"""Metadata-only registration for the sealed WP5 CNN comparison holdout."""

from __future__ import annotations

from collections.abc import Mapping, Set
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA_KEYS = {
    "image_path",
    "mask_path",
    "rgb_path",
    "semantic_path",
    "video_path",
}


class HoldoutRegistrationError(ValueError):
    """The proposed holdout is not sealed and disjoint."""


def _fail(code: str) -> HoldoutRegistrationError:
    return HoldoutRegistrationError(code)


def _contains_media_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _MEDIA_KEYS or _contains_media_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_media_field(child) for child in value)
    return False


def build_preregistered_holdout(
    source_plan: Mapping[str, object],
    *,
    source_plan_sha256: str,
    consumed_case_ids: Set[str],
    forbidden_run_ids: Set[str],
) -> dict[str, object]:
    """Select every whole V4 evaluation run without opening any media.

    ``source_plan`` is the unrendered metadata plan.  The returned declaration
    intentionally contains no file locator, recipe volume, or label field.
    """

    if not isinstance(source_plan_sha256, str) or not _SHA256.fullmatch(
        source_plan_sha256
    ):
        raise _fail("source_plan_sha256_invalid")
    if source_plan.get("artifact_type") != "physx_f2_varied_run_residual_render_plan_v4":
        raise _fail("source_plan_identity_invalid")
    if _contains_media_field(source_plan):
        raise _fail("media_field_present")

    runs = source_plan.get("runs")
    batches = source_plan.get("batches")
    if not isinstance(runs, list) or not isinstance(batches, list):
        raise _fail("source_plan_runs_invalid")
    selected = [
        run
        for run in runs
        if isinstance(run, Mapping) and run.get("split_role") == "evaluation"
    ]
    if len(selected) != 3:
        raise _fail("evaluation_run_denominator_invalid")

    batch_by_run: dict[str, Mapping[str, object]] = {}
    for batch in batches:
        if not isinstance(batch, Mapping) or not isinstance(batch.get("run_id"), str):
            raise _fail("source_plan_batch_invalid")
        batch_by_run[str(batch["run_id"])] = batch

    run_ids: list[str] = []
    case_ids: list[str] = []
    render_seeds: list[int] = []
    runs_out: list[dict[str, object]] = []
    for run in selected:
        run_id = run.get("run_id")
        run_cases = run.get("case_ids")
        if not isinstance(run_id, str) or not isinstance(run_cases, list):
            raise _fail("source_plan_run_invalid")
        if run_id in forbidden_run_ids:
            raise _fail("forbidden_run_overlap")
        if any(not isinstance(case_id, str) for case_id in run_cases):
            raise _fail("source_plan_case_invalid")
        batch = batch_by_run.get(run_id)
        if batch is None or batch.get("case_ids") != run_cases:
            raise _fail("whole_run_case_mismatch")
        seeds = batch.get("render_seed_by_vial")
        if len(run_cases) != 3 or not isinstance(seeds, list) or len(seeds) != 3:
            raise _fail("whole_run_case_mismatch")
        if any(case_id in consumed_case_ids for case_id in run_cases):
            raise _fail("consumed_case_overlap")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise _fail("render_seed_invalid")
        run_ids.append(run_id)
        case_ids.extend(run_cases)
        render_seeds.extend(seeds)
        runs_out.append(
            {
                "run_id": run_id,
                "case_ids": list(run_cases),
                "render_seeds": list(seeds),
            }
        )

    if len(set(case_ids)) != 9 or len(set(render_seeds)) != 9:
        raise _fail("holdout_identity_not_unique")

    return {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_preregistered_holdout",
        "status": "sealed_metadata_only",
        "source_plan_sha256": source_plan_sha256,
        "selection_rule": "all_three_whole_v4_evaluation_runs_in_source_order",
        "run_ids": run_ids,
        "case_ids": case_ids,
        "render_seeds": render_seeds,
        "runs": runs_out,
        "denominators": {"runs": 3, "cases": 9},
        "pixel_access_before_candidate_freeze": False,
        "media_open_count_at_preregistration": 0,
        "excluded_sources": [
            "all_v3_v4_development_runs",
            "wp1_test_runs",
            "real_video_5",
            "previously_consumed_v3_evaluation_cases",
        ],
        "forbidden_run_ids": sorted(forbidden_run_ids),
        "consumed_case_ids": sorted(consumed_case_ids),
        "scoring_gates": {
            "mean_dice_min": 0.85,
            "p05_dice_min": 0.70,
            "visible_height_error_px_max": 5.0,
            "empty_identical_controls_abstain": True,
            "empty_vial_mask_must_be_empty": True,
            "must_beat_frozen_approach_a": True,
            "low_fill_first_dispense_reported_separately": True,
        },
        "claim_boundary": (
            "synthetic heldout metadata only; no media was opened and no physical or "
            "sim-to-real accuracy is claimed"
        ),
    }


__all__ = ["HoldoutRegistrationError", "build_preregistered_holdout"]

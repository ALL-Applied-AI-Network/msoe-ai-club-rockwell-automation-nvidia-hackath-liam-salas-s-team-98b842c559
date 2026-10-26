"""Build a canonical model-input plan from verified stage captures.

This module is deliberately small and CPU-only.  A batch context describes
schedule/provenance; it is never copied into a model input as a label or a
render parameter.  Pixels remain in the capture artifacts and are bound later
by :func:`fluidmix.capture_manifest_adapter.build_measured_evidence`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import math
import os

from .capture_manifest_adapter import bind_stage_capture_manifest
from .render_process import load_strict_json_snapshot
from .vision_measurement import _publish_json_no_clobber


SPLIT_VERSION = "inn-7-splits-v2"
TANKS = frozenset(("t100", "t200", "t400"))
TANK_KEYS = ("t100", "t200", "t400")
VIALS = ("V1", "V2", "V3")
_BATCH_KEYS = {
    "batch_id",
    "cohort",
    "family_id",
    "current_recipe_id",
    "previous_recipe_id",
    "run_id",
    "programmed_recipe",
    "split_role",
}
_TOP_KEYS = {"schema_version", "artifact_type", "split_version", "batches"}
_RECIPE_KEYS = {"dispense_order", "previous_ml_by_tank", "current_ml_by_tank"}
_ORACLE_KEYS = {
    "label",
    "labels",
    "latent_fault",
    "latent_parameters",
    "oracle",
    "oracle_fields",
    "expected_disposition",
    "render_seed",
    "synthetic_profile_id",
    "appearance_blend_alpha",
    "appearance_blend_alpha_by_vial",
    "synthetic_onset_id",
    "clean_stage_linear_rgb",
    "observed_stage_linear_rgb",
    "semantic_aov",
    "residual_particle_count",
    "residual_particle_count_by_vial",
    "residual_particle_plan",
    "synthetic_oracle_record",
}


class ModelInputPlanError(ValueError):
    """A context, capture set, or model-input-plan boundary failed closed."""


def _reject_oracle_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _ORACLE_KEYS:
                raise ModelInputPlanError(f"forbidden_oracle_field: {key}")
            _reject_oracle_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_oracle_fields(child)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelInputPlanError(f"{field}_invalid")
    return value


def _load_context(value: object) -> tuple[Mapping[str, object], str | None]:
    if isinstance(value, Mapping):
        payload = value
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return payload, hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if not isinstance(value, (str, Path)):
        raise ModelInputPlanError("batch_context_invalid")
    try:
        payload, snapshot = load_strict_json_snapshot(value)
    except (OSError, ValueError) as error:
        raise ModelInputPlanError("batch_context_invalid") from error
    if not isinstance(payload, Mapping):
        raise ModelInputPlanError("batch_context_invalid")
    return payload, str(snapshot["sha256"])


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelInputPlanError(f"{field}_invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ModelInputPlanError(f"{field}_invalid")
    return result


def _parse_programmed_recipe(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RECIPE_KEYS:
        raise ModelInputPlanError("programmed_recipe_schema_invalid")
    order = value.get("dispense_order")
    previous = value.get("previous_ml_by_tank")
    current = value.get("current_ml_by_tank")
    if not isinstance(order, (list, tuple)) or len(order) != 3 or set(order) != TANKS:
        raise ModelInputPlanError("programmed_dispense_order_invalid")
    normalized_order = list(order)
    if not isinstance(previous, Mapping) or set(previous) != TANKS:
        raise ModelInputPlanError("previous_programmed_ml_schema_invalid")
    if not isinstance(current, Mapping) or set(current) != TANKS:
        raise ModelInputPlanError("programmed_ml_schema_invalid")
    previous_normalized = {
        tank: _number(previous[tank], f"previous_programmed_ml.{tank}")
        for tank in TANK_KEYS
    }
    current_normalized = {
        tank: _number(current[tank], f"current_programmed_ml.{tank}")
        for tank in TANK_KEYS
    }
    return {
        "dispense_order": normalized_order,
        "previous_ml_by_tank": previous_normalized,
        "current_ml_by_tank": current_normalized,
    }


def _validate_context(payload: Mapping[str, object]) -> list[dict[str, object]]:
    if set(payload) != _TOP_KEYS:
        raise ModelInputPlanError("batch_context_schema_invalid")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != "stage_batch_context":
        raise ModelInputPlanError("batch_context_schema_invalid")
    if payload.get("split_version") != SPLIT_VERSION:
        raise ModelInputPlanError("batch_context_split_invalid")
    batches = payload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ModelInputPlanError("batch_context_batches_invalid")
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for raw in batches:
        if not isinstance(raw, Mapping) or not set(raw).issubset(_BATCH_KEYS):
            raise ModelInputPlanError("batch_context_batch_schema_invalid")
        required = {
            "batch_id", "cohort", "family_id", "current_recipe_id",
            "previous_recipe_id", "run_id", "programmed_recipe",
        }
        if not required.issubset(raw):
            raise ModelInputPlanError("batch_context_batch_fields_missing")
        if set(raw) not in (required, required | {"split_role"}):
            raise ModelInputPlanError("batch_context_batch_schema_invalid")
        batch_id = _text(raw.get("batch_id"), "batch_id")
        if batch_id in seen:
            raise ModelInputPlanError("duplicate_context_batch")
        seen.add(batch_id)
        cohort = raw.get("cohort")
        if cohort not in {"reference_only", "scored_target"}:
            raise ModelInputPlanError("batch_cohort_invalid")
        previous = raw.get("previous_recipe_id")
        if previous is not None:
            previous = _text(previous, "previous_recipe_id")
        split_role = raw.get("split_role")
        if split_role is not None:
            split_role = _text(split_role, "split_role")
        result.append(
            {
                "batch_id": batch_id,
                "cohort": cohort,
                "family_id": _text(raw.get("family_id"), "family_id"),
                "current_recipe_id": _text(raw.get("current_recipe_id"), "current_recipe_id"),
                "previous_recipe_id": previous,
                "run_id": _text(raw.get("run_id"), "run_id"),
                "programmed_recipe": _parse_programmed_recipe(raw.get("programmed_recipe")),
                "split_role": split_role,
            }
        )
    return sorted(result, key=lambda row: str(row["batch_id"]))


def _bind_captures(capture_paths: Sequence[str | os.PathLike[str]]) -> tuple[list[dict[str, object]], dict[str, str]]:
    if not capture_paths:
        raise ModelInputPlanError("capture_manifests_missing")
    bound: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for item in capture_paths:
        path = Path(item)
        try:
            payload, snapshot = load_strict_json_snapshot(path)
            capture = bind_stage_capture_manifest(payload, base_dir=path.parent)
        except (OSError, ValueError) as error:
            raise ModelInputPlanError("capture_manifest_invalid") from error
        batch_id = _text(capture.get("batch_id"), "capture batch_id")
        if batch_id in seen:
            raise ModelInputPlanError("duplicate_capture_batch")
        seen.add(batch_id)
        hashes[batch_id] = str(snapshot["sha256"])
        bound.append(capture)
    return bound, hashes


def _candidate(
    batch: Mapping[str, object], vial: str, stage_id: str, *, origin: str = "reference_bank"
) -> dict[str, object]:
    return {
        "peer_image_id": f"{batch['batch_id']}-{vial}-{stage_id}",
        "batch_id": batch["batch_id"],
        "vial_id": vial,
        "origin": origin,
        "family_id": batch["family_id"],
        "current_sim_recipe_id": batch["current_recipe_id"],
        "partition": batch["split_role"],
    }


def _capture_stage_slots(capture: Mapping[str, object]) -> tuple[tuple[str, int, int], ...]:
    rows = capture.get("stage_statistics")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ModelInputPlanError("capture_stage_slots_invalid")
    slots: dict[int, tuple[str, int, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ModelInputPlanError("capture_stage_slots_invalid")
        ordinal = row.get("stage_ordinal")
        stage_id = row.get("stage_id")
        time_code = row.get("time_code")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not isinstance(stage_id, str) or not isinstance(time_code, int):
            raise ModelInputPlanError("capture_stage_slots_invalid")
        slot = (stage_id, ordinal, time_code)
        prior = slots.get(ordinal)
        if prior is not None and prior != slot:
            raise ModelInputPlanError("capture_stage_slots_invalid")
        slots[ordinal] = slot
    if set(slots) != {1, 2, 3}:
        raise ModelInputPlanError("capture_stage_slots_invalid")
    return tuple(slots[index] for index in (1, 2, 3))


def build_model_input_plan(
    capture_manifests: Sequence[str | os.PathLike[str]],
    batch_context: object,
) -> dict[str, object]:
    """Build one deterministic allowlisted model-input plan.

    ``capture_manifests`` are paths to strict raw manifests.  ``batch_context``
    may be its already-loaded object or a strict JSON path.
    """
    context_payload, context_hash = _load_context(batch_context)
    _reject_oracle_fields(context_payload)
    context_rows = _validate_context(context_payload)
    captures, capture_hashes = _bind_captures(capture_manifests)
    capture_by_batch = {_text(row.get("batch_id"), "capture batch_id"): row for row in captures}
    context_by_batch = {str(row["batch_id"]): row for row in context_rows}
    if set(capture_by_batch) != set(context_by_batch):
        raise ModelInputPlanError("context_capture_batch_set_mismatch")

    provenance_keys: set[tuple[str, str, str]] = set()
    for batch_id, context in context_by_batch.items():
        capture = capture_by_batch[batch_id]
        for context_field, capture_field in (
            ("family_id", "family_id"),
            ("current_recipe_id", "recipe_id"),
        ):
            if context[context_field] != capture[capture_field]:
                raise ModelInputPlanError("context_capture_provenance_mismatch")
        if capture.get("split_version") != SPLIT_VERSION:
            raise ModelInputPlanError("capture_split_invalid")
        split_role = _text(capture.get("split_role"), "capture split_role")
        if context["split_role"] is not None and context["split_role"] != split_role:
            raise ModelInputPlanError("context_capture_split_mismatch")
        context["split_role"] = split_role
        provenance_keys.add(
            (str(context["family_id"]), str(context["current_recipe_id"]), split_role)
        )
    if len(provenance_keys) != 1:
        raise ModelInputPlanError("cross_family_recipe_or_split")
    first_context = context_rows[0]
    first_recipe = first_context["programmed_recipe"]
    assert isinstance(first_recipe, Mapping)
    for context in context_rows[1:]:
        recipe = context["programmed_recipe"]
        if recipe != first_recipe:
            raise ModelInputPlanError("context_programmed_recipe_mismatch")
    stage_slots = _capture_stage_slots(captures[0])
    if any(_capture_stage_slots(capture) != stage_slots for capture in captures[1:]):
        raise ModelInputPlanError("capture_stage_slots_mismatch")
    expected_stage_ids = tuple(f"after-{tank}" for tank in first_recipe["dispense_order"])
    if tuple(slot[0] for slot in stage_slots) != expected_stage_ids:
        raise ModelInputPlanError("context_capture_stage_order_mismatch")

    # References are grouped with targets by the complete provenance key.
    groups: dict[tuple[str, str, str], dict[str, list[dict[str, object]]]] = {}
    for row in context_rows:
        capture = capture_by_batch[str(row["batch_id"])]
        role = str(capture["split_role"])
        key = (str(row["family_id"]), str(row["current_recipe_id"]), role)
        groups.setdefault(key, {"reference_only": [], "scored_target": []})[str(row["cohort"])].append(row)
    targets = [row for row in context_rows if row["cohort"] == "scored_target"]
    if not targets:
        raise ModelInputPlanError("scored_targets_missing")
    if any(row["previous_recipe_id"] is None for row in targets):
        raise ModelInputPlanError("target_previous_recipe_missing")

    decisions: list[dict[str, object]] = []
    for target in targets:
        capture = capture_by_batch[str(target["batch_id"])]
        split_role = str(capture["split_role"])
        key = (str(target["family_id"]), str(target["current_recipe_id"]), split_role)
        references = groups[key]["reference_only"]
        reference_ids = {str(row["batch_id"]) for row in references}
        if str(target["batch_id"]) in reference_ids:
            raise ModelInputPlanError("target_as_reference")
        independent_references: list[dict[str, object]] = []
        seen_reference_runs: set[str] = set()
        for reference in references:
            run_id = str(reference["run_id"])
            if run_id == target["run_id"] or run_id in seen_reference_runs:
                continue
            seen_reference_runs.add(run_id)
            independent_references.append(reference)
        independent_reference_ready = len(independent_references) >= 2
        target_id = str(target["batch_id"])
        recipe = target["programmed_recipe"]
        assert isinstance(recipe, Mapping)
        for vial_position, vial in enumerate(VIALS, start=1):
            stages: list[dict[str, object]] = []
            for ordinal, (stage_id, _, time_code) in enumerate(stage_slots, start=1):
                later = [
                    _candidate(target, later_vial, stage_id, origin="later_same_target_batch")
                    for later_vial in VIALS[vial_position:]
                ]
                all_reference_peers = [
                    *(later if independent_reference_ready else []),
                    *[
                        _candidate(ref, ref_vial, stage_id)
                        for ref in independent_references
                        for ref_vial in VIALS
                    ],
                ]
                cases: list[dict[str, object]] = [
                    {
                        "case_id": "same_batch_only",
                        "mode": "same_batch_only",
                        "alternative_id": None,
                        "candidate_peers": later,
                        "disposition_source": False,
                    }
                ]
                for ref in references:
                    cases.append(
                        {
                            "case_id": f"one_reference_batch:{ref['batch_id']}",
                            "mode": "one_reference_batch",
                            "alternative_id": ref["batch_id"],
                            "candidate_peers": [
                                *later,
                                *[_candidate(ref, ref_vial, stage_id) for ref_vial in VIALS],
                            ],
                            "disposition_source": False,
                        }
                    )
                cases.append(
                    {
                        "case_id": "all_reference_batches",
                        "mode": "all_reference_batches",
                        "alternative_id": None,
                        "candidate_peers": all_reference_peers,
                        "disposition_source": True,
                    }
                )
                stages.append(
                    {
                        "evidence_id": f"EVID-{target_id}-{vial}-{stage_id}",
                        "measurement_id": f"MEAS-{target_id}-{vial}-{stage_id}",
                        "stage_id": stage_id,
                        "stage_ordinal": ordinal,
                        "time_code": time_code,
                        "target_image_id": f"{target_id}-{vial}-{stage_id}",
                        "reference_cases": cases,
                        "primary_mode": "all_reference_batches",
                        "unit": "delta_e76",
                    }
                )
            decisions.append(
                {
                    "decision_id": f"DEC-{target_id}-{vial}",
                    "family_id": target["family_id"],
                    "batch_id": target_id,
                    "cohort": "scored_target",
                    "partition": split_role,
                    "previous_sim_recipe_id": target["previous_recipe_id"],
                    "current_sim_recipe_id": target["current_recipe_id"],
                    "previous_programmed_ml": dict(recipe["previous_ml_by_tank"]),
                    "current_programmed_ml": dict(recipe["current_ml_by_tank"]),
                    "programmed_dispense_order": list(recipe["dispense_order"]),
                    "run_id": target["run_id"],
                    "vial_id": vial,
                    "vial_position": vial_position,
                    "stages": stages,
                    "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
                }
            )

    return {
        "schema_version": 1,
        "artifact_type": "model_input_plan",
        "data_origin": "twin",
        "split_version": SPLIT_VERSION,
        "config_id": "manifest-context-model-input-plan-v1",
        "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
        "reference_policy": {
            "primary_mode": "all_reference_batches",
            "minimum_peer_count": 2,
            "minimum_distinct_batch_ids": 2,
            "minimum_distinct_run_ids": 2,
            "tight_cluster_max_diameter_delta_e76": 2.0,
            "unique_largest_cluster_required": True,
            "same_batch_case_is_diagnostic_only": True,
        },
        "source_batch_context_sha256": context_hash,
        "source_capture_manifest_sha256": {
            batch_id: capture_hashes[batch_id] for batch_id in sorted(capture_hashes)
        },
        "decision_records": decisions,
        "real_data_used": False,
        "claim_boundary": (
            "Synthetic colour-distance planning only; no physical carryover, defect, "
            "cause, plant-action, or sim-to-real claim."
        ),
    }


def write_model_input_plan(
    capture_manifests: Sequence[str | os.PathLike[str]],
    batch_context: object,
    output_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Build and atomically publish a new plan, refusing an existing target."""
    plan = build_model_input_plan(capture_manifests, batch_context)
    _publish_json_no_clobber(Path(output_path), plan)
    return plan


build_model_input_plan_from_manifests = build_model_input_plan

"""Pure corrections for the frozen WP5 CNN scoring harness.

This module does not run a model or alter a prediction.  It computes the
preregistered height metric from already-frozen masks, identifies content-level
train/holdout overlap, and reports original and cleaned score populations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
import hashlib
import math
import re

import numpy as np


_SHA256 = re.compile(r"[0-9a-f]{64}")


def summarize_replayed_mask(mask: object) -> dict[str, object]:
    """Record immutable mask identity and occupied bounds without saving the mask."""

    value = np.asarray(mask)
    if value.shape != (480, 640) or value.dtype != np.bool_:
        raise ValueError("replayed mask must be one 480x640 boolean array")
    encoded = value.astype(np.uint8).tobytes(order="C")
    bounds = _occupied_bounds(value)
    return {
        "mask_sha256": hashlib.sha256(encoded).hexdigest(),
        "mask_hash_encoding": "480x640-row-major-uint8-0-or-1",
        "first_occupied_row": None if bounds is None else bounds[0],
        "last_occupied_row": None if bounds is None else bounds[1],
        "visible_height_px": None if bounds is None else bounds[1] - bounds[0] + 1,
        "mask_pixel_count": int(np.count_nonzero(value)),
    }


def verify_replay_dice_tolerance(
    frozen_rows: Sequence[Mapping[str, object]],
    replayed_rows: Sequence[Mapping[str, object]],
    *,
    tolerance: float = 0.001,
) -> dict[str, object]:
    """Require every replayed Dice value to be within the declared tolerance."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    frozen = {
        str(row.get("case_id")): row.get("dice")
        for row in frozen_rows
        if row.get("control_kind") == "positive"
    }
    replayed = {str(row.get("case_id")): row.get("dice") for row in replayed_rows}
    if len(frozen) != len([row for row in frozen_rows if row.get("control_kind") == "positive"]):
        raise ValueError("frozen positive row identities are not unique")
    if len(replayed) != len(replayed_rows) or set(frozen) != set(replayed):
        raise ValueError("frozen and replayed row identities do not match exactly")
    maximum = 0.0
    for row_id, frozen_dice in frozen.items():
        replayed_dice = replayed[row_id]
        if (
            isinstance(frozen_dice, bool)
            or not isinstance(frozen_dice, (int, float))
            or isinstance(replayed_dice, bool)
            or not isinstance(replayed_dice, (int, float))
        ):
            raise ValueError(f"Dice values for {row_id} must be numeric")
        difference = abs(float(replayed_dice) - float(frozen_dice))
        maximum = max(maximum, difference)
        if difference > tolerance and not math.isclose(
            difference, tolerance, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                f"determinism mismatch for {row_id}: absolute Dice difference "
                f"{difference!r} exceeds {tolerance!r}"
            )
    return {
        "status": "pass",
        "comparison": "absolute_dice_difference_lte_tolerance",
        "tolerance": tolerance,
        "max_absolute_difference": maximum,
        "matched_rows": len(frozen),
        "mismatch_count": 0,
    }


def select_nonidentical_empty_control(
    preregister: Mapping[str, object],
    frozen_comparison: Mapping[str, object],
) -> dict[str, object]:
    """Select one later pre-pour frame and frame-zero reference from one run."""

    runs = preregister.get("runs")
    opened = frozen_comparison.get("opened_files")
    if not isinstance(runs, list) or not runs or not isinstance(runs[0], Mapping):
        raise ValueError("preregister is missing its first run")
    if not isinstance(opened, list) or any(not isinstance(row, Mapping) for row in opened):
        raise ValueError("comparison is missing its opened-file ledger")

    first_run = runs[0]
    run_id = first_run.get("run_id")
    case_ids = first_run.get("case_ids")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("first run is missing its run_id")
    if not isinstance(case_ids, list) or len(case_ids) < 2:
        raise ValueError("first run is missing two ordered case_ids")
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids[:2]):
        raise ValueError("first run has an invalid case_id")

    reference_case_id, current_case_id = case_ids[:2]

    def _reference(case_id: str) -> Mapping[str, object]:
        matches = [
            row
            for row in opened
            if row.get("case_id") == case_id and row.get("role") == "reference"
        ]
        if len(matches) != 1:
            raise ValueError(f"opened-file ledger is missing one reference for {case_id}")
        row = matches[0]
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError(f"reference path for {case_id} must be absolute")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"reference sha256 for {case_id} is invalid")
        return row

    reference = _reference(reference_case_id)
    current = _reference(current_case_id)
    if current["sha256"] == reference["sha256"]:
        raise ValueError("empty-control inputs must be non-identical")
    return {
        "selection_rule": (
            "first_run_second_case_prepour_vs_first_case_prepour_reference"
        ),
        "run_id": run_id,
        "current_case_id": current_case_id,
        "current_path": current["path"],
        "current_sha256": current["sha256"],
        "reference_case_id": reference_case_id,
        "reference_path": reference["path"],
        "reference_sha256": reference["sha256"],
        "model_inference_denominator": 1,
    }


def _occupied_bounds(mask: object) -> tuple[int, int] | None:
    value = np.asarray(mask)
    if value.shape != (480, 640) or value.dtype != np.bool_:
        raise ValueError("mask must be one 480x640 boolean array")
    rows = np.flatnonzero(np.any(value, axis=1))
    if not len(rows):
        return None
    return int(rows[0]), int(rows[-1])


def visible_height_error_px(predicted_mask: object, truth_mask: object) -> float:
    """Compare true visible heights derived from first and last occupied rows."""

    predicted = _occupied_bounds(predicted_mask)
    truth = _occupied_bounds(truth_mask)
    if predicted is None or truth is None:
        return 0.0 if predicted is None and truth is None else 480.0
    predicted_height = predicted[1] - predicted[0] + 1
    truth_height = truth[1] - truth[0] + 1
    return float(abs(predicted_height - truth_height))


def classify_overlap_exclusions(
    rows: Sequence[Mapping[str, object]],
    *,
    training_mask_hashes: Set[str],
    training_rgb_hashes: Set[str],
) -> dict[str, object]:
    """Exclude every score row whose truth mask or reference RGB overlaps train."""

    mask_overlaps = sorted(
        {
            str(row.get("truth_mask_sha256"))
            for row in rows
            if str(row.get("truth_mask_sha256")) in training_mask_hashes
        }
    )
    reference_overlaps = sorted(
        {
            str(row.get("reference_rgb_sha256"))
            for row in rows
            if str(row.get("reference_rgb_sha256")) in training_rgb_hashes
        }
    )
    excluded: list[str] = []
    clean: list[str] = []
    reasons: dict[str, list[str]] = {}
    for row in rows:
        row_id = row.get("case_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("every score row requires a nonempty case_id")
        row_reasons = []
        if row.get("truth_mask_sha256") in training_mask_hashes:
            row_reasons.append("training_mask_content_overlap")
        if row.get("reference_rgb_sha256") in training_rgb_hashes:
            row_reasons.append("training_reference_rgb_content_overlap")
        if row_reasons:
            excluded.append(row_id)
            reasons[row_id] = row_reasons
        else:
            clean.append(row_id)
    return {
        "classification": "likely_deterministic_simulator_duplicates",
        "overlap_content_identities": {
            "mask_sha256": mask_overlaps,
            "reference_rgb_sha256": reference_overlaps,
        },
        "excluded_positive_row_ids": excluded,
        "clean_positive_row_ids": clean,
        "exclusion_reasons_by_row_id": reasons,
    }


def _finite(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _score(
    cnn_rows: Sequence[Mapping[str, object]],
    approach_rows: Sequence[Mapping[str, object]],
    *,
    empty_control: Mapping[str, object],
) -> dict[str, object]:
    if not cnn_rows:
        raise ValueError("at least one positive score row is required")
    approach = {str(row.get("case_id")): row for row in approach_rows}
    row_ids = [str(row.get("case_id")) for row in cnn_rows]
    if len(set(row_ids)) != len(row_ids) or set(approach) != set(row_ids):
        raise ValueError("CNN and Approach-A rows must match one-to-one")
    dice = [_finite(row, "dice") for row in cnn_rows]
    height = [_finite(row, "visible_height_error_px") for row in cnn_rows]
    approach_dice = [_finite(approach[row_id], "dice") for row_id in row_ids]
    metrics = {
        "mean_dice": math.fsum(dice) / len(dice),
        "p05_dice": float(np.quantile(np.asarray(dice), 0.05)),
        "visible_height_error_px_max": max(height),
        "approach_a_mean_dice": math.fsum(approach_dice) / len(approach_dice),
    }
    empty_pass = bool(
        empty_control.get("control_kind") == "empty_vial_nonidentical"
        and empty_control.get("inference_executed") is True
        and empty_control.get("abstained") is True
        and empty_control.get("mask_pixel_count") == 0
    )
    gates = {
        "mean_dice": metrics["mean_dice"] >= 0.85,
        "p05_dice": metrics["p05_dice"] >= 0.70,
        "visible_height_error": metrics["visible_height_error_px_max"] <= 5.0,
        "nonidentical_empty_vial": empty_pass,
        "beats_approach_a": metrics["mean_dice"] > metrics["approach_a_mean_dice"],
    }
    low_fill = [row for row in cnn_rows if row.get("low_fill_first_dispense") is True]
    passed = all(gates.values())
    return {
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": passed,
        "decision": "replace_approach_a" if passed else "retain_approach_a",
        "low_fill_first_dispense": {
            "count": len(low_fill),
            "mean_dice": (
                math.fsum(_finite(row, "dice") for row in low_fill) / len(low_fill)
                if low_fill
                else None
            ),
        },
    }


def score_all_and_cleaned(
    cnn_rows: Sequence[Mapping[str, object]],
    approach_a_rows: Sequence[Mapping[str, object]],
    *,
    excluded_positive_row_ids: Set[str],
    empty_control: Mapping[str, object],
) -> dict[str, object]:
    """Score immutable rows before and after declared content-overlap exclusions."""

    if any(row.get("control_kind") != "positive" for row in cnn_rows):
        raise ValueError("cnn_rows must contain positive rows only")
    known = {str(row.get("case_id")) for row in cnn_rows}
    if not excluded_positive_row_ids <= known:
        raise ValueError("excluded row identity is not present")
    cleaned_cnn = [
        row for row in cnn_rows if str(row.get("case_id")) not in excluded_positive_row_ids
    ]
    cleaned_ids = {str(row.get("case_id")) for row in cleaned_cnn}
    cleaned_approach = [
        row for row in approach_a_rows if str(row.get("case_id")) in cleaned_ids
    ]
    identical_count = int(empty_control.get("control_kind") == "identical")
    executed_empty_count = int(
        empty_control.get("control_kind") == "empty_vial_nonidentical"
        and empty_control.get("inference_executed") is True
    )
    return {
        "denominators": {
            "all_positive_rows": len(cnn_rows),
            "excluded_positive_rows": len(excluded_positive_row_ids),
            "clean_positive_rows": len(cleaned_cnn),
            "cnn_executed_empty_controls": executed_empty_count,
            "identical_guard_controls": identical_count,
        },
        "all": _score(cnn_rows, approach_a_rows, empty_control=empty_control),
        "cleaned": _score(cleaned_cnn, cleaned_approach, empty_control=empty_control),
    }


def _dice_summary(
    cnn_rows: Sequence[Mapping[str, object]],
    approach_rows: Sequence[Mapping[str, object]],
) -> dict[str, float] | None:
    if not cnn_rows:
        return None
    approach = {str(row.get("case_id")): row for row in approach_rows}
    row_ids = [str(row.get("case_id")) for row in cnn_rows]
    if set(approach) != set(row_ids):
        raise ValueError("CNN and Approach-A rows must match one-to-one")
    dice = [_finite(row, "dice") for row in cnn_rows]
    baseline = [_finite(approach[row_id], "dice") for row_id in row_ids]
    return {
        "mean_dice": math.fsum(dice) / len(dice),
        "p05_dice": float(np.quantile(np.asarray(dice), 0.05)),
        "approach_a_mean_dice": math.fsum(baseline) / len(baseline),
    }


def build_recovery_audit(
    training_inventory: Mapping[str, object],
    frozen_comparison: Mapping[str, object],
) -> dict[str, object]:
    """Account for overlap using only the frozen inventory/comparison JSON."""

    samples = training_inventory.get("samples")
    cnn_raw = frozen_comparison.get("cnn_rows")
    approach_raw = frozen_comparison.get("approach_a_rows")
    opened = frozen_comparison.get("opened_files")
    if not isinstance(samples, list) or not isinstance(cnn_raw, list):
        raise ValueError("training inventory and CNN rows must be lists")
    if not isinstance(approach_raw, list) or not isinstance(opened, list):
        raise ValueError("Approach-A rows and opened-file ledger must be lists")
    if any(not isinstance(row, Mapping) for row in samples + cnn_raw + approach_raw + opened):
        raise ValueError("artifact rows must be objects")

    training_masks = {
        str(row["mask_sha256"])
        for row in samples
        if isinstance(row.get("mask_sha256"), str)
    }
    training_rgb = {
        str(row[key])
        for row in samples
        for key in ("current_rgb_sha256", "reference_rgb_sha256")
        if isinstance(row.get(key), str)
    }
    references: dict[str, str] = {}
    masks: dict[str, list[str]] = {}
    for row in opened:
        case_id = row.get("case_id")
        role = row.get("role")
        digest = row.get("sha256")
        if not all(isinstance(value, str) and value for value in (case_id, role, digest)):
            raise ValueError("opened-file identity is invalid")
        if role == "reference":
            previous = references.setdefault(case_id, digest)
            if previous != digest:
                raise ValueError("case has multiple reference hashes")
        elif role == "simulator_mask":
            masks.setdefault(case_id, []).append(digest)

    positives = [dict(row) for row in cnn_raw if row.get("control_kind") == "positive"]
    enriched: list[dict[str, object]] = []
    for row in positives:
        row_id = row.get("case_id")
        if not isinstance(row_id, str) or ":stage-" not in row_id:
            raise ValueError("positive row identity is invalid")
        source_case, ordinal_text = row_id.rsplit(":stage-", 1)
        try:
            ordinal = int(ordinal_text)
            truth_hash = masks[source_case][ordinal - 1]
            reference_hash = references[source_case]
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError("positive row cannot be joined to frozen opened files") from error
        row.update(
            source_case_id=source_case,
            truth_mask_sha256=truth_hash,
            reference_rgb_sha256=reference_hash,
        )
        enriched.append(row)

    exclusions = classify_overlap_exclusions(
        enriched,
        training_mask_hashes=training_masks,
        training_rgb_hashes=training_rgb,
    )
    excluded = set(exclusions["excluded_positive_row_ids"])
    clean_cnn = [row for row in enriched if str(row["case_id"]) not in excluded]
    clean_ids = {str(row["case_id"]) for row in clean_cnn}
    clean_approach = [row for row in approach_raw if str(row.get("case_id")) in clean_ids]
    overlap = exclusions["overlap_content_identities"]
    assert isinstance(overlap, Mapping)
    mask_overlap = overlap["mask_sha256"]
    reference_overlap = overlap["reference_rgb_sha256"]
    return {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_scoring_harness_recovery_audit",
        "status": "blocked_pending_frozen_height_and_empty_evidence",
        "model_inference_count": 0,
        "overlap_exclusions": exclusions,
        "denominators": {
            "all_positive_rows": len(enriched),
            "excluded_positive_rows": len(excluded),
            "clean_positive_rows": len(clean_cnn),
            "overlap_content_identities": len(mask_overlap) + len(reference_overlap),
            "overlap_mask_content_identities": len(mask_overlap),
            "overlap_reference_rgb_content_identities": len(reference_overlap),
        },
        "dice_metrics": {
            "all": _dice_summary(enriched, approach_raw),
            "cleaned": _dice_summary(clean_cnn, clean_approach),
        },
        "corrected_height_metric": {
            "definition": "abs((pred_last-pred_first+1)-(truth_last-truth_first+1))",
            "status": "blocked_no_frozen_prediction_bounds",
            "reason": (
                "The frozen comparison stores Dice and first-row offsets only; it has "
                "no predicted masks, run lengths, probabilities, or last occupied rows."
            ),
        },
        "empty_control": {
            "status": "blocked_no_authorized_nonidentical_empty_input",
            "identical_guard_counts_as_cnn_inference": False,
        },
    }


__all__ = [
    "build_recovery_audit",
    "classify_overlap_exclusions",
    "score_all_and_cleaned",
    "select_nonidentical_empty_control",
    "summarize_replayed_mask",
    "verify_replay_dice_tolerance",
    "visible_height_error_px",
]

"""Validation-only surface-row evaluation for the FillCam segmenter.

The row rule is deliberately inspectable. In the vial-centre columns, the
first row with enough liquid pixels is the visible liquid top. If the row
directly above it is blocked, the endpoints of that contiguous blocked span
are returned as a hidden-surface range.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from fluidmix import fill_level_training as training


ROWS_PER_ML = 3.04
DEFAULT_MEASUREMENT_COLUMNS_FULL = (298, 316)
DEFAULT_MEASUREMENT_ROWS_FULL = (training.CROP_XYXY[1], training.CROP_XYXY[3])
DEFAULT_MIN_LIQUID_PIXELS_PER_ROW = 1
DEFAULT_MIN_BLOCKED_PIXELS_PER_ROW = 1
CASE_NAMES = ("clean", "foam", "sticker", "edge", "hidden")


@dataclass(frozen=True)
class ValidationMetadata:
    """Manifest facts needed to slice one validation prediction."""

    seed: int
    level_ml: float
    sampling_band: str
    foam: bool
    sticker: bool
    geometry_row: float

    @property
    def cases(self):
        cases = []
        if not self.foam and not self.sticker:
            cases.append("clean")
        if self.foam:
            cases.append("foam")
        if self.sticker:
            cases.append("sticker")
        if "edge" in self.sampling_band:
            cases.append("edge")
        if "hidden" in self.sampling_band:
            cases.append("hidden")
        return tuple(cases)


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _complete_attempt(chunk_dir, expected_attempt):
    chunk_dir = Path(chunk_dir)
    complete = []
    for attempt in sorted(chunk_dir.iterdir(), key=lambda item: item.name):
        if not attempt.is_dir():
            continue
        try:
            state = (attempt / "state.txt").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if state == "COMPLETE":
            complete.append(attempt)
    if len(complete) != 1:
        raise ValueError(
            f"validation chunk {chunk_dir.name} needs exactly one COMPLETE attempt; "
            f"found {len(complete)}"
        )
    if complete[0].name != expected_attempt:
        raise ValueError(
            f"validation chunk {chunk_dir.name} COMPLETE attempt is "
            f"{complete[0].name}; approved attempt is {expected_attempt}"
        )
    return complete[0]


def _safe_output_path(output_dir, relative_path, description):
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {description} path: {relative_path}")
    output_resolved = Path(output_dir).resolve()
    resolved = (output_resolved / relative).resolve()
    if resolved != output_resolved and output_resolved not in resolved.parents:
        raise ValueError(f"unsafe {description} path: {relative_path}")
    return resolved


def discover_validation_frames(dataset_root, audit):
    """Read approved validation manifests only; train/test manifests stay unopened."""
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    approved = {
        name: contract
        for name, contract in audit["approved_chunks"].items()
        if contract["role"] == "val"
    }
    if not approved:
        raise ValueError("exception audit has no approved validation chunks")

    observed_val = set()
    for path in dataset_root.iterdir():
        match = training.CHUNK_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None and match.group("role") == "val":
            observed_val.add(path.name)
    if observed_val != set(approved):
        raise ValueError(
            "validation chunk set does not match approved batch summary: "
            f"missing={sorted(set(approved) - observed_val)}, "
            f"extra={sorted(observed_val - set(approved))}"
        )

    samples = []
    metadata = {}
    for chunk_name in sorted(approved):
        contract = approved[chunk_name]
        attempt = _complete_attempt(dataset_root / chunk_name, contract["attempt"])
        output_dir = attempt / "output"
        manifest_path = output_dir / "manifest.jsonl"
        try:
            manifest_bytes = manifest_path.read_bytes()
            lines = manifest_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(
                f"could not read validation manifest {manifest_path}: {error}"
            ) from error
        if hashlib.sha256(manifest_bytes).hexdigest() != contract["manifest_sha256"]:
            raise ValueError(f"validation chunk {chunk_name} manifest hash is not approved")

        for line_number, raw_line in enumerate(lines, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid validation manifest JSON {manifest_path}:{line_number}: "
                    f"{error.msg}"
                ) from error
            if not isinstance(record, dict) or record.get("role") != "val":
                raise ValueError(
                    f"validation manifest record {manifest_path}:{line_number} "
                    "must be a val object"
                )
            seed = int(record["seed"])
            if not contract["seed_start"] <= seed <= contract["seed_end"]:
                raise ValueError(f"validation seed {seed} is outside approved chunk range")
            if seed in metadata:
                raise ValueError(f"duplicate validation seed {seed}")
            outputs = record.get("outputs")
            if not isinstance(outputs, dict):
                raise ValueError(f"validation seed {seed} has no outputs object")
            try:
                rgb = outputs["rgb"]
                mask = outputs["mask"]
                rgb_hash = rgb["sha256"]
                mask_hash = mask["sha256"]
                level_ml = float(record["level_ml"])
                geometry_row = float(record["surface"]["geometry_row"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"validation seed {seed} has invalid metadata") from error
            if not math.isfinite(level_ml) or not math.isfinite(geometry_row):
                raise ValueError(f"validation seed {seed} has a non-finite level or row")
            if not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (rgb_hash, mask_hash)
            ):
                raise ValueError(f"validation seed {seed} has invalid output hashes")

            samples.append(
                training.FillLevelSample(
                    role="val",
                    seed=seed,
                    chunk=chunk_name,
                    frame_index=int(record["frame_index"]),
                    rgb_path=_safe_output_path(output_dir, rgb["path"], "RGB"),
                    mask_path=_safe_output_path(output_dir, mask["path"], "mask"),
                    rgb_sha256=rgb_hash,
                    mask_sha256=mask_hash,
                )
            )
            metadata[seed] = ValidationMetadata(
                seed=seed,
                level_ml=level_ml,
                sampling_band=str(record.get("level_sampling_band", "")),
                foam=bool(record.get("foam", {}).get("enabled")),
                sticker=bool(record.get("sticker_pose", {}).get("present")),
                geometry_row=geometry_row,
            )

    samples.sort(key=lambda sample: (sample.seed, sample.frame_index))
    expected_count = int(audit["eligible_counts"]["val"])
    if len(samples) != expected_count:
        raise ValueError(
            f"validation sample count must be {expected_count}; got {len(samples)}"
        )
    excluded = set(int(seed) for seed in audit.get("excluded_seeds", ()))
    if excluded & set(metadata):
        raise ValueError("excluded validation seed remains in evaluation selection")
    return tuple(samples), metadata


def validate_training_artifacts(training_summary_path, checkpoint_path, audit):
    """Bind evaluation to the completed full-run, validation-selected checkpoint."""
    summary = _read_json(training_summary_path, "training summary")
    checkpoint_path = Path(checkpoint_path)
    if summary.get("status") != "complete":
        raise ValueError("training summary status must be complete")
    if summary.get("profile") != "full" or not summary.get("checkpoint_candidate"):
        raise ValueError("surface evaluation requires a full checkpoint candidate")
    if summary.get("selection_split") != "validation":
        raise ValueError("checkpoint selection must use validation")
    dataset = summary.get("dataset", {})
    if dataset.get("test_manifests_opened") != 0 or dataset.get("test_images_loaded") != 0:
        raise ValueError("training summary must show untouched test data")
    if dataset.get("used_counts", {}).get("val") != audit["eligible_counts"]["val"]:
        raise ValueError("training summary validation count does not match audit")
    inputs = summary.get("inputs", {})
    if inputs.get("batch_summary_sha256") != audit["batch_summary_sha256"]:
        raise ValueError("training summary batch-summary hash does not match audit")
    if inputs.get("exception_review_sha256") != audit["review_sha256"]:
        raise ValueError("training summary exception-review hash does not match audit")
    observed_training_module_hash = training.file_sha256(training.__file__)
    if inputs.get("training_module_sha256") != observed_training_module_hash:
        raise ValueError("current training module SHA-256 does not match training summary")
    artifact = summary.get("artifacts", {})
    if checkpoint_path.name != artifact.get("best_weights"):
        raise ValueError("checkpoint filename does not match training summary")
    observed_hash = training.file_sha256(checkpoint_path)
    if observed_hash != artifact.get("best_weights_sha256"):
        raise ValueError("checkpoint SHA-256 does not match training summary")
    config = summary.get("config", {})
    if config.get("crop_xyxy") != list(training.CROP_XYXY):
        raise ValueError("training crop does not match evaluator crop")
    return {
        "training_summary_sha256": training.file_sha256(training_summary_path),
        "training_module_sha256": observed_training_module_hash,
        "checkpoint_sha256": observed_hash,
        "checkpoint_filename": checkpoint_path.name,
        "best_epoch": int(summary["best_epoch"]),
        "base_channels": int(config["base_channels"]),
        "training_seed": int(config["seed"]),
    }


def derive_vial_interior_rows(
    label_masks,
    *,
    measurement_columns_full=DEFAULT_MEASUREMENT_COLUMNS_FULL,
    crop_xyxy=training.CROP_XYXY,
):
    """Derive a common search interval from synthetic label-mask outlines.

    Each label mask marks pixels outside the vial as ignore (255), the nozzle
    and base as blocked (3), and the interior as a scored class.  For every
    frame, the interval starts immediately after the contiguous blocked nozzle
    span at the top of the vial outline and ends immediately before the
    contiguous blocked base span.  The intersection across frames is therefore
    inside the scored vial interior for every supplied synthetic frame.
    """

    x0, y0, x1, y1 = (int(value) for value in crop_xyxy)
    expected_shape = (y1 - y0, x1 - x0)
    first_full, last_full = (int(value) for value in measurement_columns_full)
    first = first_full - x0
    last = last_full - x0
    if not 0 <= first < last <= expected_shape[1]:
        raise ValueError("measurement columns are outside the mask crop")

    per_frame_starts = []
    per_frame_ends = []
    for index, raw_mask in enumerate(label_masks):
        mask = np.asarray(raw_mask)
        if mask.shape != expected_shape:
            raise ValueError(
                f"label mask {index} shape must be {expected_shape}; got {mask.shape}"
            )
        strip = mask[:, first:last]
        labeled_counts = np.count_nonzero(strip != training.IGNORE_LABEL, axis=1)
        labeled_rows = np.flatnonzero(labeled_counts > 0)
        if not labeled_rows.size:
            raise ValueError(f"label mask {index} has no vial outline in the measurement band")

        blocked_counts = np.count_nonzero(strip == 3, axis=1)
        first_labeled = int(labeled_rows[0])
        last_labeled = int(labeled_rows[-1])

        interior_start = first_labeled
        while interior_start <= last_labeled and blocked_counts[interior_start] > 0:
            interior_start += 1
        interior_last = last_labeled
        while interior_last >= interior_start and blocked_counts[interior_last] > 0:
            interior_last -= 1
        if interior_start > interior_last:
            raise ValueError(f"label mask {index} has no interior between nozzle and base")

        per_frame_starts.append(y0 + interior_start)
        per_frame_ends.append(y0 + interior_last + 1)

    if not per_frame_starts:
        raise ValueError("at least one synthetic label mask is required")
    selected_start = max(per_frame_starts)
    selected_end = min(per_frame_ends)
    if selected_start >= selected_end:
        raise ValueError("synthetic vial-interior row intervals do not overlap")

    return {
        "measurement_rows_full_half_open": [selected_start, selected_end],
        "selection_source": "synthetic_validation_label_outline_intersection",
        "frame_count": len(per_frame_starts),
        "per_frame_start_min": min(per_frame_starts),
        "per_frame_start_max": max(per_frame_starts),
        "per_frame_end_exclusive_min": min(per_frame_ends),
        "per_frame_end_exclusive_max": max(per_frame_ends),
        "outline_rule": (
            "intersection of per-frame centre-band intervals after the contiguous "
            "top blocked nozzle span and before the contiguous bottom blocked base span"
        ),
    }


def surface_prediction(
    mask,
    *,
    measurement_columns_full=DEFAULT_MEASUREMENT_COLUMNS_FULL,
    measurement_rows_full=DEFAULT_MEASUREMENT_ROWS_FULL,
    min_liquid_pixels=DEFAULT_MIN_LIQUID_PIXELS_PER_ROW,
    min_blocked_pixels=DEFAULT_MIN_BLOCKED_PIXELS_PER_ROW,
    crop_xyxy=training.CROP_XYXY,
):
    """Apply the frozen row/hidden rule to one cropped class mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D; got shape {mask.shape}")
    x0, y0, x1, y1 = (int(value) for value in crop_xyxy)
    expected_shape = (y1 - y0, x1 - x0)
    if mask.shape != expected_shape:
        raise ValueError(f"mask shape must be {expected_shape}; got {mask.shape}")
    first_full, last_full = (int(value) for value in measurement_columns_full)
    first = first_full - x0
    last = last_full - x0
    if not 0 <= first < last <= mask.shape[1]:
        raise ValueError("measurement columns are outside the mask crop")
    first_row_full, last_row_full = (int(value) for value in measurement_rows_full)
    first_row = first_row_full - y0
    last_row = last_row_full - y0
    if not 0 <= first_row < last_row <= mask.shape[0]:
        raise ValueError("measurement rows are outside the mask crop")
    if min_liquid_pixels <= 0 or min_liquid_pixels > last - first:
        raise ValueError("invalid minimum liquid pixels")
    if min_blocked_pixels <= 0 or min_blocked_pixels > last - first:
        raise ValueError("invalid minimum blocked pixels")

    strip = mask[first_row:last_row, first:last]
    liquid_counts = np.count_nonzero(strip == 1, axis=1)
    liquid_rows = np.flatnonzero(liquid_counts >= int(min_liquid_pixels))
    if not liquid_rows.size:
        return {
            "status": "abstain",
            "liquid_top_row": None,
            "range_top_row": None,
            "range_bottom_row": None,
        }
    liquid_row = first_row + int(liquid_rows[0])
    full_liquid_row = float(y0 + liquid_row)
    blocked_counts = np.count_nonzero(mask[:, first:last] == 3, axis=1)
    if (
        liquid_row == first_row
        or blocked_counts[liquid_row - 1] < int(min_blocked_pixels)
    ):
        return {
            "status": "visible",
            "liquid_top_row": full_liquid_row,
            "range_top_row": full_liquid_row,
            "range_bottom_row": full_liquid_row,
        }

    blocked_top = liquid_row - 1
    while blocked_top > first_row and blocked_counts[blocked_top - 1] >= int(
        min_blocked_pixels
    ):
        blocked_top -= 1
    return {
        "status": "hidden",
        "liquid_top_row": full_liquid_row,
        "range_top_row": float(y0 + blocked_top),
        "range_bottom_row": float(y0 + liquid_row - 1),
    }


def frozen_rule_config(
    *,
    measurement_columns_full=DEFAULT_MEASUREMENT_COLUMNS_FULL,
    measurement_rows_full=DEFAULT_MEASUREMENT_ROWS_FULL,
    min_liquid_pixels=DEFAULT_MIN_LIQUID_PIXELS_PER_ROW,
    min_blocked_pixels=DEFAULT_MIN_BLOCKED_PIXELS_PER_ROW,
    row_bounds_evidence=None,
):
    """Return the exact rule parameters to persist with validation evidence."""
    return {
        "selection_source": "synthetic_validation_only",
        "crop_xyxy": list(training.CROP_XYXY),
        "class_ids": {"liquid": 1, "blocked": 3},
        "measurement_columns_full_half_open": [
            int(measurement_columns_full[0]),
            int(measurement_columns_full[1]),
        ],
        "measurement_rows_full_half_open": [
            int(measurement_rows_full[0]),
            int(measurement_rows_full[1]),
        ],
        "measurement_row_derivation": row_bounds_evidence,
        "minimum_liquid_pixels_per_row": int(min_liquid_pixels),
        "minimum_blocked_pixels_per_row": int(min_blocked_pixels),
        "visible_row_rule": (
            "first row with minimum liquid pixels in the measurement columns "
            "and frozen vial-interior row interval"
        ),
        "hidden_rule": (
            "when the row immediately above the visible liquid top has the minimum "
            "blocked pixels, return the contiguous blocked span immediately above "
            "the visible top"
        ),
        "truth_source": "top liquid row from the validation label mask",
        "hidden_range_check": "manifest geometry row contained by predicted row range",
        "hidden_supplemental_error": (
            "distance from label-mask liquid top to the nearest predicted range endpoint"
        ),
        "rows_per_ml_for_reporting": ROWS_PER_ML,
    }


def level_bin_10ml(level_ml):
    level = float(level_ml)
    if not math.isfinite(level) or not 0.0 <= level <= 90.0:
        raise ValueError(f"level outside supported 0-90 mL validation bins: {level}")
    lower = min(int(level // 10) * 10, 80)
    return f"{lower:02d}-{lower + 10:02d}"


def frame_result(metadata, truth_mask, predicted_mask, *, rule_config=None):
    """Score a prediction against the label-mask liquid top."""
    if rule_config is None:
        rule_config = frozen_rule_config()
    if rule_config.get("crop_xyxy") != list(training.CROP_XYXY):
        raise ValueError("surface-rule crop does not match the trained model crop")
    columns = tuple(rule_config["measurement_columns_full_half_open"])
    rows = tuple(
        rule_config.get(
            "measurement_rows_full_half_open", DEFAULT_MEASUREMENT_ROWS_FULL
        )
    )
    min_liquid = int(rule_config["minimum_liquid_pixels_per_row"])
    min_blocked = int(rule_config["minimum_blocked_pixels_per_row"])
    options = {
        "measurement_columns_full": columns,
        "measurement_rows_full": rows,
        "min_liquid_pixels": min_liquid,
        "min_blocked_pixels": min_blocked,
    }
    # Label truth is literal: the first label-mask liquid pixel in the same band.
    # Prediction threshold tuning must never move the comparison target.
    truth = surface_prediction(
        truth_mask,
        measurement_columns_full=columns,
        measurement_rows_full=rows,
        min_liquid_pixels=1,
        min_blocked_pixels=1,
    )
    if truth["liquid_top_row"] is None:
        raise ValueError(f"validation seed {metadata.seed} label has no liquid top")
    prediction = surface_prediction(predicted_mask, **options)
    true_row = float(truth["liquid_top_row"])
    result = {
        "seed": int(metadata.seed),
        "level_ml": float(metadata.level_ml),
        "level_bin_10ml": level_bin_10ml(metadata.level_ml),
        "sampling_band": metadata.sampling_band,
        "cases": list(metadata.cases),
        "true_top_liquid_row": true_row,
        "manifest_geometry_row": float(metadata.geometry_row),
        "prediction_status": prediction["status"],
        "predicted_liquid_top_row": prediction["liquid_top_row"],
        "predicted_range_top_row": prediction["range_top_row"],
        "predicted_range_bottom_row": prediction["range_bottom_row"],
        "geometry_in_predicted_range": None,
        "visible_signed_error_rows": None,
        "visible_absolute_error_rows": None,
        "visible_signed_error_ml": None,
        "visible_absolute_error_ml": None,
        "hidden_interval_distance_to_label_rows": None,
        "hidden_interval_distance_to_label_ml": None,
    }
    if prediction["status"] == "abstain":
        return result

    top = float(prediction["range_top_row"])
    bottom = float(prediction["range_bottom_row"])
    if prediction["status"] == "visible":
        signed_rows = float(prediction["liquid_top_row"]) - true_row
        # Image rows increase downward while inferred volume increases upward.
        signed_ml = -signed_rows / ROWS_PER_ML
        result.update(
            {
                "visible_signed_error_rows": signed_rows,
                "visible_absolute_error_rows": abs(signed_rows),
                "visible_signed_error_ml": signed_ml,
                "visible_absolute_error_ml": abs(signed_ml),
            }
        )
    else:
        result["geometry_in_predicted_range"] = (
            top <= float(metadata.geometry_row) <= bottom
        )
        if true_row < top:
            distance = top - true_row
        elif true_row > bottom:
            distance = true_row - bottom
        else:
            distance = 0.0
        result["hidden_interval_distance_to_label_rows"] = distance
        result["hidden_interval_distance_to_label_ml"] = distance / ROWS_PER_ML
    return result


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _aggregate(items):
    visible = [
        item for item in items if item["visible_absolute_error_rows"] is not None
    ]
    predicted_hidden = [item for item in items if item["prediction_status"] == "hidden"]
    contained = [item for item in predicted_hidden if item["geometry_in_predicted_range"]]
    abstained = [item for item in items if item["prediction_status"] == "abstain"]
    abs_rows = [item["visible_absolute_error_rows"] for item in visible]
    abs_ml = [item["visible_absolute_error_ml"] for item in visible]
    interval_rows = [
        item["hidden_interval_distance_to_label_rows"] for item in predicted_hidden
    ]
    interval_ml = [
        item["hidden_interval_distance_to_label_ml"] for item in predicted_hidden
    ]
    result = {
        "frames": len(items),
        "visible_predictions": len(visible),
        "visible_prediction_rate": (len(visible) / len(items)) if items else None,
        "abstained": len(abstained),
        "abstention_rate": (len(abstained) / len(items)) if items else None,
        "predicted_hidden": len(predicted_hidden),
        "predicted_hidden_rate": (len(predicted_hidden) / len(items)) if items else None,
        "geometry_in_predicted_range": len(contained),
        "true_in_range_rate": (
            len(contained) / len(predicted_hidden) if predicted_hidden else None
        ),
        "true_in_range_over_all_frames_rate": (
            len(contained) / len(items) if items else None
        ),
    }
    metric_names = (
        "mae_rows",
        "median_absolute_error_rows",
        "p95_absolute_error_rows",
        "max_absolute_error_rows",
        "mae_ml",
        "median_absolute_error_ml",
        "p95_absolute_error_ml",
        "max_absolute_error_ml",
        "hidden_interval_median_distance_to_label_rows",
        "hidden_interval_p95_distance_to_label_rows",
        "hidden_interval_max_distance_to_label_rows",
        "hidden_interval_median_distance_to_label_ml",
        "hidden_interval_p95_distance_to_label_ml",
        "hidden_interval_max_distance_to_label_ml",
    )
    result.update({name: None for name in metric_names})
    if visible:
        result.update({
            "mae_rows": float(np.mean(abs_rows)),
            "median_absolute_error_rows": _percentile(abs_rows, 50),
            "p95_absolute_error_rows": _percentile(abs_rows, 95),
            "max_absolute_error_rows": float(np.max(abs_rows)),
            "mae_ml": float(np.mean(abs_ml)),
            "median_absolute_error_ml": _percentile(abs_ml, 50),
            "p95_absolute_error_ml": _percentile(abs_ml, 95),
            "max_absolute_error_ml": float(np.max(abs_ml)),
        })
    if predicted_hidden:
        result.update({
            "hidden_interval_median_distance_to_label_rows": _percentile(interval_rows, 50),
            "hidden_interval_p95_distance_to_label_rows": _percentile(interval_rows, 95),
            "hidden_interval_max_distance_to_label_rows": float(np.max(interval_rows)),
            "hidden_interval_median_distance_to_label_ml": _percentile(interval_ml, 50),
            "hidden_interval_p95_distance_to_label_ml": _percentile(interval_ml, 95),
            "hidden_interval_max_distance_to_label_ml": float(np.max(interval_ml)),
        })
    return result


def summarize_results(results):
    """Aggregate all validation frames, requested cases and fixed 10 mL bins."""
    results = tuple(results)
    by_case = {"all": _aggregate(list(results))}
    for name in CASE_NAMES:
        by_case[name] = _aggregate([item for item in results if name in item["cases"]])
    by_level_bin = {}
    for lower in range(0, 90, 10):
        name = f"{lower:02d}-{lower + 10:02d}"
        by_level_bin[name] = _aggregate(
            [item for item in results if item["level_bin_10ml"] == name]
        )
    return {"by_case": by_case, "by_level_bin_10ml": by_level_bin}


def validate_case_counts(batch_summary_path, aggregate):
    """Cross-check case membership against the integrity-approved Step 6 summary."""
    summary = _read_json(batch_summary_path, "batch summary")
    expected = summary.get("case_coverage", {}).get("by_role", {}).get("val", {})
    mapping = {
        "clean": "clean",
        "foam": "foam",
        "sticker": "sticker",
        "edge": "edge_zone",
        "hidden": "hidden",
    }
    by_case = aggregate["by_case"]
    for observed_name, expected_name in mapping.items():
        expected_count = expected.get(expected_name)
        if not isinstance(expected_count, int):
            raise ValueError(f"batch summary lacks validation case count {expected_name}")
        if by_case[observed_name]["frames"] != expected_count:
            raise ValueError(
                f"validation {observed_name} count must be {expected_count}; "
                f"got {by_case[observed_name]['frames']}"
            )
    return {name: by_case[name]["frames"] for name in CASE_NAMES}

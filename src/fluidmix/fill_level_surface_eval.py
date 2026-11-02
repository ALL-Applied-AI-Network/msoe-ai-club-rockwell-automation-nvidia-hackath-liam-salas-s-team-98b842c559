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
DEFAULT_SURFACE_WIDTH_WINDOW_ROWS = 32
DEFAULT_VISIBILITY_LIQUID_FRACTION = 0.05
CASE_NAMES = ("clean", "foam", "sticker", "edge", "hidden")
STEP6_VALIDATION_BATCH_SUMMARY_SHA256 = (
    "e736d17ee7db384fa9b15cadc97422bb095b2a82ef1fb6b879040df0424d88c0"
)
STEP6_VALIDATION_LEDGER_SHA256 = (
    "8969e9727149dc2f987b2afb0524fc042f46d5396adbb199411ce1c2ca2066dd"
)
STEP6_VALIDATION_FRAME_COUNT = 300
STEP6_VALIDATION_SEED_RANGE = (1100000, 1100299)
APPROVED_HIDDEN_RULE_CONSTANTS = {
    "holder_band_rows_full_inclusive": [217, 275],
    "surface_oval_height_rows": 17,
    "visibility_columns_full_half_open": [279, 340],
    "minimum_liquid_pixels_per_visibility_row": 3,
}


@dataclass(frozen=True)
class ValidationMetadata:
    """Manifest facts needed to slice one validation prediction."""

    seed: int
    level_ml: float
    sampling_band: str
    foam: bool
    sticker: bool
    geometry_row: float
    holder_rows_inclusive: tuple[int, int] | None = None

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


def step6_hidden_rule_derivation_indices(samples):
    """Select the complete, ordered Step 6 validation cohort from a combined audit."""

    first, last = STEP6_VALIDATION_SEED_RANGE
    selected = tuple(
        index
        for index, sample in enumerate(samples)
        if first <= int(sample.seed) <= last
    )
    observed = [int(samples[index].seed) for index in selected]
    expected = list(range(first, last + 1))
    if observed != expected or len(selected) != STEP6_VALIDATION_FRAME_COUNT:
        raise ValueError(
            "hidden-rule derivation requires the complete ordered Step 6 "
            "validation cohort"
        )
    return selected


def bind_step6_hidden_rule_derivation(evidence, preflight):
    """Bind derived hidden-rule constants to the approved Step 6 validation bytes."""

    if not isinstance(evidence, dict) or evidence.get("selection_source") != (
        "synthetic_validation_labels_and_manifest_geometry"
    ):
        raise ValueError("hidden-rule derivation source is invalid")
    if evidence.get("frame_count") != STEP6_VALIDATION_FRAME_COUNT:
        raise ValueError("hidden-rule derivation frame count is not Step 6")
    for name, expected in APPROVED_HIDDEN_RULE_CONSTANTS.items():
        if evidence.get(name) != expected:
            raise ValueError(f"hidden-rule derivation {name} is not the approved value")
    if not isinstance(preflight, dict) or (
        preflight.get("samples_checked") != STEP6_VALIDATION_FRAME_COUNT
        or preflight.get("files_checked") != 2 * STEP6_VALIDATION_FRAME_COUNT
        or preflight.get("ordered_hash_ledger_sha256")
        != STEP6_VALIDATION_LEDGER_SHA256
    ):
        raise ValueError("hidden-rule derivation preflight is not the approved Step 6 ledger")
    return {
        **evidence,
        "role": "val",
        "ordered_hash_ledger_sha256": STEP6_VALIDATION_LEDGER_SHA256,
        "source_component": "step6",
        "source_batch_summary_sha256": STEP6_VALIDATION_BATCH_SUMMARY_SHA256,
    }


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
                holder_rows = record["holder_projection"]["selected_rows_inclusive"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"validation seed {seed} has invalid metadata") from error
            if (
                not isinstance(holder_rows, list)
                or len(holder_rows) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in holder_rows)
                or not 0 <= holder_rows[0] <= holder_rows[1] < 480
            ):
                raise ValueError(f"validation seed {seed} has invalid holder rows")
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
                holder_rows_inclusive=(int(holder_rows[0]), int(holder_rows[1])),
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


def _upper_coverage_order_statistic(values, coverage):
    values = sorted(int(value) for value in values)
    if not values:
        raise ValueError("at least one value is required")
    coverage = float(coverage)
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    index = max(0, min(len(values) - 1, int(math.ceil(coverage * len(values))) - 1))
    return values[index]


def derive_hidden_surface_rule(
    label_masks,
    metadata,
    *,
    crop_xyxy=training.CROP_XYXY,
    surface_width_window_rows=DEFAULT_SURFACE_WIDTH_WINDOW_ROWS,
    surface_height_coverage=0.95,
    visibility_liquid_fraction=DEFAULT_VISIBILITY_LIQUID_FRACTION,
):
    """Derive the holder band and surface-oval margin from validation labels only.

    The holder band uses the median projected carrier endpoints from the validation
    manifests.  The visibility columns are the intersection of every validation vial
    outline, so they cannot include pixels outside any supplied vial.  Surface-oval
    height is measured only where foam, stickers and the holder do not cover the first
    ``surface_width_window_rows`` liquid rows.  It is the first row at which the liquid
    label reaches its maximum width in that window; the returned margin is the upper
    coverage order statistic across eligible validation frames.
    """

    masks = tuple(np.asarray(mask) for mask in label_masks)
    items = tuple(metadata)
    if not masks or len(masks) != len(items):
        raise ValueError("label masks and validation metadata must have equal nonzero length")
    x0, y0, x1, y1 = (int(value) for value in crop_xyxy)
    expected_shape = (y1 - y0, x1 - x0)
    window = int(surface_width_window_rows)
    if window <= 1 or window > expected_shape[0]:
        raise ValueError("surface-width window is outside the mask crop")
    visibility_fraction = float(visibility_liquid_fraction)
    if not 0.0 < visibility_fraction <= 1.0:
        raise ValueError("visibility liquid fraction must be in (0, 1]")

    holder_tops = []
    holder_bottoms = []
    vial_lefts = []
    vial_rights = []
    for index, (mask, item) in enumerate(zip(masks, items)):
        if mask.shape != expected_shape:
            raise ValueError(
                f"label mask {index} shape must be {expected_shape}; got {mask.shape}"
            )
        holder = item.holder_rows_inclusive
        if holder is None or len(holder) != 2:
            raise ValueError(f"validation seed {item.seed} has no holder-row evidence")
        holder_top, holder_bottom = (int(value) for value in holder)
        if not y0 <= holder_top <= holder_bottom < y1:
            raise ValueError(f"validation seed {item.seed} holder rows are outside the crop")
        holder_tops.append(holder_top)
        holder_bottoms.append(holder_bottom)

        scored_columns = np.flatnonzero(np.any(mask != training.IGNORE_LABEL, axis=0))
        if not scored_columns.size:
            raise ValueError(f"validation seed {item.seed} has no vial outline")
        vial_lefts.append(int(scored_columns[0]))
        vial_rights.append(int(scored_columns[-1]) + 1)

    holder_top = int(math.floor(float(np.median(holder_tops)) + 0.5))
    holder_bottom = int(math.floor(float(np.median(holder_bottoms)) + 0.5))
    visibility_first = max(vial_lefts)
    visibility_last = min(vial_rights)
    if visibility_first >= visibility_last:
        raise ValueError("validation vial-outline columns do not overlap")
    visibility_width = visibility_last - visibility_first
    min_visibility_liquid = max(1, int(round(visibility_width * visibility_fraction)))

    oval_heights = []
    eligible_seeds = []
    for mask, item in zip(masks, items):
        if item.level_ml <= 0.0 or item.foam or item.sticker:
            continue
        frame_holder_top, frame_holder_bottom = item.holder_rows_inclusive
        geometry_row = float(item.geometry_row)
        if not (
            geometry_row + window < frame_holder_top
            or geometry_row > frame_holder_bottom
        ):
            continue
        liquid_widths = np.count_nonzero(
            mask[:, visibility_first:visibility_last] == 1,
            axis=1,
        )
        liquid_rows = np.flatnonzero(liquid_widths > 0)
        if not liquid_rows.size:
            continue
        liquid_top = int(liquid_rows[0])
        full_liquid_top = y0 + liquid_top
        if abs(full_liquid_top - geometry_row) > 2.0:
            continue
        width_window = liquid_widths[liquid_top : liquid_top + window]
        if width_window.size != window or int(width_window.max()) <= 0:
            continue
        first_full_width = int(np.flatnonzero(width_window == width_window.max())[0])
        oval_heights.append(first_full_width + 1)
        eligible_seeds.append(int(item.seed))

    if not oval_heights:
        raise ValueError("no unobstructed validation surfaces support oval-height derivation")
    surface_oval_height = _upper_coverage_order_statistic(
        oval_heights,
        surface_height_coverage,
    )
    return {
        "selection_source": "synthetic_validation_labels_and_manifest_geometry",
        "frame_count": len(masks),
        "holder_band_rows_full_inclusive": [holder_top, holder_bottom],
        "holder_endpoint_statistic": "nearest-integer median across validation frames",
        "holder_top_rows": {
            "minimum": min(holder_tops),
            "median": float(np.median(holder_tops)),
            "maximum": max(holder_tops),
        },
        "holder_bottom_rows": {
            "minimum": min(holder_bottoms),
            "median": float(np.median(holder_bottoms)),
            "maximum": max(holder_bottoms),
        },
        "visibility_columns_full_half_open": [
            x0 + visibility_first,
            x0 + visibility_last,
        ],
        "visibility_column_derivation": (
            "intersection of validation label-mask vial-outline columns"
        ),
        "minimum_liquid_pixels_per_visibility_row": min_visibility_liquid,
        "minimum_visibility_liquid_fraction_requested": visibility_fraction,
        "surface_oval_height_rows": surface_oval_height,
        "surface_oval_height_coverage": float(surface_height_coverage),
        "surface_oval_height_statistic": (
            "upper coverage order statistic of first maximum-width liquid row"
        ),
        "surface_width_window_rows": window,
        "surface_oval_eligible_frame_count": len(oval_heights),
        "surface_oval_eligible_seeds": eligible_seeds,
        "surface_oval_height_rows_distribution": {
            "minimum": min(oval_heights),
            "median": float(np.median(oval_heights)),
            "maximum": max(oval_heights),
        },
        "surface_oval_eligibility": (
            "positive-level validation frame without foam or sticker, with the first "
            "surface-width window clear of the frame-specific holder band and label "
            "top within two rows of manifest geometry"
        ),
    }


def surface_prediction(
    mask,
    *,
    measurement_columns_full=DEFAULT_MEASUREMENT_COLUMNS_FULL,
    measurement_rows_full=DEFAULT_MEASUREMENT_ROWS_FULL,
    min_liquid_pixels=DEFAULT_MIN_LIQUID_PIXELS_PER_ROW,
    min_blocked_pixels=DEFAULT_MIN_BLOCKED_PIXELS_PER_ROW,
    holder_band_rows_full=None,
    surface_oval_height_rows=None,
    visibility_columns_full=None,
    min_visibility_liquid_pixels=None,
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
    liquid_row = first_row + int(liquid_rows[0]) if liquid_rows.size else None
    full_liquid_row = float(y0 + liquid_row) if liquid_row is not None else None

    holder_options = (
        holder_band_rows_full,
        surface_oval_height_rows,
        visibility_columns_full,
        min_visibility_liquid_pixels,
    )
    if any(value is not None for value in holder_options):
        if any(value is None for value in holder_options):
            raise ValueError("holder-band rule options must be supplied together")
        band_top, band_bottom = (int(value) for value in holder_band_rows_full)
        oval_height = int(surface_oval_height_rows)
        visibility_first_full, visibility_last_full = (
            int(value) for value in visibility_columns_full
        )
        visibility_first = visibility_first_full - x0
        visibility_last = visibility_last_full - x0
        min_visibility = int(min_visibility_liquid_pixels)
        if not first_row_full <= band_top <= band_bottom < last_row_full:
            raise ValueError("holder band is outside the measurement rows")
        if oval_height <= 0:
            raise ValueError("surface-oval height must be positive")
        if not 0 <= visibility_first < visibility_last <= mask.shape[1]:
            raise ValueError("visibility columns are outside the mask crop")
        if not 0 < min_visibility <= visibility_last - visibility_first:
            raise ValueError("invalid minimum visibility liquid pixels")

        visibility_counts = np.count_nonzero(
            mask[:, visibility_first:visibility_last] == 1,
            axis=1,
        )
        qualifying_visibility_rows = np.flatnonzero(
            visibility_counts >= min_visibility
        ) + y0
        widened_top = max(first_row_full, band_top - oval_height)
        widened_bottom = min(last_row_full - 1, band_bottom + oval_height)
        first_visibility_row = (
            int(qualifying_visibility_rows[0])
            if qualifying_visibility_rows.size
            else None
        )
        edge_flag = (
            first_visibility_row is not None
            and widened_top <= first_visibility_row <= band_top + oval_height
        )
        liquid_above_band = np.any(
            (qualifying_visibility_rows >= widened_top)
            & (qualifying_visibility_rows < band_top)
        )
        liquid_below_band = np.any(
            (qualifying_visibility_rows > band_bottom)
            & (qualifying_visibility_rows <= widened_bottom)
        )
        if liquid_below_band and not liquid_above_band and not edge_flag:
            below = qualifying_visibility_rows[
                (qualifying_visibility_rows > band_bottom)
                & (qualifying_visibility_rows <= widened_bottom)
            ]
            return {
                "status": "hidden",
                "liquid_top_row": (
                    full_liquid_row if full_liquid_row is not None else float(below[0])
                ),
                "range_top_row": float(widened_top),
                "range_bottom_row": float(widened_bottom),
                "edge_flag": False,
            }
        if not liquid_rows.size and not edge_flag:
            return {
                "status": "abstain",
                "liquid_top_row": None,
                "range_top_row": None,
                "range_bottom_row": None,
                "edge_flag": False,
            }
        reported_liquid_row = (
            full_liquid_row
            if full_liquid_row is not None
            else float(first_visibility_row)
        )
        return {
            "status": "visible",
            "liquid_top_row": reported_liquid_row,
            "range_top_row": reported_liquid_row,
            "range_bottom_row": reported_liquid_row,
            "edge_flag": bool(edge_flag),
        }

    if not liquid_rows.size:
        return {
            "status": "abstain",
            "liquid_top_row": None,
            "range_top_row": None,
            "range_bottom_row": None,
        }
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
    hidden_rule_evidence=None,
):
    """Return the exact rule parameters to persist with validation evidence."""
    result = {
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
    if hidden_rule_evidence is not None:
        evidence = dict(hidden_rule_evidence)
        required = (
            "holder_band_rows_full_inclusive",
            "surface_oval_height_rows",
            "visibility_columns_full_half_open",
            "minimum_liquid_pixels_per_visibility_row",
        )
        missing = [name for name in required if name not in evidence]
        if missing:
            raise ValueError(f"hidden-rule evidence lacks {missing}")
        result.update({
            "hidden_rule_version": "holder_band_v2",
            "hidden_rule_derivation": evidence,
            "holder_band_rows_full_inclusive": list(
                evidence["holder_band_rows_full_inclusive"]
            ),
            "surface_oval_height_rows": int(evidence["surface_oval_height_rows"]),
            "visibility_columns_full_half_open": list(
                evidence["visibility_columns_full_half_open"]
            ),
            "minimum_liquid_pixels_per_visibility_row": int(
                evidence["minimum_liquid_pixels_per_visibility_row"]
            ),
            "hidden_rule": (
                "call hidden when liquid is visible within one validation-derived "
                "surface-oval height below the fixed holder band and no liquid is "
                "visible within that height above the band; return the holder band "
                "widened at both ends by the surface-oval height"
            ),
            "edge_rule": (
                "flag a visible prediction when its first broad-vial liquid row is "
                "within one surface-oval height of the holder band's top edge"
            ),
        })
    else:
        result["hidden_rule_version"] = "contiguous_blocked_v1"
    return result


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
    if rule_config.get("hidden_rule_version") == "holder_band_v2":
        options.update({
            "holder_band_rows_full": tuple(
                rule_config["holder_band_rows_full_inclusive"]
            ),
            "surface_oval_height_rows": int(
                rule_config["surface_oval_height_rows"]
            ),
            "visibility_columns_full": tuple(
                rule_config["visibility_columns_full_half_open"]
            ),
            "min_visibility_liquid_pixels": int(
                rule_config["minimum_liquid_pixels_per_visibility_row"]
            ),
        })
    # Label truth is literal: the first label-mask liquid pixel in the same
    # columns. Prediction threshold tuning must never move the comparison
    # target. The frozen prediction rows are the common interior intersection;
    # a low surface can begin exactly below that intersection in an otherwise
    # valid jittered frame, so truth alone may fall back to the full crop rows.
    truth_row_source = "label_measurement_rows"
    truth = surface_prediction(
        truth_mask,
        measurement_columns_full=columns,
        measurement_rows_full=rows,
        min_liquid_pixels=1,
        min_blocked_pixels=1,
    )
    if truth["liquid_top_row"] is None:
        _, crop_y0, _, crop_y1 = (
            int(value) for value in rule_config["crop_xyxy"]
        )
        truth = surface_prediction(
            truth_mask,
            measurement_columns_full=columns,
            measurement_rows_full=(crop_y0, crop_y1),
            min_liquid_pixels=1,
            min_blocked_pixels=1,
            crop_xyxy=tuple(rule_config["crop_xyxy"]),
        )
        truth_row_source = "label_same_columns_full_crop_fallback"
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
        "truth_row_source": truth_row_source,
        "manifest_geometry_row": float(metadata.geometry_row),
        "prediction_status": prediction["status"],
        "edge_flag": bool(prediction.get("edge_flag", False)),
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
    edge_flagged = [item for item in items if item.get("edge_flag")]
    truth_crop_fallbacks = [
        item
        for item in items
        if item.get("truth_row_source")
        == "label_same_columns_full_crop_fallback"
    ]
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
        "edge_flagged": len(edge_flagged),
        "edge_flag_rate": (len(edge_flagged) / len(items)) if items else None,
        "truth_label_crop_fallbacks": len(truth_crop_fallbacks),
        "truth_label_crop_fallback_rate": (
            len(truth_crop_fallbacks) / len(items) if items else None
        ),
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

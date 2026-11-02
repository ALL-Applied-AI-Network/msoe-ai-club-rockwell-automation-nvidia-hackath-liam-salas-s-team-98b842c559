"""Pure contracts for the frozen Step 10 real-video evaluation.

The command-line evaluator owns OpenCV/Torch setup and evidence writes.  This
module keeps the approved-window validation, exact frame decoding, per-vial
aggregation, and summary arithmetic independently testable without a GPU.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from statistics import median

import numpy as np

from . import fill_level_ruler


EXPECTED_WINDOWS_SHA256 = (
    "19174a4b706fe896b177ba71c8ba2bdf850ceb9cf3a40522dcdaa42d53904229"
)
EXPECTED_RECIPES_SHA256 = (
    "2157df0b154ec356c0c2518758e4b426271a48cc2906f712f28663d4cb039291"
)

APPROVED_VIALS = tuple(range(2, 10)) + (11, 12) + tuple(range(14, 31))
CALIBRATION_VIALS = (13,)
INELIGIBLE_VIALS = (1, 10)
TRANSITION_VIALS = (4, 7, 16, 19, 22, 25, 28)

# These inclusive spans are the already-settled windows in the frozen CSV.
# The 0.3-second skip was applied when that CSV was created and must not be
# applied a second time by the decoder.
APPROVED_SPANS = {
    2: (2219, 2237, 19),
    3: (3625, 3631, 7),
    4: (4849, 4865, 17),
    5: (6102, 6123, 22),
    6: (7401, 7409, 9),
    7: (9250, 9282, 33),
    8: (10839, 10874, 36),
    9: (12437, 12490, 54),
    11: (15858, 15895, 38),
    12: (17606, 17659, 54),
    14: (21789, 21841, 53),
    15: (23935, 23989, 55),
    16: (25689, 25753, 65),
    17: (27417, 27468, 52),
    18: (29107, 29170, 64),
    19: (30938, 30996, 59),
    20: (32777, 32822, 46),
    21: (34608, 34661, 54),
    22: (36628, 36697, 70),
    23: (38706, 38769, 64),
    24: (40805, 40843, 39),
    25: (42719, 42841, 123),
    26: (44629, 44741, 113),
    27: (46557, 46702, 146),
    28: (48044, 48097, 54),
    29: (49408, 49480, 73),
    30: (50824, 50874, 51),
}

APPROVED_VOLUMES_ML = {
    2: 30.0,
    3: 30.0,
    4: 40.0,
    5: 40.0,
    6: 40.0,
    7: 70.0,
    8: 70.0,
    9: 70.0,
    11: 60.0,
    12: 60.0,
    14: 80.0,
    15: 80.0,
    16: 75.0,
    17: 75.0,
    18: 75.0,
    19: 70.0,
    20: 70.0,
    21: 70.0,
    22: 70.0,
    23: 70.0,
    24: 70.0,
    25: 70.0,
    26: 70.0,
    27: 70.0,
    28: 70.0,
    29: 70.0,
    30: 70.0,
}

WINDOWS_FIELDS = (
    "video_id",
    "vial",
    "recipe_id",
    "prev_recipe_id",
    "recipe_changed",
    "end_s",
    "departure_after_end_s",
    "window_start_s",
    "window_end_s",
    "start_frame",
    "end_frame",
    "n_frames",
    "role",
    "eligible",
    "exclusion_reason",
)
RECIPES_FIELDS = (
    "vial",
    "recipe_id",
    "type",
    "t100_ml",
    "t200_ml",
    "t400_ml",
    "total_ml",
    "dispense_order",
    "color_purpose",
    "start_time",
    "end_time",
    "start_s",
    "end_s",
    "duration_s",
    "start_frame",
    "end_frame",
)
STATUS_KEYS = ("visible", "hidden", "abstain", "no_clean_frame")


@dataclass(frozen=True)
class RealWindow:
    video_id: int
    vial: int
    recipe_id: int
    previous_recipe_id: int
    recipe_changed: bool
    transition_or_repeat: str
    zone: str
    start_frame: int
    end_frame: int
    frame_count: int
    recipe_volume_ml: float


class FrameDecodeError(ValueError):
    """A decode failure carrying the number of frames already exposed."""

    def __init__(self, message, *, frames_decoded=0):
        super().__init__(message)
        self.frames_decoded = int(frames_decoded)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_frozen_hash(path, supplied, expected, label):
    if str(supplied).lower() != expected:
        raise ValueError(f"{label} SHA-256 argument is not the frozen value")
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _read_csv(path, fields, label):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"{label} CSV header does not match the frozen schema")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{label} CSV must not be empty")
    return rows


def _integer(row, name, label):
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} {name} must be an integer") from error
    return value


def _number(row, name, label):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} {name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} {name} must be finite")
    return value


def _rows_by_vial(rows, label):
    result = {}
    for index, row in enumerate(rows, start=2):
        vial = _integer(row, "vial", f"{label} row {index}")
        if vial in result:
            raise ValueError(f"{label} contains duplicate vial {vial}")
        result[vial] = row
    if set(result) != set(range(1, 31)):
        raise ValueError(f"{label} must contain exactly vials 1 through 30")
    return result


def _zone_for_volume(volume_ml):
    if volume_ml == 60.0:
        return "hidden"
    if volume_ml == 70.0:
        return "edge"
    if volume_ml in (30.0, 40.0, 75.0, 80.0):
        return "visible"
    raise ValueError(f"unsupported frozen real-check volume {volume_ml:g} mL")


def validate_real_inputs(
    windows_path,
    expected_windows_hash,
    recipes_path,
    expected_recipes_hash,
):
    """Validate the frozen metadata and return only the approved 27 windows."""

    _require_frozen_hash(
        windows_path,
        expected_windows_hash,
        EXPECTED_WINDOWS_SHA256,
        "windows CSV",
    )
    _require_frozen_hash(
        recipes_path,
        expected_recipes_hash,
        EXPECTED_RECIPES_SHA256,
        "recipes CSV",
    )
    windows_by_vial = _rows_by_vial(
        _read_csv(windows_path, WINDOWS_FIELDS, "windows"), "windows CSV"
    )
    recipes_by_vial = _rows_by_vial(
        _read_csv(recipes_path, RECIPES_FIELDS, "recipes"), "recipes CSV"
    )

    if windows_by_vial[1]["eligible"] != "0" or windows_by_vial[1][
        "exclusion_reason"
    ] != "previous_recipe_unknown":
        raise ValueError("vial 1 must remain ineligible because its predecessor is unknown")
    if windows_by_vial[10]["eligible"] != "0" or windows_by_vial[10][
        "exclusion_reason"
    ] != "window_too_short":
        raise ValueError("vial 10 must remain ineligible because its window has two frames")
    if windows_by_vial[13]["eligible"] != "1":
        raise ValueError("vial 13 must remain metadata-eligible and reserved for calibration")

    approved = []
    for vial in APPROVED_VIALS:
        label = f"windows vial {vial}"
        row = windows_by_vial[vial]
        recipe = recipes_by_vial[vial]
        video_id = _integer(row, "video_id", label)
        recipe_id = _integer(row, "recipe_id", label)
        previous_recipe_id = _integer(row, "prev_recipe_id", label)
        recipe_changed_value = _integer(row, "recipe_changed", label)
        eligible = _integer(row, "eligible", label)
        start_frame = _integer(row, "start_frame", label)
        end_frame = _integer(row, "end_frame", label)
        frame_count = _integer(row, "n_frames", label)
        recipe_volume_ml = _number(recipe, "total_ml", f"recipes vial {vial}")

        if video_id != 5 or row["role"] != "real_validation":
            raise ValueError(f"{label} is not an approved video-5 real-validation row")
        if eligible != 1 or row["exclusion_reason"]:
            raise ValueError(f"{label} must be eligible with no exclusion reason")
        if recipe_id != _integer(recipe, "recipe_id", f"recipes vial {vial}"):
            raise ValueError(f"{label} recipe ID does not match the recipe ledger")
        if recipe_volume_ml != APPROVED_VOLUMES_ML[vial]:
            raise ValueError(f"recipes vial {vial} total volume is not frozen")
        if (start_frame, end_frame, frame_count) != APPROVED_SPANS[vial]:
            raise ValueError(f"{label} inclusive frame span is not frozen")
        if frame_count != end_frame - start_frame + 1 or frame_count < 3:
            raise ValueError(f"{label} must have an inclusive window of at least 3 frames")
        if not math.isclose(
            _number(row, "window_start_s", label) - _number(row, "end_s", label),
            0.3,
            abs_tol=0.001,
        ):
            raise ValueError(f"{label} does not contain the frozen 0.3-second settled skip")

        recipe_changed = recipe_changed_value == 1
        if recipe_changed_value not in (0, 1):
            raise ValueError(f"{label} recipe_changed must be 0 or 1")
        if recipe_changed != (vial in TRANSITION_VIALS):
            raise ValueError(f"{label} transition/repeat classification is not frozen")
        if recipe_changed != (recipe_id != previous_recipe_id):
            raise ValueError(f"{label} recipe change disagrees with predecessor IDs")

        approved.append(
            RealWindow(
                video_id=video_id,
                vial=vial,
                recipe_id=recipe_id,
                previous_recipe_id=previous_recipe_id,
                recipe_changed=recipe_changed,
                transition_or_repeat="transition" if recipe_changed else "repeat",
                zone=_zone_for_volume(recipe_volume_ml),
                start_frame=start_frame,
                end_frame=end_frame,
                frame_count=frame_count,
                recipe_volume_ml=recipe_volume_ml,
            )
        )

    zone_counts = {
        name: sum(item.zone == name for item in approved)
        for name in ("visible", "edge", "hidden")
    }
    transition_counts = {
        name: sum(item.transition_or_repeat == name for item in approved)
        for name in ("transition", "repeat")
    }
    if zone_counts != {"visible": 10, "edge": 15, "hidden": 2}:
        raise ValueError("approved zone counts do not match the frozen 27-vial design")
    if transition_counts != {"transition": 7, "repeat": 20}:
        raise ValueError("approved transition/repeat counts do not match the frozen design")
    frame_count = sum(item.frame_count for item in approved)
    if frame_count != 1470:
        raise ValueError("approved real-check windows must contain exactly 1,470 frames")
    return {
        "windows": approved,
        "frame_count": frame_count,
        "zone_counts": zone_counts,
        "transition_or_repeat_counts": transition_counts,
        "excluded_vials": {
            "ineligible": list(INELIGIBLE_VIALS),
            "calibration": list(CALIBRATION_VIALS),
        },
    }


def bgr_to_rgb(bgr):
    """Convert an OpenCV BGR frame to an independent contiguous RGB array."""

    frame = np.asarray(bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"decoded frame must have three BGR channels; got {tuple(frame.shape)}")
    if frame.dtype != np.uint8:
        raise ValueError(f"decoded frame must use uint8 pixels; got {frame.dtype}")
    return np.ascontiguousarray(frame[:, :, ::-1]).copy()


def decode_window(capture, window, cv2):
    """Seek once and decode exactly the approved inclusive frame window."""

    if not capture.isOpened():
        raise FrameDecodeError("video capture is not open")
    if window.frame_count != window.end_frame - window.start_frame + 1:
        raise FrameDecodeError("window frame count is not inclusive")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(window.start_frame)):
        raise FrameDecodeError(f"failed to seek to frame {window.start_frame}")
    position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if not math.isfinite(position) or abs(position - window.start_frame) > 0.01:
        raise FrameDecodeError(
            f"inexact seek for frame {window.start_frame}: capture reports {position}"
        )

    decoded = []
    for frame_index in range(window.start_frame, window.end_frame + 1):
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise FrameDecodeError(
                f"failed to decode approved frame {frame_index}",
                frames_decoded=len(decoded),
            )
        if getattr(bgr, "shape", None) != (480, 640, 3):
            raise FrameDecodeError(
                f"approved frame {frame_index} must be 640x480 BGR; "
                f"got {getattr(bgr, 'shape', None)}",
                frames_decoded=len(decoded),
            )
        try:
            rgb = bgr_to_rgb(bgr)
        except Exception as error:
            wrapped = FrameDecodeError(
                f"approved frame {frame_index} must be 640x480 uint8 BGR: {error}",
                frames_decoded=len(decoded),
            )
            raise wrapped from error
        decoded.append({"frame_index": frame_index, "rgb": rgb})
        next_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not math.isfinite(next_position) or abs(next_position - (frame_index + 1)) > 0.01:
            raise FrameDecodeError(
                f"capture position drift after frame {frame_index}: {next_position}",
                frames_decoded=len(decoded),
            )
    return decoded


def _finite(value, description):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _window_metadata(window):
    return {
        "video_id": int(window.video_id),
        "vial": int(window.vial),
        "recipe_id": int(window.recipe_id),
        "previous_recipe_id": int(window.previous_recipe_id),
        "recipe_changed": bool(window.recipe_changed),
        "transition_or_repeat": window.transition_or_repeat,
        "zone": window.zone,
        "start_frame": int(window.start_frame),
        "end_frame": int(window.end_frame),
        "window_frames": int(window.frame_count),
        "recipe_volume_ml": float(window.recipe_volume_ml),
    }


def aggregate_vial(window, frame_results, ruler):
    """Apply the strict-majority rule and aggregate one vial without tuning."""

    results = list(frame_results)
    base = _window_metadata(window)
    counts = {name: 0 for name in ("visible", "hidden", "abstain")}
    seen = set()
    edge_flagged = 0
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"frame result {index} must be an object")
        try:
            frame_index = int(result["frame_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"frame result {index} has no integer frame_index") from error
        if frame_index in seen:
            raise ValueError(f"duplicate frame result {frame_index}")
        if not window.start_frame <= frame_index <= window.end_frame:
            raise ValueError(f"frame result {frame_index} is outside vial {window.vial}")
        seen.add(frame_index)
        status = result.get("status")
        if status not in counts:
            raise ValueError(f"frame result {frame_index} has unsupported status {status!r}")
        counts[status] += 1
        edge_flagged += bool(result.get("edge_flag", False))

    base.update(
        {
            "decoded_frames": len(results),
            "frame_status_counts": counts,
            "edge_flagged_frames": int(edge_flagged),
        }
    )
    if len(results) < 3:
        return {**base, "status": "no_clean_frame", "frames_used": 0}

    majority_status = next(
        (name for name, count in counts.items() if count > len(results) / 2.0),
        "abstain",
    )
    if majority_status == "abstain":
        return {**base, "status": "abstain", "frames_used": 0}
    selected = [item for item in results if item["status"] == majority_status]

    if majority_status == "visible":
        rows = [
            _finite(item.get("liquid_top_row"), "visible liquid_top_row")
            for item in selected
        ]
        surface_row = float(median(rows))
        predicted_ml = fill_level_ruler.row_to_ml(ruler, surface_row)
        signed_error_ml = predicted_ml - window.recipe_volume_ml
        return {
            **base,
            "status": "visible",
            "frames_used": len(selected),
            "raw_surface_row": surface_row,
            "predicted_ml": predicted_ml,
            "signed_error_ml": signed_error_ml,
            "absolute_error_ml": abs(signed_error_ml),
        }

    top_rows = [
        _finite(item.get("range_top_row"), "hidden range_top_row")
        for item in selected
    ]
    bottom_rows = [
        _finite(item.get("range_bottom_row"), "hidden range_bottom_row")
        for item in selected
    ]
    top_row = float(median(top_rows))
    bottom_row = float(median(bottom_rows))
    if top_row > bottom_row:
        raise ValueError("hidden median top row must not be below its bottom row")
    predicted_range = fill_level_ruler.row_range_to_ml(
        ruler, [top_row, bottom_row]
    )
    low_ml, high_ml = predicted_range
    recipe = window.recipe_volume_ml
    in_range = low_ml <= recipe <= high_ml
    distance = 0.0 if in_range else min(abs(recipe - low_ml), abs(recipe - high_ml))
    return {
        **base,
        "status": "hidden",
        "frames_used": len(selected),
        "raw_row_range": [top_row, bottom_row],
        "predicted_ml_range": predicted_range,
        "range_width_ml": high_ml - low_ml,
        "recipe_in_range": in_range,
        "distance_to_range_ml": float(distance),
    }


def _percentile(values, quantile):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _summary_group(results):
    status_counts = {
        name: sum(item["status"] == name for item in results) for name in STATUS_KEYS
    }
    visible = [item for item in results if item["status"] == "visible"]
    hidden = [item for item in results if item["status"] == "hidden"]
    errors = [float(item["absolute_error_ml"]) for item in visible]
    signed = [float(item["signed_error_ml"]) for item in visible]
    distances = [float(item["distance_to_range_ml"]) for item in hidden]
    widths = [float(item["range_width_ml"]) for item in hidden]
    value = {
        "vials": len(results),
        "window_frames": sum(int(item["window_frames"]) for item in results),
        "decoded_frames": sum(int(item["decoded_frames"]) for item in results),
        "frames_used": sum(int(item["frames_used"]) for item in results),
        "edge_flagged_frames": sum(int(item["edge_flagged_frames"]) for item in results),
        "status_counts": status_counts,
        "visible_count": status_counts["visible"],
        "hidden_count": status_counts["hidden"],
        "abstain_count": status_counts["abstain"],
        "no_clean_frame_count": status_counts["no_clean_frame"],
        "visible_absolute_error_median_ml": float(median(errors)) if errors else None,
        "visible_absolute_error_p95_ml": _percentile(errors, 95),
        "visible_absolute_error_max_ml": max(errors) if errors else None,
        "visible_signed_error_median_ml": float(median(signed)) if signed else None,
        "hidden_recipe_in_range_count": sum(
            bool(item["recipe_in_range"]) for item in hidden
        ),
        "hidden_distance_to_range_median_ml": (
            float(median(distances)) if distances else None
        ),
        "hidden_distance_to_range_max_ml": max(distances) if distances else None,
        "hidden_range_width_median_ml": float(median(widths)) if widths else None,
        "hidden_range_width_max_ml": max(widths) if widths else None,
    }
    for threshold in (1, 2, 5):
        count = sum(error <= threshold for error in errors)
        value[f"visible_within_{threshold}_ml_count"] = count
        value[f"visible_within_{threshold}_ml_rate"] = (
            count / len(visible) if visible else None
        )
    value["hidden_recipe_in_range_rate"] = (
        value["hidden_recipe_in_range_count"] / len(hidden) if hidden else None
    )
    return value


def summarize_vials(per_vial):
    """Summarize final per-vial records overall and by frozen design strata."""

    results = list(per_vial)
    for index, item in enumerate(results):
        if not isinstance(item, dict) or item.get("status") not in STATUS_KEYS:
            raise ValueError(f"per-vial result {index} has an unsupported status")
    return {
        "overall": _summary_group(results),
        "by_zone": {
            zone: _summary_group([item for item in results if item["zone"] == zone])
            for zone in ("visible", "edge", "hidden")
        },
        "by_transition_or_repeat": {
            group: _summary_group(
                [item for item in results if item["transition_or_repeat"] == group]
            )
            for group in ("transition", "repeat")
        },
    }

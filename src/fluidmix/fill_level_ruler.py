"""Pure helpers for fitting and applying the frozen FillCam row-to-volume ruler.

Step 8 supplies one settled-frame median at 30 mL and one at 80 mL.  Those
points define the affine conversion used by the held-out twin test.  Image
decoding and Torch inference live in the command-line script so these contracts
remain testable without a GPU installation.
"""

from __future__ import annotations

import math
from statistics import median


CALIBRATION_VOLUMES_ML = {1: 30.0, 13: 80.0}
CALIBRATION_FRAME_SPANS = {1: (816, 843), 13: (19700, 19769)}
TWIN_REFERENCE_ROWS = {30.0: 329.0, 80.0: 177.0}
REFERENCE_ROWS_PER_ML = 3.04
MIN_VISIBLE_FRAMES = 3


def _finite_number(value, description):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def median_visible_row(frame_results, *, min_visible=MIN_VISIBLE_FRAMES):
    """Return the median visible row and explicit rule-status counts.

    A hidden range is never silently converted to a point estimate.  Only
    records whose frozen surface-rule status is ``visible`` contribute.
    """

    try:
        minimum = int(min_visible)
    except (TypeError, ValueError) as error:
        raise ValueError("min_visible must be an integer") from error
    if minimum < 3:
        raise ValueError("calibration requires at least three visible frames")

    results = list(frame_results)
    if not results:
        raise ValueError("calibration frame results must not be empty")
    counts = {"visible": 0, "hidden": 0, "abstain": 0}
    rows = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"frame result {index} must be an object")
        status = result.get("status")
        if status not in counts:
            raise ValueError(f"frame result {index} has unsupported status {status!r}")
        counts[status] += 1
        if status == "visible":
            rows.append(
                _finite_number(
                    result.get("liquid_top_row"), f"frame result {index} liquid_top_row"
                )
            )
    if len(rows) < minimum:
        raise ValueError(
            f"calibration needs at least {minimum} visible rows; found {len(rows)}"
        )
    return {
        "frame_count": len(results),
        "status_counts": counts,
        "visible_row_count": len(rows),
        "median_row": float(median(rows)),
        "visible_row_min": float(min(rows)),
        "visible_row_max": float(max(rows)),
    }


def fit_affine_ruler(calibration_rows):
    """Fit ``volume_ml = slope_ml_per_row * row + intercept_ml``."""

    if not isinstance(calibration_rows, dict) or set(calibration_rows) != set(
        CALIBRATION_VOLUMES_ML
    ):
        raise ValueError("calibration rows must contain exactly vials 1 and 13")
    low_row = _finite_number(calibration_rows[1], "vial 1 median row")
    high_row = _finite_number(calibration_rows[13], "vial 13 median row")
    if low_row <= high_row:
        raise ValueError("30 mL row must be below the 80 mL row")

    low_ml = CALIBRATION_VOLUMES_ML[1]
    high_ml = CALIBRATION_VOLUMES_ML[13]
    slope = (high_ml - low_ml) / (high_row - low_row)
    intercept = low_ml - slope * low_row
    rows_per_ml = abs(1.0 / slope)
    result = {
        "formula": "volume_ml = slope_ml_per_row * row + intercept_ml",
        "slope_ml_per_row": float(slope),
        "intercept_ml": float(intercept),
        "rows_per_ml": float(rows_per_ml),
        "ml_per_row": float(abs(slope)),
        "calibration_points": [
            {"vial": 1, "volume_ml": low_ml, "median_row": low_row},
            {"vial": 13, "volume_ml": high_ml, "median_row": high_row},
        ],
    }
    result["crosscheck"] = crosscheck_ruler(result)
    return result


def row_to_ml(ruler, row):
    slope = _finite_number(ruler.get("slope_ml_per_row"), "ruler slope")
    intercept = _finite_number(ruler.get("intercept_ml"), "ruler intercept")
    return float(slope * _finite_number(row, "surface row") + intercept)


def row_range_to_ml(ruler, rows):
    if not isinstance(rows, (list, tuple)) or len(rows) != 2:
        raise ValueError("row range must contain top and bottom rows")
    values = [row_to_ml(ruler, row) for row in rows]
    return [float(min(values)), float(max(values))]


def ml_to_row(ruler, volume_ml):
    slope = _finite_number(ruler.get("slope_ml_per_row"), "ruler slope")
    intercept = _finite_number(ruler.get("intercept_ml"), "ruler intercept")
    if slope == 0.0:
        raise ValueError("ruler slope must be non-zero")
    return float((_finite_number(volume_ml, "volume") - intercept) / slope)


def crosscheck_ruler(ruler):
    rows_per_ml = _finite_number(ruler.get("rows_per_ml"), "rows_per_ml")
    targets = []
    for volume_ml, reference_row in sorted(TWIN_REFERENCE_ROWS.items()):
        fitted_row = ml_to_row(ruler, volume_ml)
        targets.append(
            {
                "volume_ml": volume_ml,
                "twin_reference_row": reference_row,
                "fitted_row": fitted_row,
                "delta_rows": fitted_row - reference_row,
                "absolute_delta_rows": abs(fitted_row - reference_row),
            }
        )
    return {
        "reference_rows_per_ml": REFERENCE_ROWS_PER_ML,
        "rows_per_ml_delta": rows_per_ml - REFERENCE_ROWS_PER_ML,
        "rows_per_ml_absolute_delta": abs(rows_per_ml - REFERENCE_ROWS_PER_ML),
        "twin_reference_points": targets,
        "diagnostic_only": True,
    }

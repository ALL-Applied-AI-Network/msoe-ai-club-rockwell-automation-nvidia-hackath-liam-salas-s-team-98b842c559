"""Evidence contracts for the Step 8 Revision 2 real-look render pilot.

The pilot remains synthetic.  Real calibration frames are used only to measure
appearance and, when explicitly requested, to build an ignored visual montage.
Only the two already-opened representative frames named in
``ALLOWED_REAL_MONTAGE_FRAMES`` may be decoded by this module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw


EXPECTED_FRAME_COUNT = 20
EXPECTED_ELIGIBLE_COUNT = 19
EXPECTED_PROFILE = "real-look-v2"
EXPECTED_PILOT_GRID = (
    (28001, 0.0),
    (28002, 15.0),
    (28003, 30.0),
    (28025, 39.5),
    (28005, 40.1),
    (28006, 44.0),
    (28007, 47.9),
    (28008, 48.1),
    (28009, 50.0),
    (28010, 55.0),
    (28011, 60.0),
    (28012, 65.0),
    (28013, 66.5),
    (28014, 68.0),
    (28015, 70.0),
    (28016, 70.5),
    (28017, 72.0),
    (28018, 75.0),
    (28019, 80.0),
    (28020, 85.0),
)
LOW_V_THRESHOLD = 0.2
LOW_V_COUNT_RANGE = (6, 8)
REAL_LIQUID_MEAN_V = (0.088, 0.155)
SATURATED_S_RANGE = (0.30, 0.85)
SATURATED_COUNT_RANGE = (9, 10)
DARKEST_TAIL_FRACTION = 0.10
DARKEST_TAIL_V_RANGE = (0.05, 0.09)
REAL_HOLDER_V = 0.18
BACKDROP_NEUTRAL_TARGET = {"luma": 0.66, "saturation": 0.04}
EXPECTED_GLASS_IOR = 1.46
EXPECTED_GLASS_F0 = (
    (EXPECTED_GLASS_IOR - 1.0) / (EXPECTED_GLASS_IOR + 1.0)
) ** 2
EXPECTED_GLASS_PARAMS = {
    "diffuse_rgb": [0.92, 0.96, 0.98],
    "opacity": 0.14,
    "roughness": 0.08,
    "ior": EXPECTED_GLASS_IOR,
    "specular_rgb": [EXPECTED_GLASS_F0] * 3,
    "use_specular_workflow": 1.0,
    "metallic": 0.0,
}
ALLOWED_MASK_IDS = {0, 1, 2, 3, 255}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SCENE_METRIC_NAMES = (
    "scene_grayscale_p05",
    "scene_grayscale_p50",
    "scene_grayscale_p95",
    "scene_robust_contrast",
    "scene_grayscale_std",
    "backdrop_median_luma",
    "backdrop_luma_p05",
    "backdrop_luma_p95",
    "backdrop_robust_contrast",
    "backdrop_median_saturation",
    "backdrop_median_r",
    "backdrop_median_g",
    "backdrop_median_b",
    "backdrop_tint_r_minus_g",
    "backdrop_tint_b_minus_g",
    "blur_laplacian_variance_8bit",
    "blur_tenengrad_normalized",
    "noise_normalized_sigma",
    "haze_backdrop_luma_floor_p05",
    "haze_backdrop_local_robust_contrast",
)
STICKER_METRIC_NAMES = (
    "bbox_width_px",
    "bbox_height_px",
    "center_y_px",
    "grayscale_p10",
    "grayscale_p25",
    "grayscale_p50",
    "grayscale_p75",
    "grayscale_p90",
    "dark_print_fraction_proxy",
    "paper_fraction_proxy",
    "median_r",
    "median_g",
    "median_b",
)
ALLOWED_REAL_MONTAGE_FRAMES = (
    {
        "path": "vial-01-030ml/frame-000830.png",
        "frame_index": 830,
        "vial": 1,
        "level_ml": 30.0,
    },
    {
        "path": "vial-13-080ml/frame-019735.png",
        "frame_index": 19735,
        "vial": 13,
        "level_ml": 80.0,
    },
)
ALLOWED_REAL_SURFACE_ZOOMS = {
    830: [277, 329, 341, 345],
    19735: [277, 177, 341, 200],
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT_CHECKER_PATH = REPOSITORY_ROOT / "scripts" / "check_fill_level_capture.py"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_with_independent_checker(manifest_path):
    """Run the repository capture checker and return deterministic evidence.

    The checker is loaded from its source file instead of copied into this module,
    so this pilot analysis cannot silently drift to a weaker duplicate contract.
    Path-bearing fields are omitted from the embedded report to keep the final
    evidence deterministic across safe copy destinations.
    """
    checker_path = INDEPENDENT_CHECKER_PATH.resolve()
    if not checker_path.is_file():
        raise ValueError(
            f"independent capture checker does not exist: {checker_path}"
        )
    specification = importlib.util.spec_from_file_location(
        "_fluidmix_fill_level_capture_checker", checker_path
    )
    if specification is None or specification.loader is None:
        raise ValueError(
            f"could not load independent capture checker: {checker_path}"
        )
    checker = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(checker)
    except Exception as error:
        raise ValueError(
            f"could not import independent capture checker: {error}"
        ) from error
    validator = getattr(checker, "validate_manifest", None)
    if not callable(validator):
        raise ValueError(
            "independent capture checker has no validate_manifest API"
        )
    try:
        raw_report = validator(Path(manifest_path).resolve())
    except Exception as error:
        raise ValueError(
            f"independent capture checker raised an exception: {error}"
        ) from error
    if not isinstance(raw_report, dict):
        raise ValueError("independent capture checker returned no report object")
    records = raw_report.get("records")
    record_count = raw_report.get("record_count")
    if not isinstance(records, list):
        raise ValueError("independent capture checker report has no record list")
    if record_count != EXPECTED_FRAME_COUNT or len(records) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            "independent capture checker did not report exactly "
            f"{EXPECTED_FRAME_COUNT} records"
        )
    errors = raw_report.get("errors")
    if not isinstance(errors, list):
        raise ValueError("independent capture checker report has no error list")
    if raw_report.get("passed") is not True or errors:
        detail = "; ".join(str(error) for error in errors[:5])
        if len(errors) > 5:
            detail += f"; ... ({len(errors) - 5} more)"
        raise ValueError(
            "independent capture checker did not pass"
            + (f": {detail}" if detail else "")
        )
    normalized_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"independent capture checker record {index} is not an object"
            )
        record_errors = record.get("errors")
        if record_errors != []:
            raise ValueError(
                f"independent capture checker record {index} contains errors"
            )
        normalized_records.append(
            {
                "index": record.get("index"),
                "seed": record.get("seed"),
                "role": record.get("role"),
                "label_counts": record.get("label_counts"),
                "errors": [],
            }
        )
    return {
        "checker_path": "scripts/check_fill_level_capture.py",
        "checker_sha256": file_sha256(checker_path),
        "status": "passed",
        "passed": True,
        "record_count": record_count,
        "report": {
            "schema_version": raw_report.get("schema_version"),
            "record_count": record_count,
            "errors": [],
            "passed": True,
            "records": normalized_records,
        },
    }


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _finite(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _sha(value, description):
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise ValueError(f"{description} must be a SHA-256")
    return value.lower()


def _safe_child(root, relative, description):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{description} path must be a non-empty string")
    candidate = (Path(root) / relative).resolve()
    root = Path(root).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} path escapes its evidence root") from error
    if not candidate.is_file():
        raise ValueError(f"{description} file does not exist: {relative}")
    return candidate


def _descriptor_path(root, descriptor, description):
    if not isinstance(descriptor, dict):
        raise ValueError(f"{description} descriptor must be an object")
    path = _safe_child(root, descriptor.get("path"), description)
    expected = _sha(descriptor.get("sha256"), f"{description} expected hash")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{description} SHA-256 mismatch")
    return path


def _roi(value, image_shape, description):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{description} must be [x0, y0, x1, y1]")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{description} coordinates must be integers")
    x0, y0, x1, y1 = value
    height, width = image_shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{description} lies outside the image")
    return x0, y0, x1, y1


def _rgb_hsv(rgb):
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    difference = maximum - minimum
    saturation = np.divide(
        difference,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    hue = np.zeros_like(maximum)
    active = difference > 0
    red = active & (maximum == values[..., 0])
    green = active & (maximum == values[..., 1])
    blue = active & (maximum == values[..., 2])
    hue[red] = ((values[..., 1][red] - values[..., 2][red]) / difference[red]) % 6
    hue[green] = (values[..., 2][green] - values[..., 0][green]) / difference[green] + 2
    hue[blue] = (values[..., 0][blue] - values[..., 1][blue]) / difference[blue] + 4
    hue /= 6.0
    return np.stack((hue, saturation, maximum), axis=-1)


def _luma(rgb):
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    return (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )


def _laplacian(gray):
    gray = np.asarray(gray, dtype=np.float64)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        raise ValueError("scene ROI must be at least 3x3 pixels")
    return (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )


def _circular_hue_mean(hue, saturation):
    hue = np.asarray(hue, dtype=np.float64).ravel()
    weights = np.asarray(saturation, dtype=np.float64).ravel()
    if float(weights.sum()) <= 1e-12:
        return 0.0
    angle = hue * 2.0 * np.pi
    value = math.atan2(
        float(np.sum(np.sin(angle) * weights)),
        float(np.sum(np.cos(angle) * weights)),
    )
    return float((value / (2.0 * np.pi)) % 1.0)


def _circular_hue_distance(first, second):
    """Return the shortest normalized distance between two hue angles."""
    absolute = abs(float(first) - float(second)) % 1.0
    return float(min(absolute, 1.0 - absolute))


def liquid_hsv_metrics(rgb, liquid_mask):
    """Measure rendered liquid HSV without assigning a hue to grey pixels."""
    rgb = np.asarray(rgb)
    selected = np.asarray(liquid_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be an HxWx3 uint8 array")
    if selected.shape != rgb.shape[:2]:
        raise ValueError("liquid mask dimensions differ from RGB")
    pixel_count = int(np.count_nonzero(selected))
    if not pixel_count:
        return {
            "pixel_count": 0,
            "mean_hue_circular_saturation_weighted_0_to_1": None,
            "mean_saturation": None,
            "mean_value": None,
            "saturation_weight_sum": 0.0,
        }
    hsv = _rgb_hsv(rgb)[selected]
    saturation_weight_sum = float(np.sum(hsv[:, 1]))
    mean_hue = (
        _circular_hue_mean(hsv[:, 0], hsv[:, 1])
        if saturation_weight_sum > 1e-12
        else None
    )
    return {
        "pixel_count": pixel_count,
        "mean_hue_circular_saturation_weighted_0_to_1": mean_hue,
        "mean_saturation": float(np.mean(hsv[:, 1])),
        "mean_value": float(np.mean(hsv[:, 2])),
        "saturation_weight_sum": saturation_weight_sum,
    }


def _material_rendered_hsv_comparison(rendered, material_hsv):
    material_hue, material_saturation, material_value = (
        float(value) for value in material_hsv
    )
    rendered_hue = rendered[
        "mean_hue_circular_saturation_weighted_0_to_1"
    ]
    rendered_saturation = rendered["mean_saturation"]
    return {
        "material_hue_0_to_1": material_hue,
        "material_saturation": material_saturation,
        "material_value": material_value,
        "rendered_hue_0_to_1": rendered_hue,
        "rendered_mean_saturation": rendered_saturation,
        "rendered_mean_value": rendered["mean_value"],
        "circular_hue_error": (
            _circular_hue_distance(rendered_hue, material_hue)
            if rendered_hue is not None and material_saturation > 1e-12
            else None
        ),
        "saturation_retention_ratio": (
            float(rendered_saturation / material_saturation)
            if rendered_saturation is not None and material_saturation > 1e-12
            else None
        ),
    }


def holder_proxy_metrics(rgb, mask, sticker_bbox=None):
    """Measure blocked non-sticker pixels as an explicit holder proxy.

    The semantic mask does not separate holder, probe, and other blocked
    geometry. This is a documented proxy, not a holder segmentation claim.
    """
    rgb = np.asarray(rgb)
    mask = np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be an HxWx3 uint8 array")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("holder proxy mask dimensions differ from RGB")
    selected = mask == 3
    original_count = int(np.count_nonzero(selected))
    excluded_count = 0
    if sticker_bbox is not None:
        x0, y0, x1, y1 = _roi(
            list(sticker_bbox), rgb.shape, "holder proxy sticker bbox"
        )
        excluded_count = int(np.count_nonzero(selected[y0:y1, x0:x1]))
        selected = selected.copy()
        selected[y0:y1, x0:x1] = False
    pixel_count = int(np.count_nonzero(selected))
    result = {
        "proxy_name": "decoded_blocked_non_sticker",
        "blocked_pixel_count_before_sticker_exclusion": original_count,
        "sticker_bbox_excluded_blocked_pixel_count": excluded_count,
        "pixel_count": pixel_count,
        "mean_value": None,
        "median_value": None,
        "p05_value": None,
        "p95_value": None,
    }
    if pixel_count:
        values = _rgb_hsv(rgb)[..., 2][selected]
        result.update(
            {
                "mean_value": float(np.mean(values)),
                "median_value": float(np.median(values)),
                "p05_value": float(np.quantile(values, 0.05)),
                "p95_value": float(np.quantile(values, 0.95)),
            }
        )
    return result


def glass_boundary_proxy_metrics(rgb, mask, vial_bbox):
    """Measure edge/highlight proxies on the one-pixel semantic vial boundary."""
    rgb = np.asarray(rgb)
    mask = np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be an HxWx3 uint8 array")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("glass proxy mask dimensions differ from RGB")
    if vial_bbox is None:
        return None
    x0, y0, x1, y1 = _roi(list(vial_bbox), rgb.shape, "vial bbox")
    semantic_vial = np.zeros(mask.shape, dtype=np.uint8)
    semantic_vial[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] != 255).astype(
        np.uint8
    )
    eroded = cv2.erode(
        semantic_vial, np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    boundary = (semantic_vial > 0) & (eroded == 0)
    pixel_count = int(np.count_nonzero(boundary))
    if not pixel_count:
        return None
    values = _rgb_hsv(rgb)[..., 2][boundary]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)[boundary]
    return {
        "proxy_name": "semantic_vial_one_pixel_boundary",
        "bbox_xyxy_half_open": [x0, y0, x1, y1],
        "pixel_count": pixel_count,
        "value_p05": float(np.quantile(values, 0.05)),
        "value_median": float(np.median(values)),
        "value_p95": float(np.quantile(values, 0.95)),
        "value_p95_minus_p05": float(
            np.quantile(values, 0.95) - np.quantile(values, 0.05)
        ),
        "highlight_fraction_v_ge_0_80": float(np.mean(values >= 0.80)),
        "gradient_magnitude_mean": float(np.mean(gradient)),
        "gradient_magnitude_p90": float(np.quantile(gradient, 0.90)),
    }


def backdrop_neutral_target_comparison(scene):
    """Compare one measured backdrop ROI with the review's neutral target."""
    luma = float(scene["backdrop_median_luma"])
    saturation = float(scene["backdrop_median_saturation"])
    return {
        "target_median_luma": BACKDROP_NEUTRAL_TARGET["luma"],
        "target_median_saturation": BACKDROP_NEUTRAL_TARGET["saturation"],
        "observed_median_luma": luma,
        "observed_median_saturation": saturation,
        "absolute_luma_error": abs(luma - BACKDROP_NEUTRAL_TARGET["luma"]),
        "absolute_saturation_error": abs(
            saturation - BACKDROP_NEUTRAL_TARGET["saturation"]
        ),
    }


def _darkest_tail_summary(frames):
    """Summarize whether the actual darkest ten-percent tail reaches 0.05--0.09."""
    eligible = [frame for frame in frames if frame["liquid_mean_v"] is not None]
    target_count = (
        max(1, int(math.floor(len(eligible) * DARKEST_TAIL_FRACTION + 0.5)))
        if eligible
        else 0
    )
    ordered = sorted(
        eligible,
        key=lambda frame: (
            frame["liquid_mean_v"],
            frame.get("index", 0),
            frame["seed"],
        ),
    )
    low, high = DARKEST_TAIL_V_RANGE
    range_members = [
        frame for frame in ordered if low <= frame["liquid_mean_v"] <= high
    ]
    darkest = ordered[:target_count]
    darkest_all_in_range = bool(darkest) and all(
        low <= frame["liquid_mean_v"] <= high for frame in darkest
    )
    no_values_below_range = all(
        frame["liquid_mean_v"] >= low for frame in ordered
    )
    range_count_matches = len(range_members) == target_count
    return {
        "review_source": "2026-11-01 ~12:25 first batch chunk check",
        "eligible_count": len(eligible),
        "target_fraction": DARKEST_TAIL_FRACTION,
        "target_count_nearest_integer": target_count,
        "target_v_range_inclusive": [low, high],
        "observed_count_in_target_range": len(range_members),
        "observed_count_below_target_range": sum(
            frame["liquid_mean_v"] < low for frame in ordered
        ),
        "darkest_target_frames": [
            {"seed": frame["seed"], "mean_value": frame["liquid_mean_v"]}
            for frame in darkest
        ],
        "target_range_members": [
            {"seed": frame["seed"], "mean_value": frame["liquid_mean_v"]}
            for frame in range_members
        ],
        "gates": {
            "darkest_target_frames_all_in_0_05_to_0_09": darkest_all_in_range,
            "no_rendered_liquid_below_0_05": no_values_below_range,
            "target_range_count_equals_nearest_10_percent_count": (
                range_count_matches
            ),
            "passed": bool(
                darkest_all_in_range
                and no_values_below_range
                and range_count_matches
            ),
        },
    }


def scene_metrics(rgb, scene_rois):
    """Measure the declared scene and backdrop ROIs using frozen formulas."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be an HxWx3 uint8 array")
    if not isinstance(scene_rois, dict):
        raise ValueError("scene_rois must be an object")
    rows = scene_rois.get("scene_rows_half_open")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in rows)
    ):
        raise ValueError("scene rows must be a two-integer half-open range")
    sy0, sy1 = rows
    if not 0 <= sy0 < sy1 <= rgb.shape[0]:
        raise ValueError("scene rows lie outside the image")
    backdrop_box = _roi(scene_rois.get("backdrop_xyxy"), rgb.shape, "backdrop ROI")
    bx0, by0, bx1, by1 = backdrop_box
    scene_rgb = rgb[sy0:sy1]
    backdrop_rgb = rgb[by0:by1, bx0:bx1]
    scene_gray = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    backdrop_gray = cv2.cvtColor(
        backdrop_rgb, cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0
    backdrop = backdrop_rgb.astype(np.float32) / 255.0
    backdrop_hsv = cv2.cvtColor(backdrop_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    backdrop_hsv[..., 1:] /= 255.0
    residual = backdrop_gray - cv2.GaussianBlur(backdrop_gray, (0, 0), 1.5)
    residual_median = float(np.median(residual))
    laplacian = cv2.Laplacian(
        np.rint(scene_gray * 255.0).astype(np.uint8), cv2.CV_32F
    )
    gradient_x = cv2.Sobel(scene_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(scene_gray, cv2.CV_32F, 0, 1, ksize=3)
    backdrop_luma = np.dot(backdrop, [0.2126, 0.7152, 0.0722])
    scene_p05 = float(np.quantile(scene_gray, 0.05))
    scene_p50 = float(np.median(scene_gray))
    scene_p95 = float(np.quantile(scene_gray, 0.95))
    backdrop_p05 = float(np.quantile(backdrop_luma, 0.05))
    backdrop_p95 = float(np.quantile(backdrop_luma, 0.95))
    backdrop_median_rgb = np.median(backdrop.reshape(-1, 3), axis=0)
    return {
        "scene_grayscale_p05": scene_p05,
        "scene_grayscale_p50": scene_p50,
        "scene_grayscale_p95": scene_p95,
        "scene_robust_contrast": scene_p95 - scene_p05,
        "scene_grayscale_std": float(np.std(scene_gray)),
        "backdrop_median_luma": float(np.median(backdrop_luma)),
        "backdrop_luma_p05": backdrop_p05,
        "backdrop_luma_p95": backdrop_p95,
        "backdrop_robust_contrast": backdrop_p95 - backdrop_p05,
        "backdrop_median_saturation": float(np.median(backdrop_hsv[..., 1])),
        "backdrop_median_r": float(backdrop_median_rgb[0]),
        "backdrop_median_g": float(backdrop_median_rgb[1]),
        "backdrop_median_b": float(backdrop_median_rgb[2]),
        "backdrop_tint_r_minus_g": float(
            backdrop_median_rgb[0] - backdrop_median_rgb[1]
        ),
        "backdrop_tint_b_minus_g": float(
            backdrop_median_rgb[2] - backdrop_median_rgb[1]
        ),
        "blur_laplacian_variance_8bit": float(np.var(laplacian)),
        "blur_tenengrad_normalized": float(
            np.mean(gradient_x**2 + gradient_y**2)
        ),
        "noise_normalized_sigma": 1.4826
        * float(np.median(np.abs(residual - residual_median))),
        "haze_backdrop_luma_floor_p05": backdrop_p05,
        "haze_backdrop_local_robust_contrast": backdrop_p95 - backdrop_p05,
    }


def _range(value, description):
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object with min/max")
    low = _finite(value.get("min"), f"{description}.min")
    high = _finite(value.get("max"), f"{description}.max")
    if low > high:
        raise ValueError(f"{description}.min exceeds max")
    return low, high


def compare_range(values, real_range):
    values = [_finite(value, "synthetic metric") for value in values]
    if not values:
        return {
            "available": False,
            "synthetic_count": 0,
            "real_range": list(_range(real_range, "real metric range")),
            "overlaps_real": None,
            "spans_real": None,
        }
    real_low, real_high = _range(real_range, "real metric range")
    synthetic_low = float(min(values))
    synthetic_high = float(max(values))
    return {
        "available": True,
        "synthetic_count": len(values),
        "synthetic_range": [synthetic_low, synthetic_high],
        "real_range": [real_low, real_high],
        "overlaps_real": bool(
            synthetic_high >= real_low and synthetic_low <= real_high
        ),
        "spans_real": bool(
            synthetic_low <= real_low and synthetic_high >= real_high
        ),
    }


def _stat_range(value, description):
    if isinstance(value, dict):
        return {
            "min": _finite(value.get("min"), f"{description}.min"),
            "max": _finite(value.get("max"), f"{description}.max"),
        }
    scalar = _finite(value, description)
    return {"min": scalar, "max": scalar}


def validate_real_appearance(real):
    schema_version = real.get("schema_version")
    labels = {
        1: "measured_on_real_calibration_vials_1_and_13_only",
        2: "measured_real_appearance_without_real_model_training",
    }
    if schema_version not in labels:
        raise ValueError("real appearance uses an unsupported schema")
    if real.get("evidence_label") != labels[schema_version]:
        raise ValueError("real appearance evidence label does not match its schema")
    provenance = real.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("real appearance has no provenance")
    if provenance.get("frame_count") != 98 or provenance.get("allowed_vials") != [1, 13]:
        raise ValueError("real appearance must contain only 98 vial-1/vial-13 frames")
    if provenance.get("vial_counts") != {"1": 28, "13": 70}:
        raise ValueError("real appearance vial counts are invalid")
    _sha(provenance.get("index_sha256"), "real appearance calibration index hash")
    if real.get("other_real_frames_opened") != 0:
        raise ValueError("real appearance reports opening other real frames")
    if schema_version == 2:
        if real.get("additional_video5_frames_opened") != 0:
            raise ValueError(
                "expanded real appearance reports additional video-5 frames"
            )
        expected_legacy_definition = (
            "legacy video-5-only counter: no video-5 calibration or real-check "
            "frame outside the approved closed 98-frame vial-1/vial-13 set was "
            "opened; videos-1--4 appearance-reference counts are recorded separately"
        )
        if real.get("other_real_frames_opened_definition") != (
            expected_legacy_definition
        ):
            raise ValueError(
                "expanded real appearance does not define its legacy video-5 counter"
            )
        calibration_surface_foam = real.get("calibration_top_surface_and_foam")
        if not isinstance(calibration_surface_foam, dict):
            raise ValueError(
                "expanded real appearance has no calibration surface/foam evidence"
            )
        if calibration_surface_foam.get("surface_oval_height_reference_px") != [
            17,
            24,
        ] or any(
            not isinstance(calibration_surface_foam.get(name), dict)
            for name in ("vial_1_30ml", "vial_13_80ml")
        ):
            raise ValueError(
                "expanded calibration surface/foam evidence is incomplete"
            )
        external = real.get("external_reference_sample")
        external_provenance = (
            external.get("provenance") if isinstance(external, dict) else None
        )
        if not isinstance(external_provenance, dict):
            raise ValueError("expanded real appearance has no external provenance")
        expected_external_contract = {
            "video5_paths_resolved_hashed_or_decoded": 0,
            "video5_vial_colors_csv_opened": 0,
            "candidate_pair_count": 68,
            "candidate_image_count_hashed_before_first_decode": 136,
            "vial_candidate_frames_decoded": 68,
            "tank_selected_frames_decoded": 34,
            "selected_pair_count": 34,
            "allowed_video_ids": [1, 2, 3, 4],
            "forbidden_video_ids": [5],
        }
        for name, expected in expected_external_contract.items():
            if external_provenance.get(name) != expected:
                raise ValueError(
                    f"expanded real appearance external provenance {name} is invalid"
                )
        for name in (
            "vial_candidate_allowlist_sha256",
            "tank_candidate_allowlist_sha256",
        ):
            _sha(
                external_provenance.get(name),
                f"expanded real appearance {name}",
            )
        for name, count_name in (
            ("candidate_pairs", "candidate_pair_count"),
            ("selected_pairs", "selected_pair_count"),
        ):
            records = external_provenance.get(name)
            if not isinstance(records, list) or len(records) != (
                external_provenance[count_name]
            ):
                raise ValueError(
                    f"expanded real appearance external provenance {name} is invalid"
                )
        vial_fill = external.get("vial_fill")
        tank_level = external.get("tank_level_dye_colors")
        if not isinstance(vial_fill, dict) or not isinstance(tank_level, dict):
            raise ValueError("expanded real appearance external sections are missing")
        if (
            vial_fill.get("selected_nominal_cycle_count") != 34
            or not isinstance(vial_fill.get("selection_records"), list)
            or len(vial_fill["selection_records"]) != 34
            or not 0 <= vial_fill.get("settled_proxy_pass_count", -1) <= 34
        ):
            raise ValueError("expanded vial-fill selection counts are invalid")
        for name in (
            "liquid",
            "labels",
            "top_surface_and_foam",
            "global_look",
            "settled_proxy",
        ):
            if not isinstance(vial_fill.get(name), dict):
                raise ValueError(f"expanded vial-fill section {name} is missing")
        if vial_fill["liquid"].get("colored_or_dark_settled_count") != 10:
            raise ValueError("expanded vial-fill liquid count is invalid")
        if not isinstance(tank_level.get("tanks"), dict) or len(
            tank_level["tanks"]
        ) != 4:
            raise ValueError("expanded tank dye-color evidence is incomplete")
        guidance = real.get("renderer_guidance")
        if not isinstance(guidance, dict) or any(
            not isinstance(guidance.get(name), dict)
            for name in ("liquid", "labels", "dark_liquid_top_surface")
        ):
            raise ValueError("expanded real appearance renderer guidance is incomplete")
    liquids = real.get("liquids")
    if not isinstance(liquids, dict):
        raise ValueError("real appearance has no liquid measurements")
    try:
        observed = sorted(
            (
                _finite(
                    liquids["vial_1_30ml"]["per_frame_value"]["mean"],
                    "vial-1 liquid mean V",
                ),
                _finite(
                    liquids["vial_13_80ml"]["per_frame_value"]["mean"],
                    "vial-13 liquid mean V",
                ),
            )
        )
    except (KeyError, TypeError) as error:
        raise ValueError("real appearance liquid measurements are incomplete") from error
    if len(observed) != 2 or not np.allclose(
        observed, sorted(REAL_LIQUID_MEAN_V), rtol=0.0, atol=1e-4
    ):
        raise ValueError("real liquid mean V values do not match the reviewed evidence")
    global_look = real.get("global_look")
    brady = real.get("brady_label")
    if not isinstance(global_look, dict) or not isinstance(brady, dict):
        raise ValueError("real appearance global look/BRADY label is incomplete")
    try:
        scene = global_look["scene_without_timestamp"]
        backdrop = global_look["backdrop"]
        blur = global_look["blur_proxies"]
        noise = global_look["noise_proxy"]
        haze = global_look["haze_proxy"]
        scene_rois = {
            "scene_rows_half_open": list(scene["rows_half_open"]),
            "backdrop_xyxy": list(backdrop["roi_xyxy_half_open"]),
        }
        scene_metrics = {
            "scene_grayscale_p05": _stat_range(scene["grayscale_p05"], "scene p05"),
            "scene_grayscale_p50": _stat_range(scene["grayscale_p50"], "scene p50"),
            "scene_grayscale_p95": _stat_range(scene["grayscale_p95"], "scene p95"),
            "scene_robust_contrast": _stat_range(
                scene["robust_contrast_p95_minus_p05"], "scene robust contrast"
            ),
            "scene_grayscale_std": _stat_range(
                scene["grayscale_standard_deviation"], "scene grayscale std"
            ),
            "backdrop_median_luma": _stat_range(backdrop["median_luma"], "backdrop luma"),
            "backdrop_luma_p05": _stat_range(backdrop["luma_p05"], "backdrop p05"),
            "backdrop_luma_p95": _stat_range(backdrop["luma_p95"], "backdrop p95"),
            "backdrop_robust_contrast": _stat_range(
                backdrop["robust_contrast_p95_minus_p05"], "backdrop robust contrast"
            ),
            "backdrop_median_saturation": _stat_range(
                backdrop["median_saturation"], "backdrop saturation"
            ),
            "backdrop_median_r": _stat_range(backdrop["median_rgb"]["r"], "backdrop red"),
            "backdrop_median_g": _stat_range(backdrop["median_rgb"]["g"], "backdrop green"),
            "backdrop_median_b": _stat_range(backdrop["median_rgb"]["b"], "backdrop blue"),
            "backdrop_tint_r_minus_g": _stat_range(
                backdrop["tint_r_minus_g"], "backdrop red-green tint"
            ),
            "backdrop_tint_b_minus_g": _stat_range(
                backdrop["tint_b_minus_g"], "backdrop blue-green tint"
            ),
            "blur_laplacian_variance_8bit": _stat_range(
                blur["laplacian_variance_8bit"], "blur Laplacian variance"
            ),
            "blur_tenengrad_normalized": _stat_range(
                blur["tenengrad_normalized"], "blur Tenengrad"
            ),
            "noise_normalized_sigma": _stat_range(
                noise["normalized_sigma"], "noise sigma"
            ),
            "haze_backdrop_luma_floor_p05": _stat_range(
                haze["backdrop_luma_floor_p05"], "haze luma floor"
            ),
            "haze_backdrop_local_robust_contrast": _stat_range(
                haze["backdrop_local_robust_contrast"], "haze local contrast"
            ),
        }
        sticker = {
            "bbox_width_px": _stat_range(brady["width_px"], "BRADY width"),
            "bbox_height_px": _stat_range(brady["height_px"], "BRADY height"),
            "center_y_px": _stat_range(brady["center_xy"][1], "BRADY center y"),
            "grayscale_p10": _stat_range(brady["grayscale_p10"], "BRADY p10"),
            "grayscale_p25": _stat_range(brady["grayscale_p25"], "BRADY p25"),
            "grayscale_p50": _stat_range(brady["grayscale_p50"], "BRADY p50"),
            "grayscale_p75": _stat_range(brady["grayscale_p75"], "BRADY p75"),
            "grayscale_p90": _stat_range(brady["grayscale_p90"], "BRADY p90"),
            "dark_print_fraction_proxy": _stat_range(
                brady["dark_print_fraction_proxy"], "BRADY dark fraction"
            ),
            "paper_fraction_proxy": _stat_range(
                brady["paper_fraction_proxy"], "BRADY paper fraction"
            ),
            "median_r": _stat_range(brady["median_rgb"]["r"], "BRADY median red"),
            "median_g": _stat_range(brady["median_rgb"]["g"], "BRADY median green"),
            "median_b": _stat_range(brady["median_rgb"]["b"], "BRADY median blue"),
        }
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("real appearance look measurements are incomplete") from error
    normalized = dict(real)
    normalized["scene_rois"] = scene_rois
    normalized["scene_metrics"] = scene_metrics
    normalized["sticker"] = sticker
    return normalized


def _load_manifest(path):
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read pilot manifest: {error}") from error
    for line_number, raw in enumerate(lines, 1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"manifest line {line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        records.append(value)
    if len(records) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"pilot manifest must contain {EXPECTED_FRAME_COUNT} records"
        )
    identities = []
    for index, (record, expected) in enumerate(zip(records, EXPECTED_PILOT_GRID)):
        if record.get("role") != "pilot":
            raise ValueError(f"manifest record {index} is not a pilot record")
        if record.get("variation_profile") != EXPECTED_PROFILE:
            raise ValueError(f"manifest record {index} has the wrong appearance profile")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"manifest record {index} has an invalid seed")
        level_ml = _finite(record.get("level_ml"), f"manifest record {index} level_ml")
        expected_seed, expected_level_ml = expected
        if seed != expected_seed or not math.isclose(
            level_ml, expected_level_ml, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "manifest does not match the exact planned pilot seed/level grid "
                f"at record {index}: expected {expected_seed}:{expected_level_ml}"
            )
        identities.append((record["role"], seed))
    if len(identities) != len(set(identities)):
        raise ValueError("pilot manifest contains duplicate role/seed records")
    return records


def _validate_summary(
    summary,
    summary_path,
    manifest_path,
    *,
    expected_specs_sha256=None,
):
    if summary.get("frame_count") != EXPECTED_FRAME_COUNT:
        raise ValueError("render summary frame count does not match the pilot")
    if summary.get("passed") is not True:
        raise ValueError("render summary did not pass")
    if summary.get("source_unchanged") is not True:
        raise ValueError("render summary says the source changed")
    if summary.get("integrity_passed") is not True:
        raise ValueError("render summary integrity did not pass")
    if summary.get("real_look_requested") is not True:
        raise ValueError("render summary did not request the real-look profile")
    manifest_hash = file_sha256(manifest_path)
    if summary.get("manifest_sha256") != manifest_hash:
        raise ValueError("render summary manifest hash mismatch")
    glass = _validate_exact_glass_params(
        summary.get("active_vial_glass_material_params")
    )
    for name in (
        "renderer_script_sha256",
        "variation_module_sha256",
        "specs_sha256",
        "session_layer_sha256",
    ):
        _sha(summary.get(name), f"render summary {name}")
    normalized_expected_specs = None
    if expected_specs_sha256 is not None:
        normalized_expected_specs = _sha(
            expected_specs_sha256, "expected pilot specs hash"
        )
        if summary["specs_sha256"].lower() != normalized_expected_specs:
            raise ValueError("render summary specs hash does not match the intended pilot")
    return {
        "summary_sha256": file_sha256(summary_path),
        "manifest_sha256": manifest_hash,
        "renderer_script_sha256": summary["renderer_script_sha256"],
        "variation_module_sha256": summary["variation_module_sha256"],
        "specs_sha256": summary["specs_sha256"],
        "expected_specs_sha256": normalized_expected_specs,
        "specs_match_expected": (
            True if normalized_expected_specs is not None else None
        ),
        "session_layer_sha256": summary["session_layer_sha256"],
        "active_vial_glass_material_params": glass,
        "source_unchanged": True,
    }


def _validate_exact_glass_params(value):
    """Require the complete v5 clear-glass shader contract from the render summary."""
    if not isinstance(value, dict):
        raise ValueError("render summary lacks the clear active-vial glass override")
    normalized = {}
    for name, expected in EXPECTED_GLASS_PARAMS.items():
        actual = value.get(name)
        if isinstance(expected, list):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                raise ValueError(f"glass {name} must contain {len(expected)} channels")
            channels = [
                _finite(channel, f"glass {name} channel") for channel in actual
            ]
            if name == "specular_rgb" and max(channels) > 0.06:
                raise ValueError(
                    "glass specular_rgb exceeds the dielectric F0 guard"
                )
            if any(
                not math.isclose(channel, target, rel_tol=0.0, abs_tol=1e-12)
                for channel, target in zip(channels, expected)
            ):
                raise ValueError(f"glass {name} does not match the v5 contract")
            normalized[name] = channels
        else:
            number = _finite(actual, f"glass {name}")
            if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"glass {name} does not match the v5 contract")
            normalized[name] = (
                int(number) if name == "use_specular_workflow" else number
            )
    return normalized


def _sticker_bbox(record):
    candidates = (
        record.get("sticker_bbox_xyxy"),
        record.get("label_stats", {}).get("sticker_bbox_xyxy")
        if isinstance(record.get("label_stats"), dict)
        else None,
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _bbox_metrics(bbox, image_shape):
    x0, y0, x1, y1 = _roi(bbox, image_shape, "sticker bbox")
    return {
        "bbox_width_px": float(x1 - x0),
        "bbox_height_px": float(y1 - y0),
        "center_y_px": float((y0 + y1) / 2.0),
    }


def _sticker_appearance(rgb, sticker_mask):
    selected = sticker_mask > 0
    if int(np.count_nonzero(selected)) < 2:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    pixels = gray[selected]
    rgb_values = rgb.astype(np.float32)[selected] / 255.0
    result = {
        "grayscale_p10": float(np.quantile(pixels, 0.10)),
        "grayscale_p25": float(np.quantile(pixels, 0.25)),
        "grayscale_p50": float(np.quantile(pixels, 0.50)),
        "grayscale_p75": float(np.quantile(pixels, 0.75)),
        "grayscale_p90": float(np.quantile(pixels, 0.90)),
        "dark_print_fraction_proxy": float(np.mean(pixels <= 0.30)),
        "paper_fraction_proxy": float(np.mean(pixels >= 0.50)),
        "median_r": float(np.median(rgb_values[:, 0])),
        "median_g": float(np.median(rgb_values[:, 1])),
        "median_b": float(np.median(rgb_values[:, 2])),
    }
    unique = set(int(value) for value in np.unique(sticker_mask))
    if 1 in unique and 2 in unique:
        value = _rgb_hsv(rgb)[..., 2]
        paper = value[sticker_mask == 1]
        printed = value[sticker_mask == 2]
        method = "explicit sticker mask ids paper=1 print=2"
    else:
        value_pixels = _rgb_hsv(rgb)[..., 2][selected]
        low = float(np.quantile(value_pixels, 0.30))
        high = float(np.quantile(value_pixels, 0.70))
        printed = value_pixels[value_pixels <= low]
        paper = value_pixels[value_pixels >= high]
        method = "binary sticker mask lower/upper 30 percent V"
    if paper.size and printed.size:
        paper_mean = float(np.mean(paper))
        print_mean = float(np.mean(printed))
        result.update(
            {
                "paper_v_mean": paper_mean,
                "print_v_mean": print_mean,
                "paper_print_contrast": paper_mean - print_mean,
                "paper_print_method": method,
            }
        )
    return result


def _validate_images(pilot_dir, records, scene_rois):
    frames = []
    for index, record in enumerate(records):
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"manifest record {index} has no outputs")
        rgb_path = _descriptor_path(
            pilot_dir, outputs.get("rgb"), f"record {index} RGB"
        )
        mask_path = _descriptor_path(
            pilot_dir, outputs.get("mask"), f"record {index} mask"
        )
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError(f"record {index} mask must be 2-D uint8")
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"record {index} RGB/mask dimensions differ")
        unexpected = {int(value) for value in np.unique(mask)} - ALLOWED_MASK_IDS
        if unexpected:
            raise ValueError(
                f"record {index} mask has unexpected classes {sorted(unexpected)}"
            )
        label_stats = record.get("label_stats")
        class_pixels = (
            label_stats.get("class_pixels")
            if isinstance(label_stats, dict)
            else None
        )
        if not isinstance(class_pixels, dict):
            raise ValueError(f"record {index} has no class-pixel integrity counts")
        sticker_blocked_pixels = label_stats.get("sticker_blocked_pixels")
        if (
            isinstance(sticker_blocked_pixels, bool)
            or not isinstance(sticker_blocked_pixels, int)
            or sticker_blocked_pixels < 0
        ):
            raise ValueError(
                f"record {index} sticker_blocked_pixels must be a "
                "non-negative integer"
            )
        for name, identifier in (
            ("empty", 0),
            ("liquid", 1),
            ("foam", 2),
            ("blocked", 3),
            ("ignore", 255),
        ):
            observed_count = class_pixels.get(name)
            expected_count = int(np.count_nonzero(mask == identifier))
            if observed_count != expected_count:
                raise ValueError(
                    f"record {index} class-pixel count mismatch for {name}"
                )
        level_ml = _finite(record.get("level_ml"), f"record {index} level_ml")
        liquid_metrics = liquid_hsv_metrics(rgb, mask == 1)
        liquid_pixel_count = liquid_metrics["pixel_count"]
        liquid_mean_v = liquid_metrics["mean_value"]
        seed = int(record["seed"])
        if seed == 28001 and liquid_pixel_count:
            raise ValueError(
                "planned zero-level seed 28001 has decoded liquid pixels"
            )
        if seed != 28001 and not liquid_pixel_count:
            raise ValueError(
                f"positive-level planned seed {seed} has no decoded liquid pixels"
            )
        surface_metrics = None
        surface_bbox = None
        if liquid_pixel_count:
            liquid_y, liquid_x = np.nonzero(mask == 1)
            top = int(liquid_y.min())
            configured_height = record.get("surface", {}).get(
                "rendered_oval_height_px", 20
            ) if isinstance(record.get("surface"), dict) else 20
            if (
                isinstance(configured_height, bool)
                or not isinstance(configured_height, (int, float))
            ):
                raise ValueError(f"record {index} surface oval height is invalid")
            height = max(1, min(40, int(round(float(configured_height)))))
            bottom = min(mask.shape[0], top + height)
            band = (mask == 1) & (
                np.arange(mask.shape[0], dtype=np.int32)[:, None] >= top
            ) & (np.arange(mask.shape[0], dtype=np.int32)[:, None] < bottom)
            band_y, band_x = np.nonzero(band)
            if band_x.size:
                surface_bbox = [
                    int(band_x.min()),
                    int(band_y.min()),
                    int(band_x.max()) + 1,
                    int(band_y.max()) + 1,
                ]
                surface_hsv = _rgb_hsv(rgb)[band]
                grey_reflective = (
                    (surface_hsv[:, 1] <= 0.20) & (surface_hsv[:, 2] >= 0.20)
                )
                surface_metrics = {
                    "bbox_xyxy_half_open": surface_bbox,
                    "band_height_px": int(surface_bbox[3] - surface_bbox[1]),
                    "mean_saturation": float(np.mean(surface_hsv[:, 1])),
                    "mean_value": float(np.mean(surface_hsv[:, 2])),
                    "median_value": float(np.median(surface_hsv[:, 2])),
                    "p90_value": float(np.quantile(surface_hsv[:, 2], 0.90)),
                    "p95_value": float(np.quantile(surface_hsv[:, 2], 0.95)),
                    "p90_minus_liquid_mean_v": float(
                        np.quantile(surface_hsv[:, 2], 0.90) - liquid_mean_v
                    ),
                    "achromatic_fraction_s_le_0_20": float(
                        np.mean(surface_hsv[:, 1] <= 0.20)
                    ),
                    "grey_reflective_fraction_s_le_0_20_v_ge_0_20": float(
                        np.mean(grey_reflective)
                    ),
                    "method": (
                        "top rendered-oval-height rows of decoded liquid-mask pixels"
                    ),
                }
        foam_record = record.get("foam")
        if not isinstance(foam_record, dict):
            raise ValueError(f"record {index} has no foam material")
        foam_enabled = foam_record.get("enabled")
        if not isinstance(foam_enabled, bool):
            raise ValueError(f"record {index} foam enabled flag is invalid")
        manifest_tint = foam_record.get("tint_rgb")
        if not isinstance(manifest_tint, list) or len(manifest_tint) != 3:
            raise ValueError(f"record {index} foam tint RGB is invalid")
        manifest_tint = [
            _finite(channel, f"record {index} foam tint RGB")
            for channel in manifest_tint
        ]
        if not all(0.0 <= channel <= 1.0 for channel in manifest_tint):
            raise ValueError(f"record {index} foam tint RGB is outside [0, 1]")
        if "authored_small_bubble_count" not in foam_record:
            raise ValueError(
                f"record {index} has no authored small-bubble count"
            )
        authored_bubble_count = foam_record["authored_small_bubble_count"]
        if (
            isinstance(authored_bubble_count, bool)
            or not isinstance(authored_bubble_count, int)
            or not 0 <= authored_bubble_count <= 10
        ):
            raise ValueError(
                f"record {index} authored small-bubble count is invalid"
            )
        expected_authored = foam_enabled and level_ml > 0.0
        if expected_authored and not 3 <= authored_bubble_count <= 10:
            raise ValueError(
                f"record {index} enabled foam needs 3..10 authored bubbles"
            )
        if not expected_authored and authored_bubble_count != 0:
            raise ValueError(
                f"record {index} disabled/empty foam needs zero authored bubbles"
            )
        foam_mask = mask == 2
        visible_foam = bool(np.any(foam_mask))
        if foam_record.get("visible_in_camera") is not visible_foam:
            raise ValueError(
                f"record {index} foam visibility disagrees with decoded mask"
            )
        foam_metrics = {
            "visible_in_decoded_mask": visible_foam,
            "pixel_count": int(np.count_nonzero(foam_mask)),
            "authored_small_bubble_count": authored_bubble_count,
            "authored_bubble_definition": (
                "small semantic foam spheres authored in 3-D; not guaranteed to be "
                "distinct camera-visible connected components"
            ),
            "manifest_tint_rgb": manifest_tint,
            "mask_connected_component_count": 0,
            "small_mask_component_count_2_to_64_px_proxy": 0,
            "mean_hue_circular_0_to_1": None,
            "mean_saturation": None,
            "mean_value": None,
            "grey_fraction_s_le_0_20": None,
            "liquid_hue_circular_0_to_1": None,
            "foam_liquid_circular_hue_distance": None,
        }
        foam_bbox = None
        if visible_foam:
            foam_ys, foam_xs = np.nonzero(foam_mask)
            foam_bbox = [
                int(foam_xs.min()),
                int(foam_ys.min()),
                int(foam_xs.max()) + 1,
                int(foam_ys.max()) + 1,
            ]
            foam_hsv = _rgb_hsv(rgb)[foam_mask]
            foam_hue = _circular_hue_mean(foam_hsv[:, 0], foam_hsv[:, 1])
            component_count, _labels, component_stats, _centroids = (
                cv2.connectedComponentsWithStats(
                    foam_mask.astype(np.uint8), connectivity=8
                )
            )
            component_areas = component_stats[1:, cv2.CC_STAT_AREA]
            small_component_count = int(
                np.count_nonzero((component_areas >= 2) & (component_areas <= 64))
            )
            liquid_hue = None
            hue_distance = None
            if liquid_pixel_count:
                liquid_hue = liquid_metrics[
                    "mean_hue_circular_saturation_weighted_0_to_1"
                ]
                if liquid_hue is not None:
                    hue_distance = _circular_hue_distance(foam_hue, liquid_hue)
            foam_metrics.update({
                "mask_connected_component_count": int(component_count - 1),
                "small_mask_component_count_2_to_64_px_proxy": (
                    small_component_count
                ),
                "mean_hue_circular_0_to_1": foam_hue,
                "mean_saturation": float(np.mean(foam_hsv[:, 1])),
                "mean_value": float(np.mean(foam_hsv[:, 2])),
                "grey_fraction_s_le_0_20": float(np.mean(foam_hsv[:, 1] <= 0.20)),
                "liquid_hue_circular_0_to_1": liquid_hue,
                "foam_liquid_circular_hue_distance": hue_distance,
            })
        liquid = record.get("liquid")
        if not isinstance(liquid, dict):
            raise ValueError(f"manifest record {index} has no liquid material")
        hsv = liquid.get("hsv")
        if not isinstance(hsv, list) or len(hsv) != 3:
            raise ValueError(f"manifest record {index} has invalid liquid HSV")
        material_hsv = [
            _finite(value, f"record {index} liquid HSV") for value in hsv
        ]
        hue_family = liquid.get("hue_family")
        if hue_family is not None and (
            not isinstance(hue_family, str) or not hue_family
        ):
            raise ValueError(f"manifest record {index} liquid hue family is invalid")
        material_opacity = _finite(
            liquid.get("opacity"), f"record {index} liquid opacity"
        )
        if not all(0.0 <= value <= 1.0 for value in material_hsv):
            raise ValueError(f"manifest record {index} liquid HSV is outside [0, 1]")
        if not 0.0 <= material_opacity <= 1.0:
            raise ValueError(f"manifest record {index} opacity is outside [0, 1]")
        liquid_comparison = _material_rendered_hsv_comparison(
            liquid_metrics, material_hsv
        )
        sticker_metrics = {}
        sticker_bbox = None
        bbox = _sticker_bbox(record)
        if bbox is not None:
            sticker_bbox = list(_roi(bbox, rgb.shape, "sticker bbox"))
            sticker_metrics.update(_bbox_metrics(sticker_bbox, rgb.shape))
        sticker_descriptor = outputs.get("sticker_mask")
        if sticker_descriptor is not None:
            sticker_path = _descriptor_path(
                pilot_dir,
                sticker_descriptor,
                f"record {index} sticker mask",
            )
            with Image.open(sticker_path) as image:
                sticker_mask = np.asarray(image)
            if sticker_mask.ndim != 2 or sticker_mask.shape != mask.shape:
                raise ValueError(f"record {index} sticker mask dimensions differ")
            appearance = _sticker_appearance(rgb, sticker_mask)
            if appearance:
                sticker_metrics.update(appearance)
            if bbox is None and np.any(sticker_mask):
                ys, xs = np.nonzero(sticker_mask)
                derived = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ]
                sticker_metrics.update(_bbox_metrics(derived, rgb.shape))
                sticker_bbox = derived
        vial_bbox = label_stats.get("vial_bbox_xyxy")
        if vial_bbox is not None:
            vial_bbox = list(_roi(vial_bbox, rgb.shape, "vial bbox"))
        holder_proxy = holder_proxy_metrics(rgb, mask, sticker_bbox)
        glass_proxy = glass_boundary_proxy_metrics(rgb, mask, vial_bbox)
        measured_scene = scene_metrics(rgb, scene_rois)
        frames.append(
            {
                "index": index,
                "seed": int(record["seed"]),
                "level_ml": level_ml,
                "rgb": rgb,
                "rgb_path": rgb_path,
                "liquid_pixel_count": liquid_pixel_count,
                "liquid_mean_v": liquid_mean_v,
                "liquid_hsv": liquid_metrics,
                "liquid_material_comparison": liquid_comparison,
                "material": {
                    "variation_profile": record["variation_profile"],
                    "hsv": material_hsv,
                    "hue_family": hue_family,
                    "opacity": material_opacity,
                },
                "sticker": sticker_metrics or None,
                "sticker_bbox": sticker_bbox,
                "sticker_blocked_pixels": sticker_blocked_pixels,
                "surface": surface_metrics,
                "surface_bbox": surface_bbox,
                "foam": foam_metrics,
                "foam_bbox": foam_bbox,
                "holder_proxy": holder_proxy,
                "glass_boundary_proxy": glass_proxy,
                "scene": measured_scene,
                "backdrop_neutral_target": (
                    backdrop_neutral_target_comparison(measured_scene)
                ),
            }
        )
    return frames


def _metric_comparisons(frames, real, category, names):
    result = {}
    real_metrics = real.get(category, {})
    for name in names:
        if name not in real_metrics:
            continue
        values = []
        for frame in frames:
            source = frame["scene"] if category == "scene_metrics" else frame["sticker"]
            if source and name in source:
                values.append(source[name])
        result[name] = compare_range(values, real_metrics[name])
    return result


def _validate_real_index(root):
    root = Path(root).expanduser().resolve()
    index_path = root / "index.json"
    index = _read_json(index_path, "allowed calibration-frame index")
    frames = index.get("frames")
    if index.get("frame_count") != 98 or not isinstance(frames, list) or len(frames) != 98:
        raise ValueError("allowed calibration-frame index must contain 98 records")
    by_path = {}
    vial_frames = {1: [], 13: []}
    for record in frames:
        if not isinstance(record, dict):
            raise ValueError("allowed calibration-frame index has a non-object record")
        if record.get("role") != "calibration" or record.get("source_window_role") != (
            "real_validation"
        ):
            raise ValueError("allowed real frame has the wrong role")
        if record.get("vial") not in (1, 13):
            raise ValueError("allowed calibration index contains another vial")
        path = record.get("path")
        if not isinstance(path, str) or path in by_path:
            raise ValueError("allowed calibration index has an invalid/duplicate path")
        by_path[path] = record
        vial_frames[record["vial"]].append(record.get("frame_index"))
    expected_windows = {
        1: list(range(816, 844)),
        13: list(range(19700, 19770)),
    }
    for vial, expected in expected_windows.items():
        observed = vial_frames[vial]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in observed)
            or sorted(observed) != expected
        ):
            raise ValueError(
                f"allowed calibration index vial {vial} window is incomplete"
            )
    selected = []
    for expected in ALLOWED_REAL_MONTAGE_FRAMES:
        record = by_path.get(expected["path"])
        if record is None:
            raise ValueError(f"allowed real frame is absent: {expected['path']}")
        for name in ("frame_index", "vial", "level_ml"):
            if record.get(name) != expected[name]:
                raise ValueError(
                    f"allowed real frame metadata mismatch for {expected['path']}"
                )
        image_path = _safe_child(root, record["path"], "allowed real frame")
        expected_hash = _sha(record.get("sha256"), "allowed real frame hash")
        observed_hash = file_sha256(image_path)
        if observed_hash != expected_hash:
            raise ValueError(f"allowed real frame hash mismatch: {expected['path']}")
        selected.append((record, image_path))
    return index_path, selected


def _fit_panel(image, box, size=(240, 240)):
    image = image.crop(box)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (32, 32, 32))
    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def _expanded_box(box, image_shape, *, horizontal=12, vertical=8):
    x0, y0, x1, y1 = box
    height, width = image_shape[:2]
    return (
        max(0, x0 - horizontal),
        max(0, y0 - vertical),
        min(width, x1 + horizontal),
        min(height, y1 + vertical),
    )


def _select_scene_montage_frames(frames):
    """Select deterministic scenes that expose darkness and manifested hues."""
    selected = []
    metadata = []
    by_index = {}

    def add(frame, reason):
        position = by_index.get(frame["index"])
        if position is None:
            position = len(selected)
            by_index[frame["index"]] = position
            selected.append(frame)
            metadata.append(
                {
                    "seed": frame["seed"],
                    "reasons": [reason],
                    "hue_family": frame["material"].get("hue_family"),
                    "material_hsv_v": frame["material"]["hsv"][2],
                    "liquid_mean_v": frame["liquid_mean_v"],
                }
            )
        elif reason not in metadata[position]["reasons"]:
            metadata[position]["reasons"].append(reason)
        return position

    no_liquid = sorted(
        (frame for frame in frames if frame["liquid_mean_v"] is None),
        key=lambda frame: (frame["index"], frame["seed"]),
    )
    if no_liquid:
        add(no_liquid[0], "no_liquid")
    eligible = sorted(
        (frame for frame in frames if frame["liquid_mean_v"] is not None),
        key=lambda frame: (
            frame["liquid_mean_v"],
            frame["index"],
            frame["seed"],
        ),
    )
    if not eligible:
        return selected, metadata
    add(eligible[0], "darkest_rendered_liquid")
    add(eligible[-1], "brightest_rendered_liquid")

    families = sorted(
        {
            frame["material"].get("hue_family")
            for frame in eligible
            if frame["material"].get("hue_family") is not None
        }
    )
    if families:
        for family in families:
            candidates = [
                frame
                for frame in eligible
                if frame["material"].get("hue_family") == family
            ]
            family_mid_value = float(
                np.median([frame["material"]["hsv"][2] for frame in candidates])
            )
            representative = min(
                candidates,
                key=lambda frame: (
                    round(
                        abs(
                            frame["material"]["hsv"][2]
                            - family_mid_value
                        ),
                        12,
                    ),
                    frame["index"],
                    frame["seed"],
                ),
            )
            position = add(representative, f"mid_value_hue_family:{family}")
            metadata[position]["hue_family_material_value_median"] = (
                family_mid_value
            )
            metadata[position]["hue_family_selection"] = (
                "closest material V to the manifested family median; ties use "
                "earliest manifest index then lowest seed"
            )
    else:
        # Legacy/schema-1 fixtures do not carry hue-family names.  Keep a
        # deterministic spread so their montage remains useful and testable.
        for index in (
            len(eligible) // 4,
            len(eligible) // 2,
            3 * len(eligible) // 4,
        ):
            add(eligible[index], f"rendered_value_quantile_index:{index}")
    return selected, metadata


def _select_surface_montage_frames(frames, scene_frames):
    """Add the brightest dark-liquid surface to the selected scene surfaces."""
    selected = []
    selected_indices = set()
    for frame in scene_frames:
        if frame["surface_bbox"] is not None and frame["index"] not in selected_indices:
            selected.append(frame)
            selected_indices.add(frame["index"])
    candidates = [
        frame
        for frame in frames
        if frame["surface_bbox"] is not None
        and frame["material"]["hsv"][2] <= 0.12
        and frame["surface"] is not None
        and frame["surface"].get("mean_value") is not None
    ]
    if not candidates:
        return selected, None
    outlier = max(
        candidates,
        key=lambda frame: (
            frame["surface"]["mean_value"],
            -frame["index"],
            -frame["seed"],
        ),
    )
    if outlier["index"] not in selected_indices:
        selected.append(outlier)
    return selected, {
        "seed": outlier["seed"],
        "surface_mean_value": outlier["surface"]["mean_value"],
        "selection": (
            "brightest mean-value surface among material HSV V <= 0.12; "
            "ties use earliest manifest index then lowest seed"
        ),
    }


def _select_foam_montage_frame(frames):
    """Choose one visible foam crop with the strongest bubble evidence."""
    candidates = [
        frame
        for frame in frames
        if frame["foam_bbox"] is not None
        and frame["foam"]["visible_in_decoded_mask"]
    ]
    if not candidates:
        return None, None
    selected = max(
        candidates,
        key=lambda frame: (
            frame["foam"]["small_mask_component_count_2_to_64_px_proxy"],
            frame["foam"]["authored_small_bubble_count"],
            frame["foam"]["pixel_count"],
            -frame["index"],
            -frame["seed"],
        ),
    )
    return selected, {
        "seed": selected["seed"],
        "bbox_xyxy_half_open": list(selected["foam_bbox"]),
        "authored_small_bubble_count": selected["foam"][
            "authored_small_bubble_count"
        ],
        "small_mask_component_count_2_to_64_px_proxy": selected["foam"][
            "small_mask_component_count_2_to_64_px_proxy"
        ],
        "foam_pixel_count": selected["foam"]["pixel_count"],
        "selection": (
            "most small decoded components, then authored bubbles, then foam "
            "pixels; ties use earliest manifest index then lowest seed"
        ),
    }


def _select_sticker_montage_frame(frames):
    candidates = [
        frame
        for frame in frames
        if frame["sticker_bbox"] is not None
        and frame["sticker_blocked_pixels"] > 0
    ]
    if not candidates:
        raise ValueError("pilot has no synthetic sticker bbox for montage")
    return max(
        candidates,
        key=lambda frame: (
            frame["sticker_blocked_pixels"],
            -frame["index"],
            -frame["seed"],
        ),
    )


def write_synthetic_montage(path, frames, scene_rois):
    """Write a compact, deterministic montage containing synthetic crops only."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite synthetic montage: {path}")
    scene_picks, scene_metadata = _select_scene_montage_frames(frames)
    surface_picks, surface_outlier = _select_surface_montage_frames(
        frames, scene_picks
    )
    foam_pick, foam_metadata = _select_foam_montage_frame(frames)
    sticker_pick = _select_sticker_montage_frame(frames)
    panels = []
    labels = []
    panel_sources = []
    panel_size = (180, 160)

    sticker_box = list(
        _roi(
            sticker_pick["sticker_bbox"],
            sticker_pick["rgb"].shape,
            "selected synthetic sticker bbox",
        )
    )
    panels.append(
        _fit_panel(
            Image.fromarray(sticker_pick["rgb"], mode="RGB"),
            _expanded_box(
                sticker_box,
                sticker_pick["rgb"].shape,
                horizontal=16,
                vertical=16,
            ),
            size=panel_size,
        )
    )
    labels.append(f"sticker seed {sticker_pick['seed']}")
    panel_sources.append("synthetic_sticker_crop")

    for frame in surface_picks:
        panels.append(
            _fit_panel(
                Image.fromarray(frame["rgb"], mode="RGB"),
                _expanded_box(frame["surface_bbox"], frame["rgb"].shape),
                size=panel_size,
            )
        )
        labels.append(f"surface seed {frame['seed']}")
        panel_sources.append("synthetic_surface_crop")
    if foam_pick is not None:
        panels.append(
            _fit_panel(
                Image.fromarray(foam_pick["rgb"], mode="RGB"),
                _expanded_box(
                    foam_pick["foam_bbox"],
                    foam_pick["rgb"].shape,
                    horizontal=24,
                    vertical=24,
                ),
                size=panel_size,
            )
        )
        labels.append(f"foam seed {foam_pick['seed']}")
        panel_sources.append("synthetic_foam_crop")
    for frame in scene_picks:
        first, last = scene_rois["scene_rows_half_open"]
        scene_box = _roi(
            [0, first, frame["rgb"].shape[1], last],
            frame["rgb"].shape,
            "synthetic scene ROI",
        )
        panels.append(
            _fit_panel(
                Image.fromarray(frame["rgb"], mode="RGB"),
                scene_box,
                size=panel_size,
            )
        )
        labels.append(f"scene seed {frame['seed']}")
        panel_sources.append("synthetic_scene_crop")

    columns = 4
    label_height = 24
    rows = int(math.ceil(len(panels) / columns))
    montage = Image.new(
        "RGB",
        (columns * panel_size[0], rows * (panel_size[1] + label_height)),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(montage)
    for panel_index, (panel, label) in enumerate(zip(panels, labels)):
        x = (panel_index % columns) * panel_size[0]
        y = (panel_index // columns) * (panel_size[1] + label_height)
        montage.paste(panel, (x, y))
        draw.text((x + 4, y + panel_size[1] + 4), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)
    return {
        "status": "created_from_synthetic_pilot_only",
        "sha256": file_sha256(path),
        "contains_real_frames": False,
        "raw_full_renders_included": False,
        "panel_count": len(panels),
        "layout_dimensions_px": list(montage.size),
        "panel_sources": panel_sources,
        "synthetic_scene_selections": scene_metadata,
        "synthetic_surface_zoom_seeds": [
            frame["seed"] for frame in surface_picks
        ],
        "synthetic_surface_outlier": surface_outlier,
        "synthetic_foam_crop": foam_metadata,
        "synthetic_sticker_crop": {
            "seed": sticker_pick["seed"],
            "bbox_xyxy_half_open": sticker_box,
            "sticker_blocked_pixels": sticker_pick["sticker_blocked_pixels"],
        },
    }


def write_montage(
    path,
    frames,
    scene_rois,
    allowed_frames_root,
    *,
    expected_index_sha256,
    real_sticker_bbox,
):
    """Write an ignored montage after validating/opening only two real frames."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite montage: {path}")
    index_path, real_frames = _validate_real_index(allowed_frames_root)
    observed_index_sha256 = file_sha256(index_path)
    if observed_index_sha256 != _sha(
        expected_index_sha256, "expected calibration index hash"
    ):
        raise ValueError("allowed calibration-frame index SHA-256 mismatch")
    picks, scene_selection_metadata = _select_scene_montage_frames(frames)
    surface_picks, surface_outlier_metadata = _select_surface_montage_frames(
        frames, picks
    )
    foam_pick, foam_selection_metadata = _select_foam_montage_frame(frames)
    scene_panels = []
    scene_labels = []
    surface_panels = []
    surface_labels = []
    foam_panels = []
    foam_labels = []
    sticker_panels = []
    sticker_labels = []
    for record, image_path in real_frames:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            first, last = scene_rois["scene_rows_half_open"]
            box = _roi(
                [0, first, rgb.width, last],
                (rgb.height, rgb.width, 3),
                "real scene ROI",
            )
            scene_panels.append(_fit_panel(rgb, box))
        scene_labels.append(
            f"real vial {record['vial']} frame {record['frame_index']}"
        )
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            surface_box = _roi(
                ALLOWED_REAL_SURFACE_ZOOMS[record["frame_index"]],
                (rgb.height, rgb.width, 3),
                "allowed real surface zoom",
            )
            surface_panels.append(_fit_panel(rgb, surface_box))
        surface_labels.append(f"real surface zoom frame {record['frame_index']}")
        if record["frame_index"] == 830:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                validated_sticker_bbox = _roi(
                    real_sticker_bbox,
                    (rgb.height, rgb.width, 3),
                    "frame-830 BRADY bbox",
                )
                sticker_panels.append(
                    _fit_panel(
                        rgb,
                        _expanded_box(
                            validated_sticker_bbox,
                            (rgb.height, rgb.width, 3),
                            horizontal=16,
                            vertical=16,
                        ),
                    )
                )
            sticker_labels.append("real frame 830 BRADY bbox crop")
    for frame in picks:
        rgb = Image.fromarray(frame["rgb"], mode="RGB")
        first, last = scene_rois["scene_rows_half_open"]
        box = _roi(
            [0, first, frame["rgb"].shape[1], last],
            frame["rgb"].shape,
            "synthetic scene ROI",
        )
        scene_panels.append(_fit_panel(rgb, box))
        mean_v_label = (
            f"{frame['liquid_mean_v']:.3f}"
            if frame["liquid_mean_v"] is not None
            else "no liquid"
        )
        rendered_hue = frame["liquid_hsv"][
            "mean_hue_circular_saturation_weighted_0_to_1"
        ]
        hue_label = f"{rendered_hue:.2f}" if rendered_hue is not None else "n/a"
        rendered_saturation = frame["liquid_hsv"]["mean_saturation"]
        saturation_label = (
            f"{rendered_saturation:.2f}"
            if rendered_saturation is not None
            else "n/a"
        )
        scene_labels.append(
            f"synthetic {frame['seed']} H/S/V {hue_label}/{saturation_label}/{mean_v_label}"
        )
    for frame in surface_picks:
        rgb = Image.fromarray(frame["rgb"], mode="RGB")
        surface_box = _expanded_box(frame["surface_bbox"], frame["rgb"].shape)
        surface_panels.append(_fit_panel(rgb, surface_box))
        surface_labels.append(f"synthetic surface zoom seed {frame['seed']}")
    if foam_pick is not None:
        foam_rgb = Image.fromarray(foam_pick["rgb"], mode="RGB")
        foam_panels.append(
            _fit_panel(
                foam_rgb,
                _expanded_box(
                    foam_pick["foam_bbox"],
                    foam_pick["rgb"].shape,
                    horizontal=24,
                    vertical=24,
                ),
            )
        )
        foam_labels.append(f"synthetic foam zoom seed {foam_pick['seed']}")
    synthetic_sticker = _select_sticker_montage_frame(frames)
    synthetic_sticker_bbox = list(
        _roi(
            synthetic_sticker["sticker_bbox"],
            synthetic_sticker["rgb"].shape,
            "selected synthetic sticker bbox",
        )
    )
    synthetic_sticker_rgb = Image.fromarray(synthetic_sticker["rgb"], mode="RGB")
    sticker_panels.append(
        _fit_panel(
            synthetic_sticker_rgb,
            _expanded_box(
                synthetic_sticker_bbox,
                synthetic_sticker["rgb"].shape,
                horizontal=16,
                vertical=16,
            ),
        )
    )
    sticker_labels.append(
        f"synthetic sticker bbox seed {synthetic_sticker['seed']}"
    )
    panels = sticker_panels + surface_panels + foam_panels + scene_panels
    labels = sticker_labels + surface_labels + foam_labels + scene_labels
    columns = 4
    panel_width, panel_height = 240, 240
    label_height = 28
    rows = int(math.ceil(len(panels) / columns))
    montage = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + label_height)),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(montage)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = (index % columns) * panel_width
        y = (index // columns) * (panel_height + label_height)
        montage.paste(panel, (x, y))
        draw.text((x + 4, y + panel_height + 5), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)
    return {
        "status": "created_from_explicit_whitelist",
        "sha256": file_sha256(path),
        "real_index_sha256": observed_index_sha256,
        "real_frames": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "vial": record["vial"],
                "frame_index": record["frame_index"],
            }
            for record, _ in real_frames
        ],
        "synthetic_seeds": [frame["seed"] for frame in picks],
        "synthetic_scene_selections": scene_selection_metadata,
        "synthetic_surface_zoom_seeds": [
            frame["seed"] for frame in surface_picks
        ],
        "synthetic_surface_outlier": surface_outlier_metadata,
        "synthetic_foam_crop": foam_selection_metadata,
        "real_sticker_crop": {
            "frame_index": 830,
            "bbox_xyxy_half_open": list(real_sticker_bbox),
        },
        "synthetic_sticker_crop": {
            "seed": synthetic_sticker["seed"],
            "bbox_xyxy_half_open": synthetic_sticker_bbox,
            "sticker_blocked_pixels": synthetic_sticker[
                "sticker_blocked_pixels"
            ],
            "selection": (
                "largest sticker_blocked_pixels; ties use earliest manifest "
                "index then lowest seed"
            ),
        },
        "layout": (
            "frame-830 BRADY and synthetic sticker bbox crops first; whitelisted "
            "real surface zooms then representative and worst-outlier synthetic "
            "surface zooms share the next comparison section; a strongest-evidence "
            "foam crop and darkness/hue-stratified scene crops follow"
        ),
        "review_required": True,
    }


def summarize_pilot(
    pilot_dir,
    real_appearance_path,
    *,
    allowed_frames_root=None,
    montage_out=None,
    synthetic_montage_out=None,
    expected_specs_sha256=None,
):
    """Validate and summarize one copied 20-frame Revision-2 pilot."""
    pilot_dir = Path(pilot_dir).expanduser().resolve()
    manifest_path = pilot_dir / "manifest.jsonl"
    summary_path = pilot_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError("pilot directory needs manifest.jsonl and summary.json")
    real_path = Path(real_appearance_path).expanduser().resolve()
    real = validate_real_appearance(_read_json(real_path, "real appearance summary"))
    records = _load_manifest(manifest_path)
    render_summary = _read_json(summary_path, "render summary")
    integrity = _validate_summary(
        render_summary,
        summary_path,
        manifest_path,
        expected_specs_sha256=expected_specs_sha256,
    )
    frames = _validate_images(pilot_dir, records, real["scene_rois"])
    independent_checker = _validate_with_independent_checker(manifest_path)
    eligible = [frame for frame in frames if frame["liquid_mean_v"] is not None]
    values = [frame["liquid_mean_v"] for frame in eligible]
    low_count = sum(value <= LOW_V_THRESHOLD for value in values)
    expected_eligible = len(eligible) == EXPECTED_ELIGIBLE_COUNT
    coverage_low = bool(values and min(values) <= REAL_LIQUID_MEAN_V[0])
    coverage_high = bool(values and max(values) >= REAL_LIQUID_MEAN_V[1])
    low_count_passed = LOW_V_COUNT_RANGE[0] <= low_count <= LOW_V_COUNT_RANGE[1]
    liquid_gate = expected_eligible and coverage_low and coverage_high
    darkest_tail = _darkest_tail_summary(frames)
    hsv_per_frame = [
        {
            "seed": frame["seed"],
            "level_ml": frame["level_ml"],
            "hue_family": frame["material"].get("hue_family"),
            "liquid_pixel_count": frame["liquid_pixel_count"],
            **frame["liquid_material_comparison"],
        }
        for frame in frames
    ]
    eligible_hsv = [
        frame["liquid_material_comparison"]
        for frame in eligible
    ]
    rendered_hues = [
        comparison["rendered_hue_0_to_1"]
        for comparison in eligible_hsv
        if comparison["rendered_hue_0_to_1"] is not None
    ]
    rendered_saturations = [
        comparison["rendered_mean_saturation"] for comparison in eligible_hsv
    ]
    saturated_low, saturated_high = SATURATED_S_RANGE
    saturated_count = sum(
        saturated_low <= saturation <= saturated_high
        for saturation in rendered_saturations
    )
    saturation_gate = bool(
        expected_eligible
        and SATURATED_COUNT_RANGE[0]
        <= saturated_count
        <= SATURATED_COUNT_RANGE[1]
    )
    hue_errors = [
        comparison["circular_hue_error"]
        for comparison in eligible_hsv
        if comparison["circular_hue_error"] is not None
    ]
    saturation_retentions = [
        comparison["saturation_retention_ratio"]
        for comparison in eligible_hsv
        if comparison["saturation_retention_ratio"] is not None
    ]
    holder_frames = [
        frame for frame in frames
        if frame["holder_proxy"]["median_value"] is not None
    ]
    holder_medians = [
        frame["holder_proxy"]["median_value"] for frame in holder_frames
    ]
    holder_means = [
        frame["holder_proxy"]["mean_value"] for frame in holder_frames
    ]
    glass_frames = [
        frame for frame in frames if frame["glass_boundary_proxy"] is not None
    ]
    holder_relations = []
    for frame in eligible:
        material_v = frame["material"]["hsv"][2]
        if material_v > 0.12 and frame["liquid_mean_v"] > 0.12:
            continue
        holder_v = frame["holder_proxy"]["mean_value"]
        holder_relations.append(
            {
                "seed": frame["seed"],
                "material_value": material_v,
                "rendered_liquid_mean_value": frame["liquid_mean_v"],
                "holder_proxy_mean_value": holder_v,
                "liquid_is_as_dark_as_or_darker_than_holder": bool(
                    holder_v is not None and frame["liquid_mean_v"] <= holder_v
                ),
            }
        )
    holder_relation_gate = bool(holder_relations) and all(
        relation["liquid_is_as_dark_as_or_darker_than_holder"]
        for relation in holder_relations
    )
    automated_passed = bool(
        independent_checker["passed"]
        and liquid_gate
        and saturation_gate
        and darkest_tail["gates"]["passed"]
        and holder_relation_gate
    )
    per_frame = [
        {
            "seed": frame["seed"],
            "level_ml": frame["level_ml"],
            "liquid_pixel_count": frame["liquid_pixel_count"],
            "mean_hsv_v": frame["liquid_mean_v"],
        }
        for frame in frames
    ]
    material = [
        {"seed": frame["seed"], **frame["material"]} for frame in frames
    ]
    scene_comparisons = _metric_comparisons(
        frames, real, "scene_metrics", SCENE_METRIC_NAMES
    )
    sticker_comparisons = _metric_comparisons(
        frames, real, "sticker", STICKER_METRIC_NAMES
    )
    surfaces = [frame for frame in frames if frame["surface"]]
    dark_surfaces = [
        frame
        for frame in surfaces
        if frame["liquid_mean_v"] is not None
        and frame["liquid_mean_v"] <= LOW_V_THRESHOLD
    ]
    foams = [frame for frame in frames if frame["foam"]]
    visible_foams = [
        frame for frame in foams if frame["foam"]["visible_in_decoded_mask"]
    ]
    authored_bubble_counts = [
        frame["foam"]["authored_small_bubble_count"] for frame in foams
    ]
    positive_authored_bubble_counts = [
        count for count in authored_bubble_counts if count > 0
    ]
    if (
        montage_out is not None
        and synthetic_montage_out is not None
        and Path(montage_out).expanduser().resolve()
        == Path(synthetic_montage_out).expanduser().resolve()
    ):
        raise ValueError(
            "montage_out and synthetic_montage_out must be different paths"
        )
    for output, description in (
        (montage_out, "montage"),
        (synthetic_montage_out, "synthetic montage"),
    ):
        if output is not None and Path(output).expanduser().resolve().exists():
            raise FileExistsError(f"refusing to overwrite {description}: {output}")
    montage = {
        "status": "not_requested",
        "review_required": True,
    }
    synthetic_montage = {
        "status": "not_requested",
        "contains_real_frames": False,
    }
    if (allowed_frames_root is None) != (montage_out is None):
        raise ValueError(
            "allowed_frames_root and montage_out must be supplied together"
        )
    if allowed_frames_root is not None:
        montage = write_montage(
            montage_out,
            frames,
            real["scene_rois"],
            allowed_frames_root,
            expected_index_sha256=real["provenance"]["index_sha256"],
            real_sticker_bbox=real["brady_label"]["bbox_xyxy_half_open"],
        )
    if synthetic_montage_out is not None:
        synthetic_montage = write_synthetic_montage(
            synthetic_montage_out, frames, real["scene_rois"]
        )
    appearance_reference = {
        "schema_version": real["schema_version"],
        "evidence_label": real["evidence_label"],
        "video5_calibration_frame_count": real["provenance"]["frame_count"],
        "video5_calibration_index_sha256": real["provenance"]["index_sha256"],
        "additional_video5_frames_opened": (
            real.get("additional_video5_frames_opened", 0)
        ),
        "external_reference_counts": None,
        "renderer_guidance": None,
    }
    if real["schema_version"] == 2:
        external_provenance = real["external_reference_sample"]["provenance"]
        appearance_reference["external_reference_counts"] = {
            name: external_provenance[name]
            for name in (
                "candidate_pair_count",
                "candidate_image_count_hashed_before_first_decode",
                "vial_candidate_frames_decoded",
                "tank_selected_frames_decoded",
                "selected_pair_count",
                "video5_paths_resolved_hashed_or_decoded",
                "video5_vial_colors_csv_opened",
            )
        }
        appearance_reference["renderer_guidance"] = real["renderer_guidance"]
    report = {
        "schema_version": 1,
        "step": 8,
        "revision": 2,
        "evidence_label": (
            "measured_synthetic_pilot_compared_with_measured_calibration_appearance"
        ),
        "status": "complete" if automated_passed else "failed_automated_gate",
        "inputs": {
            "real_appearance_sha256": file_sha256(real_path),
            "real_appearance_schema_version": real["schema_version"],
            "real_appearance_evidence_label": real["evidence_label"],
            "expanded_external_reference_included": (
                real["schema_version"] == 2
            ),
            **integrity,
        },
        "independent_capture_checker": independent_checker,
        "appearance_reference": appearance_reference,
        "population": {
            "manifest_records": len(records),
            "liquid_eligible_records": len(eligible),
            "no_visible_liquid_records": len(frames) - len(eligible),
            "variation_profile": EXPECTED_PROFILE,
        },
        "pilot_identity": {
            "contract": "exact planned manifest order and seed/level association",
            "seed_level_grid": [
                {"seed": seed, "level_ml": level_ml}
                for seed, level_ml in EXPECTED_PILOT_GRID
            ],
            "decoded_liquid_ineligible_seeds": [
                frame["seed"]
                for frame in frames
                if frame["liquid_mean_v"] is None
            ],
            "only_zero_level_seed_28001_is_ineligible": True,
        },
        "rendered_liquid_v": {
            "method": "mean HSV V over RGB pixels whose decoded class mask equals liquid=1",
            "real_reference_mean_v": list(REAL_LIQUID_MEAN_V),
            "low_v_threshold": LOW_V_THRESHOLD,
            "legacy_waived_low_count_in_19": list(LOW_V_COUNT_RANGE),
            "observed_min_max": [float(min(values)), float(max(values))]
            if values
            else None,
            "low_count": low_count,
            "low_fraction_of_eligible": float(low_count / len(values))
            if values
            else None,
            "legacy_observations": {
                "review_source": "2026-11-01 ~11:22 pilot r2-v4 look review",
                "low_count_is_6_to_8": low_count_passed,
                "waived_for_v5": True,
            },
            "gates": {
                "eligible_count_is_19": expected_eligible,
                "minimum_covers_real_0_088": coverage_low,
                "maximum_covers_real_0_155": coverage_high,
                "passed": liquid_gate,
            },
            "per_frame": per_frame,
        },
        "rendered_liquid_hsv": {
            "review_source": "2026-11-01 ~12:25 first batch chunk check",
            "method": (
                "Rendered H is the saturation-weighted circular mean over RGB "
                "pixels whose decoded class mask is liquid=1; S and V are "
                "arithmetic means over those same pixels. Hue is null when all "
                "selected pixels are achromatic. Circular hue error and saturation "
                "retention compare rendered pixels with the manifested material HSV."
            ),
            "review_reference_examples": {
                "vial_1_dark_green": {"hue": 0.40, "saturation": 0.78, "value": 0.155},
                "vial_13_olive": {
                    "hue": 0.21,
                    "saturation_range": [0.34, 0.40],
                    "value": 0.088,
                },
            },
            "manifested_hue_families": sorted(
                {
                    frame["material"].get("hue_family")
                    for frame in frames
                    if frame["material"].get("hue_family") is not None
                }
            ),
            "observed_rendered_hue_range": (
                [float(min(rendered_hues)), float(max(rendered_hues))]
                if rendered_hues
                else None
            ),
            "observed_rendered_saturation_range": (
                [
                    float(min(rendered_saturations)),
                    float(max(rendered_saturations)),
                ]
                if rendered_saturations
                else None
            ),
            "target_saturation_range_inclusive": list(SATURATED_S_RANGE),
            "target_saturated_count_in_19": list(SATURATED_COUNT_RANGE),
            "observed_saturated_count": saturated_count,
            "observed_saturated_fraction_of_eligible": (
                float(saturated_count / len(rendered_saturations))
                if rendered_saturations
                else None
            ),
            "observed_rendered_value_range": (
                [float(min(values)), float(max(values))] if values else None
            ),
            "observed_circular_hue_error_range": (
                [float(min(hue_errors)), float(max(hue_errors))]
                if hue_errors
                else None
            ),
            "observed_saturation_retention_range": (
                [
                    float(min(saturation_retentions)),
                    float(max(saturation_retentions)),
                ]
                if saturation_retentions
                else None
            ),
            "per_frame": hsv_per_frame,
            "gates": {
                "eligible_count_is_19": expected_eligible,
                "saturated_count_is_9_or_10": saturation_gate,
                "passed": saturation_gate,
            },
            "visual_review_required": True,
            "automated_gate": "about half at rendered S 0.30--0.85",
        },
        "darkest_liquid_tail": darkest_tail,
        "holder_appearance": {
            "review_source": "2026-11-01 ~10:22 pilot r2-v3 look review item 4",
            "proxy_method": (
                "HSV V over decoded blocked=3 pixels after removing every pixel "
                "inside the manifested synthetic sticker bbox. The remaining class "
                "can include probe or other blocker geometry, so it is explicitly a "
                "blocked-non-sticker holder proxy rather than holder segmentation."
            ),
            "real_holder_reference_value_about": REAL_HOLDER_V,
            "available_frame_count": len(holder_frames),
            "observed_frame_median_value_range": (
                [float(min(holder_medians)), float(max(holder_medians))]
                if holder_medians
                else None
            ),
            "observed_frame_mean_value_range": (
                [float(min(holder_means)), float(max(holder_means))]
                if holder_means
                else None
            ),
            "batch_median_of_frame_medians": (
                float(np.median(holder_medians)) if holder_medians else None
            ),
            "batch_median_of_frame_means": (
                float(np.median(holder_means)) if holder_means else None
            ),
            "batch_median_minus_real_reference": (
                float(np.median(holder_medians) - REAL_HOLDER_V)
                if holder_medians
                else None
            ),
            "per_frame": [
                {"seed": frame["seed"], "metrics": frame["holder_proxy"]}
                for frame in frames
            ],
            "extreme_or_very_dark_definition": (
                "manifested material V <= 0.12 or rendered liquid mean V <= 0.12"
            ),
            "extreme_or_very_dark_liquid_holder_relations": holder_relations,
            "gates": {
                "extreme_or_very_dark_records_exist": bool(holder_relations),
                "all_are_as_dark_as_or_darker_than_holder_proxy": (
                    holder_relation_gate
                ),
                "passed": holder_relation_gate,
            },
            "absolute_holder_target_visual_review_required": True,
        },
        "glass_edge_highlight_proxies": {
            "review_source": "2026-11-01 ~10:22 pilot r2-v3 look review item 6",
            "method": (
                "A one-pixel morphological boundary is taken from all non-ignore "
                "semantic pixels inside the manifested vial bbox. HSV V percentiles, "
                "V>=0.80 fraction, and Sobel gradient magnitude are reported there. "
                "These are reproducible edge/highlight proxies, not a claim that the "
                "boundary pixels exclusively segment physical glass."
            ),
            "available_frame_count": len(glass_frames),
            "batch_ranges": {
                name: [
                    float(
                        min(frame["glass_boundary_proxy"][name] for frame in glass_frames)
                    ),
                    float(
                        max(frame["glass_boundary_proxy"][name] for frame in glass_frames)
                    ),
                ]
                if glass_frames
                else None
                for name in (
                    "value_p95",
                    "value_p95_minus_p05",
                    "highlight_fraction_v_ge_0_80",
                    "gradient_magnitude_mean",
                    "gradient_magnitude_p90",
                )
            },
            "per_frame": [
                {
                    "seed": frame["seed"],
                    "metrics": frame["glass_boundary_proxy"],
                }
                for frame in frames
            ],
            "visual_review_required": True,
            "automated_gate": None,
        },
        "material_inputs": {
            "note": "Manifested 3-D render material inputs; not rendered-pixel results.",
            "per_frame": material,
        },
        "liquid_top_surface": {
            "evidence_label": (
                "measured synthetic top-band proxies plus whitelisted visual zooms"
            ),
            "method": (
                "Top rendered-oval-height rows of decoded liquid-mask pixels; "
                "achromatic and bright-pixel fractions are appearance proxies, not "
                "a physical reflectance measurement."
            ),
            "surface_records": len(surfaces),
            "dark_liquid_surface_records": len(dark_surfaces),
            "dark_surface_grey_reflective_fraction_range": (
                [
                    float(
                        min(
                            frame["surface"][
                                "grey_reflective_fraction_s_le_0_20_v_ge_0_20"
                            ]
                            for frame in dark_surfaces
                        )
                    ),
                    float(
                        max(
                            frame["surface"][
                                "grey_reflective_fraction_s_le_0_20_v_ge_0_20"
                            ]
                            for frame in dark_surfaces
                        )
                    ),
                ]
                if dark_surfaces
                else None
            ),
            "per_frame": [
                {
                    "seed": frame["seed"],
                    "liquid_mean_v": frame["liquid_mean_v"],
                    "material_hsv_v": frame["material"]["hsv"][2],
                    "metrics": frame["surface"],
                }
                for frame in surfaces
            ],
            "visual_review_required": True,
            "automated_gate": None,
        },
        "foam_appearance": {
            "evidence_label": (
                "measured synthetic foam-mask pixels and manifested authored spheres"
            ),
            "method": (
                "Foam HSV is measured only where decoded class mask equals foam=2; "
                "hue distance is circular and becomes weak evidence for nearly grey "
                "pixels. Authored sphere count and camera-visible mask components are "
                "reported separately; neither is relabeled as the other."
            ),
            "frames_with_foam_contract": len(foams),
            "visible_foam_records": len(visible_foams),
            "authored_small_bubble_count_records": len(authored_bubble_counts),
            "authored_small_bubble_count_range_all_frames": (
                [min(authored_bubble_counts), max(authored_bubble_counts)]
                if authored_bubble_counts
                else None
            ),
            "positive_authored_small_bubble_count_range": (
                [
                    min(positive_authored_bubble_counts),
                    max(positive_authored_bubble_counts),
                ]
                if positive_authored_bubble_counts
                else None
            ),
            "per_frame": [
                {
                    "seed": frame["seed"],
                    "liquid_mean_v": frame["liquid_mean_v"],
                    "metrics": frame["foam"],
                }
                for frame in foams
            ],
            "visual_review_required": True,
            "automated_gate": None,
        },
        "sticker_appearance": {
            "comparisons": sticker_comparisons,
            "per_frame": [
                {"seed": frame["seed"], "metrics": frame["sticker"]}
                for frame in frames
                if frame["sticker"]
            ],
            "visual_review_required": True,
        },
        "scene_appearance": {
            "metric_formulas": {
                "scene_grayscale_percentiles": (
                    "OpenCV RGB-to-grayscale p05, median, and p95 in declared rows"
                ),
                "scene_robust_contrast": "scene grayscale p95 minus p05",
                "scene_grayscale_std": "standard deviation of scene grayscale",
                "backdrop_luma": (
                    "Rec.709 luma median/p05/p95 in declared backdrop ROI"
                ),
                "backdrop_robust_contrast": "backdrop luma p95 minus p05",
                "backdrop_color": (
                    "median HSV saturation and median normalized RGB/tint deltas"
                ),
                "blur_laplacian_variance_8bit": (
                    "variance of OpenCV Laplacian on 8-bit scene grayscale"
                ),
                "blur_tenengrad_normalized": (
                    "mean squared 3x3 Sobel gradients on normalized scene grayscale"
                ),
                "noise_normalized_sigma": (
                    "1.4826 times MAD after Gaussian sigma=1.5 backdrop residual"
                ),
                "haze_proxies": (
                    "backdrop luma p05 floor and backdrop p95-minus-p05 contrast"
                ),
            },
            "rois": real["scene_rois"],
            "comparisons": scene_comparisons,
            "per_frame": [
                {"seed": frame["seed"], "metrics": frame["scene"]}
                for frame in frames
            ],
            "neutral_backdrop_target_comparison": {
                "review_source": (
                    "2026-11-01 ~10:22 pilot r2-v3 look review item 5"
                ),
                "review_target": {
                    "median_luma_about": BACKDROP_NEUTRAL_TARGET["luma"],
                    "median_saturation_about": BACKDROP_NEUTRAL_TARGET[
                        "saturation"
                    ],
                },
                "method": (
                    "Per-frame absolute errors from the declared backdrop ROI to "
                    "the review's approximate real target; no tolerance is invented."
                ),
                "per_frame": [
                    {
                        "seed": frame["seed"],
                        **frame["backdrop_neutral_target"],
                    }
                    for frame in frames
                ],
                "observed_luma_range": [
                    float(
                        min(
                            frame["scene"]["backdrop_median_luma"]
                            for frame in frames
                        )
                    ),
                    float(
                        max(
                            frame["scene"]["backdrop_median_luma"]
                            for frame in frames
                        )
                    ),
                ],
                "observed_saturation_range": [
                    float(
                        min(
                            frame["scene"]["backdrop_median_saturation"]
                            for frame in frames
                        )
                    ),
                    float(
                        max(
                            frame["scene"]["backdrop_median_saturation"]
                            for frame in frames
                        )
                    ),
                ],
                "visual_review_required": True,
                "automated_gate": None,
            },
            "visual_review_required": True,
        },
        "montage": montage,
        "synthetic_montage": synthetic_montage,
        "automated_gates": {
            "integrity_passed": True,
            "independent_capture_checker_passed": True,
            "rendered_liquid_v_passed": liquid_gate,
            "rendered_liquid_saturation_passed": saturation_gate,
            "darkest_liquid_tail_passed": darkest_tail["gates"]["passed"],
            "extreme_liquid_holder_relation_passed": holder_relation_gate,
            "passed": automated_passed,
            "scope_note": (
                "The real-reference V coverage, the requested 9--10 of 19 rendered "
                "liquids at S 0.30--0.85, the requested 10-percent V 0.05--0.09 tail, "
                "the corrected full glass-material contract, and the explicit "
                "dark-liquid-versus-holder relation are automated. The prior 6--8 of "
                "19 at V<=0.2 is retained only as waived historical evidence. Hue, "
                "absolute holder V, backdrop transmission through empty glass, the "
                "visible liquid boundary, labels and overall-look resemblance remain "
                "side-by-side review decisions; proxy intervals are evidence, not "
                "physical-accuracy claims."
            ),
        },
    }
    return report


def write_json_no_overwrite(path, value):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path

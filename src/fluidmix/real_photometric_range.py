"""Deterministic, hash-bound photometric nuisance receipts for real video feeds.

This measures a configured fixed background crop. It deliberately does not identify
liquid or recover a physical lamp position; callers must supply frame identities and a
configuration whose crop excludes the clock, vial/liquid zone, cart lane, and moving-vial
zone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image


ALLOWED_VIDEO_IDS = (1, 2, 3, 4)
DEFAULT_STABLE_REGION_CONFIG = {
    "config_id": "fixed_background_v1",
    "crop_id": "fixed_background_crop_v1",
    "crop_xyxy": [0, 0, 30, 20],
    "excluded_regions": ["burned_clock", "liquid_zone", "cart_lane", "vial_motion_zone"],
    "selection_rule": "configured_fixed_background_crop_and_motion_gate",
    "max_motion_score": 0.25,
    "underexposure_luma_max": 30.0,
    "min_shadow_contrast": 15.0,
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be a finite number" + (f" >= {minimum}" if minimum is not None else ""))
    return result


def _video_id(value: object, field: str = "video_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value == 5:
        raise ValueError("video 5 is held out and cannot be calibration input")
    if value not in ALLOWED_VIDEO_IDS:
        raise ValueError(f"{field} must be one of videos 1-4")
    return value


def _config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("stable_region_config must be an object")
    config = dict(value)
    if "liquid_polygon" in config or "manual_liquid_polygon" in config:
        raise ValueError("a hand-edited liquid polygon is not permitted")
    required_exclusions = {"burned_clock", "liquid_zone", "cart_lane", "vial_motion_zone"}
    if not isinstance(config.get("config_id"), str) or not config["config_id"]:
        raise ValueError("stable_region_config.config_id is required")
    if not isinstance(config.get("crop_id"), str) or not config["crop_id"]:
        raise ValueError("stable_region_config.crop_id is required")
    if not required_exclusions.issubset(set(config.get("excluded_regions", []))):
        raise ValueError("stable_region_config must document clock, liquid, cart, and moving-vial exclusions")
    crop = config.get("crop_xyxy")
    if not isinstance(crop, list) or len(crop) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in crop):
        raise ValueError("stable_region_config.crop_xyxy must contain four integer coordinates")
    x0, y0, x1, y1 = crop
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("stable_region_config.crop_xyxy is invalid")
    _number(config.get("max_motion_score"), "stable_region_config.max_motion_score", minimum=0.0)
    _number(config.get("underexposure_luma_max"), "stable_region_config.underexposure_luma_max", minimum=0.0)
    _number(config.get("min_shadow_contrast"), "stable_region_config.min_shadow_contrast", minimum=0.0)
    return config


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _crossing(profile: np.ndarray, level: float) -> float | None:
    for index in range(len(profile) - 1):
        left, right = float(profile[index]), float(profile[index + 1])
        if left <= level <= right and right > left:
            return index + (level - left) / (right - left)
    return None


def _shadow_metrics(luma: np.ndarray, min_contrast: float) -> dict[str, object]:
    baseline = float(np.percentile(luma, 90))
    mask = luma <= baseline - min_contrast
    if not mask.any():
        return {"status": "not_measurable", "reason": "shadow_not_distinct", "centroid_xy": None,
                "edge_angle_degrees": None, "area_fraction": None, "contrast": None,
                "penumbra_10_90_px": None}
    yy, xx = np.nonzero(mask)
    contrast = baseline - float(np.median(luma[mask]))
    edge_angle: float | None = None
    gradient_y, gradient_x = np.gradient(luma)
    magnitude = np.hypot(gradient_x, gradient_y)
    edge = magnitude >= max(float(np.percentile(magnitude, 90)), min_contrast / 4.0)
    edge_y, edge_x = np.nonzero(edge)
    if len(edge_x) >= 2:
        points = np.column_stack((edge_x, edge_y)).astype(float)
        values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
        if np.isfinite(values).all() and values[-1] > 0:
            vector = vectors[:, int(np.argmax(values))]
            edge_angle = float(math.degrees(math.atan2(vector[1], vector[0])) % 180.0)
    profile = np.mean(luma, axis=0)
    low, high = float(np.percentile(profile, 5)), float(np.percentile(profile, 95))
    penumbra: float | None = None
    if high - low >= min_contrast:
        p10 = _crossing(profile, low + 0.1 * (high - low))
        p90 = _crossing(profile, low + 0.9 * (high - low))
        if p10 is not None and p90 is not None and p90 >= p10:
            penumbra = p90 - p10
    return {"status": "measured", "reason": None if penumbra is not None else "penumbra_not_measurable",
            "coordinate_system": "configured_crop_pixels",
            "centroid_xy": [_round(float(np.mean(xx))), _round(float(np.mean(yy)))],
            "edge_angle_degrees": _round(edge_angle), "area_fraction": _round(float(np.mean(mask))),
            "contrast": _round(contrast), "penumbra_10_90_px": _round(penumbra)}


def _measure_crop(crop: np.ndarray, config: Mapping[str, object]) -> tuple[dict[str, float], dict[str, object]]:
    pixels = crop.astype(np.float64)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    mean_green = float(np.mean(green))
    if mean_green <= 0:
        raise ValueError("mean green channel is zero")
    maximum, minimum = np.max(pixels, axis=2), np.min(pixels, axis=2)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    metrics = {"luma": _round(float(np.mean(luma))), "r_over_g": _round(float(np.mean(red)) / mean_green),
               "b_over_g": _round(float(np.mean(blue)) / mean_green), "saturation": _round(float(np.mean(saturation))),
               "underexposure_fraction": _round(float(np.mean(luma <= float(config["underexposure_luma_max"]))))}
    return metrics, _shadow_metrics(luma, float(config["min_shadow_contrast"]))


def _bounds(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"p05": _round(float(np.percentile(values, 5))), "p50": _round(float(np.percentile(values, 50))),
            "p95": _round(float(np.percentile(values, 95)))}


def _summary(measurements: list[dict[str, object]], candidate_count: int) -> dict[str, object]:
    metric_names = ("luma", "r_over_g", "b_over_g", "saturation", "underexposure_fraction")
    shadow_names = ("area_fraction", "contrast", "penumbra_10_90_px")
    bounds: dict[str, object] = {name: _bounds([float(row["metrics"][name]) for row in measurements]) for name in metric_names}
    bounds.update({f"shadow_{name}": _bounds([float(row["shadow"][name]) for row in measurements if row["shadow"][name] is not None]) for name in shadow_names})
    return {"candidate_count": candidate_count, "usable_count": len(measurements),
            "excluded_count": candidate_count - len(measurements), "percentile_bounds": bounds}


def build_photometric_receipt(manifest_path: str | Path) -> dict[str, object]:
    """Build one receipt from externally selected videos-1--4 PNG/JPEG frames.

    The manifest remains outside Git with its raw frames. Each frame has ``frame_id``,
    ``video_id``, ``image_path``, and a measured ``motion_score``. Frame selection is
    declarative: this function does not decode video or infer an event.
    """
    path = Path(manifest_path).resolve(strict=True)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("photometric frame manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ValueError("photometric frame manifest must have schema_version 1")
    config = _config(manifest.get("stable_region_config"))
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("photometric frame manifest.frames must be a list")
    measurements: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    x0, y0, x1, y1 = config["crop_xyxy"]
    for row in frames:
        if not isinstance(row, Mapping):
            raise ValueError("each photometric frame must be an object")
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen_ids:
            raise ValueError("each photometric frame needs a unique frame_id")
        seen_ids.add(frame_id)
        video_id = _video_id(row.get("video_id"))
        image_value = row.get("image_path")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"frame {frame_id} needs image_path")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve(strict=True)
        source_sha256 = _sha256_path(image_path)
        sources.append({"frame_id": frame_id, "sha256": source_sha256})
        motion_score = _number(row.get("motion_score"), f"frame {frame_id}.motion_score", minimum=0.0)
        if motion_score > float(config["max_motion_score"]):
            exclusions.append({"frame_id": frame_id, "reason": "motion_score_above_max", "source_sha256": source_sha256})
            continue
        try:
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB"))
        except (OSError, ValueError):
            exclusions.append({"frame_id": frame_id, "reason": "unreadable_image", "source_sha256": source_sha256})
            continue
        if pixels.shape[0] < y1 or pixels.shape[1] < x1:
            exclusions.append({"frame_id": frame_id, "reason": "crop_outside_image", "source_sha256": source_sha256})
            continue
        try:
            metrics, shadow = _measure_crop(pixels[y0:y1, x0:x1], config)
        except ValueError as error:
            exclusions.append({"frame_id": frame_id, "reason": str(error), "source_sha256": source_sha256})
            continue
        measurements.append({"frame_id": frame_id, "video_id": video_id, "source_sha256": source_sha256,
                             "motion_score": _round(motion_score), "metrics": metrics, "shadow": shadow})
    per_video = {
        str(video_id): _summary(
            [row for row in measurements if row["video_id"] == video_id],
            sum(1 for row in frames if row.get("video_id") == video_id),
        )
        for video_id in ALLOWED_VIDEO_IDS
    }
    return {"schema_version": 1, "receipt_type": "real_photometric_nuisance_range",
            "status": "measured" if measurements else "insufficient_evidence", "allowed_video_ids": list(ALLOWED_VIDEO_IDS),
            "source_manifest_sha256": _sha256_path(path), "sources": sources,
            "stable_region": {"config_id": config["config_id"], "crop_id": config["crop_id"],
                              "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
                              "crop_xyxy": config["crop_xyxy"], "excluded_regions": config["excluded_regions"], "selection_rule": config["selection_rule"]},
            "per_video": per_video, "pooled": _summary(measurements, len(frames)), "measurements": measurements, "exclusions": exclusions,
            "insufficiency_reasons": [] if measurements else ["no_usable_stable_region_frames"],
            "limits": {"measurement_scope": "configured stable background region only", "physical_lamp_position": "not_recovered",
                       "liquid_polygon": "not_used", "real_feed_claim": "photometric and shadow nuisance range only"}}


def write_no_clobber_json(output_path: str | Path, payload: object) -> Path:
    """Publish canonical JSON once, atomically refusing an existing destination."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.partial-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output

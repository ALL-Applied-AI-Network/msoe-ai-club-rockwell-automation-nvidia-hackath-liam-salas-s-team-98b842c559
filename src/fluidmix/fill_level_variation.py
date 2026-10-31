"""Deterministic specifications and class-mask rules for FillCam renders.

This module has no Isaac Sim dependency. It freezes the synthetic variation
contract and the label precedence used by the remote renderer.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np


ROLE_SEED_RANGES = {
    "train": (0, 999_999),
    "val": (1_000_000, 1_999_999),
    "test": (2_000_000, 2_999_999),
}
ROLE_CODES = {"pilot": 11, "train": 23, "val": 37, "test": 53}
LEVEL_BANDS = (
    ("full_0_85", 0.0, 85.0, 0.25),
    ("twin_lower_edge_40_1_48_1", 40.1, 48.1, 0.25),
    ("twin_hidden_48_1_66_5", 48.1, 66.5, 0.25),
    ("twin_upper_edge_66_5_70_5", 66.5, 70.5, 0.25),
)
LABEL_IDS = {
    "empty": 0,
    "liquid": 1,
    "foam": 2,
    "blocked": 3,
    "ignore": 255,
}
ML_PER_PIXEL = 1.0 / 3.04


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_role_seed(role, seed):
    if role not in ROLE_CODES:
        raise ValueError(f"role must be one of {sorted(ROLE_CODES)}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if role in ROLE_SEED_RANGES:
        first, last = ROLE_SEED_RANGES[role]
        if not first <= seed <= last:
            raise ValueError(f"{role} seed must be in [{first}, {last}]")
    return seed


def sample_spec(seed, role, level_ml=None):
    """Return a reproducible, JSON-serializable frame specification."""
    seed = validate_role_seed(role, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, ROLE_CODES[role]]))
    if level_ml is None:
        probabilities = [band[3] for band in LEVEL_BANDS]
        index = int(rng.choice(len(LEVEL_BANDS), p=probabilities))
        band_name, low, high, _ = LEVEL_BANDS[index]
        level = float(rng.uniform(low, high))
    else:
        level = _finite_number(level_ml, "level_ml")
        if not 0.0 <= level <= 85.0:
            raise ValueError("level_ml must be in [0, 85]")
        band_name = "explicit"

    hue = float(rng.uniform(0.0, 1.0))
    saturation = float(rng.beta(0.9, 0.9))
    value = float(rng.uniform(0.08, 1.0))
    rgb = [float(channel) for channel in colorsys.hsv_to_rgb(hue, saturation, value)]
    foam_enabled = bool(level > 0.0 and rng.random() < 0.5)
    sticker_present = bool(rng.random() < 0.6)
    wet_enabled = bool(rng.random() < 0.3)
    temperature_scale = float(rng.uniform(0.8, 1.2))
    spec = {
        "schema_version": 1,
        "seed": seed,
        "role": role,
        "level_ml": level,
        "level_sampling_band": band_name,
        "liquid": {
            "hsv": [hue, saturation, value],
            "rgb": rgb,
            "opacity": float(rng.uniform(0.55, 0.98)),
            "roughness": float(rng.uniform(0.12, 0.55)),
        },
        "foam": {
            "enabled": foam_enabled,
            "thickness_mm": float(rng.uniform(0.25, 10.0)) if foam_enabled else 0.0,
            "tint_rgb": [
                float(rng.uniform(0.72, 1.0)),
                float(rng.uniform(0.72, 1.0)),
                float(rng.uniform(0.72, 1.0)),
            ],
            "opacity": float(rng.uniform(0.35, 0.95)),
            "bumpiness": float(rng.uniform(0.0, 1.0)),
        },
        "sticker_pose": {
            "present": sticker_present,
            "azimuth_deg": float(rng.uniform(-180.0, 180.0)),
            "height_offset_mm": float(rng.uniform(-32.0, 32.0)),
            "roll_deg": float(rng.uniform(-8.0, 8.0)),
        },
        "lighting": {
            "brightness_scale": float(rng.uniform(0.8, 1.2)),
            "color_temperature_scale": temperature_scale,
            "color_temperature_kelvin": 5200.0 * temperature_scale,
        },
        "camera_jitter": {
            "x_px": float(rng.uniform(-3.0, 3.0)),
            "y_px": float(rng.uniform(-3.0, 3.0)),
            "zoom_scale": float(rng.uniform(0.98, 1.02)),
        },
        "wet_walls": {
            "enabled": wet_enabled,
            "coverage": float(rng.uniform(0.05, 0.35)) if wet_enabled else 0.0,
            "opacity": float(rng.uniform(0.08, 0.28)) if wet_enabled else 0.0,
            "pattern_seed": int(rng.integers(0, 2**31)),
        },
    }
    validate_spec(spec)
    return spec


def validate_spec(spec):
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported frame-spec schema")
    validate_role_seed(spec.get("role"), spec.get("seed"))
    level = _finite_number(spec.get("level_ml"), "level_ml")
    if not 0.0 <= level <= 85.0:
        raise ValueError("level_ml must be in [0, 85]")
    foam = spec["foam"]
    for name, value in (
        ("foam.enabled", foam.get("enabled")),
        ("sticker_pose.present", spec["sticker_pose"].get("present")),
        ("wet_walls.enabled", spec["wet_walls"].get("enabled")),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    thickness = _finite_number(foam["thickness_mm"], "foam thickness")
    if not 0.0 <= thickness <= 10.0:
        raise ValueError("foam thickness must be in [0, 10] mm")
    if foam["enabled"] != (thickness > 0.0):
        raise ValueError("foam enabled state and thickness disagree")
    if level == 0.0 and foam["enabled"]:
        raise ValueError("an empty vial cannot contain foam")
    for name, value in (
        ("liquid opacity", spec["liquid"]["opacity"]),
        ("liquid roughness", spec["liquid"]["roughness"]),
        ("foam opacity", foam["opacity"]),
        ("foam bumpiness", foam["bumpiness"]),
    ):
        if not 0.0 <= _finite_number(value, name) <= 1.0:
            raise ValueError(f"{name} outside [0, 1]")
    sticker = spec["sticker_pose"]
    azimuth = _finite_number(sticker["azimuth_deg"], "sticker azimuth")
    if not -180.0 <= azimuth <= 180.0:
        raise ValueError("sticker azimuth outside [-180, 180]")
    if abs(_finite_number(sticker["height_offset_mm"], "sticker height")) > 32.0:
        raise ValueError("sticker height offset exceeds 32 mm")
    if abs(_finite_number(sticker["roll_deg"], "sticker roll")) > 8.0:
        raise ValueError("sticker roll exceeds eight degrees")
    lighting = spec["lighting"]
    for name in ("brightness_scale", "color_temperature_scale"):
        if not 0.8 <= _finite_number(lighting[name], name) <= 1.2:
            raise ValueError(f"{name} outside +/-20 percent")
    camera = spec["camera_jitter"]
    if abs(_finite_number(camera["x_px"], "camera x jitter")) > 3.0:
        raise ValueError("camera x jitter exceeds three pixels")
    if abs(_finite_number(camera["y_px"], "camera y jitter")) > 3.0:
        raise ValueError("camera y jitter exceeds three pixels")
    if not 0.98 <= _finite_number(camera["zoom_scale"], "camera zoom") <= 1.02:
        raise ValueError("camera zoom outside +/-2 percent")
    for channel in spec["liquid"]["rgb"] + foam["tint_rgb"]:
        if not 0.0 <= _finite_number(channel, "color channel") <= 1.0:
            raise ValueError("color channel outside [0, 1]")
    wet = spec["wet_walls"]
    if not wet["enabled"] and (
        _finite_number(wet["coverage"], "wet coverage") != 0.0
        or _finite_number(wet["opacity"], "wet opacity") != 0.0
    ):
        raise ValueError("disabled wet walls must have zero coverage and opacity")
    if wet["enabled"]:
        if not 0.05 <= _finite_number(wet["coverage"], "wet coverage") <= 0.35:
            raise ValueError("wet coverage outside [0.05, 0.35]")
        if not 0.08 <= _finite_number(wet["opacity"], "wet opacity") <= 0.28:
            raise ValueError("wet opacity outside [0.08, 0.28]")
    return spec


def compose_class_mask(vial_silhouette, liquid, foam, front_blocked):
    """Compose plan labels; geometry behind empty glass stays empty."""
    masks = [
        np.asarray(mask, dtype=bool)
        for mask in (vial_silhouette, liquid, foam, front_blocked)
    ]
    if len({mask.shape for mask in masks}) != 1 or masks[0].ndim != 2:
        raise ValueError("all masks must share one two-dimensional shape")
    vial, liquid_mask, foam_mask, blocked = masks
    if np.any(liquid_mask & foam_mask):
        raise ValueError("liquid and foam masks must not overlap")
    if np.any((liquid_mask | foam_mask | blocked) & ~vial):
        raise ValueError("scored classes must stay inside the vial silhouette")
    result = np.full(vial.shape, LABEL_IDS["ignore"], dtype=np.uint8)
    result[vial] = LABEL_IDS["empty"]
    result[liquid_mask] = LABEL_IDS["liquid"]
    result[foam_mask] = LABEL_IDS["foam"]
    result[blocked] = LABEL_IDS["blocked"]
    return result


def validate_class_mask(mask):
    mask = np.asarray(mask)
    allowed = set(LABEL_IDS.values())
    seen = {int(value) for value in np.unique(mask)}
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("class mask must be a two-dimensional uint8 array")
    if not seen <= allowed:
        raise ValueError(f"unexpected class ids: {sorted(seen - allowed)}")
    if LABEL_IDS["ignore"] not in seen:
        raise ValueError("class mask must contain ignored pixels outside the vial")
    if LABEL_IDS["empty"] not in seen:
        raise ValueError("class mask must contain an empty-vial region")
    return mask


def volume_ml_for_row(row):
    return 30.0 + (329.0 - _finite_number(row, "row")) * ML_PER_PIXEL


def expected_surface_row(level_ml):
    level = _finite_number(level_ml, "level_ml")
    return 329.0 - (level - 30.0) / ML_PER_PIXEL

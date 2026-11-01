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
RENDER_PROFILES = ("baseline", "real-look-v2")
REAL_LOOK_PROFILE_CODE = 0xA11CE
REAL_LOOK_LIQUID_VALUE_RANGE = (0.03, 0.35)
REAL_LOOK_LIQUID_OPACITY_RANGE = (0.93, 0.97)
REAL_LOOK_EXTREME_DARK_VALUE_RANGE = (0.03, 0.04)
REAL_LOOK_VERY_DARK_VALUE_RANGE = (0.03, 0.12)
REAL_LOOK_DARK_VALUE_RANGE = (0.20, 0.35)
REAL_LOOK_SATURATED_SATURATION_RANGE = (0.30, 0.85)
REAL_LOOK_SUBDUED_SATURATION_RANGE = (0.02, 0.28)
REAL_LOOK_HUE_FAMILIES = {
    "dark_green": ((0.27, 0.46), (0.02, 0.85)),
    "olive": ((0.12, 0.23), (0.02, 0.85)),
    "brown": ((0.00, 0.10), (0.02, 0.85)),
    "blue": ((0.50, 0.70), (0.02, 0.85)),
    "near_black": ((0.00, 1.00), (0.02, 0.85)),
}
REAL_LOOK_LABEL_WIDTH_MM = (28.0, 36.0)
REAL_LOOK_LABEL_HEIGHT_MM = (34.0, 44.0)
REAL_LOOK_LABEL_ABS_AZIMUTH_DEG = (130.0, 180.0)
REAL_LOOK_LABEL_HEIGHT_OFFSET_MM = (-10.0, 54.0)
REAL_LOOK_HOLDER_BRIGHTNESS_RANGE = (0.002, 0.018)
REAL_LOOK_BACKDROP_SATURATION_RANGE = (0.01, 0.055)
REAL_LOOK_BACKDROP_VALUE_RANGE = (0.60, 0.74)
REAL_LOOK_HAZE_OPACITY_RANGE = (0.0, 0.008)
REAL_LOOK_CONTRAST_RANGE = (0.85, 1.10)
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


def _sample_real_look_liquid(rng, seed):
    """Sample a dark physical-material target with deterministic dark strata."""
    family_names = tuple(REAL_LOOK_HUE_FAMILIES)
    family = family_names[int(rng.integers(0, len(family_names)))]
    hue_bounds, _allowed_saturation_bounds = REAL_LOOK_HUE_FAMILIES[family]
    hue = float(rng.uniform(*hue_bounds))
    saturated = int(seed) % 2 == 0
    saturation_bounds = (
        REAL_LOOK_SATURATED_SATURATION_RANGE
        if saturated
        else REAL_LOOK_SUBDUED_SATURATION_RANGE
    )
    saturation = float(rng.uniform(*saturation_bounds))
    extreme_dark = int(seed) % 9 == 0
    very_dark = int(seed) % 3 == 0
    if extreme_dark:
        value_bounds = REAL_LOOK_EXTREME_DARK_VALUE_RANGE
    elif very_dark:
        value_bounds = REAL_LOOK_VERY_DARK_VALUE_RANGE
    else:
        value_bounds = REAL_LOOK_DARK_VALUE_RANGE
    value = float(rng.uniform(*value_bounds))
    return {
        "hue_family": family,
        "saturation_band": "saturated" if saturated else "subdued",
        "darkness_band": (
            "extreme_dark" if extreme_dark else "very_dark" if very_dark else "dark"
        ),
        "hsv": [hue, saturation, value],
        "rgb": [
            float(channel)
            for channel in colorsys.hsv_to_rgb(hue, saturation, value)
        ],
    }


def _sample_real_look_scene(rng):
    backdrop_hsv = [
        float(rng.uniform(0.0, 1.0)),
        float(rng.uniform(*REAL_LOOK_BACKDROP_SATURATION_RANGE)),
        float(rng.uniform(*REAL_LOOK_BACKDROP_VALUE_RANGE)),
    ]
    backdrop_rgb = [
        float(channel) for channel in colorsys.hsv_to_rgb(*backdrop_hsv)
    ]
    haze_hsv = [
        float((backdrop_hsv[0] + rng.uniform(-0.02, 0.02)) % 1.0),
        float(rng.uniform(0.00, 0.03)),
        float(rng.uniform(0.64, 0.78)),
    ]
    haze_tint_rgb = [
        float(channel) for channel in colorsys.hsv_to_rgb(*haze_hsv)
    ]
    return {
        "backdrop_hsv": backdrop_hsv,
        "backdrop_rgb": backdrop_rgb,
        "backdrop_roughness": float(rng.uniform(0.45, 0.95)),
        "haze_opacity": float(rng.uniform(*REAL_LOOK_HAZE_OPACITY_RANGE)),
        "haze_rgb": haze_tint_rgb,
        "contrast_scale": float(rng.uniform(*REAL_LOOK_CONTRAST_RANGE)),
    }


def sample_spec(seed, role, level_ml=None, *, profile="baseline"):
    """Return a reproducible, JSON-serializable frame specification."""
    seed = validate_role_seed(role, seed)
    if profile not in RENDER_PROFILES:
        raise ValueError(f"profile must be one of {list(RENDER_PROFILES)}")
    seed_parts = [seed, ROLE_CODES[role]]
    if profile == "real-look-v2":
        seed_parts.append(REAL_LOOK_PROFILE_CODE)
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
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

    real_look_liquid = None
    if profile == "real-look-v2":
        real_look_liquid = _sample_real_look_liquid(rng, seed)
        hue, saturation, value = real_look_liquid["hsv"]
        rgb = real_look_liquid["rgb"]
    else:
        hue = float(rng.uniform(0.0, 1.0))
        saturation = float(rng.beta(0.9, 0.9))
        value = float(rng.uniform(0.08, 1.0))
        rgb = [
            float(channel)
            for channel in colorsys.hsv_to_rgb(hue, saturation, value)
        ]
    foam_enabled = bool(level > 0.0 and rng.random() < 0.5)
    sticker_present = bool(rng.random() < 0.6)
    wet_enabled = bool(rng.random() < 0.3)
    temperature_scale = float(rng.uniform(0.8, 1.2))
    sampled_liquid_opacity = float(rng.uniform(0.55, 0.98))
    spec = {
        "schema_version": 1,
        "seed": seed,
        "role": role,
        "level_ml": level,
        "level_sampling_band": band_name,
        "liquid": {
            "hsv": [hue, saturation, value],
            "rgb": rgb,
            "opacity": sampled_liquid_opacity,
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
        "holder_material": {
            # Blue/green PreviewSurface diffuse value in linear scene color.
            # The source black acetal uses about 0.018; this range includes
            # clearly darker traps without making implausibly pale holders.
            "brightness": float(
                rng.uniform(
                    0.002,
                    REAL_LOOK_HOLDER_BRIGHTNESS_RANGE[1]
                    if profile == "real-look-v2"
                    else 0.05,
                )
            ),
            "roughness": float(rng.uniform(0.20, 0.85)),
        },
    }
    if profile == "real-look-v2":
        if sticker_present:
            original_azimuth = float(spec["sticker_pose"]["azimuth_deg"])
            azimuth_sign = -1.0 if original_azimuth < 0.0 else 1.0
            spec["sticker_pose"]["azimuth_deg"] = azimuth_sign * (
                REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[0]
                + (
                    REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[1]
                    - REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[0]
                )
                * abs(original_azimuth)
                / 180.0
            )
            spec["sticker_pose"]["height_offset_mm"] = min(
                REAL_LOOK_LABEL_HEIGHT_OFFSET_MM[1],
                float(spec["sticker_pose"]["height_offset_mm"]) + 22.0,
            )
        paper_hsv = [
            float(rng.uniform(0.08, 0.18)),
            float(rng.uniform(0.00, 0.08)),
            float(rng.uniform(0.78, 0.98)),
        ]
        print_value = float(rng.uniform(0.01, 0.10))
        spec["variation_profile"] = profile
        spec["liquid"].update(
            {
                "hue_family": real_look_liquid["hue_family"],
                "saturation_band": real_look_liquid["saturation_band"],
                "darkness_band": real_look_liquid["darkness_band"],
                "opacity": float(
                    REAL_LOOK_LIQUID_OPACITY_RANGE[0]
                    + (sampled_liquid_opacity - 0.55)
                    / (0.98 - 0.55)
                    * (
                        REAL_LOOK_LIQUID_OPACITY_RANGE[1]
                        - REAL_LOOK_LIQUID_OPACITY_RANGE[0]
                    )
                ),
            }
        )
        spec["sticker_pose"].update(
            {
                "width_mm": float(rng.uniform(*REAL_LOOK_LABEL_WIDTH_MM)),
                "height_mm": float(rng.uniform(*REAL_LOOK_LABEL_HEIGHT_MM)),
                "paper_rgb": [
                    float(channel)
                    for channel in colorsys.hsv_to_rgb(*paper_hsv)
                ],
                "print_rgb": [print_value, print_value, print_value],
                "print_style": "brady_text_and_barcode",
                "radial_offset_mm": float(rng.uniform(13.2, 14.4)),
            }
        )
        spec["scene_look"] = _sample_real_look_scene(rng)
    validate_spec(spec)
    return spec


def validate_spec(spec):
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported frame-spec schema")
    profile = spec.get("variation_profile", "baseline")
    if profile not in RENDER_PROFILES:
        raise ValueError(f"unsupported render profile: {profile}")
    if profile == "baseline":
        real_look_only_fields = []
        if "scene_look" in spec:
            real_look_only_fields.append("scene_look")
        for key in ("hue_family", "saturation_band", "darkness_band"):
            if key in spec.get("liquid", {}):
                real_look_only_fields.append(f"liquid.{key}")
        if "authored_small_bubble_count" in spec.get("foam", {}):
            real_look_only_fields.append("foam.authored_small_bubble_count")
        for key in (
            "width_mm",
            "height_mm",
            "paper_rgb",
            "print_rgb",
            "print_style",
            "radial_offset_mm",
        ):
            if key in spec.get("sticker_pose", {}):
                real_look_only_fields.append(f"sticker_pose.{key}")
        if real_look_only_fields:
            raise ValueError(
                "baseline spec contains real-look-only fields: "
                + ", ".join(real_look_only_fields)
            )
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
    height_offset = _finite_number(sticker["height_offset_mm"], "sticker height")
    if (
        profile == "baseline" or not sticker["present"]
    ) and abs(height_offset) > 32.0:
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
    holder = spec["holder_material"]
    holder_brightness = _finite_number(
        holder["brightness"], "holder brightness"
    )
    holder_high = (
        REAL_LOOK_HOLDER_BRIGHTNESS_RANGE[1]
        if profile == "real-look-v2"
        else 0.05
    )
    if not 0.002 <= holder_brightness <= holder_high:
        raise ValueError(
            f"holder brightness outside [0.002, {holder_high:.3f}]"
        )
    if not 0.20 <= _finite_number(
        holder["roughness"], "holder roughness"
    ) <= 0.85:
        raise ValueError("holder roughness outside [0.20, 0.85]")
    if profile == "real-look-v2":
        liquid = spec["liquid"]
        family = liquid.get("hue_family")
        if family not in REAL_LOOK_HUE_FAMILIES:
            raise ValueError("real-look liquid has an unsupported hue family")
        hue, saturation, value = (
            _finite_number(channel, "real-look liquid HSV channel")
            for channel in liquid["hsv"]
        )
        hue_bounds, saturation_bounds = REAL_LOOK_HUE_FAMILIES[family]
        if not hue_bounds[0] <= hue <= hue_bounds[1]:
            raise ValueError("real-look liquid hue is outside its family")
        if not saturation_bounds[0] <= saturation <= saturation_bounds[1]:
            raise ValueError("real-look liquid saturation is outside its family")
        saturation_band = liquid.get("saturation_band")
        saturation_ranges = {
            "saturated": REAL_LOOK_SATURATED_SATURATION_RANGE,
            "subdued": REAL_LOOK_SUBDUED_SATURATION_RANGE,
        }
        if saturation_band not in saturation_ranges:
            raise ValueError("real-look liquid has an unsupported saturation band")
        expected_saturation_band = (
            "saturated" if int(spec["seed"]) % 2 == 0 else "subdued"
        )
        if saturation_band != expected_saturation_band:
            raise ValueError("real-look liquid saturation band disagrees with seed")
        saturation_low, saturation_high = saturation_ranges[saturation_band]
        if not saturation_low <= saturation <= saturation_high:
            raise ValueError("real-look liquid saturation is outside its band")
        if not REAL_LOOK_LIQUID_VALUE_RANGE[0] <= value <= REAL_LOOK_LIQUID_VALUE_RANGE[1]:
            raise ValueError("real-look liquid value is outside [0.03, 0.35]")
        darkness_band = liquid.get("darkness_band")
        darkness_ranges = {
            "extreme_dark": REAL_LOOK_EXTREME_DARK_VALUE_RANGE,
            "very_dark": REAL_LOOK_VERY_DARK_VALUE_RANGE,
            "dark": REAL_LOOK_DARK_VALUE_RANGE,
        }
        if darkness_band not in darkness_ranges:
            raise ValueError("real-look liquid has an unsupported darkness band")
        low, high = darkness_ranges[darkness_band]
        if not low <= value <= high:
            raise ValueError("real-look liquid value is outside its darkness band")
        expected_rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        if not np.allclose(liquid["rgb"], expected_rgb, rtol=0.0, atol=1e-12):
            raise ValueError("real-look liquid RGB does not match its HSV")
        if not REAL_LOOK_LIQUID_OPACITY_RANGE[0] <= _finite_number(
            liquid["opacity"], "real-look liquid opacity"
        ) <= REAL_LOOK_LIQUID_OPACITY_RANGE[1]:
            raise ValueError(
                "real-look liquid opacity is outside its translucent range"
            )

        sticker = spec["sticker_pose"]
        if not REAL_LOOK_LABEL_WIDTH_MM[0] <= _finite_number(
            sticker["width_mm"], "sticker width"
        ) <= REAL_LOOK_LABEL_WIDTH_MM[1]:
            raise ValueError("real-look sticker width is outside its measured range")
        if not REAL_LOOK_LABEL_HEIGHT_MM[0] <= _finite_number(
            sticker["height_mm"], "sticker height"
        ) <= REAL_LOOK_LABEL_HEIGHT_MM[1]:
            raise ValueError("real-look sticker height is outside its measured range")
        if sticker.get("print_style") != "brady_text_and_barcode":
            raise ValueError("real-look sticker must use the BRADY-like print style")
        for channel in sticker["paper_rgb"] + sticker["print_rgb"]:
            if not 0.0 <= _finite_number(channel, "sticker color channel") <= 1.0:
                raise ValueError("sticker color channel outside [0, 1]")
        if min(sticker["paper_rgb"]) < 0.70:
            raise ValueError("real-look sticker paper must be light")
        if max(sticker["print_rgb"]) > 0.12:
            raise ValueError("real-look sticker print must be dark")
        if sticker["present"]:
            if not REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[0] <= abs(azimuth) <= (
                REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[1]
            ):
                raise ValueError(
                    "real-look visible sticker azimuth is outside its broad-facing range"
                )
            height_offset = _finite_number(
                sticker["height_offset_mm"], "sticker height"
            )
            if not REAL_LOOK_LABEL_HEIGHT_OFFSET_MM[0] <= height_offset <= (
                REAL_LOOK_LABEL_HEIGHT_OFFSET_MM[1]
            ):
                raise ValueError(
                    "real-look visible sticker height is outside its raised range"
                )

        if not 13.2 <= _finite_number(
            sticker["radial_offset_mm"], "sticker radial offset"
        ) <= 14.4:
            raise ValueError("real-look sticker radial offset is outside [13.2, 14.4]")

        scene = spec["scene_look"]
        backdrop_hsv = [
            _finite_number(channel, "backdrop HSV channel")
            for channel in scene["backdrop_hsv"]
        ]
        if not (
            0.0 <= backdrop_hsv[0] <= 1.0
            and REAL_LOOK_BACKDROP_SATURATION_RANGE[0]
            <= backdrop_hsv[1]
            <= REAL_LOOK_BACKDROP_SATURATION_RANGE[1]
            and REAL_LOOK_BACKDROP_VALUE_RANGE[0]
            <= backdrop_hsv[2]
            <= REAL_LOOK_BACKDROP_VALUE_RANGE[1]
        ):
            raise ValueError("real-look backdrop HSV is outside its range")
        if not np.allclose(
            scene["backdrop_rgb"], colorsys.hsv_to_rgb(*backdrop_hsv),
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("real-look backdrop RGB does not match its HSV")
        if not 0.45 <= _finite_number(
            scene["backdrop_roughness"], "backdrop roughness"
        ) <= 0.95:
            raise ValueError("real-look backdrop roughness is outside [0.45, 0.95]")
        if not REAL_LOOK_HAZE_OPACITY_RANGE[0] <= _finite_number(
            scene["haze_opacity"], "haze opacity"
        ) <= REAL_LOOK_HAZE_OPACITY_RANGE[1]:
            raise ValueError("real-look haze opacity is outside [0, 0.008]")
        for channel in scene["haze_rgb"]:
            if not 0.0 <= _finite_number(channel, "haze tint channel") <= 1.0:
                raise ValueError("haze tint channel outside [0, 1]")
        if not REAL_LOOK_CONTRAST_RANGE[0] <= _finite_number(
            scene["contrast_scale"], "physical contrast scale"
        ) <= REAL_LOOK_CONTRAST_RANGE[1]:
            raise ValueError("real-look contrast scale is outside [0.85, 1.10]")
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
    if seen == {LABEL_IDS["ignore"]}:
        raise ValueError("class mask must contain scored vial pixels")
    return mask


def volume_ml_for_row(row):
    return 30.0 + (329.0 - _finite_number(row, "row")) * ML_PER_PIXEL


def expected_surface_row(level_ml):
    level = _finite_number(level_ml, "level_ml")
    return 329.0 - (level - 30.0) / ML_PER_PIXEL

"""Synthetic fill-level variation and label-contract checks."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.fill_level_variation import (  # noqa: E402
    LABEL_IDS,
    LEVEL_BANDS,
    REAL_LOOK_BACKDROP_SATURATION_RANGE,
    REAL_LOOK_BACKDROP_VALUE_RANGE,
    REAL_LOOK_CONTRAST_RANGE,
    REAL_LOOK_DARK_VALUE_RANGE,
    REAL_LOOK_EXTREME_DARK_VALUE_RANGE,
    REAL_LOOK_LABEL_ABS_AZIMUTH_DEG,
    REAL_LOOK_HUE_FAMILIES,
    REAL_LOOK_HAZE_OPACITY_RANGE,
    REAL_LOOK_HOLDER_BRIGHTNESS_RANGE,
    REAL_LOOK_LABEL_HEIGHT_MM,
    REAL_LOOK_LABEL_HEIGHT_OFFSET_MM,
    REAL_LOOK_LABEL_WIDTH_MM,
    REAL_LOOK_LIQUID_OPACITY_RANGE,
    REAL_LOOK_LIQUID_VALUE_RANGE,
    REAL_LOOK_SATURATED_SATURATION_RANGE,
    REAL_LOOK_SUBDUED_SATURATION_RANGE,
    REAL_LOOK_VERY_DARK_VALUE_RANGE,
    compose_class_mask,
    expected_surface_row,
    sample_spec,
    validate_class_mask,
    validate_role_seed,
    validate_spec,
    volume_ml_for_row,
)


def test_spec_is_deterministic_and_complete():
    first = sample_spec(17, "pilot")
    assert first == sample_spec(17, "pilot")
    assert first != sample_spec(18, "pilot")
    for field in (
        "seed", "role", "level_ml", "foam", "sticker_pose",
        "lighting", "camera_jitter", "wet_walls", "holder_material",
    ):
        assert field in first


def test_baseline_spec_stream_is_unchanged():
    payload = json.dumps(
        sample_spec(17, "pilot"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "264674422a167fd3f82c9126cb1e79eeedba951f2686e935cbd745f4bcc6273e"
    )
    assert "variation_profile" not in sample_spec(17, "pilot")


def test_baseline_rejects_injected_real_look_scene_fields():
    spec = sample_spec(17, "pilot")
    spec["scene_look"] = {"contrast_scale": 0.65, "haze_opacity": 0.05}
    with pytest.raises(ValueError, match="real-look-only.*scene_look"):
        validate_spec(spec)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("liquid", "hue_family", "dark_green"),
        ("liquid", "saturation_band", "saturated"),
        ("liquid", "darkness_band", "dark"),
        ("foam", "authored_small_bubble_count", 4),
        ("sticker_pose", "width_mm", 32.0),
        ("sticker_pose", "height_mm", 40.0),
        ("sticker_pose", "paper_rgb", [0.9, 0.9, 0.9]),
        ("sticker_pose", "print_rgb", [0.05, 0.05, 0.05]),
        ("sticker_pose", "print_style", "brady_text_and_barcode"),
        ("sticker_pose", "radial_offset_mm", 13.7),
    ),
)
def test_baseline_rejects_injected_real_look_nested_fields(section, key, value):
    spec = sample_spec(17, "pilot")
    spec[section][key] = value
    with pytest.raises(ValueError, match=rf"real-look-only.*{key}"):
        validate_spec(spec)


def test_real_look_profile_is_deterministic_complete_and_distinct():
    spec = sample_spec(17, "pilot", profile="real-look-v2")
    assert spec == sample_spec(17, "pilot", profile="real-look-v2")
    assert spec != sample_spec(17, "pilot")
    assert spec["variation_profile"] == "real-look-v2"
    assert spec["liquid"]["hue_family"] in REAL_LOOK_HUE_FAMILIES
    assert spec["liquid"]["saturation_band"] in {"saturated", "subdued"}
    assert spec["liquid"]["darkness_band"] in {
        "extreme_dark", "very_dark", "dark"
    }
    assert spec["sticker_pose"]["print_style"] == "brady_text_and_barcode"
    assert "scene_look" in spec


def test_real_look_ranges_and_stratified_dark_prevalence():
    specs = [
        sample_spec(seed, "pilot", profile="real-look-v2")
        for seed in range(900)
    ]
    values = [spec["liquid"]["hsv"][2] for spec in specs]
    saturations = [spec["liquid"]["hsv"][1] for spec in specs]
    families = {spec["liquid"]["hue_family"] for spec in specs}
    assert min(values) >= REAL_LOOK_LIQUID_VALUE_RANGE[0]
    assert max(values) <= REAL_LOOK_LIQUID_VALUE_RANGE[1]
    assert sum(value <= 0.12 for value in values) == 300
    assert sum(value >= 0.20 for value in values) == 600
    assert families == set(REAL_LOOK_HUE_FAMILIES)
    assert sum(
        REAL_LOOK_SATURATED_SATURATION_RANGE[0]
        <= saturation
        <= REAL_LOOK_SATURATED_SATURATION_RANGE[1]
        for saturation in saturations
    ) == 450
    assert sum(
        REAL_LOOK_SUBDUED_SATURATION_RANGE[0]
        <= saturation
        <= REAL_LOOK_SUBDUED_SATURATION_RANGE[1]
        for saturation in saturations
    ) == 450
    stratum_counts = {"extreme_dark": 0, "very_dark": 0, "dark": 0}
    for seed, spec in enumerate(specs):
        value = spec["liquid"]["hsv"][2]
        saturation = spec["liquid"]["hsv"][1]
        expected_saturation_band = "saturated" if seed % 2 == 0 else "subdued"
        assert spec["liquid"]["saturation_band"] == expected_saturation_band
        saturation_low, saturation_high = (
            REAL_LOOK_SATURATED_SATURATION_RANGE
            if expected_saturation_band == "saturated"
            else REAL_LOOK_SUBDUED_SATURATION_RANGE
        )
        assert saturation_low <= saturation <= saturation_high
        if seed % 9 == 0:
            low, high = REAL_LOOK_EXTREME_DARK_VALUE_RANGE
            stratum_counts["extreme_dark"] += 1
            expected_darkness_band = "extreme_dark"
        elif seed % 3 == 0:
            low, high = REAL_LOOK_VERY_DARK_VALUE_RANGE
            stratum_counts["very_dark"] += 1
            expected_darkness_band = "very_dark"
        else:
            low, high = REAL_LOOK_DARK_VALUE_RANGE
            stratum_counts["dark"] += 1
            expected_darkness_band = "dark"
        assert low <= value <= high
        assert spec["liquid"]["darkness_band"] == expected_darkness_band
    assert stratum_counts == {
        "extreme_dark": 100,
        "very_dark": 200,
        "dark": 600,
    }
    present_heights = [
        spec["sticker_pose"]["height_offset_mm"]
        for spec in specs
        if spec["sticker_pose"]["present"]
    ]
    assert min(present_heights) < -9.9
    assert max(present_heights) > 53.8
    for spec in specs:
        sticker = spec["sticker_pose"]
        assert REAL_LOOK_LABEL_WIDTH_MM[0] <= sticker["width_mm"] <= REAL_LOOK_LABEL_WIDTH_MM[1]
        assert REAL_LOOK_LABEL_HEIGHT_MM[0] <= sticker["height_mm"] <= REAL_LOOK_LABEL_HEIGHT_MM[1]
        assert min(sticker["paper_rgb"]) >= 0.70
        assert max(sticker["print_rgb"]) <= 0.12
        assert REAL_LOOK_LIQUID_OPACITY_RANGE[0] <= spec["liquid"][
            "opacity"
        ] <= REAL_LOOK_LIQUID_OPACITY_RANGE[1]
        assert spec["liquid"]["opacity"] < 1.0
        if sticker["present"]:
            assert REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[0] <= abs(
                sticker["azimuth_deg"]
            ) <= REAL_LOOK_LABEL_ABS_AZIMUTH_DEG[1]
            assert REAL_LOOK_LABEL_HEIGHT_OFFSET_MM[0] <= sticker[
                "height_offset_mm"
            ] <= REAL_LOOK_LABEL_HEIGHT_OFFSET_MM[1]
        else:
            assert -32.0 <= sticker["height_offset_mm"] <= 32.0
        scene = spec["scene_look"]
        assert REAL_LOOK_HAZE_OPACITY_RANGE[0] <= scene[
            "haze_opacity"
        ] <= REAL_LOOK_HAZE_OPACITY_RANGE[1]
        assert REAL_LOOK_CONTRAST_RANGE[0] <= scene[
            "contrast_scale"
        ] <= REAL_LOOK_CONTRAST_RANGE[1]
        assert REAL_LOOK_BACKDROP_SATURATION_RANGE[0] <= scene[
            "backdrop_hsv"
        ][1] <= REAL_LOOK_BACKDROP_SATURATION_RANGE[1]
        assert REAL_LOOK_BACKDROP_VALUE_RANGE[0] <= scene[
            "backdrop_hsv"
        ][2] <= REAL_LOOK_BACKDROP_VALUE_RANGE[1]
        assert 13.2 <= sticker["radial_offset_mm"] <= 14.4
        assert REAL_LOOK_HOLDER_BRIGHTNESS_RANGE[0] <= spec[
            "holder_material"
        ]["brightness"] <= REAL_LOOK_HOLDER_BRIGHTNESS_RANGE[1]


def test_real_look_explicit_pilot_extreme_dark_anchors_are_deterministic():
    expected = {
        (28008, 48.1): 0.03241078268220834,
        (28017, 72.0): 0.03677704485146843,
    }
    for (seed, level_ml), value in expected.items():
        spec = sample_spec(
            seed,
            "pilot",
            level_ml=level_ml,
            profile="real-look-v2",
        )
        assert spec["liquid"]["darkness_band"] == "extreme_dark"
        assert spec["liquid"]["hsv"][2] == pytest.approx(value, abs=1e-15)


def test_real_look_hue_families_cover_measured_tank_anchors():
    # Medians in step8-revision2-real-appearance.json, measured from tank-level
    # videos 1-4. Red wraps at hue 1.0, so its 0.0136 equivalent is used here.
    anchors = {
        "brown": 0.013636,
        "olive": 0.189592,
        "dark_green": 0.427387,
        "blue": 0.563090,
    }
    for family, hue in anchors.items():
        hue_bounds, saturation_bounds = REAL_LOOK_HUE_FAMILIES[family]
        assert hue_bounds[0] <= hue <= hue_bounds[1]
        assert saturation_bounds[0] <= 0.310665


def test_unknown_render_profile_is_rejected():
    with pytest.raises(ValueError, match="profile must be one of"):
        sample_spec(17, "pilot", profile="unknown")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda spec: spec["liquid"].__setitem__("hue_family", "magenta"),
            "unsupported hue family",
        ),
        (
            lambda spec: spec["sticker_pose"].__setitem__("width_mm", 40.0),
            "sticker width",
        ),
        (
            lambda spec: spec["scene_look"].__setitem__("haze_opacity", 0.5),
            "haze opacity",
        ),
        (
            lambda spec: spec["liquid"].__setitem__("opacity", 1.0),
            "translucent range",
        ),
        (
            lambda spec: spec["sticker_pose"].update(
                {"present": True, "azimuth_deg": 90.0}
            ),
            "broad-facing",
        ),
        (
            lambda spec: spec["sticker_pose"].update(
                {"present": True, "height_offset_mm": 54.1}
            ),
            "raised range",
        ),
    ),
)
def test_real_look_validation_rejects_out_of_contract_values(mutate, message):
    spec = sample_spec(17, "pilot", profile="real-look-v2")
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        validate_spec(spec)


@pytest.mark.parametrize(
    ("role", "seed"),
    (("train", 0), ("train", 999_999), ("val", 1_000_000),
     ("val", 1_999_999), ("test", 2_000_000), ("test", 2_999_999)),
)
def test_role_seed_ranges(role, seed):
    assert validate_role_seed(role, seed) == seed


@pytest.mark.parametrize(
    ("role", "seed"),
    (("train", 1_000_000), ("val", 999_999), ("test", 1_999_999)),
)
def test_role_seed_leakage_is_rejected(role, seed):
    with pytest.raises(ValueError):
        validate_role_seed(role, seed)


def test_variation_ranges_and_prevalence():
    specs = [sample_spec(seed, "pilot") for seed in range(1000)]
    assert all(0.0 <= spec["level_ml"] <= 85.0 for spec in specs)
    foam_fraction = np.mean([spec["foam"]["enabled"] for spec in specs])
    sticker_fraction = np.mean(
        [spec["sticker_pose"]["present"] for spec in specs]
    )
    assert 0.44 <= foam_fraction <= 0.56
    assert 0.54 <= sticker_fraction <= 0.66
    band_counts = {
        name: sum(spec["level_sampling_band"] == name for spec in specs)
        for name, _, _, _ in LEVEL_BANDS
    }
    assert all(200 <= count <= 300 for count in band_counts.values())
    assert all(
        0.8 <= spec["lighting"]["brightness_scale"] <= 1.2
        for spec in specs
    )
    assert all(
        0.98 <= spec["camera_jitter"]["zoom_scale"] <= 1.02
        for spec in specs
    )
    holder_brightness = [
        spec["holder_material"]["brightness"] for spec in specs
    ]
    holder_roughness = [
        spec["holder_material"]["roughness"] for spec in specs
    ]
    assert min(holder_brightness) < 0.003
    assert max(holder_brightness) > 0.049
    assert all(0.002 <= value <= 0.05 for value in holder_brightness)
    assert all(0.20 <= value <= 0.85 for value in holder_roughness)


def test_explicit_pilot_levels():
    assert sample_spec(70, "pilot", level_ml=70)["level_ml"] == 70
    assert sample_spec(75, "pilot", level_ml=75)["level_sampling_band"] == "explicit"
    with pytest.raises(ValueError):
        sample_spec(86, "pilot", level_ml=86)


def test_boolean_variation_flags_are_not_coerced():
    spec = sample_spec(9, "pilot")
    spec["foam"]["enabled"] = "false"
    with pytest.raises(ValueError, match="must be boolean"):
        validate_spec(spec)


@pytest.mark.parametrize(
    ("field", "value"),
    (("brightness", 0.051), ("roughness", 0.86)),
)
def test_holder_material_range_is_enforced(field, value):
    spec = sample_spec(10, "pilot")
    spec["holder_material"][field] = value
    with pytest.raises(ValueError, match=f"holder {field}"):
        validate_spec(spec)


def test_mask_precedence_and_empty_through_glass():
    vial = np.zeros((6, 6), dtype=bool)
    vial[1:5, 1:5] = True
    liquid = np.zeros_like(vial)
    liquid[3:5, 1:5] = True
    foam = np.zeros_like(vial)
    foam[2, 1:5] = True
    front_blocked = np.zeros_like(vial)
    front_blocked[2:4, 2] = True
    mask = compose_class_mask(vial, liquid, foam, front_blocked)
    validate_class_mask(mask)
    assert mask[0, 0] == LABEL_IDS["ignore"]
    assert mask[1, 2] == LABEL_IDS["empty"]
    assert mask[4, 4] == LABEL_IDS["liquid"]
    assert mask[2, 4] == LABEL_IDS["foam"]
    assert mask[2, 2] == LABEL_IDS["blocked"]
    # A holder behind empty glass is deliberately absent from front_blocked.
    assert mask[1, 3] == LABEL_IDS["empty"]


def test_high_fill_mask_may_have_no_visible_empty_pixels():
    mask = np.array(
        [
            [255, 255, 255, 255],
            [255, 2, 2, 255],
            [255, 1, 3, 255],
            [255, 1, 1, 255],
        ],
        dtype=np.uint8,
    )
    validate_class_mask(mask)
    assert LABEL_IDS["empty"] not in set(np.unique(mask))


def test_all_ignore_mask_is_rejected():
    mask = np.full((4, 4), LABEL_IDS["ignore"], dtype=np.uint8)
    with pytest.raises(ValueError, match="scored vial pixels"):
        validate_class_mask(mask)


def test_row_volume_conversion_matches_frozen_targets():
    assert volume_ml_for_row(329) == pytest.approx(30.0)
    assert volume_ml_for_row(177) == pytest.approx(80.0)
    assert expected_surface_row(70) == pytest.approx(207.4)
    assert expected_surface_row(75) == pytest.approx(192.2)

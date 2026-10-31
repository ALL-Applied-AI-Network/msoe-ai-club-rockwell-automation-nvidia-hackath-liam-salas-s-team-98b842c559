"""Synthetic fill-level variation and label-contract checks."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.fill_level_variation import (  # noqa: E402
    LABEL_IDS,
    LEVEL_BANDS,
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


def test_row_volume_conversion_matches_frozen_targets():
    assert volume_ml_for_row(329) == pytest.approx(30.0)
    assert volume_ml_for_row(177) == pytest.approx(80.0)
    assert expected_surface_row(70) == pytest.approx(207.4)
    assert expected_surface_row(75) == pytest.approx(192.2)

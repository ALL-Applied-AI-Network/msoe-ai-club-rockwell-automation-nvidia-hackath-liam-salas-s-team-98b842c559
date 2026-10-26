"""Tests for the geometry-hashed CPU volume/height gate."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.volume_height_curve import (
    DEFAULT_VOLUME_HEIGHT_CURVE,
    VolumeHeightCurve,
    height_bounds_for_volume_ml,
    height_for_volume_ml,
    validate_volume_height_curve,
    volume_for_height_m,
    volume_height_curve_digest,
)
from fluidmix.physx_three_pour import build_twin_collider_specs


def test_volume_height_round_trip_and_bounds_are_derived_from_any_volume() -> None:
    height = height_for_volume_ml(17.0)
    lower, upper = height_bounds_for_volume_ml(17.0)

    assert volume_for_height_m(height) == pytest.approx(17.0, abs=1.0e-10)
    assert lower < height < upper
    assert volume_for_height_m(lower) == pytest.approx(0.95 * 17.0, abs=1.0e-10)
    assert volume_for_height_m(upper) == pytest.approx(1.05 * 17.0, abs=1.0e-10)


def test_curve_digest_binds_geometry_and_rejects_nonmonotone_or_capacity_inputs() -> None:
    validation = validate_volume_height_curve()

    assert validation["strictly_monotone"] is True
    assert validation["geometry_sha256"] == volume_height_curve_digest()
    altered = VolumeHeightCurve(
        height_offsets_m=DEFAULT_VOLUME_HEIGHT_CURVE.height_offsets_m,
        cross_section_area_m2=DEFAULT_VOLUME_HEIGHT_CURVE.cross_section_area_m2,
        geometry={**DEFAULT_VOLUME_HEIGHT_CURVE.geometry, "wall_count": 13},
    )
    assert volume_height_curve_digest(altered) != validation["geometry_sha256"]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_volume_height_curve(
            VolumeHeightCurve(
                height_offsets_m=(0.0, 0.1, 0.1),
                cross_section_area_m2=(1.0, 1.0, 1.0),
                geometry={"shape": "bad"},
            )
        )
    with pytest.raises(ValueError, match="capacity"):
        height_for_volume_ml(validation["capacity_ml"] * 1.01)
    with pytest.raises(ValueError, match="positive"):
        height_bounds_for_volume_ml(0.0)


def test_default_curve_and_invisible_colliders_match_the_authored_circle_within_one_percent() -> None:
    validation = validate_volume_height_curve()
    colliders = build_twin_collider_specs()

    assert validation["wall_count"] == 32
    assert len(colliders) == 33  # one floor plus the 32 invisible interior walls
    assert validation["interior_polygon_area_m2"] < validation["authored_circular_area_m2"]
    assert validation["area_relative_error"] <= 0.01
    assert validation["area_parity_within_one_percent"] is True


def test_curve_validation_rejects_an_interior_polygon_outside_the_one_percent_area_parity() -> None:
    with pytest.raises(ValueError, match="within one percent"):
        validate_volume_height_curve(
            VolumeHeightCurve(
                height_offsets_m=DEFAULT_VOLUME_HEIGHT_CURVE.height_offsets_m,
                cross_section_area_m2=DEFAULT_VOLUME_HEIGHT_CURVE.cross_section_area_m2,
                geometry={**DEFAULT_VOLUME_HEIGHT_CURVE.geometry, "wall_count": 12},
            )
        )

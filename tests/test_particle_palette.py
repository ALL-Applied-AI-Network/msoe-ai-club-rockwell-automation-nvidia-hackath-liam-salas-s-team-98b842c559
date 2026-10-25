"""Tests for the deterministic uncalibrated particle colour palette core."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.particle_palette import (
    DEFAULT_PALETTE_RESOLUTION,
    DEFAULT_SOURCE_COLORS,
    build_particle_palette,
    build_simplex_palette,
    map_compositions_to_palette,
    mix_linear_rgb,
)


def test_simplex_palette_is_frozen_ordered_and_finite():
    palette = build_simplex_palette(2)

    assert palette.shape == (6, 3)
    np.testing.assert_array_equal(palette[:3], np.eye(3))
    assert np.all(np.isfinite(palette))
    assert np.all(palette >= 0.0)
    np.testing.assert_allclose(palette.sum(axis=1), 1.0)
    np.testing.assert_array_equal(palette, build_simplex_palette(2))


def test_linear_light_mixture_uses_configurable_source_colors():
    source_colors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    colors = mix_linear_rgb([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25]], source_colors)

    np.testing.assert_allclose(colors, [[1.0, 0.0, 0.0], [0.25, 0.5, 0.25]])
    assert DEFAULT_SOURCE_COLORS.shape == (3, 3)


def test_palette_mapping_returns_stable_indices_and_receipt_hashes():
    compositions = np.asarray(
        [[1.0, 0.0, 0.0], [0.51, 0.49, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    palette_colors, indices, receipt = build_particle_palette(
        compositions, resolution=2
    )

    assert palette_colors.shape == (6, 3)
    np.testing.assert_array_equal(indices, [0, 3, 2])
    assert receipt["resolution"] == 2
    assert receipt["prototype_count"] == 6
    assert receipt["particle_count"] == 3
    assert receipt["mixing_rule"] == "linear_light_rgb_weighted_mixture_v1"
    assert receipt["calibrated"] is False
    assert receipt["physical_accuracy_claim"] is False
    for key in (
        "compositions_sha256",
        "palette_compositions_sha256",
        "palette_colors_sha256",
        "proto_indices_sha256",
        "source_colors_sha256",
    ):
        assert len(receipt[key]) == 64
        int(receipt[key], 16)

    expected_index_hash = hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()
    assert receipt["proto_indices_sha256"] == expected_index_hash

    second = build_particle_palette(compositions, resolution=2)
    np.testing.assert_array_equal(second[0], palette_colors)
    np.testing.assert_array_equal(second[1], indices)
    assert second[2] == receipt


def test_mapping_ties_are_stable_and_indices_are_pointinstancer_ready():
    indices = map_compositions_to_palette([[0.75, 0.25, 0.0]], resolution=2)

    assert indices.dtype == np.int64
    # The row lies exactly halfway between (1, 0, 0) and (0.5, 0.5, 0).
    # Frozen palette ordering resolves the tie to the first prototype.
    np.testing.assert_array_equal(indices, [0])


@pytest.mark.parametrize(
    "bad",
    [
        [[1.0, 0.0]],
        [[1.0, -0.1, 0.1]],
        [[0.5, 0.5, 0.5]],
        [[np.nan, 0.0, 1.0]],
        [[np.inf, 0.0, 0.0]],
        [[True, False, False]],
    ],
)
def test_palette_rejects_nonfinite_negative_or_non_simplex_compositions(bad):
    with pytest.raises((TypeError, ValueError)):
        build_particle_palette(bad)


def test_palette_rejects_an_empty_particle_set():
    with pytest.raises(ValueError, match="must not be empty"):
        build_particle_palette(np.empty((0, 3), dtype=np.float64))


@pytest.mark.parametrize(
    "bad_colors",
    [
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, -0.1, 1.1], [0.0, 0.0, 1.0]],
        [[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_palette_rejects_invalid_source_colors(bad_colors):
    with pytest.raises((TypeError, ValueError)):
        mix_linear_rgb([[1.0, 0.0, 0.0]], bad_colors)


@pytest.mark.parametrize("resolution", [0, -1, True, 1.5])
def test_palette_rejects_invalid_resolution(resolution):
    with pytest.raises((TypeError, ValueError)):
        build_simplex_palette(resolution)


def test_default_resolution_is_explicit_and_nonempty():
    palette_colors, indices, receipt = build_particle_palette([[1.0, 0.0, 0.0]])

    assert DEFAULT_PALETTE_RESOLUTION == 4
    assert palette_colors.shape == (15, 3)
    np.testing.assert_array_equal(indices, [0])
    assert receipt["index_bounds_passed"] is True

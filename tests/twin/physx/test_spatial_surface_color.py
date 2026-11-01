"""Tests for deterministic particle-composition projection onto a surface."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from twin.physx.particle_palette import DEFAULT_SOURCE_COLORS
from twin.physx.spatial_surface_color import (
    build_height_banded_transmission_colors,
    build_surface_vertex_colors,
)


def test_exact_distance_tie_uses_smallest_stable_integer_id():
    colors, nearest_ids, evidence = build_surface_vertex_colors(
        particle_ids=[17, 3],
        particle_positions_m=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        particle_compositions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        mesh_vertices_m=[[0.0, 0.0, 0.0]],
        chunk_size=1,
    )

    np.testing.assert_array_equal(nearest_ids, np.asarray([3], dtype=np.int64))
    np.testing.assert_allclose(colors, DEFAULT_SOURCE_COLORS[[1]])
    assert evidence["tie_break_rule"] == "smallest_stable_integer_particle_id"
    assert evidence["particle_count"] == 2
    assert evidence["mesh_vertex_count"] == 1


def test_chunk_size_and_particle_row_order_do_not_change_mapping_or_hashes():
    ids = np.asarray([20, 5, 11], dtype=np.int64)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    compositions = np.eye(3, dtype=np.float64)
    vertices = np.asarray(
        [[0.1, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.8, 0.0], [1.0, 1.0, 0.0]]
    )

    results = [
        build_surface_vertex_colors(
            ids[order],
            positions[order],
            compositions[order],
            vertices,
            chunk_size=chunk_size,
        )
        for order, chunk_size in (
            (np.asarray([0, 1, 2]), 1),
            (np.asarray([2, 0, 1]), 2),
            (np.asarray([1, 2, 0]), 99),
        )
    ]

    expected_ids = np.asarray([20, 5, 11, 5], dtype=np.int64)
    expected_colors = DEFAULT_SOURCE_COLORS[[0, 1, 2, 1]]
    hash_keys = [key for key in results[0][2] if key.endswith("_sha256")]
    for colors, nearest_ids, evidence in results:
        np.testing.assert_array_equal(nearest_ids, expected_ids)
        np.testing.assert_allclose(colors, expected_colors)
        assert evidence["mapping_rule"] == "nearest_particle_squared_euclidean_smallest_id_v1"
        assert evidence["calibrated"] is False
        assert evidence["physical_accuracy_claim"] is False
        assert evidence["volumetric_concentration_claim"] is False
        assert evidence["surface_projection_conserves_particle_mass"] is False
        assert evidence["all_vertex_mappings_resolved"] is True
        assert evidence["nearest_particle_ids_sha256"] == hashlib.sha256(
            np.asarray(expected_ids, dtype="<i8").tobytes()
        ).hexdigest()
    for key in hash_keys:
        assert len({result[2][key] for result in results}) == 1


def test_valid_simplex_inputs_are_not_mutated_and_component_totals_are_receipted():
    compositions = np.asarray(
        [[0.25, 0.50, 0.25], [0.40, 0.10, 0.50]], dtype=np.float64
    )
    original = compositions.copy()

    colors, nearest_ids, evidence = build_surface_vertex_colors(
        particle_ids=[8, 4],
        particle_positions_m=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        particle_compositions=compositions,
        mesh_vertices_m=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        chunk_size=1,
    )

    np.testing.assert_array_equal(compositions, original)
    np.testing.assert_array_equal(nearest_ids, [8, 4])
    np.testing.assert_allclose(colors, original @ DEFAULT_SOURCE_COLORS)
    np.testing.assert_allclose(evidence["particle_component_totals"], [0.65, 0.60, 0.75])
    expected_color_hash = hashlib.sha256(
        np.asarray(colors, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    assert evidence["vertex_colors_sha256"] == expected_color_hash


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "particle_ids": [],
                "particle_positions_m": np.empty((0, 3)),
                "particle_compositions": np.empty((0, 3)),
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "particle_ids must not be empty",
        ),
        (
            {
                "particle_ids": [1, 1],
                "particle_positions_m": [[0.0, 0.0, 0.0]] * 2,
                "particle_compositions": [[1.0, 0.0, 0.0]] * 2,
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "unique",
        ),
        (
            {
                "particle_ids": [1.0],
                "particle_positions_m": [[0.0, 0.0, 0.0]],
                "particle_compositions": [[1.0, 0.0, 0.0]],
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "integers",
        ),
        (
            {
                "particle_ids": [1],
                "particle_positions_m": [[np.nan, 0.0, 0.0]],
                "particle_compositions": [[1.0, 0.0, 0.0]],
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "finite",
        ),
        (
            {
                "particle_ids": [1],
                "particle_positions_m": [[True, False, False]],
                "particle_compositions": [[1.0, 0.0, 0.0]],
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "real numbers",
        ),
        (
            {
                "particle_ids": [1, 2],
                "particle_positions_m": [[0.0, 0.0, 0.0]],
                "particle_compositions": [[1.0, 0.0, 0.0]] * 2,
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "row count",
        ),
        (
            {
                "particle_ids": [1],
                "particle_positions_m": [[0.0, 0.0, 0.0]],
                "particle_compositions": [[0.6, 0.6, -0.2]],
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "non-negative",
        ),
        (
            {
                "particle_ids": [1],
                "particle_positions_m": [[0.0, 0.0, 0.0]],
                "particle_compositions": [[0.4, 0.4, 0.4]],
                "mesh_vertices_m": [[0.0, 0.0, 0.0]],
            },
            "sum to one",
        ),
        (
            {
                "particle_ids": [1],
                "particle_positions_m": [[0.0, 0.0, 0.0]],
                "particle_compositions": [[1.0, 0.0, 0.0]],
                "mesh_vertices_m": np.empty((0, 3)),
            },
            "mesh_vertices_m must not be empty",
        ),
        (
            {
                "particle_ids": [1],
                "particle_positions_m": [[0.0, 0.0, 0.0]],
                "particle_compositions": [[1.0, 0.0, 0.0]],
                "mesh_vertices_m": [[np.inf, 0.0, 0.0]],
            },
            "finite",
        ),
    ],
)
def test_malformed_or_nonfinite_inputs_fail_closed(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build_surface_vertex_colors(**kwargs)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_size_fails_closed(chunk_size):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        build_surface_vertex_colors(
            particle_ids=[1],
            particle_positions_m=[[0.0, 0.0, 0.0]],
            particle_compositions=[[1.0, 0.0, 0.0]],
            mesh_vertices_m=[[0.0, 0.0, 0.0]],
            chunk_size=chunk_size,
        )


def test_mdl_wrapper_delegates_to_omnisurface_with_exact_o1b_parameters():
    path = (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "mdl"
        / "FluidmixSpatialOmniSurface.mdl"
    )
    text = path.read_text(encoding="utf-8")

    assert "::OmniSurface::OmniSurface(" in text
    assert 'scene::data_lookup_color("displayColor", fallback_color)' in text
    assert "enable_specular_transmission: true" in text
    assert "specular_transmission_weight: 1.0" in text
    assert "specular_transmission_scattering_depth: 0.01" in text
    assert "thin_walled: false" in text
    assert "diffuse_reflection_weight: 0.0" in text
    assert "specular_reflection_ior: 1.333" in text
    assert "specular_reflection_roughness: 0.08" in text
    assert "enable_opacity: false" in text
    assert "geometry_opacity: 1.0" in text
    assert "geometry_opacity_threshold: 0.0" in text
    assert "OpenPBR" not in text
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64


def test_height_banded_transmission_preserves_incoming_layer_by_height():
    particle_z = np.asarray(
        [0.00, 0.01, 0.02, 0.03, 0.08, 0.09, 0.10, 0.11]
    )
    positions = np.column_stack(
        (np.zeros_like(particle_z), np.zeros_like(particle_z), particle_z)
    )
    lower_red = np.tile([1.0, 0.0, 0.0], (4, 1))
    upper_blue = np.tile([0.0, 1.0, 0.0], (4, 1))
    compositions = np.vstack((lower_red, upper_blue))
    mesh_z = np.asarray([0.00, 0.025, 0.055, 0.085, 0.11])
    mesh = np.column_stack((np.zeros_like(mesh_z), np.zeros_like(mesh_z), mesh_z))
    source_transmissions = np.asarray(
        ((0.95, 0.39, 0.26), (0.14, 0.61, 0.95), (1.0, 1.0, 1.0))
    )

    colors, evidence = build_height_banded_transmission_colors(
        positions,
        compositions,
        mesh,
        source_transmissions=source_transmissions,
        band_count=4,
    )

    np.testing.assert_allclose(colors[:2], np.tile(source_transmissions[0], (2, 1)))
    np.testing.assert_allclose(colors[-2:], np.tile(source_transmissions[1], (2, 1)))
    assert evidence["color_mode"] == "height_banded_particle_composition"
    assert evidence["band_count"] == 4
    assert evidence["nonempty_band_count"] == 4
    assert evidence["spatial_mixing_color_rendered"] is True
    assert evidence["spatial_layering_color_rendered"] is True
    assert len(evidence["vertex_colors_sha256"]) == 64


def test_height_banded_transmission_uses_diffused_particle_composition():
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
    compositions = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    mesh = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
    source = np.asarray(
        ((0.95, 0.39, 0.26), (0.14, 0.61, 0.95), (1.0, 1.0, 1.0))
    )

    colors, evidence = build_height_banded_transmission_colors(
        positions,
        compositions,
        mesh,
        source_transmissions=source,
        band_count=2,
    )

    expected = np.sqrt(source[0] * source[1])
    np.testing.assert_allclose(colors, np.tile(expected, (2, 1)))
    assert evidence["mixing_rule"] == "beer_law_extinction_weighted_transmission_v1"

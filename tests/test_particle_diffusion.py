"""Tests for the CPU particle-composition diffusion contract."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fluidmix.particle_diffusion as particle_diffusion
from fluidmix.particle_diffusion import (
    DIFFUSION_DT_S,
    DIFFUSION_NEIGHBOR_RADIUS_M,
    EFFECTIVE_DIFFUSIVITY_M2_S,
    build_update_receipt,
    diffuse_particle_compositions,
    diffuse_particle_compositions_with_receipt,
)


def test_radius_neighbor_backend_returns_canonical_pair_array():
    """The runtime neighbor graph is an ordered numeric array, not Python pairs."""

    radius = DIFFUSION_NEIGHBOR_RADIUS_M
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [0.5 * radius, 0.0, 0.0],
            [3.0 * radius, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    pairs = particle_diffusion._radius_neighbor_pairs(positions)

    assert isinstance(pairs, np.ndarray)
    assert pairs.dtype == np.int64
    assert pairs.shape == (3, 2)
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]


def test_public_diffusion_accumulates_repeated_pair_indices_without_python_iteration(
    monkeypatch,
):
    """Dense repeated endpoints use the vector accumulation path."""

    class NoPythonIteration(np.ndarray):
        def __iter__(self):
            raise AssertionError("neighbor pairs must not be iterated in Python")

    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64).view(
        NoPythonIteration
    )
    monkeypatch.setattr(
        particle_diffusion,
        "_radius_neighbor_pairs",
        lambda _positions: pairs,
    )
    ids = np.arange(3)
    positions = np.zeros((3, 3), dtype=np.float64)
    before = np.eye(3, dtype=np.float64)

    after, receipt = diffuse_particle_compositions_with_receipt(ids, positions, before)

    weight = (
        EFFECTIVE_DIFFUSIVITY_M2_S
        * DIFFUSION_DT_S
        / DIFFUSION_NEIGHBOR_RADIUS_M**2
    )
    expected = np.full((3, 3), weight, dtype=np.float64)
    np.fill_diagonal(expected, 1.0 - 2.0 * weight)
    np.testing.assert_allclose(after, expected, rtol=0.0, atol=1e-14)
    assert receipt["neighbor_count"] == 3
    assert receipt["max_degree"] == 2


@pytest.mark.parametrize(
    "ids, positions, before",
    [
        (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        ),
        (
            np.array([41], dtype=np.int64),
            np.array([[0.001, -0.002, 0.003]], dtype=np.float64),
            np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
        ),
    ],
)
def test_empty_and_singleton_graphs_are_exact_noops(ids, positions, before):
    after, receipt = diffuse_particle_compositions_with_receipt(ids, positions, before)

    np.testing.assert_array_equal(after, before)
    assert receipt["neighbor_count"] == 0
    assert receipt["max_degree"] == 0
    assert receipt["pair_weight"] == 0.0


def test_seeded_saved_like_state_matches_frozen_legacy_result():
    """The vector path preserves the pre-optimization numerical contract."""

    rng = np.random.default_rng(20261026)
    particle_count = 96
    positions = np.column_stack(
        (
            rng.uniform(-0.012, 0.012, particle_count),
            rng.uniform(-0.012, 0.012, particle_count),
            rng.uniform(0.0, 0.024, particle_count),
        )
    )
    before = rng.dirichlet([0.7, 1.1, 1.6], size=particle_count)
    ids = np.arange(5000, 5000 + particle_count)

    after, receipt = diffuse_particle_compositions_with_receipt(ids, positions, before)

    assert receipt["neighbor_count"] == 220
    assert receipt["max_degree"] == 11
    expected_selected_rows = np.array(
        [
            [0.008384706369593879, 0.5176864975970099, 0.47392879603339644],
            [0.05327063654138707, 0.30263310086352124, 0.6440962625950917],
            [0.24313618024201253, 0.49665136227078793, 0.26021245748719973],
            [0.507864881942642, 0.01729577404919848, 0.4748393440081595],
            [0.39407716230476353, 0.024657894151752888, 0.5812649435434835],
            [0.030177436638040497, 0.031341121519446485, 0.9384814418425129],
        ]
    )
    np.testing.assert_allclose(
        after[[0, 1, 7, 31, 63, 95]], expected_selected_rows, rtol=0.0, atol=1e-14
    )
    assert hashlib.sha256(np.round(after, 14).tobytes()).hexdigest() == (
        "b73501cf3109d4b6ce6dad1619f34c88d898de167070e45325499e608162b6e2"
    )
    np.testing.assert_allclose(
        after.astype(np.longdouble).sum(axis=0),
        before.astype(np.longdouble).sum(axis=0),
        rtol=0.0,
        atol=1e-12,
    )
    assert receipt["component_conservation_passed"] is True


def test_single_pair_uses_frozen_dimensionless_diffusion_step_and_preserves_ids():
    ids = np.array([101, 202], dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    before = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    after, receipt = diffuse_particle_compositions_with_receipt(ids, positions, before)

    step = EFFECTIVE_DIFFUSIVITY_M2_S * DIFFUSION_DT_S / DIFFUSION_NEIGHBOR_RADIUS_M**2
    expected = np.array([[1.0 - step, step, 0.0], [step, 1.0 - step, 0.0]])
    np.testing.assert_allclose(after, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(after.sum(axis=0), before.sum(axis=0), rtol=0.0, atol=1e-12)
    assert receipt["neighbor_count"] == 1
    assert receipt["particle_count"] == 2
    assert receipt["step_factor"] == pytest.approx(step)


def test_uniform_and_isolated_states_are_unchanged():
    ids = ["a", "b", "c"]
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 1.0, 1.0]])
    uniform = np.tile(np.array([[0.2, 0.3, 0.5]]), (3, 1))

    np.testing.assert_array_equal(
        diffuse_particle_compositions(ids, positions, uniform), uniform
    )

    isolated = np.eye(3)
    np.testing.assert_array_equal(
        diffuse_particle_compositions(ids, positions, isolated), isolated
    )


def test_dense_neighborhood_is_capped_by_explicit_per_row_stability_rule():
    ids = np.arange(60)
    positions = np.zeros((60, 3), dtype=np.float64)
    before = np.tile(np.eye(3, dtype=np.float64), (20, 1))

    after, receipt = diffuse_particle_compositions_with_receipt(ids, positions, before)

    assert receipt["neighbor_count"] == 1770
    assert receipt["max_degree"] == 59
    assert receipt["pair_weight"] == pytest.approx(0.25 / 59.0)
    assert np.isfinite(after).all()
    assert (after >= 0.0).all()
    np.testing.assert_allclose(after.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(after.sum(axis=0), before.sum(axis=0), rtol=0.0, atol=1e-12)


def test_f2_scale_conservation_check_tolerates_only_summation_roundoff():
    """The 2,221-particle F2 update must not fail on float64 reduction noise."""

    particle_count = 2221
    side = 14
    ids = np.arange(particle_count)
    positions = np.asarray(
        [
            (
                (index % side) * 0.003,
                ((index // side) % side) * 0.003,
                (index // (side * side)) * 0.003,
            )
            for index in range(particle_count)
        ],
        dtype=np.float64,
    )
    before = np.zeros((particle_count, 3), dtype=np.float64)
    before[:370, 0] = 1.0
    before[370:740, 1] = 1.0
    before[740:, 2] = 1.0

    after, receipt = diffuse_particle_compositions_with_receipt(ids, positions, before)

    long_before = before.astype(np.longdouble).sum(axis=0)
    long_after = after.astype(np.longdouble).sum(axis=0)
    np.testing.assert_allclose(long_after, long_before, rtol=0.0, atol=1e-12)
    assert receipt["component_conservation_passed"] is True


@pytest.mark.parametrize(
    "ids, positions, compositions, match",
    [
        ([1, 1], [[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0]], "duplicate"),
        ([1], [[0, 0]], [[1, 0, 0]], "shape"),
        ([1], [[0, 0, 0]], [[1.0, -1e-3, 0.0]], "non-negative"),
        ([1], [[0, 0, 0]], [[0.5, 0.5, 0.1]], "sum to one"),
        ([1], [[np.nan, 0, 0]], [[1, 0, 0]], "finite"),
    ],
)
def test_invalid_inputs_abstain(ids, positions, compositions, match):
    with pytest.raises((TypeError, ValueError), match=match):
        diffuse_particle_compositions(ids, positions, compositions)


def test_receipt_contains_conservation_bounds_hashes_and_neighbor_count():
    ids = np.array([7, 8], dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    before = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    after = diffuse_particle_compositions(ids, positions, before)

    receipt = build_update_receipt(ids, positions, before, after, neighbor_count=1)

    assert receipt["min_composition"] >= 0.0
    assert receipt["max_composition"] <= 1.0
    assert receipt["conservation_error"] <= 1e-14
    assert receipt["component_conservation_passed"] is True
    assert receipt["row_sum_passed"] is True
    assert receipt["nonnegative_passed"] is True
    assert receipt["particle_row_order_preserved"] is True
    assert receipt["neighbor_count"] == 1
    assert receipt["before_sha256"] == hashlib.sha256(
        np.ascontiguousarray(before, dtype=np.float64).tobytes()
    ).hexdigest()
    assert receipt["after_sha256"] == hashlib.sha256(
        np.ascontiguousarray(after, dtype=np.float64).tobytes()
    ).hexdigest()


def test_receipt_rejects_mismatched_update():
    ids = [1, 2]
    positions = np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    before = np.eye(2, 3)
    with pytest.raises(ValueError, match="row count"):
        build_update_receipt(ids, positions, before, before[:1], neighbor_count=1)

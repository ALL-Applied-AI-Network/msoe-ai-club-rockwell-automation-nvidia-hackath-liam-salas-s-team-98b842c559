"""C21's explicit fine-resolution and source-boundary contracts."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli/twin/physx"))

from twin.physx import paint_ball_emitter
from twin.physx.particle_contract import (
    C21_EFFECTIVE_PARTICLE_VOLUME_ML,
    C21_PARENT_PACKING_RECEIPT_SHA256,
    C21_RESOLUTION_PROFILE_ID,
    LEGACY_RESOLUTION_PROFILE_ID,
    particle_resolution_profile,
)
from twin.physx.dispense_residual import (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
)


SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence_c21",
    ROOT / "cli/twin/physx" / "physx_three_pour_sequence.py",
)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)
STAGE = ROOT / "README.md"


def _c21_args(*extra: str):
    return sequence.build_parser().parse_args(
        [
            "--out",
            "/tmp/c21-resolution",
            "--stage",
            str(STAGE),
            "--recipe8-residual-volume-ml",
            "0",
            "--solver-position-iterations",
            "16",
            "--physics-substeps",
            "2",
            "--water-pbd-preset",
            "--count-driven-twin-liquid-mesh",
            "--particle-resolution-profile",
            C21_RESOLUTION_PROFILE_ID,
            "--recipe8-effective-particle-volume-ml",
            str(C21_EFFECTIVE_PARTICLE_VOLUME_ML),
            "--plan-only",
            *extra,
        ]
    )


def test_c21_profile_is_explicit_cubic_hypothesis_not_a_measurement() -> None:
    legacy = particle_resolution_profile(LEGACY_RESOLUTION_PROFILE_ID)
    fine = particle_resolution_profile(C21_RESOLUTION_PROFILE_ID)

    assert legacy["spawn_spacing_m"] == 0.0025
    assert fine == {
        "profile_id": "c21_2p0mm",
        "spawn_spacing_m": 0.002,
        "fluid_rest_offset_m": 0.001,
        "particle_contact_offset_m": 0.0016835016832,
        "contact_offset_m": 0.0016835016832,
        "rest_offset_m": 0.0016666666664,
        "solid_rest_offset_m": 0.0016666666664,
        "nominal_particle_volume_ml": 0.008,
        "nominal_particle_mass_mg": 8.0,
        "render_radius_m": 0.001,
        "linear_resolution_scale": 0.8,
        "volume_resolution_scale": 0.512,
        "effective_particle_volume_ml": 0.005771707871971808,
        "calibration_mode": "resolution_scaled_effective_volume_hypothesis",
        "directly_measured": False,
        "parent_packing_receipt_sha256": C21_PARENT_PACKING_RECEIPT_SHA256,
    }
    assert C21_EFFECTIVE_PARTICLE_VOLUME_ML == pytest.approx(
        MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML * 0.512,
        rel=0.0,
        abs=1.0e-18,
    )


def test_fine_runtime_profile_does_not_mutate_legacy_default() -> None:
    legacy = paint_ball_emitter.build_runtime_particle_configuration()
    fine = paint_ball_emitter.build_runtime_particle_configuration(
        resolution_profile_id=C21_RESOLUTION_PROFILE_ID,
        max_particles=10_395,
        solver_position_iterations=16,
    )

    assert legacy["spawn_spacing_m"] == 0.0025
    assert legacy["fluid_rest_offset_m"] == 0.00125
    assert fine["spawn_spacing_m"] == 0.002
    assert fine["fluid_rest_offset_m"] == 0.001
    assert fine["particle_contact_offset_m"] == 0.0016835016832
    assert fine["rest_offset_m"] == 0.0016666666664
    assert fine["max_particles"] == 10_395
    assert fine["solver_position_iterations"] == 16


def test_c21_cli_is_explicit_and_builds_bound_recipe8_plan() -> None:
    args = _c21_args()
    sequence._validate_args(args)

    plan = sequence._runtime_paint_ball_plan(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert args.particle_resolution_profile == C21_RESOLUTION_PROFILE_ID
    assert [pour["particle_count"] for pour in plan["pours"]] == [3465] * 3
    assert plan["totals"]["particle_count"] == 10_395
    assert plan["particle_resolution_profile"]["calibration_mode"] == (
        "resolution_scaled_effective_volume_hypothesis"
    )
    assert plan["particle_resolution_profile"][
        "parent_packing_receipt_sha256"
    ] == C21_PARENT_PACKING_RECEIPT_SHA256
    assert plan["particle_resolution_profile"]["linear_resolution_scale"] == 0.8
    assert plan["particle_resolution_profile"]["volume_resolution_scale"] == 0.512
    assert all(
        event["source_reference_particle_count"] > 0
        for pour in plan["pours"]
        for event in pour["emission_schedule"]
    )
    assert receipt["arguments"]["particle_resolution_profile"] == (
        C21_RESOLUTION_PROFILE_ID
    )
    assert receipt["particle_system"]["runtime_configuration"][
        "spawn_spacing_m"
    ] == 0.002
    assert receipt["particle_resolution_profile"] == plan[
        "particle_resolution_profile"
    ]
    assert all(
        receipt["arguments"].get(field) is None
        for field in (
            "train_render_plan",
            "train_render_plan_sha256",
            "train_case_id",
            "accepted_qualification_receipt",
            "accepted_qualification_receipt_sha256",
        )
    )


def test_c21_profile_rejects_legacy_effective_volume() -> None:
    args = _c21_args()
    args.recipe8_effective_particle_volume_ml = (
        MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML
    )
    with pytest.raises(ValueError, match="C21.*effective particle volume"):
        sequence._validate_args(args)


def _source_event_gate() -> dict[str, object]:
    return {
        "resolution_profile_id": C21_RESOLUTION_PROFILE_ID,
        "support_gate_passed": True,
        "centroid_gate_passed": True,
        "separation_gate_passed": True,
        "flux_gate_passed": True,
        "source_velocity_m_s": [0.0, 0.0, -0.1],
    }


def test_terminal_validator_fails_closed_on_c21_lineage_or_source_gate() -> None:
    receipt = sequence._empty_paint_ball_receipt(_c21_args())
    receipt["source_emission_geometry"].update(
        {
            "events": [_source_event_gate()],
            "expected_event_count": 1,
            "observed_event_count": 1,
            "all_support_gates_passed": True,
            "all_centroid_gates_passed": True,
            "all_separation_gates_passed": True,
            "all_flux_gates_passed": True,
            "gate_passed": True,
        }
    )
    failures = paint_ball_emitter.validate_paint_ball_receipt(receipt)
    assert "particle_resolution_profile" not in failures
    assert "source_emission_geometry" not in failures

    receipt["source_emission_geometry"]["events"][0][
        "flux_gate_passed"
    ] = False
    assert "source_emission_geometry" in (
        paint_ball_emitter.validate_paint_ball_receipt(receipt)
    )

    receipt["source_emission_geometry"]["events"][0][
        "flux_gate_passed"
    ] = True
    receipt["particle_resolution_profile"][
        "parent_packing_receipt_sha256"
    ] = "0" * 64
    assert "particle_resolution_profile" in (
        paint_ball_emitter.validate_paint_ball_receipt(receipt)
    )


def test_resolution_aware_source_preserves_c20_boundary_and_centroid() -> None:
    factory = getattr(
        paint_ball_emitter, "build_resolution_aware_source_sample", None
    )
    assert callable(factory), "C21 source-boundary sampler is missing"

    sample = factory(
        particle_count=23,
        reference_particle_count=12,
        resolution_profile_id=C21_RESOLUTION_PROFILE_ID,
        source_velocity_m_s=(0.0, 0.0, -0.1),
    )
    points = np.asarray(sample["local_positions_m"], dtype=np.float64)
    receipt = sample["attestation"]

    assert points.shape == (23, 3)
    assert np.all(np.isfinite(points))
    assert receipt["reference_geometry"] == "c20_unjittered_3x3_lattice"
    assert receipt["lattice_dimensions"] == [4, 4, 3]
    assert receipt["reference_center_support_m"] == [0.005, 0.005, 0.0025]
    assert np.ptp(points, axis=0) == pytest.approx(
        [0.0055, 0.0055, 0.003], rel=0.0, abs=1.0e-15
    )
    assert max(abs(value) for value in receipt["centroid_delta_m"]) <= receipt[
        "centroid_discretization_tolerance_m"
    ]
    assert receipt["reference_physical_support_m"] == [0.0075, 0.0075, 0.005]
    assert receipt["observed_physical_support_m"] == pytest.approx(
        [0.0075, 0.0075, 0.005], rel=0.0, abs=1.0e-15
    )
    assert receipt["source_velocity_m_s"] == [0.0, 0.0, -0.1]
    assert receipt["support_gate_passed"] is True
    assert receipt["centroid_gate_passed"] is True
    assert receipt["separation_gate_passed"] is True
    assert receipt["minimum_center_separation_m"] >= 0.0015
    assert receipt["flux_gate_passed"] is True
    assert abs(receipt["event_equivalent_volume_delta_ml"]) <= receipt[
        "flux_discretization_tolerance_ml"
    ]
    assert receipt["flux_discretization_tolerance_ml"] == (
        MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML
    )
    assert receipt["documented_pattern"] == (
        "NVIDIA FluidBallEmitterDemo.particle_sphere recomputes lattice dimensions "
        "from fixed physical radius and particle spacing"
    )


def test_every_c21_recipe8_emission_passes_the_source_boundary_falsifier() -> None:
    plan = sequence._runtime_paint_ball_plan(_c21_args())
    attestations = [
        paint_ball_emitter.build_resolution_aware_source_sample(
            particle_count=event["particle_count"],
            reference_particle_count=event["source_reference_particle_count"],
            resolution_profile_id=C21_RESOLUTION_PROFILE_ID,
            source_velocity_m_s=(0.0, 0.0, -0.1),
        )["attestation"]
        for pour in plan["pours"]
        for event in pour["emission_schedule"]
    ]

    assert len(attestations) == 371
    assert all(row["support_gate_passed"] is True for row in attestations)
    assert all(row["centroid_gate_passed"] is True for row in attestations)
    assert all(row["separation_gate_passed"] is True for row in attestations)
    assert all(row["flux_gate_passed"] is True for row in attestations)
    assert min(row["minimum_center_separation_m"] for row in attestations) >= (
        0.0015 - 1.0e-15
    )


@pytest.mark.parametrize(
    "particle_count,reference_particle_count",
    [(0, 12), (23, 0), (True, 12), (23, True)],
)
def test_resolution_aware_source_rejects_invalid_counts(
    particle_count: object, reference_particle_count: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        paint_ball_emitter.build_resolution_aware_source_sample(
            particle_count=particle_count,
            reference_particle_count=reference_particle_count,
            resolution_profile_id=C21_RESOLUTION_PROFILE_ID,
            source_velocity_m_s=(0.0, 0.0, -0.1),
        )

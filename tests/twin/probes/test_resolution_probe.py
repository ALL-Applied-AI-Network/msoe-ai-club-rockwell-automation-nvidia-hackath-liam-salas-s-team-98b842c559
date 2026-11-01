from __future__ import annotations

import inspect
import math

import numpy as np

import cli.twin.probes.physx_particle_array_probe as array_runner
import cli.twin.probes.physx_resolution_probe as resolution_runner
from twin.probes.resolution_probe import (
    CAPTURE_STEPS,
    EXPECTED_PARTICLE_COUNTS,
    STAGE_TARGETS_M,
    STAGE_TOLERANCE_M,
    build_probe_plan,
    evaluate_hypothesis,
    nearest_neighbor_diagnostic,
)


def test_plan_freezes_documented_one_millimetre_resolution() -> None:
    plan = build_probe_plan()

    assert plan["resolution"]["fluid_rest_offset_m"] == 0.001
    assert plan["resolution"]["particle_contact_offset_m"] == 0.001 / (0.99 * 0.6)
    assert plan["resolution"]["contact_offset_m"] == plan["resolution"]["particle_contact_offset_m"]
    assert plan["resolution"]["rest_offset_m"] == 0.001 / 0.6
    assert plan["resolution"]["solid_rest_offset_m"] == 0.001 / 0.6
    assert math.isclose(
        plan["resolution"]["per_particle_mass_kg"],
        8.0e-6,
        rel_tol=0.0,
        abs_tol=1.0e-20,
    )
    assert plan["expected_particle_counts"] == {"1412": 1250, "2622": 2500}
    assert plan["maximum_particle_count"] == 7500
    assert plan["runtime_plan"]["pours"][0]["particle_count"] == 1250
    assert plan["runtime_plan"]["pours"][1]["particle_count"] == 1250
    assert plan["runtime_plan"]["pours"][2]["particle_count"] == 5000
    assert plan["diffusion_enabled"] is False
    assert plan["render_product_created"] is False


def test_shared_runner_has_bounded_resolution_hooks_with_unchanged_defaults() -> None:
    parameters = inspect.signature(array_runner.run_probe).parameters

    assert parameters["max_particles"].default == 2221
    assert parameters["diffusion_enabled"].default is True


def test_runtime_plan_preserves_timing_and_exact_volume_counts() -> None:
    plan = build_probe_plan()["runtime_plan"]

    assert [pour["end_step"] for pour in plan["pours"]] == [1412, 2622, 3968]
    assert [len(pour["emission_schedule"]) for pour in plan["pours"]] == [118, 101, 113]
    assert [
        sum(row["particle_count"] for row in pour["emission_schedule"])
        for pour in plan["pours"]
    ] == [1250, 1250, 5000]
    assert plan["totals"]["nominal_mass_convention_equivalent_ml"] == 60.0


def test_nn_diagnostic_reports_duplicates_without_making_them_a_gate() -> None:
    rows = np.column_stack(
        (
            np.arange(8, dtype=np.float64) * 0.002,
            np.zeros(8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
        )
    )
    rows[1] = rows[0]

    diagnostic = nearest_neighbor_diagnostic(rows)

    assert diagnostic["exact_duplicate_pair_count"] == 1
    assert diagnostic["minimum_m"] == 0.0
    assert diagnostic["acceptance_gate"] is False

    rows[2] = rows[0]
    assert nearest_neighbor_diagnostic(rows)["exact_duplicate_pair_count"] == 3


def test_hypothesis_requires_both_frozen_height_bands_and_containment() -> None:
    stages = []
    for step in CAPTURE_STEPS:
        stages.append(
            {
                "physics_step": step,
                "inside_vial_interior_count": EXPECTED_PARTICLE_COUNTS[step],
                "outside_vial_interior_count": 0,
                "escaped_particle_count": 0,
                "ids_exact": True,
                "diagnostics": {
                    "raw_simulation_positions": {
                        "wall_to_center_support_m": STAGE_TARGETS_M[step]
                    }
                },
            }
        )
    receipt = {"status": "pass", "errors": [], "stages": stages}

    assert evaluate_hypothesis(receipt)["decision"] == "pass"

    receipt["stages"][1]["diagnostics"]["raw_simulation_positions"][
        "wall_to_center_support_m"
    ] = STAGE_TARGETS_M[2622] - STAGE_TOLERANCE_M - 1.0e-6
    result = evaluate_hypothesis(receipt)
    assert result["decision"] == "falsified_resolution_convergence"
    assert "stage_2622_support_outside_frozen_band" in result["failed_predicates"]


def test_mass_transition_validator_binds_every_resolution_event() -> None:
    records = []
    for frozen in resolution_runner._expected_transitions():
        count = frozen["active_particle_count"]
        total = count * build_probe_plan()["resolution"]["per_particle_mass_kg"]
        records.append(
            {
                **frozen,
                "per_particle_mass_kg": build_probe_plan()["resolution"][
                    "per_particle_mass_kg"
                ],
                "requested_total_mass_kg": total,
                "readback_total_mass_kg": total,
                "readback_matches": True,
            }
        )

    assert resolution_runner._validate_mass_transitions(records) == ()
    records[-1]["active_particle_count"] -= 1
    assert "mass_transition_schedule_mismatch" in (
        resolution_runner._validate_mass_transitions(records)
    )

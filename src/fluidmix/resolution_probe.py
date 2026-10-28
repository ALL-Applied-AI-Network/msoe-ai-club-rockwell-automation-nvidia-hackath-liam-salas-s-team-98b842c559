"""Pure contract for the bounded A40 particle-resolution probe."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .physx_three_pour import (
    build_paint_ball_three_pour_plan,
    distribute_particle_count,
)
from .particle_array_probe import build_probe_plan as build_a36_probe_plan


EXPERIMENT = "a40_physx_one_millimetre_resolution_probe"
SCHEMA_VERSION = 1
SEED = 20261025
PHYSICS_HZ = 120
CAPTURE_STEPS = (1412, 2622)
EXPECTED_PARTICLE_COUNTS = {1412: 1250, 2622: 2500}
MAXIMUM_PARTICLE_COUNT = 7500
FLUID_REST_OFFSET_M = 0.001
PARTICLE_CONTACT_OFFSET_M = FLUID_REST_OFFSET_M / (0.99 * 0.6)
CONTACT_OFFSET_M = PARTICLE_CONTACT_OFFSET_M
REST_OFFSET_M = 0.99 * PARTICLE_CONTACT_OFFSET_M
SOLID_REST_OFFSET_M = REST_OFFSET_M
DENSITY_KG_M3 = 1000.0
PARTICLE_MASS_KG = 8.0 * FLUID_REST_OFFSET_M**3 * DENSITY_KG_M3
MAX_VELOCITY_M_S = 2.5
STAGE_TARGETS_M = {1412: 0.01912223523598308, 2622: 0.03824447047196616}
STAGE_TOLERANCE_M = 0.006
MASS_READBACK_ABSOLUTE_TOLERANCE_KG = 1.0e-8
NVIDIA_PARTICLE_GUIDE_URL = (
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/"
    "dev_guide/particles/particles.html"
)


def build_resolution_runtime_plan() -> dict[str, object]:
    """Return the unchanged F2 timing with 1 mm-resolution particle counts."""

    plan = deepcopy(build_paint_ball_three_pour_plan())
    counts = (1250, 1250, 5000)
    previous_end = 0
    for pour, count in zip(plan["pours"], counts):
        duration = int(pour["duration_steps"])
        chunks = int(pour["emission_chunk_count"])
        distributed = distribute_particle_count(count, chunks)
        schedule = []
        for index, chunk_count in enumerate(distributed):
            schedule.append(
                {
                    "ordinal": index + 1,
                    "physics_step": previous_end
                    + math.ceil((index + 1) * duration / chunks),
                    "particle_count": chunk_count,
                }
            )
        pour["particle_count"] = count
        pour["emission_schedule"] = schedule
        pour["nominal_mass_convention_equivalent_ml"] = (
            count * PARTICLE_MASS_KG * 1.0e6 / DENSITY_KG_M3
        )
        pour["nominal_error_ml"] = (
            pour["nominal_mass_convention_equivalent_ml"]
            - float(pour["programmed_volume_ml"])
        )
        previous_end = int(pour["end_step"])
    plan["fluid_rest_offset_m"] = FLUID_REST_OFFSET_M
    plan["nominal_particle_volume_ml"] = PARTICLE_MASS_KG * 1.0e6 / DENSITY_KG_M3
    plan["totals"] = {
        "programmed_volume_ml": 60,
        "particle_count": MAXIMUM_PARTICLE_COUNT,
        "nominal_mass_convention_equivalent_ml": 60.0,
        "nominal_error_ml": 0.0,
    }
    plan["paint_ball_emitter"]["max_particles"] = MAXIMUM_PARTICLE_COUNT
    return plan


def build_probe_plan() -> dict[str, object]:
    """Return the frozen A40 plan."""

    runtime_plan = build_resolution_runtime_plan()
    plan = deepcopy(build_a36_probe_plan())
    plan.update({
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "seed": SEED,
        "attempt_limit": 1,
        "hypothesis": (
            "Changing only the F2 particle resolution to a documented-offset "
            "1.0 mm fluid-rest radius makes both 10 mL and 20 mL particle "
            "support heights enter the already frozen +/-6 mm bands with "
            "finite, identity-preserving, fully contained particle state"
        ),
        "falsifiers": [
            "application_crash_or_hang",
            "timeline_callback_drift",
            "missing_nonfinite_or_unreadable_array",
            "particle_identity_or_containment_failure",
            "mass_transition_missing_or_readback_mismatch",
            "stage1_support_outside_frozen_band",
            "stage2_support_outside_frozen_band",
        ],
        "resolution": {
            "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
            "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
            "contact_offset_m": CONTACT_OFFSET_M,
            "rest_offset_m": REST_OFFSET_M,
            "solid_rest_offset_m": SOLID_REST_OFFSET_M,
            "per_particle_mass_kg": PARTICLE_MASS_KG,
            "density_kg_m3": DENSITY_KG_M3,
            "spawn_spacing_m": 2.0 * FLUID_REST_OFFSET_M,
            "source": NVIDIA_PARTICLE_GUIDE_URL,
            "documented_formula": {
                "contact": "1.0*particle_contact_offset",
                "rest": "0.99*particle_contact_offset",
                "fluid_rest": "0.99*0.6*particle_contact_offset",
                "solid_rest": "0.99*particle_contact_offset",
                "fluid_mass": "8*fluid_rest_offset^3*density",
            },
        },
        "expected_particle_counts": {
            str(step): count for step, count in EXPECTED_PARTICLE_COUNTS.items()
        },
        "maximum_particle_count": MAXIMUM_PARTICLE_COUNT,
        "runtime_plan": runtime_plan,
        "diffusion_enabled": False,
        "render_product_created": False,
        "decision": {
            str(step): {
                "target_m": target,
                "lower_inclusive_m": target - STAGE_TOLERANCE_M,
                "upper_inclusive_m": target + STAGE_TOLERANCE_M,
            }
            for step, target in STAGE_TARGETS_M.items()
        },
        "unchanged_from_a37": [
            "twin_stage_and_colliders",
            "water_pbd_material",
            "physics_hz_120",
            "real_like_pour_timing",
            "seed_20261025",
            "emission_cadence_10_hz",
            "ccd_enabled",
            "max_velocity_2.5_m_s",
            "one_shared_point_instancer",
            "explicit_runtime_total_mass_readback",
        ],
        "claims": {
            "resolution_convergence_probe_only": True,
            "volume_stage_accepted": False,
            "render_or_rgb_created": False,
            "pbd_validated_navier_stokes": False,
            "real_fluid_accuracy_validated": False,
            "sim_to_real_validated": False,
            "rescores_prior_runs": False,
        },
    })
    plan["particle_system"].update(
        {
            "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
            "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
            "contact_offset_m": CONTACT_OFFSET_M,
            "rest_offset_m": REST_OFFSET_M,
            "solid_rest_offset_m": SOLID_REST_OFFSET_M,
            "particle_mass_argument_kg": PARTICLE_MASS_KG,
            "density_argument_kg_m3": DENSITY_KG_M3,
        }
    )
    return plan


def expected_total_mass_kg(active_particle_count: int) -> float:
    if isinstance(active_particle_count, bool) or not isinstance(
        active_particle_count, int
    ):
        raise TypeError("active_particle_count must be an integer")
    if active_particle_count <= 0:
        raise ValueError("active_particle_count must be positive")
    return active_particle_count * PARTICLE_MASS_KG


def nearest_neighbor_diagnostic(positions_m: object) -> dict[str, object]:
    """Compute exact nearest-neighbour diagnostics in bounded memory."""

    rows = np.asarray(positions_m, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (3,) or len(rows) < 2:
        raise ValueError("positions_m must have shape (N,3), N>=2")
    if not np.all(np.isfinite(rows)):
        raise ValueError("positions_m must be finite")
    nearest_squared = np.full(len(rows), np.inf, dtype=np.float64)
    block = 256
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        delta = rows[start:stop, None, :] - rows[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        local = np.arange(stop - start)
        squared[local, start + local] = np.inf
        nearest_squared[start:stop] = np.min(squared, axis=1)
    nearest = np.sqrt(nearest_squared)
    ordered = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    adjacent_equal = np.all(ordered[1:] == ordered[:-1], axis=1)
    exact_duplicates = 0
    run_length = 1
    for equal in adjacent_equal:
        if bool(equal):
            run_length += 1
        else:
            exact_duplicates += run_length * (run_length - 1) // 2
            run_length = 1
    exact_duplicates += run_length * (run_length - 1) // 2
    return {
        "count": len(rows),
        "q05_m": float(np.quantile(nearest, 0.05)),
        "q50_m": float(np.quantile(nearest, 0.50)),
        "q95_m": float(np.quantile(nearest, 0.95)),
        "minimum_m": float(np.min(nearest)),
        "maximum_m": float(np.max(nearest)),
        "exact_duplicate_pair_count": exact_duplicates,
        "documented_fluid_rest_distance_m": 2.0 * FLUID_REST_OFFSET_M,
        "acceptance_gate": False,
    }


def evaluate_hypothesis(receipt: object) -> dict[str, object]:
    failures: list[str] = []
    supports: dict[str, float | None] = {str(step): None for step in CAPTURE_STEPS}
    if not isinstance(receipt, Mapping):
        failures.append("receipt_not_mapping")
        stages: list[object] = []
    else:
        stages_value = receipt.get("stages")
        stages = stages_value if isinstance(stages_value, list) else []
        if receipt.get("status") != "pass" or receipt.get("errors") != []:
            failures.append("execution_not_pass")
    if len(stages) != 2:
        failures.append("stage_count_mismatch")
    for stage in stages:
        if not isinstance(stage, Mapping):
            failures.append("stage_invalid")
            continue
        step = stage.get("physics_step")
        expected = EXPECTED_PARTICLE_COUNTS.get(step)
        if expected is None:
            failures.append("unexpected_stage")
            continue
        if (
            stage.get("inside_vial_interior_count") != expected
            or stage.get("outside_vial_interior_count") != 0
            or stage.get("escaped_particle_count") != 0
            or stage.get("ids_exact") is not True
        ):
            failures.append(f"stage_{step}_containment_or_identity")
        diagnostic = stage.get("diagnostics")
        raw = diagnostic.get("raw_simulation_positions") if isinstance(diagnostic, Mapping) else None
        support = raw.get("wall_to_center_support_m") if isinstance(raw, Mapping) else None
        if isinstance(support, (int, float)) and not isinstance(support, bool) and math.isfinite(float(support)):
            supports[str(step)] = float(support)
        else:
            failures.append(f"stage_{step}_support_missing_or_nonfinite")
    for step, target in STAGE_TARGETS_M.items():
        support = supports[str(step)]
        if support is None:
            continue
        if not target - STAGE_TOLERANCE_M <= support <= target + STAGE_TOLERANCE_M:
            failures.append(f"stage_{step}_support_outside_frozen_band")
    decision = "pass" if not failures else "falsified_resolution_convergence"
    return {
        "decision": decision,
        "passed": not failures,
        "failed_predicates": failures,
        "stage_support_m": supports,
        "rescores_prior_runs": False,
        "claim_limit": "synthetic PBD resolution diagnostic only",
    }

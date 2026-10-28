"""Pure contract for the A39 pre-playback particle-registration probe."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


EXPERIMENT = "a39_physx_preplayback_registration_probe"
SCHEMA_VERSION = 1
SEED = 20261025
EXPECTED_PARTICLE_COUNT = 740
PHYSICS_HZ = 120
PHYSICS_STEPS = 1412
FLUID_REST_OFFSET_M = 0.0015
FLUID_REST_DISTANCE_M = 2.0 * FLUID_REST_OFFSET_M
PARTICLE_TO_RIGID_REST_OFFSET_M = 0.0029
DENSITY_KG_M3 = 1000.0
PARTICLE_MASS_KG = 8.0 * FLUID_REST_OFFSET_M**3 * DENSITY_KG_M3
TOTAL_MASS_KG = EXPECTED_PARTICLE_COUNT * PARTICLE_MASS_KG
STAGE2_TARGET_M = 0.03824447047196616
STAGE2_TOLERANCE_M = 0.006
STAGE2_LOWER_M = STAGE2_TARGET_M - STAGE2_TOLERANCE_M
STAGE2_UPPER_M = STAGE2_TARGET_M + STAGE2_TOLERANCE_M
MINIMUM_FINAL_MEDIAN_NEAREST_NEIGHBOR_M = 0.0027
EXACT_DUPLICATE_TOLERANCE_M = 1.0e-9
HIGH_TAIL_THRESHOLD_Z_M = 1.0262065645
A37_RECEIPT_SHA256 = (
    "dcd8def930620684909c8b4a2ad74bb7c4c6aedd4824189f5149e7bceefa5501"
)
A37_MANIFEST_SHA256 = (
    "e3e419338440879109b3eb111396d7cf2d96c6faa03f0fdb88aedb23e3adc7a5"
)
A37_INPUT_ARRAYS: dict[str, dict[str, object]] = {
    "raw_simulation_positions": {
        "path": "step-02622/raw_simulation_positions.npy",
        "sha256": "04b9751932d7ac438d4cbfa7737f21fc31262c929415225fc8ee9e2b7472b97f",
        "shape": [EXPECTED_PARTICLE_COUNT, 3],
        "dtype": "float64",
    },
    "velocities": {
        "path": "step-02622/velocities.npy",
        "sha256": "e748d22fece8bfc3e7323bd4a9da8db4dfd8bd074039197304a913b28761c6b7",
        "shape": [EXPECTED_PARTICLE_COUNT, 3],
        "dtype": "float64",
    },
    "scales": {
        "path": "step-02622/scales.npy",
        "sha256": "0e45b8228ac3e77c33a14f4e313d221de2d2fb085263743542cfcc695bcb96b8",
        "shape": [EXPECTED_PARTICLE_COUNT, 3],
        "dtype": "float64",
    },
    "ids": {
        "path": "step-02622/ids.npy",
        "sha256": "bef7bebf5df2ac6184de5fde9cd26f2305b98d2369d75738b8497f3a06d2a9ee",
        "shape": [EXPECTED_PARTICLE_COUNT],
        "dtype": "int64",
    },
    "proto_indices": {
        "path": "step-02622/proto_indices.npy",
        "sha256": "51b4771d3cc251951aa51cc1a38bafb842ed838c51ca6393f94baaad0808140a",
        "shape": [EXPECTED_PARTICLE_COUNT],
        "dtype": "int64",
    },
    "compositions": {
        "path": "step-02622/compositions.npy",
        "sha256": "9e0bb975914dcc3cc0e74ca5fdb38a38c0319bea9cd2fef244e21b7a49ebfb4d",
        "shape": [EXPECTED_PARTICLE_COUNT, 3],
        "dtype": "float64",
    },
}


def build_probe_plan() -> dict[str, object]:
    """Return the frozen A39 diagnosis contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "seed": SEED,
        "physics_hz": PHYSICS_HZ,
        "physics_steps": PHYSICS_STEPS,
        "hypothesis": (
            "A37 remained compact because a particle set created empty and extended "
            "after timeline play was not registered as a fluid set by PhysX; loading "
            "the identical hash-bound A37 stage-2 state as a nonempty set before the "
            "first post-authoring update and before timeline play expands its bulk "
            "into the frozen stage-2 height band, removes exact duplicates, and moves "
            "median nearest-neighbour spacing to at least 2.7 mm without leakage"
        ),
        "falsifiers": [
            "application_crash_or_hang",
            "missing_nonfinite_or_hash_mismatched_input",
            "fluid_self_collision_or_particle_system_relation_readback_mismatch",
            "particle_identity_or_containment_failure",
            "timeline_callback_drift",
            "final_support_outside_frozen_stage2_band",
            "final_median_nearest_neighbor_below_0.0027_m",
            "final_exact_duplicate_positions_present",
        ],
        "input": {
            "source_run": "20261025-local62-a37-explicit-mass-attempt1",
            "receipt_sha256": A37_RECEIPT_SHA256,
            "manifest_sha256": A37_MANIFEST_SHA256,
            "stage": 2,
            "physics_step": 2622,
            "particle_count": EXPECTED_PARTICLE_COUNT,
            "arrays": A37_INPUT_ARRAYS,
            "saved_velocities_used_unchanged": True,
            "saved_ids_used_unchanged": True,
            "saved_compositions_used_unchanged": True,
            "orientation_deviation": (
                "A37 did not save orientation rows; identity orientations are authored, "
                "which does not affect sphere-particle dynamics"
            ),
        },
        "control": {
            "a37_initial_wall_to_center_support_m": 0.029267833180104643,
            "a37_initial_nearest_neighbor_q50_m": 0.0014566168517088974,
            "a37_initial_exact_duplicate_pair_count": 10,
            "a37_experiment_outcome": "falsified_stage2_below_band",
        },
        "intervention": {
            "name": "nonempty_particle_set_before_physx_parse_and_timeline_play",
            "particle_set_nonempty_at_helper_call": True,
            "particle_set_nonempty_before_first_post_authoring_update": True,
            "particle_set_nonempty_before_timeline_play": True,
            "runtime_append_count": 0,
            "total_mass_kg": TOTAL_MASS_KG,
            "per_particle_mass_kg": PARTICLE_MASS_KG,
            "density_kg_m3": DENSITY_KG_M3,
            "all_other_physics_material_collider_settings_match_a37": True,
        },
        "decision": {
            "support_metric": "particle_to_rigid_rest_offset+(q95_z-q05_z)/0.90",
            "support_target_m": STAGE2_TARGET_M,
            "support_tolerance_m": STAGE2_TOLERANCE_M,
            "support_lower_inclusive_m": STAGE2_LOWER_M,
            "support_upper_inclusive_m": STAGE2_UPPER_M,
            "minimum_final_median_nearest_neighbor_m": (
                MINIMUM_FINAL_MEDIAN_NEAREST_NEIGHBOR_M
            ),
            "maximum_final_exact_duplicate_pair_count": 0,
            "tail_rows_reported_separately": True,
            "rescores_a34_or_a37": False,
            "physical_volume_gate_changed": False,
        },
        "claims": {
            "diagnoses_parse_time_registration_path": True,
            "volume_stage_accepted": False,
            "render_or_rgb_created": False,
            "pbd_validated_navier_stokes": False,
            "real_fluid_accuracy_validated": False,
            "sim_to_real_validated": False,
        },
    }


def validate_input_arrays(arrays: object) -> tuple[str, ...]:
    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    if not isinstance(arrays, Mapping) or set(arrays) != set(A37_INPUT_ARRAYS):
        return ("input_array_set_mismatch",)
    for name, specification in A37_INPUT_ARRAYS.items():
        value = arrays.get(name)
        if not isinstance(value, np.ndarray):
            fail("input_array_type_mismatch")
            continue
        if list(value.shape) != specification["shape"] or str(value.dtype) != specification["dtype"]:
            fail("input_array_shape_or_dtype_mismatch")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            fail("input_nonfinite")
    ids = arrays.get("ids")
    if isinstance(ids, np.ndarray) and not np.array_equal(
        ids, np.arange(EXPECTED_PARTICLE_COUNT, dtype=np.int64)
    ):
        fail("input_ids_mismatch")
    compositions = arrays.get("compositions")
    if isinstance(compositions, np.ndarray) and (
        np.any(compositions < 0.0)
        or not np.allclose(
            compositions.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
        )
    ):
        fail("input_compositions_mismatch")
    proto_indices = arrays.get("proto_indices")
    if isinstance(proto_indices, np.ndarray) and np.any(proto_indices < 0):
        fail("input_proto_indices_mismatch")
    return tuple(failures)


def nearest_neighbor_distances(positions: object) -> np.ndarray:
    rows = np.asarray(positions, dtype=np.float64)
    if rows.shape != (EXPECTED_PARTICLE_COUNT, 3) or not np.all(np.isfinite(rows)):
        raise ValueError("positions must be finite with shape (740, 3)")
    difference = rows[:, None, :] - rows[None, :, :]
    squared = np.einsum("ijk,ijk->ij", difference, difference)
    np.fill_diagonal(squared, np.inf)
    return np.sqrt(np.min(squared, axis=1))


def particle_metrics(positions: object) -> dict[str, float | int]:
    rows = np.asarray(positions, dtype=np.float64)
    nearest = nearest_neighbor_distances(rows)
    q05, q50, q95 = np.quantile(rows[:, 2], (0.05, 0.50, 0.95))
    difference = rows[:, None, :] - rows[None, :, :]
    squared = np.einsum("ijk,ijk->ij", difference, difference)
    upper = squared[np.triu_indices(EXPECTED_PARTICLE_COUNT, k=1)]
    return {
        "z_min_m": float(np.min(rows[:, 2])),
        "z_q05_m": float(q05),
        "z_q50_m": float(q50),
        "z_q95_m": float(q95),
        "z_max_m": float(np.max(rows[:, 2])),
        "center_support_width_m": float((q95 - q05) / 0.90),
        "wall_to_center_support_m": float(
            PARTICLE_TO_RIGID_REST_OFFSET_M + (q95 - q05) / 0.90
        ),
        "nearest_neighbor_min_m": float(np.min(nearest)),
        "nearest_neighbor_q05_m": float(np.quantile(nearest, 0.05)),
        "nearest_neighbor_q50_m": float(np.quantile(nearest, 0.50)),
        "nearest_neighbor_q95_m": float(np.quantile(nearest, 0.95)),
        "nearest_neighbor_max_m": float(np.max(nearest)),
        "fraction_nearest_neighbor_below_0_0027": float(
            np.mean(nearest < MINIMUM_FINAL_MEDIAN_NEAREST_NEIGHBOR_M)
        ),
        "exact_duplicate_pair_count": int(
            np.count_nonzero(upper <= EXACT_DUPLICATE_TOLERANCE_M**2)
        ),
        "high_tail_count_above_1_0262065645_m": int(
            np.count_nonzero(rows[:, 2] > HIGH_TAIL_THRESHOLD_Z_M)
        ),
    }


def evaluate_probe(
    initial_metrics: Mapping[str, object], final_metrics: Mapping[str, object]
) -> dict[str, object]:
    failures: list[str] = []
    support = float(final_metrics["wall_to_center_support_m"])
    median_nn = float(final_metrics["nearest_neighbor_q50_m"])
    duplicates = int(final_metrics["exact_duplicate_pair_count"])
    if not STAGE2_LOWER_M <= support <= STAGE2_UPPER_M:
        failures.append("final_support_outside_frozen_stage2_band")
    if median_nn < MINIMUM_FINAL_MEDIAN_NEAREST_NEIGHBOR_M:
        failures.append("final_median_nearest_neighbor_below_frozen_minimum")
    if duplicates != 0:
        failures.append("final_exact_duplicate_positions_present")
    passed = not failures
    return {
        "passed": passed,
        "decision": (
            "supports_runtime_registration_cause"
            if passed
            else "falsified_parse_time_registration_cause"
        ),
        "failed_predicates": failures,
        "initial_wall_to_center_support_m": float(
            initial_metrics["wall_to_center_support_m"]
        ),
        "final_wall_to_center_support_m": support,
        "support_lower_inclusive_m": STAGE2_LOWER_M,
        "support_upper_inclusive_m": STAGE2_UPPER_M,
        "initial_nearest_neighbor_q50_m": float(
            initial_metrics["nearest_neighbor_q50_m"]
        ),
        "final_nearest_neighbor_q50_m": median_nn,
        "initial_exact_duplicate_pair_count": int(
            initial_metrics["exact_duplicate_pair_count"]
        ),
        "final_exact_duplicate_pair_count": duplicates,
        "initial_high_tail_count": int(
            initial_metrics.get("high_tail_count_above_1_0262065645_m", 0)
        ),
        "final_high_tail_count": int(
            final_metrics.get("high_tail_count_above_1_0262065645_m", 0)
        ),
        "rescores_a34_or_a37": False,
    }


def validate_probe_receipt(receipt: object) -> tuple[str, ...]:
    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    if not isinstance(receipt, Mapping):
        return ("receipt_not_mapping",)
    plan = build_probe_plan()
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("experiment") != EXPERIMENT
        or receipt.get("plan") != plan
        or receipt.get("status") != "pass"
        or receipt.get("errors") != []
    ):
        fail("receipt_identity_or_status_mismatch")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        fail("runtime_missing")
    else:
        if any(
            runtime.get(name) is not False
            for name in ("render_product_created", "rgb_captured", "video_created")
        ):
            fail("unexpected_render_output")
        if (
            runtime.get("physics_callbacks") != PHYSICS_STEPS
            or runtime.get("callback_contract_passed") is not True
        ):
            fail("timeline_contract_mismatch")
    source_input = receipt.get("input")
    if not isinstance(source_input, Mapping) or any(
        source_input.get(name) is not True
        for name in ("hashes_match_frozen", "arrays_valid")
    ):
        fail("input_evidence_mismatch")
    particle = receipt.get("particle_contract")
    expected_system = ["/FillStation/FluidmixF2/ParticleSystem"]
    if not isinstance(particle, Mapping) or (
        particle.get("nonempty_at_helper_call") is not True
        or particle.get("nonempty_before_first_post_authoring_update") is not True
        or particle.get("nonempty_before_timeline_play") is not True
        or particle.get("runtime_append_count") != 0
        or particle.get("fluid") is not True
        or particle.get("self_collision") is not True
        or particle.get("particle_group") != 0
        or particle.get("particle_system_targets") != expected_system
        or particle.get("mass_readback_matches") is not True
        or particle.get("density_readback_matches") is not True
    ):
        fail("particle_contract_mismatch")
    snapshots = receipt.get("snapshots")
    for name, failure in (
        ("initial_authored", "initial_snapshot_mismatch"),
        ("final_simulated", "final_snapshot_mismatch"),
    ):
        row = snapshots.get(name) if isinstance(snapshots, Mapping) else None
        if not isinstance(row, Mapping) or (
            row.get("count") != EXPECTED_PARTICLE_COUNT
            or row.get("inside_count") != EXPECTED_PARTICLE_COUNT
            or row.get("outside_count") != 0
            or row.get("escaped_count") != 0
            or row.get("ids_exact") is not True
            or row.get("finite") is not True
            or not isinstance(row.get("metrics"), Mapping)
        ):
            fail(failure)
    evaluation = receipt.get("hypothesis_evaluation")
    if not isinstance(evaluation, Mapping):
        fail("hypothesis_evaluation_missing")
    elif receipt.get("experiment_outcome") != evaluation.get("decision"):
        fail("hypothesis_decision_mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or (
        artifacts.get("manifest_valid") is not True
        or artifacts.get("required_arrays_valid") is not True
    ):
        fail("artifact_contract_mismatch")
    return tuple(failures)


__all__ = [
    "A37_INPUT_ARRAYS",
    "A37_MANIFEST_SHA256",
    "A37_RECEIPT_SHA256",
    "DENSITY_KG_M3",
    "EXPECTED_PARTICLE_COUNT",
    "EXPERIMENT",
    "FLUID_REST_DISTANCE_M",
    "PARTICLE_MASS_KG",
    "PHYSICS_HZ",
    "PHYSICS_STEPS",
    "SCHEMA_VERSION",
    "SEED",
    "STAGE2_LOWER_M",
    "STAGE2_UPPER_M",
    "TOTAL_MASS_KG",
    "build_probe_plan",
    "evaluate_probe",
    "nearest_neighbor_distances",
    "particle_metrics",
    "validate_input_arrays",
    "validate_probe_receipt",
]

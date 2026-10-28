"""Frozen A37 explicit-equivalent-mass particle diagnostic contract."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fluidmix.particle_array_probe import (
    CAPTURE_STEPS,
    PHYSICS_HZ,
    SEED,
    build_probe_plan as build_a36_probe_plan,
    validate_probe_receipt as validate_base_probe_receipt,
)
from fluidmix.physx_three_pour import build_paint_ball_three_pour_plan


EXPERIMENT = "a37_physx_explicit_equivalent_mass_probe"
SCHEMA_VERSION = 1
FLUID_REST_OFFSET_M = 0.0015
DENSITY_KG_M3 = 1000.0
PARTICLE_MASS_KG = 8.0 * FLUID_REST_OFFSET_M**3 * DENSITY_KG_M3
MASS_READBACK_ABSOLUTE_TOLERANCE_KG = 1.0e-8
TRANSITION_COUNT = 219
STAGE1_TARGET_M = 0.01912223523598308
STAGE1_TOLERANCE_M = 0.006
STAGE1_LOWER_M = STAGE1_TARGET_M - STAGE1_TOLERANCE_M
STAGE1_UPPER_M = STAGE1_TARGET_M + STAGE1_TOLERANCE_M
STAGE2_TARGET_M = 0.03824447047196616
STAGE2_TOLERANCE_M = 0.006
STAGE2_LOWER_M = STAGE2_TARGET_M - STAGE2_TOLERANCE_M
STAGE2_UPPER_M = STAGE2_TARGET_M + STAGE2_TOLERANCE_M
A36_ATTEMPT2_RECEIPT_SHA256 = (
    "008fd317e8022caaefbb0af7874280fa5cfb252c48a1aec3b22efed468dc2a9b"
)
A36_STAGE2_RAW_WALL_TO_CENTER_SUPPORT_M = 0.030403725518327596
INSTALLED_PARTICLE_UTILS_SOURCE = (
    "/home/hackathon5/liamreporoot/isaacsim62-env/.venv/lib/python3.12/"
    "site-packages/isaacsim/extscache/omni.physx-110.4.4+110.4.0.la64.r.cp312.u7f4/"
    "omni/physx/scripts/particleUtils.py"
)
INSTALLED_PARTICLE_UTILS_SHA256 = (
    "85da0b0a39931eff3e54cea5ca8987c07d1c903557d87cb609b205ab85f9b3df"
)

NVIDIA_PARTICLE_GUIDE_URL = (
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/"
    "dev_guide/particles/particles.html"
)
NVIDIA_RUNTIME_MASS_CHANGELOG_URL = (
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/104.0/"
    "source/extensions/omni.physx.bundle/docs/CHANGELOG.html"
)
NVIDIA_PHYSX_PBD_URL = (
    "https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/"
    "ParticleSystem.html"
)


def build_probe_plan() -> dict[str, object]:
    """Return the A37 plan frozen before the single Isaac application run."""

    plan = deepcopy(build_a36_probe_plan())
    plan.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "hypothesis": (
                "A36's compact PBD bulk was caused by the runtime-empty "
                "zero-total-mass path; explicit density-equivalent total mass "
                "after every append moves stage-2 wall-to-center support into "
                "the frozen 38.244 mm +/- 6 mm band without leakage"
            ),
            "falsifiers": [
                "stage1_raw_wall_to_center_support_below_0.01312223523598308_m",
                "stage1_raw_wall_to_center_support_above_0.02512223523598308_m",
                "stage2_raw_wall_to_center_support_below_0.03224447047196616_m",
                "stage2_raw_wall_to_center_support_above_0.04424447047196616_m",
                "containment_or_identity_failure",
                "missing_nonfinite_or_unreadable_array",
                "mass_transition_missing_or_readback_mismatch",
                "timeline_callback_drift",
                "application_crash_or_hang",
            ],
            "control": {
                "run_id": "20261025-local62-a36-particle-arrays-attempt2",
                "a36_attempt2_receipt_sha256": A36_ATTEMPT2_RECEIPT_SHA256,
                "stage2_raw_wall_to_center_support_m": (
                    A36_STAGE2_RAW_WALL_TO_CENTER_SUPPORT_M
                ),
                "interpretation": (
                    "runtime-empty MassAPI diagnosis control; no A34 rescore"
                ),
            },
            "intervention": {
                "name": "explicit_runtime_total_mass",
                "formula": "8*fluid_rest_offset_m^3*density_kg_m3",
                "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
                "density_kg_m3": DENSITY_KG_M3,
                "per_particle_mass_kg": PARTICLE_MASS_KG,
                "after_each_emission_append": True,
                "total_mass_formula": (
                    "active_particle_count*per_particle_mass_kg"
                ),
                "transition_count": TRANSITION_COUNT,
                "readback_absolute_tolerance_kg": (
                    MASS_READBACK_ABSOLUTE_TOLERANCE_KG
                ),
            },
            "stage1_decision": {
                "metric": "raw_wall_to_center_support_m",
                "target_m": STAGE1_TARGET_M,
                "tolerance_m": STAGE1_TOLERANCE_M,
                "lower_inclusive_m": STAGE1_LOWER_M,
                "upper_inclusive_m": STAGE1_UPPER_M,
                "pass_if_inside_band_and_all_other_gates_pass": True,
                "falsified_if_below_lower_m": True,
            },
            "stage2_decision": {
                "metric": "raw_wall_to_center_support_m",
                "target_m": STAGE2_TARGET_M,
                "tolerance_m": STAGE2_TOLERANCE_M,
                "lower_inclusive_m": STAGE2_LOWER_M,
                "upper_inclusive_m": STAGE2_UPPER_M,
                "pass_if_inside_band_and_all_other_gates_pass": True,
                "falsified_if_below_lower_m": True,
            },
            "identical_to_a36_except": [
                "point_instancer_particle_mass_argument_kg",
                "runtime_total_mass_authored_after_each_emission_append",
                "mass_transition_receipt",
            ],
            "nearest_neighbor_diagnostic": {
                "array": "raw_simulation_positions",
                "metric": "3-D Euclidean nearest-other-particle distance",
                "quantiles": [0.05, 0.50, 0.95],
                "fraction_below_m": 0.0027,
                "fraction_above_m": 0.0033,
                "acceptance_gate": False,
            },
            "documentation": [
                {
                    "url": NVIDIA_PARTICLE_GUIDE_URL,
                    "documented_claim": (
                        "fluid particle mass is 8*fluidRestOffset^3*density; "
                        "MassAPI mass takes precedence and is divided by numParticles"
                    ),
                },
                {
                    "url": NVIDIA_RUNTIME_MASS_CHANGELOG_URL,
                    "documented_claim": (
                        "runtime user changes to PBD particle mass/density are handled"
                    ),
                },
                {
                    "url": NVIDIA_PHYSX_PBD_URL,
                    "documented_claim": (
                        "PBD solves positional constraints including volume preservation"
                    ),
                },
            ],
            "installed_source_evidence": {
                "path": INSTALLED_PARTICLE_UTILS_SOURCE,
                "sha256": INSTALLED_PARTICLE_UTILS_SHA256,
                "pointinstancer_helper_lines": "256-324",
                "observed_helper_behavior": (
                    "total MassAPI mass is particle_mass*len(initial_positions); "
                    "an empty initial set authors zero and append does not rerun helper"
                ),
                "mass_precedence_lines": "327-342",
                "claim_type": "source-file observation",
            },
            "claims": {
                "diagnoses_runtime_empty_mass_path": True,
                "fluid_rest_offset_is_target_density_parameter": True,
                "fluid_rest_offset_is_render_radius": False,
                "pbd_is_validated_navier_stokes": False,
                "real_fluid_accuracy_validated": False,
                "sim_to_real_validated": False,
                "rescores_a34": False,
            },
        }
    )
    plan["particle_system"]["particle_mass_argument_kg"] = PARTICLE_MASS_KG
    return plan


def expected_total_mass_kg(active_particle_count: int) -> float:
    if isinstance(active_particle_count, bool) or not isinstance(
        active_particle_count, int
    ):
        raise TypeError("active_particle_count must be an integer")
    if active_particle_count <= 0:
        raise ValueError("active_particle_count must be positive")
    return float(active_particle_count) * PARTICLE_MASS_KG


def frozen_mass_transition_schedule() -> list[dict[str, int]]:
    """Return exact first-two-pour append boundaries and cumulative counts."""

    cumulative = 0
    records: list[dict[str, int]] = []
    for pour in build_paint_ball_three_pour_plan()["pours"][:2]:
        for event in pour["emission_schedule"]:
            cumulative += int(event["particle_count"])
            records.append(
                {
                    "transition_ordinal": len(records) + 1,
                    "physics_step": int(event["physics_step"]),
                    "active_particle_count": cumulative,
                }
            )
    return records


def validate_mass_transitions(
    records: object, *, expected_count: int = TRANSITION_COUNT
) -> tuple[str, ...]:
    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    if (
        not isinstance(records, list)
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(records) != expected_count
    ):
        return ("mass_transition_count_mismatch",)
    previous_active_count = 0
    previous_step = 0
    frozen_schedule = (
        frozen_mass_transition_schedule()
        if expected_count == TRANSITION_COUNT
        else None
    )
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            fail("mass_transition_invalid")
            continue
        active_count = record.get("active_particle_count")
        physics_step = record.get("physics_step")
        if (
            record.get("transition_ordinal") != ordinal
            or isinstance(active_count, bool)
            or not isinstance(active_count, int)
            or active_count <= previous_active_count
            or isinstance(physics_step, bool)
            or not isinstance(physics_step, int)
            or physics_step <= previous_step
        ):
            fail("mass_transition_invalid")
            continue
        if frozen_schedule is not None and any(
            record.get(field) != frozen_schedule[ordinal - 1][field]
            for field in (
                "transition_ordinal",
                "physics_step",
                "active_particle_count",
            )
        ):
            fail("mass_transition_schedule_mismatch")
        expected_total = expected_total_mass_kg(active_count)
        numeric_fields = (
            "per_particle_mass_kg",
            "requested_total_mass_kg",
            "readback_total_mass_kg",
            "readback_per_particle_mass_kg",
        )
        if any(
            not isinstance(record.get(field), (int, float))
            or isinstance(record.get(field), bool)
            or not math.isfinite(float(record[field]))
            for field in numeric_fields
        ):
            fail("mass_transition_invalid")
        else:
            if (
                not math.isclose(
                    float(record["per_particle_mass_kg"]),
                    PARTICLE_MASS_KG,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not math.isclose(
                    float(record["requested_total_mass_kg"]),
                    expected_total,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not math.isclose(
                    float(record["readback_total_mass_kg"]),
                    expected_total,
                    rel_tol=0.0,
                    abs_tol=MASS_READBACK_ABSOLUTE_TOLERANCE_KG,
                )
                or not math.isclose(
                    float(record["readback_per_particle_mass_kg"]),
                    PARTICLE_MASS_KG,
                    rel_tol=0.0,
                    abs_tol=(
                        MASS_READBACK_ABSOLUTE_TOLERANCE_KG / active_count
                    ),
                )
                or record.get("readback_matches") is not True
            ):
                fail("mass_readback_mismatch")
        if record.get("mass_attr_authored") is not True:
            fail("mass_attr_not_authored")
        density_readback = record.get("density_readback_kg_m3")
        if (
            not isinstance(density_readback, (int, float))
            or isinstance(density_readback, bool)
            or not math.isfinite(float(density_readback))
            or not math.isclose(
                float(density_readback),
                DENSITY_KG_M3,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            fail("density_readback_mismatch")
        if record.get("density_attr_authored") is not True:
            fail("density_attr_not_authored")
        if record.get("mass_overrides_density") is not True:
            fail("mass_precedence_mismatch")
        if record.get("set_succeeded") is not True:
            fail("mass_attr_set_failed")
        previous_active_count = active_count
        previous_step = physics_step
    return tuple(failures)


def nearest_neighbor_distances(positions_m: object) -> np.ndarray:
    rows = np.asarray(positions_m, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[1:] != (3,)
        or len(rows) < 2
        or not np.all(np.isfinite(rows))
    ):
        raise ValueError("positions_m must be finite with shape (N,3), N>=2")
    delta = rows[:, None, :] - rows[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(squared, np.inf)
    return np.sqrt(np.min(squared, axis=1))


def nearest_neighbor_diagnostic(positions_m: object) -> dict[str, object]:
    nearest = nearest_neighbor_distances(positions_m)
    q05, q50, q95 = np.quantile(nearest, (0.05, 0.50, 0.95))
    return {
        "count": int(len(nearest)),
        "metric": "3-D Euclidean nearest-other-particle distance",
        "q05_m": float(q05),
        "q50_m": float(q50),
        "q95_m": float(q95),
        "minimum_m": float(np.min(nearest)),
        "maximum_m": float(np.max(nearest)),
        "fraction_below_0_0027": float(np.mean(nearest < 0.0027)),
        "fraction_above_0_0033": float(np.mean(nearest > 0.0033)),
        "documented_fluid_rest_distance_m": 2.0 * FLUID_REST_OFFSET_M,
        "acceptance_gate": False,
    }


def evaluate_hypothesis(receipt: object) -> dict[str, object]:
    other_gate_failures: list[str] = []
    stage1_support: float | None = None
    stage2_support: float | None = None
    if not isinstance(receipt, dict):
        other_gate_failures.append("receipt_not_mapping")
        stages: list[object] = []
    else:
        if receipt.get("status") != "pass" or receipt.get("errors") != []:
            other_gate_failures.append("execution_not_pass")
        stages_value = receipt.get("stages")
        stages = stages_value if isinstance(stages_value, list) else []
        if len(stages) != 2:
            other_gate_failures.append("stage_count_mismatch")
    expected_counts = {1412: 370, 2622: 740}
    for stage in stages:
        if not isinstance(stage, dict):
            other_gate_failures.append("stage_invalid")
            continue
        step = stage.get("physics_step")
        expected = expected_counts.get(step)
        if expected is None:
            other_gate_failures.append("unexpected_stage")
            continue
        if (
            stage.get("inside_vial_interior_count") != expected
            or stage.get("outside_vial_interior_count") != 0
            or stage.get("escaped_particle_count") != 0
            or stage.get("ids_exact") is not True
        ):
            other_gate_failures.append(f"stage_{step}_containment_or_identity")
        diagnostics = stage.get("diagnostics")
        raw = (
            diagnostics.get("raw_simulation_positions")
            if isinstance(diagnostics, dict)
            else None
        )
        value = raw.get("wall_to_center_support_m") if isinstance(raw, dict) else None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            if step == CAPTURE_STEPS[0]:
                stage1_support = float(value)
            elif step == CAPTURE_STEPS[-1]:
                stage2_support = float(value)
        else:
            other_gate_failures.append(f"stage_{step}_support_missing_or_nonfinite")
    if other_gate_failures:
        decision = "failed_other_gate"
    elif stage1_support is None or stage2_support is None:
        decision = "failed_other_gate"
    elif stage1_support < STAGE1_LOWER_M:
        decision = "falsified_stage1_below_band"
    elif stage1_support > STAGE1_UPPER_M:
        decision = "fail_stage1_above_band"
    elif stage2_support < STAGE2_LOWER_M:
        decision = "falsified_stage2_below_band"
    elif stage2_support > STAGE2_UPPER_M:
        decision = "fail_stage2_above_band"
    else:
        decision = "pass"
    return {
        "decision": decision,
        "stage1_raw_wall_to_center_support_m": stage1_support,
        "stage1_lower_inclusive_m": STAGE1_LOWER_M,
        "stage1_upper_inclusive_m": STAGE1_UPPER_M,
        "stage2_raw_wall_to_center_support_m": stage2_support,
        "stage2_lower_inclusive_m": STAGE2_LOWER_M,
        "stage2_upper_inclusive_m": STAGE2_UPPER_M,
        "other_gate_failures": other_gate_failures,
        "rescores_a34": False,
        "claim_limit": (
            "synthetic PBD runtime-mass diagnosis only; no real-fluid or "
            "sim-to-real validation"
        ),
    }


def validate_probe_receipt(
    receipt: object, *, artifact_root: Path
) -> tuple[str, ...]:
    """Validate A36-shared evidence plus A37 mass and NN diagnostics."""

    failures = list(
        validate_base_probe_receipt(
            receipt,
            artifact_root=artifact_root,
            expected_plan=build_probe_plan(),
            expected_experiment=EXPERIMENT,
            expected_particle_mass_argument_kg=PARTICLE_MASS_KG,
        )
    )

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    if not isinstance(receipt, dict):
        return tuple(failures)
    treatment = receipt.get("explicit_mass_treatment")
    transitions = treatment.get("transitions") if isinstance(treatment, dict) else None
    for reason in validate_mass_transitions(transitions):
        fail(reason)
    if (
        not isinstance(treatment, dict)
        or treatment.get("formula")
        != "8*fluid_rest_offset_m^3*density_kg_m3"
        or treatment.get("per_particle_mass_kg") != PARTICLE_MASS_KG
        or treatment.get("transition_count") != TRANSITION_COUNT
    ):
        fail("explicit_mass_treatment_mismatch")
    stages = receipt.get("stages")
    if isinstance(stages, list):
        manifest_pairs: set[tuple[object, object]] = set()
        artifacts = receipt.get("artifacts")
        try:
            manifest_path = (
                artifact_root / artifacts["manifest_path"]
            ).resolve(strict=True)
            manifest_path.relative_to(artifact_root.resolve(strict=True))
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_pairs = {
                (record.get("path"), record.get("sha256"))
                for record in manifest.get("records", [])
                if isinstance(record, dict)
            }
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            fail("nearest_neighbor_manifest_binding_missing")
        expected_stage_transition_indices = {1412: 117, 2622: 218}
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            diagnostic = stage.get("nearest_neighbor_diagnostic")
            artifact = stage.get("nearest_neighbor_artifact")
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("acceptance_gate") is not False
                or not isinstance(artifact, dict)
                or not isinstance(artifact.get("path"), str)
                or not isinstance(artifact.get("sha256"), str)
            ):
                fail("nearest_neighbor_diagnostic_missing")
                continue
            try:
                path = (artifact_root / artifact["path"]).resolve(strict=True)
                path.relative_to(artifact_root.resolve(strict=True))
                values = np.load(path, allow_pickle=False)
            except (OSError, ValueError):
                fail("nearest_neighbor_artifact_invalid")
                continue
            import hashlib

            if (
                hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
                or values.shape != (stage.get("expected_particle_count"),)
                or not np.all(np.isfinite(values))
            ):
                fail("nearest_neighbor_artifact_invalid")
            if (artifact.get("path"), artifact.get("sha256")) not in manifest_pairs:
                fail("nearest_neighbor_manifest_binding_missing")
            raw_record = stage.get("arrays", {}).get("raw_simulation_positions")
            try:
                raw = np.load(
                    artifact_root / raw_record["path"], allow_pickle=False
                )
                expected_diagnostic = nearest_neighbor_diagnostic(raw)
            except (KeyError, OSError, TypeError, ValueError):
                fail("nearest_neighbor_diagnostic_mismatch")
            else:
                if diagnostic != expected_diagnostic or not np.array_equal(
                    values, nearest_neighbor_distances(raw)
                ):
                    fail("nearest_neighbor_diagnostic_mismatch")
            step = stage.get("physics_step")
            transition_index = expected_stage_transition_indices.get(step)
            if (
                transition_index is None
                or not isinstance(transitions, list)
                or len(transitions) <= transition_index
                or stage.get("mass_state") != transitions[transition_index]
            ):
                fail("stage_mass_state_mismatch")
    else:
        fail("nearest_neighbor_diagnostic_missing")
    expected_evaluation = evaluate_hypothesis(receipt)
    if receipt.get("hypothesis_evaluation") != expected_evaluation:
        fail("hypothesis_evaluation_mismatch")
    return tuple(failures)

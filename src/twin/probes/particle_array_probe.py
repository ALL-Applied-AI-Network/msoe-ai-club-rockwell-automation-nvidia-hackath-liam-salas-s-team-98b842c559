"""Frozen A36 full-particle-array diagnostic contract.

This module is deliberately independent of Isaac Sim so the experiment plan,
height diagnostic, and fail-closed artifact checks can be tested on CPU.  The
diagnostic describes the distribution of particle *centres*.  In particular,
``fluidRestOffset`` is not interpreted as a physical particle radius.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from twin.io import sha256_file as _sha256
from twin.physx.three_pour import (
    COMPOSITION_COMPONENTS,
    build_paint_ball_three_pour_plan,
    build_twin_collider_specs,
)


SCHEMA_VERSION = 1
EXPERIMENT = "a36_physx_first_two_stage_particle_array_probe"
SEED = 20261025
# A36 is an archived 1.5 mm diagnostic; it must not silently inherit the
# reviewed F2 2.5 mm execution contract.
FLUID_REST_OFFSET_M = 0.0015
PHYSICS_HZ = 120
CAPTURE_STEPS = (1412, 2622)
EXPECTED_PARTICLE_COUNTS = {1412: 370, 2622: 740}
ARRAY_NAMES = (
    "raw_simulation_positions",
    "smoothed_instancer_positions",
    "velocities",
    "scales",
    "ids",
    "proto_indices",
    "compositions",
)
PARTICLE_CONTACT_OFFSET_M = 0.003
CONTACT_OFFSET_M = 0.003
PARTICLE_TO_NONPARTICLE_REST_OFFSET_M = 0.0029
SOLID_REST_OFFSET_M = 0.0029
MAX_VELOCITY_M_S = 2.5
PARTICLE_MASS_ARGUMENT_KG = 0.001
PARTICLE_DENSITY_ARGUMENT_KG_M3 = 1000.0
WALL_LIMIT_SECONDS = 1800
CANONICAL_TWIN_SHA256 = (
    "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9"
)
PHYSX_FLOAT32_ABSOLUTE_TOLERANCE_M = 1.0e-9
PHYSX_FLOAT32_MATERIAL_ABSOLUTE_TOLERANCE = 1.0e-8


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def build_probe_plan() -> dict[str, object]:
    """Return the pre-run A36 plan, including the exact A34 physics plan hash."""

    a34_plan = build_paint_ball_three_pour_plan()
    collider_specs = _jsonable(build_twin_collider_specs())
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "hypothesis": (
            "complete first-two-stage particle arrays distinguish a frozen "
            "height-estimator bias or tail from an actually compact PBD bulk"
        ),
        "falsifiers": [
            "missing_or_unreadable_array",
            "nonfinite_array",
            "array_count_or_identity_mismatch",
            "containment_mismatch",
            "timeline_callback_drift",
        ],
        "denominator": 1,
        "seed": SEED,
        "runtime": "Isaac Sim 6.2.0 local GB10",
        "canonical_twin_sha256": CANONICAL_TWIN_SHA256,
        "physics_hz": PHYSICS_HZ,
        "physics_dt_s": 1.0 / PHYSICS_HZ,
        "water_pbd_preset": True,
        "programmed_pours_ml": [10, 10, 40],
        "tank_order": list(COMPOSITION_COMPONENTS),
        "capture_steps": list(CAPTURE_STEPS),
        "stop_after_physics_step": CAPTURE_STEPS[-1],
        "timeline": {
            "start_time_s": 0.0,
            "looping": False,
            "end_margin_callbacks": 1,
            "required_end_time_s": (CAPTURE_STEPS[-1] + 1) / PHYSICS_HZ,
            "configuration_source": (
                "A34 timeline boundary probe 2750/2750 monotonic callbacks"
            ),
        },
        "expected_particle_counts": {
            str(step): count for step, count in EXPECTED_PARTICLE_COUNTS.items()
        },
        "array_names": list(ARRAY_NAMES),
        "array_format": "NumPy NPY, allow_pickle=False, lossless native dtype",
        "render_product": False,
        "rgb": False,
        "video": False,
        "a34_physics_plan_sha256": _canonical_json_sha256(a34_plan),
        "a34_physics_plan_schema_version": a34_plan["schema_version"],
        "particle_system": {
            "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
            "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
            "contact_offset_m": CONTACT_OFFSET_M,
            "rest_offset_m": PARTICLE_TO_NONPARTICLE_REST_OFFSET_M,
            "solid_rest_offset_m": SOLID_REST_OFFSET_M,
            "max_velocity_m_s": MAX_VELOCITY_M_S,
            "ccd": True,
            "smoothing": {"enabled": True, "strength": 1.0},
            "anisotropy": {"enabled": True, "scale": 1.0},
            "particle_mass_argument_kg": PARTICLE_MASS_ARGUMENT_KG,
            "density_argument_kg_m3": PARTICLE_DENSITY_ARGUMENT_KG_M3,
            "float32_readback_absolute_tolerance_m": (
                PHYSX_FLOAT32_ABSOLUTE_TOLERANCE_M
            ),
        },
        "collider_specs": collider_specs,
        "diagnostic": {
            "population": "contained particle centres only",
            "quantiles": [0.05, 0.95],
            "center_support_formula": "(Q95_z-Q05_z)/0.90",
            "wall_to_center_formula": (
                "particle_to_nonparticle_rest_offset_m+center_support_width_m"
            ),
            "visual_prototype_formula": (
                "center_support_width_m+2*prototype_radius_m*prototype_scale_z"
            ),
            "fluid_rest_offset_is_physical_radius": False,
            "visual_prototype_support_is_physical": False,
            "a34_values_used_to_tune_thresholds": False,
            "rescores_a34": False,
            "decision_threshold": None,
        },
        "limits": [
            "diagnostic run only; no A34 rescore",
            "particle-centre support is not displaced liquid volume",
            "sphere-prototype support is a non-physical visual diagnostic",
            "synthetic PBD evidence does not establish real-fluid accuracy",
        ],
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
    }


def center_support_diagnostic(
    z_m: Iterable[float] | np.ndarray,
    *,
    particle_to_nonparticle_rest_offset_m: float,
    prototype_radius_m: float,
    prototype_scale_z: float = 1.0,
) -> dict[str, float | bool | str]:
    """Describe centre support without treating ``fluidRestOffset`` as radius."""

    values = np.asarray(z_m, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("z_m must be a non-empty finite one-dimensional array")
    for name, value in (
        (
            "particle_to_nonparticle_rest_offset_m",
            particle_to_nonparticle_rest_offset_m,
        ),
        ("prototype_radius_m", prototype_radius_m),
        ("prototype_scale_z", prototype_scale_z),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
    q05, q95 = np.quantile(values, (0.05, 0.95))
    center_width = float((q95 - q05) / 0.90)
    wall_to_center = float(particle_to_nonparticle_rest_offset_m) + center_width
    visual_support = center_width + 2.0 * float(prototype_radius_m) * float(
        prototype_scale_z
    )
    return {
        "population": "particle_centres",
        "count": int(values.size),
        "q05_z_m": float(q05),
        "q95_z_m": float(q95),
        "center_support_width_m": center_width,
        "particle_to_nonparticle_rest_offset_m": float(
            particle_to_nonparticle_rest_offset_m
        ),
        "wall_to_center_support_m": wall_to_center,
        "prototype_radius_m": float(prototype_radius_m),
        "prototype_scale_z": float(prototype_scale_z),
        "visual_prototype_support_m": visual_support,
        "physical_radius_interpretation": False,
        "visual_prototype_support_is_physical": False,
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_probe_receipt(
    receipt: object,
    *,
    artifact_root: Path,
    expected_plan: dict[str, object] | None = None,
    expected_experiment: str = EXPERIMENT,
    expected_particle_mass_argument_kg: float = PARTICLE_MASS_ARGUMENT_KG,
    expected_max_particles: int = 2221,
) -> tuple[str, ...]:
    """Validate a completed A36 receipt and every declared NPY artifact."""

    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    if not isinstance(receipt, dict):
        return ("receipt_not_mapping",)
    if receipt.get("schema_version") != SCHEMA_VERSION:
        fail("schema_version_mismatch")
    if receipt.get("experiment") != expected_experiment:
        fail("experiment_mismatch")
    if receipt.get("status") != "pass" or receipt.get("errors") != []:
        fail("status_not_pass")
    frozen_plan = build_probe_plan() if expected_plan is None else expected_plan
    if receipt.get("plan") != frozen_plan:
        fail("plan_mismatch")

    command = receipt.get("command")
    if (
        not isinstance(command, dict)
        or command.get("seed") != SEED
        or command.get("denominator") != 1
        or not isinstance(command.get("argv"), list)
        or not command["argv"]
        or not _valid_sha256(command.get("script_sha256"))
    ):
        fail("command_contract_mismatch")
    source = receipt.get("source")
    layers_before = source.get("layers_before") if isinstance(source, dict) else None
    layers_after = source.get("layers_after") if isinstance(source, dict) else None
    if (
        not isinstance(source, dict)
        or source.get("stage_sha256") != CANONICAL_TWIN_SHA256
        or source.get("canonical_stage_sha256") != CANONICAL_TWIN_SHA256
        or not isinstance(layers_before, list)
        or not layers_before
        or layers_before != layers_after
        or source.get("source_layers_unchanged") is not True
        or any(
            not isinstance(record, dict) or not _valid_sha256(record.get("sha256"))
            for record in layers_before
        )
    ):
        fail("source_contract_mismatch")

    runtime = receipt.get("runtime")
    if (
        not isinstance(runtime, dict)
        or not str(runtime.get("isaac_sim", "")).startswith("6.2.0")
        or not str(runtime.get("kit", "")).startswith("110.4")
        or not str(runtime.get("physx", "")).startswith("110.4")
    ):
        fail("runtime_version_mismatch")
    if not isinstance(runtime, dict) or any(
        runtime.get(field) is not False
        for field in ("render_product_created", "rgb_captured", "video_created")
    ):
        fail("render_product_forbidden")

    expected_particle_system = frozen_plan["particle_system"]
    particle_system = receipt.get("particle_system")
    expected_scalar_fields = {
        "particle_contact_offset_m": expected_particle_system[
            "particle_contact_offset_m"
        ],
        "fluid_rest_offset_m": expected_particle_system["fluid_rest_offset_m"],
        "contact_offset_m": expected_particle_system["contact_offset_m"],
        "rest_offset_m": expected_particle_system["rest_offset_m"],
        "solid_rest_offset_m": expected_particle_system["solid_rest_offset_m"],
        "max_velocity_m_s": expected_particle_system["max_velocity_m_s"],
    }
    if not isinstance(particle_system, dict) or any(
        not isinstance(particle_system.get(field), (int, float))
        or isinstance(particle_system.get(field), bool)
        or not math.isclose(
            float(particle_system[field]),
            float(expected),
            rel_tol=0.0,
            abs_tol=PHYSX_FLOAT32_ABSOLUTE_TOLERANCE_M,
        )
        for field, expected in expected_scalar_fields.items()
    ):
        fail("particle_system_offset_mismatch")
    if not isinstance(particle_system, dict) or any(
        particle_system.get(field) is not True
        for field in ("ccd_enabled", "smoothing_enabled", "anisotropy_enabled")
    ):
        fail("particle_system_treatment_mismatch")
    elif (
        particle_system.get("smoothing_strength") != 1.0
        or particle_system.get("anisotropy_scale") != 1.0
    ):
        fail("particle_system_treatment_mismatch")
    pbd = particle_system.get("pbd_material_readback") if isinstance(particle_system, dict) else None
    expected_pbd = {
        "density_kg_m3": 1000.0,
        "friction": 0.1,
        "cohesion": 0.01,
        "damping": 0.0,
        "surface_tension": 0.0074,
        "viscosity": 1.7e-6,
        "vorticity_confinement": 0.0,
    }
    if not isinstance(pbd, dict) or any(
        not isinstance(pbd.get(field), (int, float))
        or isinstance(pbd.get(field), bool)
        or not math.isclose(
            float(pbd[field]),
            expected,
            rel_tol=0.0,
            abs_tol=PHYSX_FLOAT32_MATERIAL_ABSOLUTE_TOLERANCE,
        )
        for field, expected in expected_pbd.items()
    ):
        fail("pbd_material_readback_mismatch")
    mass = particle_system.get("mass_density_readback") if isinstance(particle_system, dict) else None
    if (
        not isinstance(mass, dict)
        or mass.get("particle_mass_argument_kg")
        != expected_particle_mass_argument_kg
        or mass.get("density_argument_kg_m3") != PARTICLE_DENSITY_ARGUMENT_KG_M3
        or not all(
            isinstance(mass.get(field), (int, float))
            and not isinstance(mass.get(field), bool)
            and math.isfinite(float(mass[field]))
            for field in ("total_mass_attr_kg", "density_attr_kg_m3")
        )
    ):
        fail("mass_density_readback_mismatch")

    point_instancer = receipt.get("point_instancer")
    if (
        not isinstance(point_instancer, dict)
        or point_instancer.get("one_shared_point_instancer") is not True
        or point_instancer.get("simulation_points_authored") is not True
        or point_instancer.get("prototype_count") != 15
        or point_instancer.get("max_particles") != expected_max_particles
    ):
        fail("point_instancer_contract_mismatch")

    colliders = receipt.get("colliders")
    floor_spec = build_probe_plan()["collider_specs"][0]
    expected_floor_top = float(floor_spec["center_m"][2]) + float(
        floor_spec["size_m"][2]
    ) / 2.0
    if (
        not isinstance(colliders, dict)
        or colliders.get("specs") != build_probe_plan()["collider_specs"]
        or colliders.get("floor_path") != floor_spec["path"]
        or not isinstance(colliders.get("floor_top_z_m"), (int, float))
        or not math.isclose(
            float(colliders["floor_top_z_m"]),
            expected_floor_top,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or colliders.get("all_invisible") is not True
        or colliders.get("visible_inner_bottom_z_m") != expected_floor_top
        or colliders.get("floor_matches_visible_inner_bottom") is not True
    ):
        fail("collider_contract_mismatch")

    timing = receipt.get("timing")
    if (
        not isinstance(timing, dict)
        or timing.get("physics_hz") != PHYSICS_HZ
        or timing.get("executed_physics_steps") != CAPTURE_STEPS[-1]
        or timing.get("callbacks_observed") != CAPTURE_STEPS[-1]
        or any(
            timing.get(field) is not True
            for field in (
                "callback_indices_exact",
                "callback_dts_exact",
                "callback_times_exact",
                "callback_app_update_indices_exact",
            )
        )
        or timing.get("timeline_looping") is not False
        or not _valid_sha256(timing.get("trace_sha256"))
    ):
        fail("callback_trace_mismatch")
    declared_artifacts: list[tuple[str, str]] = []
    if isinstance(timing, dict):
        trace_path_value = timing.get("trace_path")
        trace_sha = timing.get("trace_sha256")
        if isinstance(trace_path_value, str) and _valid_sha256(trace_sha):
            trace_path = artifact_root / trace_path_value
            try:
                resolved_trace = trace_path.resolve(strict=True)
                resolved_trace.relative_to(artifact_root.resolve(strict=True))
                trace = json.loads(resolved_trace.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                fail("callback_trace_artifact_invalid")
            else:
                if _sha256(resolved_trace) != trace_sha:
                    fail("callback_trace_artifact_invalid")
                if (
                    not isinstance(trace, list)
                    or len(trace) != CAPTURE_STEPS[-1]
                    or any(
                        not isinstance(row, dict)
                        or row.get("callback_index") != index
                        or row.get("app_update_index") != index
                        or not isinstance(row.get("dt_s"), (int, float))
                        or not math.isclose(
                            float(row["dt_s"]),
                            1.0 / PHYSICS_HZ,
                            rel_tol=0.0,
                            abs_tol=1.0e-6,
                        )
                        or not isinstance(row.get("timeline_time_s"), (int, float))
                        or not math.isclose(
                            float(row["timeline_time_s"]),
                            index / PHYSICS_HZ,
                            rel_tol=0.0,
                            abs_tol=1.0e-6,
                        )
                        for index, row in enumerate(trace, start=1)
                    )
                ):
                    fail("callback_trace_artifact_invalid")
                declared_artifacts.append((trace_path_value, str(trace_sha)))
        else:
            fail("callback_trace_artifact_invalid")

    emission = receipt.get("emission")
    if (
        not isinstance(emission, dict)
        or emission.get("seed") != SEED
        or emission.get("emitted_particle_count") != EXPECTED_PARTICLE_COUNTS[CAPTURE_STEPS[-1]]
        or emission.get("event_count") != 219
    ):
        fail("emission_contract_mismatch")

    stages = receipt.get("stages")
    if not isinstance(stages, list) or len(stages) != len(CAPTURE_STEPS):
        fail("stage_count_mismatch")
        stages = []
    for stage_index, (step, expected_count) in enumerate(
        zip(CAPTURE_STEPS, EXPECTED_PARTICLE_COUNTS.values())
    ):
        if stage_index >= len(stages) or not isinstance(stages[stage_index], dict):
            fail("stage_count_mismatch")
            continue
        stage = stages[stage_index]
        if (
            stage.get("physics_step") != step
            or stage.get("expected_particle_count") != expected_count
            or stage.get("emitted_particle_count") != expected_count
        ):
            fail("particle_count_mismatch")
        if (
            stage.get("inside_vial_interior_count") != expected_count
            or stage.get("outside_vial_interior_count") != 0
            or stage.get("escaped_particle_count") != 0
        ):
            fail("containment_mismatch")
        if stage.get("ids_exact") is not True:
            fail("particle_identity_mismatch")
        arrays = stage.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_NAMES):
            fail("stage_arrays_incomplete")
            continue
        loaded: dict[str, np.ndarray] = {}
        for name in ARRAY_NAMES:
            record = arrays.get(name)
            if not isinstance(record, dict):
                fail("stage_arrays_incomplete")
                continue
            path_value = record.get("path")
            if not isinstance(path_value, str):
                fail("artifact_missing_or_unreadable")
                continue
            path = artifact_root / path_value
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(artifact_root.resolve(strict=True))
                values = np.load(resolved, allow_pickle=False)
            except (OSError, ValueError):
                fail("artifact_missing_or_unreadable")
                continue
            if not _valid_sha256(record.get("sha256")) or _sha256(resolved) != record.get(
                "sha256"
            ):
                fail("artifact_hash_mismatch")
            else:
                declared_artifacts.append((path_value, str(record["sha256"])))
            if record.get("shape") != list(values.shape) or record.get("dtype") != str(
                values.dtype
            ):
                fail("artifact_metadata_mismatch")
            if values.dtype.kind in "fc" and not bool(np.all(np.isfinite(values))):
                fail("array_nonfinite")
            if record.get("finite") is not True:
                fail("array_nonfinite")
            loaded[name] = values
        if set(loaded) == set(ARRAY_NAMES):
            vector_names = (
                "raw_simulation_positions",
                "smoothed_instancer_positions",
                "velocities",
                "scales",
            )
            if any(loaded[name].shape != (expected_count, 3) for name in vector_names):
                fail("array_count_or_shape_mismatch")
            if loaded["compositions"].shape != (
                expected_count,
                len(COMPOSITION_COMPONENTS),
            ):
                fail("array_count_or_shape_mismatch")
            if any(loaded[name].shape != (expected_count,) for name in ("ids", "proto_indices")):
                fail("array_count_or_shape_mismatch")
            if loaded["ids"].dtype.kind not in "iu" or not np.array_equal(
                loaded["ids"], np.arange(expected_count, dtype=loaded["ids"].dtype)
            ):
                fail("particle_identity_mismatch")
            if loaded["proto_indices"].dtype.kind not in "iu":
                fail("proto_indices_invalid")
            compositions = loaded["compositions"]
            if (
                compositions.shape == (expected_count, len(COMPOSITION_COMPONENTS))
                and (
                    np.any(compositions < -1e-12)
                    or not np.allclose(
                        compositions.sum(axis=1), 1.0, rtol=0.0, atol=1e-10
                    )
                )
            ):
                fail("composition_invariant_mismatch")
        diagnostics = stage.get("diagnostics")
        if not isinstance(diagnostics, dict) or set(diagnostics) != {
            "raw_simulation_positions",
            "smoothed_instancer_positions",
        }:
            fail("offset_aware_diagnostic_missing")
        else:
            for diagnostic in diagnostics.values():
                if (
                    not isinstance(diagnostic, dict)
                    or diagnostic.get("physical_radius_interpretation") is not False
                    or any(
                        not isinstance(diagnostic.get(field), (int, float))
                        or isinstance(diagnostic.get(field), bool)
                        or not math.isfinite(float(diagnostic[field]))
                        for field in (
                            "q05_z_m",
                            "q95_z_m",
                            "center_support_width_m",
                            "wall_to_center_support_m",
                            "visual_prototype_support_m",
                        )
                    )
                ):
                    fail("offset_aware_diagnostic_missing")

    artifacts = receipt.get("artifacts")
    manifest_records: list[object] | None = None
    if isinstance(artifacts, dict):
        manifest_path_value = artifacts.get("manifest_path")
        manifest_sha = artifacts.get("manifest_sha256")
        if isinstance(manifest_path_value, str) and _valid_sha256(manifest_sha):
            try:
                manifest_path = (artifact_root / manifest_path_value).resolve(strict=True)
                manifest_path.relative_to(artifact_root.resolve(strict=True))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_records = manifest.get("records") if isinstance(manifest, dict) else None
                if _sha256(manifest_path) != manifest_sha:
                    manifest_records = None
            except (OSError, ValueError, json.JSONDecodeError):
                manifest_records = None
    if (
        not isinstance(artifacts, dict)
        or not isinstance(manifest_records, list)
        or artifacts.get("record_count") != len(manifest_records)
    ):
        fail("manifest_missing_or_invalid")
    else:
        manifest_pairs = {
            (record.get("path"), record.get("sha256"))
            for record in manifest_records
            if isinstance(record, dict)
        }
        if any(pair not in manifest_pairs for pair in declared_artifacts):
            fail("manifest_missing_or_invalid")
    return tuple(failures)

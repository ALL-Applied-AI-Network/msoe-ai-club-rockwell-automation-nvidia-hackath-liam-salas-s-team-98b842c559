#!/usr/bin/env python3
"""Bounded Isaac 6.2 physics-only stepping capability probe.

The runtime implementation intentionally imports Isaac/Kit only inside ``main``.
CPU tests cover the frozen plan and fail-closed receipt contract; they cannot
establish runtime PointInstancer growth, GPU PBD motion, or renderer alignment.

NVIDIA's Isaac Sim 6.2 SDG workflow documents physics-only stepping followed by
an explicit same-state Replicator capture:
https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fluidmix.physx_three_pour import (  # noqa: E402
    CAMERA_CLIPPING_RANGE,
    CAMERA_PATH,
    NOMINAL_TANK_TRANSMISSION_COLORS,
    SOURCE_LIQUID_PATH,
    SOURCE_STREAM_PATH,
    TWIN_REFERENCE_TIME_CODE,
    build_twin_collider_specs,
)
from fluidmix.render_capture import exact_semantic_class_mask  # noqa: E402


PHYSICS_HZ = 120
MANUAL_STEPS = 120
EMISSION_STEPS = tuple(range(1, 110, 12))
PARTICLES_PER_EMISSION = 4
RT_SUBFRAMES = 64
SEED = 20261025
WIDTH = 640
HEIGHT = 480
PROBE_NAMESPACE = "/FillStation/FluidmixManualStepProbe"
REPLICATOR_ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
REPLICATOR_EXTRA_ARGS = (f"--{REPLICATOR_ASYNC_SETTING}=false",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Isaac 6.2 physics-only stepping with one explicit FillCam capture"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    return parser


def build_probe_plan(*, seed: int) -> dict[str, object]:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed != SEED:
        raise ValueError(f"seed must equal the frozen probe seed {SEED}")
    return {
        "schema_version": 1,
        "seed": seed,
        "physics_hz": PHYSICS_HZ,
        "physics_dt_s": 1.0 / PHYSICS_HZ,
        "manual_steps": MANUAL_STEPS,
        "emission_steps": list(EMISSION_STEPS),
        "particles_per_emission": PARTICLES_PER_EMISSION,
        "final_particle_count": len(EMISSION_STEPS) * PARTICLES_PER_EMISSION,
        "capture_step": MANUAL_STEPS,
        "rt_subframes": RT_SUBFRAMES,
        "update_fabric": False,
        "render_updates_between_captures": False,
        "capture": {
            "delta_time_s": 0.0,
            "wait_for_render": True,
            "reset_renderer_accumulation": True,
            "render_product_updates_enabled_only_for_capture": True,
        },
        "documentation": [
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html",
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_getting_started.html",
            "https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html",
        ],
    }


def validate_probe_receipt(receipt: object) -> tuple[str, ...]:
    """Return stable failure reasons for a completed probe receipt."""

    if not isinstance(receipt, dict):
        return ("receipt_not_mapping",)
    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    plan = receipt.get("plan")
    if not isinstance(plan, dict):
        fail("plan_missing_or_invalid")
        expected_plan = None
    else:
        try:
            expected_plan = build_probe_plan(seed=plan.get("seed"))
        except (TypeError, ValueError):
            expected_plan = None
        if plan != expected_plan:
            fail("plan_mismatch")
    if receipt.get("schema_version") != 1:
        fail("schema_version_mismatch")
    if receipt.get("experiment") != "physx_manual_step_render_decoupling_probe":
        fail("experiment_mismatch")
    if receipt.get("status") != "pass":
        fail("status_not_pass")
    if receipt.get("errors") != []:
        fail("errors_present")

    source = receipt.get("source")
    if not isinstance(source, dict):
        fail("source_layers_changed")
        fail("source_layer_evidence_invalid")
    else:
        layers_before = source.get("layers_before")
        layers_after = source.get("layers_after")
        if (
            not _valid_sha256(source.get("stage_sha256"))
            or not _valid_source_layer_records(layers_before)
            or not _valid_source_layer_records(layers_after)
        ):
            fail("source_layer_evidence_invalid")
        if (
            source.get("source_layers_unchanged") is not True
            or layers_before != layers_after
        ):
            fail("source_layers_changed")

    scene = receipt.get("scene")
    clipping_range = scene.get("clipping_range") if isinstance(scene, dict) else None
    annotation_override = (
        scene.get("transparent_annotation_override")
        if isinstance(scene, dict)
        else None
    )
    if (
        not isinstance(scene, dict)
        or scene.get("default_prim") != "/FillStation"
        or not _finite_number(scene.get("meters_per_unit"))
        or not math.isclose(
            float(scene.get("meters_per_unit", math.nan)),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or scene.get("up_axis") != "Z"
        or scene.get("camera_path") != CAMERA_PATH
        or not isinstance(clipping_range, list)
        or len(clipping_range) != 2
        or any(
            not _finite_number(value)
            or not math.isclose(
                float(value), expected, rel_tol=0.0, abs_tol=1.0e-9
            )
            for value, expected in zip(clipping_range, CAMERA_CLIPPING_RANGE)
        )
        or scene.get("source_liquid_hidden") is not True
        or scene.get("source_stream_hidden") is not True
        or not isinstance(annotation_override, dict)
        or annotation_override.get("session_layer_only") is not True
    ):
        fail("scene_contract_mismatch")

    runtime = receipt.get("runtime")
    writeback = (
        runtime.get("usd_writeback_settings")
        if isinstance(runtime, dict)
        else None
    )
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("isaac_sim"), str)
        or not runtime["isaac_sim"].startswith("6.2.0")
        or not isinstance(runtime.get("kit"), str)
        or not runtime["kit"].startswith("110.4")
        or not isinstance(runtime.get("physx"), str)
        or not runtime["physx"].startswith("110.4")
        or runtime.get("simulation_manager_method") != "SimulationManager.step"
        or runtime.get("update_fabric") is not False
        or runtime.get("gpu_dynamics") is not True
        or runtime.get("cuda_device_available") is not True
        or not isinstance(writeback, dict)
        or any(
            writeback.get(key) is not True
            for key in (
                "update_to_usd",
                "update_particles_to_usd",
                "update_velocities_to_usd",
            )
        )
    ):
        fail("runtime_contract_mismatch")

    initialization = receipt.get("physics_initialization")
    if (
        not isinstance(initialization, dict)
        or initialization.get("initialized_with_empty_particle_set") is not True
        or initialization.get("callbacks_registered_after_initialization") is not True
        or type(initialization.get("baseline_num_physics_steps")) is not int
        or not _finite_number(initialization.get("baseline_simulation_time_s"))
    ):
        fail("physics_initialization_mismatch")

    physics = receipt.get("physics")
    if not isinstance(physics, dict):
        fail("physics_evidence_missing")
    else:
        if physics.get("manual_step_calls") != MANUAL_STEPS:
            fail("manual_step_count_mismatch")
        if physics.get("callbacks_observed") != MANUAL_STEPS:
            fail("callback_count_mismatch")
        if physics.get("callback_indices") != list(range(1, MANUAL_STEPS + 1)):
            fail("callback_indices_mismatch")
        callback_dts = physics.get("callback_dt_s")
        if (
            not isinstance(callback_dts, list)
            or len(callback_dts) != MANUAL_STEPS
            or any(
                not _finite_number(value)
                or not math.isclose(
                    float(value), 1.0 / PHYSICS_HZ, rel_tol=0.0, abs_tol=1.0e-9
                )
                for value in callback_dts
            )
        ):
            fail("callback_dt_mismatch")
        if physics.get("num_physics_steps_delta") != MANUAL_STEPS:
            fail("physics_step_delta_mismatch")
        if not _finite_number(physics.get("simulation_time_delta_s")) or not math.isclose(
            float(physics.get("simulation_time_delta_s", math.nan)),
            MANUAL_STEPS / PHYSICS_HZ,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            fail("simulation_time_delta_mismatch")
        if (
            not _finite_number(physics.get("manual_loop_seconds"))
            or float(physics.get("manual_loop_seconds", 0.0)) <= 0.0
        ):
            fail("timing_evidence_missing")

    emission = receipt.get("emission")
    if not isinstance(emission, dict):
        fail("emission_evidence_missing")
    else:
        if (
            emission.get("one_shared_point_instancer") is not True
            or emission.get("point_instancer_path")
            != f"{PROBE_NAMESPACE}/Particles/Shared"
            or emission.get("max_particles")
            != len(EMISSION_STEPS) * PARTICLES_PER_EMISSION
        ):
            fail("shared_point_instancer_missing")
        if (
            emission.get("append_event_steps") != list(EMISSION_STEPS)
            or emission.get("append_event_counts")
            != [PARTICLES_PER_EMISSION] * len(EMISSION_STEPS)
        ):
            fail("emission_schedule_mismatch")
        if (
            emission.get("final_particle_count")
            != len(EMISSION_STEPS) * PARTICLES_PER_EMISSION
            or emission.get("final_particle_ids")
            != list(range(len(EMISSION_STEPS) * PARTICLES_PER_EMISSION))
            or emission.get("stable_ids_each_step") is not True
            or emission.get("equal_row_counts_each_step") is not True
            or emission.get("particle_identity_checks") != MANUAL_STEPS
            or emission.get("row_count_checks") != MANUAL_STEPS
        ):
            fail("particle_identity_mismatch")

    motion = receipt.get("motion")
    if (
        not isinstance(motion, dict)
        or motion.get("finite_positions") is not True
        or motion.get("finite_simulation_points") is not True
        or motion.get("finite_velocities") is not True
        or motion.get("finite_state_checks") != MANUAL_STEPS
        or not _finite_number(motion.get("maximum_displacement_m"))
        or float(motion.get("maximum_displacement_m", 0.0)) <= 0.0
    ):
        fail("particle_motion_missing")

    render = receipt.get("render")
    if not isinstance(render, dict):
        fail("render_evidence_missing")
    else:
        if (
            render.get("render_updates_disabled_between_captures") is not True
            or render.get("renderer_accumulation_reset") is not True
            or render.get("capture_delta_time_s") != 0.0
            or render.get("capture_wait_for_render") is not True
            or render.get("capture_rt_subframes") != RT_SUBFRAMES
            or render.get("capture_step") != MANUAL_STEPS
            or render.get("explicit_capture_count") != 1
        ):
            fail("render_not_decoupled")
        if (
            not _finite_number(render.get("capture_seconds"))
            or float(render.get("capture_seconds", 0.0)) <= 0.0
        ):
            fail("timing_evidence_missing")
        if (
            render.get("rgb_readable") is not True
            or render.get("semantic_readable") is not True
            or render.get("liquid_mask_readable") is not True
            or render.get("raster_shape") != [480, 640]
        ):
            fail("capture_artifact_unreadable")
        if type(render.get("liquid_pixel_count")) is not int or render.get(
            "liquid_pixel_count", 0
        ) <= 0:
            fail("semantic_liquid_missing")
        if not _valid_semantic_evidence(render):
            fail("semantic_mapping_missing")

    same_state = receipt.get("same_state_capture")
    if not isinstance(same_state, dict):
        fail("same_state_capture_missing")
    else:
        if (
            same_state.get("callback_count_unchanged") is not True
            or same_state.get("num_physics_steps_unchanged") is not True
            or same_state.get("simulation_time_unchanged") is not True
        ):
            fail("capture_advanced_physics")
        for prefix in ("positions", "velocities", "ids"):
            before = same_state.get(f"{prefix}_sha256_before")
            after = same_state.get(f"{prefix}_sha256_after")
            if not _valid_sha256(before) or before != after:
                fail("capture_changed_particle_state")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or any(
        not _valid_sha256(artifacts.get(key))
        for key in ("rgb_sha256", "semantic_sha256", "liquid_mask_sha256")
    ):
        fail("artifact_hash_missing")
    if not isinstance(artifacts, dict) or any(
        not isinstance(artifacts.get(key), str) or not artifacts[key]
        for key in ("rgb_path", "semantic_path", "liquid_mask_path")
    ):
        fail("artifact_path_missing")
    return tuple(failures)


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_source_layer_records(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for record in value:
        if not isinstance(record, dict):
            return False
        if any(
            not isinstance(record.get(key), str) or not record[key]
            for key in ("identifier", "resolved_path")
        ):
            return False
        if type(record.get("bytes")) is not int or record["bytes"] <= 0:
            return False
        if not _valid_sha256(record.get("sha256")):
            return False
    return True


def _valid_semantic_evidence(render: dict[str, object]) -> bool:
    visible = render.get("semantic_unique_ids")
    liquid = render.get("liquid_semantic_ids")
    mapping = render.get("semantic_id_to_class_labels")
    if (
        not isinstance(visible, list)
        or not visible
        or any(type(value) is not int or value < 0 for value in visible)
        or len(set(visible)) != len(visible)
        or not isinstance(liquid, list)
        or not liquid
        or any(type(value) is not int or value < 0 for value in liquid)
        or len(set(liquid)) != len(liquid)
        or not set(liquid).issubset(visible)
        or not isinstance(mapping, dict)
    ):
        return False
    for semantic_id in visible:
        labels = mapping.get(str(semantic_id))
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            return False
        if semantic_id in liquid and "liquid" not in labels:
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: Any, *, integer: bool = False) -> str:
    import numpy as np

    dtype = "<i8" if integer else "<f8"
    return hashlib.sha256(np.asarray(values, dtype=dtype, order="C").tobytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "physx_manual_step_render_decoupling_probe",
        "status": "insufficient_evidence",
        "plan": build_probe_plan(seed=args.seed),
        "arguments": {
            "stage": str(args.stage.resolve()),
            "out": str(args.out.resolve()),
            "seed": args.seed,
        },
        "documentation": {
            "documented": [
                "SimulationManager.step advances physics without an application/render update.",
                "Replicator SDG workflows disable render-product updates between explicit captures.",
                "A same-state capture uses delta_time=0.0 and wait_for_render=True.",
            ],
            "observed": [],
        },
        "source": {
            "stage_sha256": None,
            "layers_before": [],
            "layers_after": [],
            "source_layers_unchanged": False,
        },
        "runtime": {
            "isaac_sim": "6.2.0",
            "kit": None,
            "simulation_manager_method": "SimulationManager.step",
            "update_fabric": False,
            "gpu_dynamics": False,
            "cuda_device_available": False,
            "usd_writeback_settings": {
                "update_to_usd": False,
                "update_particles_to_usd": False,
                "update_velocities_to_usd": False,
            },
            "failure_phase": "preflight",
        },
        "scene": {
            "default_prim": None,
            "meters_per_unit": None,
            "up_axis": None,
            "camera_path": CAMERA_PATH,
            "clipping_range": None,
            "source_liquid_hidden": False,
            "source_stream_hidden": False,
            "transparent_annotation_override": None,
        },
        "physics_initialization": {
            "initialized_with_empty_particle_set": False,
            "callbacks_registered_after_initialization": False,
            "baseline_num_physics_steps": None,
            "baseline_simulation_time_s": None,
        },
        "physics": {
            "manual_step_calls": 0,
            "callbacks_observed": 0,
            "callback_indices": [],
            "callback_dt_s": [],
            "num_physics_steps_delta": None,
            "simulation_time_delta_s": None,
            "manual_loop_seconds": None,
        },
        "emission": {
            "one_shared_point_instancer": False,
            "point_instancer_path": f"{PROBE_NAMESPACE}/Particles/Shared",
            "max_particles": None,
            "append_event_steps": [],
            "append_event_counts": [],
            "final_particle_count": 0,
            "final_particle_ids": [],
            "stable_ids_each_step": False,
            "equal_row_counts_each_step": False,
            "particle_identity_checks": 0,
            "row_count_checks": 0,
        },
        "motion": {
            "finite_positions": False,
            "finite_simulation_points": False,
            "finite_velocities": False,
            "maximum_displacement_m": None,
            "finite_state_checks": 0,
        },
        "render": {
            "render_updates_disabled_between_captures": False,
            "renderer_accumulation_reset": False,
            "capture_delta_time_s": 0.0,
            "capture_wait_for_render": True,
            "capture_rt_subframes": RT_SUBFRAMES,
            "capture_step": MANUAL_STEPS,
            "capture_seconds": None,
            "explicit_capture_count": 0,
            "rgb_readable": False,
            "semantic_readable": False,
            "liquid_mask_readable": False,
            "raster_shape": [HEIGHT, WIDTH],
            "liquid_pixel_count": 0,
            "semantic_unique_ids": [],
            "liquid_semantic_ids": [],
            "semantic_id_to_class_labels": {},
        },
        "same_state_capture": {
            "callback_count_unchanged": False,
            "num_physics_steps_unchanged": False,
            "simulation_time_unchanged": False,
            "positions_sha256_before": None,
            "positions_sha256_after": None,
            "velocities_sha256_before": None,
            "velocities_sha256_after": None,
            "ids_sha256_before": None,
            "ids_sha256_after": None,
        },
        "artifacts": {
            "rgb_path": None,
            "rgb_sha256": None,
            "semantic_path": None,
            "semantic_sha256": None,
            "liquid_mask_path": None,
            "liquid_mask_sha256": None,
        },
        "validation_failures": [],
        "errors": [],
        "limitations": {
            "capability_probe_only": True,
            "continuous_surface": False,
            "palette_calibrated": False,
            "physical_fluid_validated": False,
            "sim_to_real_validated": False,
        },
    }


def _state_snapshot(instancer: Any, point_set_api: Any) -> dict[str, Any]:
    import numpy as np

    positions = np.asarray(
        [list(value) for value in (instancer.GetPositionsAttr().Get() or [])],
        dtype=np.float64,
    ).reshape((-1, 3))
    simulation_positions = np.asarray(
        [list(value) for value in (point_set_api.GetSimulationPointsAttr().Get() or [])],
        dtype=np.float64,
    ).reshape((-1, 3))
    velocities = np.asarray(
        [list(value) for value in (instancer.GetVelocitiesAttr().Get() or [])],
        dtype=np.float64,
    ).reshape((-1, 3))
    ids = [int(value) for value in (instancer.GetIdsAttr().Get() or [])]
    return {
        "positions": positions,
        "simulation_positions": simulation_positions,
        "velocities": velocities,
        "ids": ids,
        "positions_sha256": _array_sha256(positions),
        "velocities_sha256": _array_sha256(velocities),
        "ids_sha256": _array_sha256(ids, integer=True),
    }


def _append_particles(
    *,
    stage: Any,
    session_layer: Any,
    instancer: Any,
    point_set_api: Any,
    positions: list[Any],
    particle_ids: list[int],
    Usd: Any,
    Vt: Any,
    Gf: Any,
) -> None:
    with Usd.EditContext(stage, session_layer):
        old_positions = list(instancer.GetPositionsAttr().Get() or [])
        old_simulation = list(point_set_api.GetSimulationPointsAttr().Get() or [])
        old_velocities = list(instancer.GetVelocitiesAttr().Get() or [])
        old_orientations = list(instancer.GetOrientationsAttr().Get() or [])
        old_scales = list(instancer.GetScalesAttr().Get() or [])
        old_ids = [int(value) for value in (instancer.GetIdsAttr().Get() or [])]
        old_prototypes = [
            int(value) for value in (instancer.GetProtoIndicesAttr().Get() or [])
        ]
        row_counts = {
            len(old_positions),
            len(old_simulation),
            len(old_velocities),
            len(old_orientations),
            len(old_scales),
            len(old_ids),
            len(old_prototypes),
        }
        if len(row_counts) != 1 or particle_ids != list(
            range(len(old_ids), len(old_ids) + len(positions))
        ):
            raise RuntimeError("PointInstancer append lost stable row identity")
        instancer.GetPositionsAttr().Set(Vt.Vec3fArray([*old_positions, *positions]))
        point_set_api.GetSimulationPointsAttr().Set(
            Vt.Vec3fArray([*old_simulation, *positions])
        )
        instancer.GetVelocitiesAttr().Set(
            Vt.Vec3fArray(
                [*old_velocities, *[Gf.Vec3f(0.0, 0.0, -0.1)] * len(positions)]
            )
        )
        instancer.GetOrientationsAttr().Set(
            Vt.QuathArray(
                [
                    *old_orientations,
                    *[Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(positions),
                ]
            )
        )
        instancer.GetScalesAttr().Set(
            Vt.Vec3fArray(
                [*old_scales, *[Gf.Vec3f(1.0, 1.0, 1.0)] * len(positions)]
            )
        )
        instancer.GetIdsAttr().Set(Vt.Int64Array([*old_ids, *particle_ids]))
        instancer.GetProtoIndicesAttr().Set(
            Vt.IntArray([*old_prototypes, *[0] * len(positions)])
        )


def _annotator_array_and_mapping(raw: Any) -> tuple[Any, object]:
    import numpy as np

    if not isinstance(raw, dict) or "data" not in raw or not isinstance(
        raw.get("info"), dict
    ):
        raise RuntimeError("semantic annotator returned a malformed envelope")
    array = np.asarray(raw["data"])
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    mapping = raw["info"].get("idToLabels")
    if not isinstance(mapping, dict):
        raise RuntimeError("semantic annotator mapping is missing")
    return array, mapping


def run_probe(args: argparse.Namespace, app: Any, receipt: dict[str, Any]) -> None:
    """Execute one fresh-process capability probe; mutate ``receipt`` in place."""

    import carb
    import cv2
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
    from omni.physx.bindings import _physx as physx_bindings
    from omni.physx.scripts import particleUtils
    from omni.physxfoundation import get_physx_foundation_interface
    from pxr import (
        Gf,
        PhysxSchema,
        Sdf,
        Semantics as LegacySemantics,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdSemantics,
        UsdShade,
        Vt,
    )

    import physx_three_pour_sequence as f2

    output = args.out
    settings = carb.settings.get_settings()
    receipt["runtime"]["failure_phase"] = "configure_runtime"
    if settings.get_as_bool(REPLICATOR_ASYNC_SETTING) is not False:
        raise RuntimeError("Replicator asynchronous throttling is enabled")
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
    receipt["runtime"]["usd_writeback_settings"] = {
        "update_to_usd": settings.get_as_bool(
            physx_bindings.SETTING_UPDATE_TO_USD
        ),
        "update_particles_to_usd": settings.get_as_bool(
            physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD
        ),
        "update_velocities_to_usd": settings.get_as_bool(
            physx_bindings.SETTING_UPDATE_VELOCITIES_TO_USD
        ),
    }
    if not all(receipt["runtime"]["usd_writeback_settings"].values()):
        raise RuntimeError("PhysX USD position/particle/velocity writeback did not enable")
    settings.set("/rtx/pathtracing/maxBounces", 16)
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
    versions = f2._version_info(omni.kit.app.get_app())
    receipt["runtime"].update(
        {
            "isaac_sim": versions.get("isaac"),
            "kit": versions.get("kit"),
            "physx": versions.get("physx"),
            "cuda_device_available": bool(
                get_physx_foundation_interface().cuda_device_check()
            ),
        }
    )
    if receipt["runtime"]["cuda_device_available"] is not True:
        raise RuntimeError("GPU particle probe requires a CUDA device")

    context = omni.usd.get_context()
    receipt["runtime"]["failure_phase"] = "open_twin"
    if not context.open_stage(os.fspath(args.stage)):
        raise RuntimeError(f"could not open twin stage: {args.stage}")
    while context.get_stage_loading_status()[2] > 0:
        app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("opened twin stage is unavailable")
    before_layers = f2._source_layer_records(stage)
    receipt["source"].update(
        {
            "stage_sha256": _sha256(args.stage),
            "layers_before": before_layers,
        }
    )
    default_prim = stage.GetDefaultPrim()
    receipt["scene"].update(
        {
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        }
    )
    if (
        receipt["scene"]["default_prim"] != "/FillStation"
        or receipt["scene"]["meters_per_unit"] != 1.0
        or receipt["scene"]["up_axis"] != "Z"
    ):
        raise RuntimeError("stage is not the reviewed metre-scale Z-up FillStation twin")
    time_codes_per_second = float(stage.GetTimeCodesPerSecond())
    if time_codes_per_second <= 0.0:
        raise RuntimeError("twin timeCodesPerSecond is invalid")
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_current_time(TWIN_REFERENCE_TIME_CODE / time_codes_per_second)
    app.update()
    if timeline.is_playing():
        raise RuntimeError("manual-step probe requires a stopped timeline")
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError("reviewed FillCam is missing")
    clip_value = UsdGeom.Camera(camera_prim).GetClippingRangeAttr().Get(
        Usd.TimeCode(TWIN_REFERENCE_TIME_CODE)
    )
    receipt["scene"]["clipping_range"] = [
        float(clip_value[0]),
        float(clip_value[1]),
    ]
    if any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-9)
        for value, expected in zip(
            receipt["scene"]["clipping_range"], CAMERA_CLIPPING_RANGE
        )
    ):
        raise RuntimeError("FillCam clipping range differs from the twin contract")

    receipt["runtime"]["failure_phase"] = "author_probe_graph"
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        for path, key in (
            (SOURCE_LIQUID_PATH, "source_liquid_hidden"),
            (SOURCE_STREAM_PATH, "source_stream_hidden"),
        ):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"source liquid prim missing: {path}")
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.invisible
            )
            receipt["scene"][key] = True
        UsdGeom.Xform.Define(stage, PROBE_NAMESPACE)
        scene = UsdPhysics.Scene.Define(stage, f"{PROBE_NAMESPACE}/PhysicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        scene_api.CreateSolverTypeAttr().Set("TGS")
        scene_api.GetTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
        receipt["runtime"]["gpu_dynamics"] = bool(
            scene_api.GetEnableGPUDynamicsAttr().Get()
        )
        if receipt["runtime"]["gpu_dynamics"] is not True:
            raise RuntimeError("GPU dynamics did not enable on the probe physics scene")

        system_path = Sdf.Path(f"{PROBE_NAMESPACE}/ParticleSystem")
        system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
        system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
        system.CreateParticleSystemEnabledAttr().Set(True)
        system.CreateEnableCCDAttr().Set(True)
        system.CreateParticleContactOffsetAttr().Set(0.003)
        system.CreateFluidRestOffsetAttr().Set(0.0015)
        system.CreateContactOffsetAttr().Set(0.003)
        system.CreateRestOffsetAttr().Set(0.0029)
        system.CreateSolidRestOffsetAttr().Set(0.0029)
        system.CreateMaxVelocityAttr().Set(2.5)
        particleUtils.add_physx_particle_smoothing(
            stage, system_path, enabled=True, strength=1.0
        )
        particleUtils.add_physx_particle_anisotropy(
            stage, system_path, enabled=True, scale=1.0
        )

        physics_material = UsdShade.Material.Define(
            stage, f"{PROBE_NAMESPACE}/PBDMaterial"
        )
        pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(physics_material.GetPrim())
        pbd.CreateDensityAttr().Set(1000.0)
        pbd.CreateFrictionAttr().Set(0.1)
        UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(
            physics_material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )

        material = UsdShade.Material.Define(
            stage, f"{PROBE_NAMESPACE}/T200LiquidMaterial"
        )
        f2._author_omnisurface(
            stage,
            material,
            NOMINAL_TANK_TRANSMISSION_COLORS["t200"],
            Sdf,
            Gf,
            UsdShade,
        )
        instancer_path = Sdf.Path(f"{PROBE_NAMESPACE}/Particles/Shared")
        instancer_prim = particleUtils.add_physx_particleset_pointinstancer(
            stage,
            instancer_path,
            [],
            [],
            system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.001,
            density=1000.0,
            num_prototypes=0,
        )
        instancer = UsdGeom.PointInstancer(instancer_prim)
        point_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(instancer_prim)
        max_particles_attr = instancer_prim.CreateAttribute(
            "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
        )
        max_particles_attr.Set(len(EMISSION_STEPS) * PARTICLES_PER_EMISSION)
        receipt["emission"]["max_particles"] = int(max_particles_attr.Get())
        point_set_api.GetSimulationPointsAttr().Set(Vt.Vec3fArray([]))
        instancer.CreateIdsAttr().Set(Vt.Int64Array([]))
        instancer.GetProtoIndicesAttr().Set(Vt.IntArray([]))
        prototype = UsdGeom.Sphere.Define(
            stage, instancer_path.AppendChild("particlePrototype0")
        )
        prototype.CreateRadiusAttr().Set(0.0015)
        UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(material)
        instancer.GetPrototypesRel().AddTarget(prototype.GetPath())
        f2._author_dual_liquid_semantics(
            prototype.GetPrim(), UsdSemantics, LegacySemantics
        )
        f2._author_dual_liquid_semantics(
            instancer.GetPrim(), UsdSemantics, LegacySemantics
        )
        for spec in build_twin_collider_specs():
            collider_path = str(spec["path"]).replace(
                "/FillStation/FluidmixF2/Colliders",
                f"{PROBE_NAMESPACE}/Colliders",
            )
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(collider_path))
            cube.CreateSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
            cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
            cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    probe_instancers = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(PROBE_NAMESPACE)
        and prim.IsA(UsdGeom.PointInstancer)
    ]
    receipt["emission"]["one_shared_point_instancer"] = (
        len(probe_instancers) == 1
    )
    if receipt["emission"]["one_shared_point_instancer"] is not True:
        raise RuntimeError("probe did not author exactly one PointInstancer")
    for source_path, key in (
        (SOURCE_LIQUID_PATH, "source_liquid_hidden"),
        (SOURCE_STREAM_PATH, "source_stream_hidden"),
    ):
        hidden = (
            UsdGeom.Imageable(stage.GetPrimAtPath(source_path)).ComputeVisibility()
            == UsdGeom.Tokens.invisible
        )
        receipt["scene"][key] = hidden
        if not hidden:
            raise RuntimeError(f"source prim remains visible: {source_path}")
    annotation_override = f2._author_transparent_vial_annotation_override(
        stage, session_layer, Usd, Sdf
    )
    annotation_override["authored_after_untouched_reference_capture"] = False
    receipt["scene"]["transparent_annotation_override"] = annotation_override

    receipt["runtime"]["failure_phase"] = "attach_render_graph"
    rep.set_global_seed(args.seed)
    render_product = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    semantic_annotator = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation",
        init_params={"colorize": False, "semanticTypes": ["class"]},
    )
    rgb_annotator.attach([render_product])
    semantic_annotator.attach([render_product])
    rep.orchestrator.set_capture_on_play(False)
    for _ in range(60):
        app.update()
    render_product.hydra_texture.set_updates_enabled(False)
    receipt["render"]["render_updates_disabled_between_captures"] = True

    receipt["runtime"]["failure_phase"] = "initialize_manual_physics"
    if list(point_set_api.GetSimulationPointsAttr().Get() or []):
        raise RuntimeError("particle set must be empty before physics initialization")
    SimulationManager.initialize_physics()
    baseline_steps = int(SimulationManager.get_num_physics_steps())
    baseline_time = float(SimulationManager.get_simulation_time())
    receipt["physics_initialization"].update(
        {
            "initialized_with_empty_particle_set": True,
            "baseline_num_physics_steps": baseline_steps,
            "baseline_simulation_time_s": baseline_time,
        }
    )
    callbacks: list[dict[str, float | int]] = []

    def on_post_step(step_dt: float, _context: Any) -> None:
        callbacks.append(
            {"callback_index": len(callbacks) + 1, "dt_s": float(step_dt)}
        )

    SimulationManager.register_callback(
        on_post_step, event=SimulationEvent.PHYSICS_POST_STEP
    )
    receipt["physics_initialization"][
        "callbacks_registered_after_initialization"
    ] = True

    receipt["runtime"]["failure_phase"] = "manual_physics_loop"
    rng = random.Random(args.seed)
    expected_ids: list[int] = []
    initial_positions_by_id: dict[int, list[float]] = {}
    particle_identity_checks = 0
    row_count_checks = 0
    finite_state_checks = 0
    loop_started = time.monotonic()
    for step in range(1, MANUAL_STEPS + 1):
        if step in EMISSION_STEPS:
            ids = list(
                range(
                    len(expected_ids),
                    len(expected_ids) + PARTICLES_PER_EMISSION,
                )
            )
            positions = f2._twin_chunk_positions(
                Gf, PARTICLES_PER_EMISSION, rng
            )
            _append_particles(
                stage=stage,
                session_layer=session_layer,
                instancer=instancer,
                point_set_api=point_set_api,
                positions=positions,
                particle_ids=ids,
                Usd=Usd,
                Vt=Vt,
                Gf=Gf,
            )
            for particle_id, position in zip(ids, positions):
                initial_positions_by_id[particle_id] = [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ]
            expected_ids.extend(ids)
            receipt["emission"]["append_event_steps"].append(step)
            receipt["emission"]["append_event_counts"].append(len(ids))
        callbacks_before = len(callbacks)
        SimulationManager.step(steps=1, update_fabric=False)
        receipt["physics"]["manual_step_calls"] += 1
        if len(callbacks) != callbacks_before + 1:
            raise RuntimeError("manual step did not emit exactly one post-step callback")
        state = _state_snapshot(instancer, point_set_api)
        row_count = len(state["ids"])
        positions_count = len(state["positions"])
        simulation_positions_count = len(state["simulation_positions"])
        velocities_count = len(state["velocities"])
        if state["ids"] != expected_ids:
            raise RuntimeError("manual step changed PointInstancer particle IDs")
        particle_identity_checks += 1
        if len(
            {
                row_count,
                positions_count,
                simulation_positions_count,
                velocities_count,
            }
        ) != 1:
            raise RuntimeError("manual step produced unequal particle row counts")
        row_count_checks += 1
        if not np.all(np.isfinite(state["positions"])) or not np.all(
            np.isfinite(state["simulation_positions"])
        ) or not np.all(np.isfinite(state["velocities"])):
            raise RuntimeError("manual step produced non-finite particle state")
        finite_state_checks += 1
    receipt["physics"]["manual_loop_seconds"] = time.monotonic() - loop_started
    receipt["documentation"]["observed"].append(
        {
            "manual_physics_steps": MANUAL_STEPS,
            "manual_loop_seconds": receipt["physics"]["manual_loop_seconds"],
        }
    )
    receipt["physics"].update(
        {
            "callbacks_observed": len(callbacks),
            "callback_indices": [int(row["callback_index"]) for row in callbacks],
            "callback_dt_s": [float(row["dt_s"]) for row in callbacks],
            "num_physics_steps_delta": int(
                SimulationManager.get_num_physics_steps() - baseline_steps
            ),
            "simulation_time_delta_s": float(
                SimulationManager.get_simulation_time() - baseline_time
            ),
        }
    )
    before_capture = _state_snapshot(instancer, point_set_api)
    receipt["emission"].update(
        {
            "final_particle_count": len(before_capture["ids"]),
            "final_particle_ids": before_capture["ids"],
            "stable_ids_each_step": before_capture["ids"] == expected_ids,
            "equal_row_counts_each_step": len(before_capture["positions"])
            == len(before_capture["simulation_positions"])
            == len(before_capture["velocities"])
            == len(before_capture["ids"]),
            "particle_identity_checks": particle_identity_checks,
            "row_count_checks": row_count_checks,
        }
    )
    initial_rows = np.asarray(
        [initial_positions_by_id[particle_id] for particle_id in before_capture["ids"]],
        dtype=np.float64,
    )
    displacement = np.linalg.norm(before_capture["positions"] - initial_rows, axis=1)
    receipt["motion"].update(
        {
            "finite_positions": bool(np.all(np.isfinite(before_capture["positions"]))),
            "finite_simulation_points": bool(
                np.all(np.isfinite(before_capture["simulation_positions"]))
            ),
            "finite_velocities": bool(np.all(np.isfinite(before_capture["velocities"]))),
            "maximum_displacement_m": float(np.max(displacement)),
            "finite_state_checks": finite_state_checks,
        }
    )

    receipt["runtime"]["failure_phase"] = "same_state_explicit_capture"
    callbacks_before_capture = len(callbacks)
    steps_before_capture = int(SimulationManager.get_num_physics_steps())
    time_before_capture = float(SimulationManager.get_simulation_time())
    render_product.hydra_texture.set_updates_enabled(True)
    context.reset_renderer_accumulation()
    receipt["render"]["renderer_accumulation_reset"] = True
    capture_started = time.monotonic()
    try:
        rep.orchestrator.step(
            delta_time=0.0,
            rt_subframes=RT_SUBFRAMES,
            wait_for_render=True,
        )
        receipt["render"]["explicit_capture_count"] = 1
        rgb = np.asarray(rgb_annotator.get_data())
        semantic, mapping = _annotator_array_and_mapping(
            semantic_annotator.get_data()
        )
    finally:
        render_product.hydra_texture.set_updates_enabled(False)
    receipt["render"]["capture_seconds"] = time.monotonic() - capture_started
    receipt["documentation"]["observed"].append(
        {
            "explicit_capture_count": 1,
            "explicit_capture_seconds": receipt["render"]["capture_seconds"],
        }
    )
    after_capture = _state_snapshot(instancer, point_set_api)
    receipt["same_state_capture"].update(
        {
            "callback_count_unchanged": len(callbacks) == callbacks_before_capture,
            "num_physics_steps_unchanged": int(
                SimulationManager.get_num_physics_steps()
            )
            == steps_before_capture,
            "simulation_time_unchanged": math.isclose(
                float(SimulationManager.get_simulation_time()),
                time_before_capture,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "positions_sha256_before": before_capture["positions_sha256"],
            "positions_sha256_after": after_capture["positions_sha256"],
            "velocities_sha256_before": before_capture["velocities_sha256"],
            "velocities_sha256_after": after_capture["velocities_sha256"],
            "ids_sha256_before": before_capture["ids_sha256"],
            "ids_sha256_after": after_capture["ids_sha256"],
        }
    )

    if rgb.ndim != 3 or rgb.shape[:2] != (HEIGHT, WIDTH) or rgb.shape[2] < 3:
        raise RuntimeError(f"RGB capture shape is invalid: {rgb.shape}")
    rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
    if not np.any(rgb != rgb[0, 0]):
        raise RuntimeError("RGB capture is spatially constant")
    if semantic.shape != (HEIGHT, WIDTH):
        raise RuntimeError(f"semantic capture shape is invalid: {semantic.shape}")
    exact = exact_semantic_class_mask(semantic, mapping, class_name="liquid")
    rgb_path = output / "rgb.png"
    semantic_path = output / "semantic.npy"
    liquid_path = output / "liquid.png"
    if not cv2.imwrite(
        os.fspath(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError("could not save RGB artifact")
    np.save(semantic_path, semantic, allow_pickle=False)
    if not cv2.imwrite(os.fspath(liquid_path), exact.mask):
        raise RuntimeError("could not save liquid-mask artifact")
    reloaded_rgb = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
    reloaded_semantic = np.load(semantic_path, allow_pickle=False)
    reloaded_mask = cv2.imread(os.fspath(liquid_path), cv2.IMREAD_UNCHANGED)
    receipt["render"].update(
        {
            "rgb_readable": reloaded_rgb is not None
            and reloaded_rgb.shape == (HEIGHT, WIDTH, 3),
            "semantic_readable": np.array_equal(reloaded_semantic, semantic),
            "liquid_mask_readable": reloaded_mask is not None
            and np.array_equal(reloaded_mask, exact.mask),
            "liquid_pixel_count": int(exact.pixel_count),
            "semantic_unique_ids": list(exact.visible_semantic_ids),
            "liquid_semantic_ids": list(exact.semantic_ids),
            "semantic_id_to_class_labels": {
                str(semantic_id): list(labels)
                for semantic_id, labels in exact.id_to_labels
            },
        }
    )
    receipt["artifacts"].update(
        {
            "rgb_path": rgb_path.name,
            "rgb_sha256": _sha256(rgb_path),
            "semantic_path": semantic_path.name,
            "semantic_sha256": _sha256(semantic_path),
            "liquid_mask_path": liquid_path.name,
            "liquid_mask_sha256": _sha256(liquid_path),
        }
    )
    after_layers = f2._source_layer_records(stage)
    receipt["source"].update(
        {
            "layers_after": after_layers,
            "source_layers_unchanged": after_layers == before_layers,
        }
    )
    receipt["runtime"]["failure_phase"] = "validation"
    receipt["status"] = "pass"
    failures = validate_probe_receipt(receipt)
    receipt["validation_failures"] = list(failures)
    if failures:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].extend(failures)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.stage = args.stage.resolve(strict=True)
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    args.out.mkdir(parents=True, mode=0o700)
    receipt = _empty_receipt(args)
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": True,
                "hide_ui": True,
                "renderer": "PathTracing",
                "extra_args": list(REPLICATOR_EXTRA_ARGS),
            }
        )
        run_probe(args, app, receipt)
    except BaseException as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(
            {
                "phase": receipt["runtime"].get("failure_phase"),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }
        )
    finally:
        receipt_path = args.out / "receipt.json"
        try:
            # Preserve the result before Kit shutdown so shutdown cannot
            # prevent receipt persistence.
            _write_json(receipt_path, receipt)
        finally:
            if app is not None:
                try:
                    app.close()
                except BaseException as exc:
                    receipt["status"] = "insufficient_evidence"
                    receipt["errors"].append(
                        {
                            "phase": "simulation_app_close",
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": "".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                        }
                    )
                    _write_json(receipt_path, receipt)
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

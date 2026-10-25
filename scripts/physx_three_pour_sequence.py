#!/usr/bin/env python3
"""Run one bounded GPU PhysX three-pour sequence and preserve visual evidence.

This is a synthetic simulator experiment.  Programmed millilitres select a
documented nominal particle count; they are not measured displaced volume.
The shared isosurface material changes only after each settled pour under an
explicit fully-mixed visualization assumption.  No absorbance fitting occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.physx_three_pour import (
    CAMERA_PATH,
    CAMERA_CLIPPING_RANGE,
    FLUIDMIX_NAMESPACE,
    NOZZLE_EXIT_CENTER_M,
    O1B_MATERIAL_PARAMETERS,
    PARITY_THRESHOLDS,
    SOURCE_LIQUID_BOTTOM_M,
    SOURCE_LIQUID_PATH,
    SOURCE_LIQUID_RADIUS_M,
    SOURCE_NOZZLE_PATH,
    SOURCE_STREAM_PATH,
    SOURCE_VIAL_PATH,
    TWIN_REFERENCE_TIME_CODE,
    VIAL_CENTER_XY_M,
    VIAL_MOUTH_M,
    VISIBILITY_CHECK_TIME_CODES,
    build_twin_collider_specs,
    build_three_pour_plan,
    distribute_particle_count,
    validate_three_pour_receipt,
)

# Reuse the F1 routines that were exercised on both GB10 and Brev.  This script
# must be copied beside physx_particle_compat_probe.py on the remote host.
from physx_particle_compat_probe import (
    _finite_positions,
    _isosurface_mesh_evidence,
    _movement,
    _positions,
    _safe_call,
    _version_info,
)


WALL_LIMIT_SECONDS = 5400
ISOSURFACE_LIMIT_SECONDS = 2700
GPU_MEMORY_LIMIT_MIB = 40000
PHYSICS_HZ = 60
SUPPORTED_COLLIDER_TYPE = "Cube"
# Isaac Sim 6.0.1 documents skipped Replicator frames when a stopped timeline
# lets isaacsim.core.throttling enable asynchronous rendering.  Keep F2 on the
# same launch contract as the already exercised FillCam renderer.
REPLICATOR_ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
REPLICATOR_EXTRA_ARGS = (f"--{REPLICATOR_ASYNC_SETTING}=false",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--renderer", choices=("PathTracing",), default="PathTracing")
    parser.add_argument("--rt-subframes", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument(
        "--max-capture-tries",
        type=int,
        default=20,
        help="maximum same-state renderer attempts for one kept RGB frame",
    )
    parser.add_argument("--capture-stride", type=int, default=6)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--stable-min-steps", type=int, default=60)
    parser.add_argument("--stable-max-steps", type=int, default=240)
    parser.add_argument("--stable-consecutive-captures", type=int, default=3)
    parser.add_argument("--stable-mean-displacement-m", type=float, default=0.0005)
    parser.add_argument("--stable-max-displacement-m", type=float, default=0.003)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, int]:
    try:
        width_text, height_text = args.resolution.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("resolution must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    if (width, height) != (640, 480):
        raise ValueError("twin F2 resolution must be 640x480")
    if not args.stage.is_absolute() or not args.stage.is_file():
        raise ValueError("stage must be an existing absolute file")
    for name in (
        "rt_subframes",
        "warmup",
        "max_capture_tries",
        "capture_stride",
        "stable_min_steps",
        "stable_max_steps",
        "stable_consecutive_captures",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.stable_max_steps < args.stable_min_steps:
        raise ValueError("stable_max_steps must be >= stable_min_steps")
    for name in (
        "video_fps",
        "stable_mean_displacement_m",
        "stable_max_displacement_m",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("seed must be an integer")
    if args.rt_subframes != 64 or args.warmup != 60:
        raise ValueError("twin F2 requires rt_subframes=64 and warmup=60")
    return width, height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _gpu_used_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_three_pour_plan()
    script_path = Path(__file__).resolve()
    contract_path = ROOT / "src" / "fluidmix" / "physx_three_pour.py"
    return {
        "schema_version": 2,
        "experiment": "physx_three_pour_f2",
        "status": "insufficient_evidence",
        "plan": plan,
        "arguments": {
            "stage": str(args.stage),
            "seed": args.seed,
            "resolution": args.resolution,
            "renderer": args.renderer,
            "rt_subframes": args.rt_subframes,
            "warmup": args.warmup,
            "max_capture_tries": args.max_capture_tries,
            "capture_stride": args.capture_stride,
            "video_fps": args.video_fps,
            "stable_min_steps": args.stable_min_steps,
            "stable_max_steps": args.stable_max_steps,
            "stable_consecutive_captures": args.stable_consecutive_captures,
            "stable_mean_displacement_m": args.stable_mean_displacement_m,
            "stable_max_displacement_m": args.stable_max_displacement_m,
        },
        "provenance": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "script_path": str(script_path),
            "script_sha256": _sha256(script_path),
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": _sha256(contract_path),
            "requested_stage_path": str(args.stage.resolve()),
            "requested_stage_sha256": _sha256(args.stage),
        },
        "versions": {"isaac": None, "kit": None, "physx": None},
        "gpu_simulation": {
            "cuda_device_available": False,
            "scene_enable_gpu_dynamics": False,
            "broadphase": None,
        },
        "particle_system": {
            "count": 0,
            "exactly_one": False,
            "particle_set_count": 0,
            "mass_api_count": None,
            "all_particle_sets_target_one_system": False,
            "collider_types": [],
            "collider_paths": [],
            "all_colliders_invisible": False,
            "decorative_vial_collision_api": None,
            "pbd_material_binding": False,
            "render_material_binding": False,
        },
        "scene_contract": {
            "default_prim": None,
            "meters_per_unit": None,
            "up_axis": None,
            "frozen_time_code": TWIN_REFERENCE_TIME_CODE,
            "physics_namespace": FLUIDMIX_NAMESPACE,
            "source_layer_records": [],
            "source_layers_unchanged": False,
            "hidden_source_visibility": {},
            "forbidden_authored_paths": [],
            "custom_camera_count": 0,
            "custom_light_count": 0,
            "custom_backdrop_count": 0,
        },
        "camera": {
            "path": CAMERA_PATH,
            "source_has_authored_clipping_range": False,
            "clipping_range_authored_by_runner": False,
            "clipping_range_effective": None,
            "resolution": [640, 480],
        },
        "material": {
            "path": f"{FLUIDMIX_NAMESPACE}/PBDMaterial",
            "shader_path": f"{FLUIDMIX_NAMESPACE}/PBDMaterial/OmniSurface",
            "parameters": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in O1B_MATERIAL_PARAMETERS.items()
            },
            "fixed_for_entire_run": True,
        },
        "emitter": {
            "source_nozzle_path": SOURCE_NOZZLE_PATH,
            "source_nozzle_valid": False,
            "origin_m": list(NOZZLE_EXIT_CENTER_M),
            "velocity_m_s": [0.0, 0.0, -0.1],
        },
        "pre_pour_parity": {
            "thresholds": dict(PARITY_THRESHOLDS),
            "reference_path": None,
            "reference_sha256": None,
            "f2_prefill_path": None,
            "f2_prefill_sha256": None,
            "absdiff_path": None,
            "absdiff_sha256": None,
            "mean_abs_rgb": None,
            "p95_abs_rgb": None,
            "fraction_abs_rgb_gt_16": None,
            "passed": False,
        },
        "pours": [],
        "totals": {
            "programmed_volume_ml": 60,
            "emitted_particle_count": 0,
            "final_particle_count": 0,
            "nominal_mass_convention_equivalent_ml": plan["totals"]["nominal_mass_convention_equivalent_ml"],
            "nominal_error_ml": plan["totals"]["nominal_error_ml"],
        },
        "isosurface": {
            "enabled": False,
            "mesh_point_count": 0,
            "mesh_face_count": 0,
            "mesh_finite_points": False,
            "mesh_extent_nonempty": False,
        },
        "render": {
            "renderer": args.renderer,
            "resolution": [640, 480],
            "rt_subframes": args.rt_subframes,
            "warmup_updates": args.warmup,
            "frame_count": 0,
            "frame_paths": [],
            "frame_sha256": [],
            "stage_frame_paths": [],
            "stage_frame_sha256": [],
            "video_path": None,
            "video_sha256": None,
            "video_decodable": False,
            "video_decoded_frame_count": 0,
            "stage_frames_are_video_subsets": False,
            "max_capture_tries": args.max_capture_tries,
            "capture_attempt_count": 0,
            "capture_attempts": [],
            "capture_on_play": False,
            "wait_for_render": True,
            "same_state_delta_time": 0.0,
            "async_throttling_enabled": None,
            "rejected_rgb_diagnostics": [],
        },
        "memory": {
            "hard_limit_mib": GPU_MEMORY_LIMIT_MIB,
            "peak_used_mib": None,
            "hard_limit_exceeded": False,
            "samples_mib": [],
            "isosurface_enabled_seconds": None,
            "isosurface_time_limit_seconds": ISOSURFACE_LIMIT_SECONDS,
        },
        "timing": {
            "physics_hz": PHYSICS_HZ,
            "wall_seconds": None,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
        },
        "mixing_assumption": plan["mixing_assumption"],
        "claims": plan["claims"],
        "errors": [],
        "warnings": [],
    }


def _memory_sample(receipt: dict[str, Any]) -> int:
    value = _gpu_used_memory_mib()
    if value is None:
        raise RuntimeError("nvidia-smi did not return a GPU memory sample")
    receipt["memory"]["samples_mib"].append(value)
    peak = receipt["memory"]["peak_used_mib"]
    receipt["memory"]["peak_used_mib"] = value if peak is None else max(peak, value)
    if value >= GPU_MEMORY_LIMIT_MIB:
        receipt["memory"]["hard_limit_exceeded"] = True
        raise RuntimeError(f"GPU memory safety stop reached {value} MiB")
    return value


def _author_omnisurface(stage: Any, material: Any, Sdf: Any, Gf: Any, UsdShade: Any) -> Any:
    shader = UsdShade.Shader.Define(
        stage, f"{FLUIDMIX_NAMESPACE}/PBDMaterial/OmniSurface"
    )
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    values = {
        "enable_specular_transmission": (Sdf.ValueTypeNames.Bool, True),
        "specular_transmission_weight": (Sdf.ValueTypeNames.Float, 1.0),
        "specular_transmission_color": (
            Sdf.ValueTypeNames.Color3f,
            Gf.Vec3f(*O1B_MATERIAL_PARAMETERS["specular_transmission_color"]),
        ),
        "specular_transmission_scattering_depth": (
            Sdf.ValueTypeNames.Float,
            O1B_MATERIAL_PARAMETERS["specular_transmission_scattering_depth_m"],
        ),
        "thin_walled": (Sdf.ValueTypeNames.Bool, False),
        "diffuse_reflection_weight": (Sdf.ValueTypeNames.Float, 0.0),
        "specular_reflection_ior": (Sdf.ValueTypeNames.Float, 1.333),
        "specular_reflection_roughness": (Sdf.ValueTypeNames.Float, 0.08),
        "enable_opacity": (Sdf.ValueTypeNames.Bool, False),
        "geometry_opacity": (Sdf.ValueTypeNames.Float, 1.0),
        "geometry_opacity_threshold": (Sdf.ValueTypeNames.Float, 0.0),
    }
    for name, (type_name, value) in values.items():
        shader.CreateInput(name, type_name).Set(value)
    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader_output)
    material.CreateVolumeOutput("mdl").ConnectToSource(shader_output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader_output)
    return shader


def _set_stage_color(shader: Any, Gf: Any, color: tuple[float, float, float]) -> None:
    if not shader.GetInput("specular_transmission_color").Set(Gf.Vec3f(*color)):
        raise RuntimeError("could not update the shared fully-mixed stage color")


def _look_at_matrix(Gf: Any, eye: tuple[float, float, float], target: tuple[float, float, float]):
    # Camera local -Z points toward the target; inverse view gives the USD transform.
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 1.0, 0.0)).GetInverse()


def _author_and_attest_camera_clipping(
    camera: Any, Gf: Any, receipt: dict[str, Any]
) -> None:
    requested = list(CAMERA_CLIPPING_RANGE)
    receipt["camera"]["clipping_range_requested"] = requested
    attribute = camera.CreateClippingRangeAttr()
    set_ok = attribute.Set(Gf.Vec2f(*CAMERA_CLIPPING_RANGE))
    effective_value = camera.GetClippingRangeAttr().Get()
    effective = (
        [float(effective_value[0]), float(effective_value[1])]
        if effective_value is not None
        else None
    )
    authored = effective if attribute.HasAuthoredValueOpinion() else None
    receipt["camera"]["clipping_range_authored"] = authored
    receipt["camera"]["clipping_range_effective"] = effective
    if not set_ok or authored != requested or effective != requested:
        raise RuntimeError(
            "camera clipping range did not retain the requested live USD value"
        )


def _chunk_positions(Gf: Any, count: int, rng: random.Random) -> list[Any]:
    values = []
    spacing = 0.003
    for index in range(count):
        ix = index % 3
        iz = (index // 3) % 3
        iy = index // 9
        jitter_x = (rng.random() - 0.5) * 0.0003
        jitter_z = (rng.random() - 0.5) * 0.0003
        values.append(
            Gf.Vec3f(
                (ix - 1) * spacing + jitter_x,
                0.145 + iy * spacing,
                (iz - 1) * spacing + jitter_z,
            )
        )
    return values


def _all_positions(active_prims: list[Any]) -> list[list[float]]:
    combined: list[list[float]] = []
    for prim in active_prims:
        combined.extend(_positions(prim))
    return combined


def _escaped_count(positions: list[list[float]]) -> int:
    return sum(
        1
        for x, y, z in positions
        if not (-0.015 <= x <= 0.015 and -0.0005 <= y <= 0.145 and -0.015 <= z <= 0.015)
    )


def _decode_video(cv2: Any, path: Path) -> tuple[bool, int]:
    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        return False, 0
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                return False, count
            count += 1
    finally:
        capture.release()
    return count > 0, count


def _write_rejected_rgb_diagnostic(
    cv2: Any,
    np: Any,
    output: Path,
    rgb: Any,
    *,
    step: int,
    phase: str,
) -> dict[str, Any]:
    """Persist a rejected full-shape RGB without making it a sequence frame."""

    safe_phase = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in phase
    )
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    path = diagnostics / f"rejected-rgb-step-{step:05d}-{safe_phase}.png"
    if not cv2.imwrite(os.fspath(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write rejected RGB diagnostic {path}")
    channel_mean = np.mean(rgb, axis=(0, 1))
    channel_std = np.std(rgb, axis=(0, 1))
    return {
        "path": str(path.relative_to(output)),
        "sha256": _sha256(path),
        "shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "channel_mean": [float(value) for value in channel_mean],
        "channel_std": [float(value) for value in channel_std],
        "spatial_rgb_std": float(np.std(rgb)),
        "minimum": int(np.min(rgb)),
        "maximum": int(np.max(rgb)),
        "spatially_constant": not bool(np.any(rgb != rgb[0, 0])),
        "included_in_frame_count": False,
        "included_in_video": False,
        "included_as_stage": False,
    }


def _start_or_reuse_kit() -> tuple[Any, Any]:
    try:
        import omni.kit.app

        current = omni.kit.app.get_app()
        if current is not None and bool(_safe_call(current, "is_running", False)):
            return current, None
    except Exception:
        pass
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "PathTracing",
            "extra_args": list(REPLICATOR_EXTRA_ARGS),
        }
    )
    import omni.kit.app

    return omni.kit.app.get_app(), simulation_app


def _configure_replicator_capture(rep: Any) -> None:
    rep.orchestrator.set_capture_on_play(False)


def _replicator_same_state_step(rep: Any, *, rt_subframes: int) -> None:
    rep.orchestrator.step(
        delta_time=0.0,
        pause_timeline=True,
        rt_subframes=rt_subframes,
        wait_for_render=True,
    )


def _source_layer_records(stage: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if not layer.realPath:
            continue
        path = Path(layer.realPath).resolve(strict=True)
        records.append(
            {
                "identifier": str(layer.identifier),
                "resolved_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise RuntimeError("opened twin stage has no hashable non-session layers")
    return records


def _twin_chunk_positions(Gf: Any, count: int, rng: random.Random) -> list[Any]:
    values = []
    spacing = 0.003
    for index in range(count):
        ix = index % 3
        iy = (index // 3) % 3
        iz = index // 9
        values.append(
            Gf.Vec3f(
                NOZZLE_EXIT_CENTER_M[0]
                + (ix - 1) * spacing
                + (rng.random() - 0.5) * 0.0003,
                NOZZLE_EXIT_CENTER_M[1]
                + (iy - 1) * spacing
                + (rng.random() - 0.5) * 0.0003,
                min(NOZZLE_EXIT_CENTER_M[2] - 0.003, VIAL_MOUTH_M - 0.001)
                - iz * spacing,
            )
        )
    return values


def _twin_escaped_count(positions: list[list[float]]) -> int:
    return sum(
        1
        for x, y, z in positions
        if (
            math.hypot(x - VIAL_CENTER_XY_M[0], y - VIAL_CENTER_XY_M[1])
            > SOURCE_LIQUID_RADIUS_M
            or z < SOURCE_LIQUID_BOTTOM_M - 0.001
            or z > VIAL_MOUTH_M + 0.001
        )
    )


def _parity_metrics(np: Any, reference: Any, current: Any) -> dict[str, Any]:
    if reference.shape != current.shape:
        raise RuntimeError("pre-pour parity images have different shapes")
    difference = np.abs(reference.astype(np.int16) - current.astype(np.int16))
    mean_abs = float(np.mean(difference))
    p95_abs = float(np.percentile(difference, 95))
    fraction_gt_16 = float(np.mean(difference > 16))
    return {
        "mean_abs_rgb": mean_abs,
        "p95_abs_rgb": p95_abs,
        "fraction_abs_rgb_gt_16": fraction_gt_16,
        "passed": bool(
            mean_abs <= PARITY_THRESHOLDS["mean_abs_rgb_max"]
            and p95_abs <= PARITY_THRESHOLDS["p95_abs_rgb_max"]
            and fraction_gt_16
            <= PARITY_THRESHOLDS["fraction_abs_rgb_gt_16_max"]
        ),
        "difference": difference,
    }


def run_sequence(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
    """Run F2 on the existing fill-station twin; only liquid physics is added."""

    width, height = _validate_args(args)
    receipt = _empty_receipt(args)
    started = time.monotonic()
    deadline = started + WALL_LIMIT_SECONDS
    isosurface_started: float | None = None
    simulation = None
    stage_attached = False
    render_product = None
    rgb_annotator = None
    writer = None
    frame_paths: list[Path] = []
    stage_paths: list[Path] = []
    active_prims: list[Any] = []
    try:
        import carb
        import cv2
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

        settings = carb.settings.get_settings()
        receipt["render"]["async_throttling_enabled"] = settings.get_as_bool(
            REPLICATOR_ASYNC_SETTING
        )
        if receipt["render"]["async_throttling_enabled"] is not False:
            raise RuntimeError("Replicator asynchronous throttling is still enabled")
        settings.set("/rtx/pathtracing/maxBounces", 16)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
        manager = app.get_extension_manager()
        receipt["versions"] = _version_info(app)
        if not manager.is_extension_enabled("omni.physx"):
            raise RuntimeError("omni.physx extension is not enabled")
        simulation = omni.physx.get_physx_simulation_interface()
        foundation = get_physx_foundation_interface()
        receipt["gpu_simulation"]["cuda_device_available"] = bool(
            foundation.cuda_device_check()
        )
        if not receipt["gpu_simulation"]["cuda_device_available"]:
            raise RuntimeError("GPU particles require a CUDA device")

        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(args.stage)):
            raise RuntimeError(f"could not open twin stage: {args.stage}")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not return the opened twin stage")
        layer_records = _source_layer_records(stage)
        receipt["scene_contract"]["source_layer_records"] = layer_records
        default_prim = stage.GetDefaultPrim()
        default_path = str(default_prim.GetPath()) if default_prim.IsValid() else None
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        receipt["scene_contract"].update(
            {
                "default_prim": default_path,
                "meters_per_unit": meters_per_unit,
                "up_axis": up_axis,
            }
        )
        if default_path != "/FillStation" or meters_per_unit != 1.0 or up_axis != "Z":
            raise RuntimeError("opened stage is not the reviewed metre-scale Z-up FillStation twin")

        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0:
            raise RuntimeError("twin timeCodesPerSecond is invalid")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(TWIN_REFERENCE_TIME_CODE / time_codes_per_second)
        app.update()

        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"reviewed FillCam missing: {CAMERA_PATH}")
        camera = UsdGeom.Camera(camera_prim)
        clip_attr = camera.GetClippingRangeAttr()
        clip_value = clip_attr.Get(Usd.TimeCode(TWIN_REFERENCE_TIME_CODE))
        effective_clip = [float(clip_value[0]), float(clip_value[1])]
        receipt["camera"].update(
            {
                "source_has_authored_clipping_range": bool(
                    clip_attr.HasAuthoredValueOpinion()
                ),
                "clipping_range_effective": effective_clip,
            }
        )
        if not clip_attr.HasAuthoredValueOpinion() or any(
            not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
            for value, expected in zip(effective_clip, CAMERA_CLIPPING_RANGE)
        ):
            raise RuntimeError("FillCam authored clipping range differs from source contract")

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(CAMERA_PATH, (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        _configure_replicator_capture(rep)

        parity_dir = output / "pre-pour-parity"
        frames_dir = output / "frames"
        stages_dir = output / "stages"
        parity_dir.mkdir()
        frames_dir.mkdir()
        stages_dir.mkdir()
        for _ in range(args.warmup):
            app.update()

        def check_time_limits() -> None:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError("F2 wall-time safety stop reached")
            if isosurface_started is not None and now - isosurface_started > ISOSURFACE_LIMIT_SECONDS:
                raise RuntimeError("F2 isosurface-enabled time safety stop reached")

        def acquire_rgb(step: int, phase: str, *, included_in_video: bool) -> Any:
            invalid_reasons: list[str] = []
            rgb = None
            last_rejected_rgb = None
            for _attempt in range(1, args.max_capture_tries + 1):
                check_time_limits()
                _replicator_same_state_step(rep, rt_subframes=args.rt_subframes)
                candidate = np.asarray(rgb_annotator.get_data())
                if candidate.ndim != 3 or candidate.shape[:2] != (height, width) or candidate.shape[2] < 3:
                    invalid_reasons.append(f"shape={candidate.shape}")
                    continue
                candidate = np.asarray(candidate[..., :3], dtype=np.uint8)
                if not np.any(candidate != candidate[0, 0]):
                    invalid_reasons.append("spatially_constant")
                    last_rejected_rgb = candidate.copy()
                    continue
                rgb = candidate
                break
            tries = len(invalid_reasons) + (1 if rgb is not None else 0)
            receipt["render"]["capture_attempt_count"] += tries
            attempt_record = {
                "frame_index": len(frame_paths) + 1 if included_in_video else None,
                "physics_step": step,
                "phase": phase,
                "tries": tries,
                "invalid_reasons": invalid_reasons,
                "included_in_video": included_in_video,
                "rejected_rgb_diagnostic": None,
            }
            receipt["render"]["capture_attempts"].append(attempt_record)
            if rgb is None:
                if last_rejected_rgb is not None:
                    diagnostic = _write_rejected_rgb_diagnostic(
                        cv2, np, output, last_rejected_rgb, step=step, phase=phase
                    )
                    attempt_record["rejected_rgb_diagnostic"] = diagnostic
                    receipt["render"]["rejected_rgb_diagnostics"].append(diagnostic)
                raise RuntimeError(
                    f"RGB capture failed after {args.max_capture_tries} tries: "
                    f"{invalid_reasons[-1] if invalid_reasons else 'unknown'}"
                )
            return rgb

        reference_rgb = acquire_rgb(0, "twin-reference-time-0001", included_in_video=False)
        reference_path = parity_dir / "twin-reference-time-0001.png"
        if not cv2.imwrite(os.fspath(reference_path), cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write twin reference image")

        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particles_setting = getattr(physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None)
        if particles_setting is None:
            raise RuntimeError("PhysX lacks SETTING_UPDATE_PARTICLES_TO_USD")
        settings.set(particles_setting, True)
        session_layer = stage.GetSessionLayer()
        authored_paths: list[str] = []
        with Usd.EditContext(stage, session_layer):
            for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                prim = stage.GetPrimAtPath(hidden_path)
                if not prim.IsValid():
                    raise RuntimeError(f"source liquid prim missing: {hidden_path}")
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                authored_paths.append(hidden_path)

            UsdGeom.Xform.Define(stage, FLUIDMIX_NAMESPACE)
            scene = UsdPhysics.Scene.Define(stage, f"{FLUIDMIX_NAMESPACE}/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            scene_api.CreateSolverTypeAttr().Set("TGS")
            receipt["gpu_simulation"].update(
                {
                    "scene_enable_gpu_dynamics": bool(scene_api.GetEnableGPUDynamicsAttr().Get()),
                    "broadphase": str(scene_api.GetBroadphaseTypeAttr().Get()),
                }
            )

            system_path = Sdf.Path(f"{FLUIDMIX_NAMESPACE}/ParticleSystem")
            system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateParticleContactOffsetAttr().Set(0.003)
            system.CreateFluidRestOffsetAttr().Set(0.0015)
            system.CreateContactOffsetAttr().Set(0.003)
            system.CreateRestOffsetAttr().Set(0.0029)
            system.CreateSolidRestOffsetAttr().Set(0.0029)

            material = UsdShade.Material.Define(stage, f"{FLUIDMIX_NAMESPACE}/PBDMaterial")
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
            pbd.CreateDensityAttr().Set(1000.0)
            pbd.CreateFrictionAttr().Set(0.1)
            binding = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            _author_omnisurface(stage, material, Sdf, Gf, UsdShade)
            binding.Bind(material)
            receipt["particle_system"].update(
                {
                    "pbd_material_binding": bool(binding.GetDirectBinding("physics").GetMaterial()),
                    "render_material_binding": bool(binding.GetDirectBinding().GetMaterial()),
                }
            )

            collider_prims = []
            for spec in build_twin_collider_specs():
                cube = UsdGeom.Cube.Define(stage, Sdf.Path(str(spec["path"])))
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
                cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())
                authored_paths.append(str(spec["path"]))
            receipt["particle_system"].update(
                {
                    "collider_types": ["Cube"] * len(collider_prims),
                    "collider_paths": [str(prim.GetPath()) for prim in collider_prims],
                    "all_colliders_invisible": all(
                        UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
                        for prim in collider_prims
                    ),
                    "decorative_vial_collision_api": any(
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        for prim in Usd.PrimRange(stage.GetPrimAtPath(SOURCE_VIAL_PATH))
                    ),
                }
            )

            iso_api = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(system.GetPrim())
            iso_api.CreateIsosurfaceEnabledAttr().Set(True)
            iso_api.CreateGridSpacingAttr().Set(0.002)
            iso_api.CreateMaxVerticesAttr().Set(250000)
            iso_api.CreateMaxTrianglesAttr().Set(500000)
            receipt["isosurface"]["enabled"] = bool(iso_api.GetIsosurfaceEnabledAttr().Get())

            plan = receipt["plan"]
            rng = random.Random(args.seed)
            chunk_records: list[tuple[Any, Any, int, str]] = []
            for pour in plan["pours"]:
                counts = distribute_particle_count(pour["particle_count"], pour["emission_chunk_count"])
                for chunk_ordinal, count in enumerate(counts, start=1):
                    path = Sdf.Path(f"{FLUIDMIX_NAMESPACE}/FluidParticles/{pour['tank_id'].upper()}/Chunk{chunk_ordinal:03d}")
                    points = UsdGeom.Points.Define(stage, path)
                    positions = _twin_chunk_positions(Gf, count, rng)
                    points.GetPointsAttr().Set(Vt.Vec3fArray(positions))
                    points.GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, -0.10)] * count))
                    points.GetWidthsAttr().Set(Vt.FloatArray([0.003] * count))
                    set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
                    particle_api = PhysxSchema.PhysxParticleAPI(set_api)
                    particle_api.CreateParticleSystemRel().SetTargets([system_path])
                    particle_api.CreateSelfCollisionAttr().Set(True)
                    particle_api.CreateParticleEnabledAttr().Set(False)
                    particle_api.CreateParticleGroupAttr().Set(0)
                    set_api.CreateFluidAttr().Set(True)
                    chunk_records.append((points.GetPrim(), particle_api, count, pour["tank_id"]))

        receipt["emitter"]["source_nozzle_valid"] = stage.GetPrimAtPath(SOURCE_NOZZLE_PATH).IsValid()
        visibility_record: dict[str, dict[str, str]] = {}
        for hidden_path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(hidden_path))
            visibility_record[hidden_path] = {
                str(code): str(imageable.GetVisibilityAttr().Get(Usd.TimeCode(code)))
                for code in VISIBILITY_CHECK_TIME_CODES
            }
        receipt["scene_contract"]["hidden_source_visibility"] = visibility_record
        if any(value != "invisible" for row in visibility_record.values() for value in row.values()):
            raise RuntimeError("source liquid visibility override did not compose at all critical times")
        camera_property = Sdf.Path(CAMERA_PATH).AppendProperty("clippingRange")
        receipt["camera"]["clipping_range_authored_by_runner"] = bool(
            session_layer.GetPropertyAtPath(camera_property)
        )
        if receipt["camera"]["clipping_range_authored_by_runner"]:
            raise RuntimeError("runner unexpectedly authored FillCam clippingRange")
        forbidden = []
        if session_layer.GetPrimAtPath("/World"):
            forbidden.append("/World")
        namespace_prims = [
            prim for prim in stage.Traverse() if str(prim.GetPath()).startswith(FLUIDMIX_NAMESPACE)
        ]
        receipt["scene_contract"].update(
            {
                "forbidden_authored_paths": forbidden,
                "custom_camera_count": sum(prim.IsA(UsdGeom.Camera) for prim in namespace_prims),
                "custom_light_count": sum(prim.GetTypeName().endswith("Light") for prim in namespace_prims),
                "custom_backdrop_count": sum("backdrop" in str(prim.GetPath()).lower() for prim in namespace_prims),
                "authored_paths": authored_paths,
            }
        )
        if forbidden or any(receipt["scene_contract"][key] for key in ("custom_camera_count", "custom_light_count", "custom_backdrop_count")):
            raise RuntimeError("F2 authored a forbidden replacement-scene prim")

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("twin F2 stage has no StageCache ID")
        simulation.attach_stage(stage_id)
        stage_attached = True
        isosurface_started = time.monotonic()
        for _ in range(2):
            app.update()
        _memory_sample(receipt)

        video_path = output / "three-pour.mp4"
        writer = cv2.VideoWriter(os.fspath(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.video_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v writer did not open")

        def capture(step: int, phase: str) -> tuple[Path, list[list[float]], Any]:
            rgb = acquire_rgb(step, phase, included_in_video=True)
            index = len(frame_paths) + 1
            path = frames_dir / f"frame-{index:04d}-step-{step:05d}-{phase}.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(os.fspath(path), bgr):
                raise RuntimeError(f"could not write RGB frame {path}")
            writer.write(bgr)
            frame_paths.append(path)
            return path, _all_positions(active_prims), rgb

        prefill_path, _, prefill_rgb = capture(0, "prefill-time-0001")
        parity_prefill_path = parity_dir / "f2-prefill-time-0001.png"
        shutil.copyfile(prefill_path, parity_prefill_path)
        parity = _parity_metrics(np, reference_rgb, prefill_rgb)
        difference_image = np.clip(parity.pop("difference") * 8, 0, 255).astype(np.uint8)
        difference_path = parity_dir / "absdiff-times-8.png"
        if not cv2.imwrite(os.fspath(difference_path), cv2.cvtColor(difference_image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write parity absolute-difference image")
        receipt["pre_pour_parity"].update(
            {
                "reference_path": str(reference_path.relative_to(output)),
                "reference_sha256": _sha256(reference_path),
                "f2_prefill_path": str(parity_prefill_path.relative_to(output)),
                "f2_prefill_sha256": _sha256(parity_prefill_path),
                "absdiff_path": str(difference_path.relative_to(output)),
                "absdiff_sha256": _sha256(difference_path),
                **parity,
            }
        )
        if not parity["passed"]:
            raise RuntimeError("F2 pre-pour frame does not match the source twin within frozen noise limits")

        dt = 1.0 / PHYSICS_HZ
        global_step = 0
        chunk_cursor = 0
        cumulative = 0
        for stage_index, pour in enumerate(receipt["plan"]["pours"], start=1):
            pre_memory = _memory_sample(receipt)
            stage_memory_samples = [pre_memory]
            pour_chunks = pour["emission_chunk_count"]
            for _ in range(pour_chunks):
                prim, particle_api, _count, tank_id = chunk_records[chunk_cursor]
                chunk_cursor += 1
                if tank_id != pour["tank_id"]:
                    raise RuntimeError("pre-authored emission order drifted")
                if not particle_api.GetParticleEnabledAttr().Set(True):
                    raise RuntimeError("could not enable a particle emission chunk")
                active_prims.append(prim)
                check_time_limits()
                global_step += 1
                simulation.simulate(dt, global_step * dt)
                simulation.fetch_results()
                app.update()
                if global_step % args.capture_stride == 0:
                    capture(global_step, f"{pour['tank_id']}-pour")
                    stage_memory_samples.append(_memory_sample(receipt))

            previous = _all_positions(active_prims)
            stable_streak = 0
            stable = False
            stable_frame_path = None
            stable_motion = None
            settling_steps = 0
            while settling_steps < args.stable_max_steps:
                check_time_limits()
                global_step += 1
                settling_steps += 1
                simulation.simulate(dt, global_step * dt)
                simulation.fetch_results()
                app.update()
                if global_step % args.capture_stride != 0:
                    continue
                frame_path, current, _ = capture(global_step, f"{pour['tank_id']}-settle")
                stage_memory_samples.append(_memory_sample(receipt))
                motion = _movement(previous, current)
                previous = current
                if (settling_steps >= args.stable_min_steps and motion is not None and motion["mean_m"] <= args.stable_mean_displacement_m and motion["max_m"] <= args.stable_max_displacement_m):
                    stable_streak += 1
                else:
                    stable_streak = 0
                if stable_streak >= args.stable_consecutive_captures:
                    stable = True
                    stable_motion = motion
                    stable_frame_path, _, _ = capture(global_step, f"{pour['tank_id']}-stable-stage")
                    break
            positions = _all_positions(active_prims)
            cumulative += pour["particle_count"]
            iso_evidence = _isosurface_mesh_evidence(stage, system_path.AppendChild("Isosurface"))
            if stable_frame_path is None:
                stable_digest = None
                stable_index = None
            else:
                destination = stages_dir / f"stage-{stage_index}-{pour['tank_id']}.png"
                shutil.copyfile(stable_frame_path, destination)
                stable_digest = _sha256(destination)
                if stable_digest != _sha256(stable_frame_path):
                    raise RuntimeError("stage-frame copy differs from source video frame")
                stable_index = len(frame_paths)
                stage_paths.append(destination)
            final_memory = stage_memory_samples[-1]
            peak_memory = max(stage_memory_samples)
            receipt["pours"].append(
                {
                    "ordinal": stage_index,
                    "tank_id": pour["tank_id"],
                    "programmed_volume_ml": pour["programmed_volume_ml"],
                    "emitted_particle_count": pour["particle_count"],
                    "cumulative_particle_count": cumulative,
                    "emission_chunk_count": pour_chunks,
                    "settling_steps": settling_steps,
                    "finite_positions": _finite_positions(positions),
                    "escaped_particle_count": _twin_escaped_count(positions),
                    "stable": stable,
                    "stable_motion": stable_motion,
                    "stable_frame_index": stable_index,
                    "stable_frame_path": str(stage_paths[-1].relative_to(output)) if stable_frame_path is not None else None,
                    "stable_source_frame_path": str(stable_frame_path.relative_to(output)) if stable_frame_path is not None else None,
                    "stable_frame_sha256": stable_digest,
                    "isosurface_nonempty": bool(iso_evidence["mesh_point_count"] > 0 and iso_evidence["mesh_face_count"] > 0 and iso_evidence["mesh_finite_points"] and iso_evidence["mesh_extent_nonempty"]),
                    "used_gpu_memory_mib_before": pre_memory,
                    "used_gpu_memory_mib_peak": peak_memory,
                    "used_gpu_memory_mib_after": final_memory,
                    "memory_recovered_during_settle": bool(peak_memory - pre_memory < 4096 or final_memory < peak_memory),
                    "fixed_o1b_material": True,
                }
            )
            row = receipt["pours"][-1]
            if not stable:
                raise RuntimeError(f"{pour['tank_id']} did not meet the frozen stability threshold")
            if row["escaped_particle_count"]:
                raise RuntimeError(f"{pour['tank_id']} has escaped particles")
            if not row["isosurface_nonempty"]:
                raise RuntimeError(f"{pour['tank_id']} has no nonempty isosurface")
            if peak_memory - pre_memory >= 4096 and final_memory >= peak_memory:
                raise RuntimeError(f"{pour['tank_id']} isosurface memory rose >=4096 MiB without recovery")

        systems = [prim for prim in stage.Traverse() if PhysxSchema.PhysxParticleSystem(prim)]
        particle_prims = [prim for prim in stage.Traverse() if prim.HasAPI(PhysxSchema.PhysxParticleSetAPI)]
        receipt["particle_system"].update(
            {
                "count": len(systems),
                "exactly_one": len(systems) == 1,
                "particle_set_count": len(particle_prims),
                "mass_api_count": sum(1 for prim in particle_prims if prim.HasAPI(UsdPhysics.MassAPI)),
                "all_particle_sets_target_one_system": all(
                    [str(path) for path in PhysxSchema.PhysxParticleAPI(PhysxSchema.PhysxParticleSetAPI(prim)).GetParticleSystemRel().GetTargets()] == [str(system_path)]
                    for prim in particle_prims
                ),
            }
        )
        final_positions = _all_positions(active_prims)
        receipt["totals"].update({"emitted_particle_count": cumulative, "final_particle_count": len(final_positions)})
        receipt["isosurface"].update(_isosurface_mesh_evidence(stage, system_path.AppendChild("Isosurface")))
        receipt["memory"]["isosurface_enabled_seconds"] = time.monotonic() - isosurface_started
        writer.release()
        writer = None
        decodable, decoded_count = _decode_video(cv2, video_path)
        frame_hashes = [_sha256(path) for path in frame_paths]
        stage_hashes = [_sha256(path) for path in stage_paths]
        receipt["render"].update(
            {
                "frame_count": len(frame_paths),
                "frame_paths": [str(path.relative_to(output)) for path in frame_paths],
                "frame_sha256": frame_hashes,
                "stage_frame_paths": [str(path.relative_to(output)) for path in stage_paths],
                "stage_frame_sha256": stage_hashes,
                "video_path": str(video_path.relative_to(output)),
                "video_sha256": _sha256(video_path),
                "video_decodable": decodable,
                "video_decoded_frame_count": decoded_count,
                "stage_frames_are_video_subsets": bool(len(stage_hashes) == 3 and set(stage_hashes).issubset(set(frame_hashes))),
            }
        )
        receipt["scene_contract"]["source_layers_unchanged"] = all(
            Path(row["resolved_path"]).is_file()
            and _sha256(Path(row["resolved_path"])) == row["sha256"]
            for row in layer_records
        )
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
        receipt["status"] = "pass"
        failures = validate_three_pour_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(f"F2 execution failed: {type(exc).__name__}: {exc}")
    finally:
        if writer is not None:
            writer.release()
        if rgb_annotator is not None and render_product is not None:
            try:
                rgb_annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"RGB annotator detach failed: {exc}")
        if stage_attached and simulation is not None:
            try:
                simulation.detach_stage()
            except Exception as exc:
                receipt["warnings"].append(f"PhysX stage detach failed: {exc}")
        if isosurface_started is not None and receipt["memory"]["isosurface_enabled_seconds"] is None:
            receipt["memory"]["isosurface_enabled_seconds"] = time.monotonic() - isosurface_started
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_only:
        if args.out.exists():
            raise FileExistsError(f"output path already exists: {args.out}")
        args.out.mkdir(parents=True)
        _write_json(args.out / "plan.json", build_three_pour_plan())
        return 0
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    owned_app = None
    exit_code = 2
    try:
        app, owned_app = _start_or_reuse_kit()
        receipt = run_sequence(args, app, args.out)
    except Exception as exc:
        receipt = _empty_receipt(args)
        receipt["errors"].append(
            f"Isaac application startup failed: {type(exc).__name__}: {exc}"
        )
    try:
        receipt_path = args.out / "receipt.json"
        _write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if receipt["status"] == "pass" else 2
    finally:
        if owned_app is not None:
            owned_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

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
    CAMERA_CLIPPING_RANGE,
    build_render_guide_specs,
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
STAGE_COLORS = (
    (0.95, 0.55, 0.18),
    (0.48, 0.34, 0.58),
    (0.72, 0.64, 0.76),
)
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
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--renderer", choices=("PathTracing",), default="PathTracing")
    parser.add_argument("--rt-subframes", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
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
        "schema_version": 1,
        "experiment": "physx_three_pour_f2",
        "status": "insufficient_evidence",
        "plan": plan,
        "arguments": {
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
            "pbd_material_binding": False,
            "render_material_binding": False,
        },
        "render_guides": {
            "expected_count": len(build_render_guide_specs()),
            "authored_count": 0,
            "collision_api_count": None,
            "paths": [],
        },
        "camera": {
            "path": "/World/Camera",
            "clipping_range_requested": list(CAMERA_CLIPPING_RANGE),
            "clipping_range_authored": None,
            "clipping_range_effective": None,
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
    shader = UsdShade.Shader.Define(stage, "/World/PBDMaterial/OmniSurface")
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    values = {
        "enable_specular_transmission": (Sdf.ValueTypeNames.Bool, True),
        "specular_transmission_weight": (Sdf.ValueTypeNames.Float, 1.0),
        "specular_transmission_color": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.72, 0.82, 0.92)),
        "specular_transmission_scattering_depth": (Sdf.ValueTypeNames.Float, 0.01),
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


def run_sequence(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
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
        import omni.kit.app
        import omni.physx
        import omni.replicator.core as rep
        import omni.usd
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt

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
        context.new_stage()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not create the F2 stage")
        app.update()
        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particles_setting = getattr(
            physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None
        )
        if particles_setting is None:
            raise RuntimeError("PhysX lacks SETTING_UPDATE_PARTICLES_TO_USD")
        settings.set(particles_setting, True)

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        stage.SetTimeCodesPerSecond(float(PHYSICS_HZ))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        scene_api.CreateSolverTypeAttr().Set("TGS")
        receipt["gpu_simulation"]["scene_enable_gpu_dynamics"] = bool(
            scene_api.GetEnableGPUDynamicsAttr().Get()
        )
        receipt["gpu_simulation"]["broadphase"] = str(
            scene_api.GetBroadphaseTypeAttr().Get()
        )

        system_path = Sdf.Path("/World/ParticleSystem")
        system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
        system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
        system.CreateParticleSystemEnabledAttr().Set(True)
        system.CreateParticleContactOffsetAttr().Set(0.003)
        system.CreateFluidRestOffsetAttr().Set(0.0015)
        system.CreateContactOffsetAttr().Set(0.003)
        system.CreateRestOffsetAttr().Set(0.0029)
        system.CreateSolidRestOffsetAttr().Set(0.0029)

        material = UsdShade.Material.Define(stage, "/World/PBDMaterial")
        pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
        pbd.CreateDensityAttr().Set(1000.0)
        pbd.CreateFrictionAttr().Set(0.1)
        binding = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        shader = _author_omnisurface(stage, material, Sdf, Gf, UsdShade)
        binding.Bind(material)
        receipt["particle_system"]["pbd_material_binding"] = bool(
            binding.GetDirectBinding("physics").GetMaterial()
        )
        receipt["particle_system"]["render_material_binding"] = bool(
            binding.GetDirectBinding().GetMaterial()
        )

        def box(path: str, center: tuple[float, float, float], scale: tuple[float, float, float], *, visible: bool = False):
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
            cube.CreateSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3f(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if not visible:
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            return cube

        box("/World/VialCollider/Floor", (0.0, -0.001, 0.0), (0.034, 0.002, 0.034))
        box("/World/VialCollider/WallXMin", (-0.016, 0.075, 0.0), (0.002, 0.15, 0.034))
        box("/World/VialCollider/WallXMax", (0.016, 0.075, 0.0), (0.002, 0.15, 0.034))
        box("/World/VialCollider/WallZMin", (0.0, 0.075, -0.016), (0.034, 0.15, 0.002))
        box("/World/VialCollider/WallZMax", (0.0, 0.075, 0.016), (0.034, 0.15, 0.002))
        receipt["particle_system"]["collider_types"] = [SUPPORTED_COLLIDER_TYPE]

        UsdGeom.Xform.Define(stage, "/World/RenderGuides")
        guide_material = UsdShade.Material.Define(stage, "/World/RenderGuideMaterial")
        guide_shader = UsdShade.Shader.Define(
            stage, "/World/RenderGuideMaterial/Preview"
        )
        guide_shader.CreateIdAttr().Set("UsdPreviewSurface")
        guide_shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(0.12, 0.14, 0.17))
        guide_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
        guide_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        guide_material.CreateSurfaceOutput().ConnectToSource(
            guide_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        guide_prims = []
        guide_paths = []
        for spec in build_render_guide_specs():
            guide_path = Sdf.Path(f"/World/RenderGuides/{spec['name']}")
            guide = UsdGeom.Cube.Define(stage, guide_path)
            guide.CreateSizeAttr().Set(1.0)
            guide.AddTranslateOp().Set(Gf.Vec3f(*spec["center_m"]))
            guide.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
            UsdShade.MaterialBindingAPI.Apply(guide.GetPrim()).Bind(guide_material)
            guide_prims.append(guide.GetPrim())
            guide_paths.append(str(guide_path))
        guide_collision_api_count = sum(
            1 for prim in guide_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        receipt["render_guides"].update(
            {
                "authored_count": len(guide_prims),
                "collision_api_count": guide_collision_api_count,
                "paths": guide_paths,
            }
        )
        if len(guide_prims) != receipt["render_guides"]["expected_count"]:
            raise RuntimeError("render-only vessel guide count differs from contract")
        if guide_collision_api_count != 0:
            raise RuntimeError("render-only vessel guide acquired CollisionAPI")

        backdrop = UsdGeom.Cube.Define(stage, "/World/Backdrop")
        backdrop.CreateSizeAttr().Set(1.0)
        backdrop.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.075, -0.04))
        backdrop.AddScaleOp().Set(Gf.Vec3f(0.16, 0.18, 0.002))
        backdrop_material = UsdShade.Material.Define(stage, "/World/BackdropMaterial")
        backdrop_shader = UsdShade.Shader.Define(stage, "/World/BackdropMaterial/Preview")
        backdrop_shader.CreateIdAttr().Set("UsdPreviewSurface")
        backdrop_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.72, 0.74, 0.78)
        )
        backdrop_material.CreateSurfaceOutput().ConnectToSource(
            backdrop_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        UsdShade.MaterialBindingAPI.Apply(backdrop.GetPrim()).Bind(backdrop_material)

        light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        light.CreateIntensityAttr().Set(1000.0)
        key = UsdLux.RectLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr().Set(18000.0)
        key.CreateWidthAttr().Set(0.10)
        key.CreateHeightAttr().Set(0.10)
        UsdGeom.Xformable(key.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3f(0.08, 0.18, 0.08)
        )

        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr().Set(45.0)
        camera.AddTransformOp().Set(
            _look_at_matrix(Gf, (0.11, 0.085, 0.12), (0.0, 0.07, 0.0))
        )
        _author_and_attest_camera_clipping(camera, Gf, receipt)
        render_product = rep.create.render_product(str(camera.GetPath()), (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        _configure_replicator_capture(rep)

        iso_api = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(system.GetPrim())
        iso_api.CreateIsosurfaceEnabledAttr().Set(True)
        iso_api.CreateGridSpacingAttr().Set(0.002)
        iso_api.CreateMaxVerticesAttr().Set(250000)
        iso_api.CreateMaxTrianglesAttr().Set(500000)
        receipt["isosurface"]["enabled"] = bool(
            iso_api.GetIsosurfaceEnabledAttr().Get()
        )
        isosurface_started = time.monotonic()

        def check_time_limits() -> None:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError("F2 wall-time safety stop reached")
            if now - isosurface_started > ISOSURFACE_LIMIT_SECONDS:
                raise RuntimeError("F2 isosurface-enabled time safety stop reached")

        plan = receipt["plan"]
        rng = random.Random(args.seed)
        chunk_records: list[tuple[Any, Any, int, str]] = []
        for pour in plan["pours"]:
            counts = distribute_particle_count(
                pour["particle_count"], pour["emission_chunk_count"]
            )
            for chunk_ordinal, count in enumerate(counts, start=1):
                path = Sdf.Path(
                    f"/World/FluidParticles/{pour['tank_id'].upper()}/Chunk{chunk_ordinal:03d}"
                )
                points = UsdGeom.Points.Define(stage, path)
                positions = _chunk_positions(Gf, count, rng)
                points.GetPointsAttr().Set(Vt.Vec3fArray(positions))
                points.GetVelocitiesAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(0.0, -0.10, 0.0)] * count)
                )
                points.GetWidthsAttr().Set(Vt.FloatArray([0.003] * count))
                set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
                particle_api = PhysxSchema.PhysxParticleAPI(set_api)
                particle_api.CreateParticleSystemRel().SetTargets([system_path])
                particle_api.CreateSelfCollisionAttr().Set(True)
                particle_api.CreateParticleEnabledAttr().Set(False)
                particle_api.CreateParticleGroupAttr().Set(0)
                set_api.CreateFluidAttr().Set(True)
                chunk_records.append((points.GetPrim(), particle_api, count, pour["tank_id"]))

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("F2 stage has no StageCache ID")
        simulation.attach_stage(stage_id)
        stage_attached = True
        for _ in range(2):
            app.update()
        _memory_sample(receipt)

        frames_dir = output / "frames"
        stages_dir = output / "stages"
        frames_dir.mkdir()
        stages_dir.mkdir()
        video_path = output / "three-pour.mp4"
        writer = cv2.VideoWriter(
            os.fspath(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.video_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v writer did not open")

        def capture(step: int, phase: str) -> tuple[Path, list[list[float]]]:
            invalid_reasons: list[str] = []
            rgb = None
            last_rejected_rgb = None
            for attempt in range(1, args.max_capture_tries + 1):
                check_time_limits()
                # delta_time=0.0 and pause_timeline=True deliberately retry the
                # exact same simulated state; only renderer convergence advances.
                _replicator_same_state_step(rep, rt_subframes=args.rt_subframes)
                raw = rgb_annotator.get_data()
                candidate = np.asarray(raw)
                if (
                    candidate.ndim != 3
                    or candidate.shape[0] != height
                    or candidate.shape[1] != width
                    or candidate.shape[2] < 3
                ):
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
                "frame_index": len(frame_paths) + 1,
                "physics_step": step,
                "phase": phase,
                "tries": tries,
                "invalid_reasons": invalid_reasons,
                "rejected_rgb_diagnostic": None,
            }
            receipt["render"]["capture_attempts"].append(attempt_record)
            if rgb is None:
                if last_rejected_rgb is not None:
                    diagnostic = _write_rejected_rgb_diagnostic(
                        cv2,
                        np,
                        output,
                        last_rejected_rgb,
                        step=step,
                        phase=phase,
                    )
                    attempt_record["rejected_rgb_diagnostic"] = diagnostic
                    receipt["render"]["rejected_rgb_diagnostics"].append(
                        diagnostic
                    )
                detail = invalid_reasons[-1] if invalid_reasons else "unknown"
                raise RuntimeError(
                    f"RGB capture failed after {args.max_capture_tries} tries: {detail}"
                )
            index = len(frame_paths) + 1
            path = frames_dir / f"frame-{index:04d}-step-{step:05d}-{phase}.png"
            if not cv2.imwrite(os.fspath(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"could not write RGB frame {path}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frame_paths.append(path)
            return path, _all_positions(active_prims)

        for _ in range(args.warmup):
            check_time_limits()
            _replicator_same_state_step(rep, rt_subframes=args.rt_subframes)
        capture(0, "prefill")

        dt = 1.0 / PHYSICS_HZ
        global_step = 0
        chunk_cursor = 0
        cumulative = 0
        for stage_index, pour in enumerate(plan["pours"], start=1):
            pre_memory = _memory_sample(receipt)
            stage_memory_samples = [pre_memory]
            pour_chunks = pour["emission_chunk_count"]
            for _ in range(pour_chunks):
                prim, particle_api, count, tank_id = chunk_records[chunk_cursor]
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
            stable_frame_path: Path | None = None
            stable_motion: dict[str, float] | None = None
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
                frame_path, current = capture(global_step, f"{pour['tank_id']}-settle")
                stage_memory_samples.append(_memory_sample(receipt))
                motion = _movement(previous, current)
                previous = current
                if (
                    settling_steps >= args.stable_min_steps
                    and motion is not None
                    and motion["mean_m"] <= args.stable_mean_displacement_m
                    and motion["max_m"] <= args.stable_max_displacement_m
                ):
                    stable_streak += 1
                else:
                    stable_streak = 0
                if stable_streak >= args.stable_consecutive_captures:
                    stable = True
                    stable_motion = motion
                    _set_stage_color(shader, Gf, STAGE_COLORS[stage_index - 1])
                    stable_frame_path, current = capture(
                        global_step, f"{pour['tank_id']}-stable-stage"
                    )
                    stage_memory_samples.append(_memory_sample(receipt))
                    break
            positions = _all_positions(active_prims)
            cumulative += pour["particle_count"]
            iso_path = system_path.AppendChild("Isosurface")
            iso_evidence = _isosurface_mesh_evidence(stage, iso_path)
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
                    "emission_end_step": global_step - settling_steps,
                    "settling_steps": settling_steps,
                    "finite_positions": _finite_positions(positions),
                    "escaped_particle_count": _escaped_count(positions),
                    "stable": stable,
                    "stable_motion": stable_motion,
                    "stable_frame_index": stable_index,
                    "stable_frame_path": (
                        str(stage_paths[-1].relative_to(output))
                        if stable_frame_path is not None
                        else None
                    ),
                    "stable_source_frame_path": (
                        str(stable_frame_path.relative_to(output))
                        if stable_frame_path is not None
                        else None
                    ),
                    "stable_frame_sha256": stable_digest,
                    "isosurface_nonempty": bool(
                        iso_evidence["mesh_point_count"] > 0
                        and iso_evidence["mesh_face_count"] > 0
                        and iso_evidence["mesh_finite_points"]
                        and iso_evidence["mesh_extent_nonempty"]
                    ),
                    "used_gpu_memory_mib_before": pre_memory,
                    "used_gpu_memory_mib_peak": peak_memory,
                    "used_gpu_memory_mib_after": final_memory,
                    "memory_recovered_during_settle": bool(
                        peak_memory - pre_memory < 4096 or final_memory < peak_memory
                    ),
                    "uncalibrated_stage_transmission_color": list(
                        STAGE_COLORS[stage_index - 1]
                    ),
                }
            )
            if not stable:
                raise RuntimeError(
                    f"{pour['tank_id']} did not meet the frozen stability threshold"
                )
            if receipt["pours"][-1]["escaped_particle_count"]:
                raise RuntimeError(f"{pour['tank_id']} has escaped particles")
            if not receipt["pours"][-1]["isosurface_nonempty"]:
                raise RuntimeError(f"{pour['tank_id']} has no nonempty isosurface")
            if peak_memory - pre_memory >= 4096 and final_memory >= peak_memory:
                raise RuntimeError(
                    f"{pour['tank_id']} isosurface memory rose >=4096 MiB without recovery"
                )

        systems = [
            prim
            for prim in stage.Traverse()
            if PhysxSchema.PhysxParticleSystem(prim)
        ]
        particle_prims = [
            prim for prim in stage.Traverse() if prim.HasAPI(PhysxSchema.PhysxParticleSetAPI)
        ]
        mass_api_count = sum(
            1 for prim in particle_prims if prim.HasAPI(UsdPhysics.MassAPI)
        )
        relation_ok = all(
            [str(path) for path in PhysxSchema.PhysxParticleAPI(
                PhysxSchema.PhysxParticleSetAPI(prim)
            ).GetParticleSystemRel().GetTargets()]
            == [str(system_path)]
            for prim in particle_prims
        )
        receipt["particle_system"].update(
            {
                "count": len(systems),
                "exactly_one": len(systems) == 1,
                "particle_set_count": len(particle_prims),
                "mass_api_count": mass_api_count,
                "all_particle_sets_target_one_system": relation_ok,
            }
        )
        final_positions = _all_positions(active_prims)
        receipt["totals"].update(
            {
                "emitted_particle_count": cumulative,
                "final_particle_count": len(final_positions),
            }
        )
        iso_path = system_path.AppendChild("Isosurface")
        receipt["isosurface"].update(_isosurface_mesh_evidence(stage, iso_path))
        receipt["memory"]["isosurface_enabled_seconds"] = (
            time.monotonic() - isosurface_started
        )
        check_time_limits()
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
                "stage_frame_paths": [
                    str(path.relative_to(output)) for path in stage_paths
                ],
                "stage_frame_sha256": stage_hashes,
                "video_path": str(video_path.relative_to(output)),
                "video_sha256": _sha256(video_path),
                "video_decodable": decodable,
                "video_decoded_frame_count": decoded_count,
                "stage_frames_are_video_subsets": bool(
                    len(stage_hashes) == 3
                    and set(stage_hashes).issubset(set(frame_hashes))
                ),
            }
        )
        receipt["timing"]["wall_seconds"] = time.monotonic() - started
        receipt["status"] = "pass"
        failures = validate_three_pour_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(
            f"F2 execution failed: {type(exc).__name__}: {exc}"
        )
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

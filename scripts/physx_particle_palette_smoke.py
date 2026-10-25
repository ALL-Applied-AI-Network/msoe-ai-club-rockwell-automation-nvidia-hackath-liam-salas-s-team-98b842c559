#!/usr/bin/env python3
"""Test whether Isaac 6.0.1 renders mutable per-particle prototype colours.

This is a bounded renderer-compatibility smoke, not a liquid-mixing or physical-
colour result.  A top-level visual review of the saved images is required.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.physx_three_pour import (
    CAMERA_PATH,
    SOURCE_LIQUID_PATH,
    SOURCE_STREAM_PATH,
    VIAL_CENTER_XY_M,
    build_twin_collider_specs,
)


NAMESPACE = "/FillStation/FluidmixPaletteSmoke"
PARTICLE_COUNT = 12
INITIAL_PROTO_INDICES = [0, 1, 2] * 4
TANK_COLORS = {
    "t200": (0.95, 0.38, 0.08),
    "t400": (0.18, 0.55, 0.75),
    "t100": (0.92, 0.98, 1.0),
}
ASYNC_SETTING = "/exts/isaacsim.core.throttling/enable_async"
EXTRA_ARGS = (f"--{ASYNC_SETTING}=false",)


def build_smoke_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "physx_particle_palette_smoke",
        "seed": 20261025,
        "particle_count": PARTICLE_COUNT,
        "particle_system_count": 1,
        "prototype_tanks": list(TANK_COLORS),
        "prototype_transmission_colors": {
            tank: list(color) for tank, color in TANK_COLORS.items()
        },
        "initial_proto_indices": list(INITIAL_PROTO_INDICES),
        "swapped_proto_indices": swap_orange_blue(INITIAL_PROTO_INDICES),
        "isaac_sim_version": "6.0.1",
        "frozen_time_code": 1.0,
        "physics_steps_before_swap": 30,
        "physics_steps_during_swap": 0,
        "physics_hz": 60,
        "renderer": "PathTracing",
        "resolution": [640, 480],
        "warmup_updates": 60,
        "rt_subframes": 64,
        "max_bounces": 16,
        "max_specular_and_transmission_bounces": 16,
        "async_throttling_enabled": False,
        "camera_path": CAMERA_PATH,
        "material_controls": {
            "specular_transmission_scattering_depth_m": 0.01,
            "thin_walled": False,
        },
        "visual_decision": "pending_top_level_review",
        "claim_limit": (
            "Renderer/API feasibility only; not smooth fluid, measured tank colour, "
            "mixing validation, physical accuracy, or sim-to-real evidence."
        ),
    }


def swap_orange_blue(indices: list[int]) -> list[int]:
    if any(type(value) is not int or value not in (0, 1, 2) for value in indices):
        raise ValueError("prototype index outside frozen palette")
    return [1 if value == 0 else 0 if value == 1 else 2 for value in indices]


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_technical_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    failures: list[str] = []
    if receipt.get("schema_version") != 1 or receipt.get("experiment") != "physx_particle_palette_smoke":
        failures.append("receipt_identity")
    if receipt.get("status") != "pending_visual":
        failures.append("status_must_remain_pending_visual")
    scene = receipt.get("scene_contract")
    scene = scene if isinstance(scene, Mapping) else {}
    frozen_time_code = scene.get("frozen_time_code")
    if (
        scene.get("default_prim") != "/FillStation"
        or scene.get("meters_per_unit") != 1.0
        or scene.get("up_axis") != "Z"
        or isinstance(frozen_time_code, bool)
        or not isinstance(frozen_time_code, (int, float))
        or not math.isclose(
            float(frozen_time_code), 1.0, rel_tol=0.0, abs_tol=1e-9
        )
        or scene.get("session_layer_only") is not True
        or scene.get("hidden_source_paths") != [SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH]
        or scene.get("custom_camera_count") != 0
        or scene.get("custom_light_count") != 0
        or scene.get("custom_backdrop_count") != 0
        or scene.get("forbidden_authored_paths") != []
    ):
        failures.append("twin_scene_contract")
    technical = receipt.get("technical")
    technical = technical if isinstance(technical, Mapping) else {}
    isaac_version = technical.get("isaac_sim_version")
    if not isinstance(isaac_version, str) or not isaac_version.startswith("6.0.1"):
        failures.append("isaac_version")
    if technical.get("cuda_device_available") is not True or technical.get("gpu_dynamics") is not True:
        failures.append("gpu_dynamics")
    if technical.get("particle_system_count") != 1:
        failures.append("particle_system_count")
    if technical.get("particle_count") != PARTICLE_COUNT or technical.get("prototype_count") != 3:
        failures.append("particle_or_prototype_count")
    if (
        technical.get("fluid_particle_set") is not True
        or technical.get("creation_api")
        != "omni.physx.scripts.particleUtils.add_physx_particleset_pointinstancer"
        or technical.get("particle_system_target_count") != 1
        or technical.get("sphere_prototype_count") != 3
    ):
        failures.append("pointinstancer_contract")
    if technical.get("prototype_material_bindings") != 3:
        failures.append("prototype_material_binding")
    expected_materials = [
        {
            "tank_id": tank,
            "specular_transmission_color": list(color),
            "specular_transmission_scattering_depth_m": 0.01,
            "thin_walled": False,
            "bound": True,
        }
        for tank, color in TANK_COLORS.items()
    ]
    if technical.get("material_variants") != expected_materials:
        failures.append("material_variant_contract")
    if (
        technical.get("collider_count") != len(build_twin_collider_specs())
        or technical.get("all_colliders_invisible") is not True
    ):
        failures.append("collider_contract")
    maximum_movement = technical.get("maximum_movement_m")
    if (
        technical.get("physics_steps_before_swap") != 30
        or technical.get("physics_steps_during_swap") != 0
        or technical.get("finite_positions") is not True
        or technical.get("moved_under_physics") is not True
        or isinstance(maximum_movement, bool)
        or not isinstance(maximum_movement, (int, float))
        or not math.isfinite(float(maximum_movement))
        or float(maximum_movement) <= 1e-6
    ):
        failures.append("finite_motion")
    if technical.get("indices_before") != INITIAL_PROTO_INDICES or technical.get("indices_after") != swap_orange_blue(INITIAL_PROTO_INDICES):
        failures.append("prototype_swap")
    if technical.get("positions_identical_across_swap") is not True:
        failures.append("swap_changed_positions")
    before_position_hash = technical.get("positions_before_sha256")
    after_position_hash = technical.get("positions_after_sha256")
    if (
        technical.get("position_bytes_identical_across_swap") is not True
        or not _valid_hash(before_position_hash)
        or before_position_hash != after_position_hash
    ):
        failures.append("swap_changed_position_bytes")
    render = receipt.get("render")
    render = render if isinstance(render, Mapping) else {}
    if (
        render.get("renderer") != "PathTracing"
        or render.get("camera_path") != CAMERA_PATH
        or render.get("resolution") != [640, 480]
        or render.get("async_throttling_enabled") is not False
        or render.get("warmup_updates") != 60
        or render.get("rt_subframes") != 64
        or render.get("max_bounces") != 16
        or render.get("max_specular_and_transmission_bounces") != 16
    ):
        failures.append("render_configuration")
    if (
        render.get("before_readable") is not True
        or render.get("after_readable") is not True
        or render.get("before_nonconstant") is not True
        or render.get("after_nonconstant") is not True
        or any(
            not _valid_hash(render.get(key))
            for key in (
                "before_sha256",
                "after_sha256",
                "before_crop_sha256",
                "after_crop_sha256",
                "contact_pair_sha256",
            )
        )
    ):
        failures.append("render_missing_or_constant")
    visual = receipt.get("visual_review")
    visual = visual if isinstance(visual, Mapping) else {}
    if visual.get("status") != "pending_top_level_review":
        failures.append("visual_review_not_pending")
    errors = receipt.get("errors")
    if not isinstance(errors, list) or errors:
        failures.append("run_error_recorded")
    return tuple(dict.fromkeys(failures))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.stage.is_absolute():
        raise ValueError("stage must be an absolute path")
    if not args.stage.is_file():
        raise ValueError("stage must be an existing file")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("seed must be an integer")


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


def _empty_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "physx_particle_palette_smoke",
        "status": "insufficient_evidence",
        "plan": build_smoke_plan(),
        "arguments": {"stage": str(args.stage), "seed": args.seed},
        "provenance": {
            "argv": list(sys.argv),
            "stage_sha256": _sha256(args.stage),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "scene_contract": {
            "default_prim": None,
            "meters_per_unit": None,
            "up_axis": None,
            "frozen_time_code": None,
            "session_layer_only": False,
            "hidden_source_paths": [],
            "custom_camera_count": None,
            "custom_light_count": None,
            "custom_backdrop_count": None,
            "forbidden_authored_paths": None,
        },
        "technical": {
            "isaac_sim_version": None,
            "cuda_device_available": False,
            "gpu_dynamics": False,
            "particle_system_count": 0,
            "particle_count": 0,
            "fluid_particle_set": False,
            "creation_api": None,
            "particle_system_target_count": 0,
            "prototype_count": 0,
            "sphere_prototype_count": 0,
            "prototype_material_bindings": 0,
            "material_variants": [],
            "collider_count": 0,
            "all_colliders_invisible": False,
            "physics_steps_before_swap": 0,
            "physics_steps_during_swap": None,
            "finite_positions": False,
            "moved_under_physics": False,
            "maximum_movement_m": None,
            "indices_before": [],
            "indices_after": [],
            "positions_identical_across_swap": False,
            "position_bytes_identical_across_swap": False,
            "positions_before_sha256": None,
            "positions_after_sha256": None,
            "positions_before_swap": [],
            "positions_after_swap": [],
        },
        "render": {
            "renderer": "PathTracing",
            "camera_path": CAMERA_PATH,
            "resolution": [640, 480],
            "async_throttling_enabled": None,
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_bounces": 16,
            "max_specular_and_transmission_bounces": 16,
            "before_path": None,
            "before_sha256": None,
            "after_path": None,
            "after_sha256": None,
            "before_crop_path": None,
            "before_crop_sha256": None,
            "after_crop_path": None,
            "after_crop_sha256": None,
            "contact_pair_path": None,
            "contact_pair_sha256": None,
            "before_readable": False,
            "after_readable": False,
            "before_nonconstant": False,
            "after_nonconstant": False,
        },
        "visual_review": {
            "status": "pending_top_level_review",
            "criteria": [
                "three prototype colours visibly distinct",
                "orange and blue visibly swap on the same particles",
                "particle rendering readable enough to justify the bead fallback",
            ],
        },
        "errors": [],
        "warnings": [],
    }


def _author_omnisurface(stage: Any, path: str, color: tuple[float, float, float], Sdf: Any, Gf: Any, UsdShade: Any) -> Any:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/OmniSurface")
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    values = {
        "enable_specular_transmission": (Sdf.ValueTypeNames.Bool, True),
        "specular_transmission_weight": (Sdf.ValueTypeNames.Float, 1.0),
        "specular_transmission_color": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
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
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    material.CreateVolumeOutput("mdl").ConnectToSource(output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(output)
    return material, shader


def _initial_positions(Gf: Any) -> list[Any]:
    positions = []
    for index in range(PARTICLE_COUNT):
        column = index % 4
        row = (index // 4) % 3
        positions.append(
            Gf.Vec3f(
                VIAL_CENTER_XY_M[0] + (column - 1.5) * 0.004,
                VIAL_CENTER_XY_M[1] + (row - 1.0) * 0.004,
                1.035 + (index // 6) * 0.008,
            )
        )
    return positions


def _finite_positions(values: Any) -> bool:
    return bool(values) and all(
        math.isfinite(float(component)) for row in values for component in row
    )


def _capture_rgb(rep: Any, annotator: Any, np: Any, *, tries: int = 20) -> Any:
    for _ in range(tries):
        rep.orchestrator.step(
            delta_time=0.0,
            pause_timeline=True,
            rt_subframes=64,
            wait_for_render=True,
        )
        candidate = np.asarray(annotator.get_data())
        if candidate.ndim == 3 and candidate.shape[:2] == (480, 640) and candidate.shape[2] >= 3:
            candidate = np.asarray(candidate[..., :3], dtype=np.uint8)
            if np.any(candidate != candidate[0, 0]):
                return candidate
    raise RuntimeError("RGB capture remained missing or spatially constant after 20 tries")


def run_smoke(args: argparse.Namespace, app: Any, output: Path) -> dict[str, Any]:
    receipt = _empty_receipt(args)
    simulation = None
    attached = False
    annotator = None
    render_product = None
    try:
        import carb
        import cv2
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from omni.physx.scripts import particleUtils
        from omni.physxfoundation import get_physx_foundation_interface
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

        settings = carb.settings.get_settings()
        receipt["technical"]["isaac_sim_version"] = version("isaacsim")
        if not receipt["technical"]["isaac_sim_version"].startswith("6.0.1"):
            raise RuntimeError(
                "palette smoke is frozen to Isaac Sim 6.0.1, observed "
                f"{receipt['technical']['isaac_sim_version']}"
            )
        receipt["render"]["async_throttling_enabled"] = settings.get_as_bool(
            ASYNC_SETTING
        )
        if receipt["render"]["async_throttling_enabled"] is not False:
            raise RuntimeError("Replicator asynchronous throttling is enabled")
        settings.set("/rtx/pathtracing/maxBounces", 16)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
        receipt["render"]["max_bounces"] = settings.get_as_int(
            "/rtx/pathtracing/maxBounces"
        )
        receipt["render"][
            "max_specular_and_transmission_bounces"
        ] = settings.get_as_int(
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces"
        )
        if (
            receipt["render"]["max_bounces"] != 16
            or receipt["render"]["max_specular_and_transmission_bounces"] != 16
        ):
            raise RuntimeError("path-tracing bounce settings were not retained")
        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(args.stage)):
            raise RuntimeError("could not open canonical twin stage")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not return the opened twin stage")
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
            raise RuntimeError("opened stage is not the reviewed FillStation twin")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0.0:
            raise RuntimeError("canonical twin has invalid timeCodesPerSecond")
        timeline.set_current_time(1.0 / time_codes_per_second)
        app.update()
        frozen_time_code = float(timeline.get_current_time()) * time_codes_per_second
        receipt["scene_contract"]["frozen_time_code"] = frozen_time_code
        if not math.isclose(frozen_time_code, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("canonical twin did not freeze at time code 1")
        camera = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera.IsValid() or not camera.IsA(UsdGeom.Camera):
            raise RuntimeError("reviewed FillCam is missing")

        foundation = get_physx_foundation_interface()
        receipt["technical"]["cuda_device_available"] = bool(foundation.cuda_device_check())
        if not receipt["technical"]["cuda_device_available"]:
            raise RuntimeError("GPU particles require a CUDA device")
        simulation = omni.physx.get_physx_simulation_interface()
        from omni.physx.bindings import _physx as physx_bindings

        settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
        particle_update = getattr(physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None)
        if particle_update is None:
            raise RuntimeError("PhysX lacks particle-to-USD updates")
        settings.set(particle_update, True)

        session = stage.GetSessionLayer()
        collider_prims = []
        material_records = []
        with Usd.EditContext(stage, session):
            for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
                source_prim = stage.GetPrimAtPath(path)
                if not source_prim.IsValid():
                    raise RuntimeError(f"canonical twin source prim is missing: {path}")
                UsdGeom.Imageable(source_prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
            UsdGeom.Xform.Define(stage, NAMESPACE)
            scene = UsdPhysics.Scene.Define(stage, f"{NAMESPACE}/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            receipt["technical"]["gpu_dynamics"] = bool(
                scene_api.GetEnableGPUDynamicsAttr().Get()
            ) and str(scene_api.GetBroadphaseTypeAttr().Get()) == "GPU"

            system_path = Sdf.Path(f"{NAMESPACE}/ParticleSystem")
            system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateParticleContactOffsetAttr().Set(0.003)
            system.CreateFluidRestOffsetAttr().Set(0.0015)
            system.CreateContactOffsetAttr().Set(0.003)
            system.CreateRestOffsetAttr().Set(0.0029)
            system.CreateSolidRestOffsetAttr().Set(0.0029)
            pbd_material = UsdShade.Material.Define(stage, f"{NAMESPACE}/PBDMaterial")
            pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(pbd_material.GetPrim())
            pbd.CreateDensityAttr().Set(1000.0)
            UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(
                pbd_material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

            for spec in build_twin_collider_specs():
                smoke_path = str(spec["path"]).replace(
                    "/FillStation/FluidmixF2", NAMESPACE
                )
                cube = UsdGeom.Cube.Define(stage, smoke_path)
                cube.CreateSizeAttr().Set(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*spec["center_m"]))
                cube.AddRotateZOp().Set(float(spec["rotate_z_degrees"]))
                cube.AddScaleOp().Set(Gf.Vec3f(*spec["size_m"]))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                collider_prims.append(cube.GetPrim())

            initial_positions = _initial_positions(Gf)
            initial_velocities = [Gf.Vec3f(0.0, 0.0, -0.05)] * PARTICLE_COUNT
            instancer_path = Sdf.Path(f"{NAMESPACE}/Particles")
            instancer_prim = particleUtils.add_physx_particleset_pointinstancer(
                stage=stage,
                path=instancer_path,
                positions=initial_positions,
                velocities=initial_velocities,
                particle_system_path=system_path,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=0.0,
                density=1000.0,
                num_prototypes=3,
                prototype_indices=list(INITIAL_PROTO_INDICES),
            )
            bindings = 0
            for ordinal, (tank, color) in enumerate(TANK_COLORS.items()):
                prototype_path = f"{instancer_path}/particlePrototype{ordinal}"
                sphere = UsdGeom.Sphere.Get(stage, prototype_path)
                sphere.CreateRadiusAttr().Set(0.0025)
                material, shader = _author_omnisurface(
                    stage,
                    f"{NAMESPACE}/Materials/{tank}",
                    color,
                    Sdf,
                    Gf,
                    UsdShade,
                )
                binding = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
                binding.Bind(material)
                bound = bool(binding.GetDirectBinding().GetMaterial())
                bindings += bound
                effective_color = shader.GetInput(
                    "specular_transmission_color"
                ).Get()
                if any(
                    not math.isclose(
                        float(observed), expected, rel_tol=0.0, abs_tol=1e-6
                    )
                    for observed, expected in zip(effective_color, color)
                ):
                    raise RuntimeError(f"{tank} transmission color was not retained")
                effective_depth = float(
                    shader.GetInput(
                        "specular_transmission_scattering_depth"
                    ).Get()
                )
                effective_thin_walled = bool(
                    shader.GetInput("thin_walled").Get()
                )
                if not math.isclose(
                    effective_depth, 0.01, rel_tol=0.0, abs_tol=1e-8
                ) or effective_thin_walled:
                    raise RuntimeError(f"{tank} nonthin depth controls were not retained")
                material_records.append(
                    {
                        "tank_id": tank,
                        "specular_transmission_color": list(color),
                        "specular_transmission_scattering_depth_m": 0.01,
                        "thin_walled": False,
                        "bound": bound,
                    }
                )
            receipt["technical"]["prototype_material_bindings"] = int(bindings)

        hidden_paths = []
        for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH):
            property_path = Sdf.Path(path).AppendProperty("visibility")
            session_spec = session.GetPropertyAtPath(property_path)
            visibility = UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Get(
                Usd.TimeCode(1.0)
            )
            if session_spec and visibility == UsdGeom.Tokens.invisible:
                hidden_paths.append(path)
        namespace_prims = [
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(NAMESPACE)
        ]
        forbidden = ["/World"] if session.GetPrimAtPath("/World") else []
        receipt["scene_contract"].update(
            {
                "session_layer_only": bool(session.GetPrimAtPath(NAMESPACE))
                and not bool(stage.GetRootLayer().GetPrimAtPath(NAMESPACE)),
                "hidden_source_paths": hidden_paths,
                "custom_camera_count": sum(
                    prim.IsA(UsdGeom.Camera) for prim in namespace_prims
                ),
                "custom_light_count": sum(
                    prim.GetTypeName().endswith("Light") for prim in namespace_prims
                ),
                "custom_backdrop_count": sum(
                    "backdrop" in str(prim.GetPath()).lower()
                    for prim in namespace_prims
                ),
                "forbidden_authored_paths": forbidden,
            }
        )
        particle_set_api = PhysxSchema.PhysxParticleSetAPI(instancer_prim)
        particle_api = PhysxSchema.PhysxParticleAPI(particle_set_api)
        instancer_for_contract = UsdGeom.PointInstancer(instancer_prim)
        prototype_targets = instancer_for_contract.GetPrototypesRel().GetTargets()
        receipt["technical"].update(
            {
                "creation_api": "omni.physx.scripts.particleUtils.add_physx_particleset_pointinstancer",
                "fluid_particle_set": bool(particle_set_api.GetFluidAttr().Get()),
                "particle_system_target_count": len(
                    particle_api.GetParticleSystemRel().GetTargets()
                ),
                "sphere_prototype_count": sum(
                    stage.GetPrimAtPath(path).IsA(UsdGeom.Sphere)
                    for path in prototype_targets
                ),
                "material_variants": material_records,
                "collider_count": len(collider_prims),
                "all_colliders_invisible": all(
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
                    == UsdGeom.Tokens.invisible
                    for prim in collider_prims
                ),
            }
        )

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("opened twin has no stage-cache ID")
        simulation.attach_stage(stage_id)
        attached = True
        for _ in range(2):
            app.update()
        initial_array = np.asarray(
            [[float(component) for component in row] for row in initial_positions],
            dtype=np.float64,
        )
        dt = 1.0 / 60.0
        for step in range(1, 31):
            simulation.simulate(dt, step * dt)
            simulation.fetch_results()
            app.update()
        receipt["technical"]["physics_steps_before_swap"] = 30

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(CAMERA_PATH, (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(60):
            app.update()
        receipt["render"]["warmup_updates"] = 60

        instancer = UsdGeom.PointInstancer(instancer_prim)
        before = _capture_rgb(rep, annotator, np)
        positions_before_value = instancer.GetPositionsAttr().Get()
        positions_before = np.asarray(
            [[float(component) for component in row] for row in positions_before_value],
            dtype=np.float64,
        )
        indices_before = [int(value) for value in instancer.GetProtoIndicesAttr().Get()]
        before_path = output / "before-swap.png"
        if not cv2.imwrite(os.fspath(before_path), cv2.cvtColor(before, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write before-swap image")

        indices_after_expected = swap_orange_blue(indices_before)
        if not instancer.GetProtoIndicesAttr().Set(indices_after_expected):
            raise RuntimeError("could not author swapped prototype indices")
        app.update()
        after = _capture_rgb(rep, annotator, np)
        positions_after_value = instancer.GetPositionsAttr().Get()
        positions_after = np.asarray(
            [[float(component) for component in row] for row in positions_after_value],
            dtype=np.float64,
        )
        receipt["technical"]["physics_steps_during_swap"] = 0
        after_path = output / "after-swap.png"
        if not cv2.imwrite(os.fspath(after_path), cv2.cvtColor(after, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write after-swap image")

        crop_slice = (slice(80, 465), slice(100, 430))
        before_crop = before[crop_slice]
        after_crop = after[crop_slice]
        before_crop_path = output / "before-swap-crop.png"
        after_crop_path = output / "after-swap-crop.png"
        if not cv2.imwrite(os.fspath(before_crop_path), cv2.cvtColor(before_crop, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write before crop")
        if not cv2.imwrite(os.fspath(after_crop_path), cv2.cvtColor(after_crop, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write after crop")
        contact_pair = np.concatenate([before_crop, after_crop], axis=1)
        contact_pair_path = output / "swap-contact-pair.png"
        if not cv2.imwrite(
            os.fspath(contact_pair_path),
            cv2.cvtColor(contact_pair, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError("could not write swap contact pair")
        before_reopened = cv2.imread(os.fspath(before_path), cv2.IMREAD_COLOR)
        after_reopened = cv2.imread(os.fspath(after_path), cv2.IMREAD_COLOR)
        before_readable = (
            before_reopened is not None and before_reopened.shape[:2] == (480, 640)
        )
        after_readable = (
            after_reopened is not None and after_reopened.shape[:2] == (480, 640)
        )

        systems = [
            prim for prim in stage.Traverse() if PhysxSchema.PhysxParticleSystem(prim)
        ]
        movement = np.linalg.norm(positions_before - initial_array, axis=1)
        receipt["technical"].update(
            {
                "particle_system_count": len(systems),
                "particle_count": int(len(positions_before)),
                "prototype_count": len(instancer.GetPrototypesRel().GetTargets()),
                "finite_positions": bool(
                    _finite_positions(positions_before_value)
                    and _finite_positions(positions_after_value)
                ),
                "moved_under_physics": bool(np.max(movement) > 1e-6),
                "maximum_movement_m": float(np.max(movement)),
                "indices_before": indices_before,
                "indices_after": [
                    int(value) for value in instancer.GetProtoIndicesAttr().Get()
                ],
                "positions_identical_across_swap": bool(
                    np.array_equal(positions_before, positions_after)
                ),
                "position_bytes_identical_across_swap": bool(
                    positions_before.tobytes(order="C")
                    == positions_after.tobytes(order="C")
                ),
                "positions_before_sha256": hashlib.sha256(
                    positions_before.tobytes(order="C")
                ).hexdigest(),
                "positions_after_sha256": hashlib.sha256(
                    positions_after.tobytes(order="C")
                ).hexdigest(),
                "positions_before_swap": positions_before.tolist(),
                "positions_after_swap": positions_after.tolist(),
            }
        )
        receipt["render"].update(
            {
                "before_path": before_path.name,
                "before_sha256": _sha256(before_path),
                "after_path": after_path.name,
                "after_sha256": _sha256(after_path),
                "before_crop_path": before_crop_path.name,
                "before_crop_sha256": _sha256(before_crop_path),
                "after_crop_path": after_crop_path.name,
                "after_crop_sha256": _sha256(after_crop_path),
                "contact_pair_path": contact_pair_path.name,
                "contact_pair_sha256": _sha256(contact_pair_path),
                "before_readable": bool(before_readable),
                "after_readable": bool(after_readable),
                "before_nonconstant": bool(np.any(before != before[0, 0])),
                "after_nonconstant": bool(np.any(after != after[0, 0])),
            }
        )
        receipt["status"] = "pending_visual"
        failures = validate_technical_receipt(receipt)
        if failures:
            receipt["status"] = "insufficient_evidence"
            receipt["errors"].extend(failures)
    except Exception as exc:
        receipt["status"] = "insufficient_evidence"
        receipt["errors"].append(
            f"palette smoke failed: {type(exc).__name__}: {exc}"
        )
    finally:
        if annotator is not None and render_product is not None:
            try:
                annotator.detach([render_product])
            except Exception as exc:
                receipt["warnings"].append(f"annotator detach failed: {exc}")
        if attached and simulation is not None:
            try:
                simulation.detach_stage()
            except Exception as exc:
                receipt["warnings"].append(f"stage detach failed: {exc}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    from isaacsim import SimulationApp

    kit = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "PathTracing",
            "width": 640,
            "height": 480,
            "extra_args": list(EXTRA_ARGS),
        }
    )
    exit_code = 2
    try:
        import omni.kit.app

        receipt = run_smoke(args, omni.kit.app.get_app(), args.out)
        _write_json(args.out / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if receipt["status"] == "pending_visual" else 2
    finally:
        kit.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

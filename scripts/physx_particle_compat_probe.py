#!/usr/bin/env python3
"""Run the bounded, GPU-only F1 PhysX particle compatibility probe.

The script starts at most one ``SimulationApp`` when it is not already running
and never selects a CPU fallback. Under ordinary Python it can still parse
arguments and emit an explicit ``unsupported`` receipt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any


MIN_STEPS = 1
MAX_STEPS = 120
DEFAULT_STEPS = 8
DEFAULT_SEED = 20261025


def _bounded_steps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be an integer") from exc
    if not MIN_STEPS <= parsed <= MAX_STEPS:
        raise argparse.ArgumentTypeError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    return parsed


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="JSON receipt path")
    parser.add_argument(
        "--steps",
        type=_bounded_steps,
        default=DEFAULT_STEPS,
        help=f"fixed physics frames ({MIN_STEPS}-{MAX_STEPS}, default: {DEFAULT_STEPS})",
    )
    parser.add_argument("--seed", type=_seed, default=DEFAULT_SEED, help="deterministic seed")
    return parser


def empty_record(out: Path, *, steps: int, seed: int) -> dict[str, Any]:
    """Return the stable receipt envelope used for success and explicit failures."""
    return {
        "probe": "physx_particle_compatibility_f1",
        "status": "insufficient_evidence",
        "arguments": {"out": str(out), "steps": steps, "seed": seed},
        "versions": {"isaac": None, "kit": None, "physx": None},
        "extensions": {},
        "gpu_simulation": {
            "scene_enable_gpu_dynamics": None,
            "broadphase": None,
            "cuda_device_available": None,
            "simulation_interface": None,
            "stage_attached": None,
            "update_to_usd": None,
            "update_particles_to_usd": None,
        },
        "schema_api": {
            "PhysxParticleSystem": None,
            "PhysxParticleSetAPI": None,
            "PhysxParticleAPI": None,
            "PhysxPBDMaterialAPI": None,
            "PhysxParticleIsosurfaceAPI": None,
        },
        "particle_set": {
            "path": "/World/FluidParticles",
            "kind": "Points",
            "count_before": None,
            "count_after": None,
            "finite_before": None,
            "finite_after": None,
            "mean_position_before": None,
            "mean_position_after": None,
            "movement": None,
            "gravity_moved_down": None,
            "box_contained_after": None,
            "system_relation": None,
            "mass_api_present": None,
        },
        "particle_system": {
            "count": None,
            "path": "/World/ParticleSystem",
            "exactly_one": None,
            "simulation_owner": None,
            "pbd_material_binding": None,
            "render_material_binding": None,
            "render_material_path": None,
        },
        "isosurface": {
            "api_present": None,
            "enabled": None,
            "prim_path": None,
            "prim_present": None,
            "mesh_is_mesh": None,
            "mesh_point_count": None,
            "mesh_face_count": None,
            "mesh_finite_points": None,
            "mesh_extent_nonempty": None,
        },
        "timing": {
            "wall_seconds": None,
            "steps": steps,
            "fixed_dt_seconds": 1.0 / 60.0,
            "simulation_route": None,
        },
        "errors": [],
        "warnings": [],
        "recommended_brev_command": (
            "brev exec claude-isaac-smoke-1024 "
            "\"cd ~/mixture && OMNI_KIT_ACCEPT_EULA=YES "
            "/home/ubuntu/isaac/.venv/bin/python scripts/physx_particle_compat_probe.py "
            "--out outputs/f1_particle_probe.json --steps 8 --seed 20261025\""
        ),
    }


def _truthy(value: Any) -> bool:
    return value is True


def validate_record(record: dict[str, Any]) -> list[str]:
    """Validate receipt shape and reject an unproven pass.

    This is deliberately conservative: a receipt with a missing field is not a
    pass, while an unsupported/insufficient-evidence receipt only needs a reason.
    """
    errors: list[str] = []
    required = {"probe", "status", "arguments", "versions", "extensions", "gpu_simulation", "schema_api", "particle_set", "particle_system", "isosurface", "timing", "errors"}
    errors.extend(f"missing receipt field: {key}" for key in sorted(required - set(record)))
    status = record.get("status")
    if status not in {"pass", "unsupported", "insufficient_evidence"}:
        errors.append("status must be pass, unsupported, or insufficient_evidence")
    if status in {"unsupported", "insufficient_evidence"} and not record.get("errors"):
        errors.append(f"{status} receipt requires at least one error")
    if status == "pass":
        p = record.get("particle_set", {})
        s = record.get("particle_system", {})
        i = record.get("isosurface", {})
        g = record.get("gpu_simulation", {})
        versions = record.get("versions", {})
        extensions = record.get("extensions", {})
        schema = record.get("schema_api", {})
        required_truths = (
            all(versions.get(key) is not None for key in ("isaac", "kit", "physx")),
            extensions.get("omni.physx") is True,
            all(schema.get(key) is True for key in schema),
            g.get("scene_enable_gpu_dynamics"),
            g.get("simulation_interface"),
            g.get("stage_attached"),
            g.get("broadphase") == "GPU",
            g.get("cuda_device_available"),
            g.get("update_to_usd"),
            g.get("update_particles_to_usd"),
            p.get("finite_before"),
            p.get("finite_after"),
            p.get("gravity_moved_down"),
            p.get("box_contained_after"),
            p.get("system_relation"),
            p.get("mass_api_present") is False,
            s.get("exactly_one"),
            s.get("simulation_owner"),
            s.get("pbd_material_binding"),
            s.get("render_material_binding"),
            i.get("api_present"),
            i.get("enabled"),
            i.get("prim_present"),
            i.get("mesh_is_mesh"),
            (i.get("mesh_point_count") or 0) > 0,
            (i.get("mesh_face_count") or 0) > 0,
            i.get("mesh_finite_points"),
            i.get("mesh_extent_nonempty"),
            bool(record.get("timing", {}).get("simulation_route")),
        )
        if record.get("errors") or not all(_truthy(value) for value in required_truths):
            errors.append("pass requires all required evidence")
        if (
            not all(
                _truthy(i.get(key))
                for key in ("mesh_is_mesh", "mesh_finite_points", "mesh_extent_nonempty")
            )
            or not (i.get("mesh_point_count") or 0) > 0
            or not (i.get("mesh_face_count") or 0) > 0
        ):
            errors.append("pass requires nonempty finite isosurface mesh evidence")
    return errors


def _safe_call(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:
        return default


def _version_info(app: Any) -> dict[str, Any]:
    versions: dict[str, Any] = {"isaac": None, "kit": None, "physx": None}
    versions["kit"] = _safe_call(app, "get_build_version") or _safe_call(app, "get_version")
    try:
        import isaacsim

        versions["isaac"] = getattr(isaacsim, "__version__", None)
    except Exception:
        pass
    try:
        import importlib.metadata as metadata

        for dist_name in ("isaacsim", "isaacsim-kernel", "omni-physx"):
            try:
                value = metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                continue
            if dist_name == "omni-physx":
                versions["physx"] = value
            else:
                versions["isaac"] = value
    except Exception:
        pass
    try:
        manager = app.get_extension_manager()
        extension_id = manager.get_enabled_extension_id("omni.physx")
        extension = manager.get_extension_dict(extension_id) if extension_id else None
        if isinstance(extension, dict):
            package = extension.get("package")
            package_version = package.get("version") if isinstance(package, dict) else None
            versions["physx"] = (
                extension.get("version")
                or extension.get("packageVersion")
                or package_version
            )
    except Exception:
        pass
    return versions


def _positions(points_prim: Any, *, instancer: bool = False) -> list[list[float]]:
    from pxr import UsdGeom

    geom = UsdGeom.PointInstancer(points_prim) if instancer else UsdGeom.Points(points_prim)
    attr = geom.GetPositionsAttr() if instancer else geom.GetPointsAttr()
    values = attr.Get() or []
    return [[float(point[0]), float(point[1]), float(point[2])] for point in values]


def _finite_positions(values: list[list[float]]) -> bool:
    return bool(values) and all(math.isfinite(component) for point in values for component in point)


def _mean_position(values: list[list[float]]) -> list[float] | None:
    if not values:
        return None
    return [sum(point[i] for point in values) / len(values) for i in range(3)]


def _movement(before: list[list[float]], after: list[list[float]]) -> dict[str, float] | None:
    if len(before) != len(after) or not before:
        return None
    distances = [math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) for a, b in zip(before, after)]
    return {"max_m": max(distances), "mean_m": sum(distances) / len(distances)}


def _isosurface_mesh_evidence(stage: Any, path: Any) -> dict[str, Any]:
    """Inspect the deferred isosurface as a real, finite, nonempty Mesh."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    points = (mesh.GetPointsAttr().Get() or []) if mesh else []
    face_counts = (mesh.GetFaceVertexCountsAttr().Get() or []) if mesh else []
    point_values = [[float(point[0]), float(point[1]), float(point[2])] for point in points]
    finite_points = _finite_positions(point_values)
    extent = mesh.GetExtentAttr().Get() if mesh else None
    if not extent and point_values:
        extent = UsdGeom.PointBased.ComputeExtent(points)
    finite_extent = bool(extent) and all(
        math.isfinite(float(component)) for bound in extent for component in bound
    )
    extent_nonempty = bool(
        finite_extent
        and all(float(extent[1][axis]) > float(extent[0][axis]) for axis in range(3))
    )
    return {
        "prim_present": bool(prim),
        "mesh_is_mesh": bool(mesh),
        "mesh_point_count": len(point_values),
        "mesh_face_count": len(face_counts),
        "mesh_finite_points": finite_points,
        "mesh_extent_nonempty": extent_nonempty,
    }


def _extension_state(manager: Any, name: str) -> Any:
    try:
        return bool(manager.is_extension_enabled(name))
    except Exception:
        return None


def _author_preview_surface(stage: Any, material: Any, Sdf: Any, Gf: Any, UsdShade: Any) -> Any:
    """Author one valid preview-surface output on the shared PBD material."""

    shader = UsdShade.Shader.Define(stage, "/World/PBDMaterial/UsdPreviewSurface")
    shader.CreateIdAttr().Set("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.18, 0.55, 0.75)
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)
    return shader


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run one stage and one PhysX particle system using a retained Kit app."""
    record = empty_record(args.out, steps=args.steps, seed=args.seed)
    started = time.monotonic()
    simulation = None
    stage_attached = False
    try:
        # These imports are intentionally inside the runner: plain Python --help
        # and CPU contract tests must not require Isaac/Kit or pxr.
        import carb
        import omni.kit.app
        import omni.physx
        from omni.physxfoundation import get_physx_foundation_interface
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt
    except Exception as exc:
        record["status"] = "unsupported"
        record["errors"].append(f"Isaac/Kit particle runtime unavailable: {type(exc).__name__}: {exc}")
        record["timing"]["wall_seconds"] = time.monotonic() - started
        return record

    try:
        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()
        record["versions"] = _version_info(app)
        record["extensions"] = {name: _extension_state(manager, name) for name in ("omni.physx", "omni.physics", "omni.usd")}
        if record["extensions"].get("omni.physx") is not True:
            record["status"] = "unsupported"
            record["errors"].append("omni.physx extension is not enabled")
            return record

        simulation = omni.physx.get_physx_simulation_interface()
        foundation = get_physx_foundation_interface()
        record["gpu_simulation"]["simulation_interface"] = simulation is not None
        record["gpu_simulation"]["cuda_device_available"] = bool(foundation.cuda_device_check())
        if not record["gpu_simulation"]["cuda_device_available"]:
            record["status"] = "unsupported"
            record["errors"].append("GPU particle simulation requires a CUDA device")
            return record

        # Attach the fresh stage to the active USD context so PhysX's stage
        # update and GPU particle synchronization operate on the same stage.
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd context did not create a fresh stage")
        app.update()
        try:
            from omni.physx.bindings import _physx as physx_bindings

            settings = carb.settings.get_settings()
            settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
            record["gpu_simulation"]["update_to_usd"] = bool(
                settings.get_as_bool(physx_bindings.SETTING_UPDATE_TO_USD)
            )
            particles_setting = getattr(physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None)
            if particles_setting is not None:
                settings.set(particles_setting, True)
                record["gpu_simulation"]["update_particles_to_usd"] = bool(
                    settings.get_as_bool(particles_setting)
                )
        except Exception as exc:
            record["warnings"].append(f"could not set PhysX update-to-USD setting: {exc}")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetTimeCodesPerSecond(60.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        scene_api.CreateSolverTypeAttr().Set("TGS")
        record["gpu_simulation"]["scene_enable_gpu_dynamics"] = bool(scene_api.GetEnableGPUDynamicsAttr().Get())
        record["gpu_simulation"]["broadphase"] = str(scene_api.GetBroadphaseTypeAttr().Get())

        system_path = Sdf.Path("/World/ParticleSystem")
        system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
        system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
        system.CreateParticleSystemEnabledAttr().Set(True)
        system.CreateParticleContactOffsetAttr().Set(0.03)
        system.CreateFluidRestOffsetAttr().Set(0.015)
        system.CreateContactOffsetAttr().Set(0.03)
        system.CreateRestOffsetAttr().Set(0.029)
        system.CreateSolidRestOffsetAttr().Set(0.029)

        material_path = Sdf.Path("/World/PBDMaterial")
        material = UsdShade.Material.Define(stage, material_path)
        pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
        pbd.CreateDensityAttr().Set(1000.0)
        pbd.CreateFrictionAttr().Set(0.1)
        binding_api = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
        binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        record["particle_system"]["pbd_material_binding"] = bool(binding_api.GetDirectBinding("physics").GetMaterial())

        # The same material carries the PBD API and a render-output placeholder;
        # there is one render-material binding, rather than a second material or
        # a second particle system.  Isosurface extraction consumes this binding.
        _author_preview_surface(stage, material, Sdf, Gf, UsdShade)
        render_binding = UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
        render_binding.Bind(material)
        record["particle_system"]["render_material_binding"] = bool(render_binding.GetDirectBinding().GetMaterial())
        record["particle_system"]["render_material_path"] = str(material_path)

        # Box-only basin: floor and four walls. No cylinder, cone, decorative mesh,
        # or mesh/SDF collider is introduced by this probe.
        def box(path: str, center: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
            cube.CreateSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3f(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        box("/World/Basin/Floor", (0.0, -0.01, 0.0), (0.30, 0.02, 0.30))
        box("/World/Basin/WallXMin", (-0.15, 0.15, 0.0), (0.02, 0.32, 0.30))
        box("/World/Basin/WallXMax", (0.15, 0.15, 0.0), (0.02, 0.32, 0.30))
        box("/World/Basin/WallZMin", (0.0, 0.15, -0.15), (0.30, 0.32, 0.02))
        box("/World/Basin/WallZMax", (0.0, 0.15, 0.15), (0.30, 0.32, 0.02))

        # Small fluid Points set. We explicitly avoid MassAPI: material density
        # must remain the governing mass source per the frozen F1 contract.
        rng = random.Random(args.seed)
        positions: list[Any] = []
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    jitter = (rng.random() - 0.5) * 0.001
                    positions.append(Gf.Vec3f(-0.045 + ix * 0.03 + jitter, 0.075 + iy * 0.03, -0.045 + iz * 0.03 - jitter))
        points = UsdGeom.Points.Define(stage, "/World/FluidParticles")
        points.GetPointsAttr().Set(Vt.Vec3fArray(positions))
        points.GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(positions)))
        points.GetWidthsAttr().Set(Vt.FloatArray([0.03] * len(positions)))
        particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
        # PhysxParticleSetAPI carries the particle API in the retained 110.1
        # schema; wrapping it mirrors NVIDIA's installed particleUtils helper
        # without applying UsdPhysics.MassAPI.
        particle_api = PhysxSchema.PhysxParticleAPI(particle_set_api)
        particle_api.CreateParticleSystemRel().SetTargets([system_path])
        particle_api.CreateSelfCollisionAttr().Set(True)
        particle_api.CreateParticleEnabledAttr().Set(True)
        particle_api.CreateParticleGroupAttr().Set(0)
        particle_set_api.CreateFluidAttr().Set(True)
        record["particle_set"]["system_relation"] = [str(x) for x in particle_api.GetParticleSystemRel().GetTargets()] == [str(system_path)]
        record["particle_set"]["mass_api_present"] = points.GetPrim().HasAPI(UsdPhysics.MassAPI)

        iso_api = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(system.GetPrim())
        iso_api.CreateIsosurfaceEnabledAttr().Set(True)
        iso_api.CreateGridSpacingAttr().Set(0.01)
        iso_api.CreateMaxVerticesAttr().Set(20000)
        iso_api.CreateMaxTrianglesAttr().Set(40000)
        record["isosurface"]["api_present"] = system.GetPrim().HasAPI(PhysxSchema.PhysxParticleIsosurfaceAPI)
        record["isosurface"]["enabled"] = bool(iso_api.GetIsosurfaceEnabledAttr().Get())
        record["schema_api"] = {
            "PhysxParticleSystem": bool(system),
            "PhysxParticleSetAPI": bool(particle_set_api),
            "PhysxParticleAPI": bool(particle_api),
            "PhysxPBDMaterialAPI": bool(pbd),
            "PhysxParticleIsosurfaceAPI": bool(iso_api),
        }
        record["particle_system"]["simulation_owner"] = [str(x) for x in system.GetSimulationOwnerRel().GetTargets()] == [str(scene.GetPath())]

        # Match the installed 110.1 tests: memory stages are explicitly attached
        # before direct simulate/fetch_results calls, and isosurface creation is
        # deferred until two Kit update cycles have completed.
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("fresh stage has no UsdUtils.StageCache id")
        simulation.attach_stage(stage_id)
        stage_attached = True
        record["gpu_simulation"]["stage_attached"] = True
        record["timing"]["simulation_route"] = "attach_stage_simulate_fetch_results"
        iso_path = system_path.AppendChild("Isosurface")
        for _ in range(2):
            app.update()
        iso_before = _isosurface_mesh_evidence(stage, iso_path)
        record["isosurface"].update(iso_before)

        before = _positions(points.GetPrim())
        record["particle_set"]["count_before"] = len(before)
        record["particle_set"]["finite_before"] = _finite_positions(before)
        dt = 1.0 / 60.0
        for step in range(args.steps):
            simulation.simulate(dt, step * dt)
            simulation.fetch_results()
            # GPU particle positions are synchronized to USD by the stage update.
            app.update()
        after = _positions(points.GetPrim())
        record["particle_set"]["count_after"] = len(after)
        record["particle_set"]["finite_after"] = _finite_positions(after)
        record["particle_set"]["mean_position_before"] = _mean_position(before)
        record["particle_set"]["mean_position_after"] = _mean_position(after)
        record["particle_set"]["movement"] = _movement(before, after)
        record["particle_set"]["gravity_moved_down"] = bool(
            record["particle_set"]["mean_position_before"]
            and record["particle_set"]["mean_position_after"]
            and record["particle_set"]["mean_position_after"][1] < record["particle_set"]["mean_position_before"][1] - 1e-7
        )
        record["particle_set"]["box_contained_after"] = bool(
            after and all(-0.14 <= p[0] <= 0.14 and 0.015 <= p[1] <= 0.31 and -0.14 <= p[2] <= 0.14 for p in after)
        )
        systems = [prim for prim in stage.Traverse() if PhysxSchema.PhysxParticleSystem(prim)]
        record["particle_system"]["count"] = len(systems)
        record["particle_system"]["exactly_one"] = len(systems) == 1
        record["isosurface"]["prim_path"] = str(iso_path)
        record["isosurface"].update(_isosurface_mesh_evidence(stage, iso_path))
        record["status"] = "pass"
        if validate_record(record):
            record["status"] = "insufficient_evidence"
            record["errors"].append("one or more F1 evidence gates did not pass")
    except Exception as exc:
        record["status"] = "insufficient_evidence"
        record["errors"].append(f"probe execution failed: {type(exc).__name__}: {exc}")
    finally:
        if stage_attached and simulation is not None:
            try:
                simulation.detach_stage()
            except Exception as exc:
                record["errors"].append(f"could not detach PhysX stage: {type(exc).__name__}: {exc}")
        record["timing"]["wall_seconds"] = time.monotonic() - started
    return record


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _start_or_reuse_kit() -> tuple[Any, Any]:
    """Return ``(app, owned_simulation_app)`` with at most one startup attempt."""
    try:
        import omni.kit.app

        current = omni.kit.app.get_app()
        if current is not None and bool(_safe_call(current, "is_running", False)):
            return current, None
    except Exception:
        pass
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "hide_ui": True})
    import omni.kit.app

    return omni.kit.app.get_app(), simulation_app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    owned_app = None
    exit_code = 2
    try:
        app, owned_app = _start_or_reuse_kit()
        record = run_probe(args)
    except Exception as exc:
        record = empty_record(args.out, steps=args.steps, seed=args.seed)
        record["status"] = "unsupported"
        record["errors"].append(f"Isaac application startup unavailable: {type(exc).__name__}: {exc}")
    try:
        validation_errors = validate_record(record)
        if validation_errors:
            record["errors"].extend(
                error for error in validation_errors if error not in record["errors"]
            )
            if record["status"] == "pass":
                record["status"] = "insufficient_evidence"
        exit_code = 0 if record["status"] == "pass" else 2
        # SimulationApp's documented fast-shutdown path may terminate the process
        # inside close(). Persist and flush the complete receipt before calling it.
        _write_record(args.out, record)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    finally:
        if owned_app is not None:
            owned_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

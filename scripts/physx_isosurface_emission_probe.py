#!/usr/bin/env python3
"""Probe F2 particle-emission and generated-isosurface update behavior without rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.isosurface_geometry import analyze_triangle_mesh_geometry


FROZEN_SEED = 20261025
CONDITION_NAMES = (
    "a12_multi_disabled_toggle",
    "nvidia_demo_shared_append_visibility_ablation",
)


def _frozen_seed(value: str) -> int:
    parsed = int(value)
    if parsed != FROZEN_SEED:
        raise argparse.ArgumentTypeError(f"seed must remain frozen at {FROZEN_SEED}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=_frozen_seed, default=FROZEN_SEED)
    parser.add_argument("--condition", choices=CONDITION_NAMES, required=True)
    return parser


def build_probe_plan(*, seed: int) -> dict[str, object]:
    if seed != FROZEN_SEED:
        raise ValueError(f"seed must remain frozen at {FROZEN_SEED}")
    return {
        "schema_version": 1,
        "seed": seed,
        "fixed_dt_s": 1.0 / 60.0,
        "emission_steps": list(range(6, 121, 6)),
        "particles_per_emission": 4,
        "snapshot_steps": [
            0,
            1,
            6,
            12,
            18,
            21,
            23,
            24,
            27,
            60,
            66,
            120,
            126,
        ],
        "conditions": [
            {
                "name": "a12_multi_disabled_toggle",
                "particle_set_model": "twenty_preauthored_disabled_sets_then_enable",
                "source_visibility": "inherited_for_entire_condition",
                "max_step": 120,
            },
            {
                "name": "nvidia_demo_shared_append_visibility_ablation",
                "particle_set_model": "one_enabled_points_set_runtime_array_append",
                "source_visibility": "inherited_through_step_60_hidden_61_120_restored_121_126",
                "max_step": 126,
            },
        ],
        "hypotheses": {
            "emission_model": (
                "the demo-style shared runtime-append set yields substantive generated "
                "isosurface geometry while A12-style disabled-set toggles do not"
            ),
            "visibility": (
                "after a substantive shared-set surface exists, hiding the generated "
                "source prevents later USD mesh updates"
            ),
        },
        "falsifiers": [
            (
                "shared append never produces substantive geometry, or the always-visible "
                "multi-disabled condition does produce it"
            ),
            (
                "with growing/moving particles and prior visible mesh updates, the generated "
                "mesh hash changes while its visibility is invisible"
            ),
        ],
        "claim_limits": {
            "renderer_exercised": False,
            "twin_scene_exercised": False,
            "physical_fluid_accuracy_validated": False,
            "purpose": "engine_emission_and_generated_mesh_update_ablation_only",
        },
        "nvidia_demo_reference": {
            "installed_file": (
                "omni.physx.demos/omni/physxdemos/scenes/FluidBallEmitterDemo.py"
            ),
            "observed_pattern": (
                "one shared Points or PointInstancer particle set; runtime positions, "
                "velocities, and widths arrays are extended in a session sublayer"
            ),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _particle_positions(Gf: Any, *, emission_ordinal: int, seed: int) -> list[Any]:
    rng = random.Random(seed + emission_ordinal)
    rows = []
    for index in range(4):
        rows.append(
            Gf.Vec3f(
                (index % 2 - 0.5) * 0.003 + (rng.random() - 0.5) * 0.0003,
                (index // 2 - 0.5) * 0.003 + (rng.random() - 0.5) * 0.0003,
                0.15,
            )
        )
    return rows


def _mesh_snapshot(stage: Any, path: Any, UsdGeom: Any) -> dict[str, object]:
    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    points = (mesh.GetPointsAttr().Get() or []) if mesh else []
    counts = (mesh.GetFaceVertexCountsAttr().Get() or []) if mesh else []
    indices = (mesh.GetFaceVertexIndicesAttr().Get() or []) if mesh else []
    point_rows = [
        [float(point[0]), float(point[1]), float(point[2])] for point in points
    ]
    count_rows = [int(value) for value in counts]
    index_rows = [int(value) for value in indices]
    evidence = {
        "prim_present": bool(prim),
        "mesh_is_mesh": bool(mesh),
        **analyze_triangle_mesh_geometry(point_rows, count_rows, index_rows),
    }
    if evidence["mesh_finite_points"] is True:
        evidence.update(
            {
                "points_sha256": _value_hash(point_rows),
                "topology_sha256": _value_hash([count_rows, index_rows]),
                "mesh_sha256": _value_hash([point_rows, count_rows, index_rows]),
            }
        )
    else:
        evidence.update(
            {
                "points_sha256": None,
                "topology_sha256": None,
                "mesh_sha256": None,
            }
        )
    evidence["visibility"] = (
        str(UsdGeom.Imageable(prim).ComputeVisibility()) if prim.IsValid() else None
    )
    return evidence


def _rows_from_points(prims: list[Any], UsdGeom: Any) -> list[list[float]]:
    rows: list[list[float]] = []
    for prim in prims:
        values = UsdGeom.Points(prim).GetPointsAttr().Get() or []
        rows.extend(
            [float(value[0]), float(value[1]), float(value[2])]
            for value in values
        )
    return rows


def _particle_snapshot(
    prims: list[Any],
    UsdGeom: Any,
    *,
    expected_count: int,
    initial_positions: list[list[float]],
) -> dict[str, object]:
    rows = _rows_from_points(prims, UsdGeom)
    finite = all(math.isfinite(value) for row in rows for value in row)
    if rows and finite:
        extent = [
            max(row[axis] for row in rows) - min(row[axis] for row in rows)
            for axis in range(3)
        ]
        position_hash = _value_hash(rows)
    else:
        extent = [None, None, None]
        position_hash = _value_hash([]) if not rows else None
    return {
        "expected_active_particle_count": expected_count,
        "usd_active_particle_count": len(rows),
        "count_matches": len(rows) == expected_count,
        "finite_positions": finite,
        "extent_m": extent,
        "positions_sha256": position_hash,
        "movement": _movement_evidence(initial=initial_positions, current=rows),
    }


def _movement_evidence(
    *, initial: list[list[float]], current: list[list[float]]
) -> dict[str, object]:
    count_matches = len(initial) == len(current)
    finite = all(
        math.isfinite(value)
        for rows in (initial, current)
        for row in rows
        for value in row
    )
    if not count_matches or not initial or not finite:
        return {
            "row_count_matches": count_matches,
            "maximum_displacement_m": None,
            "mean_displacement_m": None,
            "mean_initial_z_m": None,
            "mean_current_z_m": None,
            "motion_observed": False,
            "gravity_moved_down": False,
        }
    distances = [
        math.sqrt(sum((after[axis] - before[axis]) ** 2 for axis in range(3)))
        for before, after in zip(initial, current)
    ]
    mean_initial_z = sum(row[2] for row in initial) / len(initial)
    mean_current_z = sum(row[2] for row in current) / len(current)
    maximum = max(distances)
    mean = sum(distances) / len(distances)
    return {
        "row_count_matches": True,
        "maximum_displacement_m": maximum,
        "mean_displacement_m": mean,
        "mean_initial_z_m": mean_initial_z,
        "mean_current_z_m": mean_current_z,
        "motion_observed": maximum > 0.0,
        "gravity_moved_down": mean_current_z < mean_initial_z - 1.0e-7,
    }


def _author_box(
    stage: Any,
    path: str,
    center: tuple[float, float, float],
    scale: tuple[float, float, float],
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> None:
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3f(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _author_common_stage(
    stage: Any,
    *,
    Gf: Any,
    PhysxSchema: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> tuple[Any, Any]:
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(60.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateEnableGPUDynamicsAttr().Set(True)
    scene_api.CreateBroadphaseTypeAttr().Set("GPU")
    scene_api.CreateSolverTypeAttr().Set("TGS")

    system_path = Sdf.Path("/World/ParticleSystem")
    system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
    system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
    system.CreateParticleSystemEnabledAttr().Set(True)
    system.CreateEnableCCDAttr().Set(True)
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
    shader = UsdShade.Shader.Define(stage, "/World/PBDMaterial/Preview")
    shader.CreateIdAttr().Set("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.18, 0.55, 0.75)
    )
    output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(output)
    binding.Bind(material)

    iso_api = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(system.GetPrim())
    iso_api.CreateIsosurfaceEnabledAttr().Set(True)
    iso_api.CreateGridSpacingAttr().Set(0.002)
    iso_api.CreateMaxVerticesAttr().Set(250000)
    iso_api.CreateMaxTrianglesAttr().Set(500000)

    _author_box(
        stage,
        "/World/Basin/Floor",
        (0.0, 0.0, -0.002),
        (0.05, 0.05, 0.004),
        Gf,
        Sdf,
        UsdGeom,
        UsdPhysics,
    )
    for name, center, scale in (
        ("XMin", (-0.025, 0.0, 0.10), (0.002, 0.05, 0.20)),
        ("XMax", (0.025, 0.0, 0.10), (0.002, 0.05, 0.20)),
        ("YMin", (0.0, -0.025, 0.10), (0.05, 0.002, 0.20)),
        ("YMax", (0.0, 0.025, 0.10), (0.05, 0.002, 0.20)),
    ):
        _author_box(
            stage,
            f"/World/Basin/Wall{name}",
            center,
            scale,
            Gf,
            Sdf,
            UsdGeom,
            UsdPhysics,
        )
    return system, system_path


def _author_particle_set(
    stage: Any,
    path: Any,
    positions: list[Any],
    *,
    enabled: bool,
    system_path: Any,
    Gf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    Vt: Any,
) -> tuple[Any, Any]:
    points = UsdGeom.Points.Define(stage, path)
    points.GetPointsAttr().Set(Vt.Vec3fArray(positions))
    points.GetVelocitiesAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, -0.10)] * len(positions))
    )
    points.GetWidthsAttr().Set(Vt.FloatArray([0.003] * len(positions)))
    set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
    particle_api = PhysxSchema.PhysxParticleAPI(set_api)
    particle_api.CreateParticleSystemRel().SetTargets([system_path])
    particle_api.CreateSelfCollisionAttr().Set(True)
    particle_api.CreateParticleEnabledAttr().Set(enabled)
    particle_api.CreateParticleGroupAttr().Set(0)
    set_api.CreateFluidAttr().Set(True)
    return points.GetPrim(), particle_api


def _run_condition(
    *,
    condition: dict[str, object],
    plan: dict[str, object],
    output: Path,
    app: Any,
    context: Any,
    simulation: Any,
    modules: dict[str, Any],
) -> dict[str, object]:
    Gf = modules["Gf"]
    PhysxSchema = modules["PhysxSchema"]
    Sdf = modules["Sdf"]
    Usd = modules["Usd"]
    UsdGeom = modules["UsdGeom"]
    UsdPhysics = modules["UsdPhysics"]
    UsdShade = modules["UsdShade"]
    UsdUtils = modules["UsdUtils"]
    Vt = modules["Vt"]

    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("omni.usd did not create a fresh probe stage")
    app.update()
    session = stage.GetSessionLayer()
    stage_attached = False
    snapshots: list[dict[str, object]] = []
    active_prims: list[Any] = []
    try:
        with Usd.EditContext(stage, session):
            _system, system_path = _author_common_stage(
                stage,
                Gf=Gf,
                PhysxSchema=PhysxSchema,
                Sdf=Sdf,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                UsdShade=UsdShade,
            )
            particle_records: list[dict[str, Any]] = []
            if condition["name"] == "a12_multi_disabled_toggle":
                for ordinal, _step in enumerate(plan["emission_steps"], start=1):
                    prim, api = _author_particle_set(
                        stage,
                        Sdf.Path(f"/World/Particles/Chunk{ordinal:03d}"),
                        _particle_positions(
                            Gf, emission_ordinal=ordinal, seed=int(plan["seed"])
                        ),
                        enabled=False,
                        system_path=system_path,
                        Gf=Gf,
                        PhysxSchema=PhysxSchema,
                        UsdGeom=UsdGeom,
                        Vt=Vt,
                    )
                    particle_records.append({"prim": prim, "api": api})
            else:
                prim, api = _author_particle_set(
                    stage,
                    Sdf.Path("/World/Particles/Shared"),
                    [],
                    enabled=True,
                    system_path=system_path,
                    Gf=Gf,
                    PhysxSchema=PhysxSchema,
                    UsdGeom=UsdGeom,
                    Vt=Vt,
                )
                prim.CreateAttribute(
                    "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
                ).Set(
                    len(plan["emission_steps"])
                    * int(plan["particles_per_emission"])
                )
                particle_records.append({"prim": prim, "api": api})

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not stage_id:
            raise RuntimeError("probe stage has no StageCache ID")
        simulation.attach_stage(stage_id)
        stage_attached = True
        for _ in range(2):
            app.update()

        iso_path = system_path.AppendChild("Isosurface")
        expected_count = 0
        initial_active_positions: list[list[float]] = []

        def save_snapshot(step: int) -> dict[str, object]:
            record = {
                "physics_step": step,
                "particle_set_count": len(particle_records),
                "particles": _particle_snapshot(
                    active_prims,
                    UsdGeom,
                    expected_count=expected_count,
                    initial_positions=initial_active_positions,
                ),
                "isosurface": _mesh_snapshot(stage, iso_path, UsdGeom),
            }
            path = output / str(condition["name"]) / f"step-{step:03d}.json"
            _write_json(path, record)
            record["artifact_path"] = str(path.relative_to(output))
            record["artifact_sha256"] = _sha256(path)
            snapshots.append(record)
            return record

        save_snapshot(0)
        emission_by_step = {
            int(step): ordinal
            for ordinal, step in enumerate(plan["emission_steps"], start=1)
        }
        max_step = int(condition["max_step"])
        snapshot_steps = set(int(step) for step in plan["snapshot_steps"])
        first_substantive_step = None
        for step in range(1, max_step + 1):
            ordinal = emission_by_step.get(step)
            if ordinal is not None:
                emitted_positions = _particle_positions(
                    Gf,
                    emission_ordinal=ordinal,
                    seed=int(plan["seed"]),
                )
                with Usd.EditContext(stage, session):
                    if condition["name"] == "a12_multi_disabled_toggle":
                        record = particle_records[ordinal - 1]
                        if not record["api"].GetParticleEnabledAttr().Set(True):
                            raise RuntimeError("could not enable preauthored particle set")
                        active_prims.append(record["prim"])
                    else:
                        shared = UsdGeom.Points(particle_records[0]["prim"])
                        positions = list(shared.GetPointsAttr().Get() or [])
                        velocities = list(shared.GetVelocitiesAttr().Get() or [])
                        widths = list(shared.GetWidthsAttr().Get() or [])
                        positions.extend(
                            emitted_positions
                        )
                        velocities.extend(
                            [Gf.Vec3f(0.0, 0.0, -0.10)]
                            * int(plan["particles_per_emission"])
                        )
                        widths.extend(
                            [0.003] * int(plan["particles_per_emission"])
                        )
                        shared.GetPointsAttr().Set(Vt.Vec3fArray(positions))
                        shared.GetVelocitiesAttr().Set(Vt.Vec3fArray(velocities))
                        shared.GetWidthsAttr().Set(Vt.FloatArray(widths))
                        if not active_prims:
                            active_prims.append(particle_records[0]["prim"])
                expected_count += int(plan["particles_per_emission"])
                initial_active_positions.extend(
                    [
                        [float(value[0]), float(value[1]), float(value[2])]
                        for value in emitted_positions
                    ]
                )

            simulation.simulate(float(plan["fixed_dt_s"]), step * float(plan["fixed_dt_s"]))
            simulation.fetch_results()
            app.update()

            live_mesh = _mesh_snapshot(stage, iso_path, UsdGeom)
            became_substantive = False
            if (
                first_substantive_step is None
                and live_mesh["mesh_geometry_valid"] is True
            ):
                first_substantive_step = step
                became_substantive = True

            if step in snapshot_steps or became_substantive:
                save_snapshot(step)

            if (
                condition["name"]
                == "nvidia_demo_shared_append_visibility_ablation"
                and step == 60
            ):
                prim = stage.GetPrimAtPath(iso_path)
                if prim.IsValid():
                    with Usd.EditContext(stage, session):
                        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                            UsdGeom.Tokens.invisible
                        )
                    app.update()
            if (
                condition["name"]
                == "nvidia_demo_shared_append_visibility_ablation"
                and step == 120
            ):
                prim = stage.GetPrimAtPath(iso_path)
                if prim.IsValid():
                    with Usd.EditContext(stage, session):
                        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                            UsdGeom.Tokens.inherited
                        )
                    app.update()

        first_substantive_artifact = next(
            (
                {
                    "path": row["artifact_path"],
                    "sha256": row["artifact_sha256"],
                    "mesh_sha256": row["isosurface"]["mesh_sha256"],
                }
                for row in snapshots
                if row["physics_step"] == first_substantive_step
            ),
            None,
        )
        return {
            "name": condition["name"],
            "completed": True,
            "first_substantive_isosurface_step": first_substantive_step,
            "substantive_isosurface_observed": first_substantive_step is not None,
            "first_substantive_artifact": first_substantive_artifact,
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
        }
    finally:
        if stage_attached:
            simulation.detach_stage()
            app.update()


def _snapshot_by_step(condition: dict[str, object], step: int) -> dict[str, object]:
    for row in condition.get("snapshots", []):
        if row.get("physics_step") == step:
            return row
    return {}


def _decide_hypotheses(conditions: list[dict[str, object]]) -> dict[str, object]:
    rows = {str(row.get("name")): row for row in conditions}
    multi = rows.get("a12_multi_disabled_toggle", {})
    shared = rows.get("nvidia_demo_shared_append_visibility_ablation", {})
    multi_valid = multi.get("substantive_isosurface_observed") is True
    shared_valid = shared.get("substantive_isosurface_observed") is True
    if shared_valid and not multi_valid:
        emission = "supported"
    elif not shared_valid or multi_valid:
        emission = "falsified"
    else:
        emission = "insufficient_evidence"

    shared_24 = _snapshot_by_step(shared, 24)
    shared_60 = _snapshot_by_step(shared, 60)
    shared_120 = _snapshot_by_step(shared, 120)
    shared_126 = _snapshot_by_step(shared, 126)
    hashes = [
        row.get("isosurface", {}).get("mesh_sha256")
        for row in (shared_24, shared_60, shared_120, shared_126)
    ]
    counts = [
        row.get("particles", {}).get("usd_active_particle_count")
        for row in (shared_24, shared_60, shared_120, shared_126)
    ]
    visible_updates = bool(
        hashes[0]
        and hashes[1]
        and hashes[0] != hashes[1]
        and counts[0] < counts[1]
    )
    hidden_particles_grew = bool(
        counts[1] is not None
        and counts[2] is not None
        and counts[2] > counts[1]
    )
    if not shared_valid or not visible_updates or not hidden_particles_grew:
        visibility = "insufficient_evidence"
    elif hashes[1] == hashes[2]:
        visibility = "supported"
    else:
        visibility = "falsified"
    return {
        "emission_model": emission,
        "visibility": visibility,
        "visibility_evidence": {
            "visible_updates_observed_step24_to_60": visible_updates,
            "particles_grew_while_hidden_step60_to_120": hidden_particles_grew,
            "mesh_changed_while_hidden_step60_to_120": bool(
                hashes[1] and hashes[2] and hashes[1] != hashes[2]
            ),
            "mesh_changed_after_restore_step120_to_126": bool(
                hashes[2] and hashes[3] and hashes[2] != hashes[3]
            ),
            "mesh_hashes_steps_24_60_120_126": hashes,
            "particle_counts_steps_24_60_120_126": counts,
        },
    }


def _start_or_reuse_kit() -> tuple[Any, Any]:
    try:
        import omni.kit.app

        current = omni.kit.app.get_app()
        if current is not None and bool(current.is_running()):
            return current, None
    except Exception:
        pass
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "hide_ui": True})
    import omni.kit.app

    return omni.kit.app.get_app(), simulation_app


def run_probe(args: argparse.Namespace, app: Any) -> dict[str, object]:
    import carb
    import omni.physx
    import omni.usd
    from omni.physxfoundation import get_physx_foundation_interface
    from pxr import (
        Gf,
        PhysxSchema,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        UsdUtils,
        Vt,
    )
    from omni.physx.bindings import _physx as physx_bindings

    plan = build_probe_plan(seed=args.seed)
    foundation = get_physx_foundation_interface()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "experiment": "physx_isosurface_emission_update_probe",
        "status": "insufficient_evidence",
        "plan": plan,
        "runtime": {
            "cuda_device_available": bool(foundation.cuda_device_check()),
            "update_to_usd": None,
            "update_particles_to_usd": None,
        },
        "conditions": [],
        "hypothesis_outcomes": {},
        "errors": [],
    }
    if receipt["runtime"]["cuda_device_available"] is not True:
        receipt["errors"].append("GPU particle simulation requires a CUDA device")
        return receipt
    settings = carb.settings.get_settings()
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
    particle_setting = getattr(
        physx_bindings, "SETTING_UPDATE_PARTICLES_TO_USD", None
    )
    if particle_setting is None:
        receipt["errors"].append("PhysX lacks SETTING_UPDATE_PARTICLES_TO_USD")
        return receipt
    settings.set(particle_setting, True)
    receipt["runtime"]["update_to_usd"] = bool(
        settings.get_as_bool(physx_bindings.SETTING_UPDATE_TO_USD)
    )
    receipt["runtime"]["update_particles_to_usd"] = bool(
        settings.get_as_bool(particle_setting)
    )

    modules = {
        "Gf": Gf,
        "PhysxSchema": PhysxSchema,
        "Sdf": Sdf,
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "UsdPhysics": UsdPhysics,
        "UsdShade": UsdShade,
        "UsdUtils": UsdUtils,
        "Vt": Vt,
    }
    context = omni.usd.get_context()
    simulation = omni.physx.get_physx_simulation_interface()
    selected = next(
        condition
        for condition in plan["conditions"]
        if condition["name"] == args.condition
    )
    receipt["condition_requested"] = args.condition
    receipt["conditions"].append(
        _run_condition(
            condition=selected,
            plan=plan,
            output=args.out,
            app=app,
            context=context,
            simulation=simulation,
            modules=modules,
        )
    )
    receipt["hypothesis_outcomes"] = {
        "emission_model": "requires_both_fresh_processes",
        "visibility": (
            _decide_hypotheses(receipt["conditions"])["visibility"]
            if args.condition
            == "nvidia_demo_shared_append_visibility_ablation"
            else "not_applicable"
        ),
    }
    final_snapshot = receipt["conditions"][0]["snapshots"][-1]
    final_movement = final_snapshot["particles"]["movement"]
    if (
        len(receipt["conditions"]) == 1
        and all(row.get("completed") is True for row in receipt["conditions"])
        and all(
            snapshot.get("particles", {}).get("finite_positions") is True
            and snapshot.get("particles", {}).get("count_matches") is True
            for row in receipt["conditions"]
            for snapshot in row.get("snapshots", [])
        )
        and final_movement.get("motion_observed") is True
        and final_movement.get("gravity_moved_down") is True
    ):
        receipt["status"] = "completed"
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    print("physx isosurface emission probe starting", flush=True)
    owned_app = None
    started = time.monotonic()
    receipt: dict[str, object]
    try:
        app, owned_app = _start_or_reuse_kit()
        receipt = run_probe(args, app)
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "experiment": "physx_isosurface_emission_update_probe",
            "status": "insufficient_evidence",
            "plan": build_probe_plan(seed=args.seed),
            "conditions": [],
            "hypothesis_outcomes": {},
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    receipt["wall_seconds"] = time.monotonic() - started
    receipt_path = args.out / "receipt.json"
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    exit_code = 0 if receipt.get("status") == "completed" else 2
    if owned_app is not None:
        owned_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

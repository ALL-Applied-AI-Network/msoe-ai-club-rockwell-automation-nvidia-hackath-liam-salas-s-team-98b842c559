#!/usr/bin/env python3
"""Run one no-render, timeline-owned, 12-callback PhysX emission probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.isosurface_geometry import analyze_triangle_mesh_geometry
from fluidmix.timeline_probe import (
    EMISSION_CALLBACK_INDEX,
    FIXED_HZ,
    FROZEN_SEED,
    PARTICLES_PER_EMISSION,
    TARGET_CALLBACK_COUNT,
    build_timeline_probe_plan,
    movement_evidence,
)
from physx_isosurface_emission_probe import (
    _author_common_stage,
    _author_particle_set,
    _particle_positions,
    _sha256,
    _value_hash,
    _write_json,
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
    return parser


def _point_rows(prim: Any, UsdGeom: Any) -> list[list[float]]:
    values = UsdGeom.Points(prim).GetPointsAttr().Get() or []
    return [[float(value[0]), float(value[1]), float(value[2])] for value in values]


def _mesh_artifact(stage: Any, path: Any, UsdGeom: Any) -> dict[str, object]:
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
    geometry = analyze_triangle_mesh_geometry(point_rows, count_rows, index_rows)
    mesh_hash = _value_hash([point_rows, count_rows, index_rows])
    return {
        "prim_path": str(path),
        "prim_present": bool(prim),
        "mesh_is_mesh": bool(mesh),
        "points_m": point_rows,
        "face_vertex_counts": count_rows,
        "face_vertex_indices": index_rows,
        **geometry,
        "mesh_sha256": mesh_hash,
    }


def _record_artifact(output: Path, relative: str, value: object) -> dict[str, str]:
    path = output / relative
    _write_json(path, value)
    return {"path": relative, "sha256": _sha256(path)}


def run_probe(*, output: Path, seed: int) -> int:
    plan = build_timeline_probe_plan(seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", plan)
    print(f"TIMELINE_PROBE_PLAN {json.dumps(plan, sort_keys=True)}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "hide_ui": True})
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "failed",
        "plan": plan,
        "errors": [],
        "callbacks": [],
    }
    errors: list[str] = []
    timeline = None
    physics_step_subscription = None
    app_update_index = 0
    try:
        import carb
        import omni.physx
        import omni.timeline
        import omni.usd
        from omni.physx.bindings._physx import (
            SETTING_MIN_FRAME_RATE,
            SETTING_UPDATE_PARTICLES_TO_USD,
            SETTING_UPDATE_TO_USD,
        )
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd did not create a fresh probe stage")
        app_update_index += 1
        app.update()

        session = stage.GetSessionLayer()
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
            scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
            scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
            scene_api.GetTimeStepsPerSecondAttr().Set(FIXED_HZ)

        settings = carb.settings.get_settings()
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(SETTING_MIN_FRAME_RATE, FIXED_HZ)
        settings.set_bool(SETTING_UPDATE_TO_USD, True)
        settings.set_bool(SETTING_UPDATE_PARTICLES_TO_USD, True)

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(0.0)
        timeline.set_time_codes_per_second(float(FIXED_HZ))
        timeline.set_ticks_per_frame(1)
        timeline.set_target_framerate(float(FIXED_HZ))
        timeline.set_play_every_frame(True)
        timeline.commit()
        app_update_index += 1
        app.update()

        settings_observed = {
            "stage_time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
            "timeline_time_codes_per_second": float(
                timeline.get_time_codes_per_second()
            ),
            "timeline_target_framerate": float(timeline.get_target_framerate()),
            "timeline_ticks_per_frame": int(timeline.get_ticks_per_frame()),
            "timeline_play_every_frame": bool(timeline.get_play_every_frame()),
            "use_fixed_time_stepping": bool(
                settings.get_as_bool("/app/player/useFixedTimeStepping")
            ),
            "physics_time_steps_per_second": int(
                scene_api.GetTimeStepsPerSecondAttr().Get()
            ),
            "physics_min_frame_rate": int(settings.get(SETTING_MIN_FRAME_RATE)),
        }
        receipt["settings_observed"] = settings_observed
        print(
            f"TIMELINE_PROBE_SETTINGS {json.dumps(settings_observed, sort_keys=True)}",
            flush=True,
        )

        callbacks: list[dict[str, object]] = []

        def on_physics_step(dt: float) -> None:
            callbacks.append(
                {
                    "callback_index": len(callbacks) + 1,
                    "dt_s": float(dt),
                    "app_update_index": app_update_index,
                    "timeline_time_s": float(timeline.get_current_time()),
                }
            )

        physics_step_subscription = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                on_physics_step
            )
        )
        timeline.play()
        timeline.commit()

        particle_prim = None
        initial_positions: list[list[float]] = []
        emission: dict[str, object] | None = None
        pump_limit = TARGET_CALLBACK_COUNT * 4
        pump_count = 0
        while len(callbacks) < TARGET_CALLBACK_COUNT and pump_count < pump_limit:
            if len(callbacks) == EMISSION_CALLBACK_INDEX - 1 and particle_prim is None:
                positions = _particle_positions(
                    Gf, emission_ordinal=1, seed=seed
                )
                with Usd.EditContext(stage, session):
                    particle_prim, _particle_api = _author_particle_set(
                        stage,
                        Sdf.Path("/World/Particles/Shared"),
                        positions,
                        enabled=True,
                        system_path=system_path,
                        Gf=Gf,
                        PhysxSchema=PhysxSchema,
                        UsdGeom=UsdGeom,
                        Vt=Vt,
                    )
                    particle_prim.CreateAttribute(
                        "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
                    ).Set(PARTICLES_PER_EMISSION)
                initial_positions = [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in positions
                ]
                emission = {
                    "scheduled_before_callback_index": EMISSION_CALLBACK_INDEX,
                    "authored_after_callback_count": len(callbacks),
                    "authored_at_app_update_index": app_update_index,
                    "particle_count": len(initial_positions),
                }
                print(
                    f"TIMELINE_PROBE_EMISSION {json.dumps(emission, sort_keys=True)}",
                    flush=True,
                )

            before = len(callbacks)
            app_update_index += 1
            app.update()
            pump_count += 1
            produced = len(callbacks) - before
            if produced > 1:
                errors.append("multiple_callbacks_in_one_app_update")
                break
            if produced == 1:
                print(
                    "TIMELINE_PROBE_CALLBACK "
                    + json.dumps(callbacks[-1], sort_keys=True),
                    flush=True,
                )
            if len(callbacks) > TARGET_CALLBACK_COUNT:
                errors.append("extra_callback_observed")
                break

        timeline.pause()
        timeline.commit()
        before_pause_flush = len(callbacks)
        app_update_index += 1
        app.update()
        if len(callbacks) != before_pause_flush:
            errors.append("callback_after_committed_pause")

        current_positions = (
            _point_rows(particle_prim, UsdGeom) if particle_prim is not None else []
        )
        movement = movement_evidence(
            initial=initial_positions, current=current_positions
        )
        mesh_artifact = _mesh_artifact(
            stage, system_path.AppendChild("Isosurface"), UsdGeom
        )
        callback_artifact = _record_artifact(output, "callback-trace.json", callbacks)
        particle_artifact = _record_artifact(
            output,
            "particle-positions.json",
            {
                "initial_positions_m": initial_positions,
                "current_positions_m": current_positions,
                "initial_positions_sha256": _value_hash(initial_positions),
                "current_positions_sha256": _value_hash(current_positions),
                "movement": movement,
            },
        )
        mesh_record = _record_artifact(output, "mesh-final.json", mesh_artifact)
        artifacts = {
            "plan": {
                "path": "plan.json",
                "sha256": _sha256(output / "plan.json"),
            },
            "callbacks": callback_artifact,
            "particles": particle_artifact,
            "mesh": mesh_record,
        }
        receipt.update(
            {
                "status": "completed",
                "errors": errors,
                "callbacks": callbacks,
                "explicit_app_update_count": app_update_index,
                "emission": emission,
                "initial_positions_m": initial_positions,
                "current_positions_m": current_positions,
                "movement": movement,
                "mesh_final": {
                    key: value
                    for key, value in mesh_artifact.items()
                    if key
                    not in {
                        "points_m",
                        "face_vertex_counts",
                        "face_vertex_indices",
                    }
                },
                "artifacts": artifacts,
            }
        )
        _write_json(output / "receipt.json", receipt)
        print(
            "TIMELINE_PROBE_OBSERVATION "
            + json.dumps(
                {
                    "callback_count": len(callbacks),
                    "errors": errors,
                    "movement": movement,
                    "mesh_geometry_valid": mesh_artifact["mesh_geometry_valid"],
                    "mesh_sha256": mesh_artifact["mesh_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if not errors else 2
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        receipt["errors"] = errors
        receipt["explicit_app_update_count"] = app_update_index
        _write_json(output / "receipt.json", receipt)
        print(f"TIMELINE_PROBE_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if timeline is not None:
            timeline.pause()
            timeline.commit()
        physics_step_subscription = None
        app.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_probe(output=args.out, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())

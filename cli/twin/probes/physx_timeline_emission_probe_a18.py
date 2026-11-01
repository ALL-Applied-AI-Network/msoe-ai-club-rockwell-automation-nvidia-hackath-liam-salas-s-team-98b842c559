#!/usr/bin/env python3
"""Run one layer-corrected, no-render, timeline-owned PhysX emission probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from twin.probes.timeline_probe import (
    CORRECTED_PARTICLES_PER_EMISSION,
    CORRECTED_TARGET_CALLBACK_COUNT,
    EMISSION_CALLBACK_INDEX,
    FIXED_HZ,
    FROZEN_SEED,
    build_corrected_timeline_probe_plan,
    corrected_callback_completion_failures,
    movement_evidence,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cli.twin.probes.physx_isosurface_emission_probe import (
    _author_common_stage,
    _author_particle_set,
    _particle_positions,
    _sha256,
    _value_hash,
    _write_json,
)
from cli.twin.probes.physx_timeline_emission_probe import (
    _mesh_artifact,
    _point_rows,
    _record_artifact,
)


POSITION_SOURCE = "a15_emission_ordinals_1_through_3_concatenated_unchanged"


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


def _app_update_in_session(stage: Any, session: Any, app: Any, Usd: Any) -> None:
    with Usd.EditContext(stage, session):
        app.update()


def _a15_first_twelve_positions(Gf: Any, *, seed: int) -> list[Any]:
    positions: list[Any] = []
    for ordinal in range(1, 4):
        positions.extend(
            _particle_positions(Gf, emission_ordinal=ordinal, seed=seed)
        )
    if len(positions) != CORRECTED_PARTICLES_PER_EMISSION:
        raise RuntimeError("frozen A15 position source did not yield exactly 12 rows")
    return positions


def _runtime_layer_specs(
    stage: Any, particle_prim: Any, Sdf: Any
) -> dict[str, bool]:
    path = particle_prim.GetPath()
    session = stage.GetSessionLayer()
    root = stage.GetRootLayer()
    rows: dict[str, bool] = {}
    for name in ("points", "velocities", "widths"):
        property_path = Sdf.Path(path).AppendProperty(name)
        rows[f"{name}_session_layer_spec"] = (
            session.GetPropertyAtPath(property_path) is not None
        )
        rows[f"{name}_root_layer_spec"] = (
            root.GetPropertyAtPath(property_path) is not None
        )
    rows["final_edit_target_is_session"] = (
        stage.GetEditTarget().GetLayer().identifier == session.identifier
    )
    return rows


def run_probe(*, output: Path, seed: int) -> int:
    plan = build_corrected_timeline_probe_plan(seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", plan)
    print(f"TIMELINE_A18_PLAN {json.dumps(plan, sort_keys=True)}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "hide_ui": True})
    receipt: dict[str, object] = {
        "schema_version": 2,
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
            scene_api = PhysxSchema.PhysxSceneAPI(
                stage.GetPrimAtPath("/World/PhysicsScene")
            )
            scene_api.GetTimeStepsPerSecondAttr().Set(FIXED_HZ)

        settings = carb.settings.get_settings()
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(SETTING_MIN_FRAME_RATE, FIXED_HZ)
        settings.set_bool(SETTING_UPDATE_TO_USD, True)
        settings.set_bool(SETTING_UPDATE_PARTICLES_TO_USD, True)

        timeline = omni.timeline.get_timeline_interface()
        with Usd.EditContext(stage, session):
            timeline.stop()
            timeline.set_current_time(0.0)
            timeline.set_time_codes_per_second(float(FIXED_HZ))
            timeline.set_ticks_per_frame(1)
            timeline.set_target_framerate(float(FIXED_HZ))
            timeline.set_play_every_frame(True)
            timeline.commit()
        app_update_index += 1
        _app_update_in_session(stage, session, app, Usd)

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
            f"TIMELINE_A18_SETTINGS {json.dumps(settings_observed, sort_keys=True)}",
            flush=True,
        )

        callbacks: list[dict[str, object]] = []
        with Usd.EditContext(stage, session):
            timeline.play()
            timeline.commit()
        timeline_initialization = {
            "play_committed_before_subscription": True,
            "timeline_time_before_subscription_s": float(
                timeline.get_current_time()
            ),
            "callbacks_before_subscription": len(callbacks),
        }
        receipt["timeline_initialization"] = timeline_initialization
        print(
            "TIMELINE_A18_INITIALIZATION "
            + json.dumps(timeline_initialization, sort_keys=True),
            flush=True,
        )

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

        particle_prim = None
        initial_positions: list[list[float]] = []
        emission: dict[str, object] | None = None
        pump_limit = CORRECTED_TARGET_CALLBACK_COUNT * 4
        pump_count = 0
        while (
            len(callbacks) < CORRECTED_TARGET_CALLBACK_COUNT
            and pump_count < pump_limit
        ):
            if len(callbacks) == EMISSION_CALLBACK_INDEX - 1 and particle_prim is None:
                positions = _a15_first_twelve_positions(Gf, seed=seed)
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
                    ).Set(CORRECTED_PARTICLES_PER_EMISSION)
                initial_positions = [
                    [float(value[0]), float(value[1]), float(value[2])]
                    for value in positions
                ]
                emission = {
                    "scheduled_before_callback_index": EMISSION_CALLBACK_INDEX,
                    "authored_after_callback_count": len(callbacks),
                    "authored_at_app_update_index": app_update_index,
                    "particle_count": len(initial_positions),
                    "position_source": POSITION_SOURCE,
                }
                print(
                    f"TIMELINE_A18_EMISSION {json.dumps(emission, sort_keys=True)}",
                    flush=True,
                )

            before = len(callbacks)
            app_update_index += 1
            _app_update_in_session(stage, session, app, Usd)
            pump_count += 1
            produced = len(callbacks) - before
            if produced > 1:
                errors.append("multiple_callbacks_in_one_app_update")
                break
            if produced == 1:
                print(
                    "TIMELINE_A18_CALLBACK "
                    + json.dumps(callbacks[-1], sort_keys=True),
                    flush=True,
                )
            if len(callbacks) > CORRECTED_TARGET_CALLBACK_COUNT:
                errors.append("extra_callback_observed")
                break
        errors.extend(corrected_callback_completion_failures(len(callbacks)))

        with Usd.EditContext(stage, session):
            timeline.pause()
            timeline.commit()
        before_pause_flush = len(callbacks)
        app_update_index += 1
        _app_update_in_session(stage, session, app, Usd)
        if len(callbacks) != before_pause_flush:
            errors.append("callback_after_committed_pause")

        with Usd.EditContext(stage, session):
            current_positions = (
                _point_rows(particle_prim, UsdGeom)
                if particle_prim is not None
                else []
            )
            runtime_layer_specs = (
                _runtime_layer_specs(stage, particle_prim, Sdf)
                if particle_prim is not None
                else {}
            )
            mesh_artifact = _mesh_artifact(
                stage, system_path.AppendChild("Isosurface"), UsdGeom
            )
        movement = movement_evidence(
            initial=initial_positions, current=current_positions
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
                "runtime_layer_specs": runtime_layer_specs,
            },
        )
        mesh_record = _record_artifact(output, "mesh-final.json", mesh_artifact)
        receipt.update(
            {
                "status": "completed",
                "errors": errors,
                "callbacks": callbacks,
                "explicit_app_update_count": app_update_index,
                "emission": emission,
                "runtime_layer_specs": runtime_layer_specs,
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
                "artifacts": {
                    "plan": {
                        "path": "plan.json",
                        "sha256": _sha256(output / "plan.json"),
                    },
                    "callbacks": callback_artifact,
                    "particles": particle_artifact,
                    "mesh": mesh_record,
                },
            }
        )
        _write_json(output / "receipt.json", receipt)
        print(
            "TIMELINE_A18_OBSERVATION "
            + json.dumps(
                {
                    "callback_count": len(callbacks),
                    "errors": errors,
                    "movement": movement,
                    "runtime_layer_specs": runtime_layer_specs,
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
        print(f"TIMELINE_A18_ERROR {type(exc).__name__}: {exc}", flush=True)
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

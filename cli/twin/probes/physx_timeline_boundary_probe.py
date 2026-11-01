#!/usr/bin/env python3
"""Run one bounded no-capture Isaac trace across the former twin timeline wrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cli.twin.physx.physx_three_pour_sequence import (
    _app_update_in_session,
    _configure_zero_origin_timeline,
    _start_timeline_non_looping,
    _timeline_state_readback,
)


PHYSICS_HZ = 120
TARGET_CALLBACK_COUNT = 2750
FINAL_CALLBACK_INDEX = 4004
SOURCE_STAGE_END_TIME_CODE = 686.0
SOURCE_STAGE_TIME_CODES_PER_SECOND = 30.0
BOUNDARY_WINDOW_START = 2743
BOUNDARY_WINDOW_END = 2750


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=_positive_int, default=300)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_callback_trace(
    callbacks: list[dict[str, object]],
    *,
    physics_hz: int,
    target_callback_count: int,
) -> dict[str, object]:
    """Validate the exact fixed-step callback/time contract without Isaac imports."""

    failures: list[str] = []
    if len(callbacks) != target_callback_count:
        failures.append("callback_count_not_target")
    expected_dt = 1.0 / float(physics_hz)
    timing_errors: list[float] = []
    callback_indices: list[int] = []
    timeline_times: list[float] = []
    update_indices: list[int] = []
    for row in callbacks:
        callback_index = int(row["callback_index"])
        callback_indices.append(callback_index)
        dt_s = float(row["dt_s"])
        timeline_time_s = float(row["timeline_time_s"])
        timeline_times.append(timeline_time_s)
        update_indices.append(int(row["app_update_index"]))
        timing_errors.append(abs(timeline_time_s - callback_index / physics_hz))
        if not math.isclose(dt_s, expected_dt, rel_tol=0.0, abs_tol=1.0e-9):
            if "callback_dt_not_120hz" not in failures:
                failures.append("callback_dt_not_120hz")
    if callback_indices != list(range(1, len(callbacks) + 1)):
        failures.append("callback_indices_not_contiguous")
    if any(
        later <= earlier
        for earlier, later in zip(timeline_times, timeline_times[1:])
    ):
        failures.append("timeline_time_not_strictly_monotonic")
    if timing_errors and max(timing_errors) > 1.0e-9:
        failures.append("timeline_time_not_callback_index_over_hz")
    if update_indices != list(range(1, len(callbacks) + 1)):
        failures.append("not_exactly_one_callback_per_app_update")
    source_stage_end_time_s = (
        SOURCE_STAGE_END_TIME_CODE / SOURCE_STAGE_TIME_CODES_PER_SECOND
    )
    crossed_source_stage_end = bool(
        timeline_times and timeline_times[-1] > source_stage_end_time_s
    )
    if not crossed_source_stage_end:
        failures.append("trace_did_not_cross_source_stage_end")
    boundary_window = [
        dict(row)
        for row in callbacks
        if BOUNDARY_WINDOW_START
        <= int(row["callback_index"])
        <= BOUNDARY_WINDOW_END
    ]
    if len(boundary_window) != BOUNDARY_WINDOW_END - BOUNDARY_WINDOW_START + 1:
        failures.append("former_wrap_boundary_window_incomplete")
    return {
        "passed": not failures,
        "failed_predicates": failures,
        "callback_count": len(callbacks),
        "physics_hz": physics_hz,
        "expected_dt_s": expected_dt,
        "max_abs_timing_error_s": max(timing_errors, default=None),
        "source_stage_end_time_s": source_stage_end_time_s,
        "crossed_source_stage_end": crossed_source_stage_end,
        "boundary_window": boundary_window,
    }


def run_probe(*, stage_path: Path, output: Path, wall_seconds: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    print(
        "TIMELINE_BOUNDARY_BOOT "
        + json.dumps(
            {
                "stage": os.fspath(stage_path),
                "target_callbacks": TARGET_CALLBACK_COUNT,
                "final_callback_index": FINAL_CALLBACK_INDEX,
                "physics_hz": PHYSICS_HZ,
                "render_capture": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "disable_viewport_updates": True,
        }
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "failed",
        "hypothesis": (
            "An explicit 4005/120 s end and explicit non-looping playback keep "
            "the canonical twin timeline monotonic through callback 2750."
        ),
        "falsifier": (
            "Any missing/non-finite callback, non-1/120 dt, non-monotonic or "
            "non-index/120 time, incomplete former-wrap window, or failed readback."
        ),
        "render_capture": False,
        "pbd_or_volume_settings_authored": False,
        "stage": os.fspath(stage_path),
        "stage_sha256": _sha256(stage_path),
        "target_callback_count": TARGET_CALLBACK_COUNT,
        "production_final_callback_index": FINAL_CALLBACK_INDEX,
        "errors": [],
    }
    errors: list[str] = []
    callbacks: list[dict[str, object]] = []
    physics_step_subscription = None
    timeline = None
    app_update_index = 0
    deadline = time.monotonic() + wall_seconds
    try:
        import carb
        import omni.kit.app
        import omni.physx
        import omni.timeline
        import omni.usd
        from omni.physx.bindings._physx import SETTING_MIN_FRAME_RATE
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        app = omni.kit.app.get_app()
        context = omni.usd.get_context()
        if not context.open_stage(os.fspath(stage_path)):
            raise RuntimeError(f"could not open canonical twin stage: {stage_path}")
        while context.get_stage_loading_status()[2] > 0:
            if time.monotonic() > deadline:
                raise RuntimeError("stage-load wall-time stop reached")
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("omni.usd returned no stage after open_stage")
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid() or str(default_prim.GetPath()) != "/FillStation":
            raise RuntimeError("probe stage is not the canonical FillStation twin")
        if not math.isclose(
            float(UsdGeom.GetStageMetersPerUnit(stage)),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("probe stage is not metre-scale")

        session_layer = stage.GetSessionLayer()
        with Usd.EditContext(stage, session_layer):
            UsdGeom.Xform.Define(stage, "/FillStation/FluidmixTimelineBoundaryProbe")
            scene = UsdPhysics.Scene.Define(
                stage, "/FillStation/FluidmixTimelineBoundaryProbe/PhysicsScene"
            )
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            scene_api.CreateTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
        scene_tps = int(scene_api.GetTimeStepsPerSecondAttr().Get())
        if scene_tps != PHYSICS_HZ:
            raise RuntimeError("probe PhysicsScene did not retain 120 Hz")

        settings = carb.settings.get_settings()
        settings.set_bool("/app/player/useFixedTimeStepping", True)
        settings.set_int(SETTING_MIN_FRAME_RATE, PHYSICS_HZ)
        timeline = omni.timeline.get_timeline_interface()
        configured = _configure_zero_origin_timeline(
            stage,
            session_layer,
            app,
            timeline,
            Usd,
            physics_hz=PHYSICS_HZ,
            final_callback_index=FINAL_CALLBACK_INDEX,
        )
        receipt["timeline_configuration"] = configured
        print(
            "TIMELINE_BOUNDARY_CONFIGURED "
            + json.dumps(configured, sort_keys=True),
            flush=True,
        )
        playback = _start_timeline_non_looping(
            stage,
            session_layer,
            timeline,
            Usd,
            configured=configured,
        )
        receipt["timeline_playback_start"] = playback
        print(
            "TIMELINE_BOUNDARY_PLAYBACK "
            + json.dumps(playback, sort_keys=True),
            flush=True,
        )

        def on_physics_step(dt_s: float) -> None:
            callbacks.append(
                {
                    "callback_index": len(callbacks) + 1,
                    "dt_s": float(dt_s),
                    "timeline_time_s": float(timeline.get_current_time()),
                    "app_update_index": app_update_index,
                }
            )

        physics_step_subscription = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                on_physics_step
            )
        )
        pump_limit = TARGET_CALLBACK_COUNT * 4
        pump_count = 0
        checkpoints = {1, *range(BOUNDARY_WINDOW_START, BOUNDARY_WINDOW_END + 1)}
        while len(callbacks) < TARGET_CALLBACK_COUNT and pump_count < pump_limit:
            if time.monotonic() > deadline:
                raise RuntimeError("callback-trace wall-time stop reached")
            before = len(callbacks)
            app_update_index += 1
            _app_update_in_session(stage, session_layer, app, Usd)
            pump_count += 1
            produced = len(callbacks) - before
            if produced > 1:
                errors.append("multiple_callbacks_in_one_app_update")
                break
            if produced == 1 and len(callbacks) in checkpoints:
                print(
                    "TIMELINE_BOUNDARY_CALLBACK "
                    + json.dumps(callbacks[-1], sort_keys=True),
                    flush=True,
                )
        with Usd.EditContext(stage, session_layer):
            timeline.pause()
            timeline.commit()
        receipt["timeline_after_trace"] = _timeline_state_readback(stage, timeline)
        receipt["explicit_app_update_count"] = app_update_index
        receipt["pump_count"] = pump_count
        receipt["physics_scene_time_steps_per_second"] = scene_tps
        validation = validate_callback_trace(
            callbacks,
            physics_hz=PHYSICS_HZ,
            target_callback_count=TARGET_CALLBACK_COUNT,
        )
        errors.extend(str(item) for item in validation["failed_predicates"])
        trace_path = output / "callback-trace.json"
        _write_json(trace_path, callbacks)
        receipt["trace_validation"] = validation
        receipt["artifacts"] = {
            "callback_trace": {
                "path": trace_path.name,
                "sha256": _sha256(trace_path),
                "rows": len(callbacks),
            }
        }
        receipt["errors"] = errors
        receipt["status"] = "passed" if not errors else "failed"
        _write_json(output / "receipt.json", receipt)
        print(
            "TIMELINE_BOUNDARY_RESULT "
            + json.dumps(
                {
                    "status": receipt["status"],
                    "callback_count": len(callbacks),
                    "errors": errors,
                    "max_abs_timing_error_s": validation[
                        "max_abs_timing_error_s"
                    ],
                    "boundary_window": validation["boundary_window"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if not errors else 2
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        receipt["errors"] = errors
        receipt["callbacks_observed"] = len(callbacks)
        receipt["explicit_app_update_count"] = app_update_index
        receipt["traceback"] = traceback.format_exc()
        if callbacks:
            trace_path = output / "callback-trace.json"
            _write_json(trace_path, callbacks)
            receipt["artifacts"] = {
                "callback_trace": {
                    "path": trace_path.name,
                    "sha256": _sha256(trace_path),
                    "rows": len(callbacks),
                }
            }
        _write_json(output / "receipt.json", receipt)
        print(f"TIMELINE_BOUNDARY_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        physics_step_subscription = None
        if timeline is not None:
            timeline.pause()
            timeline.set_looping(False)
            timeline.commit()
        simulation_app.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_probe(
        stage_path=args.stage.resolve(),
        output=args.out.resolve(),
        wall_seconds=args.wall_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())

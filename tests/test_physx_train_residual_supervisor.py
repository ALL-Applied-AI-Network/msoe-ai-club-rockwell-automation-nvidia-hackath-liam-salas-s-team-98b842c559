from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_physx_train_residual_cohort",
    ROOT / "scripts" / "run_physx_train_residual_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)

from fluidmix.physx_train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_inputs(tmp_path: Path):
    plan = build_train_residual_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    plan_path = (tmp_path / "render-plan.json").resolve()
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stage = (tmp_path / "fill_station.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    renderer = (ROOT / "scripts" / "physx_three_pour_sequence.py").resolve()
    qualification_args = supervisor.sequence.build_parser().parse_args(
        [
            "--out",
            str((tmp_path / "qualification-output").resolve()),
            "--stage",
            str(stage),
            "--recipe8-residual-volume-ml",
            "0",
            "--solver-position-iterations",
            "16",
            "--physics-substeps",
            "1",
            "--water-pbd-preset",
            "--count-driven-twin-liquid-mesh",
            "--particle-resolution-profile",
            "c20_2p5mm",
            "--recipe8-effective-particle-volume-ml",
            "0.011272866937444935",
            "--plan-only",
        ]
    )
    qualification_plan = supervisor.sequence._runtime_paint_ball_plan(
        qualification_args
    )
    qualification_schedule = supervisor.sequence.build_runtime_capture_schedule(
        qualification_plan, capture_stride_steps=312
    )
    qualification = {
        "status": "pass",
        "render_path": "count_driven_twin_liquid_mesh",
        "qualification": {"first_stage_only": False},
        "arguments": {
            "seed": 20261025,
            "recipe8_residual_volume_ml": 0.0,
            "solver_position_iterations": 16,
            "physics_substeps": 1,
            "particle_resolution_profile": "c20_2p5mm",
            "recipe8_effective_particle_volume_ml": 0.011272866937444935,
            "count_driven_twin_liquid_mesh": True,
            "rt_subframes": 64,
            "pathtracing_total_spp": 64,
            "capture_stride": 312,
            "video_fps": 10.0,
            "wall_limit_seconds": 10800,
            "nuisance_receipt": None,
            "nuisance_receipt_sha256": None,
            "nuisance_case_id": None,
        },
        "versions": {"isaac": "6.2.0.0", "physx": "110.4.4"},
        "provenance": {
            "requested_stage_sha256": _sha256(stage),
            "script_sha256": _sha256(renderer),
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 64,
            "pathtracing_total_spp_requested": 64,
            "pathtracing_total_spp_readback": 64,
            "pathtracing_spp_requested": 64,
            "pathtracing_spp_readback": 64,
            "warmup_updates": 60,
            "frame_count": len(qualification_schedule["captures"]),
        },
        "plan": qualification_plan,
        "runtime_capture_schedule": qualification_schedule,
        "particle_resolution_profile": qualification_plan[
            "particle_resolution_profile"
        ],
        "source_emission_geometry": {
            "profile_id": "c20_2p5mm",
            "events": [],
            "expected_event_count": sum(
                len(pour["emission_schedule"])
                for pour in qualification_plan["pours"]
            ),
            "observed_event_count": 0,
            "all_support_gates_passed": None,
            "all_centroid_gates_passed": None,
            "all_separation_gates_passed": None,
            "all_flux_gates_passed": None,
            "gate_passed": True,
        },
        "timing": {
            "report_hz": 120,
            "physics_steps_per_second": 120,
            "physics_substeps_per_report_interval": 1,
            "report_interval_count": qualification_schedule["final_plus_3_0_step"],
            "executed_report_intervals": qualification_schedule["final_plus_3_0_step"],
            "physics_callback_count": qualification_schedule["final_plus_3_0_step"],
            "executed_physics_steps": qualification_schedule["final_plus_3_0_step"],
            "callbacks_expected": qualification_schedule["final_plus_3_0_step"],
            "callbacks_observed": qualification_schedule["final_plus_3_0_step"],
            "diffusion_update_count": qualification_schedule["final_plus_3_0_step"],
            "diffusion_updates_per_report_interval": 1,
            "final_edit_target_is_session": True,
            "positions_session_layer_only": True,
            "velocities_session_layer_only": True,
            "orientations_session_layer_only": True,
            "scales_session_layer_only": True,
            "proto_indices_session_layer_only": True,
            "ids_session_layer_only": True,
            "simulation_points_root_layer_absent": True,
            "simulation_points_runtime_readback_passed": True,
        },
    }
    qualification_path = (tmp_path / "qualification.json").resolve()
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_root = (tmp_path / "cohort").resolve()
    return {
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": _canonical_sha256(plan),
        "qualification_path": qualification_path,
        "qualification_sha256": _sha256(qualification_path),
        "stage": stage,
        "renderer": renderer,
        "output_root": output_root,
    }


def _args(inputs: dict[str, object], *extra: str):
    return supervisor.build_parser().parse_args(
        [
            "--plan",
            str(inputs["plan_path"]),
            "--plan-sha256",
            str(inputs["plan_sha256"]),
            "--accepted-qualification-receipt",
            str(inputs["qualification_path"]),
            "--accepted-qualification-receipt-sha256",
            str(inputs["qualification_sha256"]),
            "--output-root",
            str(inputs["output_root"]),
            "--stage",
            str(inputs["stage"]),
            "--renderer-script",
            str(inputs["renderer"]),
            "--python-executable",
            sys.executable,
            "--host-lock",
            str(Path(inputs["output_root"]).parent / "host.lock"),
            *extra,
        ]
    )


def _completed_receipt(command: list[str], inputs: dict[str, object]) -> dict[str, object]:
    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    case_id = value("--train-case-id")
    schedule = next(
        row
        for row in inputs["plan"]["execution_schedules"]
        if row["case_binding"]["case_id"] == case_id
    )
    runtime_schedule = deepcopy(schedule)
    runtime_schedule["timing"]["capture_hz"] = 10
    capture_schedule = supervisor.sequence.build_runtime_capture_schedule(
        runtime_schedule, capture_stride_steps=312
    )
    capture_steps = [
        int(row["physics_step"]) for row in capture_schedule["captures"]
    ]
    video = supervisor.sequence.build_time_preserving_video_repeat_plan(
        capture_steps, physics_hz=120, playback_fps=10.0
    )
    final_step = capture_schedule["final_plus_3_0_step"]
    return {
        "status": "pass",
        "render_path": "count_driven_twin_liquid_mesh",
        "arguments": {
            "seed": int(value("--seed")),
            "physics_substeps": 1,
            "particle_resolution_profile": "c20_2p5mm",
            "recipe8_effective_particle_volume_ml": 0.011272866937444935,
            "resolution": "640x480",
            "rt_subframes": 64,
            "pathtracing_total_spp": 64,
            "capture_stride": 312,
            "video_fps": 10.0,
            "wall_limit_seconds": 10800,
            "train_render_plan_sha256": value("--train-render-plan-sha256"),
            "train_case_id": value("--train-case-id"),
            "accepted_qualification_receipt_sha256": value(
                "--accepted-qualification-receipt-sha256"
            ),
            "count_driven_twin_liquid_mesh": True,
            "nuisance_receipt": None,
            "nuisance_receipt_sha256": None,
            "nuisance_case_id": None,
        },
        "provenance": {
            "requested_stage_sha256": _sha256(Path(inputs["stage"])),
            "script_sha256": _sha256(Path(inputs["renderer"])),
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 64,
            "pathtracing_total_spp_requested": 64,
            "pathtracing_total_spp_readback": 64,
            "pathtracing_spp_requested": 64,
            "pathtracing_spp_readback": 64,
            "warmup_updates": 60,
            "frame_count": len(capture_steps),
            "video_decoded_frame_count": video["video_frame_count"],
        },
        "nuisance_profile_binding": None,
        "nuisance_runtime": None,
        "timing": {
            "report_hz": 120,
            "physics_steps_per_second": 120,
            "physics_substeps_per_report_interval": 1,
            "report_interval_count": final_step,
            "physics_callback_count": final_step,
            "diffusion_update_count": final_step,
            "diffusion_updates_per_report_interval": 1,
            "positions_session_layer_only": True,
            "velocities_session_layer_only": True,
            "orientations_session_layer_only": True,
            "scales_session_layer_only": True,
            "proto_indices_session_layer_only": True,
            "ids_session_layer_only": True,
            "simulation_points_root_layer_absent": True,
            "simulation_points_runtime_readback_passed": True,
        },
        "execution_schedule": {
            "case_id": value("--train-case-id"),
            "cohort_plan_sha256": value("--train-render-plan-sha256"),
            "schedule_file_sha256": _sha256(Path(inputs["plan_path"])),
            "qualification_receipt_sha256": value(
                "--accepted-qualification-receipt-sha256"
            ),
        },
    }


def _success_runner(inputs: dict[str, object], commands: list[list[str]]):
    active = 0

    def run(command, **kwargs):
        nonlocal active
        active += 1
        assert active == 1
        commands.append(list(command))
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "receipt.json").write_text(
            json.dumps(_completed_receipt(command, inputs), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        active -= 1
        return subprocess.CompletedProcess(command, 0)

    return run


def test_supervisor_runs_all_source_bound_cases_serially_with_qualified_mesh_contract(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    commands: list[list[str]] = []
    summary = supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    assert summary["status"] == "pass"
    assert summary["planned_case_count"] == 18
    assert summary["attempts_started"] == 18
    assert summary["processes_completed"] == 18
    assert summary["validated_completed_case_count"] == 18
    assert summary["observed_source_capture_count"] == 444
    assert summary["observed_decoded_video_frame_count"] == 8532
    manifest = json.loads(
        (Path(inputs["output_root"]) / "cohort-launch.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["expected_output_contract"]["case_count"] == 18
    assert manifest["expected_output_contract"]["keep_count"] == 12
    assert manifest["expected_output_contract"]["discard_count"] == 6
    assert manifest["expected_output_contract"]["source_capture_count"] == 444
    assert manifest["expected_output_contract"]["decoded_video_frame_count"] == 8532
    assert len(commands) == 18
    expected_cases = [
        row["case_binding"]["case_id"] for row in inputs["plan"]["execution_schedules"]
    ]
    assert [command[command.index("--train-case-id") + 1] for command in commands] == expected_cases
    assert [int(command[command.index("--seed") + 1]) for command in commands] == [
        330001,
        330002,
        330003,
        330004,
        330005,
        330006,
        330101,
        330102,
        330103,
        330101,
        330102,
        330103,
        330101,
        330102,
        330103,
        330101,
        330102,
        330103,
    ]
    for command in commands:
        assert command[command.index("--rt-subframes") + 1] == "64"
        assert command[command.index("--pathtracing-total-spp") + 1] == "64"
        assert command.count("--count-driven-twin-liquid-mesh") == 1
        assert command[command.index("--warmup") + 1] == "60"
        assert command[command.index("--resolution") + 1] == "640x480"
        assert command[command.index("--capture-stride") + 1] == "312"
        assert command[command.index("--video-fps") + 1] == "10.0"
        assert command[command.index("--physics-substeps") + 1] == "1"
        assert command[command.index("--particle-resolution-profile") + 1] == (
            "c20_2p5mm"
        )
        assert command[
            command.index("--recipe8-effective-particle-volume-ml") + 1
        ] == "0.011272866937444935"
        assert "--nuisance-receipt" not in command
        case_id = command[command.index("--train-case-id") + 1]
        assert (Path(inputs["output_root"]) / "cases" / case_id / "launch-receipt.json").is_file()


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (
            lambda receipt: receipt["timing"].update(
                physics_callback_count=(
                    receipt["timing"]["physics_callback_count"] - 1
                )
            ),
            "physics_callback_count",
        ),
        (
            lambda receipt: receipt["timing"].update(
                positions_session_layer_only=False
            ),
            "authored_rows_session_only",
        ),
        (
            lambda receipt: receipt["timing"].update(
                simulation_points_root_layer_absent=False
            ),
            "simulation_points_root_absent",
        ),
        (
            lambda receipt: receipt["arguments"].update(
                nuisance_case_id="nuisance-case"
            ),
            "nuisance_arguments_absent",
        ),
        (
            lambda receipt: receipt["render"].update(
                frame_count=receipt["render"]["frame_count"] - 1
            ),
            "source_capture_count",
        ),
    ],
)
def test_case_receipt_validator_rejects_legacy_contract_drift(
    tmp_path, monkeypatch, mutation, failure
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    args = _args(inputs)
    binding = supervisor._bind_inputs(args)
    case_id = binding["plan"]["execution_schedules"][0]["case_binding"][
        "case_id"
    ]
    seed = binding["case_seeds"][case_id]
    command = supervisor._case_command(
        args,
        case_id=case_id,
        seed=seed,
        app_output=(tmp_path / "case-output").resolve(),
    )
    receipt = _completed_receipt(command, inputs)
    mutation(receipt)
    receipt_path = (tmp_path / "receipt.json").resolve()
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert failure in supervisor._case_receipt_failures(
        receipt_path,
        args=args,
        binding=binding,
        case_id=case_id,
        seed=seed,
    )


def test_supervisor_stops_on_first_failed_process_and_preserves_denominator(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    commands: list[list[str]] = []
    success = _success_runner(inputs, commands)

    def fail_second(command, **kwargs):
        if len(commands) == 1:
            commands.append(list(command))
            return subprocess.CompletedProcess(command, 7)
        return success(command, **kwargs)

    summary = supervisor.run_supervisor(
        _args(inputs),
        run_process=fail_second,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    case_ids = [
        row["case_binding"]["case_id"] for row in inputs["plan"]["execution_schedules"]
    ]
    assert summary["status"] == "failed"
    assert summary["attempts_started"] == 2
    assert summary["processes_completed"] == 2
    assert summary["cases_passed_this_invocation"] == 1
    assert summary["cases_failed_this_invocation"] == 1
    assert len(commands) == 2
    assert (Path(inputs["output_root"]) / "cases" / case_ids[1] / "exit-code.txt").read_text() == "7\n"
    assert not (Path(inputs["output_root"]) / "cases" / case_ids[2]).exists()


def test_resume_is_explicit_and_only_recognizes_valid_completed_cases(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    first_commands: list[list[str]] = []
    probes = 0

    def stop_before_third():
        nonlocal probes
        probes += 1
        return [] if probes <= 2 else [{"pid": 999, "command": "isaac-sim"}]

    first = supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, first_commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=stop_before_third,
    )
    assert first["status"] == "failed"
    assert first["attempts_started"] == 2
    assert first["validated_completed_case_count"] == 2

    with pytest.raises(ValueError, match="--resume-completed is required"):
        supervisor.run_supervisor(
            _args(inputs),
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )

    resumed_commands: list[list[str]] = []
    resumed = supervisor.run_supervisor(
        _args(inputs, "--resume-completed"),
        run_process=_success_runner(inputs, resumed_commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )
    assert resumed["status"] == "pass"
    assert resumed["previous_completed_cases_recognized"] == 2
    assert resumed["attempts_started"] == 16
    assert resumed["processes_completed"] == 16
    assert resumed["validated_completed_case_count"] == 18
    assert len(resumed_commands) == 16
    assert len(list((Path(inputs["output_root"]) / "supervisor-attempts").glob("attempt-*.json"))) == 2


def test_resume_refuses_an_existing_case_with_drift_instead_of_overwriting(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    probes = 0

    def stop_after_first():
        nonlocal probes
        probes += 1
        return [] if probes == 1 else [{"pid": 999, "command": "isaac-sim"}]

    supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, []),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=stop_after_first,
    )
    first_case = inputs["plan"]["execution_schedules"][0]["case_binding"]["case_id"]
    launch_path = Path(inputs["output_root"]) / "cases" / first_case / "launch-receipt.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["seed"] += 1
    launch_path.write_text(json.dumps(launch, sort_keys=True) + "\n", encoding="utf-8")
    launches = 0

    def forbidden_launch(*args, **kwargs):
        nonlocal launches
        launches += 1
        return subprocess.CompletedProcess([], 0)

    resumed = supervisor.run_supervisor(
        _args(inputs, "--resume-completed"),
        run_process=forbidden_launch,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )
    assert resumed["status"] == "failed"
    assert resumed["failure"]["reason"] == "existing_case_is_not_a_valid_explicit_resume"
    assert resumed["attempts_started"] == 0
    assert launches == 0


def test_resume_rechecks_the_fixed_renderer_contract_in_the_completed_receipt(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    probes = 0

    def stop_after_first():
        nonlocal probes
        probes += 1
        return [] if probes == 1 else [{"pid": 999, "command": "isaac-sim"}]

    supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, []),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=stop_after_first,
    )
    first_case = inputs["plan"]["execution_schedules"][0]["case_binding"]["case_id"]
    receipt_path = Path(inputs["output_root"]) / "cases" / first_case / "app-output" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arguments"]["capture_stride"] = 99
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    launches = 0

    def forbidden_launch(*args, **kwargs):
        nonlocal launches
        launches += 1
        return subprocess.CompletedProcess([], 0)

    resumed = supervisor.run_supervisor(
        _args(inputs, "--resume-completed"),
        run_process=forbidden_launch,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )
    assert resumed["status"] == "failed"
    assert "capture_stride" in resumed["failure"]["evidence"]["receipt_failures"]
    assert resumed["attempts_started"] == 0
    assert launches == 0


def test_supervisor_refuses_an_existing_isaac_process_before_creating_a_case(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    summary = supervisor.run_supervisor(
        _args(inputs),
        run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [{"pid": 123, "command": "isaac-sim"}],
    )
    assert summary["status"] == "failed"
    assert summary["failure"]["reason"] == "other_isaac_or_kit_process_is_running"
    assert summary["attempts_started"] == 0
    assert list((Path(inputs["output_root"]) / "cases").iterdir()) == []


def test_supervisor_rejects_nonfrozen_plan_hash_before_creating_output(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    args = _args(inputs)
    args.plan_sha256 = "0" * 64
    with pytest.raises(ValueError, match="frozen source-bound plan"):
        supervisor.run_supervisor(
            args,
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
    assert not Path(inputs["output_root"]).exists()

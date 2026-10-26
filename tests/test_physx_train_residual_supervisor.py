from __future__ import annotations

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
    qualification = {
        "status": "pass",
        "render_path": "point_instancer",
        "qualification": {"first_stage_only": False},
        "arguments": {
            "recipe8_residual_volume_ml": 0.0,
            "solver_position_iterations": 16,
            "count_driven_twin_liquid_mesh": False,
        },
        "versions": {"isaac": "6.2.0.0", "physx": "110.4.4"},
        "provenance": {
            "requested_stage_sha256": _sha256(stage),
            "script_sha256": _sha256(renderer),
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 16,
            "pathtracing_total_spp_requested": 64,
            "pathtracing_total_spp_readback": 64,
            "pathtracing_spp_requested": 64,
            "pathtracing_spp_readback": 64,
            "warmup_updates": 60,
            "frame_count": 401,
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

    return {
        "status": "pass",
        "arguments": {
            "seed": int(value("--seed")),
            "resolution": "640x480",
            "capture_stride": 12,
            "video_fps": 10.0,
            "wall_limit_seconds": 10800,
            "train_render_plan_sha256": value("--train-render-plan-sha256"),
            "train_case_id": value("--train-case-id"),
            "accepted_qualification_receipt_sha256": value(
                "--accepted-qualification-receipt-sha256"
            ),
        },
        "provenance": {
            "requested_stage_sha256": _sha256(Path(inputs["stage"])),
            "script_sha256": _sha256(Path(inputs["renderer"])),
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 16,
            "pathtracing_total_spp_requested": 64,
            "pathtracing_total_spp_readback": 64,
            "pathtracing_spp_requested": 64,
            "pathtracing_spp_readback": 64,
            "warmup_updates": 60,
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


def test_supervisor_runs_all_source_bound_cases_serially_with_fixed_c10_contract(
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
    assert summary["planned_case_count"] == 9
    assert summary["attempts_started"] == 9
    assert summary["processes_completed"] == 9
    assert summary["validated_completed_case_count"] == 9
    assert len(commands) == 9
    expected_cases = [
        row["case_binding"]["case_id"] for row in inputs["plan"]["execution_schedules"]
    ]
    assert [command[command.index("--train-case-id") + 1] for command in commands] == expected_cases
    assert [int(command[command.index("--seed") + 1]) for command in commands] == [
        340101,
        340102,
        340103,
        340101,
        340102,
        340103,
        340101,
        340102,
        340103,
    ]
    for command in commands:
        assert command[command.index("--rt-subframes") + 1] == "16"
        assert command[command.index("--pathtracing-total-spp") + 1] == "64"
        assert command[command.index("--warmup") + 1] == "60"
        assert command[command.index("--resolution") + 1] == "640x480"
        assert command[command.index("--capture-stride") + 1] == "12"
        assert command[command.index("--video-fps") + 1] == "10"
        case_id = command[command.index("--train-case-id") + 1]
        assert (Path(inputs["output_root"]) / "cases" / case_id / "launch-receipt.json").is_file()


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
    assert resumed["attempts_started"] == 7
    assert resumed["processes_completed"] == 7
    assert resumed["validated_completed_case_count"] == 9
    assert len(resumed_commands) == 7
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
    with pytest.raises(ValueError, match="frozen nine-case plan"):
        supervisor.run_supervisor(
            args,
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
    assert not Path(inputs["output_root"]).exists()

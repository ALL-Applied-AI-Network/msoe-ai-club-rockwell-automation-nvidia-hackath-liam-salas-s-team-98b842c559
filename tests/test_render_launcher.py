"""CPU tests for the external Task 2A launcher."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_twin_dataset.py"


def load_launcher():
    name = "fluidmix_test_render_twin_dataset"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


SETTINGS = {
    "camera": "FillCam",
    "stage": "stations/fill_station/fill_station.usda",
    "frames": "1:2:1",
    "subframes": 32,
    "warmup": 60,
    "max_tries": 20,
    "seed": 20261024,
    "require_liquid_pixels": True,
    "renderer": "PathTracing",
}


def test_launcher_builds_exact_fixed_fillcam_list_command(tmp_path):
    module = load_launcher()
    command = module.build_render_command(
        "/isaac/python",
        "/repo/scripts/twin_render_sequence.py",
        "/public/twin",
        tmp_path / "capture",
        SETTINGS,
    )

    assert type(command) is list
    assert command == [
        "/isaac/python",
        "/repo/scripts/twin_render_sequence.py",
        "--twin",
        "/public/twin",
        "--stage",
        "stations/fill_station/fill_station.usda",
        "--camera",
        "FillCam",
        "--frames",
        "1:2:1",
        "--subframes",
        "32",
        "--warmup",
        "60",
        "--max-tries",
        "20",
        "--seed",
        "20261024",
        "--renderer",
        "PathTracing",
        "--require-liquid-pixels",
        "--out",
        str(tmp_path / "capture"),
    ]
    assert "shell" not in command


def test_launcher_timeout_is_bounded_from_1800_through_7200():
    module = load_launcher()
    assert module.timeout_seconds("1800") == 1800
    assert module.timeout_seconds("7200") == 7200
    for invalid in ("1799", "7201", "1.5"):
        with pytest.raises(Exception):
            module.timeout_seconds(invalid)


def test_launcher_is_import_and_help_safe_without_isaac():
    module = load_launcher()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert hasattr(module, "main")
    assert completed.returncode == 0, completed.stderr
    assert "--experiment-manifest" in completed.stdout
    assert "--environment-profile" in completed.stdout


def _write_experiment(path, run_id):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "hypothesis": "mechanical capture",
                "falsifier": "any gate failure",
                "command_plan": ["fixed FillCam"],
                "seed": 20261024,
                "denominators": {"frames": 1},
                "stop_conditions": ["one attempt"],
            }
        )
    )


def test_launcher_writes_success_manifest_from_verified_fixture(
    tmp_path, monkeypatch
):
    module = load_launcher()
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_render_process import write_complete_capture
    finally:
        sys.path.pop(0)

    twin = tmp_path / "twin"
    stage = twin / "stations/fill_station/fill_station.usda"
    stage.parent.mkdir(parents=True)
    stage.write_text("#usda 1.0\n")
    camera = twin / "cameras/FillCam.json"
    camera.parent.mkdir(parents=True)
    camera.write_text('{"render_px":[3,2]}\n')
    isaac = tmp_path / "isaac-python"
    isaac.write_text("#!/bin/sh\nexit 99\n")
    isaac.chmod(0o700)
    runtime_lock = tmp_path / "requirements.lock"
    runtime_lock.write_text("isaacsim==6.0.1.0\n")
    experiment = tmp_path / "experiment.json"
    _write_experiment(experiment, "run-a1")
    out_parent = tmp_path / "runs"
    out_parent.mkdir()
    calls = []

    def fake_run(command, log_path, *, timeout_seconds, environment):
        calls.append((command, timeout_seconds, environment))
        capture = Path(command[command.index("--out") + 1])
        write_complete_capture(capture)
        Path(log_path).write_text("fixture child\n")
        return {
            "command": command,
            "started_utc": "2026-10-24T00:00:00+00:00",
            "ended_utc": "2026-10-24T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "returncode": 0,
            "timed_out": False,
            "spawn_error": None,
            "termination_escalation": "none",
            "log_path": str(log_path),
            "log_sha256": "a" * 64,
        }

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(
        [
            "--isaac-python",
            str(isaac),
            "--runtime-lock",
            str(runtime_lock),
            "--experiment-manifest",
            str(experiment),
            "--twin",
            str(twin),
            "--run-id",
            "run-a1",
            "--out-parent",
            str(out_parent),
            "--frames",
            "1:2:1",
            "--require-liquid-pixels",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    attempt = json.loads((out_parent / "run-a1/attempt.json").read_text())
    assert attempt["schema_version"] == 1
    assert attempt["status"] == "mechanical_gates_passed"
    assert attempt["visual_check_status"] == "pending"
    assert attempt["failure_reasons"] == []
    assert attempt["provenance"]["twin"]["unchanged"] is True
    assert attempt["provenance"]["experiment_manifest"]["unchanged"] is True
    assert attempt["capture"]["verified"] is True
    assert "child_environment" not in attempt
    assert attempt["environment_policy"] == {
        "name": "explicit-runtime-allowlist-v1",
        "profile": "generic",
    }

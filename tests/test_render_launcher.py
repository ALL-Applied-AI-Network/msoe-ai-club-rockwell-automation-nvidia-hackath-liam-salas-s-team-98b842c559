"""CPU tests for the external Task 2A launcher."""

import importlib.util
import hashlib
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
                "command_plan": {"view": "fixed FillCam"},
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
            "process_group_cleanup_required": False,
            "process_group_quiescent": True,
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
    assert calls[0][0][0] == str(isaac.absolute())
    attempt = json.loads((out_parent / "run-a1/attempt.json").read_text())
    assert attempt["schema_version"] == 1
    assert attempt["status"] == "mechanical_gates_passed"
    assert attempt["visual_check_status"] == "pending"
    assert attempt["failure_reasons"] == []
    assert attempt["provenance"]["twin"]["unchanged"] is True
    assert attempt["provenance"]["experiment_manifest"]["unchanged"] is True
    assert attempt["capture"]["verified"] is True
    assert attempt["provenance"]["files"]["isaac_python"]["before"][
        "lexical_identity"
    ]["kind"] == "regular"
    assert "child_environment" not in attempt
    assert attempt["environment_policy"] == {
        "name": "explicit-runtime-allowlist-v1",
        "profile": "generic",
    }


def test_launcher_preserves_absolute_python_environment_entry_and_binds_target(
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
    target = tmp_path / "managed-python"
    target.write_text("#!/bin/sh\nexit 99\n")
    target.chmod(0o700)
    environment_root = tmp_path / "isaac-venv"
    entry = environment_root / "bin/python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(target)
    (environment_root / "pyvenv.cfg").write_text("home = /managed\n")
    runtime_lock = tmp_path / "requirements.lock"
    runtime_lock.write_text("isaacsim==6.0.1.0\n")
    experiment = tmp_path / "experiment.json"
    _write_experiment(experiment, "lexical-python-a1")
    out_parent = tmp_path / "runs"
    out_parent.mkdir()
    commands = []

    def fake_run(command, log_path, *, timeout_seconds, environment):
        commands.append(command)
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
            "process_group_cleanup_required": False,
            "process_group_quiescent": True,
            "log_path": str(log_path),
            "log_sha256": "f" * 64,
        }

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(
        [
            "--isaac-python",
            str(entry),
            "--runtime-lock",
            str(runtime_lock),
            "--experiment-manifest",
            str(experiment),
            "--twin",
            str(twin),
            "--run-id",
            "lexical-python-a1",
            "--out-parent",
            str(out_parent),
            "--frames",
            "1:2:1",
            "--require-liquid-pixels",
        ]
    )

    assert result == 0
    assert commands[0][0] == str(entry.absolute())
    assert commands[0][0] != str(target.resolve())
    attempt = json.loads(
        (out_parent / "lexical-python-a1/attempt.json").read_text()
    )
    executable = attempt["provenance"]["files"]["isaac_python"]
    assert executable["before"]["lexical_path"] == str(entry.absolute())
    assert executable["before"]["lexical_identity"]["kind"] == "symlink"
    assert executable["before"]["resolved_target"]["path"] == str(target.resolve())
    assert executable["before"]["resolved_target"]["sha256"] == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    assert executable["unchanged"] is True


def test_executable_reverification_rejects_lexical_symlink_retarget(
    tmp_path
):
    module = load_launcher()
    first = tmp_path / "python-first"
    first.write_text("#!/bin/sh\nexit 0\n")
    first.chmod(0o700)
    second = tmp_path / "python-second"
    second.write_text("#!/bin/sh\nexit 0\n")
    second.chmod(0o700)
    entry = tmp_path / "venv/bin/python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(first)
    before = module._resolve_executable(str(entry))
    entry.unlink()
    entry.symlink_to(second)

    verified = module._reverify_executable(before)

    assert verified["unchanged"] is False
    assert verified["before"] == before
    assert verified["after"]["lexical_path"] == str(entry.absolute())
    assert verified["after"]["resolved_target"]["path"] == str(second.resolve())


def test_executable_reverification_rejects_resolved_target_byte_drift(tmp_path):
    module = load_launcher()
    target = tmp_path / "managed-python"
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o700)
    entry = tmp_path / "venv/bin/python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(target)
    before = module._resolve_executable(str(entry))
    target.write_text("#!/bin/sh\nexit 7\n")

    verified = module._reverify_executable(before)

    assert verified["unchanged"] is False
    assert verified["after"]["resolved_target"]["sha256"] != before[
        "resolved_target"
    ]["sha256"]


@pytest.mark.parametrize("mutation", ["retarget", "target_bytes"])
def test_launcher_reports_executable_identity_drift(
    tmp_path, monkeypatch, mutation
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
    target = tmp_path / "managed-python"
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o700)
    replacement = tmp_path / "replacement-python"
    replacement.write_text("#!/bin/sh\nexit 0\n")
    replacement.chmod(0o700)
    entry = tmp_path / "isaac-venv/bin/python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(target)
    runtime_lock = tmp_path / "requirements.lock"
    runtime_lock.write_text("isaacsim==6.0.1.0\n")
    experiment = tmp_path / "experiment.json"
    _write_experiment(experiment, f"executable-{mutation}-a1")
    out_parent = tmp_path / "runs"
    out_parent.mkdir()

    def fake_run(command, log_path, *, timeout_seconds, environment):
        capture = Path(command[command.index("--out") + 1])
        write_complete_capture(capture)
        Path(log_path).write_text("fixture child\n")
        if mutation == "retarget":
            entry.unlink()
            entry.symlink_to(replacement)
        else:
            target.write_text("#!/bin/sh\nexit 7\n")
        return {
            "command": command,
            "started_utc": "2026-10-24T00:00:00+00:00",
            "ended_utc": "2026-10-24T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "returncode": 0,
            "timed_out": False,
            "spawn_error": None,
            "termination_escalation": "none",
            "process_group_cleanup_required": False,
            "process_group_quiescent": True,
            "log_path": str(log_path),
            "log_sha256": "e" * 64,
        }

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    run_id = f"executable-{mutation}-a1"
    result = module.main(
        [
            "--isaac-python",
            str(entry),
            "--runtime-lock",
            str(runtime_lock),
            "--experiment-manifest",
            str(experiment),
            "--twin",
            str(twin),
            "--run-id",
            run_id,
            "--out-parent",
            str(out_parent),
            "--frames",
            "1:2:1",
            "--require-liquid-pixels",
        ]
    )

    assert result == 1
    attempt = json.loads((out_parent / run_id / "attempt.json").read_text())
    assert "isaac_python_changed" in attempt["failure_reasons"]
    executable = attempt["provenance"]["files"]["isaac_python"]
    assert executable["unchanged"] is False
    assert executable["before"] != executable["after"]


def test_executable_entry_normalizes_relative_and_bare_inputs_to_absolute_lexical_paths(
    tmp_path, monkeypatch
):
    module = load_launcher()
    executable = tmp_path / "bin/isaac-python"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.chdir(tmp_path)

    relative = module._resolve_executable("bin/isaac-python")
    monkeypatch.setenv("PATH", str(executable.parent))
    bare = module._resolve_executable("isaac-python")

    assert relative["lexical_path"] == str(executable.absolute())
    assert bare["lexical_path"] == str(executable.absolute())
    assert Path(relative["lexical_path"]).is_absolute()
    assert Path(bare["lexical_path"]).is_absolute()


def test_executable_entry_rejects_dangling_directory_and_nonexecutable_inputs(
    tmp_path
):
    module = load_launcher()
    dangling = tmp_path / "dangling-python"
    dangling.symlink_to(tmp_path / "missing-target")
    directory = tmp_path / "python-directory"
    directory.mkdir()
    nonexecutable = tmp_path / "python-no-exec"
    nonexecutable.write_text("#!/bin/sh\nexit 0\n")
    nonexecutable.chmod(0o600)

    for invalid in (dangling, directory, nonexecutable):
        with pytest.raises(ValueError, match="Isaac Python"):
            module._resolve_executable(str(invalid))


def test_launcher_fails_when_normal_exit_required_process_group_cleanup(
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
    _write_experiment(experiment, "cleanup-a1")
    out_parent = tmp_path / "runs"
    out_parent.mkdir()

    def fake_run(command, log_path, *, timeout_seconds, environment):
        capture = Path(command[command.index("--out") + 1])
        write_complete_capture(capture)
        Path(log_path).write_text("leader exited; descendant terminated\n")
        return {
            "command": command,
            "started_utc": "2026-10-24T00:00:00+00:00",
            "ended_utc": "2026-10-24T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "returncode": 0,
            "timed_out": False,
            "spawn_error": None,
            "termination_escalation": "sigterm",
            "process_group_cleanup_required": True,
            "process_group_quiescent": True,
            "log_path": str(log_path),
            "log_sha256": "b" * 64,
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
            "cleanup-a1",
            "--out-parent",
            str(out_parent),
            "--frames",
            "1:2:1",
            "--require-liquid-pixels",
        ]
    )

    assert result == 1
    attempt = json.loads((out_parent / "cleanup-a1/attempt.json").read_text())
    assert attempt["capture"]["verified"] is True
    assert attempt["status"] == "failed"
    assert "renderer_process_group_cleanup_required" in attempt["failure_reasons"]


def test_launcher_records_isolated_gb10_preload_provenance_for_regular_lexical_file(
    tmp_path, monkeypatch
):
    module = load_launcher()
    import fluidmix.render_process as render_process

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
    _write_experiment(experiment, "gb10-a1")
    out_parent = tmp_path / "runs"
    out_parent.mkdir()
    preload_snapshot = {
        "lexical_path": "/lib/aarch64-linux-gnu/libgomp.so.1",
        "lexical_identity": {
            "kind": "regular",
            "link_target": None,
            "device": 1,
            "inode": 2,
            "mode": 0o100644,
            "size": 8,
            "mtime_ns": 3,
            "ctime_ns": 4,
        },
        "resolved_target": {
            "path": "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
            "device": 1,
            "inode": 2,
            "mode": 0o100644,
            "size": 8,
            "mtime_ns": 3,
            "ctime_ns": 4,
            "sha256": "e" * 64,
            "bytes": 8,
        },
    }
    snapshot_calls = []

    def fake_snapshot():
        snapshot_calls.append("before")
        return preload_snapshot

    def fake_reverify(before):
        assert before == preload_snapshot
        return {
            "before": before,
            "after": preload_snapshot,
            "unchanged": True,
        }

    monkeypatch.setattr(render_process, "GB10_PRELOAD", tmp_path / "missing.so")
    monkeypatch.setattr(render_process, "snapshot_gb10_preload", fake_snapshot)
    monkeypatch.setattr(module, "snapshot_gb10_preload", fake_snapshot)
    monkeypatch.setattr(module, "reverify_gb10_preload", fake_reverify)

    def fake_run(command, log_path, *, timeout_seconds, environment):
        assert environment["LD_PRELOAD"] == "/lib/aarch64-linux-gnu/libgomp.so.1"
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
            "process_group_cleanup_required": False,
            "process_group_quiescent": True,
            "log_path": str(log_path),
            "log_sha256": "c" * 64,
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
            "gb10-a1",
            "--out-parent",
            str(out_parent),
            "--frames",
            "1:2:1",
            "--require-liquid-pixels",
            "--environment-profile",
            "gb10-aarch64",
        ]
    )

    assert result == 0
    attempt = json.loads((out_parent / "gb10-a1/attempt.json").read_text())
    preload = attempt["provenance"]["gb10_preload"]
    assert snapshot_calls == ["before", "before"]
    assert preload["unchanged"] is True
    assert preload["before"] == preload_snapshot
    assert preload["before"]["lexical_identity"]["kind"] == "regular"
    assert len(preload["before"]["resolved_target"]["sha256"]) == 64


def test_launcher_fails_with_truthful_hashes_when_experiment_manifest_drifts(
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
    _write_experiment(experiment, "manifest-drift-a1")
    before_hash = hashlib.sha256(experiment.read_bytes()).hexdigest()
    out_parent = tmp_path / "runs"
    out_parent.mkdir()

    def fake_run(command, log_path, *, timeout_seconds, environment):
        payload = json.loads(experiment.read_text())
        payload["hypothesis"] = "changed during the child attempt"
        experiment.write_text(json.dumps(payload, sort_keys=True) + "\n")
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
            "process_group_cleanup_required": False,
            "process_group_quiescent": True,
            "log_path": str(log_path),
            "log_sha256": "d" * 64,
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
            "manifest-drift-a1",
            "--out-parent",
            str(out_parent),
            "--frames",
            "1:2:1",
            "--require-liquid-pixels",
        ]
    )

    after_hash = hashlib.sha256(experiment.read_bytes()).hexdigest()
    attempt = json.loads(
        (out_parent / "manifest-drift-a1/attempt.json").read_text()
    )
    recorded = attempt["provenance"]["experiment_manifest"]
    assert result == 1
    assert attempt["status"] == "failed"
    assert "experiment_manifest_changed" in attempt["failure_reasons"]
    assert recorded["sha256"] == before_hash
    assert recorded["sha256_after"] == after_hash
    assert recorded["sha256"] != recorded["sha256_after"]
    assert recorded["unchanged"] is False

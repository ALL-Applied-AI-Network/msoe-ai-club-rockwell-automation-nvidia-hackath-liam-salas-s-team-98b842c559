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


def _write_appearance_experiment(path, run_id, appearance_plan):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "hypothesis": "direct appearance survives the bounded capture boundary",
                "falsifier": "any lifecycle, provenance, or capture gate fails",
                "command_plan": {
                    "view": "fixed FillCam",
                    "appearance_layer": appearance_plan,
                },
                "seed": 20261024,
                "denominators": {"frames": 1},
                "stop_conditions": ["one attempt"],
            },
            sort_keys=True,
        )
        + "\n"
    )


def _appearance_launcher_fixture(tmp_path, run_id="appearance-a1"):
    from fluidmix.twin_appearance import (
        appearance_command_plan,
        create_appearance_layer,
    )

    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_twin_appearance import _write_twin
    finally:
        sys.path.pop(0)
    twin = tmp_path / "twin"
    _write_twin(twin)
    camera = twin / "cameras/FillCam.json"
    camera.parent.mkdir(parents=True)
    camera.write_text('{"render_px":[3,2]}\n')
    out_parent = tmp_path / "runs"
    out_parent.mkdir()
    appearance = create_appearance_layer(
        twin, out_parent, "R002", [0.1392, 0.176, 0.213], 0.6
    )
    isaac = tmp_path / "isaac-python"
    isaac.write_text("#!/bin/sh\nexit 99\n")
    isaac.chmod(0o700)
    runtime_lock = tmp_path / "requirements.lock"
    runtime_lock.write_text("isaacsim==6.0.1.0\n")
    experiment = tmp_path / "experiment.json"
    _write_appearance_experiment(
        experiment, run_id, appearance_command_plan(appearance)
    )
    return {
        "twin": twin,
        "out_parent": out_parent,
        "appearance": appearance,
        "isaac": isaac,
        "runtime_lock": runtime_lock,
        "experiment": experiment,
        "run_id": run_id,
    }


def _stage_appearance_launcher_fixture(tmp_path, run_id="stage-appearance-a1"):
    from fluidmix.twin_stage_appearance import (
        create_stage_appearance_layer,
        stage_appearance_command_plan,
    )

    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_twin_stage_appearance import _write_twin
    finally:
        sys.path.pop(0)
    twin = tmp_path / "twin"
    _write_twin(twin)
    camera = twin / "cameras/FillCam.json"
    camera.parent.mkdir(parents=True)
    camera.write_text('{"render_px":[3,2]}\n')
    out_parent = tmp_path / "runs"
    out_parent.mkdir()
    appearance = create_stage_appearance_layer(
        twin,
        out_parent,
        "STAGE001",
        [
            {
                "time_code": 171,
                "diffuse_color_linear": [0.1, 0.2, 0.3],
                "opacity": 0.6,
            },
            {
                "time_code": 342,
                "diffuse_color_linear": [0.2, 0.3, 0.4],
                "opacity": 0.6,
            },
            {
                "time_code": 686,
                "diffuse_color_linear": [0.3, 0.4, 0.5],
                "opacity": 0.6,
            },
        ],
    )
    isaac = tmp_path / "isaac-python"
    isaac.write_text("#!/bin/sh\nexit 99\n")
    isaac.chmod(0o700)
    runtime_lock = tmp_path / "requirements.lock"
    runtime_lock.write_text("isaacsim==6.2.0.0\n")
    experiment = tmp_path / "experiment.json"
    _write_appearance_experiment(
        experiment, run_id, stage_appearance_command_plan(appearance)
    )
    return {
        "twin": twin,
        "out_parent": out_parent,
        "appearance": appearance,
        "isaac": isaac,
        "runtime_lock": runtime_lock,
        "experiment": experiment,
        "run_id": run_id,
    }


def _write_appearance_capture(command, appearance, *, stack_damage=None):
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_render_process import write_complete_capture
    finally:
        sys.path.pop(0)
    capture = Path(command[command.index("--out") + 1])
    summary = write_complete_capture(capture)
    summary["inputs"]["stage"] = {
        "requested": appearance["path"],
        "resolved_path": appearance["path"],
        "sha256": appearance["stage_sha256"],
    }
    stack = [
        {**record, "identifier": record["resolved_path"]}
        for record in appearance["non_session_usd_layer_stack"]
    ]
    if stack_damage == "reordered":
        stack[1], stack[2] = stack[2], stack[1]
    elif stack_damage == "extra":
        stack.append(dict(stack[-1], ordinal=len(stack) + 1))
    summary["inputs"]["non_session_usd_layer_stack"] = stack
    (capture / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _appearance_argv(fixture):
    return [
        "--isaac-python",
        str(fixture["isaac"]),
        "--runtime-lock",
        str(fixture["runtime_lock"]),
        "--experiment-manifest",
        str(fixture["experiment"]),
        "--twin",
        str(fixture["twin"]),
        "--run-id",
        fixture["run_id"],
        "--out-parent",
        str(fixture["out_parent"]),
        "--appearance-stage",
        fixture["appearance"]["path"],
        "--frames",
        "1:2:1",
        "--require-liquid-pixels",
    ]


def _stage_appearance_argv(fixture):
    argv = _appearance_argv(fixture)
    argv[argv.index("--appearance-stage")] = "--stage-appearance-stage"
    argv[argv.index("--frames") + 1] = "171,342,686"
    return argv


def _write_stage_appearance_capture(command, appearance):
    _write_appearance_capture(command, appearance)
    capture = Path(command[command.index("--out") + 1])
    summary_path = capture / "summary.json"
    summary = json.loads(summary_path.read_text())
    base_frame = summary["frames"][0]
    frames = []
    source_paths = {
        role: capture / record["path"]
        for role, record in base_frame["artifacts"].items()
    }
    for ordinal, time_code in enumerate((171, 342, 686), start=1):
        stem = f"{time_code:04d}"
        frame = json.loads(json.dumps(base_frame))
        frame["capture_ordinal"] = ordinal
        frame["time_code"] = time_code
        for role, source in source_paths.items():
            suffix = source.suffix
            destination = capture / role / f"{stem}{suffix}"
            destination.write_bytes(source.read_bytes())
            frame["artifacts"][role]["path"] = f"{role}/{stem}{suffix}"
        frames.append(frame)
    for source in source_paths.values():
        source.unlink()
    summary["render_settings"]["frame_spec"] = "171,342,686"
    summary["render_settings"]["time_codes"] = [171, 342, 686]
    summary["frame_count"] = 3
    summary["frames"] = frames
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _successful_process(command, log_path):
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
        "log_sha256": "9" * 64,
    }


def test_launcher_stage_options_are_mutually_exclusive_and_help_safe_without_pxr():
    module = load_launcher()
    completed = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--appearance-stage" in completed.stdout
    assert "--stage-appearance-stage" in completed.stdout
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--isaac-python",
                "/python",
                "--runtime-lock",
                "/lock",
                "--experiment-manifest",
                "/experiment",
                "--twin",
                "/twin",
                "--run-id",
                "run-a1",
                "--out-parent",
                "/runs",
                "--stage",
                "stations/fill_station/fill_station.usda",
                "--appearance-stage",
                "/runs/appearance-layers/R000.usda",
            ]
        )


def test_launcher_accepts_exact_time_sampled_stage_appearance_and_three_frames(
    tmp_path, monkeypatch
):
    module = load_launcher()
    fixture = _stage_appearance_launcher_fixture(tmp_path)
    calls = []

    def fake_run(command, log_path, *, timeout_seconds, environment):
        calls.append(command)
        _write_stage_appearance_capture(command, fixture["appearance"])
        return _successful_process(command, log_path)

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(_stage_appearance_argv(fixture))

    assert result == 0
    assert calls[0][calls[0].index("--frames") + 1] == "171,342,686"
    assert calls[0][calls[0].index("--stage") + 1] == fixture["appearance"]["path"]
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert attempt["status"] == "mechanical_gates_passed"
    assert attempt["capture"]["verified"] is True
    assert attempt["capture"]["appearance_provenance_verified"] is True
    assert attempt["requested_settings"]["frames"] == "171,342,686"
    provenance = attempt["provenance"]["appearance_layer"]
    assert provenance["unchanged"] is True
    assert provenance["inspection_before"]["schema"] == "direct_stage_appearance_v1"
    assert provenance["inspection_after"] == provenance["inspection_before"]
    assert attempt["provenance"]["files"]["stage_appearance_module"][
        "unchanged"
    ] is True


def test_launcher_rejects_nonmatching_time_sampled_preregistration_before_child(
    tmp_path, monkeypatch
):
    module = load_launcher()
    fixture = _stage_appearance_launcher_fixture(tmp_path, "stage-prereg-drift-a1")
    payload = json.loads(fixture["experiment"].read_text())
    payload["command_plan"]["appearance_layer"]["samples"][1]["opacity"] = 0.7
    fixture["experiment"].write_text(json.dumps(payload) + "\n")
    calls = []
    monkeypatch.setattr(
        module, "run_bounded_command", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(SystemExit):
        module.main(_stage_appearance_argv(fixture))

    assert calls == []
    assert not (fixture["out_parent"] / fixture["run_id"]).exists()


def test_launcher_rejects_stage_frames_missing_an_authored_sample_time(
    tmp_path, monkeypatch
):
    module = load_launcher()
    fixture = _stage_appearance_launcher_fixture(tmp_path, "stage-frame-drift-a1")
    argv = _stage_appearance_argv(fixture)
    argv[argv.index("--frames") + 1] = "171,686"
    calls = []
    monkeypatch.setattr(
        module, "run_bounded_command", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(SystemExit):
        module.main(argv)

    assert calls == []
    assert not (fixture["out_parent"] / fixture["run_id"]).exists()


def test_launcher_accepts_video_frames_when_all_authored_stage_samples_are_included(
    tmp_path, monkeypatch
):
    module = load_launcher()
    fixture = _stage_appearance_launcher_fixture(tmp_path, "stage-video-subset-a1")
    argv = _stage_appearance_argv(fixture)
    time_codes = sorted(set(range(31, 687, 5)) | {342})
    argv[argv.index("--frames") + 1] = ",".join(str(code) for code in time_codes)
    calls = []

    def accepted_child(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("accepted stage-subset probe")

    monkeypatch.setattr(module, "run_bounded_command", accepted_child)

    result = module.main(argv)

    assert result == 1
    assert len(calls) == 1
    command = calls[0][0][0]
    assert command[command.index("--frames") + 1] == argv[argv.index("--frames") + 1]
    assert {171, 342, 686}.issubset(time_codes)


def test_launcher_accepts_exact_preregistered_appearance_and_binds_capture_stack(
    tmp_path, monkeypatch
):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(tmp_path)
    calls = []

    def fake_run(command, log_path, *, timeout_seconds, environment):
        calls.append(command)
        _write_appearance_capture(command, fixture["appearance"])
        return _successful_process(command, log_path)

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(_appearance_argv(fixture))

    assert result == 0
    assert calls[0][calls[0].index("--stage") + 1] == fixture["appearance"]["path"]
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert attempt["status"] == "mechanical_gates_passed"
    assert attempt["capture"]["verified"] is True
    assert attempt["capture"]["appearance_provenance_verified"] is True
    assert attempt["requested_settings"]["stage"] == fixture["appearance"]["path"]
    appearance_provenance = attempt["provenance"]["appearance_layer"]
    assert set(appearance_provenance) == {
        "required",
        "path",
        "sha256",
        "bytes",
        "sha256_after",
        "bytes_after",
        "unchanged",
        "inspection_before",
        "inspection_after",
    }
    assert appearance_provenance["unchanged"] is True
    assert appearance_provenance["sha256"] == fixture["appearance"]["stage_sha256"]
    assert appearance_provenance["sha256_after"] == fixture["appearance"][
        "stage_sha256"
    ]
    assert appearance_provenance["inspection_before"] == fixture["appearance"]
    assert appearance_provenance["inspection_after"] == fixture["appearance"]
    assert attempt["provenance"]["files"]["appearance_module"]["unchanged"] is True


@pytest.mark.parametrize("manifest_damage", ["wrong_value", "extra_key"])
def test_launcher_rejects_nonmatching_appearance_preregistration_before_child(
    tmp_path, monkeypatch, manifest_damage
):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(
        tmp_path, f"appearance-prereg-{manifest_damage}-a1"
    )
    payload = json.loads(fixture["experiment"].read_text())
    if manifest_damage == "wrong_value":
        payload["command_plan"]["appearance_layer"]["opacity"] = 0.7
    else:
        payload["command_plan"]["appearance_layer"]["unexpected"] = True
    fixture["experiment"].write_text(json.dumps(payload) + "\n")
    calls = []
    monkeypatch.setattr(module, "run_bounded_command", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit):
        module.main(_appearance_argv(fixture))

    assert calls == []
    assert not (fixture["out_parent"] / fixture["run_id"]).exists()


@pytest.mark.parametrize("path_damage", ["relative", "outside", "symlink"])
def test_launcher_rejects_unconfined_or_symlinked_appearance_paths_before_child(
    tmp_path, monkeypatch, path_damage
):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(
        tmp_path, f"appearance-path-{path_damage}-a1"
    )
    argv = _appearance_argv(fixture)
    stage_index = argv.index("--appearance-stage") + 1
    if path_damage == "relative":
        argv[stage_index] = "appearance-layers/R002.usda"
    elif path_damage == "outside":
        outside = tmp_path / "R002.usda"
        outside.write_bytes(Path(fixture["appearance"]["path"]).read_bytes())
        argv[stage_index] = str(outside)
    else:
        linked = fixture["out_parent"] / "appearance-layers/LINK.usda"
        linked.symlink_to(fixture["appearance"]["path"])
        argv[stage_index] = str(linked)
    calls = []
    monkeypatch.setattr(module, "run_bounded_command", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit):
        module.main(argv)

    assert calls == []
    assert not (fixture["out_parent"] / fixture["run_id"]).exists()


def test_launcher_turns_missing_openusd_into_a_preflight_cli_error(
    tmp_path, monkeypatch
):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(tmp_path, "appearance-no-pxr-a1")
    calls = []
    monkeypatch.setattr(
        module,
        "inspect_appearance_layer",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pxr unavailable")),
    )
    monkeypatch.setattr(module, "run_bounded_command", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit):
        module.main(_appearance_argv(fixture))

    assert calls == []
    assert not (fixture["out_parent"] / fixture["run_id"]).exists()


def test_launcher_records_appearance_byte_drift_after_child(tmp_path, monkeypatch):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(tmp_path, "appearance-drift-a1")

    def fake_run(command, log_path, *, timeout_seconds, environment):
        _write_appearance_capture(command, fixture["appearance"])
        with Path(fixture["appearance"]["path"]).open("a") as stream:
            stream.write("\n# drift after child\n")
        return _successful_process(command, log_path)

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(_appearance_argv(fixture))

    assert result == 1
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert "appearance_stage_changed" in attempt["failure_reasons"]
    assert attempt["provenance"]["appearance_layer"]["unchanged"] is False


def test_launcher_records_appearance_unreadable_after_child(tmp_path, monkeypatch):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(tmp_path, "appearance-missing-a1")

    def fake_run(command, log_path, *, timeout_seconds, environment):
        _write_appearance_capture(command, fixture["appearance"])
        Path(fixture["appearance"]["path"]).unlink()
        return _successful_process(command, log_path)

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(_appearance_argv(fixture))

    assert result == 1
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert "appearance_stage_unreadable_after" in attempt["failure_reasons"]
    assert attempt["provenance"]["appearance_layer"]["unchanged"] is False


def test_launcher_rejects_capture_root_stage_hash_mismatch(tmp_path, monkeypatch):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(tmp_path, "appearance-stage-hash-a1")

    def fake_run(command, log_path, *, timeout_seconds, environment):
        _write_appearance_capture(command, fixture["appearance"])
        capture = Path(command[command.index("--out") + 1])
        summary_path = capture / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["inputs"]["stage"]["sha256"] = "0" * 64
        summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
        return _successful_process(command, log_path)

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(_appearance_argv(fixture))

    assert result == 1
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert "capture_stage_provenance_mismatch" in attempt["failure_reasons"]


@pytest.mark.parametrize("stack_damage", ["reordered", "extra"])
def test_launcher_rejects_nonexact_capture_layer_stack(
    tmp_path, monkeypatch, stack_damage
):
    module = load_launcher()
    fixture = _appearance_launcher_fixture(
        tmp_path, f"appearance-stack-{stack_damage}-a1"
    )

    def fake_run(command, log_path, *, timeout_seconds, environment):
        _write_appearance_capture(
            command, fixture["appearance"], stack_damage=stack_damage
        )
        return _successful_process(command, log_path)

    monkeypatch.setattr(module, "run_bounded_command", fake_run)
    result = module.main(_appearance_argv(fixture))

    assert result == 1
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert "capture_stage_provenance_mismatch" in attempt["failure_reasons"]
    assert attempt["capture"]["verified"] is True
    assert attempt["capture"]["appearance_provenance_verified"] is False


@pytest.mark.parametrize("renderer_returncode", [0, 7])
def test_launcher_requires_zero_returncode_for_verified_fixture(
    tmp_path, monkeypatch, renderer_returncode
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
            "returncode": renderer_returncode,
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

    assert result == (0 if renderer_returncode == 0 else 1)
    assert len(calls) == 1
    assert calls[0][0][0] == str(isaac.absolute())
    attempt = json.loads((out_parent / "run-a1/attempt.json").read_text())
    assert attempt["schema_version"] == 1
    assert attempt["status"] == (
        "mechanical_gates_passed" if renderer_returncode == 0 else "failed"
    )
    assert attempt["visual_check_status"] == "pending"
    assert attempt["failure_reasons"] == (
        [] if renderer_returncode == 0 else ["renderer_returncode_nonzero"]
    )
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

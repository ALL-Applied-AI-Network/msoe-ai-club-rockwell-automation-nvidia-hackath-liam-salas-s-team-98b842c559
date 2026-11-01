"""CPU-only contracts for the local Isaac Sim 6.2 process watchdog."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from copy import deepcopy

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import twin.render.render_process as render_process


READY = "Simulation App Startup Complete"


def _policy(**overrides: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "max_process_attempts": 2,
        "no_output_timeout_seconds": 0.12,
        "startup_readiness_timeout_seconds": 0.30,
        "fast_crash_retry_seconds": 0.20,
        "startup_readiness_marker": READY,
    }
    policy.update(overrides)
    return policy


def _logical_root(tmp_path: Path, name: str) -> Path:
    parent = tmp_path / "runs"
    parent.mkdir()
    return render_process.create_attempt_root(parent, name)


def _write_fake_child(tmp_path: Path) -> Path:
    child = tmp_path / "fake_renderer.py"
    child.write_text(
        """
import json
import os
from pathlib import Path
import sys
import time

mode, counter_path, observations_path = sys.argv[1:4]
counter = Path(counter_path)
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
with Path(observations_path).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "argv": sys.argv[1:],
        "frozen_setting": os.environ.get("FROZEN_SETTING"),
    }, sort_keys=True) + "\\n")

if mode == "no-output-then-success" and attempt == 1:
    time.sleep(5)
elif mode == "not-ready-then-success" and attempt == 1:
    print("Kit boot began", flush=True)
    time.sleep(5)
elif mode == "crash-then-success" and attempt == 1:
    raise SystemExit(17)
elif mode == "ready-crash-then-success" and attempt == 1:
    print("Simulation App Startup Complete", flush=True)
    raise SystemExit(139)
elif mode == "always-crash":
    raise SystemExit(19)
else:
    print("Simulation App Startup Complete", flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    return child


def _run_scenario(tmp_path: Path, mode: str) -> tuple[dict[str, object], Path, Path]:
    child = _write_fake_child(tmp_path)
    counter = tmp_path / "counter.txt"
    observations = tmp_path / "observations.jsonl"
    root = _logical_root(tmp_path, "logical-render")
    command = [
        sys.executable,
        str(child),
        mode,
        str(counter),
        str(observations),
        "--seed",
        "20261024",
        "--frames",
        "171:687:171",
        "--out",
        "capture",
    ]
    result = render_process.run_retriable_render_command(
        command,
        root,
        timeout_seconds=2,
        environment={**os.environ, "FROZEN_SETTING": "unchanged"},
        attempt_policy=_policy(),
    )
    return result, root, observations


def test_no_output_watchdog_records_attempt_then_retries_once(tmp_path):
    result, root, _ = _run_scenario(tmp_path, "no-output-then-success")

    assert result["status"] == "process_succeeded"
    assert result["attempt_count"] == 2
    first, second = result["attempts"]
    assert first["retry_trigger"] == "no_log_output_180s"
    assert first["process"]["watchdog_trigger"] == "no_log_output"
    assert first["process"]["process_group_quiescent"] is True
    assert second["retry_trigger"] is None
    assert second["outcome"] == "succeeded"
    assert (root / "process-attempt-01/render.log").exists()
    assert (root / "process-attempt-02/render.log").read_text() == READY + "\n"


def test_readiness_watchdog_records_attempt_then_retries_once(tmp_path):
    result, _, _ = _run_scenario(tmp_path, "not-ready-then-success")

    assert result["status"] == "process_succeeded"
    assert result["attempt_count"] == 2
    first = result["attempts"][0]
    assert first["retry_trigger"] == "startup_readiness_missing_420s"
    assert first["process"]["watchdog_trigger"] == "startup_readiness_missing"
    assert first["process"]["log_output_observed"] is True
    assert first["process"]["startup_readiness_observed"] is False


def test_fast_nonzero_crash_retries_once(tmp_path):
    result, _, _ = _run_scenario(tmp_path, "crash-then-success")

    assert result["status"] == "process_succeeded"
    assert result["attempt_count"] == 2
    assert result["attempts"][0]["retry_trigger"] == "fast_nonzero_exit"
    assert result["attempts"][0]["process"]["returncode"] == 17
    assert result["attempts"][1]["outcome"] == "succeeded"


def test_fast_nonzero_crash_after_readiness_still_retries_once(tmp_path):
    result, _, _ = _run_scenario(tmp_path, "ready-crash-then-success")

    assert result["status"] == "process_succeeded"
    assert result["attempt_count"] == 2
    assert result["attempts"][0]["process"]["startup_readiness_observed"] is True
    assert result["attempts"][0]["process"]["returncode"] == 139
    assert result["attempts"][0]["retry_trigger"] == "fast_nonzero_exit"
    assert result["attempts"][1]["outcome"] == "succeeded"


def test_second_failure_stops_without_a_third_attempt(tmp_path):
    result, root, _ = _run_scenario(tmp_path, "always-crash")

    assert result["status"] == "failed"
    assert result["attempt_count"] == 2
    assert [row["process"]["returncode"] for row in result["attempts"]] == [19, 19]
    assert result["attempts"][0]["retry_trigger"] == "fast_nonzero_exit"
    assert result["attempts"][1]["retry_trigger"] is None
    assert not (root / "process-attempt-03").exists()


def test_success_does_not_retry(tmp_path):
    result, root, _ = _run_scenario(tmp_path, "success")

    assert result["status"] == "process_succeeded"
    assert result["attempt_count"] == 1
    assert result["selected_attempt_number"] == 1
    assert not (root / "process-attempt-02").exists()


def test_attempts_receive_exact_same_argv_and_environment_settings(tmp_path):
    result, _, observations = _run_scenario(tmp_path, "crash-then-success")
    observed = [json.loads(line) for line in observations.read_text().splitlines()]

    assert result["attempt_count"] == 2
    assert result["attempts"][0]["command"] == result["attempts"][1]["command"]
    assert observed[0] == observed[1]
    assert observed[0]["frozen_setting"] == "unchanged"
    assert result["attempts"][0]["environment_sha256"] == result["attempts"][1][
        "environment_sha256"
    ]


def test_local_isaac62_manifest_binds_exact_host_runtime_version_and_two_attempts():
    payload = {
        "render_runtime": dict(render_process.LOCAL_ISAAC62_RUNTIME_CONTRACT),
        "process_attempt_policy": dict(render_process.LOCAL_ISAAC62_ATTEMPT_POLICY),
    }

    validated = render_process.validate_local_isaac62_manifest_binding(
        payload,
        isaac_python=render_process.LOCAL_ISAAC62_PYTHON,
        host_name="spark5",
        machine="aarch64",
    )

    assert validated["render_runtime"]["isaac_sim_version"] == "6.2.0.0"
    assert validated["render_runtime"]["kit_version"] == "110.4.0"
    assert validated["process_attempt_policy"]["max_process_attempts"] == 2


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("render_runtime", "host_name", "some-other-host"),
        ("render_runtime", "isaac_python", "/tmp/python"),
        ("render_runtime", "isaac_sim_version", "6.0.1.0"),
        ("process_attempt_policy", "max_process_attempts", 3),
        ("process_attempt_policy", "no_output_timeout_seconds", 181),
    ],
)
def test_local_isaac62_manifest_rejects_runtime_or_watchdog_drift(
    section, field, value
):
    payload = {
        "render_runtime": dict(render_process.LOCAL_ISAAC62_RUNTIME_CONTRACT),
        "process_attempt_policy": dict(render_process.LOCAL_ISAAC62_ATTEMPT_POLICY),
    }
    payload[section][field] = value

    with pytest.raises(ValueError, match="local Isaac 6.2"):
        render_process.validate_local_isaac62_manifest_binding(
            payload,
            isaac_python=render_process.LOCAL_ISAAC62_PYTHON,
            host_name="spark5",
            machine="aarch64",
        )


def test_local_isaac62_environment_keeps_exact_preload_and_eula(monkeypatch):
    monkeypatch.setattr(
        render_process, "snapshot_gb10_preload", lambda: {"checked": True}
    )

    child = render_process.build_child_environment(
        {"PATH": "/usr/bin", "LD_PRELOAD": "/wrong.so"},
        render_process.LOCAL_ISAAC62_PROFILE,
    )

    assert child["LD_PRELOAD"] == "/lib/aarch64-linux-gnu/libgomp.so.1"
    assert child["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_launcher_local62_profile_uses_retriable_boundary_and_relative_output(
    tmp_path, monkeypatch
):
    tests_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(tests_root))
    try:
        import tests.twin.render.test_launcher as launcher_fixture

        module = launcher_fixture.load_launcher()
        fixture = launcher_fixture._appearance_launcher_fixture(
            tmp_path, run_id="local62-a1"
        )
    finally:
        sys.path.pop(0)

    manifest = json.loads(fixture["experiment"].read_text())
    manifest["render_runtime"] = dict(render_process.LOCAL_ISAAC62_RUNTIME_CONTRACT)
    manifest["process_attempt_policy"] = dict(
        render_process.LOCAL_ISAAC62_ATTEMPT_POLICY
    )
    fixture["experiment"].write_text(json.dumps(manifest), encoding="utf-8")
    fixture["runtime_lock"].write_text("isaacsim==6.2.0.0\n", encoding="utf-8")

    executable = module._resolve_executable(str(fixture["isaac"]))
    executable["lexical_path"] = render_process.LOCAL_ISAAC62_PYTHON
    monkeypatch.setattr(module, "_resolve_executable", lambda unused: deepcopy(executable))
    metadata_file = tmp_path / "isaacsim-METADATA"
    metadata_file.write_text("Name: isaacsim\nVersion: 6.2.0.0\n", encoding="utf-8")
    metadata = {
        "path": str(metadata_file),
        **render_process.hash_regular_file(metadata_file),
        "distribution_name": "isaacsim",
        "distribution_version": "6.2.0.0",
    }
    monkeypatch.setattr(
        module, "snapshot_local_isaac62_install_metadata", lambda: dict(metadata)
    )
    monkeypatch.setattr(module.platform, "node", lambda: "spark5")
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        module, "snapshot_gb10_preload", lambda: {"preload": "stable"}
    )
    monkeypatch.setattr(
        module,
        "reverify_gb10_preload",
        lambda before: {"before": before, "after": before, "unchanged": True},
    )
    monkeypatch.setattr(
        module,
        "build_child_environment",
        lambda source, profile: {
            "PATH": source.get("PATH", ""),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "LD_PRELOAD": "/lib/aarch64-linux-gnu/libgomp.so.1",
        },
    )

    calls = []

    def fake_retriable(command, logical_root, *, timeout_seconds, environment, attempt_policy):
        calls.append(
            {
                "command": list(command),
                "timeout_seconds": timeout_seconds,
                "environment": dict(environment),
                "attempt_policy": dict(attempt_policy),
            }
        )
        attempt_root = render_process.create_attempt_root(
            logical_root, "process-attempt-01"
        )
        absolute_command = list(command)
        absolute_command[absolute_command.index("--out") + 1] = str(
            attempt_root / "capture"
        )
        launcher_fixture._write_appearance_capture(
            absolute_command, fixture["appearance"]
        )
        log = attempt_root / "render.log"
        log.write_text(READY + "\n", encoding="utf-8")
        process = {
            "command": list(command),
            "started_utc": "2026-10-24T00:00:00+00:00",
            "ended_utc": "2026-10-24T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "returncode": 0,
            "timed_out": False,
            "watchdog_trigger": None,
            "interrupted": None,
            "log_output_observed": True,
            "startup_readiness_marker": READY,
            "startup_readiness_observed": True,
            "spawn_error": None,
            "termination_escalation": "none",
            "process_group_cleanup_required": False,
            "process_group_quiescent": True,
            "log_path": str(log),
            "cwd": str(attempt_root),
            "log_sha256": "a" * 64,
            "log_hash_error": None,
        }
        row = {
            "attempt_number": 1,
            "capture_root": str(attempt_root / "capture"),
            "process": process,
        }
        return {
            "status": "process_succeeded",
            "max_process_attempts": 2,
            "attempt_count": 1,
            "selected_attempt_number": 1,
            "attempts": [row],
        }

    monkeypatch.setattr(module, "run_retriable_render_command", fake_retriable)
    argv = launcher_fixture._appearance_argv(fixture)
    argv[argv.index("--isaac-python") + 1] = render_process.LOCAL_ISAAC62_PYTHON
    argv.extend(["--environment-profile", render_process.LOCAL_ISAAC62_PROFILE])

    assert module.main(argv) == 0
    assert calls[0]["command"][calls[0]["command"].index("--out") + 1] == "capture"
    assert calls[0]["attempt_policy"] == render_process.LOCAL_ISAAC62_ATTEMPT_POLICY
    attempt = json.loads(
        (fixture["out_parent"] / fixture["run_id"] / "attempt.json").read_text()
    )
    assert attempt["environment_policy"]["profile"] == render_process.LOCAL_ISAAC62_PROFILE
    assert attempt["local_isaac62_binding"]["render_runtime"][
        "isaac_sim_version"
    ] == "6.2.0.0"
    assert attempt["process_attempts"]["max_process_attempts"] == 2

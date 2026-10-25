"""Process-boundary tests for the replay-safe remote F2 launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_physx_f2_remote.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _fixture(tmp_path: Path, *, memory_body: str, isaac_body: str):
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "physx_three_pour_sequence.py"
    runner.write_text("# fake runner identity\n", encoding="utf-8")
    experiment = tmp_path / "experiment.json"
    experiment.write_text('{"experiment":"physx_three_pour_f2"}\n', encoding="utf-8")
    fake_isaac = tmp_path / "fake-isaac"
    _write_executable(fake_isaac, isaac_body)
    fake_smi = tmp_path / "fake-nvidia-smi"
    _write_executable(fake_smi, memory_body)
    run = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "FLUIDMIX_F2_TEST_MODE": "1",
            "FLUIDMIX_F2_NVIDIA_SMI": str(fake_smi),
            "FLUIDMIX_F2_MONITOR_INTERVAL_SECONDS": "0.02",
            "FLUIDMIX_F2_FAKE_COUNT": str(tmp_path / "fake-count.txt"),
        }
    )
    return source, run, experiment, fake_isaac, env


def _launch(source: Path, run: Path, experiment: Path, isaac: Path, env: dict[str, str]):
    return subprocess.run(
        [
            str(LAUNCHER),
            "launch",
            str(source),
            str(run),
            str(experiment),
            str(isaac),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


def _wait_for_completion(run: Path, timeout: float = 8.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    completion = run / "completion.json"
    while time.monotonic() < deadline:
        if completion.is_file():
            return json.loads(completion.read_text(encoding="utf-8"))
        time.sleep(0.02)
    details = ""
    log = run / "supervisor.log"
    if log.is_file():
        details = log.read_text(encoding="utf-8", errors="replace")
    raise AssertionError(f"launcher did not complete; supervisor log:\n{details}")


def _pid(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def test_replayed_launch_runs_one_separate_process_tree_and_completes_atomically(tmp_path):
    source, run, experiment, isaac, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body=(
            'printf "run\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )

    first = _launch(source, run, experiment, isaac, env)
    replay = _launch(source, run, experiment, isaac, env)
    completion = _wait_for_completion(run)

    assert first.returncode == 0, first.stderr
    assert replay.returncode == 0, replay.stderr
    assert "already launched" in replay.stdout.lower()
    assert (run / "launch.guard").is_dir()
    assert Path(env["FLUIDMIX_F2_FAKE_COUNT"]).read_text(encoding="utf-8").splitlines() == ["run"]
    assert completion["status"] == "completed"
    assert completion["stop_reason"] == "completed"
    assert completion["exit_code"] == 0
    assert completion["wall_limit_seconds"] == 5400
    assert completion["gpu_memory_limit_mib"] == 40000
    assert completion["experiment_sha256"] == __import__("hashlib").sha256(
        experiment.read_bytes()
    ).hexdigest()
    for name in (
        "started-at.txt",
        "gpu-memory.csv",
        "exit-code.txt",
        "stop-reason.txt",
        "finished-at.txt",
        "completion.json",
        "supervisor.pid",
        "supervisor.pgid",
        "isaac.pid",
        "isaac.pgid",
        "monitor.pid",
        "monitor.pgid",
        "isaac.log",
        "supervisor.log",
    ):
        assert (run / name).exists(), name
    pgids = {
        _pid(run / "supervisor.pgid"),
        _pid(run / "isaac.pgid"),
        _pid(run / "monitor.pgid"),
    }
    assert len(pgids) == 3
    assert (run / "gpu-memory.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "timestamp_utc,used_mib"
    )


@pytest.mark.parametrize(
    ("memory_body", "expected_reason"),
    [
        ('printf "41000\\n"\n', "gpu_memory_limit_exceeded"),
        ("exit 7\n", "gpu_memory_unreadable"),
    ],
)
def test_memory_monitor_fails_closed_and_kills_only_f2_group(
    tmp_path, memory_body, expected_reason
):
    source, run, experiment, isaac, env = _fixture(
        tmp_path,
        memory_body=memory_body,
        isaac_body=(
            'printf "run\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n'
            "trap 'exit 42' TERM\n"
            "while :; do sleep 0.05; done\n"
        ),
    )
    unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        launched = _launch(source, run, experiment, isaac, env)
        completion = _wait_for_completion(run)

        assert launched.returncode == 0, launched.stderr
        assert completion["status"] == "failed"
        assert completion["stop_reason"] == expected_reason
        assert completion["exit_code"] != 0
        assert unrelated.poll() is None
        assert (run / "stop-reason.txt").read_text(encoding="utf-8").strip() == expected_reason
    finally:
        if unrelated.poll() is None:
            os.killpg(unrelated.pid, signal.SIGTERM)
            unrelated.wait(timeout=5)

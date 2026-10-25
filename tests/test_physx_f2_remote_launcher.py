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
    twin_stage = tmp_path / "fill_station.usda"
    twin_stage.write_text("#usda 1.0\n", encoding="utf-8")
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
    return source, run, experiment, fake_isaac, twin_stage, env


def _launch(
    source: Path,
    run: Path,
    experiment: Path,
    isaac: Path,
    twin_stage: Path,
    env: dict[str, str],
):
    return subprocess.run(
        [
            str(LAUNCHER),
            "launch",
            str(source),
            str(run),
            str(experiment),
            str(isaac),
            str(twin_stage),
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
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body=(
            'test "$#" -eq 7\n'
            'printf "run\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )

    first = _launch(source, run, experiment, isaac, twin_stage, env)
    replay = _launch(source, run, experiment, isaac, twin_stage, env)
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
    assert completion["qualification_mode"] is False
    assert "water_pbd_preset" not in completion
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
    assert (run / "qualification-mode.txt").read_text(encoding="utf-8").strip() == "0"
    assert not (run / "water-pbd-preset.txt").exists()


def test_qualification_mode_is_frozen_and_forwarded_without_changing_default(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body=(
            'printf "%s\\n" "$@" > "$FLUIDMIX_F2_FAKE_ARGV"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )
    argv_record = tmp_path / "isaac-argv.txt"
    env.update(
        {
            "FLUIDMIX_F2_FAKE_ARGV": str(argv_record),
            "FLUIDMIX_F2_QUALIFICATION_MODE": "1",
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "completed"
    assert completion["qualification_mode"] is True
    assert (run / "qualification-mode.txt").read_text(encoding="utf-8").strip() == "1"
    assert argv_record.read_text(encoding="utf-8").splitlines()[-1] == (
        "--qualification-mode"
    )


def test_water_pbd_preset_is_frozen_and_forwarded_with_qualification(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body=(
            'printf "%s\\n" "$@" > "$FLUIDMIX_F2_FAKE_ARGV"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )
    argv_record = tmp_path / "isaac-argv.txt"
    env.update(
        {
            "FLUIDMIX_F2_FAKE_ARGV": str(argv_record),
            "FLUIDMIX_F2_QUALIFICATION_MODE": "1",
            "FLUIDMIX_F2_WATER_PBD_PRESET": "1",
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "completed"
    assert completion["qualification_mode"] is True
    assert completion["water_pbd_preset"] is True
    assert (run / "water-pbd-preset.txt").read_text(encoding="utf-8").strip() == "1"
    assert argv_record.read_text(encoding="utf-8").splitlines()[-2:] == [
        "--qualification-mode",
        "--qualification-water-pbd-preset",
    ]


def test_invalid_qualification_mode_is_rejected_before_launch(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body='printf "unexpected\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n',
    )
    env["FLUIDMIX_F2_QUALIFICATION_MODE"] = "yes"

    launched = _launch(source, run, experiment, isaac, twin_stage, env)

    assert launched.returncode == 64
    assert "qualification mode must be 0 or 1" in launched.stderr.lower()
    assert not run.exists()
    assert not Path(env["FLUIDMIX_F2_FAKE_COUNT"]).exists()


def test_water_pbd_preset_is_forwarded_for_full_run(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body=(
            'printf "%s\\n" "$@" > "$FLUIDMIX_F2_FAKE_ARGV"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )
    argv_record = tmp_path / "isaac-argv.txt"
    env.update(
        {
            "FLUIDMIX_F2_FAKE_ARGV": str(argv_record),
            "FLUIDMIX_F2_WATER_PBD_PRESET": "1",
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "completed"
    assert completion["qualification_mode"] is False
    assert completion["water_pbd_preset"] is True
    assert argv_record.read_text(encoding="utf-8").splitlines()[-1] == (
        "--water-pbd-preset"
    )


def test_full_run_wall_limit_is_frozen_forwarded_and_recorded(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body='printf "1234\\n"\n',
        isaac_body=(
            'printf "%s\\n" "$@" > "$FLUIDMIX_F2_FAKE_ARGV"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )
    argv_record = tmp_path / "isaac-argv.txt"
    env.update(
        {
            "FLUIDMIX_F2_FAKE_ARGV": str(argv_record),
            "FLUIDMIX_F2_WALL_LIMIT_SECONDS": "10800",
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "completed"
    assert completion["wall_limit_seconds"] == 10800
    assert (run / "wall-limit-seconds.txt").read_text(encoding="utf-8").strip() == (
        "10800"
    )
    assert argv_record.read_text(encoding="utf-8").splitlines()[-2:] == [
        "--wall-limit-seconds",
        "10800",
    ]


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
    source, run, experiment, isaac, twin_stage, env = _fixture(
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
        launched = _launch(source, run, experiment, isaac, twin_stage, env)
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


def test_explicit_local_uma_monitor_samples_system_memory_and_records_source(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body="exit 99\n",
        isaac_body=(
            'printf "run\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n'
            'mkdir -p "$3"\n'
            'printf "{}\\n" > "$3/receipt.json"\n'
            "sleep 0.15\n"
        ),
    )
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 131073 kB\nMemAvailable: 98304 kB\n",
        encoding="ascii",
    )
    env.update(
        {
            "FLUIDMIX_F2_MEMORY_MODE": "local_uma",
            "FLUIDMIX_F2_MEMINFO": str(meminfo),
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "completed"
    rows = (run / "gpu-memory.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "timestamp_utc,used_mib"
    assert {row.rsplit(",", 1)[1] for row in rows[1:]} == {"33"}
    assert (run / "memory-monitor-mode.txt").read_text(encoding="utf-8").strip() == (
        "local_uma"
    )
    assert (run / "memory-monitor-source.txt").read_text(encoding="utf-8").strip() == str(
        meminfo
    )


@pytest.mark.parametrize(
    "meminfo_text",
    [
        None,
        "MemTotal: 131072 kB\n",
        "MemTotal: many kB\nMemAvailable: 98304 kB\n",
        "MemTotal: 65536 kB\nMemAvailable: 98304 kB\n",
        "MemTotal: 131072 kB\nMemTotal: 131072 kB\nMemAvailable: 98304 kB\n",
    ],
)
def test_local_uma_monitor_rejects_unavailable_or_malformed_meminfo(
    tmp_path, meminfo_text
):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body="exit 99\n",
        isaac_body=(
            'printf "run\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n'
            "trap 'exit 42' TERM\n"
            "while :; do sleep 0.05; done\n"
        ),
    )
    meminfo = tmp_path / "meminfo"
    if meminfo_text is not None:
        meminfo.write_text(meminfo_text, encoding="ascii")
    env.update(
        {
            "FLUIDMIX_F2_MEMORY_MODE": "local_uma",
            "FLUIDMIX_F2_MEMINFO": str(meminfo),
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "failed"
    assert completion["stop_reason"] == "gpu_memory_unreadable"
    assert (run / "gpu-memory.csv").read_text(encoding="utf-8").splitlines() == [
        "timestamp_utc,used_mib"
    ]


def test_local_uma_monitor_enforces_same_40000_mib_limit(tmp_path):
    source, run, experiment, isaac, twin_stage, env = _fixture(
        tmp_path,
        memory_body="exit 99\n",
        isaac_body=(
            'printf "run\\n" >> "$FLUIDMIX_F2_FAKE_COUNT"\n'
            "trap 'exit 42' TERM\n"
            "while :; do sleep 0.05; done\n"
        ),
    )
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 131072000 kB\nMemAvailable: 90112000 kB\n",
        encoding="ascii",
    )
    env.update(
        {
            "FLUIDMIX_F2_MEMORY_MODE": "local_uma",
            "FLUIDMIX_F2_MEMINFO": str(meminfo),
        }
    )

    launched = _launch(source, run, experiment, isaac, twin_stage, env)
    completion = _wait_for_completion(run)

    assert launched.returncode == 0, launched.stderr
    assert completion["status"] == "failed"
    assert completion["stop_reason"] == "gpu_memory_limit_exceeded"
    assert (run / "gpu-memory.csv").read_text(encoding="utf-8").splitlines()[1].endswith(
        ",40000"
    )

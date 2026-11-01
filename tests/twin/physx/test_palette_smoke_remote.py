"""Replay-boundary test for the palette smoke remote launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "cli/twin/physx" / "run_physx_palette_smoke_remote.sh"


def executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_replay_runs_one_guarded_process_and_records_completion(tmp_path):
    source = tmp_path / "source"
    (source / "cli/twin/probes").mkdir(parents=True)
    (source / "cli/twin/probes" / "physx_particle_palette_smoke.py").write_text("# identity\n", encoding="utf-8")
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    experiment = tmp_path / "experiment.json"
    experiment.write_text('{"experiment":"physx_particle_palette_smoke"}\n', encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    executable(
        fake_python,
        'printf "run\\n" >> "$PALETTE_FAKE_COUNT"\n'
        'mkdir -p "$5"\n'
        'printf "{}\\n" > "$5/receipt.json"\n'
        'sleep 0.15\n',
    )
    fake_smi = tmp_path / "fake-smi"
    executable(fake_smi, 'printf "1234\\n"\n')
    run = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "PALETTE_SMOKE_TEST_MODE": "1",
            "PALETTE_SMOKE_NVIDIA_SMI": str(fake_smi),
            "PALETTE_SMOKE_MONITOR_INTERVAL": "0.02",
            "PALETTE_FAKE_COUNT": str(tmp_path / "count.txt"),
        }
    )
    command = [str(LAUNCHER), "launch", str(source), str(run), str(experiment), str(fake_python), str(stage)]

    first = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=10)
    replay = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=10)
    deadline = time.monotonic() + 8
    while not (run / "completion.json").is_file() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert first.returncode == 0, first.stderr
    assert replay.returncode == 0, replay.stderr
    assert "already launched" in replay.stdout.lower()
    assert (tmp_path / "count.txt").read_text(encoding="utf-8").splitlines() == ["run"]
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "completed"
    assert completion["exit_code"] == 0
    assert completion["wall_limit_seconds"] == 1800
    assert completion["gpu_memory_limit_mib"] == 40000
    assert (run / "gpu-memory.csv").read_text(encoding="utf-8").splitlines()[0] == "timestamp_utc,used_mib"

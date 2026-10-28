"""Checks for the dispatch-only multi-process supervisor wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType


def _module(
    status: str = "pass",
    *,
    conflicts: list[dict[str, object]] | None = None,
) -> ModuleType:
    module = ModuleType("stub_supervisor")

    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--case", required=True)
        return parser

    def run_supervisor(args, *, active_processes):
        assert args.case == "22"
        assert active_processes() == []
        return {"status": status, "case": args.case}

    def find_other_isaac_processes():
        return list(conflicts or [])

    module.build_parser = build_parser
    module.run_supervisor = run_supervisor
    module._find_other_isaac_processes = find_other_isaac_processes
    return module


def _wrapper_args() -> list[str]:
    return [
        "--supervisor-script",
        str(Path(__file__)),
        "--peer-renderer-script",
        "/source/scripts/physx_three_pour_sequence.py",
        "--peer-plan",
        "/payload/render-schedule-plan.json",
        "--peer-output-prefix",
        "/runs/varied-v4-",
        "--",
        "--case",
        "22",
    ]


def test_wrapper_disables_only_the_host_wide_process_probe(capsys):
    from tools.sim_runner.concurrent_supervisor import run

    result = run(
        _wrapper_args(),
        module=_module(),
    )

    captured = capsys.readouterr()
    assert result == 0
    assert '"status": "pass"' in captured.out
    assert "dispatch-only deviation" in captured.err
    assert "physics inputs" in captured.err


def test_wrapper_preserves_failed_supervisor_exit_status():
    from tools.sim_runner.concurrent_supervisor import run

    result = run(
        _wrapper_args(),
        module=_module("failed"),
    )

    assert result == 1


def test_wrapper_allows_only_exact_same_cohort_peer_processes():
    from tools.sim_runner.concurrent_supervisor import run

    peer = {
        "pid": 42,
        "command": (
            "/source/scripts/physx_three_pour_sequence.py "
            "--train-render-plan /payload/render-schedule-plan.json "
            "--out /runs/varied-v4-case-23"
        ),
    }

    assert run(_wrapper_args(), module=_module(conflicts=[peer])) == 0


def test_wrapper_allows_exact_same_cohort_wrapper_process():
    from tools.sim_runner import concurrent_supervisor
    from tools.sim_runner.concurrent_supervisor import run

    peer = {
        "pid": 43,
        "command": (
            f"python {Path(concurrent_supervisor.__file__).resolve()} "
            f"--supervisor-script {Path(__file__).resolve()} "
            "--peer-renderer-script /source/scripts/physx_three_pour_sequence.py "
            "--peer-plan /payload/render-schedule-plan.json "
            "--peer-output-prefix /runs/varied-v4- "
            "--output-root /runs/varied-v4-case-23"
        ),
    }

    assert run(_wrapper_args(), module=_module(conflicts=[peer])) == 0


def test_wrapper_keeps_unrelated_isaac_process_as_a_conflict():
    from tools.sim_runner.concurrent_supervisor import run

    unrelated = {"pid": 99, "command": "python omni.kit unrelated.py"}
    module = _module(conflicts=[unrelated])

    def run_supervisor(args, *, active_processes):
        assert active_processes() == [unrelated]
        return {"status": "failed"}

    module.run_supervisor = run_supervisor
    assert run(_wrapper_args(), module=module) == 1

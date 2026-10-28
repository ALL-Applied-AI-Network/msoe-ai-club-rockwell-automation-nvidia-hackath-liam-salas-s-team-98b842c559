#!/usr/bin/env python3
"""Dispatch-only wrapper that permits isolated supervisors to share one GPU.

The imported supervisor and renderer files stay byte-identical. This wrapper
only replaces the supervisor's host-wide process probe; each dispatcher slot
still has its own process, lock, output root, logs, and one-case shard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType


def _load_supervisor(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("supervisor script must be a regular file")
    spec = importlib.util.spec_from_file_location(
        "fluidmix_concurrent_source_supervisor", resolved
    )
    if spec is None or spec.loader is None:
        raise ValueError("could not load supervisor script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dispatch_process_probe(
    module: ModuleType,
    *,
    peer_supervisor_script: str,
    peer_renderer_script: str,
    peer_plan: str,
    peer_output_prefix: str,
):
    """Keep unrelated Isaac/Kit conflicts while allowing exact cohort peers."""

    renderer_tokens = (
        peer_renderer_script,
        f"--train-render-plan {peer_plan}",
        f"--out {peer_output_prefix}",
    )
    wrapper_tokens = (
        os.fspath(Path(__file__).resolve()),
        f"--supervisor-script {peer_supervisor_script}",
        f"--peer-renderer-script {peer_renderer_script}",
        f"--peer-plan {peer_plan}",
        f"--peer-output-prefix {peer_output_prefix}",
        f"--output-root {peer_output_prefix}",
    )

    def exact_peer(command: object) -> bool:
        return isinstance(command, str) and (
            all(token in command for token in renderer_tokens)
            or all(token in command for token in wrapper_tokens)
        )

    def parent_pid(pid: int) -> int | None:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("PPid:"):
                    return int(line.split(":", 1)[1].strip())
        except (OSError, ValueError):
            return None
        return None

    def descends_from(pid: object, allowed: set[int]) -> bool:
        if isinstance(pid, bool) or not isinstance(pid, int):
            return False
        seen: set[int] = set()
        current = pid
        while current > 1 and current not in seen:
            seen.add(current)
            current_parent = parent_pid(current)
            if current_parent is None:
                return False
            if current_parent in allowed:
                return True
            current = current_parent
        return False

    def probe() -> list[dict[str, object]]:
        conflicts = module._find_other_isaac_processes()
        allowed_pids = {
            row["pid"]
            for row in conflicts
            if isinstance(row, dict)
            and isinstance(row.get("pid"), int)
            and not isinstance(row.get("pid"), bool)
            and exact_peer(row.get("command"))
        }
        return [
            row
            for row in conflicts
            if not (
                isinstance(row, dict)
                and (
                    exact_peer(row.get("command"))
                    or descends_from(row.get("pid"), allowed_pids)
                )
            )
        ]

    return probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-script", required=True, type=Path)
    parser.add_argument("--peer-renderer-script", required=True)
    parser.add_argument("--peer-plan", required=True)
    parser.add_argument("--peer-output-prefix", required=True)
    parser.add_argument("supervisor_args", nargs=argparse.REMAINDER)
    return parser


def run(argv: list[str] | None = None, *, module: ModuleType | None = None) -> int:
    wrapper_args = build_parser().parse_args(argv)
    supervisor_args = list(wrapper_args.supervisor_args)
    if supervisor_args[:1] == ["--"]:
        supervisor_args.pop(0)
    if not supervisor_args:
        raise ValueError("supervisor arguments are required after --")
    loaded = module or _load_supervisor(wrapper_args.supervisor_script)
    args = loaded.build_parser().parse_args(supervisor_args)
    print(
        "dispatch-only deviation: host-wide Isaac process guard relaxed only for "
        "exact same-cohort renderer peers; unrelated Isaac/Kit processes still block; "
        "supervisor, renderer, physics inputs, per-slot lock, and output roots unchanged",
        file=sys.stderr,
    )
    try:
        summary = loaded.run_supervisor(
            args,
            active_processes=_dispatch_process_probe(
                loaded,
                peer_supervisor_script=os.fspath(
                    wrapper_args.supervisor_script.expanduser().resolve(strict=True)
                ),
                peer_renderer_script=wrapper_args.peer_renderer_script,
                peer_plan=wrapper_args.peer_plan,
                peer_output_prefix=wrapper_args.peer_output_prefix,
            ),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"supervisor input/infrastructure failure: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary.get("status") == "pass" else 1


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

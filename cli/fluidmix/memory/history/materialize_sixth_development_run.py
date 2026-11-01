#!/usr/bin/env python3
"""Publish the bounded sixth WP5 development run without oracle inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.fspath(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner-thread-id", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args(argv)

    from fluidmix.memory.history.sixth_development_run import build_six_run_development_store

    receipt = build_six_run_development_store(
        output_dir=args.output_dir,
        owner_thread_id=args.owner_thread_id,
        code_commit=args.code_commit,
    )
    print(json.dumps({"status": receipt["status"], "denominators": receipt["denominators"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

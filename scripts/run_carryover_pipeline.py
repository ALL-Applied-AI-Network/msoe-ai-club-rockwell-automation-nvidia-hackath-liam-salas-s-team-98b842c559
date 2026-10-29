#!/usr/bin/env python3
"""Thin command line entry point for the production carryover scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.carryover_pipeline import (  # noqa: E402
    freeze_validation,
    run_real_check,
    run_test,
)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--recipe-sequence", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--owner-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze validation or seal a one-pass carryover production receipt."
    )
    phases = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = phases.add_parser("freeze-validation")
    freeze_parser.add_argument("--validation-results", required=True, type=Path)
    freeze_parser.add_argument("--out", required=True, type=Path)
    _run_arguments(phases.add_parser("run-test"))
    _run_arguments(phases.add_parser("run-real-check"))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.phase == "freeze-validation":
        receipt = freeze_validation(arguments.validation_results, arguments.out)
    else:
        function = run_test if arguments.phase == "run-test" else run_real_check
        receipt = function(
            video=arguments.video,
            recipe_sequence=arguments.recipe_sequence,
            freeze=arguments.freeze,
            expected_freeze_sha256=arguments.freeze_sha256,
            store=arguments.store,
            out=arguments.out,
            claim=arguments.claim,
            owner_id=arguments.owner_id,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

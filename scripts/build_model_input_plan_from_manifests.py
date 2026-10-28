#!/usr/bin/env python3
"""Build an allowlisted model-input plan from verified twin captures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-manifest",
        action="append",
        required=True,
        help="raw verified twin_stage_capture_manifest JSON (repeatable)",
    )
    parser.add_argument(
        "--batch-context",
        required=True,
        help="strict stage_batch_context JSON",
    )
    parser.add_argument("--out", required=True, help="new model_input_plan JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.model_input_plan_builder import write_model_input_plan

    write_model_input_plan(args.capture_manifest, args.batch_context, args.out)
    print(f"wrote model input plan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

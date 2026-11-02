#!/usr/bin/env python3
"""Run one hash-bound WP7 VLM attempt; never loads multiple models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from evaluation.rca_bench.model_bakeoff_runner import invoke_stock_runtime, load_stock_runtime
from fluidmix.explanation.vlm.decision import run_one_model_attempt_from_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one exact pinned WP7 VLM against a reviewed development freeze"
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--image-locators", required=True)
    parser.add_argument("--tracker-receipt", action="append", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--reviewed-freeze-sha256", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--production", action="store_true",
        help="require production camera metadata and bind a new tracker cohort",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_one_model_attempt_from_files(
        args.cohort,
        args.image_locators,
        args.tracker_receipt,
        freeze_path=args.freeze,
        expected_freeze_sha256=args.reviewed_freeze_sha256,
        repository=args.repository,
        revision=args.revision,
        snapshot_path=args.snapshot,
        production=args.production,
        output_path=args.output,
        runtime_loader=load_stock_runtime,
        runtime_invoker=invoke_stock_runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

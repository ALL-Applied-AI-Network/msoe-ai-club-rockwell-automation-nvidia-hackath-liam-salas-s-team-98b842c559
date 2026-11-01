#!/usr/bin/env python3
"""CPU-only aggregation of zero to three hash-bound WP7 VLM attempts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from fluidmix.decision.vlm import aggregate_model_attempts_from_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate zero to three WP7 VLM attempt receipts"
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--image-locators", required=True)
    parser.add_argument("--tracker-receipt", action="append", required=True)
    parser.add_argument("--attempt-receipt", action="append", default=[])
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--reviewed-freeze-sha256", required=True)
    parser.add_argument(
        "--production", action="store_true",
        help="require production camera metadata and bind a new tracker cohort",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate_model_attempts_from_files(
        args.cohort,
        args.image_locators,
        args.tracker_receipt,
        args.attempt_receipt,
        freeze_path=args.freeze,
        expected_freeze_sha256=args.reviewed_freeze_sha256,
        production=args.production,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

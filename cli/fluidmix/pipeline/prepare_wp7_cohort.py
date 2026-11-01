#!/usr/bin/env python3
"""Prepare WP7 cohort-goldens inputs without reading camera media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.pipeline.cohort_preparation import (  # noqa: E402
    prepare_real_check_inputs,
    prepare_run_test_inputs,
    prepare_validation_freeze,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish metadata-only WP7 cohort preparation bundles."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("prepare-run-test")
    capture.add_argument("--recipe-sequence", required=True, type=Path)
    capture.add_argument("--recipe-sequence-sha256", required=True)
    capture.add_argument("--composition-receipt", required=True, type=Path)
    capture.add_argument("--composition-receipt-sha256", required=True)
    capture.add_argument("--final-review", required=True, type=Path)
    capture.add_argument("--final-review-sha256", required=True)
    for index in (1, 2):
        capture.add_argument(f"--run-receipt-{index}", required=True, type=Path)
        capture.add_argument(f"--run-receipt-{index}-sha256", required=True)
    capture.add_argument("--destination", required=True, type=Path)

    real_check = commands.add_parser("prepare-real-check")
    real_check.add_argument("--destination", required=True, type=Path)

    freeze = commands.add_parser("prepare-freeze")
    freeze.add_argument("--checkpoint", required=True, type=Path)
    freeze.add_argument("--checkpoint-sha256", required=True)
    freeze.add_argument("--replay", required=True, type=Path)
    freeze.add_argument("--replay-sha256", required=True)
    freeze.add_argument("--review", required=True, type=Path)
    freeze.add_argument("--review-sha256", required=True)
    freeze.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-run-test":
        result = prepare_run_test_inputs(
            recipe_sequence=args.recipe_sequence,
            expected_recipe_sequence_sha256=args.recipe_sequence_sha256,
            composition_receipt=args.composition_receipt,
            expected_composition_receipt_sha256=args.composition_receipt_sha256,
            final_review=args.final_review,
            expected_final_review_sha256=args.final_review_sha256,
            run_receipts=(args.run_receipt_1, args.run_receipt_2),
            expected_run_receipt_sha256s=(
                args.run_receipt_1_sha256, args.run_receipt_2_sha256,
            ),
            destination=args.destination,
        )
        output = {
            "publication_sha256": result.publication_receipt["publication_sha256"],
            "output_paths": [str(path) for path in result.output_paths],
        }
    elif args.command == "prepare-real-check":
        result = prepare_real_check_inputs(destination=args.destination)
        output = {
            "publication_sha256": result.publication_receipt["publication_sha256"],
            "output_paths": [str(path) for path in result.output_paths],
        }
    else:
        result = prepare_validation_freeze(
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            replay=args.replay,
            expected_replay_sha256=args.replay_sha256,
            review=args.review,
            expected_review_sha256=args.review_sha256,
            destination=args.destination,
        )
        output = {
            "validation_results_path": str(result.validation_results_path),
            "freeze_path": str(result.freeze_path),
            "preparation_receipt_path": str(result.preparation_receipt_path),
        }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the preregistered single-RGB mask validation, freeze, evaluation, or publish phase."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    validation = subparsers.add_parser("validation")
    validation.add_argument("--index", required=True)
    validation.add_argument("--merged-root", required=True)
    validation.add_argument("--preregistration", required=True)
    validation.add_argument("--out", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--validation-dir", required=True)
    freeze.add_argument("--out", required=True)

    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("--index", required=True)
    evaluation.add_argument("--merged-root", required=True)
    evaluation.add_argument("--validation-dir", required=True)
    evaluation.add_argument("--freeze-receipt", required=True)
    evaluation.add_argument("--sentinel", required=True)
    evaluation.add_argument("--out", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--validation-dir", required=True)
    publish.add_argument("--evaluation-dir", required=True)
    publish.add_argument("--freeze-receipt", required=True)
    publish.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.spatial_rgb_mask_predictor import (
        publish_locator_artifact,
        run_evaluation_once,
        run_validation_experiment,
        write_freeze_receipt,
    )

    if args.phase == "validation":
        result = run_validation_experiment(
            index_path=args.index,
            merged_root=args.merged_root,
            preregistration_path=args.preregistration,
            output_dir=args.out,
        )
        print(
            f"{result['status']}: validation images={result['validation_metrics']['image_count']}; "
            f"mean Dice={result['validation_metrics']['mean_dice']:.6f}; "
            f"mean height error={result['validation_metrics']['mean_absolute_height_error_px']:.6f}px"
        )
        return 0 if result["status"] == "validation_pass" else 2
    if args.phase == "freeze":
        receipt = write_freeze_receipt(
            validation_dir=args.validation_dir, output_path=args.out
        )
        print(f"frozen_before_evaluation: model={receipt['model_sha256']}; receipt={args.out}")
        return 0
    if args.phase == "evaluation":
        result = run_evaluation_once(
            index_path=args.index,
            merged_root=args.merged_root,
            validation_dir=args.validation_dir,
            freeze_receipt_path=args.freeze_receipt,
            sentinel_path=args.sentinel,
            output_dir=args.out,
        )
        print(
            f"{result['status']}: evaluation images={result['evaluation_metrics']['image_count']}; "
            f"mean Dice={result['evaluation_metrics']['mean_dice']:.6f}; "
            f"mean height error={result['evaluation_metrics']['mean_absolute_height_error_px']:.6f}px"
        )
        return 0 if result["status"] == "evaluation_pass" else 2
    artifact = publish_locator_artifact(
        validation_dir=args.validation_dir,
        evaluation_dir=args.evaluation_dir,
        freeze_receipt_path=args.freeze_receipt,
        output_path=args.out,
    )
    print(f"published schema-v2 locators: images={len(artifact['images'])}; path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

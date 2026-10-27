#!/usr/bin/env python3
"""Run the frozen validation-only temporal RGB mask baseline experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--merged-root", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.temporal_rgb_mask_baseline import run_validation_experiment

    result = run_validation_experiment(
        index_path=args.index,
        merged_root=args.merged_root,
        preregistration_path=args.preregistration,
        output_dir=args.out,
        expected_denominators={
            "train_vials": 12,
            "train_stage_images": 30,
            "validation_vials": 3,
            "validation_stage_images": 6,
            "evaluation_vials_declared_not_opened": 3,
            "evaluation_stage_images_declared_not_opened": 6,
        },
    )
    print(
        f"{result['status']}: {result['candidate_count']} candidates; "
        f"validation mean Dice={result['validation_metrics']['mean_dice']:.6f}; "
        f"evaluation pixels opened={result['evaluation_pixels_opened']}; "
        f"result={Path(args.out) / 'result.json'}"
    )
    return 0 if result["status"] == "validation_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

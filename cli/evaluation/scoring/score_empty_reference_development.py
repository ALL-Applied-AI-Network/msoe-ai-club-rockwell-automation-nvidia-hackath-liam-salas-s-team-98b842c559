#!/usr/bin/env python3
"""Score one hash-frozen Approach-A development prediction artifact."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--image-locators", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    from evaluation.scoring.empty_reference_development_scoring import (
        score_frozen_development_predictions,
    )

    result = score_frozen_development_predictions(
        prediction_path=args.predictions,
        expected_prediction_sha256=args.predictions_sha256,
        image_locators_path=args.image_locators,
        output_path=args.out,
    )
    metrics = result["metrics"]
    print(
        f"{result['status']}: images=6; mean_dice={metrics['mean_dice']:.6f}; "
        f"mean_surface_error_px={metrics['mean_surface_row_absolute_error_px']:.6f}; "
        f"output={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

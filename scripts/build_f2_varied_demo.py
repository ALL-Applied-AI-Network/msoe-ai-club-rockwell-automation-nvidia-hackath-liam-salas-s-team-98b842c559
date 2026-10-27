#!/usr/bin/env python3
"""Build the immutable Varied V3 synthetic diagnostic review bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--image-locators", type=Path, required=True)
    parser.add_argument("--cosmos-score", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.f2_varied_demo import build_varied_demo_bundle

    receipt = build_varied_demo_bundle(
        args.decisions,
        args.image_locators,
        args.cosmos_score,
        args.out,
    )
    receipt_path = args.out / "receipt.json"
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    print(
        f"published diagnostic-only review for "
        f"{receipt['denominators']['decisions']} abstaining vial decisions and "
        f"{receipt['denominators']['images']} oracle-mask images: {args.out}; "
        f"receipt_sha256={receipt_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

#!/usr/bin/env python3
"""Publish safe active-fill inputs from the accepted Varied V3 receipts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-validation-receipt", required=True)
    parser.add_argument("--image-locators", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from twin.cohorts.fill_clip_inputs import build_f2_fill_clip_input_manifest

    path = build_f2_fill_clip_input_manifest(
        downstream_validation_receipt=args.downstream_validation_receipt,
        image_locators=args.image_locators,
        output_path=args.output,
    )
    print(f"published 18 safe V3 fill-clip inputs: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

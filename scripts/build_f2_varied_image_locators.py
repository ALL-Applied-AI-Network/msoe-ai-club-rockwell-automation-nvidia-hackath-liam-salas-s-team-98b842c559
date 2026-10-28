#!/usr/bin/env python3
"""Publish the label-free image locator for six corrected Varied V3 runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-manifest",
        action="append",
        required=True,
        help="corrected schema-3 capture manifest (repeat exactly six times)",
    )
    parser.add_argument(
        "--validation-receipt",
        required=True,
        help="corrected V3 downstream validation receipt",
    )
    parser.add_argument("--out", required=True, help="new locator JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.f2_varied_image_locator import write_varied_image_locators

    result = write_varied_image_locators(
        args.capture_manifest,
        args.validation_receipt,
        args.out,
    )
    denominators = result["denominators"]
    print(
        f"published {denominators['images']} image locators from "
        f"{denominators['source_manifests']} manifests and "
        f"{denominators['runs']} runs: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create one neutral, uncalibrated direct-appearance USD override layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def finite_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument(
        "--linear-rgb", required=True, nargs=3, type=finite_number, metavar=("R", "G", "B")
    )
    parser.add_argument("--opacity", required=True, type=finite_number)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from fluidmix.twin_appearance import create_appearance_layer

    try:
        record = create_appearance_layer(
            args.twin,
            args.output_root,
            args.variant_id,
            args.linear_rgb,
            args.opacity,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

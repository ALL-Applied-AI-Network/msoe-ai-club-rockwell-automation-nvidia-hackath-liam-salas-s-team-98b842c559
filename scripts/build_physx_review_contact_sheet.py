#!/usr/bin/env python3
"""Build a receipt-bound RGB/liquid-mask contact sheet for PhysX review."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _physics_steps(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError("steps must be comma-separated integers")
    try:
        steps = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if any(step < 0 for step in steps) or len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must be unique nonnegative integers")
    return steps


def _crop(value: str) -> tuple[int, int, int, int]:
    pieces = value.split(",")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2")
    try:
        values = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2") from error
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="accepted PhysX receipt.json")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new directory for contact-sheet.png and its provenance JSON",
    )
    parser.add_argument(
        "--steps",
        required=True,
        type=_physics_steps,
        help="comma-separated physics steps in the desired review order",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="positive grid column count (default: 3)",
    )
    parser.add_argument(
        "--crop",
        type=_crop,
        help="optional exclusive x1,y1,x2,y2 crop applied equally to every frame",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.review_contact_sheet import create_physx_review_contact_sheet

    record = create_physx_review_contact_sheet(
        args.receipt,
        args.output_dir,
        physics_steps=args.steps,
        columns=args.columns,
        crop_xyxy=args.crop,
    )
    print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

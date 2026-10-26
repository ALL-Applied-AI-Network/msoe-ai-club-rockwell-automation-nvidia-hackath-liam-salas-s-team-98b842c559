#!/usr/bin/env python3
"""Encode receipt-bound PhysX PNG frames as a decoded, verified review MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="PhysX render receipt.json")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory to publish sequence.mp4 and video.json into",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Positive integer review playback rate (default: 6)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.review_video import encode_physx_receipt_video

    record = encode_physx_receipt_video(
        args.receipt,
        args.output_dir,
        fps=args.fps,
    )
    print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

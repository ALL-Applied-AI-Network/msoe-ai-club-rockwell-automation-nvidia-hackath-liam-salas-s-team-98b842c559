#!/usr/bin/env python3
"""Encode one verified 133-frame FillCam capture as a review-only MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from twin.render.twin_video import encode_capture_video

    record = encode_capture_video(args.capture_root, args.output_dir)
    print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

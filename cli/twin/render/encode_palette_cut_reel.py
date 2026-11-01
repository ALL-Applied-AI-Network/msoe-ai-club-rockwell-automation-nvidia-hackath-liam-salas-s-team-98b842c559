#!/usr/bin/env python3
"""Encode accepted run MP4s with explicit no-transport palette cut cards."""

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
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        help="JSON request containing run_order and ordered path/hash/run-id rows",
    )
    parser.add_argument(
        "--owner-thread-id",
        required=True,
        help="Owner identifier recorded in the reel receipt",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for the MP4, JSON receipt, and SHA256SUMS",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    try:
        receipt = encode_palette_cut_reel(
            args.request,
            args.output_dir,
            owner_thread_id=args.owner_thread_id,
            repo_root=ROOT,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"palette-cut reel failed: {error}", file=sys.stderr)
        return 2
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish three accepted F2 vial receipts as one downstream stage batch."""

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
    parser.add_argument(
        "--request",
        required=True,
        help="strict f2_stage_batch_publish_request JSON",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new no-clobber directory for the manifest, context and receipts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.f2_receipt_manifest_publisher import publish_f2_stage_batch

    receipt = publish_f2_stage_batch(args.request, args.output_dir)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

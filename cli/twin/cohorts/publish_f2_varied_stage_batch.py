#!/usr/bin/env python3
"""Publish one accepted Varied V3 three-vial batch into stage manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from twin.cohorts.receipt_manifest_publisher import (
        publish_f2_varied_stage_batch,
    )

    receipt = publish_f2_varied_stage_batch(args.request, args.output_dir)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

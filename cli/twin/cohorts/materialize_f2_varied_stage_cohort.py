#!/usr/bin/env python3
"""Materialize Varied V3 stage-publication requests from a merged cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-plan", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from twin.cohorts.varied_stage_materializer import (
        materialize_varied_publish_requests,
    )

    receipt = materialize_varied_publish_requests(
        model_input_plan=args.model_input_plan,
        cohort_root=args.cohort_root,
        runtime_id=args.runtime_id,
        output_dir=args.output_dir,
    )
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

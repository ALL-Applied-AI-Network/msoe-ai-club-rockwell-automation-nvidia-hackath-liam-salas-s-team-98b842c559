#!/usr/bin/env python3
"""Measure the frozen synthetic VA stage evidence from SHA-bound capture PNGs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-plan", required=True)
    parser.add_argument("--image-locators", required=True)
    parser.add_argument("--binding-receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.validation_stage_measurement import (
        write_validation_stage_evidence,
    )

    record = write_validation_stage_evidence(
        args.model_plan,
        args.image_locators,
        args.binding_receipt,
        args.output,
    )
    denominators = record["denominators"]
    print(
        "measured "
        f"{denominators['image_measurements']} synthetic stage images for "
        f"{denominators['vial_decisions']} vial decisions: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

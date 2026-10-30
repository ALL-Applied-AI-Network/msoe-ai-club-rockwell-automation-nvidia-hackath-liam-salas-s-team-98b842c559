#!/usr/bin/env python3
"""Build label-free per-stage colour evidence from Varied V3 publications."""

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
        help="published schema-3 f2-stage-capture-manifest.json (repeatable)",
    )
    parser.add_argument(
        "--predicted-image-locators",
        help=(
            "optional strict schema-v2 RGB-only predicted-mask locator artifact; "
            "when omitted, the synthetic renderer masks remain the diagnostic input"
        ),
    )
    parser.add_argument(
        "--bridge-compatible",
        action="store_true",
        help=(
            "publish a measured artifact with the exact WP5 bridge header; "
            "the prediction-locator hash is retained in measurement_contract"
        ),
    )
    parser.add_argument("--out", required=True, help="new measured-evidence JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.f2_varied_stage_measurement import (
        write_varied_stage_measurements,
    )

    result = write_varied_stage_measurements(
        args.capture_manifest,
        args.out,
        predicted_image_locators_path=args.predicted_image_locators,
        bridge_compatible=args.bridge_compatible,
    )
    denominators = result["denominators"]
    print(
        f"measured {denominators['stage_measurements']} stages from "
        f"{denominators['vials']} vials in {denominators['runs']} runs; "
        f"published {denominators['same_run_stage_pairs']} same-stage pairs: "
        f"{args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

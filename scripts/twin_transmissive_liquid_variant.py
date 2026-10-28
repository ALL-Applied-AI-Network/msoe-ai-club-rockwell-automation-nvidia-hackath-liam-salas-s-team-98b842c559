#!/usr/bin/env python3
"""Author an uncalibrated OmniSurface transmissive-liquid USD override.

This tool only authors and CPU-inspects a small USD layer.  It does not render,
fit absorbance, calculate a mixture, or establish physical validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fluidmix.transmissive_optics import create_transmissive_layer


def _color(value: str) -> list[float]:
    try:
        result = [float(piece) for piece in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("transmission color must be comma-separated numbers") from error
    if len(result) != 3:
        raise argparse.ArgumentTypeError("transmission color must have exactly three channels")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("scattering depth must be a number") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("scattering depth must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", required=True, help="csi_digital_twin_v1 folder")
    parser.add_argument("--out-root", required=True, help="existing output parent directory")
    parser.add_argument("--variant-id", required=True)
    parser.add_argument(
        "--transmission-color", type=_color, required=True,
        help="uncalibrated linear RGB passed to OmniSurface transmission",
    )
    parser.add_argument(
        "--scattering-depth-m", type=_positive_float, required=True,
        help="positive OmniSurface transmission scattering depth in scene meters",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = create_transmissive_layer(
        args.twin,
        args.out_root,
        args.variant_id,
        args.transmission_color,
        args.scattering_depth_m,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

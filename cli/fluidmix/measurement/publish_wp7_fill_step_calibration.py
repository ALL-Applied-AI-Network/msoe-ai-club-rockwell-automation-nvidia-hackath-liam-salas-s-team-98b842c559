#!/usr/bin/env python3
"""Publish the already-frozen attempt04 calibration as a standalone artifact.

This performs no search and opens no RGB, masks, labels, recipes, or video.  It
replays the accepted dense surface-domain rule and refuses any payload whose
self-hash differs from the attempt04 materialization receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.measurement.fill_step_calibration import build_development_calibration
from twin.physx.volume_gate import visible_height_from_surface_row_m
from twin.physx.volume_height_curve import (
    DEFAULT_GEOMETRY,
    volume_for_height_m,
    volume_height_curve_digest,
)


ACCEPTED_CALIBRATION_SHA256 = (
    "9d2c2941244ef807cf02cc4bc3107dc6f621aa68bc59b11627484fb0d2c46c04"
)
CALIBRATION_ID = "wp7-fillcam-dense-source-geometry-v1"


def publish(path: Path) -> None:
    samples = tuple(
        (float(417 - surface_row), float(surface_row))
        for surface_row in range(383, 66, -1)
    )
    calibration = build_development_calibration(
        calibration_id=CALIBRATION_ID,
        fill_height_surface_row_samples=samples,
        geometry=DEFAULT_GEOMETRY,
        visible_height_from_surface_row_m=visible_height_from_surface_row_m,
        volume_for_height_m=volume_for_height_m,
        volume_height_curve_digest=volume_height_curve_digest,
    )
    if calibration.artifact_sha256 != ACCEPTED_CALIBRATION_SHA256:
        raise RuntimeError("accepted attempt04 calibration replay drifted")
    payload = json.dumps(
        asdict(calibration), sort_keys=True, indent=2, allow_nan=False
    ).encode() + b"\n"
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the hash-locked WP7 attempt04 pixel calibration"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish(args.output)


if __name__ == "__main__":
    main()

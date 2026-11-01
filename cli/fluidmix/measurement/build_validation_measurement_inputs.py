#!/usr/bin/env python3
"""Bind one completed synthetic VA cohort to deterministic measurement inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-plan", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.measurement.validation_measurement_binding import (
        write_validation_image_locators,
        write_validation_model_slice,
    )

    candidate = Path(args.output_dir).expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"output directory already exists: {candidate}")
    parent = candidate.parent.resolve(strict=True)
    output = parent / candidate.name
    os.mkdir(output, 0o700)
    os.chmod(output, 0o700)

    sliced = output / "validation-model-input.json"
    locator = output / "validation-image-locators.json"
    receipt = output / "validation-binding-receipt.json"
    write_validation_model_slice(args.source_model_plan, sliced)
    records = write_validation_image_locators(
        sliced,
        args.cohort_root,
        locator,
        receipt,
    )
    print(
        f"bound {len(records['locator']['images'])} synthetic stage images: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

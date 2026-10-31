#!/usr/bin/env python3
"""Assemble measured stage evidence from strict twin capture manifests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-input-plan",
        required=True,
        help="strict JSON model_input_plan artifact",
    )
    parser.add_argument(
        "--capture-manifest",
        action="append",
        required=True,
        help="raw verified twin_stage_capture_manifest JSON (repeatable)",
    )
    parser.add_argument("--out", required=True, help="new measured-evidence JSON path")
    parser.add_argument(
        "--image-locators-out",
        required=True,
        help="new exact image-locator JSON path for downstream measurement consumers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        build_measured_evidence,
    )
    from fluidmix.core.io import load_strict_json_snapshot
    from fluidmix.measurement.vision_measurement import _publish_json_no_clobber

    model_path = Path(args.model_input_plan)
    model_plan, model_snapshot = load_strict_json_snapshot(model_path)
    captures = []
    capture_hashes: list[str] = []
    for capture_arg in args.capture_manifest:
        capture_path = Path(capture_arg)
        payload, snapshot = load_strict_json_snapshot(capture_path)
        capture_hashes.append(str(snapshot["sha256"]))
        captures.append(
            bind_stage_capture_manifest(payload, base_dir=capture_path.parent)
        )

    measured = build_measured_evidence(model_plan, captures)
    measured["source_model_input_sha256"] = str(model_snapshot["sha256"])
    measured["source_capture_manifest_sha256"] = capture_hashes

    output_path = Path(args.out)
    locator_output_path = Path(args.image_locators_out)
    if output_path.absolute() == locator_output_path.absolute():
        raise ValueError("measured evidence and image locator outputs must differ")
    if os.path.lexists(output_path):
        raise FileExistsError(f"measured evidence output already exists: {output_path}")
    if os.path.lexists(locator_output_path):
        raise FileExistsError(
            f"image locator output already exists: {locator_output_path}"
        )

    locator_images: dict[str, object] = {}
    for capture in captures:
        locators = capture["image_locators"]
        assert isinstance(locators, dict)
        images = locators["images"]
        assert isinstance(images, dict)
        locator_images.update(images)
    image_locators = {
        "schema_version": 1,
        "data_origin": "twin",
        "images": locator_images,
    }

    locator_published = False
    try:
        _publish_json_no_clobber(locator_output_path, image_locators)
        locator_published = True
        _publish_json_no_clobber(output_path, measured)
    except Exception:
        if locator_published:
            try:
                locator_output_path.unlink()
            except FileNotFoundError:
                pass
        raise
    print(f"wrote measured stage evidence: {output_path}")
    print(f"wrote image locators: {locator_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

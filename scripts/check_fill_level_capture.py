"""Validate FillCam RGB/mask pairs and their per-frame JSONL manifest.

Paths in ``outputs`` are relative to the manifest by default.  The checker is
CPU-only and deliberately independent of Isaac Sim so copied pilot captures can
be validated before any raw output is used for training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
LABEL_IDS = {
    "empty": 0,
    "liquid": 1,
    "foam": 2,
    "blocked": 3,
    "ignore": 255,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_path(manifest_path, descriptor):
    if not isinstance(descriptor, dict):
        raise ValueError("output descriptor must be an object")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("output descriptor requires a non-empty path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _read_image(path, mode, name, errors):
    if not path.is_file():
        errors.append(f"{name} file does not exist: {path}")
        return None
    image = cv2.imread(str(path), mode)
    if image is None:
        errors.append(f"{name} could not be decoded: {path}")
    return image


def _check_hash(path, descriptor, name, errors):
    expected = descriptor.get("sha256") if isinstance(descriptor, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{name} requires a 64-character sha256")
    elif path.is_file():
        actual = file_sha256(path)
        if actual.casefold() != expected.casefold():
            errors.append(f"{name} sha256 mismatch: expected {expected}, got {actual}")


def check_record(record, manifest_path, index):
    """Return one frame report with an error list and measured label counts."""
    errors = []
    prefix = f"record {index}"
    if not isinstance(record, dict):
        return {"index": index, "errors": [f"{prefix} must be a JSON object"]}
    for field in ("seed", "role", "level_ml", "foam", "sticker_pose", "outputs"):
        if field not in record:
            errors.append(f"{prefix} missing {field}")
    outputs = record.get("outputs", {})
    rgb_descriptor = outputs.get("rgb") if isinstance(outputs, dict) else None
    mask_descriptor = outputs.get("mask") if isinstance(outputs, dict) else None
    rgb_path = mask_path = None
    try:
        rgb_path = _output_path(manifest_path, rgb_descriptor)
    except (TypeError, ValueError) as exc:
        errors.append(f"{prefix} RGB {exc}")
    try:
        mask_path = _output_path(manifest_path, mask_descriptor)
    except (TypeError, ValueError) as exc:
        errors.append(f"{prefix} mask {exc}")

    rgb = mask = None
    if rgb_path is not None:
        rgb = _read_image(rgb_path, cv2.IMREAD_UNCHANGED, "RGB", errors)
        _check_hash(rgb_path, rgb_descriptor, "RGB", errors)
    if mask_path is not None:
        mask = _read_image(mask_path, cv2.IMREAD_UNCHANGED, "mask", errors)
        _check_hash(mask_path, mask_descriptor, "mask", errors)

    if rgb is not None:
        if rgb.shape != (HEIGHT, WIDTH, 3):
            errors.append(
                f"RGB shape must be {(HEIGHT, WIDTH, 3)}, got {rgb.shape}"
            )
        if rgb.dtype != np.uint8:
            errors.append(f"RGB dtype must be uint8, got {rgb.dtype}")
        elif float(rgb.astype(np.float64).std(axis=(0, 1)).max()) <= 1.0:
            errors.append("RGB is blank or nearly constant")

    label_counts = {}
    if mask is not None:
        if mask.shape != (HEIGHT, WIDTH):
            errors.append(f"mask shape must be {(HEIGHT, WIDTH)}, got {mask.shape}")
        if mask.dtype != np.uint8:
            errors.append(f"mask dtype must be uint8, got {mask.dtype}")
        if mask.shape == (HEIGHT, WIDTH) and mask.dtype == np.uint8:
            identifiers, counts = np.unique(mask, return_counts=True)
            label_counts = {
                str(int(identifier)): int(count)
                for identifier, count in zip(identifiers, counts)
            }
            seen = {int(identifier) for identifier in identifiers}
            unexpected = seen - set(LABEL_IDS.values())
            if unexpected:
                errors.append(f"mask has unexpected class ids: {sorted(unexpected)}")
            for required in ("ignore", "empty", "blocked"):
                if LABEL_IDS[required] not in seen:
                    errors.append(f"mask is missing required {required} pixels")
            border = np.concatenate(
                (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
            )
            if np.any(border != LABEL_IDS["ignore"]):
                errors.append("mask boundary must be ignored outside the vial")

            try:
                level_ml = float(record["level_ml"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{prefix} level_ml must be numeric")
                level_ml = None
            if level_ml is not None and level_ml > 0 and LABEL_IDS["liquid"] not in seen:
                errors.append("positive liquid level requires visible liquid pixels")

            foam = record.get("foam", {})
            foam_enabled = bool(foam.get("enabled")) if isinstance(foam, dict) else False
            foam_expected_visible = (
                bool(foam.get("visible_in_camera", foam_enabled))
                if isinstance(foam, dict) else False
            )
            try:
                foam_thickness = float(foam.get("thickness_mm", 0.0))
            except (TypeError, ValueError):
                errors.append(f"{prefix} foam thickness_mm must be numeric")
                foam_thickness = 0.0
            expect_foam = (
                foam_enabled and foam_thickness > 0.0 and foam_expected_visible
            )
            if expect_foam and LABEL_IDS["foam"] not in seen:
                errors.append("enabled foam requires visible foam pixels")
            if (not foam_enabled or not foam_expected_visible) and LABEL_IDS["foam"] in seen:
                errors.append("disabled foam must not produce foam pixels")

    sticker = record.get("sticker_pose", {})
    sticker_present = bool(sticker.get("present")) if isinstance(sticker, dict) else False
    sticker_expected_visible = (
        bool(sticker.get("visible_in_camera", sticker_present))
        if isinstance(sticker, dict)
        else False
    )
    label_stats = record.get("label_stats", {})
    try:
        sticker_pixels = int(label_stats.get("sticker_blocked_pixels", 0))
    except (AttributeError, TypeError, ValueError):
        errors.append(f"{prefix} sticker_blocked_pixels must be an integer")
        sticker_pixels = 0
    if sticker_pixels < 0:
        errors.append(f"{prefix} sticker_blocked_pixels must be non-negative")
    if sticker_present and sticker_expected_visible and sticker_pixels <= 0:
        errors.append("visible sticker requires sticker_blocked_pixels > 0")
    if not sticker_present and sticker_pixels != 0:
        errors.append("absent sticker must have zero sticker_blocked_pixels")

    return {
        "index": index,
        "seed": record.get("seed"),
        "role": record.get("role"),
        "rgb": str(rgb_path) if rgb_path is not None else None,
        "mask": str(mask_path) if mask_path is not None else None,
        "label_counts": label_counts,
        "errors": errors,
    }


def validate_manifest(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        return {
            "schema_version": 1,
            "manifest": str(manifest_path),
            "records": [],
            "errors": [f"manifest does not exist: {manifest_path}"],
            "passed": False,
        }
    records = []
    parse_errors = []
    seen_keys = set()
    for line_number, raw_line in enumerate(manifest_path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"line {line_number}: {exc.msg}")
            continue
        report = check_record(record, manifest_path, line_number)
        key = (report.get("role"), report.get("seed"))
        if key in seen_keys:
            report["errors"].append(f"duplicate role/seed pair: {key}")
        seen_keys.add(key)
        records.append(report)
    if not records and not parse_errors:
        parse_errors.append("manifest contains no records")
    errors = list(parse_errors)
    errors.extend(
        error
        for report in records
        for error in report["errors"]
    )
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "records": records,
        "record_count": len(records),
        "errors": errors,
        "passed": not errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_manifest(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

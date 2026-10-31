"""Offline integrity checks for copied synthetic FillCam captures."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_fill_level_capture.py"
SPEC = importlib.util.spec_from_file_location("check_fill_level_capture", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_capture(tmp_path, *, foam=True, sticker=True):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
    rgb[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
    rgb[:, :, 2] = 77
    mask = np.full((480, 640), 255, dtype=np.uint8)
    mask[80:450, 280:340] = 0
    mask[280:450, 280:340] = 1
    if foam:
        mask[270:280, 280:340] = 2
    mask[210:260, 280:287] = 3
    rgb_path = tmp_path / "frame.png"
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(mask_path), mask)
    identifiers, counts = np.unique(mask, return_counts=True)
    measured_counts = {
        int(identifier): int(count)
        for identifier, count in zip(identifiers, counts)
    }
    record = {
        "seed": 7,
        "role": "pilot",
        "level_ml": 70.0,
        "foam": {"enabled": foam, "thickness_mm": 4.0 if foam else 0.0},
        "sticker_pose": {
            "present": sticker,
            "visible_in_camera": sticker,
        },
        "label_stats": {
            "class_pixels": {
                name: measured_counts.get(identifier, 0)
                for name, identifier in CHECKER.LABEL_IDS.items()
            },
            "vial_pixels": int(np.count_nonzero(mask != CHECKER.LABEL_IDS["ignore"])),
            "sticker_blocked_pixels": 12 if sticker else 0,
        },
        "outputs": {
            "rgb": {"path": rgb_path.name, "sha256": sha256(rgb_path)},
            "mask": {"path": mask_path.name, "sha256": sha256(mask_path)},
        },
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    return manifest, record, mask_path


def test_valid_manifest_passes(tmp_path):
    manifest, _, _ = write_capture(tmp_path)
    summary = CHECKER.validate_manifest(manifest)
    assert summary["passed"]
    assert summary["record_count"] == 1
    assert summary["records"][0]["label_counts"]["255"] > 0


def test_missing_file_and_hash_mismatch_fail(tmp_path):
    manifest, record, _ = write_capture(tmp_path)
    record["outputs"]["rgb"]["sha256"] = "0" * 64
    record["outputs"]["mask"]["path"] = "missing.png"
    manifest.write_text(json.dumps(record) + "\n")
    summary = CHECKER.validate_manifest(manifest)
    assert not summary["passed"]
    assert any("RGB sha256 mismatch" in error for error in summary["errors"])
    assert any("mask file does not exist" in error for error in summary["errors"])


def test_illegal_id_and_nonignored_boundary_fail(tmp_path):
    manifest, record, mask_path = write_capture(tmp_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask[0, 0] = 9
    assert cv2.imwrite(str(mask_path), mask)
    record["outputs"]["mask"]["sha256"] = sha256(mask_path)
    manifest.write_text(json.dumps(record) + "\n")
    summary = CHECKER.validate_manifest(manifest)
    assert any("unexpected class ids" in error for error in summary["errors"])
    assert any("boundary must be ignored" in error for error in summary["errors"])


def test_manifest_label_counts_must_match_mask(tmp_path):
    manifest, record, _ = write_capture(tmp_path)
    record["label_stats"]["class_pixels"]["liquid"] += 1
    record["label_stats"]["vial_pixels"] -= 1
    manifest.write_text(json.dumps(record) + "\n")
    errors = CHECKER.validate_manifest(manifest)["errors"]
    assert any("class_pixels.liquid" in error for error in errors)
    assert any("vial_pixels" in error for error in errors)


def test_missing_conditional_classes_fail(tmp_path):
    manifest, record, mask_path = write_capture(tmp_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask[mask == 1] = 0
    mask[mask == 2] = 0
    assert cv2.imwrite(str(mask_path), mask)
    record["outputs"]["mask"]["sha256"] = sha256(mask_path)
    record["label_stats"]["sticker_blocked_pixels"] = 0
    manifest.write_text(json.dumps(record) + "\n")
    summary = CHECKER.validate_manifest(manifest)
    assert any("positive liquid level" in error for error in summary["errors"])
    assert any("enabled foam" in error for error in summary["errors"])
    assert any("visible sticker" in error for error in summary["errors"])


def test_disabled_foam_and_absent_sticker_have_no_labels(tmp_path):
    manifest, _, _ = write_capture(tmp_path, foam=False, sticker=False)
    assert CHECKER.validate_manifest(manifest)["passed"]


def test_enabled_but_occluded_foam_may_have_no_visible_pixels(tmp_path):
    manifest, record, mask_path = write_capture(tmp_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    foam_pixels = int(np.count_nonzero(mask == CHECKER.LABEL_IDS["foam"]))
    mask[mask == 2] = 0
    assert cv2.imwrite(str(mask_path), mask)
    record["foam"]["visible_in_camera"] = False
    record["label_stats"]["class_pixels"]["foam"] = 0
    record["label_stats"]["class_pixels"]["empty"] += foam_pixels
    record["outputs"]["mask"]["sha256"] = sha256(mask_path)
    manifest.write_text(json.dumps(record) + "\n")
    assert CHECKER.validate_manifest(manifest)["passed"]


def test_empty_or_invalid_json_manifest_fails(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    assert not CHECKER.validate_manifest(empty)["passed"]
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{broken\n")
    result = CHECKER.validate_manifest(invalid)
    assert not result["passed"]
    assert any("line 1" in error for error in result["errors"])

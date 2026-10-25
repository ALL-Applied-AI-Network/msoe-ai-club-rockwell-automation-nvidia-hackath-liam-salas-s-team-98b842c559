"""Offline checks must validate artifact bytes rather than receipt metadata alone."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_physx_f2_artifacts",
    ROOT / "scripts" / "validate_physx_f2_artifacts.py",
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.mark.parametrize(
    "unsafe",
    ["", ".", "../escape.png", "frames/../../escape.png", "/tmp/escape.png"],
)
def test_resolve_artifact_rejects_empty_absolute_and_traversal(tmp_path, unsafe):
    with pytest.raises(ValueError):
        validator.resolve_artifact(tmp_path, unsafe)


def test_resolve_artifact_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-artifact.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "link.bin"
    link.symlink_to(outside)

    with pytest.raises(ValueError):
        validator.resolve_artifact(tmp_path, "link.bin")


def test_resolve_artifact_accepts_existing_regular_file(tmp_path):
    target = tmp_path / "frames" / "frame.png"
    target.parent.mkdir()
    target.write_bytes(b"frame")

    assert validator.resolve_artifact(tmp_path, "frames/frame.png") == target.resolve()


def test_verify_hash_reads_actual_bytes_and_rejects_metadata_only_match(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual bytes")
    expected = hashlib.sha256(b"actual bytes").hexdigest()

    validator.verify_hash(artifact, expected)
    with pytest.raises(validator.ArtifactValidationError, match="SHA-256"):
        validator.verify_hash(artifact, "0" * 64)


def test_rgb_mask_and_semantic_readability_checks_actual_shapes(tmp_path):
    rgb = np.full((4, 5, 3), 127, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    semantic = np.arange(20, dtype=np.uint32).reshape(4, 5)
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    semantic_path = tmp_path / "semantic.npy"
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(mask_path), mask)
    np.save(semantic_path, semantic, allow_pickle=False)

    assert validator.read_rgb(rgb_path, 5, 4).shape == (4, 5, 3)
    assert int(np.count_nonzero(validator.read_mask(mask_path, 5, 4))) == 4
    np.testing.assert_array_equal(
        validator.read_semantic(semantic_path, 5, 4), semantic
    )

    with pytest.raises(validator.ArtifactValidationError, match="shape"):
        validator.read_rgb(rgb_path, 6, 4)


def test_mask_must_contain_liquid_and_semantic_must_be_integer(tmp_path):
    empty_mask = tmp_path / "empty.png"
    floating_semantic = tmp_path / "semantic.npy"
    assert cv2.imwrite(str(empty_mask), np.zeros((3, 4), dtype=np.uint8))
    np.save(
        floating_semantic,
        np.zeros((3, 4), dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(validator.ArtifactValidationError, match="nonempty"):
        validator.read_mask(empty_mask, 4, 3)
    with pytest.raises(validator.ArtifactValidationError, match="integer"):
        validator.read_semantic(floating_semantic, 4, 3)


def test_semantic_artifact_must_reproduce_exact_class_mask_mapping_and_counts():
    semantic = np.array([[0, 5, 5], [7, 5, 0]], dtype=np.uint32)
    mask = np.array([[0, 255, 255], [0, 255, 0]], dtype=np.uint8)
    pair = {
        "semantic_unique_ids": [0, 5, 7],
        "liquid_semantic_ids": [5],
        "semantic_id_to_class_labels": {
            "0": ["BACKGROUND"],
            "5": ["container", "liquid"],
            "7": ["vial"],
        },
        "liquid_pixel_counts_by_id": {"5": 3},
        "liquid_pixel_count": 3,
    }

    validator.validate_semantic_mask_evidence(semantic, mask, pair)

    tampered = mask.copy()
    tampered[1, 0] = 255
    with pytest.raises(validator.ArtifactValidationError, match="semantic class IDs"):
        validator.validate_semantic_mask_evidence(semantic, tampered, pair)

    bad_counts = dict(pair, liquid_pixel_counts_by_id={"5": 2})
    with pytest.raises(validator.ArtifactValidationError, match="per-ID pixel counts"):
        validator.validate_semantic_mask_evidence(semantic, mask, bad_counts)

    bad_mapping = dict(
        pair,
        semantic_id_to_class_labels={
            "0": ["BACKGROUND"],
            "05": ["liquid"],
            "5": ["vial"],
            "7": ["vial"],
        },
    )
    with pytest.raises(validator.ArtifactValidationError, match="mapping"):
        validator.validate_semantic_mask_evidence(semantic, mask, bad_mapping)


def test_video_decoder_checks_actual_frame_count_and_resolution(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (8, 6)
    )
    assert writer.isOpened()
    for value in (20, 80, 140):
        writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    writer.release()

    assert validator.validate_video(video_path, 8, 6, 3) == 3
    with pytest.raises(validator.ArtifactValidationError, match="frame count"):
        validator.validate_video(video_path, 8, 6, 4)
    with pytest.raises(validator.ArtifactValidationError, match="resolution"):
        validator.validate_video(video_path, 9, 6, 3)


def test_validate_output_calls_schema_contract_and_fails_closed(tmp_path):
    (tmp_path / "receipt.json").write_text("{}\n", encoding="utf-8")

    result = validator.validate_output(tmp_path)

    assert result["status"] == "insufficient_evidence"
    assert "receipt:receipt_identity" in result["failures"]
    assert "receipt:run_status_not_pass" in result["failures"]


def test_validate_output_reports_unsafe_paths_even_when_receipt_is_invalid(tmp_path):
    receipt = {
        "render": {
            "resolution": [640, 480],
            "frame_paths": ["../outside.png"],
            "frame_sha256": ["0" * 64],
        }
    }
    (tmp_path / "receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    result = validator.validate_output(tmp_path)

    assert result["status"] == "insufficient_evidence"
    assert any("unsafe" in failure or "safe relative" in failure for failure in result["failures"])


def test_validate_output_reports_missing_or_unreadable_receipt(tmp_path):
    missing = validator.validate_output(tmp_path)
    assert missing["status"] == "insufficient_evidence"
    assert missing["failures"] == ["receipt_missing_or_unreadable"]

    (tmp_path / "receipt.json").write_text("not json", encoding="utf-8")
    unreadable = validator.validate_output(tmp_path)
    assert unreadable["status"] == "insufficient_evidence"
    assert unreadable["failures"] == ["receipt_missing_or_unreadable"]


def test_validate_output_fails_closed_when_output_directory_is_missing(tmp_path):
    missing_root = tmp_path / "does-not-exist"

    result = validator.validate_output(missing_root)

    assert result["status"] == "insufficient_evidence"
    assert result["failures"] == ["receipt_missing_or_unreadable"]
    assert validator.main([str(missing_root)]) == 2


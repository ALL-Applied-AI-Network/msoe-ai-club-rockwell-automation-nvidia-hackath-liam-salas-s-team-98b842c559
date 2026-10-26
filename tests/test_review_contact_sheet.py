"""Tests for receipt-bound PhysX visual-review contact sheets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidmix.review_contact_sheet import create_physx_review_contact_sheet


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    frames = root / "frames"
    masks = root / "liquid"
    frames.mkdir(parents=True)
    masks.mkdir()
    pairs: list[dict[str, object]] = []
    frame_paths: list[str] = []
    frame_hashes: list[str] = []
    for ordinal, (step, channel, value) in enumerate(
        ((12, 0, 100), (24, 1, 140)), start=1
    ):
        rgb_path = frames / f"frame-{ordinal:04d}-step-{step:05d}.png"
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[:, :, channel] = value
        Image.fromarray(rgb, mode="RGB").save(rgb_path)

        mask_path = masks / f"frame-{ordinal:04d}-step-{step:05d}.png"
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[ordinal : ordinal + 3, ordinal + 1 : ordinal + 4] = 255
        Image.fromarray(mask, mode="L").save(mask_path)

        relative_rgb = rgb_path.relative_to(root).as_posix()
        relative_mask = mask_path.relative_to(root).as_posix()
        rgb_hash = _sha256(rgb_path)
        frame_paths.append(relative_rgb)
        frame_hashes.append(rgb_hash)
        pairs.append(
            {
                "physics_step": step,
                "phase": f"pour-{ordinal}",
                "rgb_path": relative_rgb,
                "rgb_sha256": rgb_hash,
                "mask_path": relative_mask,
                "mask_sha256": _sha256(mask_path),
                "liquid_pixel_count": 9,
            }
        )
    receipt = {
        "schema_version": 14,
        "status": "pass",
        "render": {
            "resolution": [64, 48],
            "frame_count": 2,
            "frame_paths": frame_paths,
            "frame_sha256": frame_hashes,
            "frame_physics_steps": [12, 24],
        },
        "semantics": {"pairs": pairs},
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def test_contact_sheet_publishes_receipt_bound_rgb_and_mask_overlay_grid(tmp_path):
    receipt = _write_fixture(tmp_path / "capture")
    output = tmp_path / "review-sheet"

    result = create_physx_review_contact_sheet(
        receipt,
        output,
        physics_steps=(24, 12),
        columns=2,
    )

    assert result == output / "contact-sheet.json"
    record = json.loads(result.read_text())
    assert record["status"] == "contact_sheet_verified"
    assert record["selection"] == {
        "columns": 2,
        "physics_steps": [24, 12],
        "rows": 1,
    }
    assert record["source_receipt"]["sha256"] == _sha256(receipt)
    assert [row["phase"] for row in record["source_frames"]] == ["pour-2", "pour-1"]
    assert record["contact_sheet"]["sha256"] == _sha256(
        output / "contact-sheet.png"
    )

    with Image.open(output / "contact-sheet.png") as image:
        assert image.mode == "RGB"
        assert image.size == (128, 120)
        pixels = np.asarray(image)
    # Requested order is preserved: step 24 (green) is the first tile.
    assert pixels[24, 0].tolist() == [0, 140, 0]
    # The receipt-bound binary mask is shown as magenta in the lower overlay.
    assert pixels[72 + 2, 3].tolist() == [255, 0, 255]
    # Step 12 (red) occupies the second tile.
    assert pixels[24, 64].tolist() == [100, 0, 0]
    assert pixels[72 + 1, 64 + 2].tolist() == [255, 0, 255]


def test_contact_sheet_rejects_mask_hash_mismatch_without_publishing(tmp_path):
    receipt = _write_fixture(tmp_path / "capture")
    value = json.loads(receipt.read_text())
    value["semantics"]["pairs"][1]["mask_sha256"] = "0" * 64
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "review-sheet"

    with pytest.raises(ValueError, match="liquid mask hash does not match receipt"):
        create_physx_review_contact_sheet(
            receipt,
            output,
            physics_steps=(24,),
        )

    assert not output.exists()


def test_contact_sheet_records_and_applies_one_fixed_review_crop(tmp_path):
    receipt = _write_fixture(tmp_path / "capture")
    output = tmp_path / "review-crop"

    result = create_physx_review_contact_sheet(
        receipt,
        output,
        physics_steps=(12,),
        columns=1,
        crop_xyxy=(1, 1, 7, 6),
    )

    record = json.loads(result.read_text())
    assert record["selection"]["crop_xyxy_exclusive"] == [1, 1, 7, 6]
    with Image.open(output / "contact-sheet.png") as image:
        assert image.size == (6, 34)


def test_contact_sheet_cli_is_help_safe_and_exposes_selection_flags():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_physx_review_contact_sheet.py"
    )

    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--receipt" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--steps" in completed.stdout
    assert "--columns" in completed.stdout
    assert "--crop" in completed.stdout


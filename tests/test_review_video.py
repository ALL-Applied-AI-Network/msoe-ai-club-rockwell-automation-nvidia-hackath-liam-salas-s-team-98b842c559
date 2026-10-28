"""Tests for receipt-bound PhysX review-video encoding."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidmix.review_video import encode_physx_receipt_video


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path, *, corrupt_hash: bool = False) -> Path:
    frames = root / "frames"
    frames.mkdir(parents=True)
    paths: list[str] = []
    hashes: list[str] = []
    steps = [12, 24, 36]
    for ordinal, step in enumerate(steps, start=1):
        path = frames / f"frame-{ordinal:04d}-step-{step:05d}.png"
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[:, :, ordinal - 1] = 80 + 40 * ordinal
        pixels[ordinal : ordinal + 8, ordinal * 3 : ordinal * 3 + 12] = 255
        Image.fromarray(pixels, mode="RGB").save(path)
        paths.append(path.relative_to(root).as_posix())
        hashes.append(_sha256(path))
    if corrupt_hash:
        hashes[1] = "0" * 64
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "render": {
            "resolution": [64, 48],
            "frame_count": len(paths),
            "frame_paths": paths,
            "frame_sha256": hashes,
            "frame_physics_steps": steps,
        },
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


@pytest.mark.skipif(importlib.util.find_spec("av") is None, reason="PyAV is optional")
def test_encoder_publishes_decoded_h264_mp4_bound_to_png_receipt(tmp_path):
    receipt = _write_fixture(tmp_path / "capture")
    output = tmp_path / "review-video"

    result = encode_physx_receipt_video(receipt, output, fps=6)

    assert result == output / "video.json"
    record = json.loads(result.read_text())
    assert record["status"] == "encoded_verified"
    assert record["encoding"] == {
        "codec_requested": "libx264",
        "container": "mp4",
        "fps": 6,
        "height": 48,
        "pixel_format": "yuv420p",
        "width": 64,
    }
    assert record["verification"]["decoded_frame_count"] == 3
    assert record["verification"]["decoded_pts"] == [0, 1, 2]
    assert record["verification"]["decoded_width"] == 64
    assert record["verification"]["decoded_height"] == 48
    assert record["verification"]["reported_average_rate"] == "6"
    assert record["source_frames"] == [
        {
            "ordinal": ordinal,
            "physics_step": step,
            "path": path,
            "sha256": sha256,
        }
        for ordinal, (step, path, sha256) in enumerate(
            zip(
                [12, 24, 36],
                [
                    "frames/frame-0001-step-00012.png",
                    "frames/frame-0002-step-00024.png",
                    "frames/frame-0003-step-00036.png",
                ],
                [
                    _sha256(tmp_path / "capture" / "frames/frame-0001-step-00012.png"),
                    _sha256(tmp_path / "capture" / "frames/frame-0002-step-00024.png"),
                    _sha256(tmp_path / "capture" / "frames/frame-0003-step-00036.png"),
                ],
            ),
            start=1,
        )
    ]
    video = output / "sequence.mp4"
    assert video.stat().st_size > 0
    assert record["video"]["sha256"] == _sha256(video)
    assert record["limits"]["measurement_source"] == "hash-bound PNGs only"
    assert record["limits"]["video_role"] == "lossy review visualization only"


def test_encoder_rejects_receipt_hash_mismatch_before_publishing(tmp_path):
    receipt = _write_fixture(tmp_path / "capture", corrupt_hash=True)
    output = tmp_path / "review-video"

    with pytest.raises(ValueError, match="hash does not match"):
        encode_physx_receipt_video(receipt, output, fps=6)

    assert not output.exists()


@pytest.mark.parametrize("fps", [0, -1, 0.5, True, "6"])
def test_encoder_rejects_invalid_fps(tmp_path, fps):
    receipt = _write_fixture(tmp_path / "capture")

    with pytest.raises((TypeError, ValueError), match="fps"):
        encode_physx_receipt_video(receipt, tmp_path / "review-video", fps=fps)


def test_encoder_does_not_clobber_existing_output(tmp_path):
    receipt = _write_fixture(tmp_path / "capture")
    output = tmp_path / "review-video"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError):
        encode_physx_receipt_video(receipt, output, fps=6)

    assert marker.read_text() == "keep"


def test_encoder_cli_is_help_safe_and_exposes_receipt_output_and_fps_flags():
    script = Path(__file__).resolve().parents[1] / "scripts/encode_physx_review_video.py"

    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--receipt" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--fps" in completed.stdout

"""Synthetic-only tests for the organizer-still photometric manifest builder."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from fluidmix.storage.artifacts.real_photometric_manifest import build_photometric_frame_manifest
from fluidmix.measurement.real_photometric_range import DEFAULT_STABLE_REGION_CONFIG, build_photometric_receipt


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    images = root / "images"
    images.mkdir()
    for name, value in (("v1-a.png", 0), ("v1-b.png", 100), ("v2-a.png", 200)):
        Image.fromarray(np.full((20, 30, 3), value, dtype=np.uint8), "RGB").save(images / name)
    index = root / "image_index.csv"
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "timestamp", "video_id"])
        writer.writeheader()
        writer.writerows([
            {"filename": "v1-b.png", "timestamp": "2026-09-22 17:04:34.000000", "video_id": "1"},
            {"filename": "v2-a.png", "timestamp": "2026-09-22 17:04:35.000000", "video_id": "2"},
            {"filename": "v1-a.png", "timestamp": "2026-09-22 17:04:32.000000", "video_id": "1"},
        ])
    config = root / "stable-region.json"
    config.write_text(json.dumps(DEFAULT_STABLE_REGION_CONFIG), encoding="utf-8")
    return index, images, config


def test_builder_hashes_inputs_and_computes_within_video_crop_motion(tmp_path):
    index, images, config = _write_inputs(tmp_path)

    manifest = build_photometric_frame_manifest(index, images, config)

    assert manifest["schema_version"] == 1
    assert manifest["source_image_index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert manifest["stable_region_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert [row["frame_id"] for row in manifest["frames"]] == ["v1-a.png", "v1-b.png", "v2-a.png"]
    assert manifest["frames"][0]["motion_score"] is None
    assert manifest["frames"][0]["motion_status"] == "no_previous_frame_in_video"
    assert manifest["frames"][1]["motion_score"] == pytest.approx(100 / 255)
    assert manifest["frames"][1]["motion_reference_frame_id"] == "v1-a.png"
    assert manifest["frames"][1]["source_sha256"] == hashlib.sha256((images / "v1-b.png").read_bytes()).hexdigest()
    assert manifest["motion_metric"] == "mean_absolute_rgb_difference_divided_by_255_in_configured_crop"
    assert all("liquid_label" not in row and "event_label" not in row for row in manifest["frames"])


def test_builder_rejects_held_out_video_five(tmp_path):
    index, images, config = _write_inputs(tmp_path)
    with index.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=["filename", "timestamp", "video_id"]).writerow(
            {"filename": "v5-a.png", "timestamp": "2026-09-22 17:04:36.000000", "video_id": "5"}
        )

    with pytest.raises(ValueError, match="video 5"):
        build_photometric_frame_manifest(index, images, config)


def test_receipt_honestly_excludes_first_frames_without_motion_scores(tmp_path):
    index, images, config = _write_inputs(tmp_path)
    config.write_text(json.dumps({**DEFAULT_STABLE_REGION_CONFIG, "max_motion_score": 0.5}), encoding="utf-8")
    manifest = build_photometric_frame_manifest(index, images, config)
    manifest_path = tmp_path / "frames.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    receipt = build_photometric_receipt(manifest_path)

    assert receipt["pooled"]["candidate_count"] == 3
    assert receipt["pooled"]["usable_count"] == 1
    assert {row["reason"] for row in receipt["exclusions"]} == {"no_previous_frame_in_video"}


def test_builder_cli_writes_a_no_clobber_manifest_below_outputs(tmp_path):
    index, images, config = _write_inputs(tmp_path)
    output_root = tmp_path / "outputs"
    script = Path(__file__).resolve().parents[4] / "cli/fluidmix/storage/artifacts/build_real_photometric_manifest.py"
    environment = {**os.environ, "FLUIDMIX_OUTPUTS": str(output_root)}

    completed = subprocess.run(
        [sys.executable, str(script), "--image-index", str(index), "--images-root", str(images),
         "--stable-region-config", str(config), "--output", "receipts/frames.json"],
        capture_output=True, text=True, env=environment,
    )

    output = output_root / "receipts/frames.json"
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["frames"][1]["motion_score"] == pytest.approx(100 / 255)
    second = subprocess.run(
        [sys.executable, str(script), "--image-index", str(index), "--images-root", str(images),
         "--stable-region-config", str(config), "--output", "receipts/frames.json"],
        capture_output=True, text=True, env=environment,
    )
    assert second.returncode != 0

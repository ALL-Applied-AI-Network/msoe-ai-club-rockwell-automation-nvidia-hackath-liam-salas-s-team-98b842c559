"""Synthetic-only tests for real-feed photometric receipt mechanics."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.real_photometric_range import (
    DEFAULT_STABLE_REGION_CONFIG,
    build_photometric_receipt,
    write_no_clobber_json,
)


def _frame(path: Path, *, shadow: bool = False) -> None:
    pixels = np.full((20, 30, 3), 200, dtype=np.uint8)
    if shadow:
        pixels[:, :10] = 40
        pixels[:, 10:14] = np.array([80, 100, 130], dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)


def _manifest(root: Path) -> Path:
    first, second, moving = root / "v1-a.png", root / "v2-a.png", root / "v3-moving.png"
    _frame(first, shadow=True)
    _frame(second)
    _frame(moving)
    payload = {
        "schema_version": 1,
        "stable_region_config": DEFAULT_STABLE_REGION_CONFIG,
        "frames": [
            {"frame_id": "v1-a", "video_id": 1, "image_path": str(first), "motion_score": 0.0},
            {"frame_id": "v2-a", "video_id": 2, "image_path": str(second), "motion_score": 0.0},
            {"frame_id": "v3-moving", "video_id": 3, "image_path": str(moving), "motion_score": 1.0},
        ],
    }
    path = root / "frames.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_photometric_receipt_is_deterministic_hash_bound_and_excludes_motion(tmp_path):
    manifest = _manifest(tmp_path)

    receipt = build_photometric_receipt(manifest)

    assert receipt["status"] == "measured"
    assert receipt["photometric_status"] == "measured"
    assert receipt["shadow_status"] == "measured"
    assert receipt["allowed_video_ids"] == [1, 2, 3, 4]
    assert receipt["source_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["stable_region"]["config_id"] == "fixed_background_v1"
    assert receipt["pooled"]["usable_count"] == 2
    assert receipt["pooled"]["candidate_count"] == 3
    assert receipt["pooled"]["excluded_count"] == 1
    assert receipt["per_video"]["1"]["usable_count"] == 1
    assert receipt["per_video"]["3"]["usable_count"] == 0
    assert receipt["exclusions"] == [{
        "frame_id": "v3-moving",
        "reason": "motion_score_above_max",
        "source_sha256": hashlib.sha256((tmp_path / "v3-moving.png").read_bytes()).hexdigest(),
    }]
    first = receipt["measurements"][0]
    assert first["source_sha256"] == hashlib.sha256((tmp_path / "v1-a.png").read_bytes()).hexdigest()
    assert first["metrics"]["luma"] > 0
    assert first["shadow"]["status"] == "measured"
    assert first["shadow"]["centroid_xy"] is not None
    assert first["shadow"]["edge_angle_degrees"] is not None
    assert first["shadow"]["penumbra_10_90_px"] is not None
    assert receipt["pooled"]["percentile_bounds"]["shadow_centroid_x"] is not None
    assert receipt["per_video"]["1"]["percentile_bounds"]["shadow_edge_angle_degrees"] is not None
    assert receipt == build_photometric_receipt(manifest)


def test_photometric_receipt_records_an_honest_empty_denominator(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    for row in payload["frames"]:
        row["motion_score"] = 1.0
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    receipt = build_photometric_receipt(manifest)

    assert receipt["status"] == "insufficient_evidence"
    assert receipt["pooled"]["candidate_count"] == 3
    assert receipt["pooled"]["usable_count"] == 0
    assert receipt["pooled"]["excluded_count"] == 3
    assert receipt["insufficiency_reasons"] == ["no_usable_stable_region_frames"]


def test_photometric_receipt_keeps_photometry_when_all_shadows_are_not_measurable(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["frames"] = payload["frames"][1:2]
    payload["frames"][0]["motion_score"] = 0.0
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    receipt = build_photometric_receipt(manifest)

    assert receipt["status"] == "measured"
    assert receipt["photometric_status"] == "measured"
    assert receipt["shadow_status"] == "insufficient_evidence"
    assert receipt["shadow_insufficiency_reasons"] == ["no_measurable_shadow_observations"]
    assert receipt["pooled"]["percentile_bounds"]["shadow_centroid_x"] is None


def test_photometric_receipt_rejects_held_out_video_five_and_manual_liquid_polygon(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["frames"][0]["video_id"] = 5
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="video 5"):
        build_photometric_receipt(manifest)

    payload["frames"][0]["video_id"] = 1
    payload["stable_region_config"]["liquid_polygon"] = [[0, 0], [1, 0], [1, 1]]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="liquid polygon"):
        build_photometric_receipt(manifest)


def test_photometric_no_clobber_preserves_existing_receipt(tmp_path):
    output = tmp_path / "receipt.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_no_clobber_json(output, {"status": "measured"})

    assert output.read_text(encoding="utf-8") == "keep"


def test_photometric_cli_writes_no_clobber_receipt_below_outputs(tmp_path):
    manifest = _manifest(tmp_path)
    output_root = tmp_path / "outputs"
    script = Path(__file__).resolve().parents[1] / "scripts/measure_real_photometric_ranges.py"
    environment = {**os.environ, "FLUIDMIX_OUTPUTS": str(output_root)}

    first = subprocess.run(
        [sys.executable, str(script), "--input", str(manifest), "--output", "receipts/photometric.json"],
        capture_output=True, text=True, env=environment,
    )

    output = output_root / "receipts/photometric.json"
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text())["status"] == "measured"
    second = subprocess.run(
        [sys.executable, str(script), "--input", str(manifest), "--output", "receipts/photometric.json"],
        capture_output=True, text=True, env=environment,
    )
    assert second.returncode != 0

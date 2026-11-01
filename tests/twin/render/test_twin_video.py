"""Tests for deterministic review-video encoding of verified twin captures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import twin.render.twin_video as twin_video
from twin.render.twin_video import encode_capture_video


VIDEO_TIME_CODES = tuple(sorted(set(range(31, 687, 5)) | {342}))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video_capture(root: Path, time_codes=VIDEO_TIME_CODES) -> Path:
    for kind in ("rgb", "semantic", "instance", "liquid"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    frames = []
    for ordinal, time_code in enumerate(time_codes, start=1):
        stem = f"{time_code:04d}"
        rgb = root / "rgb" / f"{stem}.png"
        liquid = root / "liquid" / f"{stem}.png"
        semantic = root / "semantic" / f"{stem}.npy"
        instance = root / "instance" / f"{stem}.npy"
        Image.new(
            "RGB",
            (640, 480),
            (ordinal % 251, (ordinal * 3) % 251, (ordinal * 7) % 251),
        ).save(rgb)
        Image.new("L", (640, 480), 255).save(liquid)
        np.save(semantic, np.array([[5]], dtype=np.uint32), allow_pickle=False)
        np.save(instance, np.array([[45]], dtype=np.uint32), allow_pickle=False)
        artifacts = {}
        for kind, path in (
            ("rgb", rgb),
            ("semantic", semantic),
            ("instance", instance),
            ("liquid", liquid),
        ):
            artifacts[kind] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        frames.append(
            {
                "capture_ordinal": ordinal,
                "time_code": time_code,
                "artifacts": artifacts,
            }
        )
    frame_spec = ",".join(str(code) for code in time_codes)
    summary = {
        "schema_version": 1,
        "mechanical_gates_passed": True,
        "visual_check_status": "pending",
        "inputs": {
            "stage": {"requested": "/synthetic/stage.usda"},
            "camera": {"name": "FillCam"},
        },
        "render_settings": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "frame_spec": frame_spec,
            "time_codes": list(time_codes),
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_tries_per_frame": 20,
            "require_liquid_pixels": True,
            "replicator_global_seed": 1310001,
        },
        "frame_count": len(frames),
        "frames": frames,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return root


def test_encoder_writes_verified_mp4_and_sha_bound_provenance(tmp_path):
    capture = _write_video_capture(tmp_path / "capture")
    output = tmp_path / "encoded"

    result = encode_capture_video(capture, output)

    assert result == output / "video.json"
    record = json.loads(result.read_text())
    assert record["status"] == "encoded_verified"
    assert record["encoding"] == {
        "codec_requested": "mp4v",
        "container": "mp4",
        "fps": 6.0,
        "height": 480,
        "width": 640,
    }
    assert record["verification"]["decoded_frame_count"] == 133
    assert record["verification"]["reported_fourcc"] in {"FMP4", "mp4v", "MP4V"}
    assert record["time_codes"] == list(VIDEO_TIME_CODES)
    assert record["time_codes_sha256"] == (
        "234222dbeccd22636d0738ac40ae96ebb411c1c144c1264a14378736c763b112"
    )
    assert record["stage_frames"] == [
        {
            "capture_ordinal": 29,
            "rgb_sha256": record["source_frames"][28]["rgb_sha256"],
            "time_code": 171,
        },
        {
            "capture_ordinal": 64,
            "rgb_sha256": record["source_frames"][63]["rgb_sha256"],
            "time_code": 342,
        },
        {
            "capture_ordinal": 133,
            "rgb_sha256": record["source_frames"][132]["rgb_sha256"],
            "time_code": 686,
        },
    ]
    assert [row["capture_ordinal"] for row in record["source_frames"]] == list(
        range(1, 134)
    )
    assert (output / "sequence.mp4").stat().st_size > 0
    assert record["video"]["sha256"] == _sha256(output / "sequence.mp4")
    assert record["limits"]["measurement_source"] == "hash-bound capture PNGs only"
    assert record["limits"]["video_role"] == "lossy review visualization only"


def test_encoder_rejects_existing_output_without_clobbering(tmp_path):
    capture = _write_video_capture(tmp_path / "capture")
    output = tmp_path / "encoded"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError):
        encode_capture_video(capture, output)

    assert marker.read_text() == "keep"


def test_encoder_rejects_other_mechanically_valid_frame_schedule(tmp_path):
    capture = _write_video_capture(tmp_path / "capture", VIDEO_TIME_CODES[:-1])

    with pytest.raises(ValueError, match="frozen 133-frame"):
        encode_capture_video(capture, tmp_path / "encoded")

    assert not (tmp_path / "encoded").exists()


def test_encoder_reverification_rejects_capture_drift_and_cleans_partial(
    tmp_path, monkeypatch
):
    capture = _write_video_capture(tmp_path / "capture")
    original_verify = twin_video.verify_capture_summary
    calls = 0

    def drift_before_second_verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            (capture / "rgb/0031.png").write_bytes(b"drift")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(twin_video, "verify_capture_summary", drift_before_second_verify)

    with pytest.raises(ValueError, match="hash drifted"):
        encode_capture_video(capture, tmp_path / "encoded")

    assert calls == 2
    assert not (tmp_path / "encoded").exists()
    assert not list(tmp_path.glob(".encoded.partial-*"))


def test_encoder_rejects_output_nested_inside_capture(tmp_path):
    capture = _write_video_capture(tmp_path / "capture")

    with pytest.raises(ValueError, match="outside the capture"):
        encode_capture_video(capture, capture / "encoded")


def test_encoder_cli_is_help_safe_and_exposes_capture_and_output_flags():
    script = Path(__file__).resolve().parents[3] / "cli/twin/render/encode_twin_capture_video.py"
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--capture-root" in completed.stdout
    assert "--output-dir" in completed.stdout

"""CPU-only contracts for palette-boundary review reels."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _video(
    path: Path,
    color: tuple[int, int, int],
    *,
    frames: int = 2,
    rate: int = 2,
    size: tuple[int, int] = (320, 180),
    pixel_format: str = "yuv420p",
) -> Path:
    with av.open(os.fspath(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = size
        stream.pix_fmt = pixel_format
        stream.codec_context.time_base = Fraction(1, rate)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {"crf": "12", "preset": "ultrafast"}
        for index in range(frames):
            pixels = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _request(path: Path, videos: list[tuple[str, Path]]) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_type": "fluidmix_palette_cut_reel_request_v1",
        "run_order": [run_id for run_id, _ in videos],
        "runs": [
            {
                "ordinal": ordinal,
                "run_id": run_id,
                "video_path": os.fspath(video),
                "video_sha256": _sha256(video),
            }
            for ordinal, (run_id, video) in enumerate(videos, start=1)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_encoder_inserts_exact_visible_card_and_publishes_bound_inventory(tmp_path):
    from fluidmix.palette_cut_reel import CUT_TEXT_LINES, encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (160, 20, 20))
    second = _video(tmp_path / "run-02.mp4", (20, 20, 160))
    request = _request(
        tmp_path / "request.json",
        [("RUN-TRAIN-01", first), ("RUN-TRAIN-02", second)],
    )
    output = tmp_path / "reel"

    receipt_path = encode_palette_cut_reel(
        request,
        output,
        owner_thread_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        repo_root=ROOT,
    )

    assert receipt_path == output / "palette-cut-reel.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "encoded_verified"
    assert receipt["owner_thread_id"] == "01a18128-8dc4-7780-ace1-4d816b49d176"
    assert receipt["source_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert [row["run_id"] for row in receipt["ordered_inputs"]] == [
        "RUN-TRAIN-01",
        "RUN-TRAIN-02",
    ]
    assert [row["video_sha256"] for row in receipt["ordered_inputs"]] == [
        _sha256(first),
        _sha256(second),
    ]
    assert receipt["cut_cards"] == [
        {
            "boundary_ordinal": 1,
            "after_run_id": "RUN-TRAIN-01",
            "before_run_id": "RUN-TRAIN-02",
            "start_frame_index": 2,
            "stop_frame_index": 6,
            "frame_count": 4,
            "text_lines": list(CUT_TEXT_LINES),
            "transport_crosses_boundary": False,
        }
    ]
    assert receipt["output"]["frame_count"] == 8
    assert receipt["output"]["fps"] == "2"
    assert receipt["output"]["geometry"] == {"width": 320, "height": 180}
    assert receipt["output"]["sha256"] == _sha256(output / "palette-cut-reel.mp4")
    assert receipt["transport_boundary_policy"] == {
        "transport_frames_added_by_reel": False,
        "transport_crosses_palette_boundaries": False,
        "boundary_mechanism": "fixed-duration visible cut card only",
    }
    inventory = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert inventory == [
        f"{_sha256(output / 'palette-cut-reel.mp4')}  palette-cut-reel.mp4",
        f"{_sha256(receipt_path)}  palette-cut-reel.json",
    ]

    with av.open(os.fspath(output / "palette-cut-reel.mp4"), mode="r") as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(decoded) == 8
    # The card is visibly distinct from both source runs, not a silent hard cut.
    assert np.mean(decoded[2]) > 25.0
    assert np.std(decoded[2]) > 20.0
    assert np.mean(np.abs(decoded[2].astype(float) - decoded[3].astype(float))) < 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["runs"].pop(), "at least two"),
        (
            lambda value: value["runs"][1].update({"ordinal": 3}),
            "ordinals must be consecutive",
        ),
        (
            lambda value: value.update({"run_order": list(reversed(value["run_order"]))}),
            "run_order does not match",
        ),
        (
            lambda value: (
                value["runs"][1].update({"run_id": value["runs"][0]["run_id"]}),
                value.update({"run_order": [value["runs"][0]["run_id"]] * 2}),
            ),
            "duplicate run_id",
        ),
        (
            lambda value: value["runs"][1].update(
                {
                    "video_path": value["runs"][0]["video_path"],
                    "video_sha256": value["runs"][0]["video_sha256"],
                }
            ),
            "duplicate run video paths",
        ),
    ],
)
def test_encoder_rejects_count_order_and_duplicate_contracts(
    tmp_path, mutation, message
):
    from fluidmix.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    value = json.loads(request.read_text(encoding="utf-8"))
    mutation(value)
    # Keep the top-level count coherent in the minimum-count case so that the
    # specific two-input gate is what fails.
    if len(value["runs"]) == 1:
        value["run_order"] = value["run_order"][:1]
    request.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        encode_palette_cut_reel(
            request,
            tmp_path / "reel",
            owner_thread_id="owner",
            repo_root=ROOT,
        )
    assert not (tmp_path / "reel").exists()


def test_encoder_rejects_missing_and_hash_drifted_inputs(tmp_path):
    from fluidmix.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    first.write_bytes(first.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        encode_palette_cut_reel(
            request, tmp_path / "drift", owner_thread_id="owner", repo_root=ROOT
        )

    request = _request(
        tmp_path / "missing-request.json",
        [("RUN-01", second), ("RUN-02", _video(tmp_path / "run-03.mp4", (5, 5, 5)))],
    )
    (tmp_path / "run-03.mp4").unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        encode_palette_cut_reel(
            request, tmp_path / "missing", owner_thread_id="owner", repo_root=ROOT
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"size": (300, 180)},
        {"rate": 4},
        {"pixel_format": "yuv444p"},
    ],
)
def test_encoder_rejects_geometry_fps_or_profile_mismatch(tmp_path, changed):
    from fluidmix.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120), **changed)
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    with pytest.raises(ValueError, match="identical geometry, fps, and codec profile"):
        encode_palette_cut_reel(
            request, tmp_path / "reel", owner_thread_id="owner", repo_root=ROOT
        )
    assert not (tmp_path / "reel").exists()


def test_encoder_is_no_clobber_and_cleans_failed_encode(tmp_path, monkeypatch):
    import fluidmix.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    output = tmp_path / "reel"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        reel.encode_palette_cut_reel(
            request, output, owner_thread_id="owner", repo_root=ROOT
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    def fail_encode(*_args, **_kwargs):
        raise OSError("synthetic encoder failure")

    monkeypatch.setattr(reel, "_encode", fail_encode)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="encoding failed"):
        reel.encode_palette_cut_reel(
            request, failed, owner_thread_id="owner", repo_root=ROOT
        )
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.partial-*"))


def test_cli_help_is_import_safe_and_exposes_manifest_owner_and_output():
    script = ROOT / "scripts/encode_palette_cut_reel.py"
    completed = subprocess.run(
        [sys.executable, "-S", script, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout
    assert "--owner-thread-id" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_cli_encodes_the_request_boundary(tmp_path):
    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    output = tmp_path / "reel"
    completed = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/encode_palette_cut_reel.py",
            "--request",
            request,
            "--owner-thread-id",
            "owner",
            "--output-dir",
            output,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == os.fspath(output / "palette-cut-reel.json")
    assert (output / "palette-cut-reel.mp4").is_file()

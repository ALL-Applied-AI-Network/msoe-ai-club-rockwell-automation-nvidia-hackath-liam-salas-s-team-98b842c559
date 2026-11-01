"""CPU-only contracts for palette-boundary review reels."""

from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import av
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _authoritative_case_root(tmp_path: Path) -> Path:
    token = hashlib.sha256(os.fspath(tmp_path).encode()).hexdigest()[:16]
    return ROOT / "outputs/part-a/palette-cut-reels" / token


def _output(tmp_path: Path, name: str = "reel") -> Path:
    root = _authoritative_case_root(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    return root / name


@pytest.fixture(autouse=True)
def _cleanup_authoritative_test_output(tmp_path):
    yield
    owned = _authoritative_case_root(tmp_path)
    if owned.exists():
        shutil.rmtree(owned)


@pytest.fixture(autouse=True)
def _synthetic_clean_git_identity(monkeypatch):
    import twin.render.palette_cut_reel as reel

    original = reel._git_identity

    def clean_identity(root):
        head, tree, _dirty = original(root)
        return head, tree, []

    monkeypatch.setattr(reel, "_git_identity", clean_identity)


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
    pts: list[int] | None = None,
) -> Path:
    if pts is None:
        pts = list(range(frames))
    assert len(pts) == frames
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
            frame.pts = pts[index]
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_encoder_rejects_timestamp_gaps_even_when_inputs_match(tmp_path):
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    first = _video(
        tmp_path / "run-01.mp4", (160, 20, 20), frames=3, pts=[0, 1, 3]
    )
    second = _video(
        tmp_path / "run-02.mp4", (20, 20, 160), frames=3, pts=[0, 1, 3]
    )
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    with pytest.raises(ValueError, match="contiguous constant-frame-rate timestamps"):
        encode_palette_cut_reel(
            request, _output(tmp_path), owner_thread_id="owner", repo_root=ROOT,
        )


def _palette_identity(seed: int) -> tuple[dict[str, list[float]], str]:
    transmission = {
        "t100": [1.0, 1.0, 1.0],
        "t200": [round(0.2 + seed * 0.07, 3), 0.4, 0.3],
        "t400": [0.1, round(0.5 + seed * 0.05, 3), 0.8],
    }
    encoded = json.dumps(
        transmission, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return transmission, hashlib.sha256(encoded).hexdigest()


def _request(
    path: Path,
    videos: list[tuple[str, Path]],
    *,
    palette_seeds: list[int] | None = None,
) -> Path:
    from twin.postrender.run_sequence_dataset import INDEX_FIELDS

    if palette_seeds is None:
        palette_seeds = list(range(1, len(videos) + 1))
    assert len(palette_seeds) == len(videos)
    package_root = path.parent
    index_path = package_root / "videos/INDEX.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_rows = []
    run_receipts = []
    run_receipt_paths = []
    for (run_id, video), palette_seed in zip(videos, palette_seeds, strict=True):
        transmission, palette_sha256 = _palette_identity(palette_seed)
        relative_video = video.resolve().relative_to(package_root.resolve()).as_posix()
        run_receipt_path = video.with_suffix(".run.json")
        run_receipt = {
            "schema_version": 2,
            "artifact_type": "rendered_transport_run_sequence_v2",
            "status": "pass",
            "run_id": run_id,
            "case_ids": [f"{run_id}-V{ordinal:02d}" for ordinal in range(1, 4)],
            "output_video": os.fspath(video.resolve()),
            "output_video_sha256": _sha256(video),
            "output_frame_count": 2,
            "fps": 2.0,
            "resolution": [320, 180],
            "qualified_fill_receipts_changed": False,
            "camera_and_lighting_changed": False,
        }
        run_receipt_path.write_text(
            json.dumps(run_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_receipts.append(run_receipt)
        run_receipt_paths.append(run_receipt_path)
        for vial_ordinal in range(1, 4):
            row = {field: "" for field in INDEX_FIELDS}
            row.update(
                {
                    "run_id": run_id,
                    "vial_ordinal": str(vial_ordinal),
                    "case_id": f"{run_id}-V{vial_ordinal:02d}",
                    "palette_t100_rgb": json.dumps(transmission["t100"]),
                    "palette_t200_rgb": json.dumps(transmission["t200"]),
                    "palette_t400_rgb": json.dumps(transmission["t400"]),
                    "palette_sha256": palette_sha256,
                    "run_video_path": relative_video,
                }
            )
            index_rows.append(row)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(index_rows)

    inventory_paths = [
        *(video for _, video in videos),
        *run_receipt_paths,
        index_path,
    ]
    sums_path = package_root / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{_sha256(item)}  {item.resolve().relative_to(package_root.resolve()).as_posix()}\n"
            for item in sorted(inventory_paths)
        ),
        encoding="utf-8",
    )
    package_receipt_path = package_root / "package-receipt.json"
    package_receipt = {
        "schema_version": 1,
        "artifact_type": "synthetic_run_sequence_dataset_package_v1",
        "status": "pass",
        "cohort_version": "v4",
        "output_root": os.fspath(package_root.resolve()),
        "run_video_receipts": run_receipts,
        "sha256sums_path": os.fspath(sums_path.resolve()),
        "sha256sums_sha256": _sha256(sums_path),
        "hashed_payload_file_count": len(inventory_paths),
        "qualified_fill_receipts_changed": False,
        "transport_mode": "immutable_13_state_usd_overlay",
    }
    package_receipt_path.write_text(
        json.dumps(package_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
                "package_receipt_path": os.fspath(package_receipt_path),
                "package_receipt_sha256": _sha256(package_receipt_path),
                "run_receipt_path": os.fspath(run_receipt_path),
                "run_receipt_sha256": _sha256(run_receipt_path),
            }
            for ordinal, ((run_id, video), run_receipt_path) in enumerate(
                zip(videos, run_receipt_paths, strict=True), start=1
            )
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _rewrite_video(
    path: Path,
    transform,
    *,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
) -> None:
    with av.open(os.fspath(path), mode="r") as source:
        stream = source.streams.video[0]
        rate = Fraction(stream.average_rate)
        width, height = stream.width, stream.height
        frames = [frame.to_ndarray(format="rgb24") for frame in source.decode(stream)]
    frames = transform(frames)
    replacement = path.with_name("replacement.mp4")
    with av.open(os.fspath(replacement), mode="w", format="mp4") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pixel_format
        stream.codec_context.time_base = Fraction(rate.denominator, rate.numerator)
        stream.codec_context.max_b_frames = 0
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = stream.codec_context.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    replacement.replace(path)


def test_encoder_rejects_adjacent_runs_with_the_same_accepted_palette(tmp_path):
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (160, 20, 20))
    second = _video(tmp_path / "run-02.mp4", (20, 20, 160))
    request = _request(
        tmp_path / "request.json",
        [("RUN-TRAIN-01", first), ("RUN-TRAIN-02", second)],
        palette_seeds=[1, 1],
    )

    with pytest.raises(ValueError, match="adjacent accepted run palettes must differ"):
        encode_palette_cut_reel(
            request,
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )


def test_encoder_inserts_exact_visible_card_and_publishes_bound_inventory(tmp_path):
    from twin.render.palette_cut_reel import CUT_TEXT_LINES, encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (160, 20, 20))
    second = _video(tmp_path / "run-02.mp4", (20, 20, 160))
    request = _request(
        tmp_path / "request.json",
        [("RUN-TRAIN-01", first), ("RUN-TRAIN-02", second)],
    )
    output = _output(tmp_path)

    receipt_path = encode_palette_cut_reel(
        request,
        output,
        owner_thread_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        repo_root=ROOT,
    )

    assert receipt_path == output / "palette-cut-reel.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "encoded_verified"
    assert receipt["artifact_type"] == "fluidmix_palette_cut_reel_v2"
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
    assert receipt["output"]["profile"]["codec_name"] == "h264"
    assert receipt["output"]["profile"]["pixel_format"] == "yuv420p"
    assert receipt["verification"]["timing"]["duration_seconds"] == "4"
    assert receipt["verification"]["source_segments_in_declared_order"] is True
    assert receipt["verification"]["cut_cards_match_exact_rendered_text"] is True
    assert len(receipt["verification"]["source_segment_content_checks"]) == 2
    assert len(receipt["verification"]["cut_card_content_checks"]) == 1
    card_check = receipt["verification"]["cut_card_content_checks"][0]
    assert card_check["exact_canonical_decoded_frames_match"] is True
    assert card_check["comparison_tolerance_pixels"] == 0
    assert len(card_check["canonical_decoded_frame_rgb_sha256"]) == 4
    assert card_check["literal_identity_payload_sha256"]
    assert all(row["palette_sha256"] for row in receipt["ordered_inputs"])
    assert all(row["accepted_package"] for row in receipt["ordered_inputs"])
    assert receipt["source_provenance"]["commit"] == receipt["source_commit"]
    closure = {
        row["path"]: row["sha256"]
        for row in receipt["source_provenance"]["local_dependency_closure"]
    }
    assert closure == {
        relative: _sha256(ROOT / relative)
        for relative in (
            "cli/twin/render/encode_palette_cut_reel.py",
            "src/twin/render/palette_cut_reel.py",
            "src/twin/render/render_process.py",
            "src/twin/render/twin_video.py",
        )
    }
    assert receipt["part_a_artifact_boundary"]["classified_output_root"] == os.fspath(
        ROOT / "outputs/part-a/palette-cut-reels"
    )
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


def test_encoder_requires_immutable_passing_package_and_run_receipts(tmp_path):
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    value = json.loads(request.read_text(encoding="utf-8"))
    del value["runs"][0]["package_receipt_path"]
    request.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="accepted package receipt path"):
        encode_palette_cut_reel(
            request,
            _output(tmp_path, "missing-package"),
            owner_thread_id="owner",
            repo_root=ROOT,
        )

    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    package_path = tmp_path / "package-receipt.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["status"] = "candidate"
    package_path.write_text(json.dumps(package, sort_keys=True) + "\n", encoding="utf-8")
    value = json.loads(request.read_text(encoding="utf-8"))
    for row in value["runs"]:
        row["package_receipt_sha256"] = _sha256(package_path)
    request.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a passing V3/V4 run package"):
        encode_palette_cut_reel(
            request,
            _output(tmp_path, "unaccepted-package"),
            owner_thread_id="owner",
            repo_root=ROOT,
        )


@pytest.mark.parametrize("corruption", ["cut_card", "source_order", "codec"])
def test_output_verification_rejects_wrong_content_order_or_profile(
    tmp_path, monkeypatch, corruption
):
    import twin.render.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (180, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 180))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    original = reel._encode

    def corrupt_encode(*args, **kwargs):
        cuts, record = original(*args, **kwargs)
        output = args[2]
        if corruption == "cut_card":
            start, stop = cuts[0]["start_frame_index"], cuts[0]["stop_frame_index"]

            def transform(frames):
                for index in range(start, stop):
                    frames[index] = np.zeros_like(frames[index])
                return frames

            _rewrite_video(output, transform)
        elif corruption == "source_order":
            first_count = int(args[1][0]["frame_count"])
            second_count = int(args[1][1]["frame_count"])

            def transform(frames):
                left = frames[:first_count]
                right = frames[-second_count:]
                return [*right, *frames[first_count:-second_count], *left]

            _rewrite_video(output, transform)
        else:
            _rewrite_video(output, lambda frames: frames, codec="mpeg4")
        return cuts, record

    monkeypatch.setattr(reel, "_encode", corrupt_encode)
    with pytest.raises(RuntimeError, match="H.264|content|order"):
        reel.encode_palette_cut_reel(
            request,
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )
    assert not _output(tmp_path).exists()


@pytest.mark.parametrize(
    "altered_lines",
    [
        (
            "Transport does not cross this palette boundary.",
            "NEW RUN — TANK COLOURS CHANGE",
        ),
        (
            "NEW RUN — TANK COLOUR CHANGE",
            "Transport does not cross this palette boundary.",
        ),
        (
            "NEW RUN — TANK COLOURS CHANCE",
            "Transport does not cross this palette boundary.",
        ),
        (
            "NEW RUN — TANK COLOURS CHANGE",
            "Transport crosses this palette boundary.",
        ),
        (
            "NEW RUN — TANK COLOURS CHANGE",
            "Transport does not cross this paIette boundary.",
        ),
        (
            "NEW RUN — TANK COLOURSS CHANGE",
            "Transport does not cross this palette boundary.",
        ),
        (
            "NEW RUN — TANK COLOURS CHANGE",
            "This palette boundary does not cross transport.",
        ),
    ],
    ids=[
        "line-inversion",
        "one-glyph-delete",
        "one-glyph-substitute",
        "word-delete",
        "confusable-lowercase-l-uppercase-I",
        "one-glyph-add",
        "transport-sentence-inversion",
    ],
)
def test_decoded_card_semantics_reject_line_glyph_or_word_changes(
    tmp_path, monkeypatch, altered_lines
):
    import twin.render.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (180, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 180))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    original = reel._encode

    def altered_encode(*args, **kwargs):
        cuts, record = original(*args, **kwargs)
        output = args[2]
        card = reel._render_cut_card(320, 180, altered_lines)
        start, stop = cuts[0]["start_frame_index"], cuts[0]["stop_frame_index"]

        def transform(frames):
            frames[start:stop] = [card.copy() for _ in range(stop - start)]
            return frames

        _rewrite_video(output, transform)
        return cuts, record

    monkeypatch.setattr(reel, "_encode", altered_encode)
    with pytest.raises(RuntimeError, match="glyph|text|card"):
        reel.encode_palette_cut_reel(
            request,
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )


def test_source_order_verification_abstains_for_swapped_similar_runs(
    tmp_path, monkeypatch
):
    import twin.render.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (100, 100, 100))
    second = _video(tmp_path / "run-02.mp4", (110, 100, 100))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    original = reel._encode

    def swapped_encode(*args, **kwargs):
        cuts, record = original(*args, **kwargs)
        output = args[2]
        first_count = int(args[1][0]["frame_count"])
        second_count = int(args[1][1]["frame_count"])

        def transform(frames):
            return [
                *frames[-second_count:],
                *frames[first_count:-second_count],
                *frames[:first_count],
            ]

        _rewrite_video(output, transform)
        return cuts, record

    monkeypatch.setattr(reel, "_encode", swapped_encode)
    with pytest.raises(RuntimeError, match="order|separable|identity"):
        reel.encode_palette_cut_reel(
            request,
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )


def test_encoder_rejects_output_outside_classified_part_a_root(tmp_path):
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    with pytest.raises(ValueError, match="classified Part A output root"):
        encode_palette_cut_reel(
            request,
            tmp_path / "outside",
            owner_thread_id="owner",
            repo_root=ROOT,
        )


def test_encoder_rejects_output_inside_bound_accepted_package_without_mutation(tmp_path):
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    package = _authoritative_case_root(tmp_path)
    package.mkdir(parents=True, exist_ok=True)
    first = _video(package / "run-01.mp4", (120, 10, 10))
    second = _video(package / "run-02.mp4", (10, 10, 120))
    request = _request(
        package / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    before = {
        path.relative_to(package).as_posix(): _sha256(path)
        for path in package.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="outside every bound accepted package root"):
        encode_palette_cut_reel(
            request,
            package / "reel-inside-accepted-package",
            owner_thread_id="owner",
            repo_root=ROOT,
        )
    after = {
        path.relative_to(package).as_posix(): _sha256(path)
        for path in package.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("codec_name", "profile", "pixel_format", "missing_field"),
    [
        ("", "High", "yuv420p", "codec_name"),
        ("h264", None, "yuv420p", "codec_profile"),
        ("h264", "High", "unknown", "pixel_format"),
    ],
)
def test_unknown_codec_profile_fields_fail_closed(
    codec_name, profile, pixel_format, missing_field
):
    from twin.render.palette_cut_reel import _known_media_profile

    class Values:
        name = codec_name

    class Context:
        codec = Values()
        pix_fmt = pixel_format

    class Stream:
        codec_context = Context()

    Stream.codec_context.profile = profile
    with pytest.raises(ValueError, match=f"missing or unknown {missing_field}"):
        _known_media_profile(Stream(), "RUN-01")


def test_encoder_rejects_source_provenance_drift(tmp_path, monkeypatch):
    import twin.render.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    original = reel._source_provenance
    calls = 0

    def drifting(root):
        nonlocal calls
        calls += 1
        value = original(root)
        if calls == 2:
            value["local_dependency_closure"][0]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(reel, "_source_provenance", drifting)
    with pytest.raises(RuntimeError, match="source provenance changed"):
        reel.encode_palette_cut_reel(
            request,
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )
    assert calls == 2
    assert not _output(tmp_path).exists()


def test_source_provenance_rejects_dirty_local_dependency(monkeypatch):
    import twin.render.palette_cut_reel as reel

    monkeypatch.setattr(
        reel,
        "_git_identity",
        lambda _root: (
            "1" * 40,
            "2" * 40,
            [" M src/twin/render/render_process.py"],
        ),
    )
    with pytest.raises(ValueError, match="requires a clean Git worktree"):
        reel._source_provenance(ROOT)


def test_encoder_rejects_accepted_package_drift_during_encode(tmp_path, monkeypatch):
    import twin.render.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    package = tmp_path / "package-receipt.json"
    original = reel._verify_output

    def drift_after_verify(*args, **kwargs):
        result = original(*args, **kwargs)
        package.write_bytes(package.read_bytes() + b" ")
        return result

    monkeypatch.setattr(reel, "_verify_output", drift_after_verify)
    with pytest.raises(RuntimeError, match="accepted package/run evidence changed"):
        reel.encode_palette_cut_reel(
            request,
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )
    assert not _output(tmp_path).exists()


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
    from twin.render.palette_cut_reel import encode_palette_cut_reel

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
            _output(tmp_path),
            owner_thread_id="owner",
            repo_root=ROOT,
        )
    assert not _output(tmp_path).exists()


def test_encoder_rejects_missing_and_hash_drifted_inputs(tmp_path):
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    first.write_bytes(first.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        encode_palette_cut_reel(
            request, _output(tmp_path, "drift"), owner_thread_id="owner", repo_root=ROOT,
        )

    request = _request(
        tmp_path / "missing-request.json",
        [("RUN-01", second), ("RUN-02", _video(tmp_path / "run-03.mp4", (5, 5, 5)))],
    )
    (tmp_path / "run-03.mp4").unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        encode_palette_cut_reel(
            request, _output(tmp_path, "missing"), owner_thread_id="owner", repo_root=ROOT,
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
    from twin.render.palette_cut_reel import encode_palette_cut_reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120), **changed)
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    with pytest.raises(ValueError, match="identical geometry, fps, and codec profile"):
        encode_palette_cut_reel(
            request, _output(tmp_path), owner_thread_id="owner", repo_root=ROOT,
        )
    assert not _output(tmp_path).exists()


def test_encoder_is_no_clobber_and_cleans_failed_encode(tmp_path, monkeypatch):
    import twin.render.palette_cut_reel as reel

    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    output = _output(tmp_path)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        reel.encode_palette_cut_reel(
            request, output, owner_thread_id="owner", repo_root=ROOT,
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    def fail_encode(*_args, **_kwargs):
        raise OSError("synthetic encoder failure")

    monkeypatch.setattr(reel, "_encode", fail_encode)
    failed = _output(tmp_path, "failed")
    with pytest.raises(RuntimeError, match="encoding failed"):
        reel.encode_palette_cut_reel(
            request, failed, owner_thread_id="owner", repo_root=ROOT,
        )
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.partial-*"))


def test_cli_help_is_import_safe_and_exposes_manifest_owner_and_output():
    script = ROOT / "cli/twin/render/encode_palette_cut_reel.py"
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
    assert "--part-a-output-root" not in completed.stdout


def test_cli_encodes_the_request_boundary(tmp_path):
    if subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        pytest.skip("CLI execution correctly requires a clean worktree")
    first = _video(tmp_path / "run-01.mp4", (120, 10, 10))
    second = _video(tmp_path / "run-02.mp4", (10, 10, 120))
    request = _request(
        tmp_path / "request.json", [("RUN-01", first), ("RUN-02", second)]
    )
    output = _output(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            ROOT / "cli/twin/render/encode_palette_cut_reel.py",
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

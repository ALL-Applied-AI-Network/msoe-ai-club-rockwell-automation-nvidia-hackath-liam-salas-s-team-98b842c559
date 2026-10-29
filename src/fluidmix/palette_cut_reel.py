"""Build review reels separated by explicit per-run palette cut cards.

This encoder never synthesizes transport. It decodes already accepted run
videos and places a visible, fixed-duration card between each pair of runs.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fluidmix.render_process import hash_regular_file
from fluidmix.twin_video import _atomic_rename_noreplace, _write_private_json


CUT_TEXT_LINES = (
    "NEW RUN — TANK COLOURS CHANGE",
    "Transport does not cross this palette boundary.",
)
CUT_DURATION_SECONDS = 2
_VIDEO_NAME = "palette-cut-reel.mp4"
_RECEIPT_NAME = "palette-cut-reel.json"
_INVENTORY_NAME = "SHA256SUMS"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _load_av() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError("PyAV is required to encode a palette-cut reel") from error
    return av


def _regular_file(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.lstat(resolved)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return resolved


def _read_request(path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    request_path = _regular_file(path, "reel request")
    request_snapshot = hash_regular_file(request_path)
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("reel request is not readable JSON") from error
    if not isinstance(value, dict):
        raise ValueError("reel request must contain one JSON object")
    return request_path, value, request_snapshot


def _validate_request(
    request: Mapping[str, object], request_root: Path
) -> list[dict[str, object]]:
    if (
        request.get("schema_version") != 1
        or request.get("artifact_type")
        != "fluidmix_palette_cut_reel_request_v1"
    ):
        raise ValueError("reel request schema/artifact type is invalid")
    rows = request.get("runs")
    run_order = request.get("run_order")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("reel request requires at least two ordered run inputs")
    if not isinstance(run_order, list) or len(run_order) != len(rows):
        raise ValueError("run_order must list every run exactly once")

    validated: list[dict[str, object]] = []
    run_ids: list[str] = []
    resolved_paths: list[Path] = []
    hashes: list[str] = []
    for expected_ordinal, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("every run input must be an object")
        if row.get("ordinal") != expected_ordinal:
            raise ValueError("run input ordinals must be consecutive and match list order")
        run_id = row.get("run_id")
        raw_path = row.get("video_path")
        expected_hash = row.get("video_sha256")
        if (
            not isinstance(run_id, str)
            or not run_id
            or any(character.isspace() for character in run_id)
        ):
            raise ValueError("run_id must be a non-empty token")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("video_path must be a non-empty string")
        if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
            raise ValueError("video_sha256 must be an exact lowercase SHA-256")
        supplied = Path(raw_path)
        if not supplied.is_absolute():
            supplied = request_root / supplied
        resolved = _regular_file(supplied, f"run video {run_id}")
        snapshot = hash_regular_file(resolved)
        if snapshot["sha256"] != expected_hash:
            raise ValueError(f"run video SHA-256 drifted for {run_id}")
        run_ids.append(run_id)
        resolved_paths.append(resolved)
        hashes.append(expected_hash)
        validated.append(
            {
                "ordinal": expected_ordinal,
                "run_id": run_id,
                "video_path": os.fspath(resolved),
                "video_sha256": expected_hash,
                "video_bytes": snapshot["bytes"],
            }
        )

    if run_order != run_ids:
        raise ValueError("run_order does not match the ordered run inputs")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_id inputs are not allowed")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate run video paths are not allowed")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate run video content hashes are not allowed")
    return validated


def _rate(stream: Any, run_id: str) -> Fraction:
    raw = stream.average_rate
    if raw is None:
        raise ValueError(f"run video has no declared average frame rate: {run_id}")
    rate = Fraction(raw)
    if rate <= 0:
        raise ValueError(f"run video has a non-positive frame rate: {run_id}")
    if (rate * CUT_DURATION_SECONDS).denominator != 1:
        raise ValueError(
            f"run video fps cannot represent the fixed two-second cut card: {run_id}"
        )
    return rate


def _probe(av: Any, record: Mapping[str, object]) -> dict[str, object]:
    path = str(record["video_path"])
    run_id = str(record["run_id"])
    try:
        with av.open(path, mode="r") as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"run video must have exactly one video stream: {run_id}")
            if len(container.streams.audio) != 0:
                raise ValueError(f"run video must be silent: {run_id}")
            stream = container.streams.video[0]
            rate = _rate(stream, run_id)
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            profile = {
                "codec_name": stream.codec_context.codec.name,
                "codec_profile": stream.codec_context.profile,
                "pixel_format": stream.codec_context.pix_fmt,
            }
            count = 0
            for frame in container.decode(stream):
                if frame.width != width or frame.height != height:
                    raise ValueError(f"run video geometry changes while decoding: {run_id}")
                frame.to_ndarray(format="rgb24")
                count += 1
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"run video is not fully readable: {run_id}") from error
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"run video geometry must be positive and even: {run_id}")
    if count <= 0:
        raise ValueError(f"run video contains no decodable frames: {run_id}")
    return {
        "fps": str(rate),
        "width": width,
        "height": height,
        "profile": profile,
        "frame_count": count,
    }


def _font_that_fits(draw: ImageDraw.ImageDraw, text: str, width: int, maximum: int):
    for size in range(maximum, 9, -1):
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= int(width * 0.9):
            return font
    raise RuntimeError("video geometry is too small for the required cut-card text")


def _cut_card(width: int, height: int) -> np.ndarray:
    image = Image.new("RGB", (width, height), color=(8, 18, 34))
    draw = ImageDraw.Draw(image)
    maximum = max(10, min(height // 5, width // 9))
    fonts = [
        _font_that_fits(draw, text, width, maximum if index == 0 else maximum - 3)
        for index, text in enumerate(CUT_TEXT_LINES)
    ]
    bounds = [
        draw.textbbox((0, 0), text, font=font)
        for text, font in zip(CUT_TEXT_LINES, fonts, strict=True)
    ]
    heights = [bottom - top for _, top, _, bottom in bounds]
    gap = max(8, height // 14)
    total = sum(heights) + gap
    y = (height - total) // 2
    stripe = max(4, height // 22)
    draw.rectangle((0, 0, width - 1, stripe), fill=(255, 174, 0))
    draw.rectangle((0, height - stripe, width - 1, height - 1), fill=(255, 174, 0))
    for index, (text, font, bound, line_height) in enumerate(
        zip(CUT_TEXT_LINES, fonts, bounds, heights, strict=True)
    ):
        line_width = bound[2] - bound[0]
        color = (255, 204, 64) if index == 0 else (245, 248, 252)
        draw.text(((width - line_width) // 2, y - bound[1]), text, font=font, fill=color)
        y += line_height + gap
    return np.asarray(image, dtype=np.uint8)


def _source_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("repo_root does not identify a readable Git worktree") from error
    value = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Git HEAD is not a full lowercase commit hash")
    return value


def _encode(
    av: Any,
    records: list[dict[str, object]],
    output: Path,
    *,
    rate: Fraction,
    width: int,
    height: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    card_pixels = _cut_card(width, height)
    cut_frame_count = int(rate * CUT_DURATION_SECONDS)
    cuts: list[dict[str, object]] = []
    output_index = 0
    with av.open(
        os.fspath(output), mode="w", format="mp4", options={"movflags": "+faststart"}
    ) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(rate.denominator, rate.numerator)
        stream.codec_context.time_base = stream.time_base
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {"crf": "12", "preset": "medium"}

        def write_frame(pixels: np.ndarray) -> None:
            nonlocal output_index
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = output_index
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
            output_index += 1

        for run_index, record in enumerate(records):
            with av.open(str(record["video_path"]), mode="r") as source:
                source_stream = source.streams.video[0]
                for decoded in source.decode(source_stream):
                    write_frame(decoded.to_ndarray(format="rgb24"))
            if run_index < len(records) - 1:
                start = output_index
                for _ in range(cut_frame_count):
                    write_frame(card_pixels)
                cuts.append(
                    {
                        "boundary_ordinal": run_index + 1,
                        "after_run_id": records[run_index]["run_id"],
                        "before_run_id": records[run_index + 1]["run_id"],
                        "start_frame_index": start,
                        "stop_frame_index": output_index,
                        "frame_count": cut_frame_count,
                        "text_lines": list(CUT_TEXT_LINES),
                        "transport_crosses_boundary": False,
                    }
                )
        for packet in stream.encode():
            container.mux(packet)
        encoded_profile = {
            "codec_name": stream.codec_context.codec.name,
            "codec_profile": stream.codec_context.profile,
            "pixel_format": stream.codec_context.pix_fmt,
        }
    return cuts, {"frame_count": output_index, "profile": encoded_profile}


def _verify_output(
    av: Any, path: Path, *, expected_count: int, rate: Fraction, width: int, height: int
) -> dict[str, object]:
    try:
        with av.open(os.fspath(path), mode="r") as container:
            if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
                raise RuntimeError("encoded reel must contain one silent video stream")
            stream = container.streams.video[0]
            if Fraction(stream.average_rate) != rate:
                raise RuntimeError("encoded reel frame rate differs from its inputs")
            count = 0
            for frame in container.decode(stream):
                if frame.width != width or frame.height != height:
                    raise RuntimeError("encoded reel geometry differs from its inputs")
                count += 1
            profile = {
                "codec_name": stream.codec_context.codec.name,
                "codec_profile": stream.codec_context.profile,
                "pixel_format": stream.codec_context.pix_fmt,
            }
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("encoded reel could not be fully decoded") from error
    if count != expected_count:
        raise RuntimeError(f"encoded reel decoded {count} frames, expected {expected_count}")
    return {"decoded_frame_count": count, "profile": profile}


def _write_inventory(path: Path, rows: list[tuple[str, Path]]) -> None:
    content = "".join(
        f"{hash_regular_file(file)['sha256']}  {name}\n" for name, file in rows
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        data = content.encode("utf-8")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def encode_palette_cut_reel(
    request_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    owner_thread_id: str,
    repo_root: str | os.PathLike[str],
) -> Path:
    """Encode and atomically publish one hash-bound palette-change review reel.

    The request asserts that each source video was already accepted upstream.
    This function checks the declared file identities and media compatibility;
    it does not re-run or weaken upstream transport/acceptance gates.
    """

    if (
        not isinstance(owner_thread_id, str)
        or not owner_thread_id.strip()
        or any(ord(character) < 32 for character in owner_thread_id)
    ):
        raise ValueError("owner_thread_id must be a non-empty printable string")
    request_file, request, request_before = _read_request(Path(request_path))
    records = _validate_request(request, request_file.parent)
    source_commit = _source_commit(Path(repo_root).expanduser().resolve(strict=True))

    candidate = Path(output_dir).expanduser()
    parent = candidate.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("reel output parent must be a directory")
    name = candidate.name
    if not name or name in {".", ".."}:
        raise ValueError("reel output must name one directory")
    output = parent / name
    if os.path.lexists(output):
        raise FileExistsError(output)
    input_paths = [Path(str(row["video_path"])) for row in records]
    if any(path.is_relative_to(output) for path in input_paths):
        raise ValueError("reel output must be outside every source-video tree")

    av = _load_av()
    probes = [_probe(av, row) for row in records]
    first = probes[0]
    compatibility = (first["fps"], first["width"], first["height"], first["profile"])
    if any(
        (probe["fps"], probe["width"], probe["height"], probe["profile"])
        != compatibility
        for probe in probes[1:]
    ):
        raise ValueError("run videos must have identical geometry, fps, and codec profile")
    for record, probe in zip(records, probes, strict=True):
        record.update(probe)

    rate = Fraction(str(first["fps"]))
    width = int(first["width"])
    height = int(first["height"])
    expected_count = sum(int(row["frame_count"]) for row in records) + (
        len(records) - 1
    ) * int(rate * CUT_DURATION_SECONDS)

    partial: Path | None = Path(tempfile.mkdtemp(prefix=f".{name}.partial-", dir=parent))
    os.chmod(partial, 0o700)
    try:
        video_path = partial / _VIDEO_NAME
        try:
            cuts, encoding = _encode(
                av, records, video_path, rate=rate, width=width, height=height
            )
        except Exception as error:
            raise RuntimeError("palette-cut reel encoding failed") from error
        if not video_path.exists() or video_path.is_symlink():
            raise RuntimeError("palette-cut reel encoder did not create a regular MP4")
        os.chmod(video_path, 0o600)
        video_snapshot = hash_regular_file(video_path)
        if video_snapshot["bytes"] <= 0:
            raise RuntimeError("palette-cut reel MP4 is empty")
        verification = _verify_output(
            av,
            video_path,
            expected_count=expected_count,
            rate=rate,
            width=width,
            height=height,
        )

        if hash_regular_file(request_file) != request_before:
            raise RuntimeError("reel request changed while encoding")
        for row in records:
            snapshot = hash_regular_file(Path(str(row["video_path"])))
            if snapshot["sha256"] != row["video_sha256"] or snapshot["bytes"] != row["video_bytes"]:
                raise RuntimeError(f"run video changed while encoding: {row['run_id']}")

        ordered_inputs = [
            {
                "ordinal": row["ordinal"],
                "run_id": row["run_id"],
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "video_bytes": row["video_bytes"],
                "frame_count": row["frame_count"],
                "fps": row["fps"],
                "geometry": {"width": row["width"], "height": row["height"]},
                "profile": row["profile"],
            }
            for row in records
        ]
        receipt = {
            "schema_version": 1,
            "artifact_type": "fluidmix_palette_cut_reel_v1",
            "status": "encoded_verified",
            "owner_thread_id": owner_thread_id,
            "source_commit": source_commit,
            "request": {
                "path": os.fspath(request_file),
                "sha256": request_before["sha256"],
                "bytes": request_before["bytes"],
            },
            "ordered_inputs": ordered_inputs,
            "input_profile": first["profile"],
            "cut_card_duration_seconds": CUT_DURATION_SECONDS,
            "cut_cards": cuts,
            "output": {
                "path": _VIDEO_NAME,
                "sha256": video_snapshot["sha256"],
                "bytes": video_snapshot["bytes"],
                "frame_count": encoding["frame_count"],
                "fps": str(rate),
                "geometry": {"width": width, "height": height},
                "profile": verification["profile"],
            },
            "verification": verification,
            "sha_inventory": {
                "path": _INVENTORY_NAME,
                "entries": [_VIDEO_NAME, _RECEIPT_NAME],
            },
            "transport_boundary_policy": {
                "transport_frames_added_by_reel": False,
                "transport_crosses_palette_boundaries": False,
                "boundary_mechanism": "fixed-duration visible cut card only",
            },
            "limits": {
                "video_role": "lossy review reel only",
                "upstream_acceptance": "asserted by request; not re-evaluated by this encoder",
                "physical_claim": "none",
                "audio": "source videos must be silent; output is silent",
            },
        }
        receipt_path = partial / _RECEIPT_NAME
        _write_private_json(receipt_path, receipt)
        _write_inventory(
            partial / _INVENTORY_NAME,
            [(_VIDEO_NAME, video_path), (_RECEIPT_NAME, receipt_path)],
        )
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
        return output / _RECEIPT_NAME
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == parent:
                shutil.rmtree(partial)

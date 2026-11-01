"""Encode receipt-bound PhysX PNG frames as a review-only MP4.

The renderer's PNGs remain the quantitative evidence.  This module runs after
Isaac exits so a codec missing from the Isaac environment cannot silently turn
the review artifact into an unreadable AVI.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from twin.render.render_process import hash_regular_file
from twin.render.twin_video import _atomic_rename_noreplace, _write_private_json


_CODEC = "libx264"
_PIXEL_FORMAT = "yuv420p"
_MAX_MEAN_ABSOLUTE_ERROR = 8.0


def _load_av() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError(
            "PyAV is required to encode the verified review MP4"
        ) from error
    return av


def _hash_regular_png(path: Path) -> dict[str, object]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"source frame is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return {"sha256": digest.hexdigest(), "bytes": metadata.st_size}


def _read_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"source frame is not a PNG: {path}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (height, width, 3):
        raise ValueError(
            f"source frame has raster {rgb.shape}, expected {(height, width, 3)}: {path}"
        )
    return rgb


def _validated_source_frames(
    receipt: dict[str, object], receipt_root: Path
) -> tuple[list[dict[str, object]], int, int]:
    render = receipt.get("render")
    if not isinstance(render, dict):
        raise ValueError("receipt.render must be an object")
    resolution = render.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(value) is not int or value <= 0 for value in resolution)
    ):
        raise ValueError("receipt render resolution must be two positive integers")
    width, height = resolution
    count = render.get("frame_count")
    paths = render.get("frame_paths")
    hashes = render.get("frame_sha256")
    steps = render.get("frame_physics_steps")
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(paths, list)
        or not isinstance(hashes, list)
        or not isinstance(steps, list)
        or len(paths) != count
        or len(hashes) != count
        or len(steps) != count
    ):
        raise ValueError("receipt frame arrays must match its positive frame_count")
    if any(type(step) is not int or step < 0 for step in steps) or any(
        later <= earlier for earlier, later in zip(steps, steps[1:])
    ):
        raise ValueError("receipt frame physics steps must be strictly increasing integers")

    records: list[dict[str, object]] = []
    for ordinal, (relative, expected_hash, step) in enumerate(
        zip(paths, hashes, steps), start=1
    ):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("receipt frame paths must be non-empty relative paths")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("receipt frame hashes must be lowercase SHA-256 values")
        candidate = receipt_root / relative
        if candidate.is_symlink():
            raise ValueError(f"source frame is a symlink: {relative}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(receipt_root):
            raise ValueError(f"source frame escapes the receipt directory: {relative}")
        snapshot = _hash_regular_png(resolved)
        if snapshot["sha256"] != expected_hash:
            raise ValueError(f"source frame hash does not match receipt: {relative}")
        _read_rgb(resolved, width=width, height=height)
        records.append(
            {
                "ordinal": ordinal,
                "physics_step": step,
                "path": relative,
                "sha256": expected_hash,
                "bytes": snapshot["bytes"],
                "resolved_path": resolved,
            }
        )
    return records, width, height


def _encode(
    av: Any,
    frames: list[dict[str, object]],
    path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> dict[str, object]:
    with av.open(
        os.fspath(path),
        mode="w",
        format="mp4",
        options={"movflags": "+faststart"},
    ) as container:
        stream = container.add_stream(_CODEC, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = _PIXEL_FORMAT
        stream.time_base = Fraction(1, fps)
        stream.codec_context.time_base = Fraction(1, fps)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.gop_size = 1
        stream.codec_context.options = {"crf": "12", "preset": "medium"}
        for index, record in enumerate(frames):
            rgb = _read_rgb(record["resolved_path"], width=width, height=height)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        codec_name = stream.codec_context.codec.name
        codec_long_name = stream.codec_context.codec.long_name
    return {"codec_name": codec_name, "codec_long_name": codec_long_name}


def _verify(
    av: Any,
    path: Path,
    frames: list[dict[str, object]],
    *,
    fps: int,
    width: int,
    height: int,
) -> dict[str, object]:
    decoded_pts: list[int] = []
    decoded_errors: list[float] = []
    with av.open(os.fspath(path), mode="r") as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise RuntimeError(f"encoded MP4 has {len(streams)} video streams, expected one")
        stream = streams[0]
        reported_rate = stream.average_rate
        if reported_rate != Fraction(fps, 1):
            raise RuntimeError(
                f"encoded MP4 reports frame rate {reported_rate}, expected {fps}"
            )
        for index, decoded in enumerate(container.decode(stream)):
            if index >= len(frames):
                raise RuntimeError("encoded MP4 contains more frames than its source receipt")
            if decoded.width != width or decoded.height != height:
                raise RuntimeError("decoded MP4 frame has the wrong raster size")
            if decoded.pts is None or decoded.time_base is None:
                raise RuntimeError("decoded MP4 frame is missing its presentation timestamp")
            normalized = Fraction(decoded.pts) * decoded.time_base * fps
            if normalized.denominator != 1:
                raise RuntimeError("decoded MP4 presentation timestamp is not frame-aligned")
            decoded_pts.append(normalized.numerator)
            source = _read_rgb(
                frames[index]["resolved_path"], width=width, height=height
            ).astype(np.int16)
            actual = decoded.to_ndarray(format="rgb24").astype(np.int16)
            decoded_errors.append(float(np.mean(np.abs(actual - source))))
        codec_name = stream.codec_context.codec.name
        time_base = str(stream.time_base)
    if len(decoded_pts) != len(frames):
        raise RuntimeError(
            f"encoded MP4 decoded {len(decoded_pts)} frames, expected {len(frames)}"
        )
    expected_pts = list(range(len(frames)))
    if decoded_pts != expected_pts:
        raise RuntimeError(
            f"encoded MP4 presentation order {decoded_pts} does not match {expected_pts}"
        )
    maximum_error = max(decoded_errors)
    if not np.isfinite(maximum_error) or maximum_error > _MAX_MEAN_ABSOLUTE_ERROR:
        raise RuntimeError(
            "encoded MP4 does not preserve source-frame appearance within the frozen "
            f"mean-absolute-error gate: {maximum_error:.6f} > {_MAX_MEAN_ABSOLUTE_ERROR:.6f}"
        )
    return {
        "decoded_frame_count": len(decoded_pts),
        "decoded_pts": decoded_pts,
        "decoded_width": width,
        "decoded_height": height,
        "reported_average_rate": str(Fraction(fps, 1)),
        "reported_time_base": time_base,
        "reported_codec_name": codec_name,
        "per_frame_mean_absolute_error": decoded_errors,
        "max_mean_absolute_error": maximum_error,
        "max_mean_absolute_error_gate": _MAX_MEAN_ABSOLUTE_ERROR,
    }


def encode_physx_receipt_video(
    receipt_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    fps: int = 6,
) -> Path:
    """Encode and verify the ordered PNG frames declared by one PhysX receipt.

    The function verifies the receipt hashes both before and after encoding,
    decodes the MP4, checks exact presentation order, and publishes the output
    directory without replacing an existing path.
    """

    if type(fps) is not int or fps <= 0:
        raise ValueError("fps must be a positive integer")

    receipt_input = Path(receipt_path).expanduser().absolute()
    if receipt_input.is_symlink():
        raise ValueError("receipt must be a regular non-symlink file")
    receipt_candidate = receipt_input.resolve(strict=True)
    if not receipt_candidate.is_file():
        raise ValueError("receipt must be a regular non-symlink file")
    receipt_root = receipt_candidate.parent.resolve(strict=True)
    receipt_before = hash_regular_file(receipt_candidate)
    try:
        receipt = json.loads(receipt_candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("receipt is not readable JSON") from error
    if not isinstance(receipt, dict):
        raise ValueError("receipt root must be an object")
    frames, width, height = _validated_source_frames(receipt, receipt_root)

    output_candidate = Path(output_dir).expanduser()
    output_parent = output_candidate.parent.resolve(strict=True)
    if not output_parent.is_dir():
        raise ValueError("video output parent must be a directory")
    output_name = output_candidate.name
    if not output_name or output_name in {".", ".."}:
        raise ValueError("video output must name one directory")
    output = output_parent / output_name
    if os.path.lexists(output):
        raise FileExistsError(output)
    if output.is_relative_to(receipt_root) or receipt_root.is_relative_to(output):
        raise ValueError("video output must be outside the receipt tree")

    partial: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_name}.partial-", dir=output_parent)
    )
    os.chmod(partial, 0o700)
    try:
        av = _load_av()
        video_path = partial / "sequence.mp4"
        codec = _encode(
            av,
            frames,
            video_path,
            fps=fps,
            width=width,
            height=height,
        )
        if not video_path.exists() or video_path.is_symlink():
            raise RuntimeError("encoder did not create a regular MP4")
        os.chmod(video_path, 0o600)
        video_snapshot = hash_regular_file(video_path)
        if video_snapshot["bytes"] <= 0:
            raise RuntimeError("encoded MP4 is empty")
        verification = _verify(
            av,
            video_path,
            frames,
            fps=fps,
            width=width,
            height=height,
        )

        receipt_after = hash_regular_file(receipt_candidate)
        if receipt_after != receipt_before:
            raise RuntimeError("source receipt changed while video was encoded")
        frames_after, after_width, after_height = _validated_source_frames(
            receipt, receipt_root
        )
        if (
            after_width != width
            or after_height != height
            or [
                (row["path"], row["sha256"], row["physics_step"])
                for row in frames_after
            ]
            != [
                (row["path"], row["sha256"], row["physics_step"])
                for row in frames
            ]
        ):
            raise RuntimeError("source frames changed while video was encoded")

        source_records = [
            {
                "ordinal": row["ordinal"],
                "physics_step": row["physics_step"],
                "path": row["path"],
                "sha256": row["sha256"],
            }
            for row in frames
        ]
        source_order_sha256 = hashlib.sha256(
            json.dumps(source_records, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record = {
            "schema_version": 1,
            "status": "encoded_verified",
            "source_receipt": {
                "path": str(receipt_candidate),
                **receipt_before,
            },
            "encoding": {
                "codec_requested": _CODEC,
                "container": "mp4",
                "fps": fps,
                "width": width,
                "height": height,
                "pixel_format": _PIXEL_FORMAT,
            },
            "runtime": {
                "pyav_version": av.__version__,
                **codec,
            },
            "source_frames": source_records,
            "source_order_sha256": source_order_sha256,
            "video": {"path": "sequence.mp4", **video_snapshot},
            "verification": verification,
            "limits": {
                "measurement_source": "hash-bound PNGs only",
                "video_role": "lossy review visualization only",
                "physical_claim": "none; the encoder does not validate simulation physics",
                "external_player_compatibility": (
                    "H.264/yuv420p in MP4 is decoded here with PyAV; no separate player was tested"
                ),
            },
        }
        _write_private_json(partial / "video.json", record)
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
        return output / "video.json"
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == output_parent:
                shutil.rmtree(partial)

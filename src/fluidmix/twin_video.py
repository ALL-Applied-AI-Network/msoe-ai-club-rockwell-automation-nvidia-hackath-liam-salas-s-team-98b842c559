"""Encode a verified synthetic FillCam capture as review-only video.

The MP4 is deliberately not measurement evidence. Quantitative colour work
continues to use the capture's hash-bound RGB PNGs and liquid masks.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from fluidmix.render_process import hash_regular_file, verify_capture_summary


VIDEO_TIME_CODES = tuple(sorted(set(range(31, 687, 5)) | {342}))
REQUIRED_STAGE_CODES = (171, 342, 686)
VIDEO_FPS = 6.0
VIDEO_SIZE = (640, 480)
VIDEO_CODEC = "mp4v"
_ACCEPTED_REPORTED_FOURCC = {"FMP4", "mp4v", "MP4V"}


def _time_codes_sha256() -> str:
    encoded = json.dumps(VIDEO_TIME_CODES, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "atomic renameat2(RENAME_NOREPLACE) is unavailable",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(number, os.strerror(number), os.fspath(destination))
    raise OSError(number, os.strerror(number), os.fspath(destination))


def _write_private_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_fourcc(value: float) -> str:
    integer = int(round(value))
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).rstrip(
        "\x00"
    )


def _load_cv2() -> Any:
    import cv2

    return cv2


def _verify_encoded_video(cv2: Any, path: Path) -> dict[str, object]:
    reader = cv2.VideoCapture(os.fspath(path))
    if not reader.isOpened():
        reader.release()
        raise RuntimeError("encoded MP4 could not be opened")
    try:
        backend = reader.getBackendName()
        reported_fourcc = _decode_fourcc(reader.get(cv2.CAP_PROP_FOURCC))
        fps = float(reader.get(cv2.CAP_PROP_FPS))
        width = int(round(reader.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        decoded = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if frame is None or tuple(frame.shape) != (VIDEO_SIZE[1], VIDEO_SIZE[0], 3):
                raise RuntimeError("decoded video frame has the wrong raster shape")
            decoded += 1
    finally:
        reader.release()
    if decoded != len(VIDEO_TIME_CODES):
        raise RuntimeError(
            f"encoded MP4 decoded {decoded} frames, expected {len(VIDEO_TIME_CODES)}"
        )
    if (width, height) != VIDEO_SIZE:
        raise RuntimeError("encoded MP4 reports the wrong raster size")
    if abs(fps - VIDEO_FPS) > 0.01:
        raise RuntimeError("encoded MP4 reports the wrong frame rate")
    if reported_fourcc not in _ACCEPTED_REPORTED_FOURCC:
        raise RuntimeError(f"encoded MP4 reports unexpected codec {reported_fourcc!r}")
    return {
        "decoded_frame_count": decoded,
        "reported_fourcc": reported_fourcc,
        "reported_fps": fps,
        "reported_width": width,
        "reported_height": height,
        "reader_backend": backend,
    }


def encode_capture_video(
    capture_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> Path:
    """Encode one exact frozen 133-frame capture and publish it without clobbering.

    This checks hashes, order, stage inclusion, codec output, and input
    immutability. It cannot establish visual quality, physical timing,
    carryover realism, or sim-to-real validity.
    """

    capture_before = verify_capture_summary(capture_root, None)
    capture = Path(capture_before["summary_path"]).parent
    summary = capture_before["summary"]
    render = summary["render_settings"]
    time_codes = tuple(render["time_codes"])
    if time_codes != VIDEO_TIME_CODES:
        raise ValueError("capture does not use the frozen 133-frame video schedule")
    if render["resolution"] != [VIDEO_SIZE[0], VIDEO_SIZE[1]]:
        raise ValueError("capture does not use the frozen 640x480 raster")

    parent_candidate = Path(output_dir).expanduser().parent
    parent = parent_candidate.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("video output parent must be a directory")
    name = Path(output_dir).name
    if not name or name in {".", ".."}:
        raise ValueError("video output must name one directory")
    output = parent / name
    if os.path.lexists(output):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), output)
    if output.is_relative_to(capture) or capture.is_relative_to(output):
        raise ValueError("video output must be outside the capture tree")

    source_frames = []
    for ordinal, frame in enumerate(summary["frames"], start=1):
        rgb = frame["artifacts"]["rgb"]
        source_frames.append(
            {
                "capture_ordinal": ordinal,
                "time_code": frame["time_code"],
                "rgb_path": rgb["path"],
                "rgb_sha256": rgb["sha256"],
            }
        )
    positions = {row["time_code"]: row for row in source_frames}
    stage_frames = [
        {
            "capture_ordinal": positions[code]["capture_ordinal"],
            "time_code": code,
            "rgb_sha256": positions[code]["rgb_sha256"],
        }
        for code in REQUIRED_STAGE_CODES
    ]
    if [row["capture_ordinal"] for row in stage_frames] != [29, 64, 133]:
        raise ValueError("required stage frames are not at the frozen capture ordinals")

    partial = Path(tempfile.mkdtemp(prefix=f".{name}.partial-", dir=parent))
    os.chmod(partial, 0o700)
    writer = None
    try:
        cv2 = _load_cv2()
        video_path = partial / "sequence.mp4"
        writer = cv2.VideoWriter(
            os.fspath(video_path),
            cv2.VideoWriter_fourcc(*VIDEO_CODEC),
            VIDEO_FPS,
            VIDEO_SIZE,
            True,
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v VideoWriter could not be opened")
        writer_backend = writer.getBackendName()
        for row in source_frames:
            frame_path = capture / row["rgb_path"]
            image = cv2.imread(os.fspath(frame_path), cv2.IMREAD_COLOR)
            if image is None or tuple(image.shape) != (
                VIDEO_SIZE[1],
                VIDEO_SIZE[0],
                3,
            ):
                raise ValueError(
                    f"capture RGB has the wrong raster shape: {row['rgb_path']}"
                )
            writer.write(image)
        writer.release()
        writer = None
        if not video_path.exists() or video_path.is_symlink():
            raise RuntimeError("OpenCV did not create a regular MP4")
        os.chmod(video_path, 0o600)
        video_snapshot = hash_regular_file(video_path)
        if video_snapshot["bytes"] <= 0:
            raise RuntimeError("encoded MP4 is empty")
        verification = _verify_encoded_video(cv2, video_path)

        capture_after = verify_capture_summary(capture, None)
        if capture_after != capture_before:
            raise RuntimeError("capture changed while video was encoded")

        record = {
            "schema_version": 1,
            "status": "encoded_verified",
            "capture": {
                "root": str(capture),
                "summary_sha256": capture_before["summary_sha256"],
            },
            "encoding": {
                "codec_requested": VIDEO_CODEC,
                "container": "mp4",
                "fps": VIDEO_FPS,
                "width": VIDEO_SIZE[0],
                "height": VIDEO_SIZE[1],
            },
            "runtime": {
                "opencv_version": cv2.__version__,
                "writer_backend": writer_backend,
            },
            "time_codes": list(VIDEO_TIME_CODES),
            "time_codes_sha256": _time_codes_sha256(),
            "source_frames": source_frames,
            "stage_frames": stage_frames,
            "video": {"path": "sequence.mp4", **video_snapshot},
            "verification": verification,
            "limits": {
                "measurement_source": "hash-bound capture PNGs only",
                "video_role": "lossy review visualization only",
                "timing": (
                    "fixed 6 fps; inserted time code 342 makes source-time spacing "
                    "nonuniform at 341->342->346"
                ),
                "physical_claim": "none; synthetic appearance is uncalibrated",
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
        if writer is not None:
            writer.release()
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == parent:
                shutil.rmtree(partial)

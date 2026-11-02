"""Extract every settled calibration frame for fill-level ruler fitting.

This script deliberately makes no appearance-based choice.  It copies every frame in the
inclusive, committed windows for vials 1 and 13 into a new output directory and records enough
provenance to reproduce and verify the extraction.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import cv2


EXPECTED_IMAGE_SIZE = (640, 480)
EXPECTED_SOURCE_ROLE = "real_validation"
CALIBRATION_SPECS = {
    1: {"level_ml": 30.0, "recipe_id": "5"},
    13: {"level_ml": 80.0, "recipe_id": "9"},
}
REQUIRED_WINDOW_FIELDS = {
    "video_id",
    "vial",
    "recipe_id",
    "window_start_s",
    "window_end_s",
    "start_frame",
    "end_frame",
    "n_frames",
    "role",
}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_int(row, name, vial):
    value = str(row.get(name, "")).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"vial {vial} has invalid {name}: {value!r}") from exc
    return parsed


def _required_float(row, name, vial):
    value = str(row.get(name, "")).strip()
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"vial {vial} has invalid {name}: {value!r}") from exc
    if not parsed >= 0.0:
        raise ValueError(f"vial {vial} has negative {name}: {parsed}")
    return parsed


def load_calibration_windows(path):
    """Return validated vial-1/vial-13 windows in calibration order."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_WINDOW_FIELDS - fields)
        if missing:
            raise ValueError(f"window CSV is missing fields: {', '.join(missing)}")
        selected = {}
        for row_number, row in enumerate(reader, start=2):
            raw_vial = str(row.get("vial", "")).strip()
            try:
                vial = int(raw_vial)
            except ValueError:
                continue
            if vial not in CALIBRATION_SPECS:
                continue
            if vial in selected:
                raise ValueError(f"window CSV has more than one row for vial {vial}")
            spec = CALIBRATION_SPECS[vial]
            role = str(row.get("role", "")).strip()
            if role != EXPECTED_SOURCE_ROLE:
                raise ValueError(
                    f"vial {vial} source role must be {EXPECTED_SOURCE_ROLE!r}, got {role!r}"
                )
            recipe_id = str(row.get("recipe_id", "")).strip()
            if recipe_id != spec["recipe_id"]:
                raise ValueError(
                    f"vial {vial} recipe must be {spec['recipe_id']!r} for "
                    f"{spec['level_ml']:g} mL, got {recipe_id!r}"
                )
            video_id = str(row.get("video_id", "")).strip()
            if video_id != "5":
                raise ValueError(f"vial {vial} must come from video 5, got {video_id!r}")
            start_frame = _required_int(row, "start_frame", vial)
            end_frame = _required_int(row, "end_frame", vial)
            n_frames = _required_int(row, "n_frames", vial)
            if start_frame < 0 or end_frame < start_frame:
                raise ValueError(
                    f"vial {vial} has invalid inclusive frame window "
                    f"{start_frame}:{end_frame}"
                )
            expected_count = end_frame - start_frame + 1
            if n_frames != expected_count:
                raise ValueError(
                    f"vial {vial} n_frames is {n_frames}, expected {expected_count} "
                    "from the inclusive frame window"
                )
            window_start_s = _required_float(row, "window_start_s", vial)
            window_end_s = _required_float(row, "window_end_s", vial)
            if window_end_s < window_start_s:
                raise ValueError(f"vial {vial} has a reversed time window")
            selected[vial] = {
                "csv_row": row_number,
                "video_id": video_id,
                "vial": vial,
                "recipe_id": recipe_id,
                "level_ml": float(spec["level_ml"]),
                "role": "calibration",
                "source_role": role,
                "window_start_s": window_start_s,
                "window_end_s": window_end_s,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_frames": n_frames,
            }
    missing_vials = sorted(set(CALIBRATION_SPECS) - set(selected))
    if missing_vials:
        raise ValueError(
            "window CSV is missing calibration vial(s): "
            + ", ".join(str(value) for value in missing_vials)
        )
    return [selected[vial] for vial in sorted(CALIBRATION_SPECS)]


def normalize_expected_sha256(value):
    if value is None:
        return None
    value = str(value).strip()
    if not SHA256_RE.fullmatch(value):
        raise ValueError("expected video SHA-256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _capture_number(capture, property_id):
    try:
        return float(capture.get(property_id))
    except (TypeError, ValueError):
        return 0.0


def extract_calibration_frames(
    video_path,
    windows_csv,
    output_dir,
    *,
    expected_video_sha256=None,
    capture_factory=cv2.VideoCapture,
    image_writer=cv2.imwrite,
    generated_utc=None,
):
    """Extract all inclusive window frames and return the written index object."""

    video_path = Path(video_path)
    windows_csv = Path(windows_csv)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    if not windows_csv.is_file():
        raise FileNotFoundError(f"window CSV does not exist: {windows_csv}")

    windows = load_calibration_windows(windows_csv)
    expected_hash = normalize_expected_sha256(expected_video_sha256)
    video_hash = sha256_file(video_path)
    if expected_hash is not None and video_hash != expected_hash:
        raise ValueError(
            f"video SHA-256 mismatch: expected {expected_hash}, measured {video_hash}"
        )
    windows_hash = sha256_file(windows_csv)

    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    known_frame_count = int(round(_capture_number(capture, cv2.CAP_PROP_FRAME_COUNT)))
    source_fps = _capture_number(capture, cv2.CAP_PROP_FPS)
    last_requested = max(window["end_frame"] for window in windows)
    if known_frame_count > 0 and last_requested >= known_frame_count:
        capture.release()
        raise ValueError(
            f"video has {known_frame_count} frames, but window requests frame {last_requested}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for window in windows:
            vial_dir = output_dir / (
                f"vial-{window['vial']:02d}-{int(window['level_ml']):03d}ml"
            )
            vial_dir.mkdir()
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, window["start_frame"]):
                raise RuntimeError(
                    f"video backend refused seek to frame {window['start_frame']}"
                )
            position = _capture_number(capture, cv2.CAP_PROP_POS_FRAMES)
            if abs(position - window["start_frame"]) > 0.5:
                raise RuntimeError(
                    f"video backend sought to {position:g}, expected {window['start_frame']}"
                )
            for frame_index in range(window["start_frame"], window["end_frame"] + 1):
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"failed to decode video frame {frame_index}")
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(
                        f"frame {frame_index} must be a three-channel image, got {image.shape}"
                    )
                height, width = (int(image.shape[0]), int(image.shape[1]))
                if (width, height) != EXPECTED_IMAGE_SIZE:
                    raise ValueError(
                        f"frame {frame_index} is {width}x{height}; expected "
                        f"{EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}"
                    )
                next_position = _capture_number(capture, cv2.CAP_PROP_POS_FRAMES)
                if next_position > 0 and abs(next_position - (frame_index + 1)) > 0.5:
                    raise RuntimeError(
                        f"decoder position after frame {frame_index} is {next_position:g}"
                    )
                image_path = vial_dir / f"frame-{frame_index:06d}.png"
                if image_path.exists():
                    raise FileExistsError(f"refusing to overwrite frame: {image_path}")
                if not image_writer(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"failed to write lossless PNG: {image_path}")
                records.append(
                    {
                        "vial": window["vial"],
                        "recipe_id": window["recipe_id"],
                        "level_ml": window["level_ml"],
                        "role": window["role"],
                        "source_window_role": window["source_role"],
                        "frame_index": frame_index,
                        "path": image_path.relative_to(output_dir).as_posix(),
                        "sha256": sha256_file(image_path),
                        "width": width,
                        "height": height,
                    }
                )
    finally:
        capture.release()

    expected_total = sum(window["n_frames"] for window in windows)
    if len(records) != expected_total:
        raise RuntimeError(f"extracted {len(records)} frames; expected {expected_total}")
    timestamp = generated_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = {
        "schema_version": 1,
        "generated_utc": timestamp,
        "selection_policy": (
            "all frames in each committed inclusive window; no appearance-based selection"
        ),
        "source": {
            "video": {
                "path": str(video_path.resolve()),
                "sha256": video_hash,
                "expected_sha256": expected_hash,
                "opencv_frame_count": known_frame_count if known_frame_count > 0 else None,
                "opencv_fps": source_fps if source_fps > 0 else None,
            },
            "windows_csv": {
                "path": str(windows_csv.resolve()),
                "sha256": windows_hash,
            },
        },
        "image_contract": {
            "encoding": "PNG",
            "lossless": True,
            "width": EXPECTED_IMAGE_SIZE[0],
            "height": EXPECTED_IMAGE_SIZE[1],
        },
        "windows": windows,
        "frame_count": len(records),
        "frames": records,
    }
    index_path = output_dir / "index.json"
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite index: {index_path}")
    index_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="source vial-fill video")
    parser.add_argument("windows_csv", type=Path, help="committed real_check_windows.csv")
    parser.add_argument("output_dir", type=Path, help="new ignored output directory")
    parser.add_argument(
        "--expected-video-sha256",
        help="optional exact source-video SHA-256 gate",
    )
    args = parser.parse_args(argv)
    result = extract_calibration_frames(
        args.video,
        args.windows_csv,
        args.output_dir,
        expected_video_sha256=args.expected_video_sha256,
    )
    print(
        f"wrote {result['frame_count']} lossless calibration frames and index.json to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()

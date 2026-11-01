"""Build hash-bound photometric frame manifests from organizer still indexes."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...measurement.real_photometric_range import ALLOWED_VIDEO_IDS, _config, _sha256_path


MOTION_METRIC = "mean_absolute_rgb_difference_divided_by_255_in_configured_crop"


def _read_config(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("stable-region config is not valid JSON") from error
    return _config(value)


def _index_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"filename", "timestamp", "video_id"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError("image index must have filename, timestamp, and video_id columns")
    result: list[dict[str, object]] = []
    seen_filenames: set[str] = set()
    for row in rows:
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("image index filename must be a basename")
        if filename in seen_filenames:
            raise ValueError(f"image index filename is duplicated: {filename}")
        seen_filenames.add(filename)
        try:
            video_id = int(row["video_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"image index {filename} has invalid video_id") from error
        if video_id == 5:
            raise ValueError("video 5 is held out and cannot be calibration input")
        if video_id not in ALLOWED_VIDEO_IDS:
            raise ValueError(f"image index {filename} must be assigned to videos 1-4")
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError(f"image index {filename} needs timestamp")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError(f"image index {filename} has invalid timestamp") from error
        result.append({"filename": filename, "timestamp": timestamp, "parsed_timestamp": parsed_timestamp,
                       "video_id": video_id})
    return sorted(result, key=lambda row: (int(row["video_id"]), row["parsed_timestamp"], str(row["filename"])))


def _crop(path: Path, crop_xyxy: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_xyxy
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    if pixels.shape[0] < y1 or pixels.shape[1] < x1:
        raise ValueError("crop_outside_image")
    return pixels[y0:y1, x0:x1].astype(np.int16)


def build_photometric_frame_manifest(
    image_index_path: str | Path,
    images_root: str | Path,
    stable_region_config_path: str | Path,
) -> dict[str, object]:
    """Build an external videos-1--4 manifest without liquid/event inference.

    The index must itself be the reviewed videos-1--4 subset. Crop-only motion is
    compared with the prior timestamp-ordered crop in the same video. The first frame
    of each video is retained with a named unavailable state rather than fabricated.
    """
    index_path = Path(image_index_path).resolve(strict=True)
    root = Path(images_root).resolve(strict=True)
    config_path = Path(stable_region_config_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("images_root must be a directory")
    config = _read_config(config_path)
    rows = _index_rows(index_path)
    frames: list[dict[str, object]] = []
    previous_crop: dict[int, np.ndarray | None] = {video_id: None for video_id in ALLOWED_VIDEO_IDS}
    previous_frame: dict[int, str | None] = {video_id: None for video_id in ALLOWED_VIDEO_IDS}
    for row in rows:
        filename = str(row["filename"])
        video_id = int(row["video_id"])
        image_path = (root / filename).resolve(strict=True)
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"image index filename escapes images_root: {filename}") from error
        item: dict[str, object] = {
            "frame_id": filename,
            "video_id": video_id,
            "timestamp": row["timestamp"],
            "image_path": str(image_path),
            "source_sha256": _sha256_path(image_path),
            "motion_reference_frame_id": previous_frame[video_id],
        }
        try:
            current_crop = _crop(image_path, config["crop_xyxy"])
        except (OSError, ValueError) as error:
            item["motion_score"] = None
            item["motion_status"] = str(error) if str(error) else "unreadable_image"
            previous_crop[video_id] = None
        else:
            if previous_crop[video_id] is None:
                item["motion_score"] = None
                item["motion_status"] = "no_previous_frame_in_video" if previous_frame[video_id] is None else "previous_crop_unavailable"
            else:
                difference = np.mean(np.abs(current_crop - previous_crop[video_id])) / 255.0
                item["motion_score"] = round(float(difference), 6)
                item["motion_status"] = "measured"
            previous_crop[video_id] = current_crop
        previous_frame[video_id] = filename
        frames.append(item)
    return {
        "schema_version": 1,
        "manifest_type": "real_photometric_frame_manifest",
        "allowed_video_ids": list(ALLOWED_VIDEO_IDS),
        "source_image_index_sha256": _sha256_path(index_path),
        "stable_region_config_sha256": _sha256_path(config_path),
        "stable_region_config": config,
        "motion_metric": MOTION_METRIC,
        "frames": frames,
        "limits": {
            "selection": "all supplied indexed stills in videos 1-4, timestamp ordered",
            "inference": "no liquid polygon, liquid label, event label, or hidden-label inference",
        },
    }

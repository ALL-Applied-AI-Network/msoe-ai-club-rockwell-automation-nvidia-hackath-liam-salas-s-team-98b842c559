"""Measure gated real appearance for renderer design, never model training.

The validation contracts complete before the first image decode.  The video-5 gate
pins the closed vial-1/vial-13 calibration windows.  A separate videos-1--4 gate pins
the organizer indexes and a bounded paired-camera candidate allowlist while rejecting
video 5 before any external path is constructed.  All real images remain excluded
from model training.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Callable

import cv2
import numpy as np


EXPECTED_VIDEO_SHA256 = (
    "6f87cec1aeec3b2bae6dac639a317c077a3f175f27abd5fc8430d6953d9d05b9"
)
EXPECTED_WINDOWS_SHA256 = (
    "19174a4b706fe896b177ba71c8ba2bdf850ceb9cf3a40522dcdaa42d53904229"
)
EXPECTED_IMAGE_WIDTH = 640
EXPECTED_IMAGE_HEIGHT = 480
EXPECTED_WINDOWS = {
    1: {
        "count": 28,
        "first_frame": 816,
        "last_frame": 843,
        "level_ml": 30.0,
        "recipe_id": "5",
        "path_prefix": "vial-01-030ml",
        "liquid_roi_xyxy": [298, 354, 314, 420],
        "appearance_description": "dark green",
    },
    13: {
        "count": 70,
        "first_frame": 19700,
        "last_frame": 19769,
        "level_ml": 80.0,
        "recipe_id": "9",
        "path_prefix": "vial-13-080ml",
        "liquid_roi_xyxy": [300, 276, 314, 408],
        "appearance_description": "near-black olive",
    },
}
LABEL_POLYGON_XY = [
    [286, 171],
    [338, 168],
    [340, 226],
    [334, 234],
    [288, 239],
    [285, 177],
]
LABEL_BBOX_XYXY = [285, 168, 341, 240]
SCENE_ROWS_HALF_OPEN = [0, 440]
BACKDROP_ROI_XYXY = [20, 80, 240, 250]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Videos 1--4 are unlabeled appearance references, never evaluation data.  The
# hashes pin the organizer copy used for this measurement.  Video 5 rows in the
# indexes are deliberately never selected, resolved, hashed, or decoded here.
EXPECTED_EXTERNAL_CSV_SHA256 = {
    "videos.csv": "8f7857d9bc507bcb959e9482b8d6acc42489cfeda0afffc8a76393e3c50b7723",
    "vial_fill/image_index.csv": "f531c17b9ddb24d15e6277f5b85d3e651aa71117ca162add2bd7b1e2102f069c",
    "tank_level/image_index.csv": "2bdbe4176ec19640af2caf4cae73b8c2397e4b78d684c46bb9d399841c3fc53a",
}
EXPECTED_EXTERNAL_CANDIDATE_ALLOWLIST_SHA256 = {
    "vial_fill": "e4f1a1f50fda8fd7decd21dcf660218215485e20f5f18797a7f052bd30842dc0",
    "tank_level": "4295b9357aece1c39cc09db245b57c9688973f957874352f886cc4873c2a5892",
}
EXTERNAL_VIDEO_VIAL_COUNTS = {1: 19, 2: 3, 3: 6, 4: 6}
EXTERNAL_VIDEO_INDEX_COUNTS = {1: 558, 2: 78, 3: 95, 4: 157}
EXTERNAL_CANDIDATE_FRACTIONS = (0.80, 0.95)
EXTERNAL_FILENAME_RE = re.compile(
    r"^(vial_fill|tank_level)_(\d{8}_\d{6}_\d{6})\.jpg$"
)
EXTERNAL_LIQUID_ROI_XYXY = [298, 300, 316, 410]
EXTERNAL_STREAM_ROI_XYXY = [304, 70, 318, 215]
EXTERNAL_LABEL_SEARCH_ROI_XYXY = [270, 130, 345, 270]
EXTERNAL_SETTLED_STREAM_FRACTION_MAX = 0.03
EXTERNAL_TANK_ROIS_XYXY = {
    "tank_1_pale_green": [188, 330, 205, 385],
    "tank_2_pink_red": [296, 330, 315, 385],
    "tank_3_pale_neutral": [405, 330, 423, 385],
    "tank_4_blue": [514, 330, 532, 385],
}
CALIBRATION_SURFACE_ROIS_XYXY = {
    1: [292, 329, 330, 346],
    13: [292, 177, 330, 201],
}


@dataclass(frozen=True)
class ValidatedFrame:
    """One frame whose metadata, path and bytes passed the closed input gate."""

    vial: int
    frame_index: int
    relative_path: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ValidatedCalibrationSet:
    """Closed calibration set returned only after all 98 hashes pass."""

    root: Path
    index_sha256: str
    source_video_sha256: str
    windows_csv_sha256: str
    frames: tuple[ValidatedFrame, ...]


@dataclass(frozen=True)
class ExternalCandidateFrame:
    """One allowlisted videos-1--4 candidate, paired across both cameras."""

    video_id: int
    nominal_cycle: int
    fraction: float
    timestamp: str
    vial_filename: str
    vial_path: Path
    vial_sha256: str
    tank_filename: str
    tank_path: Path
    tank_sha256: str


@dataclass(frozen=True)
class ValidatedExternalCandidates:
    """All bounded candidates, hashed before the first external decode."""

    root: Path
    csv_sha256: dict[str, str]
    vial_allowlist_sha256: str
    tank_allowlist_sha256: str
    candidates: tuple[ExternalCandidateFrame, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise ValueError(f"{description} must be a 64-character SHA-256")
    return value.lower()


def _read_index(index_path: Path) -> tuple[dict, str]:
    try:
        raw = index_path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read calibration index {index_path}: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"calibration index is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("calibration index must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_index_metadata(index: dict) -> list[dict]:
    frames = index.get("frames")
    if index.get("frame_count") != 98 or not isinstance(frames, list) or len(frames) != 98:
        raise ValueError("calibration index must contain exactly 98 frame records")
    contract = index.get("image_contract")
    if not isinstance(contract, dict) or contract.get("width") != EXPECTED_IMAGE_WIDTH:
        raise ValueError("calibration image contract must specify width 640")
    if contract.get("height") != EXPECTED_IMAGE_HEIGHT:
        raise ValueError("calibration image contract must specify height 480")
    if contract.get("encoding") != "PNG" or contract.get("lossless") is not True:
        raise ValueError("calibration image contract must be lossless PNG")

    source = index.get("source")
    video = source.get("video") if isinstance(source, dict) else None
    windows_csv = source.get("windows_csv") if isinstance(source, dict) else None
    if not isinstance(video, dict) or not isinstance(windows_csv, dict):
        raise ValueError("calibration index must retain video and windows provenance")
    video_hash = _require_sha256(video.get("sha256"), "source video hash")
    expected_video_hash = _require_sha256(
        video.get("expected_sha256"), "expected source video hash"
    )
    if video_hash != EXPECTED_VIDEO_SHA256 or expected_video_hash != EXPECTED_VIDEO_SHA256:
        raise ValueError("calibration index source video SHA-256 is not approved video 5")
    windows_hash = _require_sha256(windows_csv.get("sha256"), "windows CSV hash")
    if windows_hash != EXPECTED_WINDOWS_SHA256:
        raise ValueError("calibration index windows CSV SHA-256 is not approved")

    counts: Counter[int] = Counter()
    observed_indices: dict[int, list[int]] = {1: [], 13: []}
    observed_paths: set[str] = set()
    for position, record in enumerate(frames):
        if not isinstance(record, dict):
            raise ValueError(f"frame record {position} must be a JSON object")
        vial = record.get("vial")
        if vial not in EXPECTED_WINDOWS:
            raise ValueError(f"frame record {position} uses forbidden vial {vial!r}")
        expected = EXPECTED_WINDOWS[vial]
        if record.get("role") != "calibration":
            raise ValueError(f"frame record {position} role must be calibration")
        if record.get("source_window_role") != "real_validation":
            raise ValueError(
                f"frame record {position} source_window_role must be real_validation"
            )
        if record.get("width") != EXPECTED_IMAGE_WIDTH or record.get("height") != EXPECTED_IMAGE_HEIGHT:
            raise ValueError(f"frame record {position} must declare dimensions 640x480")
        if record.get("level_ml") != expected["level_ml"]:
            raise ValueError(f"frame record {position} level does not match vial {vial}")
        if str(record.get("recipe_id")) != expected["recipe_id"]:
            raise ValueError(f"frame record {position} recipe does not match vial {vial}")
        frame_index = record.get("frame_index")
        if not isinstance(frame_index, int):
            raise ValueError(f"frame record {position} frame_index must be an integer")
        relative_text = record.get("path")
        if not isinstance(relative_text, str) or "\\" in relative_text:
            raise ValueError(f"frame record {position} path must be a POSIX relative path")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise ValueError(f"frame record {position} path traversal is forbidden")
        expected_path = (
            f"{expected['path_prefix']}/frame-{frame_index:06d}.png"
        )
        if relative.as_posix() != expected_path:
            raise ValueError(
                f"frame record {position} path must use {expected['path_prefix']} prefix"
            )
        if relative_text in observed_paths:
            raise ValueError(f"duplicate calibration frame path: {relative_text}")
        _require_sha256(record.get("sha256"), f"frame record {position} image hash")
        observed_paths.add(relative_text)
        observed_indices[vial].append(frame_index)
        counts[vial] += 1

    if counts != Counter({1: 28, 13: 70}):
        raise ValueError(f"calibration vial counts must be 28/70, observed {dict(counts)}")
    for vial, expected in EXPECTED_WINDOWS.items():
        required = list(range(expected["first_frame"], expected["last_frame"] + 1))
        if sorted(observed_indices[vial]) != required:
            raise ValueError(f"vial {vial} frame indices must be the complete approved window")
    return frames


def validate_calibration_root(root: Path) -> ValidatedCalibrationSet:
    """Validate the complete closed set, including hashes, without decoding an image."""

    root = Path(root)
    index, index_sha256 = _read_index(root / "index.json")
    records = _validate_index_metadata(index)
    root_resolved = root.resolve(strict=True)
    validated: list[ValidatedFrame] = []
    for position, record in enumerate(records):
        relative = PurePosixPath(record["path"])
        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"frame record {position} does not resolve inside calibration root"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"frame record {position} is not a regular file")
        expected_hash = record["sha256"].lower()
        actual_hash = file_sha256(resolved)
        if actual_hash != expected_hash:
            raise ValueError(
                f"frame record {position} SHA-256 mismatch for {record['path']}"
            )
        validated.append(
            ValidatedFrame(
                vial=record["vial"],
                frame_index=record["frame_index"],
                relative_path=record["path"],
                path=resolved,
                sha256=actual_hash,
            )
        )
    source = index["source"]
    return ValidatedCalibrationSet(
        root=root_resolved,
        index_sha256=index_sha256,
        source_video_sha256=source["video"]["sha256"].lower(),
        windows_csv_sha256=source["windows_csv"]["sha256"].lower(),
        frames=tuple(validated),
    )


def _read_csv_with_hash(path: Path) -> tuple[list[dict[str, str]], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read external CSV {path}: {error}") from error
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
    except UnicodeDecodeError as error:
        raise ValueError(f"external CSV is not valid UTF-8: {path}") from error
    if reader.fieldnames is None:
        raise ValueError(f"external CSV has no header: {path}")
    return rows, hashlib.sha256(raw).hexdigest()


def _external_selected_position(row_count: int, vial_count: int, cycle: int, fraction: float) -> int:
    start = cycle * row_count // vial_count
    stop = (cycle + 1) * row_count // vial_count
    if stop <= start:
        raise ValueError("external nominal cycle is empty")
    return min(start + int(fraction * (stop - start)), stop - 1)


def _validate_external_index_rows(
    rows: list[dict[str, str]], camera: str
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {
        video_id: [] for video_id in EXTERNAL_VIDEO_VIAL_COUNTS
    }
    expected_keys = {"filename", "timestamp", "video_id"}
    for position, row in enumerate(rows):
        if set(row) != expected_keys:
            raise ValueError(f"{camera} index row {position} has an unexpected schema")
        try:
            video_id = int(row["video_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{camera} index row {position} has invalid video_id") from error
        # Rows for video 5 are metadata in the pinned organizer index, but are not
        # part of this allowlist and are intentionally not inspected further.
        if video_id not in grouped:
            continue
        filename = row["filename"]
        match = EXTERNAL_FILENAME_RE.fullmatch(filename or "")
        if not match or match.group(1) != camera:
            raise ValueError(f"{camera} index row {position} has an invalid filename")
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError(f"{camera} index row {position} path traversal is forbidden")
        if not row["timestamp"]:
            raise ValueError(f"{camera} index row {position} has no timestamp")
        grouped[video_id].append(row)
    for video_id, expected_count in EXTERNAL_VIDEO_INDEX_COUNTS.items():
        observed = grouped[video_id]
        if len(observed) != expected_count:
            raise ValueError(
                f"{camera} video {video_id} must have {expected_count} index rows"
            )
        timestamps = [row["timestamp"] for row in observed]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError(f"{camera} video {video_id} timestamps must be unique and sorted")
    return grouped


def _resolve_external_image(root: Path, camera: str, filename: str) -> Path:
    root_resolved = root.resolve(strict=True)
    image_root = (root / camera / "images").resolve(strict=True)
    try:
        image_root.relative_to(root_resolved)
        candidate = (image_root / filename).resolve(strict=True)
        candidate.relative_to(image_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"external {camera} image does not resolve inside its root") from error
    if not candidate.is_file():
        raise ValueError(f"external {camera} candidate is not a regular file")
    return candidate


def _allowlist_digest(frames: list[tuple[str, str]]) -> str:
    text = "".join(f"{filename}\t{sha256}\n" for filename, sha256 in frames)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_external_reference_root(
    root: Path,
    *,
    expected_csv_sha256: dict[str, str] = EXPECTED_EXTERNAL_CSV_SHA256,
    expected_allowlist_sha256: dict[
        str, str
    ] = EXPECTED_EXTERNAL_CANDIDATE_ALLOWLIST_SHA256,
) -> ValidatedExternalCandidates:
    """Gate videos 1--4 candidates and hash all 136 files before any decode.

    Two metadata-only positions per nominal vial cycle are bounded candidates.  A
    later deterministic image proxy chooses one settled-looking frame.  Video 5
    is excluded before path construction, so this function cannot open it.
    """

    root = Path(root)
    root_resolved = root.resolve(strict=True)
    csv_paths = {
        "videos.csv": root / "videos.csv",
        "vial_fill/image_index.csv": root / "vial_fill" / "image_index.csv",
        "tank_level/image_index.csv": root / "tank_level" / "image_index.csv",
    }
    parsed: dict[str, list[dict[str, str]]] = {}
    observed_csv_hashes: dict[str, str] = {}
    for name, path in csv_paths.items():
        rows, sha256 = _read_csv_with_hash(path)
        expected = _require_sha256(expected_csv_sha256.get(name), f"expected {name} hash")
        if sha256 != expected:
            raise ValueError(f"external CSV SHA-256 mismatch: {name}")
        parsed[name] = rows
        observed_csv_hashes[name] = sha256

    video_rows = parsed["videos.csv"]
    video_metadata: dict[int, dict[str, str]] = {}
    for row in video_rows:
        try:
            video_id = int(row.get("video_id", ""))
        except ValueError:
            continue
        if video_id in EXTERNAL_VIDEO_VIAL_COUNTS:
            video_metadata[video_id] = row
    if set(video_metadata) != set(EXTERNAL_VIDEO_VIAL_COUNTS):
        raise ValueError("videos.csv must contain videos 1 through 4")
    for video_id, expected_vials in EXTERNAL_VIDEO_VIAL_COUNTS.items():
        row = video_metadata[video_id]
        if int(row.get("n_vials", "-1")) != expected_vials:
            raise ValueError(f"video {video_id} reported vial count changed")
        cameras = set(row.get("cameras", "").split(";"))
        if not {"vial_fill", "tank_level"}.issubset(cameras):
            raise ValueError(f"video {video_id} lacks required cameras")
        if row.get("labeled") != "no":
            raise ValueError(f"video {video_id} must remain an unlabeled appearance reference")

    vial_rows = _validate_external_index_rows(
        parsed["vial_fill/image_index.csv"], "vial_fill"
    )
    tank_rows = _validate_external_index_rows(
        parsed["tank_level/image_index.csv"], "tank_level"
    )

    metadata_candidates = []
    for video_id, vial_count in EXTERNAL_VIDEO_VIAL_COUNTS.items():
        vial_video = vial_rows[video_id]
        tank_video = tank_rows[video_id]
        for cycle in range(vial_count):
            for fraction in EXTERNAL_CANDIDATE_FRACTIONS:
                vial_position = _external_selected_position(
                    len(vial_video), vial_count, cycle, fraction
                )
                tank_position = _external_selected_position(
                    len(tank_video), vial_count, cycle, fraction
                )
                vial_row = vial_video[vial_position]
                tank_row = tank_video[tank_position]
                if vial_row["timestamp"] != tank_row["timestamp"]:
                    raise ValueError("paired external cameras must have identical timestamps")
                vial_match = EXTERNAL_FILENAME_RE.fullmatch(vial_row["filename"])
                tank_match = EXTERNAL_FILENAME_RE.fullmatch(tank_row["filename"])
                if vial_match is None or tank_match is None:
                    raise ValueError("paired external filename failed validation")
                if vial_match.group(2) != tank_match.group(2):
                    raise ValueError("paired external filenames must share a timestamp suffix")
                metadata_candidates.append(
                    (
                        video_id,
                        cycle + 1,
                        fraction,
                        vial_row,
                        tank_row,
                    )
                )

    # Resolve and hash the complete bounded allowlist before any caller can decode.
    candidates = []
    vial_hash_records: list[tuple[str, str]] = []
    tank_hash_records: list[tuple[str, str]] = []
    for video_id, cycle, fraction, vial_row, tank_row in metadata_candidates:
        vial_path = _resolve_external_image(root_resolved, "vial_fill", vial_row["filename"])
        tank_path = _resolve_external_image(root_resolved, "tank_level", tank_row["filename"])
        vial_hash = file_sha256(vial_path)
        tank_hash = file_sha256(tank_path)
        vial_hash_records.append((vial_row["filename"], vial_hash))
        tank_hash_records.append((tank_row["filename"], tank_hash))
        candidates.append(
            ExternalCandidateFrame(
                video_id=video_id,
                nominal_cycle=cycle,
                fraction=fraction,
                timestamp=vial_row["timestamp"],
                vial_filename=vial_row["filename"],
                vial_path=vial_path,
                vial_sha256=vial_hash,
                tank_filename=tank_row["filename"],
                tank_path=tank_path,
                tank_sha256=tank_hash,
            )
        )
    vial_digest = _allowlist_digest(vial_hash_records)
    tank_digest = _allowlist_digest(tank_hash_records)
    if vial_digest != _require_sha256(
        expected_allowlist_sha256.get("vial_fill"), "expected vial candidate allowlist hash"
    ):
        raise ValueError("external vial_fill candidate allowlist SHA-256 mismatch")
    if tank_digest != _require_sha256(
        expected_allowlist_sha256.get("tank_level"), "expected tank candidate allowlist hash"
    ):
        raise ValueError("external tank_level candidate allowlist SHA-256 mismatch")
    return ValidatedExternalCandidates(
        root=root_resolved,
        csv_sha256=observed_csv_hashes,
        vial_allowlist_sha256=vial_digest,
        tank_allowlist_sha256=tank_digest,
        candidates=tuple(candidates),
    )


def _summary(values: object) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("measurement summary requires finite values")
    return {
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p05": round(float(np.quantile(array, 0.05)), 6),
        "p95": round(float(np.quantile(array, 0.95)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _circular_hue(hue: np.ndarray, weights: np.ndarray) -> float:
    angle = 2.0 * np.pi * np.asarray(hue, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    y = float(np.sum(weights * np.sin(angle)))
    x = float(np.sum(weights * np.cos(angle)))
    if x == 0.0 and y == 0.0:
        return 0.0
    return float((math.atan2(y, x) % (2.0 * math.pi)) / (2.0 * math.pi))


def _decode_all(
    validated: ValidatedCalibrationSet,
    decoder: Callable[[str, int], np.ndarray | None],
) -> list[tuple[ValidatedFrame, np.ndarray]]:
    decoded = []
    for frame in validated.frames:
        image = decoder(str(frame.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode validated frame {frame.relative_path}")
        if image.shape != (EXPECTED_IMAGE_HEIGHT, EXPECTED_IMAGE_WIDTH, 3):
            raise ValueError(
                f"decoded frame must be 640x480 BGR: {frame.relative_path}"
            )
        decoded.append((frame, image))
    return decoded


def _normalized_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0
    hsv[..., 1:] /= 255.0
    return hsv


def _decode_external_vial_candidates(
    validated: ValidatedExternalCandidates,
    decoder: Callable[[str, int], np.ndarray | None],
) -> list[tuple[ExternalCandidateFrame, np.ndarray]]:
    decoded = []
    for candidate in validated.candidates:
        image = decoder(str(candidate.vial_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode external candidate {candidate.vial_filename}")
        if image.shape != (EXPECTED_IMAGE_HEIGHT, EXPECTED_IMAGE_WIDTH, 3):
            raise ValueError(f"external candidate must be 640x480: {candidate.vial_filename}")
        decoded.append((candidate, image))
    return decoded


def _external_candidate_metrics(bgr: np.ndarray) -> dict[str, float | bool]:
    hsv = _normalized_hsv(bgr)
    x0, y0, x1, y1 = EXTERNAL_LIQUID_ROI_XYXY
    deep = hsv[y0:y1, x0:x1]
    sx0, sy0, sx1, sy1 = EXTERNAL_STREAM_ROI_XYXY
    stream = hsv[sy0:sy1, sx0:sx1]
    saturation = float(np.mean(deep[..., 1]))
    value = float(np.mean(deep[..., 2]))
    stream_fraction = float(
        np.mean((stream[..., 1] >= 0.25) & (stream[..., 2] >= 0.07))
    )
    liquid_evidence = saturation >= 0.25 or value <= 0.20
    appearance_score = (
        saturation
        + 2.0 * max(0.0, 0.30 - value)
        - 4.0 * stream_fraction
    )
    return {
        "deep_saturation": saturation,
        "deep_value": value,
        "colored_or_dark_liquid_proxy": liquid_evidence,
        "colored_stream_fraction": stream_fraction,
        "settled_proxy_pass": stream_fraction
        <= EXTERNAL_SETTLED_STREAM_FRACTION_MAX,
        "appearance_score": appearance_score,
    }


def _select_external_frames(
    decoded: list[tuple[ExternalCandidateFrame, np.ndarray]],
) -> list[tuple[ExternalCandidateFrame, np.ndarray, dict[str, float | bool]]]:
    grouped: dict[tuple[int, int], list] = {}
    for candidate, bgr in decoded:
        metrics = _external_candidate_metrics(bgr)
        grouped.setdefault((candidate.video_id, candidate.nominal_cycle), []).append(
            (candidate, bgr, metrics)
        )
    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if len(candidates) != len(EXTERNAL_CANDIDATE_FRACTIONS):
            raise ValueError(f"external cycle {key} lacks its bounded candidates")
        settled = [item for item in candidates if item[2]["settled_proxy_pass"]]
        pool = settled or candidates
        if any(item[2]["colored_or_dark_liquid_proxy"] for item in pool):
            chosen = max(
                pool,
                key=lambda item: (
                    float(item[2]["appearance_score"]),
                    item[0].fraction,
                ),
            )
        else:
            # With no colored/dark liquid evidence, prefer the later sharp static
            # look; this avoids selecting a moving exchange frame unnecessarily.
            chosen = max(pool, key=lambda item: item[0].fraction)
        selected.append(chosen)
    if len(selected) != sum(EXTERNAL_VIDEO_VIAL_COUNTS.values()):
        raise ValueError("external selection must return one frame per nominal cycle")
    return selected


def _decode_selected_tanks(
    selected: list[tuple[ExternalCandidateFrame, np.ndarray, dict]],
    decoder: Callable[[str, int], np.ndarray | None],
) -> list[tuple[ExternalCandidateFrame, np.ndarray]]:
    decoded = []
    for candidate, _vial_bgr, _metrics in selected:
        image = decoder(str(candidate.tank_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode external tank still {candidate.tank_filename}")
        if image.shape != (EXPECTED_IMAGE_HEIGHT, EXPECTED_IMAGE_WIDTH, 3):
            raise ValueError(f"external tank still must be 640x480: {candidate.tank_filename}")
        decoded.append((candidate, image))
    return decoded


def _calibration_surface_appearance(
    decoded: list[tuple[ValidatedFrame, np.ndarray]], vial: int
) -> dict:
    x0, y0, x1, y1 = CALIBRATION_SURFACE_ROIS_XYXY[vial]
    deep_x0, deep_y0, deep_x1, deep_y1 = EXPECTED_WINDOWS[vial]["liquid_roi_xyxy"]
    records = []
    for frame, bgr in decoded:
        if frame.vial != vial:
            continue
        hsv = _normalized_hsv(bgr)
        band = hsv[y0:y1, x0:x1]
        deep = hsv[deep_y0:deep_y1, deep_x0:deep_x1]
        body_value = float(np.median(deep[..., 2]))
        gray_reflective = (band[..., 1] <= 0.25) & (
            band[..., 2] >= body_value + 0.10
        )
        bright = band[..., 2] >= max(0.30, body_value + 0.15)
        gray8 = np.rint(band[..., 2] * 255.0).astype(np.uint8)
        components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            bright.astype(np.uint8), 8
        )
        bubble_components = sum(
            1 for component in range(1, components) if 1 <= stats[component, cv2.CC_STAT_AREA] <= 80
        )
        records.append(
            {
                "hue": _circular_hue(
                    band[..., 0].ravel(),
                    (band[..., 1] * band[..., 2] + 1e-8).ravel(),
                ),
                "saturation": float(np.mean(band[..., 1])),
                "value": float(np.mean(band[..., 2])),
                "gray_reflective_fraction": float(np.mean(gray_reflective)),
                "bright_relative_fraction": float(np.mean(bright)),
                "texture_laplacian_variance": float(
                    np.var(cv2.Laplacian(gray8, cv2.CV_32F))
                ),
                "small_bright_component_count": bubble_components,
            }
        )
    return {
        "evidence": (
            "fixed manual surface/foam band from the review's permitted calibration "
            "frames; pixel values are deterministic appearance proxies"
        ),
        "roi_xyxy_half_open": [x0, y0, x1, y1],
        "height_px": y1 - y0,
        "frame_count": len(records),
        "hue_circular_0_to_1": _summary([item["hue"] for item in records]),
        "saturation": _summary([item["saturation"] for item in records]),
        "value": _summary([item["value"] for item in records]),
        "gray_reflective_fraction_proxy": _summary(
            [item["gray_reflective_fraction"] for item in records]
        ),
        "foam_or_specular_bright_fraction_proxy": _summary(
            [item["bright_relative_fraction"] for item in records]
        ),
        "texture_laplacian_variance_8bit": _summary(
            [item["texture_laplacian_variance"] for item in records]
        ),
        "small_bright_component_count_proxy": _summary(
            [item["small_bright_component_count"] for item in records]
        ),
        "proxy_caveat": (
            "Foam, bubbles and specular reflection are not semantically labeled; "
            "these fractions and component counts are appearance proxies only."
        ),
    }


def _measure_liquid(
    decoded: list[tuple[ValidatedFrame, np.ndarray]], vial: int
) -> dict:
    expected = EXPECTED_WINDOWS[vial]
    x_first, y_first, x_last, y_last = expected["liquid_roi_xyxy"]
    per_frame = []
    pixels = []
    for frame, bgr in decoded:
        if frame.vial != vial:
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1:] /= 255.0
        sample = hsv[y_first:y_last, x_first:x_last].reshape(-1, 3)
        pixels.append(sample)
        hue = _circular_hue(sample[:, 0], sample[:, 1] * sample[:, 2] + 1e-8)
        per_frame.append([hue, float(np.mean(sample[:, 1])), float(np.mean(sample[:, 2]))])
    all_pixels = np.concatenate(pixels, axis=0)
    per_frame_array = np.asarray(per_frame, dtype=np.float64)
    hue = _circular_hue(
        all_pixels[:, 0], all_pixels[:, 1] * all_pixels[:, 2] + 1e-8
    )
    return {
        "appearance_description": expected["appearance_description"],
        "frame_count": expected["count"],
        "roi_xyxy_half_open": expected["liquid_roi_xyxy"],
        "hue_circular_0_to_1": round(hue, 6),
        "hue_degrees": round(hue * 360.0, 3),
        "per_frame_hue_circular_0_to_1": _summary(per_frame_array[:, 0]),
        "per_frame_saturation": _summary(per_frame_array[:, 1]),
        "per_frame_value": _summary(per_frame_array[:, 2]),
        "pixel_saturation": _summary(all_pixels[:, 1]),
        "pixel_value": _summary(all_pixels[:, 2]),
    }


def _measure_label(decoded: list[tuple[ValidatedFrame, np.ndarray]]) -> dict:
    mask = np.zeros((EXPECTED_IMAGE_HEIGHT, EXPECTED_IMAGE_WIDTH), dtype=np.uint8)
    polygon = np.asarray(LABEL_POLYGON_XY, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    records = []
    for frame, bgr in decoded:
        if frame.vial != 1:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        values = gray[mask > 0]
        records.append(
            {
                "p10": float(np.quantile(values, 0.10)),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
                "dark_fraction_at_or_below_0_30": float(np.mean(values <= 0.30)),
                "paper_fraction_at_or_above_0_50": float(np.mean(values >= 0.50)),
                "median_rgb": np.median(rgb[mask > 0], axis=0),
            }
        )
    first, second = LABEL_POLYGON_XY[0], LABEL_POLYGON_XY[1]
    roll_degrees = math.degrees(
        math.atan2(second[1] - first[1], second[0] - first[0])
    )
    medians = np.asarray([item["median_rgb"] for item in records])
    return {
        "evidence": "manual boundary on the temporal mean of all 28 vial-1 frames",
        "frame_count": 28,
        "polygon_xy": LABEL_POLYGON_XY,
        "bbox_xyxy_half_open": LABEL_BBOX_XYXY,
        "width_px": LABEL_BBOX_XYXY[2] - LABEL_BBOX_XYXY[0],
        "height_px": LABEL_BBOX_XYXY[3] - LABEL_BBOX_XYXY[1],
        "center_xy": [
            (LABEL_BBOX_XYXY[0] + LABEL_BBOX_XYXY[2]) // 2,
            (LABEL_BBOX_XYXY[1] + LABEL_BBOX_XYXY[3]) // 2,
        ],
        "top_edge_roll_degrees": round(roll_degrees, 3),
        "visible_width_fraction_of_model_crop": 0.5,
        "look": "large white/gray wraparound paper with dense dark logo, text and barcode-like print",
        "grayscale_p10": _summary([item["p10"] for item in records]),
        "grayscale_p25": _summary([item["p25"] for item in records]),
        "grayscale_p50": _summary([item["p50"] for item in records]),
        "grayscale_p75": _summary([item["p75"] for item in records]),
        "grayscale_p90": _summary([item["p90"] for item in records]),
        "dark_print_fraction_proxy": _summary(
            [item["dark_fraction_at_or_below_0_30"] for item in records]
        ),
        "paper_fraction_proxy": _summary(
            [item["paper_fraction_at_or_above_0_50"] for item in records]
        ),
        "median_rgb": {
            "r": _summary(medians[:, 0]),
            "g": _summary(medians[:, 1]),
            "b": _summary(medians[:, 2]),
        },
    }


def _measure_global_look(decoded: list[tuple[ValidatedFrame, np.ndarray]]) -> dict:
    records = []
    x_first, y_first, x_last, y_last = BACKDROP_ROI_XYXY
    scene_first, scene_last = SCENE_ROWS_HALF_OPEN
    for _frame, bgr in decoded:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        scene = gray[scene_first:scene_last]
        backdrop = rgb[y_first:y_last, x_first:x_last]
        backdrop_gray = gray[y_first:y_last, x_first:x_last]
        backdrop_hsv = cv2.cvtColor(
            np.rint(backdrop * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV
        ).astype(np.float32)
        backdrop_hsv[..., 1:] /= 255.0
        residual = backdrop_gray - cv2.GaussianBlur(backdrop_gray, (0, 0), 1.5)
        residual_median = float(np.median(residual))
        noise_sigma = 1.4826 * float(
            np.median(np.abs(residual - residual_median))
        )
        laplacian = cv2.Laplacian(
            np.rint(scene * 255.0).astype(np.uint8), cv2.CV_32F
        )
        gradient_x = cv2.Sobel(scene, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(scene, cv2.CV_32F, 0, 1, ksize=3)
        backdrop_luma = np.dot(backdrop, [0.2126, 0.7152, 0.0722])
        p05 = float(np.quantile(scene, 0.05))
        p95 = float(np.quantile(scene, 0.95))
        backdrop_p05 = float(np.quantile(backdrop_luma, 0.05))
        backdrop_p95 = float(np.quantile(backdrop_luma, 0.95))
        records.append(
            {
                "scene_p05": p05,
                "scene_p50": float(np.median(scene)),
                "scene_p95": p95,
                "scene_robust_contrast": p95 - p05,
                "scene_std": float(np.std(scene)),
                "backdrop_luma": float(np.median(backdrop_luma)),
                "backdrop_p05": backdrop_p05,
                "backdrop_p95": backdrop_p95,
                "backdrop_contrast": backdrop_p95 - backdrop_p05,
                "backdrop_saturation": float(np.median(backdrop_hsv[..., 1])),
                "backdrop_rgb": np.median(backdrop.reshape(-1, 3), axis=0),
                "noise_sigma": noise_sigma,
                "laplacian_variance": float(np.var(laplacian)),
                "tenengrad": float(np.mean(gradient_x**2 + gradient_y**2)),
            }
        )
    backdrop_rgb = np.asarray([item["backdrop_rgb"] for item in records])
    return {
        "frame_count": 98,
        "scene_without_timestamp": {
            "rows_half_open": SCENE_ROWS_HALF_OPEN,
            "grayscale_p05": _summary([item["scene_p05"] for item in records]),
            "grayscale_p50": _summary([item["scene_p50"] for item in records]),
            "grayscale_p95": _summary([item["scene_p95"] for item in records]),
            "robust_contrast_p95_minus_p05": _summary(
                [item["scene_robust_contrast"] for item in records]
            ),
            "grayscale_standard_deviation": _summary(
                [item["scene_std"] for item in records]
            ),
        },
        "backdrop": {
            "roi_xyxy_half_open": BACKDROP_ROI_XYXY,
            "median_luma": _summary([item["backdrop_luma"] for item in records]),
            "luma_p05": _summary([item["backdrop_p05"] for item in records]),
            "luma_p95": _summary([item["backdrop_p95"] for item in records]),
            "robust_contrast_p95_minus_p05": _summary(
                [item["backdrop_contrast"] for item in records]
            ),
            "median_saturation": _summary(
                [item["backdrop_saturation"] for item in records]
            ),
            "median_rgb": {
                "r": _summary(backdrop_rgb[:, 0]),
                "g": _summary(backdrop_rgb[:, 1]),
                "b": _summary(backdrop_rgb[:, 2]),
            },
            "tint_r_minus_g": _summary(backdrop_rgb[:, 0] - backdrop_rgb[:, 1]),
            "tint_b_minus_g": _summary(backdrop_rgb[:, 2] - backdrop_rgb[:, 1]),
        },
        "blur_proxies": {
            "laplacian_variance_8bit": _summary(
                [item["laplacian_variance"] for item in records]
            ),
            "tenengrad_normalized": _summary(
                [item["tenengrad"] for item in records]
            ),
        },
        "noise_proxy": {
            "method": "1.4826 times MAD of residual after Gaussian sigma 1.5 in backdrop ROI",
            "normalized_sigma": _summary([item["noise_sigma"] for item in records]),
        },
        "haze_proxy": {
            "physical_haze_identifiable": False,
            "wording": (
                "Physical haze is not identifiable from one camera view. The backdrop "
                "luma floor and local robust contrast are appearance proxies only."
            ),
            "backdrop_luma_floor_p05": _summary(
                [item["backdrop_p05"] for item in records]
            ),
            "backdrop_local_robust_contrast": _summary(
                [item["backdrop_contrast"] for item in records]
            ),
        },
    }


def _row_runs(values: np.ndarray, first_row: int, minimum_length: int = 4) -> list[list[int]]:
    runs = []
    start = None
    for offset, value in enumerate(values.tolist()):
        if value and start is None:
            start = first_row + offset
        at_end = offset == len(values) - 1
        if start is not None and (not value or at_end):
            stop = first_row + offset + (1 if value and at_end else 0)
            if stop - start >= minimum_length:
                runs.append([start, stop])
            start = None
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= 12:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return merged


def _external_surface_row_proxy(hsv: np.ndarray, body_hue: float, body_saturation: float, body_value: float) -> tuple[int | None, list[list[int]]]:
    strip = hsv[80:430, 294:322]
    hue_distance = np.abs((strip[..., 0] - body_hue + 0.5) % 1.0 - 0.5)
    compatible = (
        (hue_distance <= max(0.06, 0.12 * (1.0 - body_saturation)))
        & (strip[..., 1] >= max(0.12, body_saturation * 0.32))
        & (strip[..., 2] <= max(0.50, body_value * 3.2))
    )
    runs = _row_runs(np.mean(compatible, axis=1) >= 0.30, 80)
    if not runs:
        return None, []
    directly_above_holder = [
        run for run in runs if run[0] < 250 and run[1] >= 210 and run[1] - run[0] >= 8
    ]
    if directly_above_holder:
        chosen = max(directly_above_holder, key=lambda run: run[1])
    else:
        chosen = max(runs, key=lambda run: (run[1] - run[0], run[1]))
    return chosen[0], runs


def _measure_external_vials(
    selected: list[tuple[ExternalCandidateFrame, np.ndarray, dict]],
) -> dict:
    all_records = []
    colored_records = []
    surface_records = []
    label_records = []
    settled_images = []
    for candidate, bgr, candidate_metrics in selected:
        hsv = _normalized_hsv(bgr)
        x0, y0, x1, y1 = EXTERNAL_LIQUID_ROI_XYXY
        deep = hsv[y0:y1, x0:x1]
        body_hue = _circular_hue(
            deep[..., 0].ravel(), (deep[..., 1] * deep[..., 2] + 1e-8).ravel()
        )
        body_saturation = float(np.mean(deep[..., 1]))
        body_value = float(np.mean(deep[..., 2]))
        record = {
            "video_id": candidate.video_id,
            "nominal_cycle": candidate.nominal_cycle,
            "candidate_fraction": candidate.fraction,
            "filename": candidate.vial_filename,
            "sha256": candidate.vial_sha256,
            "timestamp": candidate.timestamp,
            "settled_proxy_pass": bool(candidate_metrics["settled_proxy_pass"]),
            "colored_stream_fraction": round(
                float(candidate_metrics["colored_stream_fraction"]), 6
            ),
            "colored_or_dark_liquid_proxy": bool(
                candidate_metrics["colored_or_dark_liquid_proxy"]
            ),
            "deep_liquid_hue_circular_0_to_1": round(body_hue, 6),
            "deep_liquid_saturation": round(body_saturation, 6),
            "deep_liquid_value": round(body_value, 6),
        }
        all_records.append(record)
        if not candidate_metrics["settled_proxy_pass"]:
            continue
        settled_images.append((candidate, bgr))

        lx0, ly0, lx1, ly1 = EXTERNAL_LABEL_SEARCH_ROI_XYXY
        label = hsv[ly0:ly1, lx0:lx1]
        label_gray = cv2.cvtColor(bgr[ly0:ly1, lx0:lx1], cv2.COLOR_BGR2GRAY)
        label_records.append(
            {
                "paper_fraction": float(
                    np.mean((label[..., 1] <= 0.25) & (label[..., 2] >= 0.45))
                ),
                "dark_print_fraction": float(np.mean(label[..., 2] <= 0.30)),
                "edge_fraction": float(
                    np.mean(cv2.Canny(label_gray, 60, 140) > 0)
                ),
            }
        )
        if not candidate_metrics["colored_or_dark_liquid_proxy"]:
            continue
        colored_records.append(record)
        top_row, runs = _external_surface_row_proxy(
            hsv, body_hue, body_saturation, body_value
        )
        if top_row is None:
            continue
        visibility = (
            "holder_overlap_or_edge" if 245 <= top_row <= 285 else "directly_visible_proxy"
        )
        sx0, sx1 = 290, 330
        sy0, sy1 = top_row, min(top_row + 16, EXPECTED_IMAGE_HEIGHT)
        band = hsv[sy0:sy1, sx0:sx1]
        bright = band[..., 2] >= max(0.30, body_value + 0.15)
        gray_reflective = (band[..., 1] <= 0.25) & (
            band[..., 2] >= body_value + 0.10
        )
        components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            bright.astype(np.uint8), 8
        )
        surface_records.append(
            {
                "video_id": candidate.video_id,
                "nominal_cycle": candidate.nominal_cycle,
                "filename": candidate.vial_filename,
                "top_row_proxy": top_row,
                "compatible_row_runs_half_open": runs,
                "visibility": visibility,
                "band_xyxy_half_open": [sx0, sy0, sx1, sy1],
                "hue_circular_0_to_1": round(
                    _circular_hue(
                        band[..., 0].ravel(),
                        (band[..., 1] * band[..., 2] + 1e-8).ravel(),
                    ),
                    6,
                ),
                "saturation": round(float(np.mean(band[..., 1])), 6),
                "value": round(float(np.mean(band[..., 2])), 6),
                "gray_reflective_fraction_proxy": round(
                    float(np.mean(gray_reflective)), 6
                ),
                "foam_or_specular_bright_fraction_proxy": round(
                    float(np.mean(bright)), 6
                ),
                "small_bright_component_count_proxy": sum(
                    1
                    for component in range(1, components)
                    if 1 <= stats[component, cv2.CC_STAT_AREA] <= 80
                ),
            }
        )

    colored_values = [record["deep_liquid_value"] for record in colored_records]
    colored_saturation = [record["deep_liquid_saturation"] for record in colored_records]
    colored_hues = [record["deep_liquid_hue_circular_0_to_1"] for record in colored_records]
    direct_surfaces = [
        record for record in surface_records if record["visibility"] == "directly_visible_proxy"
    ]
    return {
        "selection_records": all_records,
        "selected_nominal_cycle_count": len(all_records),
        "settled_proxy_pass_count": len(settled_images),
        "settled_proxy": {
            "method": (
                "among 80% and 95% metadata-only candidates, reject candidates whose "
                "nozzle-to-vial corridor has more than 3% colored-stream pixels; prefer "
                "the strongest colored/dark liquid appearance, otherwise the later still"
            ),
            "limitation": (
                "This rejects visible colored pours but cannot prove that clear liquid is "
                "motionless; non-passing selections are logged and excluded from appearance summaries."
            ),
            "stream_roi_xyxy_half_open": EXTERNAL_STREAM_ROI_XYXY,
            "maximum_colored_stream_fraction": EXTERNAL_SETTLED_STREAM_FRACTION_MAX,
        },
        "liquid": {
            "deep_roi_xyxy_half_open": EXTERNAL_LIQUID_ROI_XYXY,
            "classification": "mean saturation >= 0.25 or mean value <= 0.20",
            "colored_or_dark_settled_count": len(colored_records),
            "per_frame_hue_circular_0_to_1": _summary(colored_hues),
            "per_frame_saturation": _summary(colored_saturation),
            "per_frame_value": _summary(colored_values),
        },
        "labels": {
            "evidence": "fixed vial-region pixel proxies on settled-proxy stills; not OCR or a label detector",
            "search_roi_xyxy_half_open": EXTERNAL_LABEL_SEARCH_ROI_XYXY,
            "frame_count": len(label_records),
            "paper_fraction_proxy": _summary(
                [record["paper_fraction"] for record in label_records]
            ),
            "dark_print_fraction_proxy": _summary(
                [record["dark_print_fraction"] for record in label_records]
            ),
            "edge_fraction_proxy": _summary(
                [record["edge_fraction"] for record in label_records]
            ),
            "geometry_reference": (
                "Use the manually measured BRADY polygon from video-5 vial 1 for size, "
                "height and roll; these videos add look variation only."
            ),
        },
        "top_surface_and_foam": {
            "evidence": (
                "hue-compatible row-run top plus a fixed 16-pixel band; rows overlapping "
                "the holder are flagged instead of claimed visible"
            ),
            "surface_records": surface_records,
            "row_proxy_count": len(surface_records),
            "directly_visible_proxy_count": len(direct_surfaces),
            "top_row_proxy": _summary(
                [record["top_row_proxy"] for record in surface_records]
            ),
            "direct_surface_saturation": _summary(
                [record["saturation"] for record in direct_surfaces]
            ),
            "direct_surface_value": _summary(
                [record["value"] for record in direct_surfaces]
            ),
            "direct_gray_reflective_fraction_proxy": _summary(
                [record["gray_reflective_fraction_proxy"] for record in direct_surfaces]
            ),
            "direct_foam_or_specular_bright_fraction_proxy": _summary(
                [record["foam_or_specular_bright_fraction_proxy"] for record in direct_surfaces]
            ),
            "proxy_caveat": (
                "There are no semantic foam or bubble labels. Bright fractions and small "
                "components mix foam, bubbles, glare and compression artifacts."
            ),
        },
        "global_look": _measure_external_global_look(settled_images),
    }


def _measure_external_global_look(
    decoded: list[tuple[ExternalCandidateFrame, np.ndarray]],
) -> dict:
    records = []
    x0, y0, x1, y1 = BACKDROP_ROI_XYXY
    for _candidate, bgr in decoded:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        scene = gray[SCENE_ROWS_HALF_OPEN[0] : SCENE_ROWS_HALF_OPEN[1]]
        backdrop = gray[y0:y1, x0:x1]
        residual = backdrop - cv2.GaussianBlur(backdrop, (0, 0), 1.5)
        p05 = float(np.quantile(scene, 0.05))
        p95 = float(np.quantile(scene, 0.95))
        records.append(
            {
                "contrast": p95 - p05,
                "scene_std": float(np.std(scene)),
                "backdrop_luma": float(np.median(backdrop)),
                "backdrop_contrast": float(np.quantile(backdrop, 0.95) - np.quantile(backdrop, 0.05)),
                "noise": 1.4826 * float(np.median(np.abs(residual - np.median(residual)))),
                "laplacian": float(
                    np.var(cv2.Laplacian(np.rint(scene * 255).astype(np.uint8), cv2.CV_32F))
                ),
            }
        )
    return {
        "frame_count": len(records),
        "robust_contrast_p95_minus_p05": _summary([item["contrast"] for item in records]),
        "grayscale_standard_deviation": _summary([item["scene_std"] for item in records]),
        "backdrop_median_luma": _summary([item["backdrop_luma"] for item in records]),
        "backdrop_local_robust_contrast": _summary([item["backdrop_contrast"] for item in records]),
        "blur_proxy_laplacian_variance_8bit": _summary([item["laplacian"] for item in records]),
        "noise_proxy_normalized_sigma": _summary([item["noise"] for item in records]),
        "haze_proxy": {
            "physical_haze_identifiable": False,
            "wording": (
                "Physical haze is not identifiable from one camera view. Backdrop luma "
                "and local contrast are appearance proxies only."
            ),
        },
    }


def _measure_tank_dyes(
    decoded: list[tuple[ExternalCandidateFrame, np.ndarray]],
) -> dict:
    tanks = {}
    all_hues = []
    all_saturations = []
    for name, roi in EXTERNAL_TANK_ROIS_XYXY.items():
        x0, y0, x1, y1 = roi
        records = []
        for _candidate, bgr in decoded:
            hsv = _normalized_hsv(bgr)[y0:y1, x0:x1]
            hue = _circular_hue(
                hsv[..., 0].ravel(), (hsv[..., 1] * hsv[..., 2] + 1e-8).ravel()
            )
            saturation = float(np.mean(hsv[..., 1]))
            records.append(
                [hue, saturation, float(np.mean(hsv[..., 2]))]
            )
            all_hues.append(hue)
            all_saturations.append(saturation)
        array = np.asarray(records)
        tanks[name] = {
            "roi_xyxy_half_open": roi,
            "frame_count": len(records),
            "hue_circular_0_to_1": _summary(array[:, 0]),
            "hue_degrees": _summary(array[:, 0] * 360.0),
            "saturation": _summary(array[:, 1]),
            "value": _summary(array[:, 2]),
        }
    chromatic_hues = [
        hue for hue, saturation in zip(all_hues, all_saturations) if saturation >= 0.10
    ]
    return {
        "evidence": (
            "four fixed left-body ROIs avoid the central white level labels and metal; "
            "camera pixels are dye-colour anchors, not concentration measurements"
        ),
        "tanks": tanks,
        "chromatic_hue_anchor_range_0_to_1": _summary(chromatic_hues),
        "renderer_dark_mix_proposal": {
            "hue_source": "measured tank hue anchors plus measured settled vial hues",
            "value_range": [0.03, 0.35],
            "evidence_label": (
                "proposal combining measured dye hues with observed dark-liquid values; "
                "not a physical dye-mixing calibration"
            ),
        },
    }


def _external_provenance(
    validated: ValidatedExternalCandidates,
    selected: list[tuple[ExternalCandidateFrame, np.ndarray, dict]],
) -> dict:
    candidates = []
    for candidate in validated.candidates:
        candidates.append(
            {
                "video_id": candidate.video_id,
                "nominal_cycle": candidate.nominal_cycle,
                "candidate_fraction": candidate.fraction,
                "timestamp": candidate.timestamp,
                "vial_fill": {
                    "path": f"vial_fill/images/{candidate.vial_filename}",
                    "sha256": candidate.vial_sha256,
                },
                "tank_level": {
                    "path": f"tank_level/images/{candidate.tank_filename}",
                    "sha256": candidate.tank_sha256,
                },
            }
        )
    selected_records = []
    for candidate, _bgr, metrics in selected:
        selected_records.append(
            {
                "video_id": candidate.video_id,
                "nominal_cycle": candidate.nominal_cycle,
                "candidate_fraction": candidate.fraction,
                "timestamp": candidate.timestamp,
                "settled_proxy_pass": bool(metrics["settled_proxy_pass"]),
                "vial_fill_path": f"vial_fill/images/{candidate.vial_filename}",
                "vial_fill_sha256": candidate.vial_sha256,
                "tank_level_path": f"tank_level/images/{candidate.tank_filename}",
                "tank_level_sha256": candidate.tank_sha256,
            }
        )
    return {
        "root_name": validated.root.name,
        "allowed_video_ids": [1, 2, 3, 4],
        "forbidden_video_ids": [5],
        "video5_paths_resolved_hashed_or_decoded": 0,
        "video5_vial_colors_csv_opened": 0,
        "csv_sha256": validated.csv_sha256,
        "candidate_fractions_within_nominal_cycle": list(EXTERNAL_CANDIDATE_FRACTIONS),
        "selection_rule": (
            "split each video's pinned camera index into videos.csv n_vials equal nominal "
            "cycles; gate paired 80% and 95% positions, then apply the logged settled proxy"
        ),
        "candidate_pair_count": len(validated.candidates),
        "candidate_image_count_hashed_before_first_decode": len(validated.candidates) * 2,
        "vial_candidate_allowlist_sha256": validated.vial_allowlist_sha256,
        "tank_candidate_allowlist_sha256": validated.tank_allowlist_sha256,
        "vial_candidate_frames_decoded": len(validated.candidates),
        "tank_selected_frames_decoded": len(selected),
        "selected_pair_count": len(selected),
        "candidate_pairs": candidates,
        "selected_pairs": selected_records,
    }


def measure_real_appearance(
    root: Path,
    *,
    external_root: Path | None = None,
    expected_external_csv_sha256: dict[str, str] = EXPECTED_EXTERNAL_CSV_SHA256,
    expected_external_allowlist_sha256: dict[
        str, str
    ] = EXPECTED_EXTERNAL_CANDIDATE_ALLOWLIST_SHA256,
    decoder: Callable[[str, int], np.ndarray | None] = cv2.imread,
) -> dict:
    """Validate all inputs, then measure calibration and optional videos-1--4 references."""

    validated = validate_calibration_root(root)
    external_validated = None
    if external_root is not None:
        external_validated = validate_external_reference_root(
            external_root,
            expected_csv_sha256=expected_external_csv_sha256,
            expected_allowlist_sha256=expected_external_allowlist_sha256,
        )
    # Both gates, including every allowlisted image hash, finish before first decode.
    decoded = _decode_all(validated, decoder)
    report = {
        "schema_version": 2,
        "purpose": (
            "renderer-design evidence from permitted video-5 calibration frames and "
            "unlabeled appearance references in external videos 1-4"
        ),
        "evidence_label": "measured_real_appearance_without_real_model_training",
        "training_policy": "appearance measurements only; real frames are never model training data",
        "additional_video5_frames_opened": 0,
        "other_real_frames_opened": 0,
        "other_real_frames_opened_definition": (
            "legacy video-5-only counter: no video-5 calibration or real-check frame "
            "outside the approved closed 98-frame vial-1/vial-13 set was opened; "
            "videos-1--4 appearance-reference counts are recorded separately"
        ),
        "provenance": {
            "calibration_root_name": validated.root.name,
            "index_filename": "index.json",
            "index_sha256": validated.index_sha256,
            "source_video_sha256": validated.source_video_sha256,
            "windows_csv_sha256": validated.windows_csv_sha256,
            "frame_count": 98,
            "vial_counts": {"1": 28, "13": 70},
            "allowed_vials": [1, 13],
            "role": "calibration",
            "source_window_role": "real_validation",
            "image_contract": {
                "encoding": "PNG",
                "lossless": True,
                "width": EXPECTED_IMAGE_WIDTH,
                "height": EXPECTED_IMAGE_HEIGHT,
            },
            "validation_order": (
                "index contract, paths and all 98 SHA-256 values pass before first decode"
            ),
        },
        "liquids": {
            "vial_1_30ml": _measure_liquid(decoded, 1),
            "vial_13_80ml": _measure_liquid(decoded, 13),
        },
        "calibration_top_surface_and_foam": {
            "vial_1_30ml": _calibration_surface_appearance(decoded, 1),
            "vial_13_80ml": _calibration_surface_appearance(decoded, 13),
            "surface_oval_height_reference_px": [
                CALIBRATION_SURFACE_ROIS_XYXY[1][3]
                - CALIBRATION_SURFACE_ROIS_XYXY[1][1],
                CALIBRATION_SURFACE_ROIS_XYXY[13][3]
                - CALIBRATION_SURFACE_ROIS_XYXY[13][1],
            ],
        },
        "brady_label": _measure_label(decoded),
        "global_look": _measure_global_look(decoded),
    }
    if external_validated is not None:
        external_candidates = _decode_external_vial_candidates(
            external_validated, decoder
        )
        selected = _select_external_frames(external_candidates)
        tanks = _decode_selected_tanks(selected, decoder)
        report["external_reference_sample"] = {
            "provenance": _external_provenance(external_validated, selected),
            "vial_fill": _measure_external_vials(selected),
            "tank_level_dye_colors": _measure_tank_dyes(tanks),
        }
        report["renderer_guidance"] = {
            "evidence_label": (
                "proposal derived from the measured ranges above; renderer settings, "
                "not additional physical measurements"
            ),
            "liquid": {
                "value_range": [0.03, 0.35],
                "dark_frame_fraction_at_or_below_value_0_20": "about one third",
                "hue_families_circular_0_to_1": {
                    "red_brown": [0.0, 0.06],
                    "olive": [0.18, 0.22],
                    "green_teal": [0.33, 0.52],
                    "blue": [0.55, 0.59],
                },
                "near_black": "apply across hue families, not as one hue",
            },
            "dark_liquid_top_surface": {
                "value_range": [0.16, 0.28],
                "saturation_range": [0.08, 0.25],
                "look": (
                    "gray reflective oval with subdued liquid tint; include small "
                    "liquid-tinted or gray foam patches and small surface bubbles"
                ),
                "oval_height_px": [17, 24],
                "basis": (
                    "calibration surface V medians near 0.172 and 0.217, vial-13 "
                    "saturation near 0.105 and gray-reflective fraction near 0.893; "
                    "external direct-surface V mean near 0.221 and p95 near 0.277"
                ),
            },
            "labels": {
                "brady_bbox_width_px": 56,
                "brady_bbox_height_px": 72,
                "brady_center_xy": [313, 204],
                "brady_top_edge_roll_degrees": -3.302,
                "variation": "large white/gray labels with dense dark print at varied heights and rolls",
            },
        }
    return report


def json_text(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json_new(path: Path, report: dict) -> None:
    """Write deterministic JSON to a fresh path and never overwrite evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json_text(report))

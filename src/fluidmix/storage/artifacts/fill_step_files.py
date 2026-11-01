"""Filesystem adapter for frozen fill-step tracker artifacts.

Perception owns arrays and tracker decisions; this outer adapter owns exclusive
publication and image-file decoding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ...perception.baselines.empty_reference_rgb_predictor import IMAGE_SHAPE, fillcam_masks
from ...perception.tracker.fill_step_production import ProductionFillStepError


class TrackerFileAdapter:
    """Injected outer capability used by the pure tracker runtime orchestrator."""

    def publish(self, path: Path, payload: bytes) -> None:
        write_new(path, payload)

    def persist_observations(
        self, *, output_root: Path, observations: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        return persist_completed_observations(
            output_root=output_root, observations=observations
        )

    def decode_rgb(self, path: Path) -> tuple[np.ndarray, list[int]]:
        return decode_rgb_image_file(path)

    def exists(self, path: Path) -> bool:
        return path.exists()

    def sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    """Publish one new file without clobbering an existing claim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def decode_rgb_image_file(path: Path) -> tuple[np.ndarray, list[int]]:
    """Decode one RGB image for an outer-pipeline locator binding."""

    with Image.open(path) as image:
        pixels = np.asarray(image, dtype=np.uint8).copy()
        dimensions = list(image.size)
    return pixels, dimensions


def persist_completed_observations(
    *, output_root: Path, observations: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Exclusively persist native RGB/mask files and a trusted binding registry."""

    root = Path(output_root).expanduser().absolute()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ProductionFillStepError("tracker output root already exists") from error
    bindings: dict[str, dict[str, object]] = {}
    for row in observations:
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in bindings:
            raise ProductionFillStepError("observation image identity is invalid")
        rgb = _rgb(row.get("observation_rgb"))
        mask = np.asarray(row.get("prediction_mask"))
        if mask.dtype != np.bool_ or mask.shape != IMAGE_SHAPE:
            raise ProductionFillStepError("observation mask is invalid")
        rgb_buffer, mask_buffer = BytesIO(), BytesIO()
        Image.fromarray(rgb, mode="RGB").save(rgb_buffer, format="PNG")
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            mask_buffer, format="PNG"
        )
        rgb_path = root / "rgb" / f"{image_id}.png"
        mask_path = root / "mask" / f"{image_id}.png"
        roi, _clamp = fillcam_masks()
        occupied_roi = np.argwhere(roi)
        y0, x0 = occupied_roi.min(axis=0)
        y1, x1 = occupied_roi.max(axis=0) + 1
        crop_buffer = BytesIO()
        Image.fromarray(rgb[y0:y1, x0:x1], mode="RGB").save(
            crop_buffer, format="PNG"
        )
        crop_path = root / "crop" / f"{image_id}.png"
        write_new(rgb_path, rgb_buffer.getvalue())
        write_new(mask_path, mask_buffer.getvalue())
        write_new(crop_path, crop_buffer.getvalue())
        bindings[image_id] = {
            "frame_index": row["frame_index"], "phase": row["phase"],
            "rgb_path": str(rgb_path),
            "rgb_sha256": hashlib.sha256(rgb_buffer.getvalue()).hexdigest(),
            "mask_path": str(mask_path),
            "mask_sha256": hashlib.sha256(mask_buffer.getvalue()).hexdigest(),
            "crop_path": str(crop_path),
            "crop_sha256": hashlib.sha256(crop_buffer.getvalue()).hexdigest(),
            "crop_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        }
    registry: dict[str, object] = {
        "schema_version": "wp7-fill-step-camera-registry-v1",
        "artifact_type": "wp7_fill_step_camera_registry",
        "images": bindings,
    }
    registry["registry_sha256"] = _canonical_hash(registry)
    write_new(
        root / "registry.json",
        json.dumps(registry, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n",
    )
    return registry


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _rgb(value: object) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.shape != (*IMAGE_SHAPE, 3):
        raise ProductionFillStepError("camera frame must be 480x640 uint8 RGB")
    return frame


def read_regular_file(path: Path, role: str) -> bytes:
    """Read exact bytes only from a regular, non-symlink artifact path."""

    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ProductionFillStepError(f"{role} must be a regular file") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise ProductionFillStepError(f"{role} must be a regular non-symlink file")
    return candidate.read_bytes()


__all__ = [
    "TrackerFileAdapter", "decode_rgb_image_file", "persist_completed_observations",
    "read_regular_file", "write_new",
]

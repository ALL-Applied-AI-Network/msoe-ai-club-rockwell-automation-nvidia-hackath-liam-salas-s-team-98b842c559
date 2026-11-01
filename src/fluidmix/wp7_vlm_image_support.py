"""Shared native-frame image support for WP7 VLM consumers."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np


class DevelopmentMaterializerError(ValueError):
    """WP7 image evidence is invalid for materialization."""


def crop_vial_roi(
    frame: object, roi: Sequence[object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Crop one native-resolution RGB/BGR frame and bind its source pixels."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise DevelopmentMaterializerError("ROI source frame must be uint8 three-channel")
    if (
        isinstance(roi, (str, bytes)) or not isinstance(roi, Sequence)
        or len(roi) != 4 or any(type(item) is not int for item in roi)
    ):
        raise DevelopmentMaterializerError("ROI coordinates are invalid")
    x1, y1, x2, y2 = (int(item) for item in roi)
    height, width = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise DevelopmentMaterializerError("ROI lies outside the native frame")
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    return crop, {
        "crop_box_xyxy": [x1, y1, x2, y2],
        "source_native_dimensions": [width, height],
        "crop_native_dimensions": [x2 - x1, y2 - y1],
        "source_full_frame_sha256": hashlib.sha256(
            np.ascontiguousarray(image).tobytes()
        ).hexdigest(),
        "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
    }


__all__ = ["DevelopmentMaterializerError", "crop_vial_roi"]

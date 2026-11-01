"""Deterministic same-vial empty-reference RGB liquid measurement.

The predictor uses only camera RGB bytes.  A run's first pre-pour frame is the
empty-vial reference.  Programmed volume is accepted by the public boundary so
callers cannot accidentally create a second recipe-shaped path, but it never
selects visible pixels or changes a visible height.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from collections.abc import Sequence
import hashlib
import math
import re

import numpy as np


IMAGE_SHAPE = (480, 640)
CHANGE_THRESHOLD = 40
MINIMUM_COMPONENT_AREA = 128
MINIMUM_DENSE_ROWS = 8
UPPER_MINIMUM_CHANGED_PIXELS = 800
UPPER_MINIMUM_DENSE_ROWS = 20
CLAMP_ROWS = (238, 270)
STREAM_ROI_XYXY = (300, 80, 320, 190)
STREAM_MINIMUM_CHANGED_PIXELS = 8
STREAM_MINIMUM_CONSECUTIVE_FRAMES = 3
FROZEN_FRAME_RATE_HZ = 10.0
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# Exact row spans of the source-bound projected vial prism.  The production
# body starts at row 194: rows 66--193 are the nozzle/stream region and cannot
# become liquid-body pixels.
_VIAL_ROW_SPANS = (
    (194, 211, 283, 338),
    (211, 220, 283, 337),
    (220, 254, 284, 337),
    (254, 258, 285, 337),
    (258, 288, 285, 336),
    (288, 305, 286, 336),
    (305, 323, 286, 335),
    (323, 352, 287, 335),
    (352, 357, 287, 334),
    (357, 392, 288, 334),
    (392, 399, 289, 334),
    (399, 403, 289, 333),
    (403, 405, 290, 332),
    (405, 408, 291, 332),
    (408, 409, 292, 330),
    (409, 410, 292, 329),
    (410, 411, 293, 328),
    (411, 412, 295, 327),
    (412, 413, 296, 326),
    (413, 414, 298, 325),
    (414, 415, 299, 323),
    (415, 416, 301, 320),
    (416, 417, 307, 316),
)

# The exact source-bound clamp raster; the first rows have two disjoint arms.
_CLAMP_ROW_RUNS = (
    (238, 239, ((284, 286), (335, 337))),
    (239, 240, ((284, 288), (333, 337))),
    (240, 241, ((284, 290), (331, 337))),
    (241, 242, ((284, 293), (328, 337))),
    (242, 243, ((284, 296), (324, 337))),
    (243, 244, ((284, 302), (319, 337))),
    (244, 253, ((284, 337),)),
    (253, 258, ((285, 337),)),
    (258, 265, ((285, 336),)),
    (265, 266, ((286, 336),)),
    (266, 267, ((295, 336),)),
    (267, 268, ((305, 336),)),
    (268, 269, ((314, 336),)),
    (269, 270, ((323, 336),)),
    (270, 271, ((332, 336),)),
)


@dataclass(frozen=True)
class EmptyReferencePrediction:
    mask: np.ndarray
    status: str
    surface_row: int | None
    visible_height_px: int | None
    changed_pixel_count: int
    median_rgb: tuple[float, float, float] | None


def fillcam_masks() -> tuple[np.ndarray, np.ndarray]:
    """Return copies of the recorded FillCam vial-body ROI and clamp raster."""

    roi = np.zeros(IMAGE_SHAPE, dtype=bool)
    for y0, y1, x0, x1 in _VIAL_ROW_SPANS:
        roi[y0:y1, x0:x1] = True
    clamp = np.zeros(IMAGE_SHAPE, dtype=bool)
    for y0, y1, runs in _CLAMP_ROW_RUNS:
        for x0, x1 in runs:
            clamp[y0:y1, x0:x1] = True
    return roi, clamp


def _validate_rgb(value: object, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError(f"{name} must be one 640x480 uint8 RGB array")
    return image


def _validate_programmed_ml(value: object) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("programmed_ml must be finite and non-negative when supplied")


def _empty(status: str) -> EmptyReferencePrediction:
    return EmptyReferencePrediction(
        mask=np.zeros(IMAGE_SHAPE, dtype=bool),
        status=status,
        surface_row=None,
        visible_height_px=None,
        changed_pixel_count=0,
        median_rgb=None,
    )


def _dense_row_count(component: np.ndarray, roi: np.ndarray) -> int:
    widths = np.count_nonzero(roi, axis=1)
    counts = np.count_nonzero(component, axis=1)
    return int(
        np.count_nonzero((widths > 0) & (counts >= np.ceil(widths / 3.0)))
    )


def predict_visible_liquid(rgb, reference_rgb, *, programmed_ml=None):
    """Measure visible liquid pixels relative to the same vial's empty frame.

    Only changed pixels survive into the returned mask and colour statistic.
    The nozzle/stream region is outside the body ROI.  Dense components on both
    sides of the exact clamp raster can survive; sparse glass/noise components
    cannot.  ``programmed_ml`` is validated but deliberately has no effect.
    """

    image = _validate_rgb(rgb, "rgb")
    reference = _validate_rgb(reference_rgb, "reference_rgb")
    _validate_programmed_ml(programmed_ml)
    if int(np.max(image)) == 0:
        return _empty("invalid_black_frame")
    if int(np.max(reference)) == 0:
        return _empty("invalid_black_reference")

    roi, clamp = fillcam_masks()
    difference = np.max(
        np.abs(image.astype(np.int16) - reference.astype(np.int16)), axis=2
    )
    candidates = (difference > CHANGE_THRESHOLD) & roi & ~clamp

    from scipy import ndimage

    labels, count = ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    selected = np.zeros(IMAGE_SHAPE, dtype=bool)
    clamp_start, clamp_end = CLAMP_ROWS
    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(np.count_nonzero(component))
        if area < MINIMUM_COMPONENT_AREA:
            continue
        rows = np.flatnonzero(np.any(component, axis=1))
        if not rows.size:
            continue
        top, bottom = int(rows[0]), int(rows[-1])
        dense_rows = _dense_row_count(component, roi)
        is_lower_body = bottom > clamp_end and dense_rows >= MINIMUM_DENSE_ROWS
        if is_lower_body:
            selected |= component

    # The curved glass can split the upper liquid appearance into multiple
    # disconnected change components.  Judge that fixed above-clamp region as
    # one camera observation while retaining only its actually changed pixels.
    upper_candidates = candidates.copy()
    upper_candidates[clamp_start:] = False
    if (
        int(np.count_nonzero(upper_candidates)) >= UPPER_MINIMUM_CHANGED_PIXELS
        and _dense_row_count(upper_candidates, roi) >= UPPER_MINIMUM_DENSE_ROWS
    ):
        selected |= upper_candidates

    if not np.any(selected):
        return _empty("insufficient_dense_change")
    rows = np.flatnonzero(np.any(selected, axis=1))
    surface = int(rows[0])
    bottom = int(rows[-1])
    median = tuple(float(value) for value in np.median(image[selected], axis=0))
    return EmptyReferencePrediction(
        mask=selected,
        status="visible_measured",
        surface_row=surface,
        visible_height_px=bottom - surface + 1,
        changed_pixel_count=int(np.count_nonzero(selected)),
        median_rgb=median,
    )


def _stream_detected(image: np.ndarray, reference: np.ndarray) -> bool:
    x0, y0, x1, y1 = STREAM_ROI_XYXY
    difference = np.max(
        np.abs(
            image[y0:y1, x0:x1].astype(np.int16)
            - reference[y0:y1, x0:x1].astype(np.int16)
        ),
        axis=2,
    )
    return int(np.count_nonzero(difference > CHANGE_THRESHOLD)) >= STREAM_MINIMUM_CHANGED_PIXELS


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def predict_fill_sequence(
    rgb_frames,
    *,
    reference_index=0,
    expected_reference_rgb_sha256,
    programmed_ml=None,
    frame_rate_hz=FROZEN_FRAME_RATE_HZ,
):
    """Measure a fill sequence and interpolate only image-supported clamp gaps.

    A hidden run is eligible only when two preceding visible rows show upward
    growth, a later visible row reappears above the run, and RGB contains a
    stream for at least three consecutive frames.  Interpolated rows outside
    the fixed clamp band remain abstentions.  No synthetic mask or colour is
    created for an inferred row.
    """

    if (
        not isinstance(rgb_frames, Sequence)
        or isinstance(rgb_frames, (str, bytes))
        or not rgb_frames
    ):
        raise ValueError("rgb_frames must be a nonempty sequence")
    if reference_index != 0:
        raise ValueError("reference_index must be 0 (the first pre-pour frame)")
    _validate_programmed_ml(programmed_ml)
    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, Real)
        or not math.isfinite(float(frame_rate_hz))
        or float(frame_rate_hz) <= 0.0
    ):
        raise ValueError("frame_rate_hz must be finite and positive")
    stream_minimum_consecutive_frames = math.ceil(
        STREAM_MINIMUM_CONSECUTIVE_FRAMES
        * float(frame_rate_hz)
        / FROZEN_FRAME_RATE_HZ
    )
    frames = [_validate_rgb(frame, "rgb frame") for frame in rgb_frames]
    reference = frames[0]
    observed_reference_sha256 = hashlib.sha256(reference.tobytes()).hexdigest()
    if (
        not isinstance(expected_reference_rgb_sha256, str)
        or _SHA256.fullmatch(expected_reference_rgb_sha256) is None
        or observed_reference_sha256 != expected_reference_rgb_sha256
    ):
        raise ValueError("first pre-pour RGB does not match its bound reference hash")
    results = [
        predict_visible_liquid(frame, reference, programmed_ml=programmed_ml)
        for frame in frames
    ]
    stream = [_stream_detected(frame, reference) for frame in frames]

    def has_upper_body(item: EmptyReferencePrediction) -> bool:
        return bool(np.any(item.mask[: CLAMP_ROWS[0]]))

    def is_lower_plateau(item: EmptyReferencePrediction) -> bool:
        return bool(
            item.status == "visible_measured"
            and item.surface_row is not None
            and CLAMP_ROWS[1] - 5 <= item.surface_row <= CLAMP_ROWS[1] + 1
            and not has_upper_body(item)
        )

    # A real hidden interval does not make the lower liquid disappear: its
    # changed-pixel mask plateaus at the clamp boundary while the unseen surface
    # continues upward.  Scan those plateaus, rather than treating an empty mask
    # as the hidden condition.
    index = 0
    while index < len(results):
        if not is_lower_plateau(results[index]):
            index += 1
            continue
        start = index
        while index < len(results) and is_lower_plateau(results[index]):
            index += 1
        stop = index
        plateau_indices = list(range(start, stop))
        plateau_rows = [results[item].surface_row for item in plateau_indices]
        assert all(row is not None for row in plateau_rows)
        if max(plateau_rows) - min(plateau_rows) > 2:
            continue
        if start < 2 or stop >= len(results):
            continue
        previous = results[start - 2]
        before = results[start - 1]
        after = results[stop]
        if (
            previous.surface_row is None
            or before.surface_row is None
            or after.surface_row is None
            or not has_upper_body(after)
            or not (
                previous.surface_row
                > before.surface_row
                > max(plateau_rows)
                > after.surface_row
            )
        ):
            continue
        if (
            _longest_true_run([stream[item] for item in plateau_indices])
            < stream_minimum_consecutive_frames
        ):
            continue
        denominator = stop - (start - 1)
        for item in plateau_indices:
            if not stream[item]:
                continue
            fraction = (item - (start - 1)) / denominator
            inferred = int(
                round(
                    before.surface_row
                    + (after.surface_row - before.surface_row) * fraction
                )
            )
            if not CLAMP_ROWS[0] <= inferred <= CLAMP_ROWS[1]:
                continue
            observed = results[item]
            rows = np.flatnonzero(np.any(observed.mask, axis=1))
            bottom = int(rows[-1])
            results[item] = EmptyReferencePrediction(
                mask=observed.mask,
                status="clamp_hidden_inferred",
                surface_row=inferred,
                visible_height_px=bottom - inferred + 1,
                changed_pixel_count=observed.changed_pixel_count,
                median_rgb=observed.median_rgb,
            )
    return results

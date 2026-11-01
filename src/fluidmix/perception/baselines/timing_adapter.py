"""Frame-rate adapter for the receipt-bound empty-reference predictor."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
import math

import numpy as np

from . import empty_reference_rgb_predictor as _frozen


FROZEN_FRAME_RATE_HZ = 10.0


def predict_fill_sequence_at_frame_rate(
    rgb_frames: Sequence[object],
    *,
    frame_rate_hz: Real,
    reference_index: int = 0,
    expected_reference_rgb_sha256: str,
    programmed_ml: Real | None = None,
):
    """Preserve the frozen predictor's 0.3-second hidden-band support."""

    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, Real)
        or not math.isfinite(float(frame_rate_hz))
        or float(frame_rate_hz) <= 0.0
    ):
        raise ValueError("frame_rate_hz must be finite and positive")
    checked_frame_rate_hz = float(frame_rate_hz)
    results = _frozen.predict_fill_sequence(
        rgb_frames,
        reference_index=reference_index,
        expected_reference_rgb_sha256=expected_reference_rgb_sha256,
        programmed_ml=programmed_ml,
    )
    if checked_frame_rate_hz == FROZEN_FRAME_RATE_HZ:
        return results

    frames = [np.asarray(frame) for frame in rgb_frames]
    reference = frames[0]
    visible = [
        (
            _frozen.predict_visible_liquid(
                frame, reference, programmed_ml=programmed_ml
            )
            if result.status == "clamp_hidden_inferred"
            else result
        )
        for frame, result in zip(frames, results)
    ]
    stream = [_frozen._stream_detected(frame, reference) for frame in frames]
    minimum_frames = math.ceil(
        _frozen.STREAM_MINIMUM_CONSECUTIVE_FRAMES
        * checked_frame_rate_hz
        / FROZEN_FRAME_RATE_HZ
    )

    def has_upper_body(item: _frozen.EmptyReferencePrediction) -> bool:
        return bool(np.any(item.mask[: _frozen.CLAMP_ROWS[0]]))

    def is_lower_plateau(item: _frozen.EmptyReferencePrediction) -> bool:
        return bool(
            item.status == "visible_measured"
            and item.surface_row is not None
            and _frozen.CLAMP_ROWS[1] - 5
            <= item.surface_row
            <= _frozen.CLAMP_ROWS[1] + 1
            and not has_upper_body(item)
        )

    adjusted = list(visible)
    index = 0
    while index < len(visible):
        if not is_lower_plateau(visible[index]):
            index += 1
            continue
        start = index
        while index < len(visible) and is_lower_plateau(visible[index]):
            index += 1
        stop = index
        plateau_indices = list(range(start, stop))
        plateau_rows = [visible[item].surface_row for item in plateau_indices]
        assert all(row is not None for row in plateau_rows)
        if max(plateau_rows) - min(plateau_rows) > 2:
            continue
        if start < 2 or stop >= len(visible):
            continue
        previous = visible[start - 2]
        before = visible[start - 1]
        after = visible[stop]
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
            _frozen._longest_true_run(
                [stream[item] for item in plateau_indices]
            )
            < minimum_frames
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
            if not _frozen.CLAMP_ROWS[0] <= inferred <= _frozen.CLAMP_ROWS[1]:
                continue
            observed = visible[item]
            rows = np.flatnonzero(np.any(observed.mask, axis=1))
            bottom = int(rows[-1])
            adjusted[item] = _frozen.EmptyReferencePrediction(
                mask=observed.mask,
                status="clamp_hidden_inferred",
                surface_row=inferred,
                visible_height_px=bottom - inferred + 1,
                changed_pixel_count=observed.changed_pixel_count,
                median_rgb=observed.median_rgb,
            )
    return adjusted

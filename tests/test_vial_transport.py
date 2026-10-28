from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.vial_transport import (  # noqa: E402
    TransportSpec,
    _fill_masked_rows,
    compose_transport_frames,
    extract_fillcam_mover_masks,
    transport_offsets,
)


def _synthetic_frame():
    frame = np.full((480, 640, 3), (12, 34, 56), dtype=np.uint8)
    outgoing = np.zeros((480, 640), dtype=np.uint8)
    incoming = np.zeros((480, 640), dtype=np.uint8)
    outgoing[180:260, 240:280] = 255
    incoming[180:260, 430:470] = 255
    frame[outgoing > 0] = (220, 30, 20)
    frame[incoming > 0] = (20, 220, 30)
    return frame, outgoing, incoming


def test_rowwise_background_fill_preserves_unmasked_pixels_and_horizontal_bands():
    frame = np.empty((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        frame[y] = (y % 251, (2 * y) % 251, (3 * y) % 251)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[80:420, 250:390] = 255
    obscured = frame.copy()
    obscured[mask > 0] = (0, 0, 0)

    filled = _fill_masked_rows(obscured, mask)

    assert np.array_equal(filled[mask == 0], obscured[mask == 0])
    assert np.array_equal(filled[mask > 0], frame[mask > 0])


def test_transport_moves_only_foregrounds_over_fixed_background():
    frame, outgoing, incoming = _synthetic_frame()
    spec = TransportSpec(
        frame_count=11,
        fps=10.0,
        outgoing_shift_px=-300,
        incoming_shift_px=-190,
        inpaint_radius_px=3.0,
        feather_radius_px=0,
    )

    frames = compose_transport_frames(
        frame,
        outgoing_mask=outgoing,
        incoming_mask=incoming,
        spec=spec,
    )

    assert len(frames) == 11
    assert np.array_equal(frames[0], frame)
    assert all(tuple(row[10, 10]) == (12, 34, 56) for row in frames)
    assert tuple(frames[-1][200, 250]) == (20, 220, 30)
    assert not np.any(np.all(frames[-1] == (220, 30, 20), axis=2))
    assert np.max(
        np.abs(frames[-1][200, 450].astype(int) - np.asarray((12, 34, 56)))
    ) <= 8
    hashes = {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in frames
    }
    assert len(hashes) == len(frames)


def test_transport_offsets_are_monotonic_and_hit_exact_endpoints():
    spec = TransportSpec(
        frame_count=9,
        outgoing_shift_px=-360,
        incoming_shift_px=-180,
    )
    offsets = transport_offsets(spec)

    assert offsets[0] == (0, 0)
    assert offsets[-1] == (-360, -180)
    assert all(
        later[0] <= earlier[0] and later[1] <= earlier[1]
        for earlier, later in zip(offsets, offsets[1:])
    )
    assert len(set(offsets)) == 9


def test_fillcam_extraction_keeps_liquid_as_outgoing_foreground():
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    frame[210:470, 250:395] = 40
    frame[215:470, 415:575] = 70
    liquid = np.zeros((480, 640), dtype=np.uint8)
    liquid[280:420, 290:345] = 255

    outgoing, incoming = extract_fillcam_mover_masks(frame, liquid)

    assert outgoing.shape == liquid.shape
    assert incoming.shape == liquid.shape
    assert outgoing.dtype == np.uint8
    assert incoming.dtype == np.uint8
    assert np.all(outgoing[liquid > 0] == 255)
    assert outgoing[250, 320] == 255
    assert incoming[300, 480] == 255
    assert np.count_nonzero(outgoing) > np.count_nonzero(liquid)
    assert np.count_nonzero(incoming) > 0
    assert np.count_nonzero((outgoing > 0) & (incoming > 0)) == 0


def test_fillcam_extraction_rejects_dark_camera_background_bands():
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    frame[45:125] = 5
    frame[185:235, 260:400] = 35
    frame[365:480, 260:400] = 70
    frame[185:235, 405:585] = 35
    frame[365:480, 405:585] = 70
    liquid = np.zeros((480, 640), dtype=np.uint8)
    liquid[285:410, 295:350] = 255

    outgoing, incoming = extract_fillcam_mover_masks(frame, liquid)

    assert outgoing[100, 385] == 0
    assert incoming[100, 550] == 0
    assert np.count_nonzero(outgoing) < 65_000
    assert np.count_nonzero(incoming) < 68_000


def test_transport_rejects_bad_frame_or_overlapping_masks():
    frame, outgoing, incoming = _synthetic_frame()
    with pytest.raises(ValueError, match="640x480"):
        compose_transport_frames(
            frame[:400],
            outgoing_mask=outgoing[:400],
            incoming_mask=incoming[:400],
        )
    incoming[200, 250] = 255
    with pytest.raises(ValueError, match="overlap"):
        compose_transport_frames(
            frame,
            outgoing_mask=outgoing,
            incoming_mask=incoming,
        )

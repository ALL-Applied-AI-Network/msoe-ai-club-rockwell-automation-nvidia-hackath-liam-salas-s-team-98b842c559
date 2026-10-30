"""Synthetic-only contracts for the bounded WP7 RGB runtime."""

from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(value: np.ndarray, *, bgr: bool = False) -> str:
    array = value[..., ::-1].copy() if bgr else value
    return hashlib.sha256(array.tobytes()).hexdigest()


def _frame(value: int = 100) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _timing(frame_counts=(5, 7)):
    starts = (0, 17, 36)
    transitions = []
    for index, start in enumerate(starts):
        transitions.append(
            {
                "segment_index": index,
                "start_frame_index": start,
                "stop_frame_index": start + 13,
                "frame_count": 13,
                "decoded_bgr_sha256": [f"{index + 1:064x}"] * 13,
            }
        )
    fills = [
        {"case_id": f"C{index + 1}", "frame_count": count}
        for index, count in enumerate(frame_counts)
    ]
    return transitions, fills


def test_composition_uses_transition_stop_minus_one_and_rejects_mismatch():
    from fluidmix.wp7_rgb_runtime import RGBRuntimeError, compose_fill_windows

    transitions, fills = _timing()
    windows = compose_fill_windows(transitions, fills)

    assert [(item.start_frame_index, item.stop_frame_index) for item in windows] == [
        (12, 17),
        (29, 36),
    ]
    bad = [dict(item) for item in fills]
    bad[0]["frame_count"] = 6
    with pytest.raises(RGBRuntimeError, match="fill frame count mismatch"):
        compose_fill_windows(transitions, bad)


def test_reference_is_exact_composed_first_fill_frame_and_hash_bound():
    from fluidmix.wp7_rgb_runtime import (
        RGBRuntimeError,
        bind_same_run_empty_references,
        compose_fill_windows,
    )

    transitions, fills = _timing()
    frames = [_frame(100 + index % 10) for index in range(50)]
    windows = compose_fill_windows(transitions, fills)
    for window in windows:
        transitions[window.fill_index]["decoded_bgr_sha256"][-1] = _sha(
            frames[window.start_frame_index], bgr=True
        )

    references = bind_same_run_empty_references(frames, transitions, windows)

    assert [item.frame_index for item in references] == [12, 29]
    assert np.array_equal(references[0].rgb, frames[12])
    assert references[0].rgb_sha256 == _sha(frames[12])
    transitions[0]["decoded_bgr_sha256"][-1] = "f" * 64
    with pytest.raises(RGBRuntimeError, match="pre-pour frame hash mismatch"):
        bind_same_run_empty_references(frames, transitions, windows)


def _paint_stream(frame: np.ndarray, value: int = 180) -> None:
    frame[80:82, 300:304] = value


def test_stream_bouts_are_maximal_and_short_or_invalid_inputs_fail_closed():
    from fluidmix.wp7_rgb_runtime import RGBRuntimeError, detect_stream_bouts

    reference = _frame()
    frames = [reference.copy() for _ in range(14)]
    for index in (2, 3, 4, 8, 9, 10, 11):
        _paint_stream(frames[index])

    bouts = detect_stream_bouts(frames, reference, start_frame_index=0, stop_frame_index=14)

    assert [(item.start_frame_index, item.stop_frame_index) for item in bouts] == [
        (2, 5),
        (8, 12),
    ]
    assert detect_stream_bouts(frames[:4], reference, start_frame_index=0, stop_frame_index=4) == ()
    with pytest.raises(RGBRuntimeError, match="uint8 RGB"):
        detect_stream_bouts(
            [np.full((480, 640, 3), np.nan)],
            reference,
            start_frame_index=0,
            stop_frame_index=1,
        )


class _Prediction:
    def __init__(self, surface: int | None, mask: np.ndarray | None = None):
        self.status = "visible_measured" if surface is not None else "empty_mask_abstain"
        self.surface_row = surface
        self.visible_height_px = None if surface is None else 405 - surface
        self.mask = np.zeros((480, 640), dtype=bool) if mask is None else mask


def test_first_stable_post_bout_endpoint_requires_next_frame_agreement():
    from fluidmix.wp7_rgb_runtime import select_dispense_endpoints

    reference = _frame()
    frames = [reference.copy() for _ in range(10)]
    for index in (1, 2, 3):
        _paint_stream(frames[index])
    for index, signal in enumerate((10, 30, 31, 70, 90, 110), start=4):
        frames[index][0, 0] = signal
    mask = np.zeros((480, 640), dtype=bool)
    mask[280:405, 290:330] = True

    def predict(current, _reference):
        signal = int(current[0, 0, 0])
        surfaces = {10: 320, 30: 300, 31: 301, 70: 260, 90: 220, 110: 190}
        return _Prediction(surfaces[signal], mask)

    result = select_dispense_endpoints(
        frames,
        reference,
        dispense_order=("t100",),
        start_frame_index=0,
        stop_frame_index=len(frames),
        predictor=predict,
    )

    assert result.status == "selected"
    assert result.limit_codes == ()
    assert [item.frame_index for item in result.endpoints] == [5]
    assert [item.tank_id for item in result.endpoints] == ["t100"]


def test_missing_ambiguous_bouts_and_unstable_endpoint_abstain_explicitly():
    from fluidmix.wp7_rgb_runtime import select_dispense_endpoints

    reference = _frame()
    missing = select_dispense_endpoints(
        [reference.copy() for _ in range(5)],
        reference,
        dispense_order=("t100",),
        start_frame_index=0,
        stop_frame_index=5,
        predictor=lambda *_: _Prediction(None),
    )
    assert missing.status == "abstain"
    assert missing.limit_codes == ("dispense_stream_bout_missing",)

    two = [reference.copy() for _ in range(12)]
    for index in (1, 2, 3, 7, 8, 9):
        _paint_stream(two[index])
    ambiguous = select_dispense_endpoints(
        two,
        reference,
        dispense_order=("t100",),
        start_frame_index=0,
        stop_frame_index=12,
        predictor=lambda *_: _Prediction(300),
    )
    assert ambiguous.limit_codes == ("dispense_stream_bout_count_ambiguous",)

    one = [reference.copy() for _ in range(9)]
    for index in (1, 2, 3):
        _paint_stream(one[index])
    for index in range(4, 9):
        one[index][0, 0] = index
    unstable_mask = np.zeros((480, 640), dtype=bool)
    unstable_mask[300:405, 290:330] = True
    unstable = select_dispense_endpoints(
        one,
        reference,
        dispense_order=("t100",),
        start_frame_index=0,
        stop_frame_index=9,
        predictor=lambda current, _ref: _Prediction(
            int(current[0, 0, 0]) * 10, unstable_mask
        ),
    )
    assert unstable.limit_codes == ("stable_post_bout_endpoint_ambiguous",)


def test_runtime_boundary_has_no_recipe_volume_or_hidden_band_guessing_route():
    from fluidmix.wp7_rgb_runtime import (
        fail_closed_temporal_hidden_band_adapter,
        select_dispense_endpoints,
    )

    parameters = set(inspect.signature(select_dispense_endpoints).parameters)
    assert not {"programmed_ml", "recipe_ml", "height_hint", "mask_hint"} & parameters
    adapter = fail_closed_temporal_hidden_band_adapter()
    assert adapter.status == "abstain"
    assert adapter.limit_codes == ("cnn_temporal_hidden_band_adapter_unavailable",)


def test_frozen_checkpoint_is_hash_bound_before_loader_and_provenance_is_complete(tmp_path):
    from fluidmix.wp7_rgb_runtime import (
        RGBRuntimeError,
        build_runtime_provenance,
        load_frozen_cnn,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    replay = tmp_path / "replay.json"
    review = tmp_path / "review.md"
    checkpoint.write_bytes(b"synthetic checkpoint")
    replay.write_text("{}", encoding="utf-8")
    review.write_text("review", encoding="utf-8")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls = []

    def loader(**paths):
        calls.append(paths)
        return object(), {"model_sha256": checkpoint_sha}, {"threshold": 0.5}, {}

    loaded = load_frozen_cnn(
        checkpoint_path=checkpoint,
        replay_path=replay,
        review_path=review,
        expected_checkpoint_sha256=checkpoint_sha,
        _loader=loader,
    )
    assert loaded.checkpoint_sha256 == checkpoint_sha
    assert len(calls) == 1
    with pytest.raises(RGBRuntimeError, match="checkpoint SHA-256 mismatch"):
        load_frozen_cnn(
            checkpoint_path=checkpoint,
            replay_path=replay,
            review_path=review,
            expected_checkpoint_sha256="0" * 64,
            _loader=loader,
        )

    provenance = build_runtime_provenance(
        video_sha256="1" * 64,
        receipt_sha256="2" * 64,
        checkpoint_sha256=checkpoint_sha,
        config_sha256="3" * 64,
        vial_records=[
            {
                "case_id": "C1",
                "reference_frame_index": 12,
                "reference_rgb_sha256": "5" * 64,
                "selected_frame_indices": [20],
                "mask_sha256s": ["6" * 64],
                "measurement_citations": ["EVIDENCE-C1-T100"],
            }
        ],
    )
    assert provenance["source_bindings"]["video_sha256"] == "1" * 64
    assert len(provenance["source_bindings"]["runtime_config_sha256"]) == 64
    assert provenance["vials"][0]["selected_frame_indices"] == [20]
    assert provenance["vials"][0]["measurement_citations"] == ["EVIDENCE-C1-T100"]
    assert len(provenance["provenance_sha256"]) == 64
    forbidden = dict(provenance["vials"][0])
    forbidden["oracle_mask"] = "not permitted"
    with pytest.raises(RGBRuntimeError, match="vial provenance fields"):
        build_runtime_provenance(
            video_sha256="1" * 64,
            receipt_sha256="2" * 64,
            checkpoint_sha256=checkpoint_sha,
            config_sha256="3" * 64,
            vial_records=[forbidden],
        )


def test_real_rgb_measurement_capability_adapter_abstains_instead_of_minting_trust():
    from fluidmix.wp7_rgb_runtime import adapt_real_rgb_measurements

    result = adapt_real_rgb_measurements()

    assert result.status == "abstain"
    assert result.limit_codes == ("trusted_real_rgb_measurement_adapter_unavailable",)

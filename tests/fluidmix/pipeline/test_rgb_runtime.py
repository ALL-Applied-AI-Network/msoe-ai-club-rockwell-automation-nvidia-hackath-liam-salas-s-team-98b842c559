"""Synthetic-only contracts for the bounded WP7 RGB runtime."""

from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import json
import sys

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _sha(value: np.ndarray, *, bgr: bool = False) -> str:
    array = value[..., ::-1].copy() if bgr else value
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    from fluidmix.pipeline.rgb_runtime import RGBRuntimeError, compose_fill_windows

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
    from fluidmix.pipeline.rgb_runtime import (
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
    from fluidmix.pipeline.rgb_runtime import RGBRuntimeError, detect_stream_bouts

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
    from fluidmix.pipeline.rgb_runtime import select_dispense_endpoints

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

    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    assert result.endpoints[0].mask_sha256 == hashlib.sha256(buffer.getvalue()).hexdigest()


def test_missing_ambiguous_bouts_and_unstable_endpoint_abstain_explicitly():
    from fluidmix.pipeline.rgb_runtime import select_dispense_endpoints

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
    from fluidmix.pipeline.rgb_runtime import (
        build_runtime_vision_measurements,
        select_dispense_endpoints,
    )

    parameters = set(inspect.signature(select_dispense_endpoints).parameters)
    assert not {"programmed_ml", "recipe_ml", "height_hint", "mask_hint"} & parameters
    measurement_parameters = set(
        inspect.signature(build_runtime_vision_measurements).parameters
    )
    assert not {"recipe_ml", "height_hint", "mask_hint"} & measurement_parameters
    assert "programmed_ml" in measurement_parameters


def test_frozen_checkpoint_and_provenance_require_live_capabilities(tmp_path, monkeypatch):
    from fluidmix.pipeline.rgb_runtime import (
        FrozenCNNPredictedMaskRuntime,
        RGBRuntimeError,
        SelectionResult,
        SelectedEndpoint,
        StreamBout,
        adapt_real_rgb_measurements,
        build_runtime_provenance,
        load_frozen_cnn,
    )
    from fluidmix.measurement.vision_measurement import (
        build_trusted_runtime_measured_evidence_v1,
        trusted_measurement_bindings_from_result_v1,
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
        return object(), {"model_sha256": checkpoint_sha}, {"probability_threshold": 0.5}, {}

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

    reference = _frame()
    visible = reference.copy()
    mask = _body_mask(300, 405)
    _paint_mask(visible, mask)
    artifact_root = tmp_path / "capabilities"
    artifact_root.mkdir()
    artifact = _real_registry_artifact(
        artifact_root,
        monkeypatch,
        [reference, visible, visible.copy()],
        [np.zeros_like(mask), mask, mask.copy()],
    )
    runtime_cnn = _loaded_test_cnn(artifact, monkeypatch)
    registry = adapt_real_rgb_measurements(artifact, runtime_cnn)
    measured = build_trusted_runtime_measured_evidence_v1(
        registry,
        {"IMAGE-001": {"measurement_id": "MEAS-1", "evidence_id": "EVID-1"}},
    )
    selection = SelectionResult(
        "selected",
        (StreamBout(100, 101, 1),),
        (
            SelectedEndpoint(
                bout_index=0,
                tank_id="t100",
                frame_index=101,
                agreement_frame_index=102,
                mask_sha256=registry.entry("IMAGE-001").mask_sha256,
                agreement_mask_sha256=registry.entry("IMAGE-002").mask_sha256,
                surface_row=300,
                visible_height_px=105,
            ),
        ),
        (),
    )
    capture_receipt = {
        "schema_version": "wp7-camera-capture-v1",
        "artifact_type": "wp7_camera_capture",
        "capture_origin": "real_camera",
        "video_sha256": "1" * 64,
        "encoded_transition_evidence": [],
        "fill_clips": [],
    }
    import fluidmix.pipeline.rgb_runtime as runtime_module

    capture_receipt["receipt_sha256"] = runtime_module._canonical_sha256(
        capture_receipt
    )
    frozen_runtime = FrozenCNNPredictedMaskRuntime(runtime_cnn, capture_receipt)
    provenance = build_runtime_provenance(
        runtime=frozen_runtime,
        registry=registry,
        measured_evidence=measured,
        endpoint_selections={"CASE-1": selection},
    )
    assert provenance["source_bindings"]["video_sha256"] == "1" * 64
    assert len(provenance["source_bindings"]["runtime_config_sha256"]) == 64
    assert provenance["vials"][0]["selected_frame_indices"] == [101]
    assert provenance["vials"][0]["agreement_frame_indices"] == [102]
    assert provenance["vials"][0]["measurement_citation_sha256s"] == [
        trusted_measurement_bindings_from_result_v1(measured)[0].measurement_sha256
    ]
    assert len(provenance["provenance_sha256"]) == 64
    with pytest.raises(RGBRuntimeError, match="endpoint abstention is invalid"):
        build_runtime_provenance(
            runtime=frozen_runtime,
            registry=registry,
            measured_evidence=measured,
            endpoint_selections={
                "CASE-1": SelectionResult(
                    "abstain", (), (), ("unreviewed_endpoint_failure",)
                )
            },
        )

    second_capture_receipt = dict(capture_receipt)
    second_capture_receipt["video_sha256"] = "2" * 64
    second_capture_receipt["receipt_sha256"] = runtime_module._canonical_sha256(
        {
            key: value
            for key, value in second_capture_receipt.items()
            if key != "receipt_sha256"
        }
    )
    second_runtime = FrozenCNNPredictedMaskRuntime(runtime_cnn, second_capture_receipt)
    second_provenance = build_runtime_provenance(
        runtime=second_runtime,
        registry=registry,
        measured_evidence=measured,
        endpoint_selections={"CASE-1": selection},
    )
    assert frozen_runtime.config_sha256 == second_runtime.config_sha256
    assert provenance["source_bindings"]["receipt_sha256"] != second_provenance[
        "source_bindings"
    ]["receipt_sha256"]
    assert provenance["source_bindings"]["video_sha256"] != second_provenance[
        "source_bindings"
    ]["video_sha256"]
    assert provenance["provenance_sha256"] != second_provenance["provenance_sha256"]

    mismatched_receipt = dict(second_capture_receipt)
    mismatched_receipt["video_sha256"] = "3" * 64
    with pytest.raises(RGBRuntimeError, match="capture receipt SHA-256 mismatch"):
        FrozenCNNPredictedMaskRuntime(runtime_cnn, mismatched_receipt)
    with pytest.raises(TypeError):
        build_runtime_provenance(
            runtime="detached",
            checkpoint_sha256=checkpoint_sha,
            config_sha256="3" * 64,
            vial_records=[{"measurement_citations": ["detached"]}],
        )


def _write_rgb(path: Path, frame: np.ndarray) -> str:
    Image.fromarray(frame, mode="RGB").save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mask(path: Path, mask: np.ndarray) -> str:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_registry_artifact(
    tmp_path, monkeypatch, frames, masks, *, origin="real_camera"
):
    import fluidmix.pipeline.rgb_runtime as runtime

    lineage = {}
    for role in ("checkpoint", "replay", "review"):
        path = tmp_path / f"cnn-{role}.bin"
        path.write_bytes(f"synthetic-{role}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setattr(runtime, f"CNN_{role.upper()}_SHA256", digest)
        lineage[role] = {"path": str(path), "sha256": digest}
    reference_path = tmp_path / "same-run-prepour.png"
    reference_sha = _write_rgb(reference_path, frames[0])
    images = {}
    for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        rgb_path = tmp_path / f"frame-{index:03d}.png"
        mask_path = tmp_path / f"mask-{index:03d}.png"
        images[f"IMAGE-{index:03d}"] = {
            "case_id": "CASE-1",
            "frame_index": 100 + index,
            "current_rgb_path": str(rgb_path),
            "current_rgb_sha256": _write_rgb(rgb_path, frame),
            "reference_frame_index": 100,
            "reference_rgb_path": str(reference_path),
            "reference_rgb_sha256": reference_sha,
            "mask_path": str(mask_path),
            "mask_sha256": _write_mask(mask_path, mask),
        }
    accepted = runtime._accepted_cnn_lineage()
    return {
        "schema_version": "wp7-camera-rgb-mask-registry-v1",
        "artifact_type": "camera_rgb_mask_registry",
        "data_origin": origin,
        "producer": {
            "producer_id": "frozen-wp5-six-channel-cnn",
            "checkpoint_sha256": accepted["checkpoint"],
            "replay_sha256": accepted["replay"],
            "review_sha256": accepted["review"],
            "probability_threshold": 0.5,
            "input_fields": ["current_rgb", "same_run_pre_pour_rgb"],
            "mask_source": "frozen_cnn_prediction",
        },
        "lineage": lineage,
        "images": images,
    }


def _loaded_test_cnn(artifact, monkeypatch):
    import fluidmix.pipeline.rgb_runtime as runtime

    checkpoint = Path(artifact["lineage"]["checkpoint"]["path"])
    replay = Path(artifact["lineage"]["replay"]["path"])
    review = Path(artifact["lineage"]["review"]["path"])
    checkpoint_sha = artifact["lineage"]["checkpoint"]["sha256"]

    def loader(**_paths):
        return (
            object(),
            {"model_sha256": checkpoint_sha},
            {"probability_threshold": 0.5},
            {},
        )

    loaded = runtime.load_frozen_cnn(
        checkpoint_path=checkpoint,
        replay_path=replay,
        review_path=review,
        expected_checkpoint_sha256=checkpoint_sha,
        _loader=loader,
    )

    def predict(_loaded, current, reference):
        from fluidmix.perception.baselines.empty_reference_rgb_predictor import fillcam_masks

        roi, clamp = fillcam_masks()
        mask = np.zeros((480, 640), dtype=bool)
        if np.any(current):
            mask = np.any(current != reference, axis=2) & roi & ~clamp
        rows = np.flatnonzero(np.any(mask, axis=1))
        surface = int(rows[0]) if rows.size else None
        return _Prediction(surface, mask)

    monkeypatch.setattr(runtime, "predict_with_frozen_cnn", predict)
    return loaded


def test_real_rgb_registry_is_distinct_hash_bound_and_rejects_oracle_provenance(
    tmp_path, monkeypatch
):
    from fluidmix.pipeline.rgb_runtime import RGBRuntimeError, adapt_real_rgb_measurements
    from fluidmix.pipeline.rgb_runtime import TrustedRealRGBMaskRegistryV1

    frame = _frame()
    mask = np.zeros((480, 640), dtype=bool)
    artifact = _real_registry_artifact(tmp_path, monkeypatch, [frame], [mask])

    loaded = _loaded_test_cnn(artifact, monkeypatch)
    registry = adapt_real_rgb_measurements(artifact, loaded)

    assert registry.image_ids == ("IMAGE-000",)
    assert registry.entry("IMAGE-000").frame_index == 100
    with pytest.raises(TypeError, match="factory-only"):
        TrustedRealRGBMaskRegistryV1({}, "0" * 64)
    forbidden = dict(artifact)
    forbidden["oracle_mask"] = "forbidden"
    with pytest.raises(RGBRuntimeError, match="forbidden real-RGB provenance field"):
        adapt_real_rgb_measurements(forbidden, loaded)
    artifact["lineage"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(RGBRuntimeError, match="lineage hash is not accepted"):
        adapt_real_rgb_measurements(artifact, loaded)


@pytest.mark.parametrize("origin", ["synthetic_camera", "real_camera"])
def test_camera_registry_preserves_truthful_origin_outside_model_inputs(
    tmp_path, monkeypatch, origin
):
    from fluidmix.pipeline.rgb_runtime import adapt_real_rgb_measurements

    frame = _frame()
    artifact = _real_registry_artifact(
        tmp_path,
        monkeypatch,
        [frame],
        [np.zeros((480, 640), dtype=bool)],
        origin=origin,
    )
    registry = adapt_real_rgb_measurements(
        artifact, _loaded_test_cnn(artifact, monkeypatch)
    )

    assert registry.image_ids == ("IMAGE-000",)
    assert registry.data_origin == origin
    assert "origin" not in registry.entry("IMAGE-000").locator()


def test_camera_registry_origin_changes_capability_and_must_match_runtime(
    tmp_path, monkeypatch
):
    from fluidmix.measurement.vision_measurement import build_trusted_runtime_measured_evidence_v1
    from fluidmix.pipeline.rgb_runtime import (
        FrozenCNNPredictedMaskRuntime,
        RGBRuntimeError,
        SelectionResult,
        SelectedEndpoint,
        StreamBout,
        adapt_real_rgb_measurements,
        build_runtime_provenance,
    )

    frame = _frame()
    mask = np.zeros((480, 640), dtype=bool)
    synthetic_root = tmp_path / "synthetic"
    real_root = tmp_path / "real"
    synthetic_root.mkdir()
    real_root.mkdir()
    synthetic_artifact = _real_registry_artifact(
        synthetic_root, monkeypatch, [frame], [mask], origin="synthetic_camera"
    )
    loaded = _loaded_test_cnn(synthetic_artifact, monkeypatch)
    synthetic_registry = adapt_real_rgb_measurements(synthetic_artifact, loaded)
    real_artifact = dict(synthetic_artifact)
    real_artifact["data_origin"] = "real_camera"
    real_registry = adapt_real_rgb_measurements(real_artifact, loaded)
    assert synthetic_registry.registry_sha256 != real_registry.registry_sha256

    entry = real_registry.entry("IMAGE-000")
    measured = build_trusted_runtime_measured_evidence_v1(
        real_registry,
        {"IMAGE-000": {"measurement_id": "MEAS-1", "evidence_id": "EVID-1"}},
    )
    selection = SelectionResult(
        "selected",
        (StreamBout(99, 100, 1),),
        (SelectedEndpoint(0, "t100", 100, 100, entry.mask_sha256, entry.mask_sha256, 300, 105),),
        (),
    )
    capture = {
        "schema_version": "wp7-camera-capture-v1",
        "artifact_type": "wp7_camera_capture",
        "capture_origin": "synthetic_camera",
        "video_sha256": "1" * 64,
        "encoded_transition_evidence": [],
        "fill_clips": [],
    }
    capture["receipt_sha256"] = _canonical_hash(capture)
    runtime = FrozenCNNPredictedMaskRuntime(loaded, capture)
    with pytest.raises(RGBRuntimeError, match="origin"):
        build_runtime_provenance(
            runtime=runtime,
            registry=real_registry,
            measured_evidence=measured,
            endpoint_selections={"CASE-1": selection},
        )


def test_camera_registry_rejects_wrong_caller_mask_and_black_reference(
    tmp_path, monkeypatch
):
    from fluidmix.pipeline.rgb_runtime import RGBRuntimeError, adapt_real_rgb_measurements

    reference = _frame()
    changed = reference.copy()
    body = _body_mask(300, 405)
    _paint_mask(changed, body)
    wrong_root = tmp_path / "wrong-mask"
    wrong_root.mkdir()
    wrong = _real_registry_artifact(
        wrong_root,
        monkeypatch,
        [reference, changed],
        [np.zeros_like(body), np.zeros_like(body)],
    )
    with pytest.raises(RGBRuntimeError, match="differs from frozen CNN"):
        adapt_real_rgb_measurements(wrong, _loaded_test_cnn(wrong, monkeypatch))

    black_root = tmp_path / "black-reference"
    black_root.mkdir()
    black = np.zeros_like(reference)
    black_artifact = _real_registry_artifact(
        black_root,
        monkeypatch,
        [black, changed],
        [np.zeros_like(body), body],
    )
    with pytest.raises(RGBRuntimeError, match="pre-pour reference is black"):
        adapt_real_rgb_measurements(
            black_artifact, _loaded_test_cnn(black_artifact, monkeypatch)
        )


def _body_mask(row_start: int, row_stop: int) -> np.ndarray:
    from fluidmix.perception.baselines.empty_reference_rgb_predictor import fillcam_masks

    roi, clamp = fillcam_masks()
    rows = np.zeros(roi.shape, dtype=bool)
    rows[row_start:row_stop] = True
    return roi & rows & ~clamp


def _paint_mask(frame: np.ndarray, mask: np.ndarray, value: int = 180) -> None:
    frame[mask] = value


def test_runtime_measurement_uses_accepted_hidden_band_inference_and_is_ml_invariant(
    tmp_path, monkeypatch
):
    from fluidmix.pipeline.rgb_runtime import (
        adapt_real_rgb_measurements,
        build_runtime_vision_measurements,
    )

    reference = _frame()
    frames = [reference.copy()]
    masks = [np.zeros((480, 640), dtype=bool)]
    for start in (310, 280):
        mask = _body_mask(start, 405)
        frame = reference.copy()
        _paint_mask(frame, mask)
        frames.append(frame)
        masks.append(mask)
    for _ in range(5):
        mask = _body_mask(271, 405)
        frame = reference.copy()
        _paint_stream(frame)
        _paint_mask(frame, mask)
        frames.append(frame)
        masks.append(mask)
    final_mask = _body_mask(194, 238) | _body_mask(271, 405)
    final_frame = reference.copy()
    _paint_mask(final_frame, final_mask)
    frames.append(final_frame)
    masks.append(final_mask)
    artifact = _real_registry_artifact(tmp_path, monkeypatch, frames, masks)
    registry = adapt_real_rgb_measurements(
        artifact, _loaded_test_cnn(artifact, monkeypatch)
    )

    low = build_runtime_vision_measurements(
        registry, registry.image_ids, programmed_ml=10.0
    )
    high = build_runtime_vision_measurements(
        registry, registry.image_ids, programmed_ml=70.0
    )

    assert low.measurements == high.measurements
    inferred = [item for item in low.measurements if item.status == "clamp_hidden_inferred"]
    assert inferred
    assert all(238 <= item.surface_row <= 270 for item in inferred)
    assert all(item.mask_sha256 for item in inferred)


def test_runtime_measurement_black_empty_and_untrusted_controls_fail_closed(
    tmp_path, monkeypatch
):
    from fluidmix.pipeline.rgb_runtime import (
        RGBRuntimeError,
        adapt_real_rgb_measurements,
        build_runtime_vision_measurements,
    )

    reference = _frame()
    frames = [reference.copy(), reference.copy(), np.zeros_like(reference)]
    masks = [np.zeros((480, 640), dtype=bool) for _ in frames]
    artifact = _real_registry_artifact(tmp_path, monkeypatch, frames, masks)
    registry = adapt_real_rgb_measurements(
        artifact, _loaded_test_cnn(artifact, monkeypatch)
    )

    result = build_runtime_vision_measurements(registry, registry.image_ids)

    assert result.measurements[0].status == "empty_reference"
    assert result.measurements[1].status == "empty_mask_abstain"
    assert result.measurements[2].status == "invalid_black_frame"
    with pytest.raises(TypeError, match="trusted real-RGB registry"):
        build_runtime_vision_measurements(object(), registry.image_ids)

    invalid_root = tmp_path / "invalid-mask"
    invalid_root.mkdir()
    invalid_mask = np.zeros((480, 640), dtype=bool)
    invalid_mask[244, 300] = True  # fixed clamp raster
    invalid_artifact = _real_registry_artifact(
        invalid_root,
        monkeypatch,
        [reference.copy()],
        [invalid_mask],
    )
    with pytest.raises(RGBRuntimeError, match="differs from frozen CNN"):
        adapt_real_rgb_measurements(
            invalid_artifact, _loaded_test_cnn(invalid_artifact, monkeypatch)
        )


@pytest.mark.parametrize("origin", ["synthetic_camera", "real_camera"])
@pytest.mark.parametrize(
    ("source_role", "evidence_role"),
    [("test", "test"), ("real_check", "real_validation")],
    ids=["test-partition", "real-check-partition"],
)
@pytest.mark.parametrize(
    "unavailable_cases",
    [set(), {"CASE-1"}, {"CASE-1", "CASE-2"}],
    ids=["all-selected", "mixed-endpoint-unavailable", "all-endpoint-unavailable"],
)
def test_concrete_frozen_runtime_projects_and_ingests_camera_measurements(
    tmp_path, monkeypatch, origin, source_role, evidence_role, unavailable_cases
):
    from fluidmix.pipeline.carryover_pipeline import (
        append_runtime_measurements,
        project_runtime_measurements,
    )
    from fluidmix.pipeline.rgb_runtime import FrozenCNNPredictedMaskRuntime

    reference = _frame()
    frames = [reference.copy() for _ in range(77)]
    lower = _body_mask(300, 405)
    upper = _body_mask(194, 238) | _body_mask(271, 405)
    for case_id, offset in (("CASE-1", 0), ("CASE-2", 32)):
        if case_id in unavailable_cases:
            continue
        for index in (13, 14, 15, 21, 22, 23):
            _paint_stream(frames[index + offset])
        for index in range(16, 21):
            _paint_mask(frames[index + offset], lower)
        for index in range(24, 32):
            _paint_mask(frames[index + offset], upper)
    transitions = [
        {
            "segment_index": 0,
            "start_frame_index": 0,
            "stop_frame_index": 13,
            "frame_count": 13,
            "decoded_bgr_sha256": [_sha(frame, bgr=True) for frame in frames[0:13]],
        },
        {
            "segment_index": 1,
            "start_frame_index": 32,
            "stop_frame_index": 45,
            "frame_count": 13,
            "decoded_bgr_sha256": [_sha(frame, bgr=True) for frame in frames[32:45]],
        },
        {
            "segment_index": 2,
            "start_frame_index": 64,
            "stop_frame_index": 77,
            "frame_count": 13,
            "decoded_bgr_sha256": [_sha(frame, bgr=True) for frame in frames[64:77]],
        },
    ]
    capture = {
        "schema_version": "wp7-camera-capture-v1",
        "artifact_type": "wp7_camera_capture",
        "capture_origin": origin,
        "video_sha256": hashlib.sha256(b"camera bytes").hexdigest(),
        "encoded_transition_evidence": transitions,
        "fill_clips": [
            {"case_id": "CASE-1", "frame_count": 20},
            {"case_id": "CASE-2", "frame_count": 20},
        ],
    }
    capture["receipt_sha256"] = _canonical_hash(capture)
    lineage_root = tmp_path / "lineage"
    lineage_root.mkdir()
    seed_artifact = _real_registry_artifact(
        lineage_root,
        monkeypatch,
        [reference],
        [np.zeros((480, 640), dtype=bool)],
        origin=origin,
    )
    loaded = _loaded_test_cnn(seed_artifact, monkeypatch)
    runtime = FrozenCNNPredictedMaskRuntime(
        loaded,
        capture,
        _frame_decoder=lambda _raw: frames,
    )
    sequence = {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": source_role,
        "split_version": "fixture-v1",
        "runs": [
            {
                "run_id": "RUN-1",
                "vials": [
                    {
                        "case_id": "CASE-1",
                        "vial_ordinal": 1,
                        "block_recipe_id": "R-1",
                        "block_position": 1,
                        "recipe_id": "R-1",
                        "previous_recipe_id": "R-0",
                        "programmed_tank_ml": {
                            "t100": 10.0,
                            "t200": 20.0,
                            "t400": 0.0,
                        },
                        "dispense_order": ["t100", "t200"],
                    },
                    {
                        "case_id": "CASE-2",
                        "vial_ordinal": 2,
                        "block_recipe_id": "R-1",
                        "block_position": 2,
                        "recipe_id": "R-1",
                        "previous_recipe_id": "R-1",
                        "programmed_tank_ml": {
                            "t100": 10.0,
                            "t200": 20.0,
                            "t400": 0.0,
                        },
                        "dispense_order": ["t100", "t200"],
                    },
                ],
            }
        ],
    }
    artifacts = tmp_path / "artifacts"

    output = runtime.analyze(
        video_bytes=b"camera bytes",
        recipe_sequence=sequence,
        artifact_root=artifacts,
    )
    assert {
        template["split_role"] for template in output.record_templates
    } <= {evidence_role}
    projected = project_runtime_measurements(output)
    ingested = None
    if projected:
        ingested = append_runtime_measurements(
            store=tmp_path / "evidence.sqlite",
            artifact_root=artifacts,
            records=projected,
            runtime_output=output,
        )

    selected_cases = [
        case_id for case_id in ("CASE-1", "CASE-2")
        if case_id not in unavailable_cases
    ]
    assert len(projected) == 2 * len(selected_cases)
    assert [row["case_id"] for row in projected] == [
        case_id for case_id in selected_cases for _ in range(2)
    ]
    assert [row["tank_id"] for row in projected] == [
        tank for _case_id in selected_cases for tank in ("t100", "t200")
    ]
    assert (None if ingested is None else ingested.inserted) == (
        None if not projected else len(projected)
    )
    assert output.runtime_provenance["source_bindings"]["capture_origin"] == origin
    assert [vial["case_id"] for vial in output.runtime_provenance["vials"]] == [
        "CASE-1",
        "CASE-2",
    ]
    provenance_by_case = {
        vial["case_id"]: vial for vial in output.runtime_provenance["vials"]
    }
    for case_id in unavailable_cases:
        assert provenance_by_case[case_id]["selection_status"] == "abstain"
        assert provenance_by_case[case_id]["limit_codes"] == [
            "dispense_stream_bout_missing"
        ]
        assert provenance_by_case[case_id]["selected_frame_indices"] == []
        assert provenance_by_case[case_id]["measurement_citation_sha256s"] == []
    from fluidmix.measurement.vision_measurement import (
        trusted_measurement_bindings_from_result_v1,
    )
    assert len(trusted_measurement_bindings_from_result_v1(
        output.trusted_adapter_result
    )) == len(projected)
    if len(unavailable_cases) == 2:
        assert output.record_templates == ()
        assert output.image_locators == {}

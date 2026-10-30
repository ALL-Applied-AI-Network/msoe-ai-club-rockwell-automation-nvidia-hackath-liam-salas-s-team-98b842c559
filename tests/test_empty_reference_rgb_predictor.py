"""Deterministic same-vial empty-reference RGB predictor contracts."""

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEVELOPMENT_INPUTS = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "rgb-mask-predictor-01a18128/approach-a-stage2-partition-fix-attempt-01a18128/"
    "development-inputs-v1.json"
)
DEVELOPMENT_LOCATORS = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "rgb-mask-predictor-01a18128/approach-a-stage2-partition-fix-attempt-01a18128/"
    "development-locators-v1.json"
)


def _reference(value: int = 100) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def _paint_body(image: np.ndarray, row_start: int, row_stop: int, value: int) -> None:
    from fluidmix.empty_reference_rgb_predictor import fillcam_masks

    roi, _clamp = fillcam_masks()
    rows = np.zeros(roi.shape, dtype=bool)
    rows[row_start:row_stop] = True
    image[roi & rows] = value


def test_visible_stage_one_rejects_nozzle_and_reports_changed_body_colour():
    from fluidmix.empty_reference_rgb_predictor import predict_visible_liquid

    reference = _reference()
    current = reference.copy()
    current[80:180, 300:320] = (230, 20, 10)  # falling stream/nozzle region
    _paint_body(current, 271, 405, 170)

    result = predict_visible_liquid(current, reference, programmed_ml=10.0)

    assert result.status == "visible_measured"
    assert result.surface_row == 271
    assert result.visible_height_px == 134
    assert result.changed_pixel_count == int(np.count_nonzero(result.mask))
    assert not np.any(result.mask[:194])
    assert result.median_rgb == (170.0, 170.0, 170.0)


def test_stage_two_retains_dense_regions_above_and_below_clamp_only():
    from fluidmix.empty_reference_rgb_predictor import fillcam_masks, predict_visible_liquid

    reference = _reference()
    current = reference.copy()
    _paint_body(current, 194, 238, 180)
    _paint_body(current, 271, 405, 180)

    result = predict_visible_liquid(current, reference, programmed_ml=70.0)
    _roi, clamp = fillcam_masks()

    assert result.status == "visible_measured"
    assert result.surface_row == 194
    assert np.any(result.mask[194:238])
    assert np.any(result.mask[271:405])
    assert not np.any(result.mask & clamp)


def test_stage_two_aggregates_disconnected_dense_change_above_clamp():
    """Glass/refraction gaps must not split one strong upper liquid region away."""

    from fluidmix.empty_reference_rgb_predictor import predict_visible_liquid

    reference = _reference()
    current = reference.copy()
    _paint_body(current, 194, 202, 180)
    _paint_body(current, 210, 226, 180)
    _paint_body(current, 271, 405, 180)

    result = predict_visible_liquid(current, reference, programmed_ml=70.0)

    assert result.status == "visible_measured"
    assert result.surface_row == 194
    assert np.any(result.mask[194:202])
    assert np.any(result.mask[210:226])


def test_empty_different_frame_sparse_change_and_black_frame_abstain():
    from fluidmix.empty_reference_rgb_predictor import predict_visible_liquid

    reference = _reference()
    later_empty = reference.copy()
    later_empty[390:394, 300:306] = 160

    empty_result = predict_visible_liquid(later_empty, reference, programmed_ml=50.0)
    black_result = predict_visible_liquid(
        np.zeros_like(reference), reference, programmed_ml=50.0
    )

    assert empty_result.status == "insufficient_dense_change"
    assert empty_result.surface_row is None
    assert not np.any(empty_result.mask)
    assert black_result.status == "invalid_black_frame"
    assert black_result.surface_row is None
    assert not np.any(black_result.mask)


def test_visible_output_is_invariant_to_programmed_ml():
    from fluidmix.empty_reference_rgb_predictor import predict_visible_liquid

    reference = _reference()
    current = reference.copy()
    _paint_body(current, 271, 405, 180)

    low = predict_visible_liquid(current, reference, programmed_ml=10.0)
    high = predict_visible_liquid(current, reference, programmed_ml=70.0)

    assert np.array_equal(low.mask, high.mask)
    assert low.surface_row == high.surface_row
    assert low.visible_height_px == high.visible_height_px
    assert low.median_rgb == high.median_rgb


def test_hidden_band_requires_pre_growth_sustained_rgb_stream_and_visible_exit():
    from fluidmix.empty_reference_rgb_predictor import predict_fill_sequence

    reference = _reference()
    before_one = reference.copy()
    before_two = reference.copy()
    _paint_body(before_one, 310, 405, 180)
    _paint_body(before_two, 280, 405, 180)
    hidden = []
    for _ in range(5):
        frame = reference.copy()
        frame[80:190, 300:320] = 180
        _paint_body(frame, 271, 405, 180)
        hidden.append(frame)
    after = reference.copy()
    _paint_body(after, 194, 238, 180)
    _paint_body(after, 271, 405, 180)

    results = predict_fill_sequence(
        [reference, before_one, before_two, *hidden, after],
        reference_index=0,
        expected_reference_rgb_sha256=_array_sha256(reference),
        programmed_ml=70.0,
    )

    inferred = [item for item in results if item.status == "clamp_hidden_inferred"]
    assert inferred
    assert all(238 <= item.surface_row <= 270 for item in inferred)
    assert all(np.any(item.mask) for item in inferred)
    assert all(item.median_rgb == (180.0, 180.0, 180.0) for item in inferred)
    assert all(item.changed_pixel_count == np.count_nonzero(item.mask) for item in inferred)
    assert results[-1].status == "visible_measured"


def test_hidden_band_does_not_infer_from_short_stream_or_without_exit():
    from fluidmix.empty_reference_rgb_predictor import predict_fill_sequence

    reference = _reference()
    before_one = reference.copy()
    before_two = reference.copy()
    _paint_body(before_one, 310, 405, 180)
    _paint_body(before_two, 280, 405, 180)
    stream = reference.copy()
    stream[80:190, 300:320] = 180
    _paint_body(stream, 271, 405, 180)
    exit_frame = reference.copy()
    _paint_body(exit_frame, 194, 238, 180)
    _paint_body(exit_frame, 271, 405, 180)

    short = predict_fill_sequence(
        [reference, before_one, before_two, stream, stream.copy(), exit_frame],
        reference_index=0,
        expected_reference_rgb_sha256=_array_sha256(reference),
        programmed_ml=70.0,
    )
    no_exit = predict_fill_sequence(
        [reference, before_one, before_two, stream, stream.copy(), stream.copy()],
        reference_index=0,
        expected_reference_rgb_sha256=_array_sha256(reference),
        programmed_ml=70.0,
    )

    assert not any(item.status == "clamp_hidden_inferred" for item in short)
    assert not any(item.status == "clamp_hidden_inferred" for item in no_exit)


def test_hidden_inference_is_programmed_ml_invariant_with_observed_lower_mask():
    from fluidmix.empty_reference_rgb_predictor import predict_fill_sequence

    reference = _reference()
    before_one = reference.copy()
    before_two = reference.copy()
    _paint_body(before_one, 310, 405, 180)
    _paint_body(before_two, 280, 405, 180)
    plateau = []
    for _ in range(5):
        frame = reference.copy()
        frame[80:190, 300:320] = 180
        _paint_body(frame, 271, 405, 180)
        plateau.append(frame)
    exit_frame = reference.copy()
    _paint_body(exit_frame, 194, 238, 180)
    _paint_body(exit_frame, 271, 405, 180)
    frames = [reference, before_one, before_two, *plateau, exit_frame]

    digest = _array_sha256(reference)
    low = predict_fill_sequence(
        frames, expected_reference_rgb_sha256=digest, programmed_ml=10.0
    )
    high = predict_fill_sequence(
        frames, expected_reference_rgb_sha256=digest, programmed_ml=70.0
    )

    assert [item.status for item in low] == [item.status for item in high]
    assert [item.surface_row for item in low] == [item.surface_row for item in high]
    assert [item.visible_height_px for item in low] == [
        item.visible_height_px for item in high
    ]
    assert all(np.array_equal(a.mask, b.mask) for a, b in zip(low, high, strict=True))


def test_sequence_rejects_nonempty_unchanged_reference_against_empty_binding():
    from fluidmix.empty_reference_rgb_predictor import predict_fill_sequence
    import pytest

    empty = _reference()
    nonempty = empty.copy()
    _paint_body(nonempty, 271, 405, 180)

    with pytest.raises(ValueError, match="bound reference hash"):
        predict_fill_sequence(
            [nonempty, nonempty.copy()],
            expected_reference_rgb_sha256=_array_sha256(empty),
            programmed_ml=50.0,
        )


def test_hidden_inference_rejects_interrupted_outside_roi_stream_and_wrong_growth():
    from fluidmix.empty_reference_rgb_predictor import predict_fill_sequence

    reference = _reference()
    growing_one = reference.copy()
    growing_two = reference.copy()
    _paint_body(growing_one, 310, 405, 180)
    _paint_body(growing_two, 280, 405, 180)
    exit_frame = reference.copy()
    _paint_body(exit_frame, 194, 238, 180)
    _paint_body(exit_frame, 271, 405, 180)

    def plateau(stream_location: str, enabled: bool = True) -> np.ndarray:
        frame = reference.copy()
        _paint_body(frame, 271, 405, 180)
        if enabled and stream_location == "inside":
            frame[80:190, 300:320] = 180
        elif enabled:
            frame[80:190, 100:120] = 180
        return frame

    cases = [
        [
            reference,
            growing_one,
            growing_two,
            plateau("inside"),
            plateau("inside", False),
            plateau("inside"),
            plateau("inside", False),
            plateau("inside"),
            exit_frame,
        ],
        [
            reference,
            growing_one,
            growing_two,
            *[plateau("outside") for _ in range(5)],
            exit_frame,
        ],
        [
            reference,
            growing_two,
            growing_one,
            *[plateau("inside") for _ in range(5)],
            exit_frame,
        ],
    ]

    for frames in cases:
        results = predict_fill_sequence(
            frames,
            expected_reference_rgb_sha256=_array_sha256(reference),
            programmed_ml=70.0,
        )
        assert not any(item.status == "clamp_hidden_inferred" for item in results)


def test_development_publisher_uses_bound_prefill_not_step12_video_frame(tmp_path):
    """The first emitted video frame is already in-pour and cannot be the reference."""

    output = tmp_path / "predictions.json"
    result = subprocess.run(
        [
            "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python",
            str(ROOT / "scripts" / "publish_empty_reference_development.py"),
            "--safe-manifest",
            str(DEVELOPMENT_INPUTS),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode in {0, 2}, result.stderr
    artifact = json.loads(output.read_bytes())
    for record in artifact["development_records"]:
        assert record["reference_source"] == "pre_pour_parity.f2_prefill_path"
        assert record["video_frame_zero_physics_step"] == 12
        assert record["reference_rgb_array_sha256"] != record["video_frame_zero_rgb_sha256"]


def test_development_publisher_accepts_only_validation_partition_artifact(tmp_path):
    output = tmp_path / "predictions.json"
    result = subprocess.run(
        [
            "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python",
            str(ROOT / "scripts" / "publish_empty_reference_development.py"),
            "--safe-manifest",
            str(DEVELOPMENT_INPUTS),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_bytes())
    assert artifact["status"] == "prediction_gates_pass"
    assert artifact["bindings"]["safe_manifest_path"] == str(DEVELOPMENT_INPUTS)


def test_semantic_scorer_consumes_frozen_predictions_without_changing_them(tmp_path):
    predictions = tmp_path / "predictions.json"
    publish = subprocess.run(
        [
            "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python",
            str(ROOT / "scripts" / "publish_empty_reference_development.py"),
            "--safe-manifest",
            str(DEVELOPMENT_INPUTS),
            "--out",
            str(predictions),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert publish.returncode in {0, 2}, publish.stderr
    before = predictions.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    artifact = json.loads(before)
    assert artifact["phase_separation"]["semantic_oracle_images_opened"] == 0
    assert all(
        "semantic_mask_path" not in stage
        for case in artifact["development_records"]
        for stage in case["stages"]
    )

    score_path = tmp_path / "score.json"
    score = subprocess.run(
        [
            "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python",
            str(ROOT / "scripts" / "score_empty_reference_development.py"),
            "--predictions",
            str(predictions),
            "--predictions-sha256",
            digest,
            "--image-locators",
            str(DEVELOPMENT_LOCATORS),
            "--out",
            str(score_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert score.returncode == 0, score.stderr
    assert predictions.read_bytes() == before
    scored = json.loads(score_path.read_bytes())
    assert scored["bindings"]["frozen_prediction_sha256"] == digest
    assert scored["phase_separation"] == {
        "phase": "post_inference_semantic_scoring",
        "prediction_artifact_written_by_scorer": False,
        "prediction_was_hash_sealed_before_locator_or_mask_open": True,
    }


def test_terminal_result_publication_normalizes_numpy_scalar_leaves(tmp_path):
    from fluidmix import empty_reference_development_scoring as scoring

    output = tmp_path / "result.json"
    scoring._publish_exclusive(
        output,
        {
            "checks": {"frozen_gate": np.bool_(True)},
            "metrics": {
                "evaluation_images": np.int64(6),
                "mean_dice": np.float32(0.875),
            },
        },
    )

    assert json.loads(output.read_bytes()) == {
        "checks": {"frozen_gate": True},
        "metrics": {"evaluation_images": 6, "mean_dice": 0.875},
    }
    with pytest.raises(FileExistsError):
        scoring._publish_exclusive(output, {"checks": {"frozen_gate": False}})


def test_scorer_cannot_open_oracle_until_prediction_hash_is_sealed(tmp_path, monkeypatch):
    from fluidmix import empty_reference_development_scoring as scoring

    predictions = tmp_path / "predictions.json"
    predictions.write_text("{}\n", encoding="utf-8")

    def forbidden_semantic_open(*_args, **_kwargs):
        raise AssertionError("semantic loader ran before prediction seal")

    monkeypatch.setattr(scoring, "_load_semantic", forbidden_semantic_open)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        scoring.score_frozen_development_predictions(
            prediction_path=predictions,
            expected_prediction_sha256="0" * 64,
            image_locators_path=tmp_path / "must-not-open.json",
            output_path=tmp_path / "must-not-write.json",
        )


def test_development_scorer_accepts_only_validation_partition_locators(tmp_path):
    from fluidmix.empty_reference_development_scoring import (
        score_frozen_development_predictions,
    )

    predictions = Path(
        "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
        "rgb-mask-predictor-01a18128/approach-a-stage2-attempt-01a18128/"
        "predictions.json"
    )
    result = score_frozen_development_predictions(
        prediction_path=predictions,
        expected_prediction_sha256=(
            "8b4ed430290e388ef0b9822fa5ba69d25516457adeda4752a0e66aa0ce482575"
        ),
        image_locators_path=DEVELOPMENT_LOCATORS,
        output_path=tmp_path / "score.json",
    )

    assert result["status"] == "development_scored"
    assert result["bindings"]["image_locators_path"] == str(DEVELOPMENT_LOCATORS)
    assert result["denominators"]["development_semantic_masks_opened"] == 6


def test_publisher_rejects_mixed_manifest_before_hash_or_open(tmp_path, monkeypatch):
    script_path = ROOT / "scripts" / "publish_empty_reference_development.py"
    spec = importlib.util.spec_from_file_location("development_publisher", script_path)
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)

    def forbidden_hash(_path):
        raise AssertionError("mixed manifest reached the file hash/open boundary")

    monkeypatch.setattr(publisher, "_sha256", forbidden_hash)
    mixed = Path(
        "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
        "wp4-oracle-population-fix2-01a18128/f2-fill-clip-inputs.json"
    )
    with pytest.raises(ValueError, match="validation-only"):
        publisher.main(
            ["--safe-manifest", str(mixed), "--out", str(tmp_path / "forbidden.json")]
        )


def test_scorer_rejects_mixed_locators_before_hash_or_open(tmp_path, monkeypatch):
    from fluidmix import empty_reference_development_scoring as scoring

    predictions = Path(
        "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
        "rgb-mask-predictor-01a18128/approach-a-stage2-attempt-01a18128/"
        "predictions.json"
    )

    def forbidden_hash(_path):
        raise AssertionError("mixed locators reached the file hash/open boundary")

    monkeypatch.setattr(scoring, "_sha256", forbidden_hash)
    mixed = Path(
        "/home/hackathon5/liamreporoot/f2-varied-v3-measurements-20261027/"
        "image-locators.json"
    )
    with pytest.raises(ValueError, match="validation-only"):
        scoring.score_frozen_development_predictions(
            prediction_path=predictions,
            expected_prediction_sha256=(
                "8b4ed430290e388ef0b9822fa5ba69d25516457adeda4752a0e66aa0ce482575"
            ),
            image_locators_path=mixed,
            output_path=tmp_path / "forbidden.json",
        )

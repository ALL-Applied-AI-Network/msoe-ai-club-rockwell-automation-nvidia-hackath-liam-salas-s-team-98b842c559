"""Contract tests for the deterministic temporal RGB liquid-mask baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_grid_is_exactly_the_preregistered_56_configs():
    from fluidmix.temporal_rgb_mask_baseline import candidate_configs

    configs = candidate_configs()

    assert len(configs) == 56
    assert len({config.canonical_json() for config in configs}) == 56
    assert {config.feature for config in configs} == {"max_abs_srgb", "lab_delta_e76"}
    assert {
        config.threshold for config in configs if config.feature == "max_abs_srgb"
    } == {4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0}
    assert {
        config.threshold for config in configs if config.feature == "lab_delta_e76"
    } == {2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0}
    assert {config.support_margin_px for config in configs} == {0, 4}
    assert {config.output_rule for config in configs} == {
        "largest_component",
        "height_fill_prior",
    }
    assert all(config.minimum_changed_pixels_per_row == 4 for config in configs)


def test_height_fill_prior_uses_rgb_change_to_choose_top_then_fills_train_support():
    from fluidmix.temporal_rgb_mask_baseline import (
        TemporalMaskConfig,
        fit_train_geometry,
        predict_mask,
    )

    support = np.zeros((9, 10), dtype=bool)
    support[2:8, 3:7] = True
    geometry = fit_train_geometry([support, support])
    prefill = np.zeros((9, 10, 3), dtype=np.uint8)
    stage = prefill.copy()
    stage[5:8, 3:7, 0] = 30
    config = TemporalMaskConfig(
        feature="max_abs_srgb",
        threshold=16.0,
        support_margin_px=0,
        output_rule="height_fill_prior",
        minimum_changed_pixels_per_row=4,
    )

    predicted = predict_mask(stage, prefill, geometry, config)

    expected = np.zeros_like(support)
    expected[5:8, 3:7] = True
    assert np.array_equal(predicted, expected)


def test_largest_component_uses_eight_connectivity_and_deterministic_largest_area():
    from fluidmix.temporal_rgb_mask_baseline import (
        TemporalMaskConfig,
        fit_train_geometry,
        predict_mask,
    )

    support = np.ones((6, 7), dtype=bool)
    geometry = fit_train_geometry([support])
    prefill = np.zeros((6, 7, 3), dtype=np.uint8)
    stage = prefill.copy()
    stage[1, 1, 0] = 20
    stage[2, 2, 0] = 20  # diagonal: same 8-connected component
    stage[4:6, 5:7, 0] = 20  # larger four-pixel component
    config = TemporalMaskConfig(
        feature="max_abs_srgb",
        threshold=16.0,
        support_margin_px=0,
        output_rule="largest_component",
        minimum_changed_pixels_per_row=4,
    )

    predicted = predict_mask(stage, prefill, geometry, config)

    expected = np.zeros((6, 7), dtype=bool)
    expected[4:6, 5:7] = True
    assert np.array_equal(predicted, expected)


@pytest.mark.parametrize(
    "forbidden",
    ["mask_path", "label", "residual_volume_ml", "palette", "composition"],
)
def test_predictor_record_rejects_oracle_and_latent_fields(forbidden):
    from fluidmix.temporal_rgb_mask_baseline import validate_predictor_record

    record = {
        "image_id": "IMAGE-1",
        "run_id": "RUN-1",
        "split_role": "train",
        "stage_rgb_path": "/tmp/stage.png",
        "stage_rgb_sha256": "a" * 64,
        "reference_rgb_path": "/tmp/prefill.png",
        "reference_rgb_sha256": "b" * 64,
        forbidden: "hidden",
    }

    with pytest.raises(ValueError, match="predictor input fields"):
        validate_predictor_record(record)


def test_mask_metrics_are_independently_checkable():
    from fluidmix.temporal_rgb_mask_baseline import score_mask

    target = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)

    result = score_mask(prediction, target)

    assert result == {
        "intersection_pixels": 1,
        "prediction_pixels": 2,
        "target_pixels": 2,
        "union_pixels": 3,
        "iou": pytest.approx(1 / 3),
        "dice": pytest.approx(1 / 2),
        "precision": pytest.approx(1 / 2),
        "recall": pytest.approx(1 / 2),
        "prediction_empty": False,
        "prediction_height_px": 2,
        "target_height_px": 1,
        "absolute_height_error_px": 1,
    }


def _png(path: Path, array: np.ndarray) -> str:
    mode = "RGB" if array.ndim == 3 else "L"
    Image.fromarray(array, mode=mode).save(path)
    return _sha(path)


def _experiment_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    merged = tmp_path / "merged"
    merged.mkdir()
    rows = []
    specifications = [
        ("CASE_TRAIN", "RUN-TRAIN", "train", 1, 4),
        ("CASE_VAL", "RUN-VAL", "validation", 1, 4),
        ("CASE_EVAL", "RUN-EVAL", "evaluation", 1, 4),
    ]
    for case_id, run_id, split, position, top in specifications:
        app = merged / "cases" / case_id / "app-output"
        frames = app / "frames"
        masks = app / "liquid"
        prefill_dir = app / "pre-pour-parity"
        frames.mkdir(parents=True)
        masks.mkdir()
        prefill_dir.mkdir()
        prefill = np.zeros((8, 8, 3), dtype=np.uint8)
        stage = prefill.copy()
        stage[top:7, 2:6, 0] = 40
        target = np.zeros((8, 8), dtype=np.uint8)
        target[top:7, 2:6] = 255
        stage_path = frames / "stage.png"
        mask_path = masks / "mask.png"
        prefill_path = prefill_dir / "f2-prefill-time-0001.png"
        stage_sha = _png(stage_path, stage)
        mask_sha = _png(mask_path, target)
        _png(prefill_path, prefill)
        rows.append(
            {
                "batch_id": f"BATCH-{split}",
                "case_id": case_id,
                "run_id": run_id,
                "split_role": split,
                "vial_position": position,
                "stages": [
                    {
                        "rgb_path": str(stage_path.relative_to(merged)),
                        "rgb_sha256": stage_sha,
                        "liquid_mask_path": str(mask_path.relative_to(merged)),
                        "liquid_mask_sha256": mask_sha,
                        "stage_ordinal": 1,
                        "tank_id": "t200",
                    }
                ],
            }
        )
    # The evaluation files must remain unopened during this validation-only runner.
    evaluation_stage = merged / "cases/CASE_EVAL/app-output/frames/stage.png"
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen test preregistration\n", encoding="utf-8")
    return index, merged, prereg


def test_validation_runner_publishes_no_clobber_and_never_opens_evaluation_pixels(
    tmp_path, monkeypatch
):
    from fluidmix.temporal_rgb_mask_baseline import run_validation_experiment

    index, merged, prereg = _experiment_fixture(tmp_path)
    output = tmp_path / "result"
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path):
        if "CASE_EVAL" in str(path):
            raise AssertionError("evaluation pixel was opened before validation handoff")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    result = run_validation_experiment(
        index_path=index,
        merged_root=merged,
        preregistration_path=prereg,
        output_dir=output,
        expected_denominators={
            "train_vials": 1,
            "train_stage_images": 1,
            "validation_vials": 1,
            "validation_stage_images": 1,
            "evaluation_vials_declared_not_opened": 1,
            "evaluation_stage_images_declared_not_opened": 1,
        },
    )

    assert result["evaluation_pixels_opened"] is False
    assert result["candidate_count"] == 56
    assert (output / "result.json").is_file()
    assert (output / "frozen-config.json").is_file()
    assert (output / "predictor-inputs.json").is_file()
    with pytest.raises(FileExistsError):
        run_validation_experiment(
            index_path=index,
            merged_root=merged,
            preregistration_path=prereg,
            output_dir=output,
            expected_denominators={
                "train_vials": 1,
                "train_stage_images": 1,
                "validation_vials": 1,
                "validation_stage_images": 1,
                "evaluation_vials_declared_not_opened": 1,
                "evaluation_stage_images_declared_not_opened": 1,
            },
        )

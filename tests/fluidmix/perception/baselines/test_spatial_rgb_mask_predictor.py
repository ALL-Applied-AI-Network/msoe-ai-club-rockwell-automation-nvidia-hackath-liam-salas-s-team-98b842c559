"""Contracts for the frozen single-RGB spatial liquid-mask predictor."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_grid_is_exactly_preregistered_32_configs():
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import candidate_configs

    configs = candidate_configs()

    assert len(configs) == 32
    assert len({config.canonical_json() for config in configs}) == 32
    assert {config.feature_set for config in configs} == {"rgb_xy", "rgb_xy_local"}
    assert {config.max_depth for config in configs} == {8, 12}
    assert {config.min_samples_leaf for config in configs} == {64, 256}
    assert {config.probability_threshold for config in configs} == {0.35, 0.5}
    assert {config.cleanup for config in configs} == {
        "largest_component",
        "largest_component_row_fill",
    }
    assert all(config.seed == 20261028 for config in configs)


def _leaf_model(shape=(5, 6), probability=0.9):
    support = np.ones(shape, dtype=np.uint8)
    return {
        "schema_version": 1,
        "artifact_type": "spatial_rgb_mask_random_forest",
        "feature_set": "rgb_xy",
        "image_shape_hw": list(shape),
        "train_union_support_rle": [[0, shape[0] * shape[1]]],
        "trees": [
            {
                "children_left": [-1],
                "children_right": [-1],
                "feature": [-2],
                "threshold": [-2.0],
                "positive_probability": [probability],
            }
        ],
    }


def test_predict_boundary_accepts_only_one_rgb_and_is_deterministic():
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import SpatialMaskConfig, predict_mask

    assert tuple(inspect.signature(predict_mask).parameters) == (
        "rgb",
        "model",
        "config",
    )
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    config = SpatialMaskConfig(
        feature_set="rgb_xy",
        max_depth=8,
        min_samples_leaf=64,
        probability_threshold=0.5,
        cleanup="largest_component",
    )

    first = predict_mask(rgb, _leaf_model(), config)
    second = predict_mask(rgb.copy(), _leaf_model(), config)

    assert np.array_equal(first, np.ones((5, 6), dtype=bool))
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((5, 6), dtype=np.uint8),
        np.zeros((5, 6, 4), dtype=np.uint8),
        np.zeros((5, 6, 3), dtype=np.float32),
    ],
)
def test_predictor_fails_closed_on_malformed_rgb(rgb):
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import SpatialMaskConfig, predict_mask

    config = SpatialMaskConfig(
        feature_set="rgb_xy",
        max_depth=8,
        min_samples_leaf=64,
        probability_threshold=0.5,
        cleanup="largest_component",
    )
    with pytest.raises(ValueError, match="uint8 RGB"):
        predict_mask(rgb, _leaf_model(), config)


def test_load_model_binds_expected_hash_and_rejects_malformed(tmp_path):
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import canonical_json_bytes, load_model

    model_path = tmp_path / "model.json"
    model_path.write_bytes(canonical_json_bytes(_leaf_model()))
    digest = _sha(model_path)

    assert load_model(model_path, digest) == _leaf_model()
    with pytest.raises(ValueError, match="model hash mismatch"):
        load_model(model_path, "0" * 64)
    model_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model fields"):
        load_model(model_path, _sha(model_path))


def _png(path: Path, array: np.ndarray) -> str:
    Image.fromarray(array, mode="RGB" if array.ndim == 3 else "L").save(path)
    return _sha(path)


def _experiment_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    merged = tmp_path / "merged"
    rows = []
    for split, case_id, top in (
        ("train", "CASE_TRAIN", 3),
        ("validation", "CASE_VAL", 2),
        ("evaluation", "CASE_EVAL", 2),
    ):
        app = merged / "cases" / case_id / "app-output"
        (app / "frames").mkdir(parents=True)
        (app / "liquid").mkdir()
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[top:7, 2:6] = (30, 10, 5)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[top:7, 2:6] = 255
        rgb_path = app / "frames" / "stage.png"
        mask_path = app / "liquid" / "mask.png"
        rgb_sha = _png(rgb_path, rgb)
        mask_sha = _png(mask_path, mask)
        rows.append(
            {
                "batch_id": f"BATCH_{split}",
                "case_id": case_id,
                "run_id": f"RUN_{split}",
                "split_role": split,
                "vial_position": 1,
                "stages": [
                    {
                        "rgb_path": str(rgb_path.relative_to(merged)),
                        "rgb_sha256": rgb_sha,
                        "liquid_mask_path": str(mask_path.relative_to(merged)),
                        "liquid_mask_sha256": mask_sha,
                        "stage_ordinal": 1,
                        "tank_id": "t200",
                    }
                ],
            }
        )
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("fixed preregistration\n", encoding="utf-8")
    return index, merged, prereg


def test_validation_loader_never_opens_evaluation_pixels(tmp_path, monkeypatch):
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import load_experiment_data

    index, merged, _ = _experiment_fixture(tmp_path)
    original = Path.read_bytes

    def guarded(path: Path):
        if "CASE_EVAL" in str(path):
            raise AssertionError("evaluation pixel opened during validation")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    data = load_experiment_data(
        index_path=index,
        merged_root=merged,
        roles=("train", "validation"),
        expected_denominators={
            "train_vials": 1,
            "train_stage_images": 1,
            "validation_vials": 1,
            "validation_stage_images": 1,
            "evaluation_vials": 1,
            "evaluation_stage_images": 1,
        },
    )

    assert set(data.examples) == {"train", "validation"}
    assert data.denominators["evaluation_stage_images"] == 1


def test_evaluation_sentinel_is_atomic_and_exactly_once(tmp_path):
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import claim_evaluation_pass

    sentinel = tmp_path / "evaluation-pass.claimed"
    claim_evaluation_pass(sentinel)
    assert sentinel.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="already claimed"):
        claim_evaluation_pass(sentinel)


def test_schema_two_locator_from_predictions_is_validator_compatible(tmp_path):
    from fluidmix.perception.masks import (
        build_trusted_predicted_mask_registry_v1,
        validate_rgb_mask_prediction_locators,
    )
    from fluidmix.perception.baselines.spatial_rgb_mask_predictor import build_locator_artifact

    rgb = tmp_path / "rgb.png"
    mask = tmp_path / "prediction.png"
    _png(rgb, np.zeros((4, 5, 3), dtype=np.uint8))
    predicted = np.zeros((4, 5), dtype=np.uint8)
    predicted[1:4, 1:4] = 255
    _png(mask, predicted)
    artifact = build_locator_artifact(
        predictions={
            "IMAGE-1": {
                "rgb_path": str(rgb.resolve()),
                "rgb_sha256": _sha(rgb),
                "mask_path": str(mask.resolve()),
                "mask_sha256": _sha(mask),
            }
        },
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        code_commit="3" * 40,
    )

    images, producer = validate_rgb_mask_prediction_locators(artifact)
    registry = build_trusted_predicted_mask_registry_v1(artifact)
    assert set(images) == {"IMAGE-1"}
    assert producer["input_fields"] == ["rgb"]
    assert registry.image_ids == ("IMAGE-1",)

"""Training-only data boundary for the WP5 CNN candidate."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def _write_rgb(path: Path, value: int) -> str:
    Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mask(path: Path) -> str:
    value = np.zeros((24, 32), dtype=np.uint8)
    value[8:18, 11:20] = 255
    Image.fromarray(value).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_inventory_accepts_only_twin_train_and_simulator_masks(tmp_path: Path):
    from fluidmix.perception.cnn.cnn_training import validate_training_inventory

    row = {
        "sample_id": "S1",
        "run_id": "RUN-TRAIN-01",
        "case_id": "C1",
        "split_role": "train",
        "data_origin": "twin",
        "current_rgb_path": "/tmp/current.png",
        "current_rgb_sha256": "a" * 64,
        "reference_rgb_path": "/tmp/reference.png",
        "reference_rgb_sha256": "b" * 64,
        "label_kind": "simulator_mask",
        "mask_path": "/tmp/mask.png",
        "mask_sha256": "c" * 64,
        "label_source": "simulator_semantic_mask",
    }
    checked = validate_training_inventory({"schema_version": 1, "samples": [row]})
    assert checked["samples"][0] == row

    bad = {"schema_version": 1, "samples": [{**row, "split_role": "evaluation"}]}
    with pytest.raises(ValueError, match="train split"):
        validate_training_inventory(bad)
    bad = {"schema_version": 1, "samples": [{**row, "recipe_ml": {"t100": 10}}]}
    with pytest.raises(ValueError, match="forbidden training field"):
        validate_training_inventory(bad)


def test_dataset_returns_current_reference_only_and_empty_negative_mask(tmp_path: Path):
    from fluidmix.perception.cnn.cnn_training import PairedMaskDataset

    current = tmp_path / "current.png"
    reference = tmp_path / "reference.png"
    mask = tmp_path / "mask.png"
    current_sha = _write_rgb(current, 130)
    reference_sha = _write_rgb(reference, 100)
    mask_sha = _write_mask(mask)
    base = {
        "run_id": "RUN-TRAIN-01",
        "case_id": "C1",
        "split_role": "train",
        "data_origin": "twin",
        "current_rgb_path": str(current),
        "current_rgb_sha256": current_sha,
        "reference_rgb_path": str(reference),
        "reference_rgb_sha256": reference_sha,
    }
    inventory = {
        "schema_version": 1,
        "samples": [
            {
                **base,
                "sample_id": "positive",
                "label_kind": "simulator_mask",
                "mask_path": str(mask),
                "mask_sha256": mask_sha,
                "label_source": "simulator_semantic_mask",
            },
            {
                **base,
                "sample_id": "empty",
                "current_rgb_path": str(reference),
                "current_rgb_sha256": reference_sha,
                "label_kind": "empty_negative",
                "label_source": "same_run_empty_prepour",
            },
        ],
    }
    dataset = PairedMaskDataset(inventory, augment=False, verify_hashes=True)

    positive = dataset[0]
    negative = dataset[1]
    assert set(positive) == {"current_rgb", "reference_rgb", "mask", "sample_id"}
    assert positive["current_rgb"].shape == (3, 24, 32)
    assert positive["reference_rgb"].shape == (3, 24, 32)
    assert positive["mask"].shape == (1, 24, 32)
    assert int(positive["mask"].sum()) == 90
    assert int(negative["mask"].sum()) == 0

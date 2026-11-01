"""CPU contract tests for Step 7 synthetic-only U-Net training."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import fill_level_training as TRAINING  # noqa: E402


SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "train_fill_level_unet",
    ROOT / "scripts" / "train_fill_level_unet.py",
)
TRAIN_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(TRAIN_SCRIPT)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_chunk(root, name, role, seeds, *, attempts=("attempt-01",)):
    chunk = root / name
    for attempt_name in attempts:
        attempt = chunk / attempt_name
        output = attempt / "output"
        output.mkdir(parents=True)
        (attempt / "state.txt").write_text("COMPLETE\n", encoding="utf-8")
        records = []
        for index, seed in enumerate(seeds):
            records.append(
                {
                    "role": role,
                    "seed": seed,
                    "frame_index": index,
                    "outputs": {
                        "rgb": {
                            "path": f"rgb/{index:05d}.png",
                            "sha256": "a" * 64,
                        },
                        "mask": {
                            "path": f"mask/{index:05d}.png",
                            "sha256": "b" * 64,
                        },
                    },
                }
            )
        (output / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    return chunk


def _review_files(tmp_path, *, status="complete", disposition="accepted_legitimate_occlusion"):
    summary = {
        "integrity": {"passed": True},
        "old_rule_review": {"exceptions": [{"seed": 100327, "role": "train"}]},
        "chunks": [
            {
                "chunk": "02-train-0100000-0100199",
                "role": "train",
                "attempt": "attempt-01",
                "manifest_sha256": "b" * 64,
                "seed_start": 100000,
                "seed_end": 100199,
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    excluded = [100327] if disposition == "excluded" else []
    review = {
        "status": status,
        "batch_summary_sha256": _sha256(summary_path),
        "eligible_counts": {"train": 3000 - len(excluded), "val": 300},
        "excluded_train_validation_seeds": excluded,
        "exceptions": [
            {
                "seed": 100327,
                "role": "train",
                "disposition": disposition,
                "explanation": "fully hidden below the base edge",
                "overlay_sha256": "a" * 64,
            }
        ],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return summary_path, review_path


def test_completed_exception_review_is_required_and_hashed(tmp_path):
    summary, review = _review_files(tmp_path)
    result = TRAINING.validate_exception_review(summary, review)
    assert result["eligible_counts"] == {"train": 3000, "val": 300}
    assert result["excluded_seeds"] == []
    value = json.loads(review.read_text())
    value["status"] = "pending"
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="status must be complete"):
        TRAINING.validate_exception_review(summary, review)


def test_cli_gate_fails_before_dataset_discovery_or_output_creation(tmp_path):
    summary, review = _review_files(tmp_path, status="pending")
    args = type(
        "Args",
        (),
        {
            "batch_summary": summary,
            "exception_review": review,
            "dataset_root": tmp_path / "must-not-be-opened",
            "output_root": tmp_path / "must-not-be-created",
        },
    )()
    with pytest.raises(ValueError, match="status must be complete"):
        TRAIN_SCRIPT.run(args)
    assert not args.output_root.exists()


def test_excluded_disposition_updates_declared_eligible_counts(tmp_path):
    summary, review = _review_files(tmp_path, disposition="excluded")
    result = TRAINING.validate_exception_review(summary, review)
    assert result["excluded_seeds"] == [100327]
    assert result["eligible_counts"] == {"train": 2999, "val": 300}


def test_exception_review_rejects_missing_inventory_item(tmp_path):
    summary, review = _review_files(tmp_path)
    value = json.loads(review.read_text())
    value["exceptions"] = []
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="do not match inventory"):
        TRAINING.validate_exception_review(summary, review)


def test_discovery_skips_test_chunk_before_opening_attempt_or_manifest(tmp_path):
    root = tmp_path / "collected"
    _write_chunk(root, "00-val-1100000-1100000", "val", [1100000])
    _write_chunk(root, "02-train-0100000-0100001", "train", [100000, 100001])
    # An intentionally empty test chunk would fail immediately if discovery opened it.
    (root / "17-test-2100000-2100199").mkdir(parents=True)
    result = TRAINING.discover_train_val_samples(
        root,
        expected_train=2,
        expected_val=1,
    )
    assert [sample.seed for sample in result["train"]] == [100000, 100001]
    assert [sample.seed for sample in result["val"]] == [1100000]


def test_discovery_applies_exclusion_and_exact_post_exclusion_counts(tmp_path):
    root = tmp_path / "collected"
    _write_chunk(root, "00-val-1100000-1100000", "val", [1100000])
    _write_chunk(root, "02-train-0100000-0100001", "train", [100000, 100001])
    result = TRAINING.discover_train_val_samples(
        root,
        expected_train=1,
        expected_val=1,
        excluded_seeds=[100001],
    )
    assert [sample.seed for sample in result["train"]] == [100000]
    with pytest.raises(ValueError, match="counts must be"):
        TRAINING.discover_train_val_samples(
            root,
            expected_train=2,
            expected_val=1,
            excluded_seeds=[100001],
        )


def test_discovery_rejects_ambiguous_complete_attempts(tmp_path):
    root = tmp_path / "collected"
    _write_chunk(root, "00-val-1100000-1100000", "val", [1100000])
    _write_chunk(
        root,
        "02-train-0100000-0100000",
        "train",
        [100000],
        attempts=("attempt-01", "attempt-02"),
    )
    with pytest.raises(ValueError, match="exactly one COMPLETE"):
        TRAINING.discover_train_val_samples(root, expected_train=1, expected_val=1)


def test_discovery_binds_attempt_and_manifest_to_approved_summary(tmp_path):
    root = tmp_path / "collected"
    val_name = "00-val-1100000-1100000"
    train_name = "02-train-0100000-0100000"
    _write_chunk(root, val_name, "val", [1100000])
    _write_chunk(root, train_name, "train", [100000])
    expected = {}
    for name, role, seed in (
        (val_name, "val", 1100000),
        (train_name, "train", 100000),
    ):
        manifest = root / name / "attempt-01" / "output" / "manifest.jsonl"
        expected[name] = {
            "role": role,
            "attempt": "attempt-01",
            "manifest_sha256": _sha256(manifest),
            "seed_start": seed,
            "seed_end": seed,
        }
    TRAINING.discover_train_val_samples(
        root,
        expected_train=1,
        expected_val=1,
        expected_chunks=expected,
    )
    manifest = root / train_name / "attempt-01" / "output" / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest hash is not approved"):
        TRAINING.discover_train_val_samples(
            root,
            expected_train=1,
            expected_val=1,
            expected_chunks=expected,
        )


def test_preflight_rejects_rgb_or_mask_tampering(tmp_path):
    rgb = tmp_path / "rgb.png"
    mask = tmp_path / "mask.png"
    rgb.write_bytes(b"approved-rgb")
    mask.write_bytes(b"approved-mask")
    sample = TRAINING.FillLevelSample(
        role="train",
        seed=100000,
        chunk="02-train-0100000-0100199",
        frame_index=0,
        rgb_path=rgb,
        mask_path=mask,
        rgb_sha256=_sha256(rgb),
        mask_sha256=_sha256(mask),
    )
    result = TRAINING.verify_sample_files([sample])
    assert result["samples_checked"] == 1
    assert result["files_checked"] == 2
    rgb.write_bytes(b"tampered-rgb")
    with pytest.raises(ValueError, match="RGB hash does not match"):
        TRAINING.verify_sample_files([sample])


def test_state_ledger_is_append_only(tmp_path):
    state = tmp_path / "state.jsonl"
    TRAIN_SCRIPT._append_state(state, "RUNNING")
    first = state.read_text()
    TRAIN_SCRIPT._append_state(state, "COMPLETE")
    lines = state.read_text().splitlines()
    assert len(lines) == 2
    assert state.read_text().startswith(first)
    assert [json.loads(line)["status"] for line in lines] == ["RUNNING", "COMPLETE"]


def test_crop_and_mask_contract():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.full((480, 640), 255, dtype=np.uint8)
    mask[48:464, 256:368] = 1
    cropped_rgb, cropped_mask = TRAINING.crop_rgb_mask(rgb, mask)
    assert cropped_rgb.shape == (416, 112, 3)
    assert cropped_mask.shape == (416, 112)
    assert set(np.unique(cropped_mask)) == {1}
    mask[100, 300] = 9
    with pytest.raises(ValueError, match="unexpected class ids"):
        TRAINING.crop_rgb_mask(rgb, mask)


def test_photometric_augmentation_preserves_dimensions_and_mask_geometry():
    rgb = np.tile(np.arange(112, dtype=np.uint8), (416, 1))
    rgb = np.stack((rgb, rgb, rgb), axis=2)
    mask = np.zeros((416, 112), dtype=np.uint8)
    mask[200:, :] = 1
    original_mask = mask.copy()
    augmented = TRAINING.augment_rgb(rgb, np.random.default_rng(7))
    assert augmented.shape == rgb.shape
    assert augmented.dtype == np.uint8
    assert np.array_equal(mask, original_mask)


def test_class_weights_use_four_positive_training_counts():
    weights = TRAINING.class_weights_from_counts([1000, 500, 10, 250])
    assert weights.shape == (4,)
    assert weights[2] > weights[1] > weights[0]
    assert weights.mean() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="four positive"):
        TRAINING.class_weights_from_counts([1, 2, 0, 4])


def test_ignore_aware_confusion_and_iou():
    target = np.array([[0, 1, 2], [3, 255, 1]], dtype=np.uint8)
    prediction = np.array([[0, 1, 1], [3, 0, 0]], dtype=np.uint8)
    confusion = TRAINING.confusion_matrix(prediction, target)
    assert confusion.sum() == 5
    assert confusion[0, 0] == 1
    assert confusion[1, 0] == 1
    assert confusion[2, 1] == 1
    per_class, macro = TRAINING.iou_from_confusion(confusion)
    assert per_class[3] == pytest.approx(1.0)
    assert 0.0 < macro < 1.0


def test_unet_output_shape_when_torch_is_available():
    torch = pytest.importorskip("torch")
    model = TRAINING.build_unet(base_channels=4)
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 416, 112)))
    assert tuple(output.shape) == (1, 4, 416, 112)

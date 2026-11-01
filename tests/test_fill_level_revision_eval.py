"""CPU-only tests for the Revision 1 synthetic comparison contracts."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import fill_level_revision_eval as REVISION  # noqa: E402
from fluidmix import fill_level_surface_eval as SURFACE  # noqa: E402
from fluidmix import fill_level_training as TRAINING  # noqa: E402


HEIGHT = 464 - 48
WIDTH = 368 - 256
FIRST = 298 - 256
LAST = 316 - 256


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _audit():
    return {
        "eligible_counts": {"train": 3000, "val": 300},
        "batch_summary_sha256": "a" * 64,
        "review_sha256": "b" * 64,
    }


def _training_summary(checkpoint, *, revision="original", test_images=0):
    module_hash = (
        REVISION.ORIGINAL_TRAINING_MODULE_SHA256
        if revision == "original"
        else TRAINING.file_sha256(TRAINING.__file__)
    )
    result = {
        "status": "complete",
        "profile": "full",
        "checkpoint_candidate": True,
        "selection_split": "validation",
        "best_epoch": 3,
        "dataset": {
            "used_counts": {"train": 3000, "val": 300},
            "test_manifests_opened": 0,
            "test_images_loaded": test_images,
        },
        "inputs": {
            "batch_summary_sha256": "a" * 64,
            "exception_review_sha256": "b" * 64,
            "training_module_sha256": module_hash,
        },
        "artifacts": {
            "best_weights": checkpoint.name,
            "best_weights_sha256": _sha(checkpoint),
        },
        "config": {
            "crop_xyxy": [256, 48, 368, 464],
            "class_ids": {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3},
            "base_channels": 16,
            "seed": 20261101,
        },
    }
    if revision == "revised":
        result["config"].update(
            {
                "learning_rate": 1e-3,
                "epochs": 4,
                "augmentation": {
                    "enabled": True,
                    "train_only": True,
                    "contract": TRAINING.training_augmentation_config(),
                },
                "learning_rate_scheduler": {
                    "name": "cosine_annealing",
                    "initial_learning_rate": 1e-3,
                    "minimum_learning_rate": 1e-5,
                    "t_max_epochs": 4,
                    "step_unit": "epoch",
                },
            }
        )
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _rule_summary(
    checkpoint_sha,
    training_summary_sha,
    *,
    revision="original",
    real_frames=0,
):
    ledger = "d" * 64
    if revision == "original":
        rule_config = SURFACE.frozen_rule_config()
        # Match the Step 8 baseline artifact, which predates explicit row bounds.
        rule_config.pop("measurement_rows_full_half_open")
        rule_config.pop("measurement_row_derivation")
        evaluator_hash = REVISION.ORIGINAL_SURFACE_MODULE_SHA256
    else:
        derivation = {
            "measurement_rows_full_half_open": [83, 427],
            "selection_source": "synthetic_validation_label_outline_intersection",
            "frame_count": 300,
            "role": "val",
            "ordered_hash_ledger_sha256": ledger,
        }
        rule_config = SURFACE.frozen_rule_config(
            measurement_rows_full=(83, 427),
            row_bounds_evidence=derivation,
        )
        evaluator_hash = TRAINING.file_sha256(SURFACE.__file__)
    return {
        "status": "complete",
        "evidence_label": "measured_on_synthetic_validation_only",
        "dataset": {
            "role": "val",
            "frame_count": 300,
            "train_manifests_opened": 0,
            "train_images_loaded": 0,
            "test_manifests_opened": 0,
            "test_images_loaded": 0,
            "real_frames_loaded": real_frames,
            "preflight": {"ordered_hash_ledger_sha256": ledger},
        },
        "model": {
            "checkpoint_sha256": checkpoint_sha,
            "training_summary_sha256": training_summary_sha,
        },
        "inputs": {
            "checkpoint_sha256": checkpoint_sha,
            "training_summary_sha256": training_summary_sha,
            "batch_summary_sha256": "a" * 64,
            "exception_review_sha256": "b" * 64,
            "evaluator_module_sha256": evaluator_hash,
        },
        "rule_config": rule_config,
        "aggregate": {"by_case": {}, "by_level_bin_10ml": {}},
    }


def _mask(row=150):
    value = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    value[row - 48 :, FIRST:LAST] = 1
    return value


def _metadata(seed=1100000):
    return SURFACE.ValidationMetadata(
        seed=seed,
        level_ml=50.0,
        sampling_band="full_0_85",
        foam=False,
        sticker=False,
        geometry_row=150.0,
    )


def test_training_binding_requires_exact_hashes_and_zero_test_access(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    summary = _write_json(
        tmp_path / "summary.json", _training_summary(checkpoint)
    )
    result = REVISION.validate_training_artifact(
        summary,
        _sha(summary),
        checkpoint,
        _sha(checkpoint),
        _audit(),
        revision="original",
    )
    assert result["checkpoint_sha256"] == _sha(checkpoint)
    bad = _training_summary(checkpoint, test_images=1)
    _write_json(summary, bad)
    with pytest.raises(ValueError, match="accessed test data"):
        REVISION.validate_training_artifact(
            summary,
            _sha(summary),
            checkpoint,
            _sha(checkpoint),
            _audit(),
            revision="original",
        )


def test_training_binding_rejects_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    summary = _write_json(
        tmp_path / "summary.json", _training_summary(checkpoint)
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        REVISION.validate_training_artifact(
            summary,
            _sha(summary),
            checkpoint,
            hashlib.sha256(b"weights").hexdigest(),
            _audit(),
            revision="original",
        )


def test_rule_binding_requires_synthetic_only_access_and_checkpoint(tmp_path):
    checkpoint_sha = "c" * 64
    training_artifact = {
        "checkpoint_sha256": checkpoint_sha,
        "summary_sha256": "e" * 64,
    }
    summary_path = _write_json(
        tmp_path / "rule.json",
        _rule_summary(checkpoint_sha, training_artifact["summary_sha256"]),
    )
    result = REVISION.validate_rule_artifact(
        summary_path,
        _sha(summary_path),
        training_artifact,
        _audit(),
        revision="original",
    )
    assert result["ordered_hash_ledger_sha256"] == "d" * 64
    _write_json(
        summary_path,
        _rule_summary(
            checkpoint_sha,
            training_artifact["summary_sha256"],
            real_frames=1,
        ),
    )
    with pytest.raises(ValueError, match="forbidden data access"):
        REVISION.validate_rule_artifact(
            summary_path,
            _sha(summary_path),
            training_artifact,
            _audit(),
            revision="original",
        )


def test_revised_bindings_require_frozen_augmentation_and_vial_rows(tmp_path):
    checkpoint = tmp_path / "revised.pt"
    checkpoint.write_bytes(b"revised")
    training_path = _write_json(
        tmp_path / "training.json",
        _training_summary(checkpoint, revision="revised"),
    )
    model = REVISION.validate_training_artifact(
        training_path,
        _sha(training_path),
        checkpoint,
        _sha(checkpoint),
        _audit(),
        revision="revised",
    )
    rule_path = _write_json(
        tmp_path / "rule.json",
        _rule_summary(
            model["checkpoint_sha256"],
            model["summary_sha256"],
            revision="revised",
        ),
    )
    rule = REVISION.validate_rule_artifact(
        rule_path,
        _sha(rule_path),
        model,
        _audit(),
        revision="revised",
    )
    assert rule["rule_config"]["measurement_rows_full_half_open"] == [83, 427]


def test_darkened_validation_is_seeded_and_changes_only_liquid_pixels():
    rgb = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
    mask = _mask()
    first = REVISION.darkened_validation_rgb(rgb, mask, 1100000)
    second = REVISION.darkened_validation_rgb(rgb, mask, 1100000)
    other = REVISION.darkened_validation_rgb(rgb, mask, 1100001)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.array_equal(first[mask != 1], rgb[mask != 1])
    assert np.any(first[mask == 1] != rgb[mask == 1])


def test_variant_metrics_reports_iou_surface_cases_and_bins():
    truth = _mask(150)
    prediction = _mask(152)
    result = REVISION.variant_metrics(
        [truth],
        [prediction],
        [_metadata()],
        SURFACE.frozen_rule_config(),
    )
    assert result["iou"]["macro"] < 1.0
    assert result["surface"]["by_case"]["all"]["frames"] == 1
    assert result["surface"]["by_case"]["all"]["p95_absolute_error_rows"] == 2.0
    assert result["surface"]["by_level_bin_10ml"]["50-60"]["frames"] == 1
    assert len(result["per_frame"]) == 1


def test_variant_metrics_confusion_rows_are_truth_and_columns_prediction():
    truth = _mask(150)
    prediction = truth.copy()
    prediction[200, 50] = 1
    truth[200, 50] = 0
    result = REVISION.variant_metrics(
        [truth],
        [prediction],
        [_metadata()],
        SURFACE.frozen_rule_config(),
    )
    assert result["iou"]["confusion"][0][1] == 1
    assert result["iou"]["confusion"][1][0] == 0


def test_normal_validation_gate_is_inclusive_at_three_rows():
    metrics = {
        "surface": {
            "by_case": {"all": {"p95_absolute_error_rows": 3.0}}
        }
    }
    assert REVISION.enforce_normal_validation_gate(metrics) == 3.0
    metrics["surface"]["by_case"]["all"]["p95_absolute_error_rows"] = 3.01
    with pytest.raises(ValueError, match="exceeds 3 rows"):
        REVISION.enforce_normal_validation_gate(metrics)


def test_compact_metrics_remove_only_per_frame_records():
    value = {
        "normal": {"iou": {"macro": 1.0}, "surface": {}, "per_frame": [1]},
        "darkened": {"iou": {"macro": 0.5}, "surface": {}, "per_frame": [2]},
    }
    compact = REVISION.compact_model_metrics(value)
    assert compact["normal"]["iou"]["macro"] == 1.0
    assert compact["darkened"]["iou"]["macro"] == 0.5
    assert "per_frame" not in compact["normal"]
    assert "per_frame" not in compact["darkened"]

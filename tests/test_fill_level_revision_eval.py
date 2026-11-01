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


def _audit(*, counts=None):
    counts = dict(counts or {"train": 3000, "val": 300})
    return {
        "expected_role_counts": counts,
        "eligible_counts": counts,
        "role_count_source": "batch_summary.batch_contract",
        "batch_summary_sha256": "a" * 64,
        "review_sha256": "b" * 64,
    }


def _training_summary(checkpoint, *, revision="original", test_images=0, audit=None):
    audit = audit or _audit()
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
            "used_counts": dict(audit["eligible_counts"]),
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
        population_contract = TRAINING.dataset_population_contract(audit)
        result["dataset"].update(population_contract)
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
                "dataset_population_contract": population_contract,
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


def test_revised_training_binding_accepts_explicit_combined_population(tmp_path):
    checkpoint = tmp_path / "combined.pt"
    checkpoint.write_bytes(b"combined")
    audit = _audit(counts={"train": 3480, "val": 360})
    training_path = _write_json(
        tmp_path / "combined-training.json",
        _training_summary(checkpoint, revision="revised", audit=audit),
    )

    artifact = REVISION.validate_training_artifact(
        training_path,
        _sha(training_path),
        checkpoint,
        _sha(checkpoint),
        audit,
        revision="revised",
    )

    assert artifact["dataset_population_contract"] == {
        "expected_role_counts": {"train": 3480, "val": 360},
        "eligible_role_counts": {"train": 3480, "val": 360},
        "role_count_source": "batch_summary.batch_contract",
    }


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


def test_darkened_validation_reports_measured_rgb8_saturation_and_value():
    first_rgb = np.array(
        [
            [[255, 0, 0], [128, 128, 128]],
            [[20, 40, 60], [1, 2, 3]],
        ],
        dtype=np.uint8,
    )
    second_rgb = np.array(
        [
            [[0, 0, 64], [0, 0, 0]],
            [[60, 40, 20], [3, 2, 1]],
        ],
        dtype=np.uint8,
    )
    liquid_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    empty_mask = np.zeros((2, 2), dtype=np.uint8)

    first_s, first_v = REVISION.liquid_hsv8_histograms(first_rgb, liquid_mask)
    second_s, second_v = REVISION.liquid_hsv8_histograms(second_rgb, liquid_mask)
    empty_s, empty_v = REVISION.liquid_hsv8_histograms(second_rgb, empty_mask)
    measured = REVISION.summarize_darkened_validation_hsv(
        np.stack((first_s, second_s, empty_s)),
        np.stack((first_v, second_v, empty_v)),
    )
    report = REVISION.darkened_validation_report(measured)

    assert measured["frames"] == {"total": 3, "with_liquid": 2, "without_liquid": 1}
    assert measured["pixels"]["count"] == 4
    assert measured["pixels"]["saturation"]["minimum"] == 0.0
    assert measured["pixels"]["saturation"]["maximum"] == 1.0
    assert measured["pixels"]["value"]["minimum"] == 0.0
    assert measured["pixels"]["value"]["maximum"] == 1.0
    assert measured["per_frame_liquid_mean"]["value"]["mean"] == pytest.approx(
        ((255 + 128) / 2 / 255 + (64 + 0) / 2 / 255) / 2
    )
    assert report["requested_ranges"] == {
        "training_recolour_liquid_saturation": [0.3, 0.85],
        "darkening_target_mean_liquid_value": [0.05, 0.09],
    }
    assert report["measured_post_transform_rgb8_hsv"] == measured
    assert report["rgb8_quantization"]["normalized_channel_step"] == pytest.approx(
        1.0 / 255.0
    )
    assert report["rgb8_quantization"]["darkening_range_applies_to"] == (
        "sampled pre-RGB8 per-frame liquid mean V target"
    )
    assert report["rgb8_quantization"]["training_recolour_saturation_applied"] is False


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

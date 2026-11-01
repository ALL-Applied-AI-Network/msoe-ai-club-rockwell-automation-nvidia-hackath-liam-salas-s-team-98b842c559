"""Contracts for the Step 8 Revision 1 synthetic comparison.

The comparison is deliberately validation-only.  It binds both checkpoints
and both rule summaries by SHA-256, constructs a deterministic darkened copy
of each validation RGB from its liquid label, and reports IoU plus the same
surface metrics used by the frozen row rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from fluidmix import fill_level_surface_eval as surface_eval
from fluidmix import fill_level_training as training


DARK_VALIDATION_SEED = 20261101
NORMAL_VISIBLE_P95_GATE_ROWS = 3.0
REVISED_MEASUREMENT_ROWS_FULL = (83, 427)
ORIGINAL_TRAINING_MODULE_SHA256 = (
    "fb941d36ca8f7b1a03227c3bf8cf46d29a4c7f1d1437e88b6d61106838b8b27d"
)
ORIGINAL_SURFACE_MODULE_SHA256 = (
    "97e260c4b138f5368962ac2773bddf0676f96976b800e3cf724056ae33b1383b"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_CLASS_IDS = {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _exact_hash(path, expected, description):
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected.lower()):
        raise ValueError(f"{description} expected SHA-256 is invalid")
    observed = file_sha256(path)
    if observed != expected.lower():
        raise ValueError(f"{description} SHA-256 mismatch")
    return observed


def validate_training_artifact(
    summary_path,
    expected_summary_sha256,
    checkpoint_path,
    expected_checkpoint_sha256,
    audit,
    *,
    revision,
):
    """Bind a validation-selected full checkpoint without opening any dataset."""
    summary_hash = _exact_hash(
        summary_path, expected_summary_sha256, f"{revision} training summary"
    )
    checkpoint_hash = _exact_hash(
        checkpoint_path, expected_checkpoint_sha256, f"{revision} checkpoint"
    )
    summary = _read_json(summary_path, f"{revision} training summary")
    if summary.get("status") != "complete" or summary.get("profile") != "full":
        raise ValueError(f"{revision} training summary must be a complete full run")
    if not summary.get("checkpoint_candidate"):
        raise ValueError(f"{revision} training summary is not a checkpoint candidate")
    if summary.get("selection_split") != "validation":
        raise ValueError(f"{revision} checkpoint was not selected on validation")
    dataset = summary.get("dataset", {})
    if dataset.get("test_manifests_opened") != 0 or dataset.get("test_images_loaded") != 0:
        raise ValueError(f"{revision} training summary accessed test data")
    if dataset.get("real_frames_loaded", 0) != 0:
        raise ValueError(f"{revision} training summary accessed real data")
    used_counts = dataset.get("used_counts", {})
    for role in ("train", "val"):
        if used_counts.get(role) != audit["eligible_counts"][role]:
            raise ValueError(f"{revision} training summary {role} count mismatch")
    inputs = summary.get("inputs", {})
    if inputs.get("batch_summary_sha256") != audit["batch_summary_sha256"]:
        raise ValueError(f"{revision} training summary batch binding mismatch")
    if inputs.get("exception_review_sha256") != audit["review_sha256"]:
        raise ValueError(f"{revision} training summary exception-review mismatch")
    artifact = summary.get("artifacts", {})
    if artifact.get("best_weights") != Path(checkpoint_path).name:
        raise ValueError(f"{revision} checkpoint filename mismatch")
    if artifact.get("best_weights_sha256") != checkpoint_hash:
        raise ValueError(f"{revision} checkpoint hash is not the selected weight")
    config = summary.get("config", {})
    if config.get("crop_xyxy") != list(training.CROP_XYXY):
        raise ValueError(f"{revision} training crop mismatch")
    if config.get("class_ids") != EXPECTED_CLASS_IDS:
        raise ValueError(f"{revision} class IDs mismatch")
    module_hash = inputs.get("training_module_sha256")
    if not isinstance(module_hash, str) or not SHA256_RE.fullmatch(module_hash):
        raise ValueError(f"{revision} training module hash is invalid")
    if revision == "original":
        if module_hash != ORIGINAL_TRAINING_MODULE_SHA256:
            raise ValueError("original training module is not the Step 7 baseline")
    elif revision == "revised":
        if module_hash != training.file_sha256(training.__file__):
            raise ValueError("revised training module is not the current implementation")
        expected_augmentation = {
            "enabled": True,
            "train_only": True,
            "contract": training.training_augmentation_config(),
        }
        if config.get("augmentation") != expected_augmentation:
            raise ValueError(
                "revised training did not use the frozen train-only augmentation"
            )
        scheduler = config.get("learning_rate_scheduler")
        if not isinstance(scheduler, dict) or scheduler.get("name") != "cosine_annealing":
            raise ValueError("revised training did not use cosine learning-rate decay")
        try:
            initial_lr = float(scheduler["initial_learning_rate"])
            minimum_lr = float(scheduler["minimum_learning_rate"])
            configured_lr = float(config["learning_rate"])
            scheduler_epochs = int(scheduler["t_max_epochs"])
            configured_epochs = int(config["epochs"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("revised learning-rate schedule is incomplete") from error
        if (
            not math.isfinite(initial_lr)
            or not math.isfinite(minimum_lr)
            or initial_lr != configured_lr
            or not 0.0 <= minimum_lr < initial_lr
            or scheduler_epochs != configured_epochs
            or scheduler.get("step_unit") != "epoch"
        ):
            raise ValueError("revised learning-rate schedule does not match the run")
    else:
        raise ValueError(f"unsupported revision label {revision!r}")
    return {
        "summary_sha256": summary_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_filename": Path(checkpoint_path).name,
        "best_epoch": int(summary["best_epoch"]),
        "base_channels": int(config["base_channels"]),
        "training_seed": int(config["seed"]),
        "training_module_sha256": module_hash,
        "augmentation": config.get("augmentation"),
        "learning_rate_scheduler": config.get("learning_rate_scheduler"),
    }


def validate_rule_artifact(
    summary_path,
    expected_summary_sha256,
    training_artifact,
    audit,
    *,
    revision,
):
    """Validate a synthetic-only rule freeze before validation image access."""
    observed = _exact_hash(
        summary_path, expected_summary_sha256, f"{revision} rule summary"
    )
    summary = _read_json(summary_path, f"{revision} rule summary")
    if summary.get("status") != "complete":
        raise ValueError(f"{revision} rule summary must be complete")
    if summary.get("evidence_label") != "measured_on_synthetic_validation_only":
        raise ValueError(f"{revision} rule was not frozen on synthetic validation")
    dataset = summary.get("dataset", {})
    expected_validation_count = int(audit["eligible_counts"]["val"])
    if dataset.get("role") != "val" or dataset.get("frame_count") != (
        expected_validation_count
    ):
        raise ValueError(f"{revision} rule summary validation population mismatch")
    forbidden = (
        "train_manifests_opened",
        "train_images_loaded",
        "test_manifests_opened",
        "test_images_loaded",
        "real_frames_loaded",
    )
    if any(dataset.get(key) != 0 for key in forbidden):
        raise ValueError(f"{revision} rule summary has forbidden data access")
    model = summary.get("model", {})
    inputs = summary.get("inputs", {})
    checkpoint_sha256 = training_artifact["checkpoint_sha256"]
    if model.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{revision} rule summary checkpoint mismatch")
    if inputs.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{revision} rule input checkpoint mismatch")
    if model.get("training_summary_sha256") != training_artifact["summary_sha256"]:
        raise ValueError(f"{revision} rule training-summary mismatch")
    if inputs.get("training_summary_sha256") != training_artifact["summary_sha256"]:
        raise ValueError(f"{revision} rule input training-summary mismatch")
    if inputs.get("batch_summary_sha256") != audit["batch_summary_sha256"]:
        raise ValueError(f"{revision} rule batch-summary mismatch")
    if inputs.get("exception_review_sha256") != audit["review_sha256"]:
        raise ValueError(f"{revision} rule exception-review mismatch")
    evaluator_module_hash = inputs.get("evaluator_module_sha256")
    if revision == "original":
        if evaluator_module_hash != ORIGINAL_SURFACE_MODULE_SHA256:
            raise ValueError("original rule module is not the Step 8 baseline")
    elif revision == "revised":
        if evaluator_module_hash != training.file_sha256(surface_eval.__file__):
            raise ValueError("revised rule module is not the current implementation")
    else:
        raise ValueError(f"unsupported revision label {revision!r}")
    rule_config = summary.get("rule_config")
    if not isinstance(rule_config, dict):
        raise ValueError(f"{revision} rule summary has no rule config")
    if rule_config.get("selection_source") != "synthetic_validation_only":
        raise ValueError(f"{revision} rule was not selected on synthetic validation")
    default_rows = list(surface_eval.DEFAULT_MEASUREMENT_ROWS_FULL)
    measurement_rows = rule_config.get(
        "measurement_rows_full_half_open", default_rows
    )
    if revision == "original":
        if measurement_rows != default_rows:
            raise ValueError("original rule must use the full baseline crop rows")
    else:
        if measurement_rows != list(REVISED_MEASUREMENT_ROWS_FULL):
            raise ValueError(
                "revised rule must use the derived vial rows [83, 427)"
            )
        derivation = rule_config.get("measurement_row_derivation")
        if not isinstance(derivation, dict):
            raise ValueError("revised rule lacks vial-row derivation evidence")
        if derivation.get("selection_source") != (
            "synthetic_validation_label_outline_intersection"
        ):
            raise ValueError("revised vial rows were not derived from validation labels")
        if derivation.get("measurement_rows_full_half_open") != measurement_rows:
            raise ValueError("revised vial-row derivation does not match the rule")
        if derivation.get("frame_count") != expected_validation_count:
            raise ValueError("revised vial-row derivation frame count mismatch")
        if derivation.get("role") != "val":
            raise ValueError("revised vial-row derivation did not use validation")
    # Exercise the pure rule so malformed bounds/thresholds fail before data access.
    height = training.CROP_XYXY[3] - training.CROP_XYXY[1]
    width = training.CROP_XYXY[2] - training.CROP_XYXY[0]
    contract_mask = np.ones((height, width), dtype=np.uint8)
    surface_eval.frame_result(
        surface_eval.ValidationMetadata(
            seed=0,
            level_ml=50.0,
            sampling_band="contract",
            foam=False,
            sticker=False,
            geometry_row=200.0,
        ),
        contract_mask,
        contract_mask,
        rule_config=rule_config,
    )
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{revision} rule summary has no validation aggregate")
    preflight = dataset.get("preflight")
    if not isinstance(preflight, dict):
        raise ValueError(f"{revision} rule summary has no validation preflight")
    ordered_ledger = preflight.get("ordered_hash_ledger_sha256")
    if not isinstance(ordered_ledger, str) or not SHA256_RE.fullmatch(ordered_ledger):
        raise ValueError(f"{revision} rule validation ledger hash is invalid")
    if revision == "revised":
        derivation_ledger = rule_config["measurement_row_derivation"].get(
            "ordered_hash_ledger_sha256"
        )
        if derivation_ledger != ordered_ledger:
            raise ValueError("revised vial-row derivation ledger mismatch")
    return {
        "summary_sha256": observed,
        "rule_config": rule_config,
        "aggregate": aggregate,
        "ordered_hash_ledger_sha256": ordered_ledger,
    }


def darkened_validation_rgb(rgb, mask, sample_seed):
    """Return the fixed-seed, always-darkened validation view."""
    contract = training.training_augmentation_config()[
        "mask_guided_liquid_darkening"
    ]
    seed = np.random.SeedSequence(
        [int(DARK_VALIDATION_SEED), int(sample_seed) & 0xFFFFFFFF]
    )
    rng = np.random.default_rng(seed)
    return training.darken_liquid_pixels(
        rgb,
        mask,
        rng,
        probability=1.0,
        value_range=tuple(contract["value_uniform"]),
    )


def variant_metrics(truth_masks, predicted_masks, metadata, rule_config):
    """Compute IoU and surface metrics for one checkpoint/input variant."""
    truths = [np.asarray(value) for value in truth_masks]
    predictions = [np.asarray(value) for value in predicted_masks]
    if len(truths) != len(predictions) or len(truths) != len(metadata):
        raise ValueError("truth, prediction and metadata counts must match")
    if not truths:
        raise ValueError("variant metrics require at least one frame")
    confusion = np.zeros((4, 4), dtype=np.int64)
    per_frame = []
    for truth, prediction, item in zip(truths, predictions, metadata):
        confusion += training.confusion_matrix(prediction, truth, num_classes=4)
        per_frame.append(
            surface_eval.frame_result(
                item, truth, prediction, rule_config=rule_config
            )
        )
    per_class, macro = training.iou_from_confusion(confusion)
    return {
        "iou": {
            "macro": float(macro),
            "per_class": [float(value) for value in per_class],
            "confusion": confusion.tolist(),
        },
        "surface": surface_eval.summarize_results(per_frame),
        "per_frame": per_frame,
    }


def enforce_normal_validation_gate(revised_normal_metrics):
    """Stop Revision 1 when normal visible-surface p95 exceeds three rows."""
    try:
        value = revised_normal_metrics["surface"]["by_case"]["all"][
            "p95_absolute_error_rows"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("revised normal metrics lack visible-surface p95") from error
    if value is None or not math.isfinite(float(value)):
        raise ValueError("revised normal visible-surface p95 is not finite")
    if float(value) > NORMAL_VISIBLE_P95_GATE_ROWS:
        raise ValueError(
            "revised normal visible-surface p95 exceeds "
            f"{NORMAL_VISIBLE_P95_GATE_ROWS:g} rows: {float(value):g}"
        )
    return float(value)


def compact_model_metrics(metrics):
    """Drop per-frame records while retaining both evidence variants."""
    return {
        variant: {key: value for key, value in result.items() if key != "per_frame"}
        for variant, result in metrics.items()
    }

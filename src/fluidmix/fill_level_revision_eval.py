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
from PIL import Image

from fluidmix import fill_level_surface_eval as surface_eval
from fluidmix import fill_level_training as training


DARK_VALIDATION_SEED = 20261101
NORMAL_VISIBLE_P95_GATE_ROWS = 3.0
HIDDEN_JOINT_TARGET_RATE = 0.95
REVISED_MEASUREMENT_ROWS_FULL = (83, 427)
ORIGINAL_TRAINING_MODULE_SHA256 = (
    "fb941d36ca8f7b1a03227c3bf8cf46d29a4c7f1d1437e88b6d61106838b8b27d"
)
ORIGINAL_SURFACE_MODULE_SHA256 = (
    "97e260c4b138f5368962ac2773bddf0676f96976b800e3cf724056ae33b1383b"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_CLASS_IDS = {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3}
REVISION2_VALIDATION_COHORTS = {
    "old_step6_validation": {
        "source_component": "step6",
        "seed_range_inclusive": list(surface_eval.STEP6_VALIDATION_SEED_RANGE),
        "frame_count": surface_eval.STEP6_VALIDATION_FRAME_COUNT,
    },
    "new_exact_v4_validation": {
        "source_component": "exact_v4",
        "seed_range_inclusive": [1101000, 1101059],
        "frame_count": 60,
    },
}


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


def revision2_validation_cohort_indices(metadata):
    """Partition the complete combined validation set into its two source cohorts."""

    items = tuple(metadata)
    result = {}
    claimed = set()
    for name, contract in REVISION2_VALIDATION_COHORTS.items():
        first, last = contract["seed_range_inclusive"]
        indices = tuple(
            index
            for index, item in enumerate(items)
            if first <= int(item.seed) <= last
        )
        observed = [int(items[index].seed) for index in indices]
        expected = list(range(first, last + 1))
        if observed != expected or len(indices) != contract["frame_count"]:
            raise ValueError(f"combined validation cohort {name} is incomplete")
        claimed.update(indices)
        result[name] = indices
    if claimed != set(range(len(items))):
        raise ValueError("combined validation contains an unrecognized cohort")
    return result


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
        expected_population_contract = training.dataset_population_contract(audit)
        if config.get("dataset_population_contract") != expected_population_contract:
            raise ValueError("revised training config population contract mismatch")
        recorded_dataset_population = {
            key: dataset.get(key) for key in expected_population_contract
        }
        if recorded_dataset_population != expected_population_contract:
            raise ValueError("revised training summary population contract mismatch")
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
        "dataset_population_contract": config.get("dataset_population_contract"),
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
        if rule_config.get("hidden_rule_version") != "holder_band_v2":
            raise ValueError("revised rule must use holder_band_v2")
        hidden_derivation = rule_config.get("hidden_rule_derivation")
        if not isinstance(hidden_derivation, dict):
            raise ValueError("revised rule lacks hidden-rule derivation evidence")
        if hidden_derivation.get("selection_source") != (
            "synthetic_validation_labels_and_manifest_geometry"
        ):
            raise ValueError(
                "revised hidden rule was not derived from validation labels and geometry"
            )
        if hidden_derivation.get("role") != "val":
            raise ValueError("revised hidden-rule derivation did not use validation")
        if hidden_derivation.get("frame_count") != (
            surface_eval.STEP6_VALIDATION_FRAME_COUNT
        ):
            raise ValueError("revised hidden-rule derivation frame count mismatch")
        if hidden_derivation.get("ordered_hash_ledger_sha256") != (
            surface_eval.STEP6_VALIDATION_LEDGER_SHA256
        ):
            raise ValueError("revised hidden-rule derivation is not bound to Step 6")
        if hidden_derivation.get("source_component") != "step6":
            raise ValueError("revised hidden-rule derivation source component mismatch")
        if hidden_derivation.get("source_batch_summary_sha256") != (
            surface_eval.STEP6_VALIDATION_BATCH_SUMMARY_SHA256
        ):
            raise ValueError("revised hidden-rule derivation source summary mismatch")
        copied_hidden_constants = (
            "holder_band_rows_full_inclusive",
            "surface_oval_height_rows",
            "visibility_columns_full_half_open",
            "minimum_liquid_pixels_per_visibility_row",
        )
        for name in copied_hidden_constants:
            if hidden_derivation.get(name) != (
                surface_eval.APPROVED_HIDDEN_RULE_CONSTANTS[name]
            ):
                raise ValueError(
                    f"revised hidden-rule constant {name} is not the approved Step 6 value"
                )
            if rule_config.get(name) != hidden_derivation.get(name):
                raise ValueError(
                    f"revised hidden-rule constant {name} does not match derivation"
                )
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


def liquid_hsv8_histograms(rgb, mask):
    """Count post-transform HSV8 saturation and value for liquid pixels."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = training.validate_mask_array(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and mask must have matching HxW geometry")
    liquid = mask == 1
    if not np.any(liquid):
        empty = np.zeros(256, dtype=np.int64)
        return empty.copy(), empty
    hsv8 = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"), dtype=np.uint8)
    saturation = np.bincount(hsv8[..., 1][liquid], minlength=256).astype(
        np.int64, copy=False
    )
    value = np.bincount(hsv8[..., 2][liquid], minlength=256).astype(
        np.int64, copy=False
    )
    return saturation, value


def _histogram_distribution(histogram):
    histogram = np.asarray(histogram, dtype=np.int64)
    count = int(histogram.sum())
    if count == 0:
        return {
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
            "quantized_levels_observed": 0,
        }
    cumulative = np.cumsum(histogram)

    def ordered_value(rank):
        return int(np.searchsorted(cumulative, int(rank), side="right"))

    def quantile(fraction):
        rank = float(fraction) * float(count - 1)
        lower_rank = int(math.floor(rank))
        upper_rank = int(math.ceil(rank))
        weight = rank - lower_rank
        lower = ordered_value(lower_rank)
        upper = ordered_value(upper_rank)
        return ((1.0 - weight) * lower + weight * upper) / 255.0

    levels = np.flatnonzero(histogram)
    weighted_sum = float(np.dot(np.arange(256, dtype=np.float64), histogram))
    return {
        "minimum": float(levels[0]) / 255.0,
        "p05": quantile(0.05),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "maximum": float(levels[-1]) / 255.0,
        "mean": weighted_sum / float(count) / 255.0,
        "quantized_levels_observed": int(levels.size),
    }


def _float_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def summarize_darkened_validation_hsv(saturation_histograms, value_histograms):
    """Summarize actual post-transform liquid HSV8 pixels and frame means."""
    saturation = np.asarray(saturation_histograms)
    value = np.asarray(value_histograms)
    if saturation.ndim != 2 or saturation.shape[1:] != (256,):
        raise ValueError("saturation histograms must have shape frames x 256")
    if value.shape != saturation.shape:
        raise ValueError("saturation and value histogram shapes must match")
    for name, histograms in (("saturation", saturation), ("value", value)):
        if not np.issubdtype(histograms.dtype, np.integer):
            raise ValueError(f"{name} histograms must contain integer counts")
        if np.any(histograms < 0):
            raise ValueError(f"{name} histograms cannot contain negative counts")
    saturation = saturation.astype(np.int64, copy=False)
    value = value.astype(np.int64, copy=False)
    saturation_counts = saturation.sum(axis=1)
    value_counts = value.sum(axis=1)
    if not np.array_equal(saturation_counts, value_counts):
        raise ValueError("saturation and value liquid-pixel counts must match per frame")

    with_liquid = value_counts > 0
    channel_values = np.arange(256, dtype=np.float64)
    frame_saturation_means = (
        saturation[with_liquid] @ channel_values / saturation_counts[with_liquid] / 255.0
    )
    frame_value_means = (
        value[with_liquid] @ channel_values / value_counts[with_liquid] / 255.0
    )
    total_saturation = saturation.sum(axis=0)
    total_value = value.sum(axis=0)
    return {
        "frames": {
            "total": int(saturation.shape[0]),
            "with_liquid": int(np.count_nonzero(with_liquid)),
            "without_liquid": int(np.count_nonzero(~with_liquid)),
        },
        "pixels": {
            "count": int(total_value.sum()),
            "saturation": _histogram_distribution(total_saturation),
            "value": _histogram_distribution(total_value),
        },
        "per_frame_liquid_mean": {
            "saturation": _float_distribution(frame_saturation_means),
            "value": _float_distribution(frame_value_means),
        },
    }


def darkened_validation_report(measured_post_transform_rgb8_hsv):
    """Build the auditable requested-versus-measured dark-validation contract."""
    augmentation = training.training_augmentation_config()
    recolour = augmentation["mask_guided_liquid_recolour"]
    darkening = augmentation["mask_guided_liquid_darkening"]
    return {
        "seed": DARK_VALIDATION_SEED,
        "per_frame_seed_derivation": (
            "numpy SeedSequence([fixed_seed, uint32(validation_seed)])"
        ),
        "probability": 1.0,
        "requested_ranges": {
            "training_recolour_liquid_saturation": list(
                recolour["saturation_uniform"]
            ),
            "darkening_target_mean_liquid_value": list(darkening["value_uniform"]),
        },
        "mask_class": "liquid",
        "non_liquid_pixels_changed": 0,
        "implementation": "training.darken_liquid_pixels",
        "training_augmentation_contract": augmentation,
        "measured_post_transform_rgb8_hsv": measured_post_transform_rgb8_hsv,
        "rgb8_quantization": {
            "source": "post-transform uint8 RGB converted with Pillow to uint8 HSV",
            "normalization": "HSV8 saturation and value divided by 255",
            "normalized_channel_step": 1.0 / 255.0,
            "darkening_range_applies_to": (
                "sampled pre-RGB8 per-frame liquid mean V target"
            ),
            "training_recolour_saturation_applied": False,
            "interpretation": (
                "The darkened-validation variant applies darkening only. Requested "
                "multiplicative liquid-texture scaling and RGB8/HSV8 round trips mean "
                "individual pixels and measured frame means need not equal the requested "
                "floating-point target or remain exactly unchanged in hue/saturation."
            ),
        },
    }


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


def hidden_rule_before_after_by_cohort(
    truth_masks,
    predicted_variants,
    metadata,
    after_rule_config,
):
    """Rescore one model's fixed predictions with old/new hidden rules by cohort."""

    truths = tuple(np.asarray(value) for value in truth_masks)
    items = tuple(metadata)
    if len(truths) != len(items):
        raise ValueError("truth and metadata counts must match")
    if set(predicted_variants) != {"normal", "darkened"}:
        raise ValueError("hidden-rule comparison requires normal and darkened predictions")
    if after_rule_config.get("hidden_rule_version") != "holder_band_v2":
        raise ValueError("hidden-rule comparison after rule must use holder_band_v2")
    cohorts = revision2_validation_cohort_indices(items)
    before_rule_config = surface_eval.frozen_rule_config(
        measurement_columns_full=tuple(
            after_rule_config["measurement_columns_full_half_open"]
        ),
        measurement_rows_full=tuple(
            after_rule_config["measurement_rows_full_half_open"]
        ),
        min_liquid_pixels=int(
            after_rule_config["minimum_liquid_pixels_per_row"]
        ),
        min_blocked_pixels=int(
            after_rule_config["minimum_blocked_pixels_per_row"]
        ),
        row_bounds_evidence=after_rule_config.get("measurement_row_derivation"),
    )

    variants = {}
    for variant in ("normal", "darkened"):
        predictions = tuple(np.asarray(value) for value in predicted_variants[variant])
        if len(predictions) != len(truths):
            raise ValueError(
                f"{variant} prediction count does not match combined validation"
            )
        cohort_metrics = {}
        for name, indices in cohorts.items():
            scored = {}
            for label, rule_config in (
                ("before", before_rule_config),
                ("after", after_rule_config),
            ):
                per_frame = [
                    surface_eval.frame_result(
                        items[index],
                        truths[index],
                        predictions[index],
                        rule_config=rule_config,
                    )
                    for index in indices
                ]
                scored[label] = surface_eval.summarize_results(per_frame)
            cohort_metrics[name] = {
                **REVISION2_VALIDATION_COHORTS[name],
                **scored,
            }
        variants[variant] = cohort_metrics
    target_cohorts = {}
    for name in REVISION2_VALIDATION_COHORTS:
        hidden = variants["normal"][name]["after"]["by_case"]["hidden"]
        frame_count = int(hidden["frames"])
        successful_frames = int(hidden["geometry_in_predicted_range"])
        predicted_hidden_rate = hidden["predicted_hidden_rate"]
        true_in_range_rate = hidden["true_in_range_over_all_frames_rate"]
        predicted_hidden_passed = (
            predicted_hidden_rate is not None
            and float(predicted_hidden_rate) >= HIDDEN_JOINT_TARGET_RATE
        )
        true_in_range_passed = (
            true_in_range_rate is not None
            and float(true_in_range_rate) >= HIDDEN_JOINT_TARGET_RATE
        )
        target_cohorts[name] = {
            "hidden_ground_truth_frames": frame_count,
            "predicted_hidden_rate": predicted_hidden_rate,
            "predicted_hidden_rate_passed": predicted_hidden_passed,
            "successful_frames": successful_frames,
            "true_in_range_over_all_frames_rate": true_in_range_rate,
            "true_in_range_over_all_frames_rate_passed": true_in_range_passed,
            "passed": predicted_hidden_passed and true_in_range_passed,
        }
    return {
        "evidence_label": (
            "same_revised_model_predictions_rescored_without_reinference"
        ),
        "before_hidden_rule_version": "contiguous_blocked_v1",
        "after_hidden_rule_version": "holder_band_v2",
        "cohorts": REVISION2_VALIDATION_COHORTS,
        "variants": variants,
        "normal_after_hidden_target": {
            "minimum_rate": HIDDEN_JOINT_TARGET_RATE,
            "cohorts": target_cohorts,
            "all_cohorts_passed": all(
                value["passed"] for value in target_cohorts.values()
            ),
        },
    }


def enforce_hidden_validation_gate(hidden_rule_before_after):
    """Require both normal/after hidden acceptance rates on both source cohorts."""

    try:
        target = hidden_rule_before_after["normal_after_hidden_target"]
        minimum = float(target["minimum_rate"])
        cohorts = target["cohorts"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("hidden acceptance gate evidence is malformed") from error
    if minimum != HIDDEN_JOINT_TARGET_RATE:
        raise ValueError("hidden acceptance gate threshold is not the frozen 95%")
    if set(cohorts) != set(REVISION2_VALIDATION_COHORTS):
        raise ValueError("hidden acceptance gate cohorts do not match Revision 2")
    failures = []
    for name in REVISION2_VALIDATION_COHORTS:
        values = cohorts[name]
        for metric in (
            "predicted_hidden_rate",
            "true_in_range_over_all_frames_rate",
        ):
            observed = values.get(metric)
            if not isinstance(observed, (int, float)) or not math.isfinite(observed):
                failures.append(f"{name} {metric}=unavailable")
            elif float(observed) < minimum:
                failures.append(f"{name} {metric}={float(observed):.6f}")
    if failures:
        raise ValueError(
            "hidden acceptance gate requires both rates >= 0.95: "
            + "; ".join(failures)
        )
    if target.get("all_cohorts_passed") is not True:
        raise ValueError("hidden acceptance gate pass flags are inconsistent")
    return target


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

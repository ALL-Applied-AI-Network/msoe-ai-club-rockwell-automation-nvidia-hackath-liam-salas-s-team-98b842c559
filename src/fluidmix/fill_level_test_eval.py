"""Held-out synthetic twin-test contracts for the FillCam segmenter.

All authority and freeze artifacts are validated before callers may discover
test manifests. Train and validation directories are skipped before any state
or manifest beneath them is read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from fluidmix import fill_level_ruler as ruler_module
from fluidmix import fill_level_surface_eval as surface_eval
from fluidmix import fill_level_training as training


EXPECTED_TEST_CHUNKS = {
    "17-test-2100000-2100199": ("old_step6_test", "step6", "17-test-2100000-2100199", 2100000, 2100199, 200),
    "18-test-2100200-2100399": ("old_step6_test", "step6", "18-test-2100200-2100399", 2100200, 2100399, 200),
    "19-test-2100400-2100499": ("old_step6_test", "step6", "19-test-2100400-2100499", 2100400, 2100499, 100),
    "29-test-2101000-2101059": ("new_exact_v4_test", "exact_v4", "09-test-2101000-2101059", 2101000, 2101059, 60),
}
EXPECTED_TEST_COUNT = 560
EXPECTED_TEST_SEEDS = tuple(range(2100000, 2100500)) + tuple(range(2101000, 2101060))
EXPECTED_COHORT_COUNTS = {"old_step6_test": 500, "new_exact_v4_test": 60}
EXPECTED_CASE_COUNTS = {
    "clean": 114,
    "foam": 289,
    "sticker": 329,
    "edge": 283,
    "hidden": 142,
}
CASE_NAMES = tuple(EXPECTED_CASE_COUNTS)
EXPECTED_COMPONENT_SUMMARIES = {
    "step6": "e736d17ee7db384fa9b15cadc97422bb095b2a82ef1fb6b879040df0424d88c0",
    "exact_v4": "e224ca0fcf18c0f833d53d1f1a61422a5067470ff85eaa079514ef9da91774ab",
}
EXPECTED_VALIDATION_LEDGER = "cffb8c32eab71867ad7b91d78e02b866d73604f06a243deaadf00d10933351bf"
EXPECTED_CHECKPOINT_REVIEW_SHA256 = "ddfba0005d5930a72cd02b69b137330aa6c4735156a92fb5fd41866da33c71d1"
EXPECTED_CALIBRATION_FAILURE_SHA256 = "bc34399aa64a26f884686a2f09f82f7bf0f2297eb8dd2ca59c44e11824878729"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class TestMetadata:
    seed: int
    cohort: str
    level_ml: float
    sampling_band: str
    foam: bool
    sticker: bool
    geometry_row: float

    @property
    def cases(self):
        cases = []
        if not self.foam and not self.sticker:
            cases.append("clean")
        if self.foam:
            cases.append("foam")
        if self.sticker:
            cases.append("sticker")
        if "edge" in self.sampling_band:
            cases.append("edge")
        if "hidden" in self.sampling_band:
            cases.append("hidden")
        return tuple(cases)


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _expected_hash(value, description):
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value.lower()):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value.lower()


def _require_file_hash(path, expected, description):
    expected = _expected_hash(expected, f"expected {description} hash")
    observed = training.file_sha256(path)
    if observed != expected:
        raise ValueError(f"{description} SHA-256 does not match the frozen hash")
    return observed


def validate_batch_summary(path, expected_sha256):
    """Validate the combined Revision 2 summary and extract exact test contracts."""
    observed_hash = _require_file_hash(path, expected_sha256, "batch summary")
    value = _read_json(path, "batch summary")
    if value.get("schema_version") != 1 or not value.get("integrity", {}).get("passed"):
        raise ValueError("batch summary integrity must pass")
    contract = value.get("batch_contract", {})
    required_contract = {
        "expected_chunks": 30,
        "expected_role_counts": {"test": 560, "train": 3480, "val": 360},
        "new_chunk_index_remap": "exact-v4 00-09 -> combined 20-29",
        "test_chunks_linked": 0,
        "training_root_roles": ["train", "val"],
    }
    if contract != required_contract:
        raise ValueError("batch summary contract does not match Revision 2")
    components = value.get("components", {})
    if any(
        components.get(name, {}).get("summary_sha256") != expected
        for name, expected in EXPECTED_COMPONENT_SUMMARIES.items()
    ):
        raise ValueError("batch summary component hashes do not match Revision 2")
    chunks = value.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("batch summary chunks must be a list")
    approved = {}
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("role") != "test":
            continue
        name = str(chunk.get("chunk"))
        if name in approved:
            raise ValueError(f"batch summary repeats test chunk {name}")
        if name not in EXPECTED_TEST_CHUNKS:
            raise ValueError(f"batch summary contains unexpected test chunk {name}")
        cohort, component, source_chunk, start, end, count = EXPECTED_TEST_CHUNKS[name]
        if (
            int(chunk.get("seed_start", -1)) != start
            or int(chunk.get("seed_end", -1)) != end
            or int(chunk.get("frame_count", -1)) != count
            or chunk.get("integrity_passed") is not True
            or chunk.get("component") != component
            or chunk.get("source_chunk") != source_chunk
        ):
            raise ValueError(f"batch summary test chunk {name} contract is invalid")
        manifest_hash = _expected_hash(
            chunk.get("manifest_sha256"), f"test chunk {name} manifest hash"
        )
        attempt = chunk.get("attempt")
        if not isinstance(attempt, str) or not attempt.startswith("attempt-"):
            raise ValueError(f"test chunk {name} has no approved attempt")
        approved[name] = {
            "role": "test",
            "attempt": attempt,
            "manifest_sha256": manifest_hash,
            "seed_start": start,
            "seed_end": end,
            "frame_count": count,
            "cohort": cohort,
            "component": component,
            "source_chunk": source_chunk,
        }
    if set(approved) != set(EXPECTED_TEST_CHUNKS):
        raise ValueError("batch summary does not approve exactly four Revision 2 test chunks")
    coverage = value.get("case_coverage", {}).get("by_role", {}).get("test", {})
    expected_coverage = {
        "clean": EXPECTED_CASE_COUNTS["clean"],
        "foam": EXPECTED_CASE_COUNTS["foam"],
        "sticker": EXPECTED_CASE_COUNTS["sticker"],
        "edge_zone": EXPECTED_CASE_COUNTS["edge"],
        "hidden": EXPECTED_CASE_COUNTS["hidden"],
    }
    if any(coverage.get(name) != count for name, count in expected_coverage.items()):
        raise ValueError("batch summary test case counts are not the frozen 560-frame split")
    return {
        "artifact_sha256": observed_hash,
        "approved_chunks": approved,
        "case_counts": dict(EXPECTED_CASE_COUNTS),
        "cohort_counts": dict(EXPECTED_COHORT_COUNTS),
    }


def validate_model_binding(
    training_summary_path,
    expected_training_summary_sha256,
    checkpoint_path,
    expected_weights_sha256,
    expected_training_module_sha256,
):
    """Bind model bytes and current loader code to the completed Step 7 run."""
    summary_hash = _require_file_hash(
        training_summary_path, expected_training_summary_sha256, "training summary"
    )
    weights_hash = _require_file_hash(
        checkpoint_path, expected_weights_sha256, "checkpoint"
    )
    module_hash = _require_file_hash(
        training.__file__, expected_training_module_sha256, "training module"
    )
    value = _read_json(training_summary_path, "training summary")
    if (
        value.get("status") != "complete"
        or value.get("profile") != "full"
        or value.get("checkpoint_candidate") is not True
        or value.get("selection_split") != "validation"
    ):
        raise ValueError("training summary is not the validation-selected full run")
    dataset = value.get("dataset", {})
    if dataset.get("test_manifests_opened") != 0 or dataset.get("test_images_loaded") != 0:
        raise ValueError("training run must leave held-out test data untouched")
    inputs = value.get("inputs", {})
    if inputs.get("training_module_sha256") != module_hash:
        raise ValueError("training summary module hash does not match current code")
    artifacts = value.get("artifacts", {})
    if (
        artifacts.get("best_weights") != Path(checkpoint_path).name
        or artifacts.get("best_weights_sha256") != weights_hash
    ):
        raise ValueError("training summary does not bind the selected checkpoint")
    config = value.get("config", {})
    if (
        config.get("crop_xyxy") != list(training.CROP_XYXY)
        or config.get("class_ids")
        != {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3}
    ):
        raise ValueError("training summary model contract is incompatible")
    return {
        "training_summary_sha256": summary_hash,
        "weights_sha256": weights_hash,
        "weights_filename": Path(checkpoint_path).name,
        "training_module_sha256": module_hash,
        "best_epoch": int(value["best_epoch"]),
        "base_channels": int(config["base_channels"]),
        "training_seed": int(config["seed"]),
        "class_ids": dict(config["class_ids"]),
        "crop_xyxy": list(config["crop_xyxy"]),
    }


def validate_frozen_rule(path, expected_sha256, model_binding, batch_binding):
    """Require the exact validation-frozen rule before test discovery."""
    artifact_hash = _require_file_hash(path, expected_sha256, "frozen rule")
    value = _read_json(path, "frozen rule")
    if (
        value.get("status") not in ("complete", "frozen_before_test")
        or value.get("purpose") != "Step 8 validation-only surface-row rule freeze"
    ):
        raise ValueError("surface rule is not the completed Step 8 validation freeze")
    dataset = value.get("dataset", {})
    required_safety = {
        "role": "val",
        "frame_count": 360,
        "case_counts": {"clean": 78, "foam": 177, "sticker": 207, "edge": 182, "hidden": 90},
        "train_manifests_opened": 0,
        "train_images_loaded": 0,
        "test_manifests_opened": 0,
        "test_images_loaded": 0,
        "real_frames_loaded": 0,
    }
    if any(dataset.get(key) != expected for key, expected in required_safety.items()):
        raise ValueError("surface rule is not the leak-free 360-frame validation freeze")
    if dataset.get("preflight") != {
        "samples_checked": 360,
        "files_checked": 720,
        "ordered_hash_ledger_sha256": EXPECTED_VALIDATION_LEDGER,
    }:
        raise ValueError("surface rule validation ledger does not match Revision 2")
    inputs = value.get("inputs", {})
    model = value.get("model", {})
    if (
        inputs.get("checkpoint_sha256") != model_binding["weights_sha256"]
        or model.get("checkpoint_sha256") != model_binding["weights_sha256"]
        or model.get("training_module_sha256")
        != model_binding["training_module_sha256"]
        or inputs.get("training_summary_sha256")
        != model_binding["training_summary_sha256"]
        or inputs.get("batch_summary_sha256") != batch_binding["artifact_sha256"]
    ):
        raise ValueError("surface rule model binding does not match Step 7")
    current_surface_hash = training.file_sha256(surface_eval.__file__)
    if inputs.get("evaluator_module_sha256") != current_surface_hash:
        raise ValueError("surface rule module hash does not match current code")
    config = value.get("rule_config")
    if not isinstance(config, dict):
        raise ValueError("surface rule artifact has no frozen config")
    if (
        config.get("selection_source") != "synthetic_validation_only"
        or config.get("crop_xyxy") != list(training.CROP_XYXY)
        or config.get("class_ids") != {"liquid": 1, "blocked": 3}
        or config.get("hidden_rule_version") != "holder_band_v2"
        or config.get("measurement_rows_full_half_open") != [83, 427]
        or config.get("holder_band_rows_full_inclusive") != [217, 275]
        or config.get("surface_oval_height_rows") != 17
        or config.get("visibility_columns_full_half_open") != [279, 340]
        or config.get("minimum_liquid_pixels_per_visibility_row") != 3
    ):
        raise ValueError("surface rule config is incompatible")
    try:
        options = {
            "measurement_columns_full": tuple(
                int(item) for item in config["measurement_columns_full_half_open"]
            ),
            "measurement_rows_full": tuple(
                int(item) for item in config["measurement_rows_full_half_open"]
            ),
            "min_liquid_pixels": int(config["minimum_liquid_pixels_per_row"]),
            "min_blocked_pixels": int(config["minimum_blocked_pixels_per_row"]),
            "holder_band_rows_full": tuple(
                int(item) for item in config["holder_band_rows_full_inclusive"]
            ),
            "surface_oval_height_rows": int(config["surface_oval_height_rows"]),
            "visibility_columns_full": tuple(
                int(item) for item in config["visibility_columns_full_half_open"]
            ),
            "min_visibility_liquid_pixels": int(
                config["minimum_liquid_pixels_per_visibility_row"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("surface rule options are incomplete") from error
    height = training.CROP_XYXY[3] - training.CROP_XYXY[1]
    width = training.CROP_XYXY[2] - training.CROP_XYXY[0]
    surface_eval.surface_prediction(np.zeros((height, width), dtype=np.uint8), **options)
    return {
        "artifact_sha256": artifact_hash,
        "surface_module_sha256": current_surface_hash,
        "rule_config": config,
        "rule_options": options,
    }


def validate_frozen_ruler(
    path,
    expected_sha256,
    model_binding,
    frozen_rule,
):
    """Require the authorized hand-measured ruler frozen before held-out access."""
    artifact_hash = _require_file_hash(path, expected_sha256, "frozen ruler")
    value = _read_json(path, "frozen ruler")
    if (
        value.get("status") != "complete"
        or value.get("freeze_state") != "frozen_before_step9_and_step10"
        or value.get("source")
        != "hand_measured_step3_targets_after_failed_system_calibration"
        or value.get("checkpoint_review_sha256")
        != EXPECTED_CHECKPOINT_REVIEW_SHA256
    ):
        raise ValueError("ruler is not the authorized pre-test hand-measured freeze")
    calibration = value.get("system_calibration", {})
    if (
        calibration.get("status") != "failed"
        or calibration.get("ruler_created") is not False
        or calibration.get("failed_medians_not_used") is not True
        or calibration.get("failure_summary_sha256")
        != EXPECTED_CALIBRATION_FAILURE_SHA256
    ):
        raise ValueError("hand ruler must disclose and exclude failed calibration medians")
    inputs = value.get("inputs", {})
    current_ruler_hash = training.file_sha256(ruler_module.__file__)
    if (
        inputs.get("frozen_validation_rule_sha256")
        != frozen_rule["artifact_sha256"]
        or inputs.get("weights_sha256") != model_binding["weights_sha256"]
        or inputs.get("training_module_sha256")
        != model_binding["training_module_sha256"]
        or inputs.get("training_summary_sha256")
        != model_binding["training_summary_sha256"]
        or inputs.get("surface_rule_module_sha256")
        != frozen_rule["surface_module_sha256"]
        or inputs.get("ruler_module_sha256") != current_ruler_hash
        or any(
            inputs.get(counter) != 0
            for counter in (
                "train_manifests_opened",
                "train_images_loaded",
                "validation_manifests_opened",
                "validation_images_loaded",
                "test_manifests_opened",
                "test_images_loaded",
                "real_check_frames_loaded",
            )
        )
    ):
        raise ValueError("frozen ruler provenance does not match the pre-test chain")
    model = value.get("model", {})
    if model.get("weights_sha256") != model_binding["weights_sha256"]:
        raise ValueError("frozen ruler model does not match the held-out checkpoint")
    fitted = value.get("ruler")
    if not isinstance(fitted, dict):
        raise ValueError("frozen ruler artifact has no ruler")
    try:
        slope = float(fitted["slope_ml_per_row"])
        intercept = float(fitted["intercept_ml"])
        rows_per_ml = float(fitted["rows_per_ml"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen ruler coefficients are invalid") from error
    if (
        not all(math.isfinite(item) for item in (slope, intercept, rows_per_ml))
        or slope >= 0.0
        or rows_per_ml <= 0.0
    ):
        raise ValueError("frozen ruler coefficients are not a descending image-row ruler")
    points = fitted.get("calibration_points")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("frozen ruler needs exactly two calibration points")
    observed_volumes = sorted(float(item.get("volume_ml")) for item in points)
    if observed_volumes != [30.0, 80.0]:
        raise ValueError("frozen ruler calibration points must be 30 and 80 mL")
    expected_points = {1: (30.0, 329.0), 13: (80.0, 177.0)}
    for point in points:
        vial = int(point.get("vial", -1))
        if (
            vial not in expected_points
            or float(point.get("volume_ml")) != expected_points[vial][0]
            or float(point.get("median_row")) != expected_points[vial][1]
            or point.get("source") != "hand_measured_step3_target"
        ):
            raise ValueError("frozen ruler points are not the Step 3 hand measurements")
        if not math.isclose(
            ruler_module.row_to_ml(fitted, point["median_row"]),
            float(point["volume_ml"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("frozen ruler does not reproduce its calibration points")
    if not math.isclose(rows_per_ml, 3.04, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("frozen ruler must use exactly the 3.04 rows/mL hand ruler")
    return {
        "artifact_sha256": artifact_hash,
        "ruler_module_sha256": current_ruler_hash,
        "ruler": fitted,
        "source": value["source"],
    }


def validate_authority(
    *,
    batch_summary_path,
    batch_summary_sha256,
    training_summary_path,
    training_summary_sha256,
    checkpoint_path,
    weights_sha256,
    training_module_sha256,
    frozen_rule_path,
    frozen_rule_sha256,
    frozen_ruler_path,
    frozen_ruler_sha256,
):
    """Run every immutable pre-test gate without accessing the test dataset root."""
    batch = validate_batch_summary(batch_summary_path, batch_summary_sha256)
    model = validate_model_binding(
        training_summary_path,
        training_summary_sha256,
        checkpoint_path,
        weights_sha256,
        training_module_sha256,
    )
    rule = validate_frozen_rule(frozen_rule_path, frozen_rule_sha256, model, batch)
    ruler = validate_frozen_ruler(frozen_ruler_path, frozen_ruler_sha256, model, rule)
    return {"model": model, "rule": rule, "ruler": ruler, "batch": batch}


def _complete_attempt(chunk_dir, expected_attempt):
    chunk_dir = Path(chunk_dir)
    complete = []
    for attempt in sorted(chunk_dir.iterdir(), key=lambda item: item.name):
        if not attempt.is_dir():
            continue
        try:
            state = (attempt / "state.txt").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if state == "COMPLETE":
            complete.append(attempt)
    if len(complete) != 1:
        raise ValueError(
            f"test chunk {chunk_dir.name} needs exactly one COMPLETE attempt; "
            f"found {len(complete)}"
        )
    if complete[0].name != expected_attempt:
        raise ValueError(
            f"test chunk {chunk_dir.name} COMPLETE attempt is {complete[0].name}; "
            f"approved attempt is {expected_attempt}"
        )
    return complete[0]


def _safe_output_path(output_dir, relative_path, description):
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {description} path: {relative_path}")
    root = Path(output_dir).resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"unsafe {description} path: {relative_path}")
    return resolved


def discover_test_frames(dataset_root, batch_contract):
    """Discover only approved test chunks; train/val children remain unopened."""
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    approved = batch_contract["approved_chunks"]
    observed_test = set()
    for path in dataset_root.iterdir():
        match = training.CHUNK_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None and match.group("role") == "test":
            observed_test.add(path.name)
    if observed_test != set(approved):
        raise ValueError(
            "test chunk set does not match the approved batch: "
            f"missing={sorted(set(approved) - observed_test)}, "
            f"extra={sorted(observed_test - set(approved))}"
        )

    samples = []
    metadata = {}
    for chunk_name in sorted(approved):
        contract = approved[chunk_name]
        attempt = _complete_attempt(dataset_root / chunk_name, contract["attempt"])
        output_dir = attempt / "output"
        manifest_path = output_dir / "manifest.jsonl"
        try:
            manifest_bytes = manifest_path.read_bytes()
            lines = manifest_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"could not read test manifest {manifest_path}: {error}") from error
        if hashlib.sha256(manifest_bytes).hexdigest() != contract["manifest_sha256"]:
            raise ValueError(f"test chunk {chunk_name} manifest hash is not approved")
        chunk_count = 0
        for line_number, raw_line in enumerate(lines, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid test manifest JSON {manifest_path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict) or record.get("role") != "test":
                raise ValueError(
                    f"test manifest record {manifest_path}:{line_number} must be a test object"
                )
            seed = int(record["seed"])
            if not contract["seed_start"] <= seed <= contract["seed_end"]:
                raise ValueError(f"test seed {seed} is outside approved chunk range")
            if seed in metadata:
                raise ValueError(f"duplicate test seed {seed}")
            outputs = record.get("outputs")
            if not isinstance(outputs, dict):
                raise ValueError(f"test seed {seed} has no outputs object")
            try:
                rgb = outputs["rgb"]
                mask = outputs["mask"]
                rgb_hash = _expected_hash(rgb["sha256"], f"seed {seed} RGB hash")
                mask_hash = _expected_hash(mask["sha256"], f"seed {seed} mask hash")
                level_ml = float(record["level_ml"])
                geometry_row = float(record["surface"]["geometry_row"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"test seed {seed} has invalid evaluation metadata") from error
            if not math.isfinite(level_ml) or not math.isfinite(geometry_row):
                raise ValueError(f"test seed {seed} has a non-finite level or row")
            samples.append(
                training.FillLevelSample(
                    role="test",
                    seed=seed,
                    chunk=chunk_name,
                    frame_index=int(record["frame_index"]),
                    rgb_path=_safe_output_path(output_dir, rgb["path"], "RGB"),
                    mask_path=_safe_output_path(output_dir, mask["path"], "mask"),
                    rgb_sha256=rgb_hash,
                    mask_sha256=mask_hash,
                )
            )
            metadata[seed] = TestMetadata(
                seed=seed,
                cohort=contract["cohort"],
                level_ml=level_ml,
                sampling_band=str(record.get("level_sampling_band", "")),
                foam=bool(record.get("foam", {}).get("enabled")),
                sticker=bool(record.get("sticker_pose", {}).get("present")),
                geometry_row=geometry_row,
            )
            chunk_count += 1
        if chunk_count != contract["frame_count"]:
            raise ValueError(
                f"test chunk {chunk_name} needs {contract['frame_count']} records; "
                f"got {chunk_count}"
            )
    samples.sort(key=lambda sample: (sample.seed, sample.frame_index))
    if len(samples) != EXPECTED_TEST_COUNT or tuple(
        item.seed for item in samples
    ) != EXPECTED_TEST_SEEDS:
        raise ValueError("test selection must contain the exact frozen 560-seed union")
    observed_cohorts = {
        name: sum(item.cohort == name for item in metadata.values())
        for name in EXPECTED_COHORT_COUNTS
    }
    if observed_cohorts != EXPECTED_COHORT_COUNTS:
        raise ValueError(
            f"test manifest cohorts must be {EXPECTED_COHORT_COUNTS}; "
            f"got {observed_cohorts}"
        )
    observed_cases = {
        name: sum(name in item.cases for item in metadata.values()) for name in CASE_NAMES
    }
    if observed_cases != batch_contract["case_counts"]:
        raise ValueError(
            f"test manifest cases must be {batch_contract['case_counts']}; "
            f"got {observed_cases}"
        )
    return tuple(samples), metadata


def level_bin_10ml(level_ml):
    level = float(level_ml)
    if not math.isfinite(level) or not 0.0 <= level <= 90.0:
        raise ValueError(f"test level outside supported 0-90 mL bins: {level}")
    lower = min(int(level // 10) * 10, 80)
    return f"{lower:02d}-{lower + 10:02d}"


def frame_result(metadata, truth_mask, predicted_mask, fitted_ruler, rule_config):
    """Score one test frame with the exact frozen validation rule and hand ruler."""
    result = surface_eval.frame_result(
        metadata,
        truth_mask,
        predicted_mask,
        rule_config=rule_config,
    )
    label_row = float(result["true_top_liquid_row"])
    label_ml = ruler_module.row_to_ml(fitted_ruler, label_row)
    result.update(
        {
            "cohort": metadata.cohort,
            "manifest_level_ml": float(metadata.level_ml),
            "manifest_hidden_case": "hidden" in metadata.cases,
            "label_top_liquid_row": label_row,
            "label_row_inferred_ml": label_ml,
            "label_row_signed_error_ml": label_ml - float(metadata.level_ml),
            "label_row_absolute_error_ml": abs(label_ml - float(metadata.level_ml)),
            "predicted_visible_ml": None,
            "visible_predicted_minus_manifest_ml": None,
            "predicted_range_ml": None,
            "predicted_range_width_rows": None,
            "predicted_range_width_ml": None,
            "manifest_level_in_predicted_range": None,
            "false_hidden_on_nonhidden": False,
        }
    )
    if result["prediction_status"] == "visible":
        predicted_ml = ruler_module.row_to_ml(
            fitted_ruler, result["predicted_liquid_top_row"]
        )
        result.update(
            {
                "predicted_visible_ml": predicted_ml,
                "visible_predicted_minus_manifest_ml": (
                    predicted_ml - float(metadata.level_ml)
                ),
            }
        )
    elif result["prediction_status"] == "hidden":
        top = float(result["predicted_range_top_row"])
        bottom = float(result["predicted_range_bottom_row"])
        converted = ruler_module.row_range_to_ml(fitted_ruler, [top, bottom])
        result.update(
            {
                "predicted_range_ml": converted,
                "predicted_range_width_rows": bottom - top,
                "predicted_range_width_ml": converted[1] - converted[0],
                "manifest_level_in_predicted_range": (
                    converted[0] <= float(metadata.level_ml) <= converted[1]
                ),
                "false_hidden_on_nonhidden": "hidden" not in metadata.cases,
            }
        )
    return result


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _error_metrics(values, prefix):
    if not values:
        return {
            f"{prefix}_mae": None,
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
        }
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mae": float(np.mean(values)),
        f"{prefix}_median": _percentile(values, 50),
        f"{prefix}_p95": _percentile(values, 95),
        f"{prefix}_max": float(np.max(values)),
    }


def _aggregate(items):
    visible = [item for item in items if item["prediction_status"] == "visible"]
    predicted_hidden = [item for item in items if item["prediction_status"] == "hidden"]
    abstained = [item for item in items if item["prediction_status"] == "abstain"]
    hidden_cases = [item for item in items if item["manifest_hidden_case"]]
    nonhidden = [item for item in items if not item["manifest_hidden_case"]]
    detected_hidden = [item for item in hidden_cases if item["prediction_status"] == "hidden"]
    false_hidden = [item for item in nonhidden if item["false_hidden_on_nonhidden"]]
    successful_hidden = [
        item
        for item in hidden_cases
        if item["prediction_status"] == "hidden"
        and item["manifest_level_in_predicted_range"]
    ]
    geometry_successful_hidden = [
        item
        for item in hidden_cases
        if item["prediction_status"] == "hidden"
        and item["geometry_in_predicted_range"]
    ]
    all_called_inside = [
        item
        for item in predicted_hidden
        if item["manifest_level_in_predicted_range"]
    ]
    truth_crop_fallbacks = [
        item
        for item in items
        if item.get("truth_row_source")
        == "label_same_columns_full_crop_fallback"
    ]
    result = {
        "frames": len(items),
        "visible_predictions": len(visible),
        "visible_prediction_rate": len(visible) / len(items) if items else None,
        "predicted_hidden": len(predicted_hidden),
        "predicted_hidden_rate": len(predicted_hidden) / len(items) if items else None,
        "abstained": len(abstained),
        "abstention_rate": len(abstained) / len(items) if items else None,
        "manifest_hidden_frames": len(hidden_cases),
        "detected_hidden_frames": len(detected_hidden),
        "hidden_case_detection_rate": (
            len(detected_hidden) / len(hidden_cases) if hidden_cases else None
        ),
        "hidden_successful_frames": len(successful_hidden),
        "hidden_truth_in_range_rate": (
            len(successful_hidden) / len(hidden_cases) if hidden_cases else None
        ),
        "hidden_truth_in_range_given_called_rate": (
            len(successful_hidden) / len(detected_hidden) if detected_hidden else None
        ),
        "hidden_geometry_successful_frames": len(geometry_successful_hidden),
        "hidden_geometry_truth_in_range_rate": (
            len(geometry_successful_hidden) / len(hidden_cases)
            if hidden_cases
            else None
        ),
        "nonhidden_frames": len(nonhidden),
        "false_hidden_on_nonhidden": len(false_hidden),
        "false_hidden_on_nonhidden_rate": (
            len(false_hidden) / len(nonhidden) if nonhidden else None
        ),
        "all_called_hidden_with_manifest_level_inside_range": len(all_called_inside),
        "all_called_hidden_manifest_level_inside_range_rate": (
            len(all_called_inside) / len(predicted_hidden)
            if predicted_hidden
            else None
        ),
        "truth_label_crop_fallbacks": len(truth_crop_fallbacks),
        "truth_label_crop_fallback_rate": (
            len(truth_crop_fallbacks) / len(items) if items else None
        ),
        "edge_flagged": sum(bool(item.get("edge_flag")) for item in items),
    }
    result.update(
        _error_metrics(
            [item["visible_absolute_error_ml"] for item in visible],
            "visible_absolute_error_ml",
        )
    )
    result.update(
        _error_metrics(
            [item["visible_absolute_error_rows"] for item in visible],
            "visible_absolute_error_rows",
        )
    )
    result.update(
        _error_metrics(
            [abs(item["visible_predicted_minus_manifest_ml"]) for item in visible],
            "visible_absolute_error_vs_manifest_ml",
        )
    )
    result.update(
        _error_metrics(
            [item["label_row_absolute_error_ml"] for item in items],
            "label_row_absolute_error_ml",
        )
    )
    result.update(
        _error_metrics(
            [item["predicted_range_width_rows"] for item in predicted_hidden],
            "predicted_range_width_rows",
        )
    )
    result.update(
        _error_metrics(
            [item["predicted_range_width_ml"] for item in predicted_hidden],
            "predicted_range_width_ml",
        )
    )
    return result


def summarize_results(results):
    """Report requested cases and bins overall and within each frozen cohort."""
    results = tuple(results)

    def grouped(subset):
        subset = tuple(subset)
        by_case = {"all": _aggregate(list(subset))}
        for name in CASE_NAMES:
            by_case[name] = _aggregate(
                [item for item in subset if name in item["cases"]]
            )
        by_bin = {}
        for lower in range(0, 90, 10):
            name = f"{lower:02d}-{lower + 10:02d}"
            by_bin[name] = _aggregate(
                [item for item in subset if item["level_bin_10ml"] == name]
            )
        return {"by_case": by_case, "by_level_bin_10ml": by_bin}

    summary = grouped(results)
    summary["by_cohort"] = {
        cohort: grouped([item for item in results if item["cohort"] == cohort])
        for cohort in EXPECTED_COHORT_COUNTS
    }
    return summary


def segmentation_summary(confusion):
    """Return held-out four-class IoU from an accumulated confusion matrix."""
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (4, 4):
        raise ValueError("test confusion matrix must be 4x4")
    per_class, macro = training.iou_from_confusion(confusion)
    names = ("empty", "liquid", "foam", "blocked")
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_iou": {
            name: float(value) if math.isfinite(value) else None
            for name, value in zip(names, per_class)
        },
        "macro_iou": float(macro) if math.isfinite(macro) else None,
    }

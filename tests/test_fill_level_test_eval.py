"""CPU-only contracts for Step 9 held-out twin-test evaluation."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import fill_level_ruler as RULER  # noqa: E402
from fluidmix import fill_level_surface_eval as SURFACE  # noqa: E402
from fluidmix import fill_level_test_eval as EVAL  # noqa: E402
from fluidmix import fill_level_training as TRAINING  # noqa: E402


CROP_HEIGHT = 416
CROP_WIDTH = 112
CENTER_FIRST = 42
CENTER_LAST = 60


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _sha256(path)


def _batch_summary(path):
    chunks = []
    manifest_hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in EVAL.EXPECTED_TEST_CHUNKS
    }
    for name, (
        _cohort,
        component,
        source_chunk,
        start,
        end,
        count,
    ) in EVAL.EXPECTED_TEST_CHUNKS.items():
        chunks.append({
            "chunk": name,
            "role": "test",
            "attempt": "attempt-01",
            "manifest_sha256": manifest_hashes[name],
            "seed_start": start,
            "seed_end": end,
            "frame_count": count,
            "integrity_passed": True,
            "component": component,
            "source_chunk": source_chunk,
        })
    value = {
        "schema_version": 1,
        "integrity": {"passed": True},
        "batch_contract": {
            "expected_chunks": 30,
            "expected_role_counts": {"test": 560, "train": 3480, "val": 360},
            "new_chunk_index_remap": "exact-v4 00-09 -> combined 20-29",
            "test_chunks_linked": 0,
            "training_root_roles": ["train", "val"],
        },
        "components": {
            name: {"summary_sha256": summary_hash}
            for name, summary_hash in EVAL.EXPECTED_COMPONENT_SUMMARIES.items()
        },
        "case_coverage": {"by_role": {"test": {
            "clean": 114,
            "foam": 289,
            "sticker": 329,
            "edge_zone": 283,
            "hidden": 142,
        }}},
        "chunks": chunks,
    }
    return value, _write_json(path, value)


def _authority_fixture(tmp_path):
    checkpoint = tmp_path / "best-epoch-028.pt"
    checkpoint.write_bytes(b"approved checkpoint")
    weights_hash = _sha256(checkpoint)
    training_hash = TRAINING.file_sha256(TRAINING.__file__)
    training_summary = {
        "status": "complete",
        "profile": "full",
        "checkpoint_candidate": True,
        "selection_split": "validation",
        "best_epoch": 28,
        "dataset": {"test_manifests_opened": 0, "test_images_loaded": 0},
        "inputs": {"training_module_sha256": training_hash},
        "artifacts": {
            "best_weights": checkpoint.name,
            "best_weights_sha256": weights_hash,
        },
        "config": {
            "crop_xyxy": list(TRAINING.CROP_XYXY),
            "class_ids": {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3},
            "base_channels": 16,
            "seed": 20261101,
        },
    }
    training_path = tmp_path / "training.json"
    training_summary_hash = _write_json(training_path, training_summary)
    batch_path = tmp_path / "batch.json"
    _batch, batch_hash = _batch_summary(batch_path)
    rule_config = SURFACE.frozen_rule_config(
        measurement_rows_full=(83, 427),
        min_liquid_pixels=1,
        hidden_rule_evidence={
            "holder_band_rows_full_inclusive": [217, 275],
            "surface_oval_height_rows": 17,
            "visibility_columns_full_half_open": [279, 340],
            "minimum_liquid_pixels_per_visibility_row": 3,
        },
    )
    rule = {
        "status": "complete",
        "purpose": "Step 8 validation-only surface-row rule freeze",
        "dataset": {
            "role": "val",
            "frame_count": 360,
            "case_counts": {"clean": 78, "foam": 177, "sticker": 207, "edge": 182, "hidden": 90},
            "preflight": {
                "samples_checked": 360,
                "files_checked": 720,
                "ordered_hash_ledger_sha256": EVAL.EXPECTED_VALIDATION_LEDGER,
            },
            "train_manifests_opened": 0,
            "train_images_loaded": 0,
            "test_manifests_opened": 0,
            "test_images_loaded": 0,
            "real_frames_loaded": 0,
        },
        "inputs": {
            "checkpoint_sha256": weights_hash,
            "training_summary_sha256": training_summary_hash,
            "evaluator_module_sha256": TRAINING.file_sha256(SURFACE.__file__),
            "batch_summary_sha256": batch_hash,
        },
        "model": {
            "checkpoint_sha256": weights_hash,
            "training_module_sha256": training_hash,
        },
        "rule_config": rule_config,
    }
    rule_path = tmp_path / "rule.json"
    rule_hash = _write_json(rule_path, rule)
    fitted = RULER.fit_affine_ruler({1: 329.0, 13: 177.0})
    frozen_ruler = {
        "status": "complete",
        "freeze_state": "frozen_before_step9_and_step10",
        "source": "hand_measured_step3_targets_after_failed_system_calibration",
        "checkpoint_review_sha256": EVAL.EXPECTED_CHECKPOINT_REVIEW_SHA256,
        "system_calibration": {
            "status": "failed",
            "ruler_created": False,
            "failed_medians_not_used": True,
            "failure_summary_sha256": EVAL.EXPECTED_CALIBRATION_FAILURE_SHA256,
        },
        "inputs": {
            "frozen_validation_rule_sha256": rule_hash,
            "weights_sha256": weights_hash,
            "training_module_sha256": training_hash,
            "training_summary_sha256": training_summary_hash,
            "surface_rule_module_sha256": TRAINING.file_sha256(SURFACE.__file__),
            "ruler_module_sha256": TRAINING.file_sha256(RULER.__file__),
            "train_manifests_opened": 0,
            "train_images_loaded": 0,
            "validation_manifests_opened": 0,
            "validation_images_loaded": 0,
            "test_manifests_opened": 0,
            "test_images_loaded": 0,
            "real_check_frames_loaded": 0,
        },
        "model": {"weights_sha256": weights_hash},
        "ruler": {
            **fitted,
            "rows_per_ml": 3.04,
            "calibration_points": [
                {**point, "source": "hand_measured_step3_target"}
                for point in fitted["calibration_points"]
            ],
        },
    }
    ruler_path = tmp_path / "ruler.json"
    ruler_hash = _write_json(ruler_path, frozen_ruler)
    kwargs = {
        "batch_summary_path": batch_path,
        "batch_summary_sha256": batch_hash,
        "training_summary_path": training_path,
        "training_summary_sha256": training_summary_hash,
        "checkpoint_path": checkpoint,
        "weights_sha256": weights_hash,
        "training_module_sha256": training_hash,
        "frozen_rule_path": rule_path,
        "frozen_rule_sha256": rule_hash,
        "frozen_ruler_path": ruler_path,
        "frozen_ruler_sha256": ruler_hash,
    }
    return kwargs, frozen_ruler


def test_authority_chain_binds_model_rule_ruler_and_batch(tmp_path):
    kwargs, _ruler = _authority_fixture(tmp_path)
    result = EVAL.validate_authority(**kwargs)
    assert result["model"]["weights_sha256"] == kwargs["weights_sha256"]
    assert result["rule"]["artifact_sha256"] == kwargs["frozen_rule_sha256"]
    assert result["ruler"]["artifact_sha256"] == kwargs["frozen_ruler_sha256"]
    assert set(result["batch"]["approved_chunks"]) == set(EVAL.EXPECTED_TEST_CHUNKS)


def test_authority_rejects_weight_or_training_module_tampering(tmp_path):
    kwargs, _ruler = _authority_fixture(tmp_path)
    kwargs["checkpoint_path"].write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        EVAL.validate_authority(**kwargs)
    kwargs["checkpoint_path"].write_bytes(b"approved checkpoint")
    kwargs["training_module_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training module SHA-256"):
        EVAL.validate_authority(**kwargs)


def test_authority_rejects_ruler_to_rule_binding_tamper(tmp_path):
    kwargs, ruler = _authority_fixture(tmp_path)
    ruler["inputs"]["frozen_validation_rule_sha256"] = "0" * 64
    kwargs["frozen_ruler_sha256"] = _write_json(kwargs["frozen_ruler_path"], ruler)
    with pytest.raises(ValueError, match="ruler provenance"):
        EVAL.validate_authority(**kwargs)


@pytest.mark.parametrize(
    "counter",
    (
        "train_manifests_opened",
        "train_images_loaded",
        "validation_manifests_opened",
        "validation_images_loaded",
        "test_manifests_opened",
        "test_images_loaded",
        "real_check_frames_loaded",
    ),
)
def test_authority_rejects_any_prefreeze_data_access(tmp_path, counter):
    kwargs, ruler = _authority_fixture(tmp_path)
    ruler["inputs"][counter] = 1
    kwargs["frozen_ruler_sha256"] = _write_json(kwargs["frozen_ruler_path"], ruler)
    with pytest.raises(ValueError, match="ruler provenance"):
        EVAL.validate_authority(**kwargs)


def _test_case_flags(index):
    foam = index < 289
    sticker = 117 <= index < 446
    band = (
        "twin_hidden_48_1_66_5"
        if index < 142
        else "twin_lower_edge_40_1_48_1"
        if index < 425
        else "full_0_85"
    )
    return foam, sticker, band


def _write_test_dataset(root, *, unsafe_seed=None):
    contracts = {}
    ordinal = 0
    for name, (
        cohort,
        component,
        source_chunk,
        start,
        end,
        count,
    ) in EVAL.EXPECTED_TEST_CHUNKS.items():
        output = root / name / "attempt-01" / "output"
        (output / "rgb").mkdir(parents=True)
        (output / "mask").mkdir()
        (output.parent / "state.txt").write_text("COMPLETE\n", encoding="utf-8")
        records = []
        for seed in range(start, end + 1):
            index = ordinal
            ordinal += 1
            foam, sticker, band = _test_case_flags(index)
            rgb_path = output / "rgb" / f"{seed}.png"
            mask_path = output / "mask" / f"{seed}.png"
            rgb_path.write_bytes(f"rgb-{seed}".encode())
            mask_path.write_bytes(f"mask-{seed}".encode())
            records.append({
                "role": "test",
                "seed": seed,
                "frame_index": seed - start,
                "level_ml": float(index % 85) + 0.5,
                "level_sampling_band": band,
                "foam": {"enabled": foam},
                "sticker_pose": {"present": sticker},
                "surface": {"geometry_row": 200.0},
                "outputs": {
                    "rgb": {
                        "path": "../escape.png" if seed == unsafe_seed else f"rgb/{seed}.png",
                        "sha256": _sha256(rgb_path),
                    },
                    "mask": {"path": f"mask/{seed}.png", "sha256": _sha256(mask_path)},
                },
            })
        manifest = output / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        contracts[name] = {
            "role": "test",
            "attempt": "attempt-01",
            "manifest_sha256": _sha256(manifest),
            "seed_start": start,
            "seed_end": end,
            "frame_count": count,
            "cohort": cohort,
            "component": component,
            "source_chunk": source_chunk,
        }
    # Poison role/real directories would fail if any children were opened as evidence.
    for name in (
        "00-val-1100000-1100199",
        "02-train-0100000-0100199",
        "real-frames-must-not-open",
    ):
        poison = root / name / "attempt-poison"
        poison.mkdir(parents=True)
        (poison / "state.txt").write_text("not a valid state", encoding="utf-8")
        (poison / "manifest.jsonl").write_text("{not-json", encoding="utf-8")
    return {"approved_chunks": contracts, "case_counts": dict(EVAL.EXPECTED_CASE_COUNTS)}


def test_discovery_reads_only_exact_test_chunks_and_verifies_all_files(tmp_path):
    root = tmp_path / "collected"
    contract = _write_test_dataset(root)
    samples, metadata = EVAL.discover_test_frames(root, contract)
    assert len(samples) == 560
    assert samples[0].seed == 2100000 and samples[-1].seed == 2101059
    assert {name: sum(item.cohort == name for item in metadata.values()) for name in EVAL.EXPECTED_COHORT_COUNTS} == EVAL.EXPECTED_COHORT_COUNTS
    assert {
        name: sum(name in item.cases for item in metadata.values())
        for name in EVAL.CASE_NAMES
    } == EVAL.EXPECTED_CASE_COUNTS
    verified = TRAINING.verify_sample_files(samples)
    assert verified["samples_checked"] == 560
    assert verified["files_checked"] == 1120
    samples[0].rgb_path.write_bytes(b"tampered after manifest discovery")
    with pytest.raises(ValueError, match="RGB hash"):
        TRAINING.verify_sample_files(samples)


def test_discovery_rejects_manifest_tamper_and_unsafe_output_paths(tmp_path):
    root = tmp_path / "tampered"
    contract = _write_test_dataset(root)
    manifest = root / "17-test-2100000-2100199" / "attempt-01" / "output" / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        EVAL.discover_test_frames(root, contract)

    unsafe_root = tmp_path / "unsafe"
    unsafe_contract = _write_test_dataset(unsafe_root, unsafe_seed=2100000)
    with pytest.raises(ValueError, match="unsafe RGB path"):
        EVAL.discover_test_frames(unsafe_root, unsafe_contract)


def _mask():
    return np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)


def _metadata(
    seed=2100000,
    level_ml=85.0,
    band="twin_hidden_48_1_66_5",
    cohort="old_step6_test",
):
    return EVAL.TestMetadata(
        seed=seed,
        cohort=cohort,
        level_ml=level_ml,
        sampling_band=band,
        foam=False,
        sticker=False,
        geometry_row=200.0,
    )


def _rule_config():
    return SURFACE.frozen_rule_config(
        measurement_rows_full=(83, 427),
        min_liquid_pixels=1,
        hidden_rule_evidence={
            "holder_band_rows_full_inclusive": [217, 275],
            "surface_oval_height_rows": 17,
            "visibility_columns_full_half_open": [279, 340],
            "minimum_liquid_pixels_per_visibility_row": 3,
        },
    )


def test_hidden_result_preserves_converted_range_without_midpoint():
    fitted = RULER.fit_affine_ruler({1: 329.0, 13: 177.0})
    config = _rule_config()
    truth = _mask()
    truth[229:, CENTER_FIRST:CENTER_LAST] = 1
    predicted = _mask()
    predicted[229:, CENTER_FIRST:CENTER_LAST] = 1
    result = EVAL.frame_result(_metadata(), truth, predicted, fitted, config)
    assert result["prediction_status"] == "hidden"
    assert result["predicted_visible_ml"] is None
    assert result["predicted_range_ml"] == pytest.approx(
        RULER.row_range_to_ml(fitted, [200.0, 292.0])
    )
    assert "midpoint" not in " ".join(result)


def test_group_and_bin_metrics_report_hidden_false_hidden_and_visible_error():
    fitted = RULER.fit_affine_ruler({1: 329.0, 13: 177.0})
    config = _rule_config()
    truth = _mask()
    truth[281:, CENTER_FIRST:CENTER_LAST] = 1  # full row 329 -> 30 mL
    visible = truth.copy()
    hidden_truth = _mask()
    hidden_truth[233:, CENTER_FIRST:CENTER_LAST] = 1  # full row 281
    hidden = _mask()
    hidden[233:, CENTER_FIRST:CENTER_LAST] = 1
    abstain = _mask()
    results = [
        EVAL.frame_result(_metadata(seed=1, level_ml=30.0, band="full_0_85"), truth, visible, fitted, config),
        EVAL.frame_result(_metadata(seed=2, level_ml=46.0), hidden_truth, hidden, fitted, config),
        EVAL.frame_result(_metadata(seed=3, level_ml=46.0, band="full_0_85"), hidden_truth, hidden, fitted, config),
        EVAL.frame_result(_metadata(seed=4, level_ml=30.0, band="full_0_85"), truth, abstain, fitted, config),
    ]
    aggregate = EVAL.summarize_results(results)
    all_frames = aggregate["by_case"]["all"]
    assert all_frames["visible_predictions"] == 1
    assert all_frames["predicted_hidden"] == 2
    assert all_frames["abstained"] == 1
    assert all_frames["false_hidden_on_nonhidden"] == 1
    assert all_frames["false_hidden_on_nonhidden_rate"] == pytest.approx(1 / 3)
    assert all_frames["visible_absolute_error_ml_max"] == pytest.approx(0.0)
    assert all_frames["hidden_successful_frames"] == 1
    assert all_frames["hidden_truth_in_range_rate"] == pytest.approx(1.0)
    assert all_frames["predicted_range_width_rows_median"] == pytest.approx(92.0)
    assert aggregate["by_case"]["hidden"]["frames"] == 1
    assert aggregate["by_level_bin_10ml"]["30-40"]["frames"] == 2
    assert aggregate["by_level_bin_10ml"]["40-50"]["frames"] == 2
    assert aggregate["by_cohort"]["old_step6_test"]["by_case"]["all"]["frames"] == 4


def test_truth_crop_fallback_and_exact_holder_rule_are_preserved():
    fitted = RULER.fit_affine_ruler({1: 329.0, 13: 177.0})
    config = _rule_config()
    truth = _mask()
    truth[427 - 48 :, CENTER_FIRST:CENTER_LAST] = 1
    predicted = _mask()
    predicted[426 - 48 :, CENTER_FIRST:CENTER_LAST] = 1
    result = EVAL.frame_result(
        _metadata(seed=2101000, level_ml=4.0, cohort="new_exact_v4_test"),
        truth,
        predicted,
        fitted,
        config,
    )
    assert result["truth_row_source"] == "label_same_columns_full_crop_fallback"
    assert result["true_top_liquid_row"] == 427.0
    aggregate = EVAL.summarize_results([result])
    assert aggregate["by_case"]["all"]["truth_label_crop_fallbacks"] == 1


def test_segmentation_summary_reports_per_class_and_macro_iou():
    confusion = np.diag([10, 20, 30, 40])
    result = EVAL.segmentation_summary(confusion)
    assert result["macro_iou"] == pytest.approx(1.0)
    assert result["per_class_iou"] == {
        "empty": 1.0,
        "liquid": 1.0,
        "foam": 1.0,
        "blocked": 1.0,
    }

"""CPU-only contracts for Step 7 validation surface-row evaluation."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import fill_level_surface_eval as EVAL  # noqa: E402
from fluidmix import fill_level_training as TRAINING  # noqa: E402


CROP_HEIGHT = 464 - 48
CROP_WIDTH = 368 - 256
CENTER_FIRST = 298 - 256
CENTER_LAST = 316 - 256


def _mask():
    return np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)


def _metadata(**updates):
    values = {
        "seed": 1100000,
        "level_ml": 55.0,
        "sampling_band": "twin_hidden_48_1_66_5",
        "foam": False,
        "sticker": False,
        "geometry_row": 140.0,
    }
    values.update(updates)
    return EVAL.ValidationMetadata(**values)


def test_step6_hidden_rule_derivation_selects_only_approved_validation_cohort():
    samples = [
        SimpleNamespace(seed=seed)
        for seed in [*range(1100000, 1100300), *range(1101000, 1101060)]
    ]

    indices = EVAL.step6_hidden_rule_derivation_indices(samples)

    assert indices == tuple(range(300))


def test_step6_hidden_rule_derivation_is_bound_to_approved_source_and_ledger():
    evidence = {
        "selection_source": "synthetic_validation_labels_and_manifest_geometry",
        "frame_count": 300,
        "holder_band_rows_full_inclusive": [217, 275],
        "surface_oval_height_rows": 17,
        "visibility_columns_full_half_open": [279, 340],
        "minimum_liquid_pixels_per_visibility_row": 3,
    }
    preflight = {
        "samples_checked": 300,
        "files_checked": 600,
        "ordered_hash_ledger_sha256": (
            "8969e9727149dc2f987b2afb0524fc042f46d5396adbb199411ce1c2ca2066dd"
        ),
    }

    bound = EVAL.bind_step6_hidden_rule_derivation(evidence, preflight)

    assert bound["role"] == "val"
    assert bound["source_component"] == "step6"
    assert bound["source_batch_summary_sha256"] == (
        "e736d17ee7db384fa9b15cadc97422bb095b2a82ef1fb6b879040df0424d88c0"
    )
    assert bound["ordered_hash_ledger_sha256"] == preflight[
        "ordered_hash_ledger_sha256"
    ]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _training_summary(checkpoint, audit, *, training_module_sha256=None):
    return {
        "status": "complete",
        "profile": "full",
        "checkpoint_candidate": True,
        "selection_split": "validation",
        "best_epoch": 28,
        "dataset": {
            "used_counts": {"train": 3000, "val": 300},
            "test_manifests_opened": 0,
            "test_images_loaded": 0,
        },
        "inputs": {
            "batch_summary_sha256": audit["batch_summary_sha256"],
            "exception_review_sha256": audit["review_sha256"],
            "training_module_sha256": (
                training_module_sha256
                if training_module_sha256 is not None
                else TRAINING.file_sha256(TRAINING.__file__)
            ),
        },
        "artifacts": {
            "best_weights": checkpoint.name,
            "best_weights_sha256": _sha256(checkpoint),
        },
        "config": {
            "crop_xyxy": [256, 48, 368, 464],
            "base_channels": 16,
            "seed": 20261101,
        },
    }


def test_surface_rule_uses_first_qualifying_liquid_row_and_frozen_band():
    mask = _mask()
    mask[90, CENTER_FIRST] = 1
    mask[100, CENTER_FIRST:CENTER_LAST] = 1
    default = EVAL.surface_prediction(mask)
    robust = EVAL.surface_prediction(mask, min_liquid_pixels=3)
    assert default == {
        "status": "visible",
        "liquid_top_row": 138.0,
        "range_top_row": 138.0,
        "range_bottom_row": 138.0,
    }
    assert robust["liquid_top_row"] == 148.0
    with pytest.raises(ValueError, match="outside the mask crop"):
        EVAL.surface_prediction(mask, measurement_columns_full=(100, 120))


def test_surface_rule_searches_only_frozen_vial_interior_rows():
    mask = _mask()
    mask[0:20, CENTER_FIRST:CENTER_LAST] = 1
    mask[100:, CENTER_FIRST:CENTER_LAST] = 1
    result = EVAL.surface_prediction(
        mask,
        measurement_rows_full=(83, 427),
    )
    assert result == {
        "status": "visible",
        "liquid_top_row": 148.0,
        "range_top_row": 148.0,
        "range_bottom_row": 148.0,
    }
    with pytest.raises(ValueError, match="measurement rows"):
        EVAL.surface_prediction(mask, measurement_rows_full=(47, 427))


def test_vial_interior_rows_are_intersection_after_nozzle_and_before_base():
    first = np.full((CROP_HEIGHT, CROP_WIDTH), 255, dtype=np.uint8)
    second = np.full((CROP_HEIGHT, CROP_WIDTH), 255, dtype=np.uint8)
    first[10:100, CENTER_FIRST:CENTER_LAST] = 0
    first[10:15, CENTER_FIRST:CENTER_LAST] = 3
    first[95:100, CENTER_FIRST:CENTER_LAST] = 3
    second[12:98, CENTER_FIRST:CENTER_LAST] = 0
    second[12:20, CENTER_FIRST:CENTER_LAST] = 3
    second[90:98, CENTER_FIRST:CENTER_LAST] = 3

    evidence = EVAL.derive_vial_interior_rows((first, second))

    assert evidence["measurement_rows_full_half_open"] == [68, 138]
    assert evidence["selection_source"] == (
        "synthetic_validation_label_outline_intersection"
    )
    assert evidence["frame_count"] == 2
    assert evidence["per_frame_start_min"] == 63
    assert evidence["per_frame_start_max"] == 68
    assert evidence["per_frame_end_exclusive_min"] == 138
    assert evidence["per_frame_end_exclusive_max"] == 143


def test_frozen_rule_config_records_vial_row_derivation():
    evidence = {
        "measurement_rows_full_half_open": [83, 427],
        "selection_source": "synthetic_validation_label_outline_intersection",
        "frame_count": 300,
    }
    config = EVAL.frozen_rule_config(
        measurement_rows_full=(83, 427),
        row_bounds_evidence=evidence,
    )
    assert config["measurement_rows_full_half_open"] == [83, 427]
    assert config["measurement_row_derivation"] == evidence


def _oval_label_mask(top_full, widths):
    mask = np.full((CROP_HEIGHT, CROP_WIDTH), 255, dtype=np.uint8)
    mask[10:400, 20:90] = 0
    top = int(top_full) - 48
    centre = 55
    for offset, width in enumerate(widths):
        first = centre - int(width) // 2
        mask[top + offset, first : first + int(width)] = 1
    maximum = int(max(widths))
    first = centre - maximum // 2
    mask[top + len(widths) : 400, first : first + maximum] = 1
    return mask


def test_hidden_rule_derivation_uses_validation_holder_and_surface_evidence():
    first = _oval_label_mask(300, [5, 15, 25, 35, 45, 61])
    second = _oval_label_mask(70, [5, 11, 19, 29, 39, 49, 57, 61])
    metadata = (
        _metadata(
            seed=1100000,
            level_ml=30.0,
            geometry_row=300.0,
            holder_rows_inclusive=(150, 220),
        ),
        _metadata(
            seed=1100001,
            level_ml=80.0,
            geometry_row=70.0,
            holder_rows_inclusive=(152, 218),
        ),
    )

    evidence = EVAL.derive_hidden_surface_rule((first, second), metadata)

    assert evidence["holder_band_rows_full_inclusive"] == [151, 219]
    assert evidence["visibility_columns_full_half_open"] == [276, 346]
    assert evidence["minimum_liquid_pixels_per_visibility_row"] == 4
    assert evidence["surface_oval_height_rows"] == 8
    assert evidence["surface_oval_eligible_frame_count"] == 2
    assert evidence["surface_oval_eligible_seeds"] == [1100000, 1100001]


def test_holder_band_rule_returns_widened_hidden_range_and_edge_flag():
    options = {
        "holder_band_rows_full": (120, 180),
        "surface_oval_height_rows": 8,
        "visibility_columns_full": (276, 346),
        "min_visibility_liquid_pixels": 4,
    }
    hidden = _mask()
    hidden[181 - 48 :, 20:90] = 1
    hidden_result = EVAL.surface_prediction(hidden, **options)
    assert hidden_result == {
        "status": "hidden",
        "liquid_top_row": 181.0,
        "range_top_row": 112.0,
        "range_bottom_row": 188.0,
        "edge_flag": False,
    }

    edge = _mask()
    edge[114 - 48 :, 20:90] = 1
    edge_result = EVAL.surface_prediction(edge, **options)
    assert edge_result == {
        "status": "visible",
        "liquid_top_row": 114.0,
        "range_top_row": 114.0,
        "range_bottom_row": 114.0,
        "edge_flag": True,
    }

    edge_off_centre = _mask()
    edge_off_centre[114 - 48 :, 20:35] = 1
    assert EVAL.surface_prediction(edge_off_centre, **options) == edge_result


@pytest.mark.parametrize("liquid_top_row", [120, 128])
def test_holder_band_rule_gives_edge_precedence_at_band_top_and_margin(
    liquid_top_row,
):
    options = {
        "holder_band_rows_full": (120, 180),
        "surface_oval_height_rows": 8,
        "visibility_columns_full": (276, 346),
        "min_visibility_liquid_pixels": 4,
    }
    mask = _mask()
    mask[liquid_top_row - 48 :, 20:90] = 1

    assert EVAL.surface_prediction(mask, **options) == {
        "status": "visible",
        "liquid_top_row": float(liquid_top_row),
        "range_top_row": float(liquid_top_row),
        "range_bottom_row": float(liquid_top_row),
        "edge_flag": True,
    }


def test_frozen_holder_rule_records_validation_derivation():
    evidence = {
        "holder_band_rows_full_inclusive": [217, 275],
        "surface_oval_height_rows": 17,
        "visibility_columns_full_half_open": [279, 340],
        "minimum_liquid_pixels_per_visibility_row": 3,
    }
    config = EVAL.frozen_rule_config(hidden_rule_evidence=evidence)
    assert config["hidden_rule_version"] == "holder_band_v2"
    assert config["holder_band_rows_full_inclusive"] == [217, 275]
    assert config["surface_oval_height_rows"] == 17
    assert config["hidden_rule_derivation"] == evidence


def test_hidden_range_contains_only_blocked_span_above_visible_liquid():
    mask = _mask()
    mask[80:100, CENTER_FIRST] = 3
    mask[100:, CENTER_FIRST:CENTER_LAST] = 1
    result = EVAL.surface_prediction(mask)
    assert result == {
        "status": "hidden",
        "liquid_top_row": 148.0,
        "range_top_row": 128.0,
        "range_bottom_row": 147.0,
    }


def test_hidden_frame_reports_geometry_containment_without_point_error():
    truth = _mask()
    truth[100:, CENTER_FIRST:CENTER_LAST] = 1
    prediction = _mask()
    prediction[80:100, CENTER_FIRST] = 3
    prediction[100:, CENTER_FIRST:CENTER_LAST] = 1
    result = EVAL.frame_result(_metadata(geometry_row=140.0), truth, prediction)
    assert result["prediction_status"] == "hidden"
    assert result["geometry_in_predicted_range"] is True
    assert result["visible_absolute_error_rows"] is None
    assert result["hidden_interval_distance_to_label_rows"] == pytest.approx(1.0)
    assert result["hidden_interval_distance_to_label_ml"] == pytest.approx(1 / 3.04)


def test_truth_is_literal_first_label_liquid_even_when_prediction_min_is_tuned():
    truth = _mask()
    truth[90, CENTER_FIRST] = 1
    truth[100:, CENTER_FIRST:CENTER_LAST] = 1
    prediction = _mask()
    prediction[100:, CENTER_FIRST:CENTER_LAST] = 1
    config = EVAL.frozen_rule_config(min_liquid_pixels=3)
    result = EVAL.frame_result(_metadata(), truth, prediction, rule_config=config)
    assert result["true_top_liquid_row"] == 138.0
    assert result["predicted_liquid_top_row"] == 148.0
    assert result["visible_absolute_error_rows"] == 10.0
    assert result["visible_absolute_error_ml"] == pytest.approx(10 / 3.04)


def test_aggregates_report_cases_bins_visible_error_and_hidden_rates():
    truth = _mask()
    truth[100:, CENTER_FIRST:CENTER_LAST] = 1
    visible = _mask()
    visible[102:, CENTER_FIRST:CENTER_LAST] = 1
    hidden = _mask()
    hidden[80:100, CENTER_FIRST] = 3
    hidden[100:, CENTER_FIRST:CENTER_LAST] = 1
    results = [
        EVAL.frame_result(_metadata(level_ml=5.0), truth, visible),
        EVAL.frame_result(
            _metadata(
                seed=1100001,
                level_ml=55.0,
                foam=True,
                geometry_row=140.0,
            ),
            truth,
            hidden,
        ),
    ]
    summary = EVAL.summarize_results(results)
    all_frames = summary["by_case"]["all"]
    assert all_frames["frames"] == 2
    assert all_frames["visible_predictions"] == 1
    assert all_frames["mae_rows"] == 2.0
    assert all_frames["predicted_hidden_rate"] == 0.5
    assert all_frames["true_in_range_rate"] == 1.0
    assert all_frames["true_in_range_over_all_frames_rate"] == 0.5
    assert summary["by_case"]["foam"]["frames"] == 1
    assert summary["by_case"]["hidden"]["frames"] == 2
    assert summary["by_level_bin_10ml"]["00-10"]["frames"] == 1
    assert summary["by_level_bin_10ml"]["50-60"]["frames"] == 1
    assert summary["by_level_bin_10ml"]["80-90"]["frames"] == 0


def test_hidden_containment_rates_keep_hidden_and_all_frame_denominators():
    truth = _mask()
    truth[100:, CENTER_FIRST:CENTER_LAST] = 1
    hidden = _mask()
    hidden[80:100, CENTER_FIRST] = 3
    hidden[100:, CENTER_FIRST:CENTER_LAST] = 1
    visible = _mask()
    visible[100:, CENTER_FIRST:CENTER_LAST] = 1
    results = [
        EVAL.frame_result(_metadata(seed=1100000, geometry_row=140.0), truth, hidden),
        EVAL.frame_result(_metadata(seed=1100001, geometry_row=120.0), truth, hidden),
        EVAL.frame_result(_metadata(seed=1100002, geometry_row=140.0), truth, visible),
    ]
    aggregate = EVAL.summarize_results(results)["by_case"]["all"]
    assert aggregate["predicted_hidden"] == 2
    assert aggregate["geometry_in_predicted_range"] == 1
    assert aggregate["true_in_range_rate"] == pytest.approx(1 / 2)
    assert aggregate["true_in_range_over_all_frames_rate"] == pytest.approx(1 / 3)


def test_validation_discovery_does_not_open_train_or_test_manifests(tmp_path):
    root = tmp_path / "collected"
    chunk_name = "00-val-1100000-1100000"
    output = root / chunk_name / "attempt-01" / "output"
    output.mkdir(parents=True)
    (output.parent / "state.txt").write_text("COMPLETE\n", encoding="utf-8")
    record = {
        "role": "val",
        "seed": 1100000,
        "frame_index": 0,
        "level_ml": 55.0,
        "level_sampling_band": "twin_hidden_48_1_66_5",
        "foam": {"enabled": False},
        "sticker_pose": {"present": False},
        "surface": {"geometry_row": 240.0},
        "holder_projection": {"selected_rows_inclusive": [217, 275]},
        "outputs": {
            "rgb": {"path": "rgb/frame.png", "sha256": "a" * 64},
            "mask": {"path": "mask/frame.png", "sha256": "b" * 64},
        },
    }
    manifest = output / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    # These role directories intentionally contain no attempt/state/manifest.
    (root / "02-train-0100000-0100199").mkdir(parents=True)
    (root / "17-test-2100000-2100199").mkdir(parents=True)
    audit = {
        "eligible_counts": {"train": 3000, "val": 1},
        "excluded_seeds": [],
        "approved_chunks": {
            chunk_name: {
                "role": "val",
                "attempt": "attempt-01",
                "manifest_sha256": _sha256(manifest),
                "seed_start": 1100000,
                "seed_end": 1100000,
            }
        },
    }
    samples, metadata = EVAL.discover_validation_frames(root, audit)
    assert [sample.seed for sample in samples] == [1100000]
    assert metadata[1100000].geometry_row == 240.0


def test_training_artifact_binding_rejects_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "best-epoch-028.pt"
    checkpoint.write_bytes(b"approved checkpoint")
    audit = {
        "eligible_counts": {"train": 3000, "val": 300},
        "batch_summary_sha256": "a" * 64,
        "review_sha256": "b" * 64,
    }
    summary = _training_summary(checkpoint, audit)
    summary_path = tmp_path / "training-summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = EVAL.validate_training_artifacts(summary_path, checkpoint, audit)
    assert result["best_epoch"] == 28
    checkpoint.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        EVAL.validate_training_artifacts(summary_path, checkpoint, audit)


def test_training_artifact_binding_rejects_training_module_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "best-epoch-028.pt"
    checkpoint.write_bytes(b"approved checkpoint")
    audit = {
        "eligible_counts": {"train": 3000, "val": 300},
        "batch_summary_sha256": "a" * 64,
        "review_sha256": "b" * 64,
    }
    summary = _training_summary(
        checkpoint,
        audit,
        training_module_sha256="0" * 64,
    )
    summary_path = tmp_path / "training-summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="training module SHA-256"):
        EVAL.validate_training_artifacts(summary_path, checkpoint, audit)


def test_level_bins_include_85_ml_and_reject_out_of_contract_values():
    assert EVAL.level_bin_10ml(0.0) == "00-10"
    assert EVAL.level_bin_10ml(10.0) == "10-20"
    assert EVAL.level_bin_10ml(85.0) == "80-90"
    with pytest.raises(ValueError, match="outside supported"):
        EVAL.level_bin_10ml(90.1)

"""Focused tests for the path-free WP7 development bundle materializer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.wp7_vlm_development_materializer import (
    DevelopmentMaterializerError,
    aggregate_masked_lab,
    build_deterministic_decision,
    select_complete_cohort,
)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _tracker(case: str, cohort: str = "DEVCOHORT_complete") -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "wp7-camera-fill-step-receipt-v1",
        "artifact_type": "wp7_camera_fill_step_receipt",
        "cohort_id": cohort,
        "vial_id": case,
        "claim_boundary": "camera_twin_geometry_equivalent_not_physical_ground_truth",
        "stages": [{
            "fill_step": 1, "tank_id": "t200", "tank_number": 200,
            "stream_step_index": 1, "expected_dispense_ml": 10.0,
            "expected_cumulative_ml": 10.0,
            "twin_geometry_equivalent_volume_ml": 9.0,
            "measured_volume_ml": 9.0, "agreement": True,
            "tank_cluster_consistent": True,
            "stream_lab": [60.0, 10.0, 15.0],
            "stream_spread_lab": [1.0, 2.0, 3.0],
            "images": {
                "early": f"v4-{case}-F0001", "mid": f"v4-{case}-F0002",
                "end": f"v4-{case}-F0003",
            },
        }],
        "calibration_sha256": "a" * 64,
        "recipe_artifact_sha256": "b" * 64,
        "source_artifact_sha256": hashlib.sha256(case.encode()).hexdigest(),
    }
    body["receipt_sha256"] = _sha(body)
    return body


def _inventory() -> dict[str, object]:
    cases = ("CASE_a", "CASE_b", "CASE_c")
    artifact: dict[str, object] = {
        "schema_version": "wp7-fill-step-development-materialization-v1",
        "artifact_type": "development_only_v3_v4_fill_step_comparison",
        "claim_boundary": "synthetic_camera_development_only_not_evaluation_not_physical_accuracy",
        "device": "cpu", "excluded_partitions": ["evaluation", "test", "video5"],
        "execution": {"device": "cpu", "gpu_excluded_reason": "test"},
        "source_commit": "4ab6d9b198e6d079165b9e40c21361e3cd745065",
        "partition_counts": {}, "support_summary": {}, "train_only_freeze": {},
        "source_bindings": {"plans": {"v4": {"sha256": "c" * 64}}},
        "per_vial": [
            {"run_id": "RUN-V4-TRAIN-01", "batch_id": "BATCH_1",
             "vial_id": f"v4:{case}", "split_role": "train",
             "receipt_cohort_id": "DEVCOHORT_complete", "new_supported": True,
             "video_sha256": hashlib.sha256(case.encode()).hexdigest()}
            for case in cases
        ],
        "supported_fill_step_receipts": {
            f"v4:{case}": _tracker(case) for case in cases
        },
    }
    artifact["receipt_sha256"] = _sha(artifact)
    return artifact


def _plan() -> dict[str, object]:
    return {
        "execution_schedules": [
            {
                "case_binding": {
                    "run_id": "RUN-V4-TRAIN-01", "batch_id": "BATCH_1",
                    "case_id": case, "vial_ordinal": ordinal,
                    "split_role": "train", "current_recipe_id": "5",
                    "previous_recipe_id": "14" if ordinal == 1 else "5",
                },
                "model_input_record": {
                    "current_recipe_id": "5", "previous_recipe_id": "14" if ordinal == 1 else "5",
                    "dispense_order": ["t200"], "programmed_ml": {"t200": 10},
                    "previous_final_tank_id": "t100",
                },
            }
            for ordinal, case in enumerate(("CASE_a", "CASE_b", "CASE_c"), 1)
        ]
    }


def test_select_complete_cohort_binds_attempt04_receipts_and_target_order() -> None:
    selected = select_complete_cohort(
        _inventory(), _plan(), run_id="RUN-V4-TRAIN-01",
        inventory_file_sha256="d" * 64, plan_file_sha256="c" * 64,
    )
    assert [row["case_id"] for row in selected["vials"]] == ["CASE_a", "CASE_b", "CASE_c"]
    assert [row["vial_position"] for row in selected["vials"]] == [1, 2, 3]
    assert selected["target_previous_recipe_id"] == "14"
    assert len(selected["tracker_receipts"]) == 3
    assert selected["development_binding"]["source_materialization_receipt_sha256"] == "d" * 64


def test_select_complete_cohort_fails_closed_on_missing_or_unaccepted_vial() -> None:
    inventory = _inventory()
    inventory["per_vial"][2]["new_supported"] = False
    inventory["receipt_sha256"] = _sha({k: v for k, v in inventory.items() if k != "receipt_sha256"})
    with pytest.raises(DevelopmentMaterializerError, match="complete accepted three-vial"):
        select_complete_cohort(
            inventory, _plan(), run_id="RUN-V4-TRAIN-01",
            inventory_file_sha256="d" * 64, plan_file_sha256="c" * 64,
        )

    inventory = _inventory()
    inventory["per_vial"][0]["run_id"] = "RUN-V4-EVALUATION-01"
    inventory["receipt_sha256"] = _sha({k: v for k, v in inventory.items() if k != "receipt_sha256"})
    with pytest.raises(DevelopmentMaterializerError, match="complete accepted three-vial"):
        select_complete_cohort(
            inventory, _plan(), run_id="RUN-V4-TRAIN-01",
            inventory_file_sha256="d" * 64, plan_file_sha256="c" * 64,
        )


def test_masked_lab_uses_all_three_cnn_masks_and_reports_dispersion() -> None:
    frames = []
    masks = []
    for value in (40, 80, 120):
        frame = np.full((4, 4, 3), value, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        frames.append(frame)
        masks.append(mask)
    measured = aggregate_masked_lab(frames, masks)
    assert measured["frame_count"] == 3
    assert measured["liquid_pixel_count"] == 12
    assert all(value >= 0 for value in measured["mad_lab"])
    assert all(value >= 0 for value in measured["iqr_lab"])
    assert measured["median_lab"][0] > 0


def test_deterministic_call_compares_target_with_both_references() -> None:
    rows = [
        {"stage_id": "stage-1", "evidence_id": "target", "median_lab": [54.0, 0.0, 0.0]},
        {"stage_id": "stage-1", "evidence_id": "ref2", "median_lab": [50.0, 0.0, 0.0]},
        {"stage_id": "stage-1", "evidence_id": "ref3", "median_lab": [50.5, 0.0, 0.0]},
    ]
    result = build_deterministic_decision(
        cohort_id="DEVCOHORT_complete", target_vial_id="CASE_a",
        stage_measurements=[rows], source_artifact_sha256="d" * 64,
    )
    assert result["disposition"] == "discard"
    assert result["stages"][0]["delta_e76_vial2"] == pytest.approx(4.0)
    assert result["stages"][0]["delta_e76_vial3"] == pytest.approx(3.5)
    assert result["receipt_sha256"] == _sha({k: v for k, v in result.items() if k != "receipt_sha256"})


def test_development_deterministic_builder_reexports_production_owner() -> None:
    from fluidmix import wp7_vlm_decision as production
    from fluidmix import wp7_vlm_development_materializer as development

    assert development.canonical_sha256 is production.canonical_sha256
    assert development.build_deterministic_decision is production.build_deterministic_decision


def test_deterministic_call_abstains_in_frozen_threshold_deadband() -> None:
    rows = [
        {"stage_id": "stage-1", "evidence_id": "target", "median_lab": [54.0, 0.0, 0.0]},
        {"stage_id": "stage-1", "evidence_id": "ref2", "median_lab": [52.5, 0.0, 0.0]},
        {"stage_id": "stage-1", "evidence_id": "ref3", "median_lab": [50.0, 0.0, 0.0]},
    ]
    result = build_deterministic_decision(
        cohort_id="DEVCOHORT_complete", target_vial_id="CASE_a",
        stage_measurements=[rows], source_artifact_sha256="d" * 64,
    )
    assert result["disposition"] == "insufficient_evidence"
    assert result["stages"][0]["limit_codes"] == ["same_cohort_threshold_deadband"]

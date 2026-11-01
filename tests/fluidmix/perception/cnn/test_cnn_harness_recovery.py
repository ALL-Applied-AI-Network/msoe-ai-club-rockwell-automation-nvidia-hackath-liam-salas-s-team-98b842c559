"""Regression contracts for the bounded CNN scoring-harness recovery."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def _mask(first: int, last: int) -> np.ndarray:
    value = np.zeros((480, 640), dtype=bool)
    value[first : last + 1, 300:310] = True
    return value


def test_visible_height_error_uses_first_and_last_occupied_rows() -> None:
    from fluidmix.perception.cnn.cnn_harness_recovery import visible_height_error_px

    # The first rows match, so the rejected surface-only metric would report 0.
    # True visible heights are 9 and 11 pixels.
    assert visible_height_error_px(_mask(10, 18), _mask(10, 20)) == 2.0


def test_overlap_exclusions_remove_every_row_using_duplicate_content() -> None:
    from fluidmix.perception.cnn.cnn_harness_recovery import classify_overlap_exclusions

    rows = [
        {
            "case_id": "case-a:stage-1",
            "source_case_id": "case-a",
            "truth_mask_sha256": "mask-overlap",
            "reference_rgb_sha256": "clean-reference",
        },
        {
            "case_id": "case-b:stage-1",
            "source_case_id": "case-b",
            "truth_mask_sha256": "mask-overlap",
            "reference_rgb_sha256": "clean-reference-2",
        },
        {
            "case_id": "case-c:stage-1",
            "source_case_id": "case-c",
            "truth_mask_sha256": "clean-mask",
            "reference_rgb_sha256": "rgb-overlap",
        },
        {
            "case_id": "case-d:stage-1",
            "source_case_id": "case-d",
            "truth_mask_sha256": "clean-mask-2",
            "reference_rgb_sha256": "clean-reference-3",
        },
    ]

    result = classify_overlap_exclusions(
        rows,
        training_mask_hashes={"mask-overlap"},
        training_rgb_hashes={"rgb-overlap"},
    )

    assert result["overlap_content_identities"] == {
        "mask_sha256": ["mask-overlap"],
        "reference_rgb_sha256": ["rgb-overlap"],
    }
    assert result["excluded_positive_row_ids"] == [
        "case-a:stage-1",
        "case-b:stage-1",
        "case-c:stage-1",
    ]
    assert result["clean_positive_row_ids"] == ["case-d:stage-1"]
    assert result["classification"] == "likely_deterministic_simulator_duplicates"


def test_all_and_cleaned_scores_have_explicit_denominators_and_real_empty_gate() -> None:
    from fluidmix.perception.cnn.cnn_harness_recovery import score_all_and_cleaned

    cnn_rows = [
        {
            "case_id": "overlap:stage-1",
            "control_kind": "positive",
            "dice": 0.99,
            "visible_height_error_px": 1.0,
            "low_fill_first_dispense": True,
        },
        {
            "case_id": "clean:stage-1",
            "control_kind": "positive",
            "dice": 0.90,
            "visible_height_error_px": 2.0,
            "low_fill_first_dispense": True,
        },
    ]
    approach_rows = [
        {"case_id": "overlap:stage-1", "dice": 0.80},
        {"case_id": "clean:stage-1", "dice": 0.70},
    ]
    empty_control = {
        "control_kind": "empty_vial_nonidentical",
        "inference_executed": True,
        "abstained": True,
        "mask_pixel_count": 0,
    }

    result = score_all_and_cleaned(
        cnn_rows,
        approach_rows,
        excluded_positive_row_ids={"overlap:stage-1"},
        empty_control=empty_control,
    )

    assert result["denominators"] == {
        "all_positive_rows": 2,
        "excluded_positive_rows": 1,
        "clean_positive_rows": 1,
        "cnn_executed_empty_controls": 1,
        "identical_guard_controls": 0,
    }
    assert result["all"]["metrics"]["mean_dice"] == pytest.approx(0.945)
    assert result["cleaned"]["metrics"]["mean_dice"] == pytest.approx(0.90)
    assert result["cleaned"]["gates"]["nonidentical_empty_vial"] is True
    assert result["cleaned"]["all_gates_pass"] is True


def test_identical_guard_cannot_substitute_for_executed_empty_control() -> None:
    from fluidmix.perception.cnn.cnn_harness_recovery import score_all_and_cleaned

    positive = {
        "case_id": "clean:stage-1",
        "control_kind": "positive",
        "dice": 0.90,
        "visible_height_error_px": 2.0,
        "low_fill_first_dispense": True,
    }
    result = score_all_and_cleaned(
        [positive],
        [{"case_id": "clean:stage-1", "dice": 0.70}],
        excluded_positive_row_ids=set(),
        empty_control={
            "control_kind": "identical",
            "inference_executed": False,
            "abstained": True,
            "mask_pixel_count": 0,
        },
    )

    assert result["cleaned"]["gates"]["nonidentical_empty_vial"] is False
    assert result["cleaned"]["all_gates_pass"] is False


def test_recovery_audit_maps_opened_hashes_to_rows_without_model_inference() -> None:
    from fluidmix.perception.cnn.cnn_harness_recovery import build_recovery_audit

    inventory = {
        "samples": [
            {
                "current_rgb_sha256": "train-current",
                "reference_rgb_sha256": "shared-reference",
                "mask_sha256": "shared-mask",
            }
        ]
    }
    comparison = {
        "cnn_rows": [
            {"case_id": "case-a:stage-1", "control_kind": "positive", "dice": 0.99},
            {"case_id": "case-b:stage-1", "control_kind": "positive", "dice": 0.90},
        ],
        "approach_a_rows": [
            {"case_id": "case-a:stage-1", "dice": 0.80},
            {"case_id": "case-b:stage-1", "dice": 0.70},
        ],
        "opened_files": [
            {"case_id": "case-a", "role": "reference", "sha256": "shared-reference"},
            {"case_id": "case-a", "role": "empty", "sha256": "shared-reference"},
            {"case_id": "case-a", "role": "stage_rgb", "sha256": "rgb-a"},
            {"case_id": "case-a", "role": "simulator_mask", "sha256": "clean-mask"},
            {"case_id": "case-b", "role": "reference", "sha256": "clean-reference"},
            {"case_id": "case-b", "role": "empty", "sha256": "clean-reference"},
            {"case_id": "case-b", "role": "stage_rgb", "sha256": "rgb-b"},
            {"case_id": "case-b", "role": "simulator_mask", "sha256": "shared-mask"},
        ],
    }

    audit = build_recovery_audit(inventory, comparison)

    assert audit["denominators"] == {
        "all_positive_rows": 2,
        "excluded_positive_rows": 2,
        "clean_positive_rows": 0,
        "overlap_content_identities": 2,
        "overlap_mask_content_identities": 1,
        "overlap_reference_rgb_content_identities": 1,
    }
    assert audit["corrected_height_metric"]["status"] == "blocked_no_frozen_prediction_bounds"
    assert audit["empty_control"]["status"] == "blocked_no_authorized_nonidentical_empty_input"
    assert audit["model_inference_count"] == 0


def test_selects_one_same_run_nonidentical_empty_pair_from_frozen_ledger() -> None:
    from fluidmix.perception.cnn.cnn_harness_recovery import select_nonidentical_empty_control

    preregister = {
        "runs": [
            {"run_id": "run-1", "case_ids": ["case-1", "case-2", "case-3"]},
            {"run_id": "run-2", "case_ids": ["case-4", "case-5", "case-6"]},
        ]
    }
    comparison = {
        "opened_files": [
            {
                "case_id": "case-1",
                "role": "reference",
                "path": "/camera/run-1/case-1/prepour.png",
                "sha256": "a" * 64,
            },
            {
                "case_id": "case-2",
                "role": "reference",
                "path": "/camera/run-1/case-2/prepour.png",
                "sha256": "b" * 64,
            },
            {
                "case_id": "case-4",
                "role": "reference",
                "path": "/camera/run-2/case-4/prepour.png",
                "sha256": "c" * 64,
            },
        ]
    }

    selected = select_nonidentical_empty_control(preregister, comparison)

    assert selected == {
        "selection_rule": "first_run_second_case_prepour_vs_first_case_prepour_reference",
        "run_id": "run-1",
        "current_case_id": "case-2",
        "current_path": "/camera/run-1/case-2/prepour.png",
        "current_sha256": "b" * 64,
        "reference_case_id": "case-1",
        "reference_path": "/camera/run-1/case-1/prepour.png",
        "reference_sha256": "a" * 64,
        "model_inference_denominator": 1,
    }


def test_empty_pair_selection_rejects_identical_or_cross_run_substitution() -> None:
    import pytest

    from fluidmix.perception.cnn.cnn_harness_recovery import select_nonidentical_empty_control

    preregister = {"runs": [{"run_id": "run-1", "case_ids": ["case-1", "case-2"]}]}
    comparison = {
        "opened_files": [
            {
                "case_id": case_id,
                "role": "reference",
                "path": f"/camera/{case_id}.png",
                "sha256": "a" * 64,
            }
            for case_id in ("case-1", "case-2")
        ]
    }
    with pytest.raises(ValueError, match="non-identical"):
        select_nonidentical_empty_control(preregister, comparison)

    comparison["opened_files"][1]["sha256"] = "b" * 64
    comparison["opened_files"][1]["case_id"] = "case-outside-run"
    with pytest.raises(ValueError, match="missing"):
        select_nonidentical_empty_control(preregister, comparison)


def test_summarizes_replayed_mask_with_canonical_hash_and_bounds() -> None:
    import hashlib

    import numpy as np

    from fluidmix.perception.cnn.cnn_harness_recovery import summarize_replayed_mask

    mask = np.zeros((480, 640), dtype=bool)
    mask[120:143, 200:205] = True

    assert summarize_replayed_mask(mask) == {
        "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
        "mask_hash_encoding": "480x640-row-major-uint8-0-or-1",
        "first_occupied_row": 120,
        "last_occupied_row": 142,
        "visible_height_px": 23,
        "mask_pixel_count": 115,
    }


def test_replay_dice_gate_allows_at_most_one_thousandth_difference() -> None:
    import pytest

    from fluidmix.perception.cnn.cnn_harness_recovery import verify_replay_dice_tolerance

    frozen = [
        {"case_id": "case-1:stage-1", "control_kind": "positive", "dice": 0.9},
        {"case_id": "case-1:stage-2", "control_kind": "positive", "dice": 0.8},
    ]
    within_tolerance = [
        {"case_id": "case-1:stage-1", "dice": 0.899},
        {"case_id": "case-1:stage-2", "dice": 0.8005},
    ]
    assert verify_replay_dice_tolerance(frozen, within_tolerance) == {
        "status": "pass",
        "comparison": "absolute_dice_difference_lte_tolerance",
        "tolerance": 0.001,
        "max_absolute_difference": pytest.approx(0.001),
        "matched_rows": 2,
        "mismatch_count": 0,
    }

    within_tolerance[1]["dice"] = 0.8011
    with pytest.raises(ValueError, match="case-1:stage-2"):
        verify_replay_dice_tolerance(frozen, within_tolerance)

    with pytest.raises(ValueError, match="identities"):
        verify_replay_dice_tolerance(frozen, within_tolerance[:1])

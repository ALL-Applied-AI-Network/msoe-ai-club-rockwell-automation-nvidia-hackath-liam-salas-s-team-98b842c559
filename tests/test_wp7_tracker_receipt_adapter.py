"""Synthetic tests for the tracker-receipt endpoint-plan seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.wp7_tracker_receipt_adapter import (
    TrackerReceiptAdapterError,
    adapt_tracker_receipts_to_endpoint_plans,
)
import fluidmix.wp7_tracker_receipt_adapter as adapter


@pytest.fixture(autouse=True)
def _synthetic_producer_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter, "_producer_validator", lambda: lambda value: dict(value)
    )


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _sequence() -> dict[str, object]:
    return {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": "real_check",
        "split_version": "synthetic-source-only-v1",
        "runs": [{
            "run_id": "VIDEO5-RUN-SYNTHETIC",
            "vials": [{
                "case_id": "VIDEO5-VIAL-01",
                "programmed_tank_ml": {"t100": 10, "t200": 20, "t400": 0},
                "dispense_order": ["t100", "t200"],
            }],
        }],
    }


def _receipt() -> dict[str, object]:
    stages = []
    cumulative = 0.0
    for ordinal, (tank_id, amount, frames) in enumerate((
        ("t100", 10.0, (10, 11, 12)),
        ("t200", 20.0, (20, 21, 22)),
    ), start=1):
        cumulative += amount
        stages.append({
            "fill_step": ordinal,
            "tank_id": tank_id,
            "tank_number": int(tank_id[1:]),
            "stream_step_index": ordinal,
            "expected_dispense_ml": amount,
            "expected_cumulative_ml": cumulative,
            "twin_geometry_equivalent_volume_ml": cumulative,
            "measured_volume_ml": cumulative,
            "agreement": True,
            "tank_cluster_consistent": True,
            "stream_lab": [50.0, float(ordinal), 0.0],
            "stream_spread_lab": [0.2, 0.3, 0.4],
            "images": {
                role: f"VIDEO5-VIAL-01-F{frame:06d}"
                for role, frame in zip(("early", "mid", "end"), frames, strict=True)
            },
        })
    value = {
        "schema_version": "wp7-camera-fill-step-receipt-v1",
        "artifact_type": "wp7_camera_fill_step_receipt",
        "cohort_id": "VIDEO5-RUN-SYNTHETIC",
        "vial_id": "VIDEO5-VIAL-01",
        "claim_boundary": "camera_twin_geometry_equivalent_not_physical_ground_truth",
        "stages": stages,
        "calibration_sha256": "a" * 64,
        "recipe_artifact_sha256": "b" * 64,
        "source_artifact_sha256": "c" * 64,
    }
    value["receipt_sha256"] = _hash(value)
    return value


def _image_bindings(receipt: dict[str, object]) -> dict[str, dict[str, object]]:
    bindings = {}
    for stage in receipt["stages"]:
        for image_id in stage["images"].values():
            bindings[image_id] = {
                "case_id": "VIDEO5-VIAL-01",
                "frame_index": int(image_id.rsplit("F", 1)[1]),
                "rgb_sha256": "d" * 64,
                "mask_sha256": "e" * 64,
            }
    return bindings


def _adapt(receipt: dict[str, object], **changes):
    arguments = {
        "receipts": [receipt],
        "recipe_sequence": _sequence(),
        "receipt_vial_id_by_case": {"VIDEO5-VIAL-01": "VIDEO5-VIAL-01"},
        "image_bindings": _image_bindings(receipt),
        "expected_cohort_id": "VIDEO5-RUN-SYNTHETIC",
        "expected_receipt_sha256_by_case": {
            "VIDEO5-VIAL-01": receipt["receipt_sha256"]
        },
        "expected_source_artifact_sha256_by_case": {
            "VIDEO5-VIAL-01": "c" * 64
        },
        "expected_calibration_sha256": "a" * 64,
        "expected_recipe_artifact_sha256": "b" * 64,
    }
    arguments.update(changes)
    return adapt_tracker_receipts_to_endpoint_plans(**arguments)


def test_receipt_projects_recipe_bound_early_mid_end_endpoint_plan() -> None:
    receipt = _receipt()

    plans = _adapt(receipt)

    assert len(plans) == 1
    assert plans[0].case_id == "VIDEO5-VIAL-01"
    assert plans[0].receipt_sha256 == receipt["receipt_sha256"]
    assert [stage.tank_id for stage in plans[0].stages] == ["t100", "t200"]
    assert [stage.expected_cumulative_ml for stage in plans[0].stages] == [10.0, 30.0]
    assert (
        plans[0].stages[1].early_frame_index,
        plans[0].stages[1].mid_frame_index,
        plans[0].stages[1].end_frame_index,
    ) == (20, 21, 22)
    assert plans[0].stages[1].stream_lab == (50.0, 2.0, 0.0)
    assert plans[0].stages[1].twin_geometry_equivalent_volume_ml == 30.0


def test_receipt_hash_or_recipe_volume_drift_fails_closed() -> None:
    receipt = _receipt()
    with pytest.raises(TrackerReceiptAdapterError, match="receipt SHA-256"):
        _adapt(
            receipt,
            expected_receipt_sha256_by_case={"VIDEO5-VIAL-01": "f" * 64},
        )

    changed = _sequence()
    changed["runs"][0]["vials"][0]["programmed_tank_ml"]["t200"] = 21
    with pytest.raises(TrackerReceiptAdapterError, match="recipe stage"):
        _adapt(receipt, recipe_sequence=changed)


def test_receipt_images_must_be_distinct_ordered_and_case_bound() -> None:
    receipt = _receipt()
    bindings = _image_bindings(receipt)
    bindings["VIDEO5-VIAL-01-F000021"]["frame_index"] = 9
    with pytest.raises(TrackerReceiptAdapterError, match="early/mid/end frame order"):
        _adapt(receipt, image_bindings=bindings)

    receipt = _receipt()
    bindings = _image_bindings(receipt)
    bindings["VIDEO5-VIAL-01-F000022"]["case_id"] = "OTHER-VIAL"
    with pytest.raises(TrackerReceiptAdapterError, match="image case binding"):
        _adapt(receipt, image_bindings=bindings)


def test_adapter_delegates_schema_validation_and_rejects_disagreement(
    monkeypatch,
) -> None:
    receipt = _receipt()
    calls = []

    def validator(value):
        calls.append(value)
        return dict(value)

    monkeypatch.setattr(adapter, "_producer_validator", lambda: validator)
    plans = _adapt(receipt)
    assert calls == [receipt]
    assert plans[0].stages[0].agreement is True

    receipt["stages"][0]["agreement"] = False
    receipt["receipt_sha256"] = _hash({
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    })
    with pytest.raises(TrackerReceiptAdapterError, match="accepted agreement"):
        _adapt(receipt)


def test_missing_producer_validator_is_an_explicit_integration_blocker(
    monkeypatch,
) -> None:
    receipt = _receipt()
    def unavailable():
        raise TrackerReceiptAdapterError("tracker producer validator is not available")

    monkeypatch.setattr(adapter, "_producer_validator", unavailable)
    with pytest.raises(TrackerReceiptAdapterError, match="producer validator"):
        _adapt(receipt)

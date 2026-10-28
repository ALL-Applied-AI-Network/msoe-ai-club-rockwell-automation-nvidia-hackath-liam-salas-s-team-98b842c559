"""Decision boundary for three-vial varied-run stage measurements."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _measurement_artifact(
    *,
    include_third_stage: bool = False,
    split_version: str = "whole-run-varied-v3",
) -> dict[str, object]:
    stages = [
        (
            "dispense-1-t200",
            1,
            "t200",
            {
                "V1": [10.0, 0.0, 0.0],
                "V2": [20.0, 0.0, 0.0],
                "V3": [20.2, 0.0, 0.0],
            },
        ),
        (
            "dispense-2-t100",
            2,
            "t100",
            {
                "V1": [30.0, 0.0, 0.0],
                "V2": [30.1, 0.0, 0.0],
                "V3": [29.9, 0.0, 0.0],
            },
        ),
    ]
    if include_third_stage:
        stages.append(
            (
                "dispense-3-t400",
                3,
                "t400",
                {
                    "V1": [40.0, 1.0, 0.0],
                    "V2": [40.2, 1.0, 0.0],
                    "V3": [39.9, 1.0, 0.0],
                },
            )
        )

    measurements: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for stage_id, ordinal, tank_id, labs in stages:
        for vial_id, lab in labs.items():
            image_id = f"IMG-BATCH-1-{vial_id}-{stage_id}"
            measurements.append(
                {
                    "evidence_id": f"EVID-{image_id}",
                    "measurement_id": f"MEAS-{image_id}",
                    "image_id": image_id,
                    "batch_id": "BATCH-1",
                    "run_id": "RUN-TRAIN-1",
                    "recipe_id": "6",
                    "split_role": "train",
                    "split_version": split_version,
                    "vial_id": vial_id,
                    "stage_id": stage_id,
                    "stage_ordinal": ordinal,
                    "tank_id": tank_id,
                    "view_id": "vial_fill",
                    "source_manifest_sha256": "a" * 64,
                    "rgb_sha256": (vial_id[-1] * 64),
                    "mask_sha256": "f" * 64,
                    "liquid_pixel_count": 2000,
                    "mask_coverage_fraction": 0.1,
                    "mask_height_px": 40,
                    "median_lab": lab,
                    "mad_lab": [0.1, 0.1, 0.1],
                    "iqr_lab": [0.2, 0.2, 0.2],
                    "median_srgb_0_255": [100.0, 110.0, 120.0],
                    "mad_srgb_0_255": [1.0, 1.0, 1.0],
                    "lab_unit": "cie_lab_d65_2deg",
                }
            )
        vial_ids = ("V1", "V2", "V3")
        for left_index in range(2):
            for right_index in range(left_index + 1, 3):
                left = vial_ids[left_index]
                right = vial_ids[right_index]
                delta = math.dist(labs[left], labs[right])
                pairs.append(
                    {
                        "pair_id": f"PAIR-BATCH-1-{stage_id}-{left}-{right}",
                        "batch_id": "BATCH-1",
                        "run_id": "RUN-TRAIN-1",
                        "recipe_id": "6",
                        "split_role": "train",
                        "split_version": split_version,
                        "stage_id": stage_id,
                        "stage_ordinal": ordinal,
                        "tank_id": tank_id,
                        "left_vial_id": left,
                        "right_vial_id": right,
                        "left_evidence_id": f"EVID-IMG-BATCH-1-{left}-{stage_id}",
                        "right_evidence_id": f"EVID-IMG-BATCH-1-{right}-{stage_id}",
                        "source_manifest_sha256": "a" * 64,
                        "left_rgb_sha256": left[-1] * 64,
                        "right_rgb_sha256": right[-1] * 64,
                        "left_mask_sha256": "f" * 64,
                        "right_mask_sha256": "f" * 64,
                        "delta_e76": delta,
                        "delta_e76_6dp": f"{delta:.6f}",
                        "unit": "delta_e76",
                    }
                )
    stage_count = len(stages)
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_stage_measured_evidence",
        "status": "measured",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": split_version,
        "source_manifests": [
            {
                "batch_id": "BATCH-1",
                "run_id": "RUN-TRAIN-1",
                "recipe_id": "6",
                "split_role": "train",
                "sha256": "a" * 64,
                "active_stage_count": stage_count,
                "stage_measurement_count": stage_count * 3,
            }
        ],
        "stage_measurements": measurements,
        "same_run_stage_pairs": pairs,
        "measurement_contract": {
            "scope": "measurements_only_no_keep_discard_or_abstain_output"
        },
        "denominators": {
            "source_manifests": 1,
            "runs": 1,
            "vials": 3,
            "active_stage_events": stage_count,
            "stage_measurements": stage_count * 3,
            "same_run_stage_pairs": stage_count * 3,
            "two_stage_runs": int(stage_count == 2),
            "three_stage_runs": int(stage_count == 3),
        },
        "claim_boundary": "Synthetic masked-colour measurements only.",
    }


def _config():
    from fluidmix.f2_varied_triage import VariedTriadConfig

    return VariedTriadConfig(
        config_id="v3-triad-relative-v1",
        tight_peer_pair_max_delta_e76=1.0,
        target_outlier_margin_delta_e76=2.0,
    )


def test_relative_outlier_is_preserved_but_cannot_be_called_discard_without_clean_anchor():
    from fluidmix.f2_varied_triage import decide_varied_run_triads

    result = decide_varied_run_triads(_measurement_artifact(), _config())

    assert result["artifact_type"] == "f2_varied_triad_decisions"
    assert result["denominators"] == {
        "runs": 1,
        "vial_decisions": 3,
        "keep": 0,
        "discard": 0,
        "insufficient_evidence": 3,
        "relative_outlier_vials": 1,
    }
    by_vial = {row["vial_id"]: row for row in result["decision_records"]}
    outlier = by_vial["V1"]
    assert outlier["disposition"] == "insufficient_evidence"
    assert outlier["relative_assessment"] == "outlier_to_two_peer_mode"
    assert "within_run_reference_cleanliness_unverified" in outlier["limit_codes"]
    stage = outlier["stages"][0]
    assert stage["relative_status"] == "outlier_to_two_peer_mode"
    assert stage["target_delta_e76"] == pytest.approx(10.1)
    assert stage["peer_pair_delta_e76"] == pytest.approx(0.2)
    assert stage["threshold_delta_e76"] == pytest.approx(2.1)
    assert stage["target_evidence_id"].endswith("V1-dispense-1-t200")
    assert stage["peer_evidence_ids"] == [
        "EVID-IMG-BATCH-1-V2-dispense-1-t200",
        "EVID-IMG-BATCH-1-V3-dispense-1-t200",
    ]
    assert by_vial["V2"]["relative_assessment"] == "peer_mode_not_tight"
    assert by_vial["V3"]["relative_assessment"] == "peer_mode_not_tight"
    assert "does not authorize" in result["claim_boundary"].lower()
    assert "sim-to-real" in result["claim_boundary"].lower()


def test_three_active_stages_are_supported_without_a_vial_position_prior():
    from fluidmix.f2_varied_triage import decide_varied_run_triads

    result = decide_varied_run_triads(
        _measurement_artifact(include_third_stage=True), _config()
    )

    assert len(result["decision_records"]) == 3
    assert all(len(row["stages"]) == 3 for row in result["decision_records"])
    assert all(row["disposition"] == "insufficient_evidence" for row in result["decision_records"])
    assert result["reference_policy"] == {
        "scope": "other_two_vials_same_run_same_stage",
        "position_prior": False,
        "cleanliness_verified": False,
        "authorization": "relative_evidence_only_no_keep_or_discard",
    }


def test_v4_split_version_is_accepted_and_propagated_without_changing_decisions():
    from fluidmix.f2_varied_triage import decide_varied_run_triads

    result = decide_varied_run_triads(
        _measurement_artifact(
            include_third_stage=True,
            split_version="whole-run-varied-v4",
        ),
        _config(),
    )

    assert result["split_version"] == "whole-run-varied-v4"
    assert {
        row["split_version"] for row in result["decision_records"]
    } == {"whole-run-varied-v4"}
    assert result["denominators"]["insufficient_evidence"] == 3


def test_uniform_three_vial_mode_still_cannot_authorize_keep():
    from fluidmix.f2_varied_triage import decide_varied_run_triads

    measured = _measurement_artifact()
    first_stage = "dispense-1-t200"
    for row in measured["stage_measurements"]:  # type: ignore[union-attr]
        if row["stage_id"] == first_stage and row["vial_id"] == "V1":
            row["median_lab"] = [20.1, 0.0, 0.0]
    for pair in measured["same_run_stage_pairs"]:  # type: ignore[union-attr]
        if pair["stage_id"] != first_stage or "V1" not in {
            pair["left_vial_id"],
            pair["right_vial_id"],
        }:
            continue
        other = (
            pair["right_vial_id"]
            if pair["left_vial_id"] == "V1"
            else pair["left_vial_id"]
        )
        delta = 0.1 if other in {"V2", "V3"} else None
        assert delta is not None
        pair["delta_e76"] = delta
        pair["delta_e76_6dp"] = f"{delta:.6f}"

    result = decide_varied_run_triads(measured, _config())

    assert result["denominators"]["keep"] == 0
    assert result["denominators"]["insufficient_evidence"] == 3
    assert {row["relative_assessment"] for row in result["decision_records"]} == {
        "consistent_with_two_peer_mode"
    }
    assert all(
        "two_peer_mode_cannot_verify_cleanliness" in row["limit_codes"]
        for row in result["decision_records"]
    )


def test_oracle_fields_and_mismatched_pair_values_fail_closed():
    from fluidmix.f2_varied_triage import (
        VariedTriadInputError,
        decide_varied_run_triads,
    )

    oracle = _measurement_artifact()
    oracle["stage_measurements"][0]["expected_disposition"] = "discard"  # type: ignore[index]
    with pytest.raises(ValueError, match="forbidden_deploy_input_field"):
        decide_varied_run_triads(oracle, _config())

    mismatch = deepcopy(_measurement_artifact())
    mismatch["same_run_stage_pairs"][0]["delta_e76"] = 999.0  # type: ignore[index]
    with pytest.raises(VariedTriadInputError, match="pair_distance_mismatch"):
        decide_varied_run_triads(mismatch, _config())

"""Grounded explanation contract for Varied V3 triad decisions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _stage(batch: str, vial: str, ordinal: int, tank: str) -> dict[str, object]:
    stage = f"dispense-{ordinal}-{tank}"
    peers = [candidate for candidate in ("V1", "V2", "V3") if candidate != vial]
    return {
        "target_evidence_id": f"EVID-{batch}-{vial}-{stage}",
        "target_measurement_id": f"MEAS-{batch}-{vial}-{stage}",
        "target_image_id": f"{batch}-{vial}-{stage}",
        "stage_id": stage,
        "stage_ordinal": ordinal,
        "tank_id": tank,
        "relative_status": "outlier_to_two_peer_mode",
        "peer_pair_id": f"PAIR-{batch}-{stage}-{peers[0]}-{peers[1]}",
        "peer_evidence_ids": [f"EVID-{batch}-{peer}-{stage}" for peer in peers],
        "peer_measurement_ids": [f"MEAS-{batch}-{peer}-{stage}" for peer in peers],
        "peer_image_ids": [f"{batch}-{peer}-{stage}" for peer in peers],
        "target_to_peer_pairs": [
            {"pair_id": f"PAIR-{batch}-{stage}-{vial}-{peer}", "delta_e76": 6.0 + index}
            for index, peer in enumerate(peers)
        ],
        "peer_pair_delta_e76": 0.5,
        "reference_center_lab": [40.0, -2.0, 3.0],
        "reference_radius_delta_e76": 0.25,
        "target_delta_e76": 6.5,
        "threshold_delta_e76": 1.25,
        "threshold_excess_delta_e76": 5.25,
        "unit": "delta_e76",
        "limit_codes": ["relative_outlier_direction_unresolved"],
    }


def _decision(run_index: int, vial_index: int, *, three_stage: bool) -> dict[str, object]:
    batch = f"BATCH_{run_index:016x}"
    vial = f"V{vial_index}"
    tanks = ["t200", "t400", "t100"] if three_stage else ["t200", "t100"]
    return {
        "decision_id": f"DEC-{batch}-{vial}",
        "batch_id": batch,
        "run_id": f"RUN-{run_index:02d}",
        "recipe_id": str(run_index),
        "split_role": "train" if run_index <= 4 else ("validation" if run_index == 5 else "evaluation"),
        "split_version": "whole-run-varied-v3",
        "vial_id": vial,
        "disposition": "insufficient_evidence",
        "explanation_status": "UNKNOWN",
        "relative_assessment": "outlier_to_two_peer_mode",
        "stages": [_stage(batch, vial, i, tank) for i, tank in enumerate(tanks, 1)],
        "limit_codes": [
            "within_run_reference_cleanliness_unverified",
            "relative_outlier_not_directional",
        ],
        "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
    }


def _artifact() -> dict[str, object]:
    decisions = [
        _decision(run, vial, three_stage=run in {1, 4})
        for run in range(1, 7)
        for vial in range(1, 4)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_triad_decisions",
        "status": "assessed",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_measurement_sha256": "a" * 64,
        "config": {
            "config_id": "v3-relative-diagnostic-inherited-stage-v1",
            "tight_peer_pair_max_delta_e76": 2.0,
            "target_outlier_margin_delta_e76": 1.0,
        },
        "config_sha256": "b" * 64,
        "reference_policy": {
            "scope": "other_two_vials_same_run_same_stage",
            "position_prior": False,
            "cleanliness_verified": False,
            "authorization": "relative_evidence_only_no_keep_or_discard",
        },
        "decision_records": decisions,
        "denominators": {
            "runs": 6,
            "vial_decisions": 18,
            "keep": 0,
            "discard": 0,
            "insufficient_evidence": 18,
            "relative_outlier_vials": 18,
        },
        "claim_boundary": "Synthetic relative evidence only.",
    }


def _raw() -> str:
    return (
        "Evidence: UNKNOWN | The supplied within-run evidence cannot establish a verified reference.\n"
        "Uncertainty: ABSTAIN | Cause, physical defect, and next-vial effect remain unverified.\n"
        "Action: COLLECT_EVIDENCE | Collect an independent verified reference before assigning a disposition."
    )


def test_forgiving_parser_and_structure_copy_only_supplied_quantitative_evidence():
    from fluidmix.f2_varied_explanation import parse_varied_response, structure_varied_response

    decision = _artifact()["decision_records"][0]
    messy = (
        "Okay, let's see.\n"
        "- **action**: collect evidence | Collect an independent verified reference.\n"
        "### Uncertainty — abstain | Cause, physical defect, and next-vial effect remain unverified.\n"
        "> Evidence: unknown | The supplied within-run evidence cannot establish a verified reference."
    )
    parsed = parse_varied_response(messy)
    record = structure_varied_response(messy, parsed, decision)

    assert parsed["evidence_meaning"] == "unknown"
    assert parsed["abstention"] == "abstain"
    assert parsed["suggested_action"] == "collect_evidence"
    assert record["schema_version"] == "f2_varied_explanation.v1"
    assert record["disposition"] == "insufficient_evidence"
    assert record["limit_codes"] == decision["limit_codes"]
    assert record["stage_evidence"] == decision["stages"]
    assert record["stage_evidence"] is not decision["stages"]
    assert record["evidence_ids"] == [
        identity
        for stage in decision["stages"]
        for identity in [stage["target_evidence_id"], *stage["peer_evidence_ids"]]
    ]
    encoded_presentation = json.dumps(record["presentation"])
    assert "6.5" not in encoded_presentation and "EVID-" not in encoded_presentation


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        (_raw().replace("UNKNOWN", "DIFFERENCE"), "evidence_meaning"),
        (_raw().replace("ABSTAIN", "LIMITED"), "abstention"),
        (_raw().replace("COLLECT_EVIDENCE", "KEEP"), "suggested_action"),
        (_raw().replace("within-run", "within-run 6"), "numeric_fidelity"),
        (_raw().replace("within-run", "EVID-BATCH_1-V1"), "evidence_id_faithfulness"),
        (_raw().replace("within-run", "delta E76"), "unit_fidelity"),
        (_raw().replace("remain unverified", "is caused by residue"), "unsupported_claim"),
        (_raw().replace("Collect an independent", "Flush the line and collect an"), "operator_control_advice"),
    ],
)
def test_content_policy_rejects_wrong_markers_or_model_owned_facts(raw, failure):
    from fluidmix.f2_varied_explanation import parse_varied_response, varied_content_failures

    parsed = parse_varied_response(raw)
    failures = varied_content_failures(raw, parsed, _artifact()["decision_records"][0])
    assert failure in failures


def test_deterministic_control_passes_same_parser_and_content_policy():
    from fluidmix.f2_varied_explanation import (
        parse_varied_response, raw_varied_format_passes,
        varied_content_failures, varied_reference_response,
    )

    decision = _artifact()["decision_records"][0]
    raw = varied_reference_response(decision)
    assert raw == _raw() and raw_varied_format_passes(raw)
    assert varied_content_failures(raw, parse_varied_response(raw), decision) == []


def test_v4_decision_uses_the_same_grounded_deterministic_explanation_boundary():
    from fluidmix.f2_varied_explanation import (
        parse_varied_response,
        structure_varied_response,
        varied_reference_response,
    )

    decision = _decision(7, 2, three_stage=True)
    decision["split_version"] = "whole-run-varied-v4"
    raw = varied_reference_response(decision)
    structured = structure_varied_response(
        raw,
        parse_varied_response(raw),
        decision,
    )

    assert structured["context"]["split_version"] == "whole-run-varied-v4"
    assert structured["disposition"] == "insufficient_evidence"
    assert structured["explanation_status"] == "UNKNOWN"


def test_attempt_plan_has_frozen_denominators_dynamic_images_and_controls():
    from fluidmix.f2_varied_explanation import build_varied_attempt_plan

    plan = build_varied_attempt_plan(_artifact(), "Explain:\n{{EVIDENCE_RECORD_JSON}}")
    assert len(plan) == 72
    assert sum(row["kind"] == "model" for row in plan) == 54
    assert sum(row["kind"] == "control" for row in plan) == 18
    grouped = {
        condition: [row for row in plan if row["condition_id"] == condition]
        for condition in {
            "numbers_only", "ordered_stage_visual_pairs",
            "irrelevant_shuffled_images", "deterministic_template",
        }
    }
    assert all(len(rows) == 18 for rows in grouped.values())
    for relevant in grouped["ordered_stage_visual_pairs"]:
        expected = 6 if relevant["run_id"] in {"RUN-01", "RUN-04"} else 4
        assert relevant["image_count"] == expected
        assert all("same-run peer" in caption or "target" in caption for caption in relevant["image_captions"])
        irrelevant = next(row for row in grouped["irrelevant_shuffled_images"] if row["decision_id"] == relevant["decision_id"])
        assert irrelevant["image_count"] == expected
        assert not set(relevant["image_ids"]).intersection(irrelevant["image_ids"])
        assert relevant["run_id"] not in irrelevant["irrelevant_source_run_ids"]
    assert all(not row["image_ids"] for row in grouped["numbers_only"])
    assert all(row["template_response"] == _raw() for row in grouped["deterministic_template"])


def test_frozen_v3_attempt_plan_rejects_a_v4_decision_mixed_into_the_artifact():
    from fluidmix.explanation_contract import ContractError
    from fluidmix.f2_varied_explanation import build_varied_attempt_plan

    mixed = _artifact()
    mixed["decision_records"][0]["split_version"] = "whole-run-varied-v4"

    with pytest.raises(ContractError, match="split version"):
        build_varied_attempt_plan(mixed, "{{EVIDENCE_RECORD_JSON}}")


def test_prompt_binds_evidence_oracles_fail_and_stage_drift_fails_closed():
    from fluidmix.explanation_contract import ContractError
    from fluidmix.f2_varied_explanation import build_varied_attempt_plan, build_varied_prompt

    decision = _artifact()["decision_records"][0]
    prompt = build_varied_prompt("Never infer. {{EVIDENCE_RECORD_JSON}}", decision)
    assert prompt.count("6.5") == len(decision["stages"])
    assert decision["decision_id"] in prompt and "{{EVIDENCE_RECORD_JSON}}" not in prompt
    oracle = _artifact()
    oracle["decision_records"][0]["expected_disposition"] = "discard"
    with pytest.raises(ValueError, match="forbidden_deploy_input_field"):
        build_varied_attempt_plan(oracle, "{{EVIDENCE_RECORD_JSON}}")
    with pytest.raises(ContractError, match="placeholder"):
        build_varied_prompt("missing", decision)
    drifted = deepcopy(decision)
    drifted["stages"][1]["stage_ordinal"] = 4
    with pytest.raises(ContractError, match="stage ordinals"):
        build_varied_prompt("{{EVIDENCE_RECORD_JSON}}", drifted)


def test_required_uncertainty_cannot_be_replaced_by_generic_limits():
    from fluidmix.f2_varied_explanation import parse_varied_response, varied_content_failures

    raw = _raw().replace(
        "Cause, physical defect, and next-vial effect remain unverified.",
        "The reference is limited.",
    )
    assert "required_limits" in varied_content_failures(
        raw, parse_varied_response(raw), _artifact()["decision_records"][0]
    )

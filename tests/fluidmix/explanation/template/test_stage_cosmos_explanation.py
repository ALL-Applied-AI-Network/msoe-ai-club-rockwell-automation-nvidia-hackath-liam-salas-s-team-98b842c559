"""Hand-written tests for the frozen stage-evidence Cosmos text boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def test_historical_stage_serializer_has_frozen_regression_only_bytes():
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_prompt

    serialized = build_stage_prompt(
        "{{EVIDENCE_RECORD_JSON}}",
        _decision("discard"),
    ).encode("utf-8")

    assert len(serialized) == 2875
    assert hashlib.sha256(serialized).hexdigest() == (
        "cd58426e498be73bc030ebd832307dfecdefad3e6292cc4672984ef57a351b07"
    )
    assert b"trusted-predicted-mask-registry" not in serialized


def _stage(decision_id: str, ordinal: int, *, outlier: bool) -> dict:
    target = f"IMG-{decision_id}-S{ordinal}-TARGET"
    representative = f"IMG-{decision_id}-S{ordinal}-REFERENCE"
    delta = 4.25 if outlier else 0.75
    return {
        "evidence_id": f"EVID-{decision_id}-S{ordinal}",
        "measurement_id": f"MEAS-{decision_id}-S{ordinal}",
        "stage_id": f"after_t{ordinal}00",
        "stage_ordinal": ordinal,
        "time_code": (171, 342, 686)[ordinal - 1],
        "target_image_id": target,
        "target_measurement": {
            "image_id": target,
            "validity": "valid",
            "limit_codes": [],
            "lab_raw": [20.0 + ordinal, 1.0, -2.0],
            "lab_6dp": [f"{20.0 + ordinal:.6f}", "1.000000", "-2.000000"],
            "unit": "cie_lab_d65_2deg",
        },
        "reference_cases": [
            {
                "case_id": "all_reference_batches",
                "mode": "all_reference_batches",
                "alternative_id": None,
                "disposition_source": True,
                "validity": "valid",
                "limit_codes": [],
                "peer_ids": [representative, f"IMG-{decision_id}-S{ordinal}-PEER2"],
                "peer_origins": ["reference_bank", "reference_bank"],
                "peer_count": 2,
                "peer_batch_ids": ["REF-A", "REF-B"],
                "peer_batch_count": 2,
                "representative_peer_id": representative,
                "center_lab_raw": [20.0, 1.0, -2.0],
                "center_lab_6dp": ["20.000000", "1.000000", "-2.000000"],
                "median_radius_delta_e76_raw": 0.4,
                "max_radius_delta_e76_raw": 0.5,
                "mad_radius_delta_e76_raw": 0.1,
                "pairwise_diameter_delta_e76_raw": 0.8,
                "threshold_delta_e76_raw": 2.0,
                "threshold_delta_e76_6dp": "2.000000",
                "target_delta_e76_raw": delta,
                "target_delta_e76_6dp": f"{delta:.6f}",
                "stage_outlier": outlier,
                "unit": "delta_e76",
            }
        ],
        "primary_mode": "all_reference_batches",
        "validity": "valid",
        "limit_codes": [],
        "target_delta_e76_raw": delta,
        "target_delta_e76_6dp": f"{delta:.6f}",
        "threshold_delta_e76_raw": 2.0,
        "threshold_delta_e76_6dp": "2.000000",
        "stage_outlier": outlier,
        "unit": "delta_e76",
    }


def _decision(disposition: str = "discard", *, suffix: str = "A") -> dict:
    status = {
        "discard": "DIFFERENCE",
        "keep": "NO_DIFFERENCE",
        "insufficient_evidence": "UNKNOWN",
    }[disposition]
    stages = [
        _stage(f"DEC-{suffix}", ordinal, outlier=disposition == "discard" and ordinal == 2)
        for ordinal in range(1, 4)
    ]
    if disposition == "insufficient_evidence":
        stage = stages[1]
        primary = stage["reference_cases"][0]
        stage.update(
            validity="insufficient_evidence",
            limit_codes=["reference_not_unique_or_too_small"],
            target_delta_e76_raw=None,
            target_delta_e76_6dp=None,
            threshold_delta_e76_raw=None,
            threshold_delta_e76_6dp=None,
            stage_outlier=None,
        )
        primary.update(
            validity="insufficient_evidence",
            limit_codes=["reference_not_unique_or_too_small"],
            peer_ids=None,
            peer_origins=None,
            peer_count=None,
            peer_batch_ids=None,
            peer_batch_count=None,
            representative_peer_id=None,
            center_lab_raw=None,
            center_lab_6dp=None,
            median_radius_delta_e76_raw=None,
            max_radius_delta_e76_raw=None,
            mad_radius_delta_e76_raw=None,
            pairwise_diameter_delta_e76_raw=None,
            threshold_delta_e76_raw=None,
            threshold_delta_e76_6dp=None,
            target_delta_e76_raw=None,
            target_delta_e76_6dp=None,
            stage_outlier=None,
        )
    return {
        "decision_id": f"DEC-{suffix}",
        "family_id": "VA",
        "batch_id": f"BATCH-{suffix}",
        "cohort": "scored_target",
        "partition": "validation",
        "previous_sim_recipe_id": "S0160",
        "current_sim_recipe_id": "S0380",
        "previous_programmed_ml": {"t100": 30, "t200": 20, "t400": 10},
        "current_programmed_ml": {"t100": 10, "t200": 30, "t400": 20},
        "programmed_dispense_order": ["t200", "t400", "t100"],
        "vial_id": "V1",
        "vial_position": 1,
        "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
        "stages": stages,
        "disposition": disposition,
        "explanation_status": status,
        "disposition_source": "deterministic_pipeline_primary_reference_evidence",
        "limit_codes": (
            ["reference_not_unique_or_too_small"]
            if disposition == "insufficient_evidence"
            else []
        ),
    }


def _use_f2_events(decision: dict, *, frame_id=None) -> dict:
    for stage in decision["stages"]:
        ordinal = stage["stage_ordinal"]
        stage.pop("time_code")
        stage["stage_event"] = {
            "event_id": stage["stage_id"],
            "stage_ordinal": ordinal,
            "tank_id": ("t200", "t400", "t100")[ordinal - 1],
            "tank_order": ordinal,
            "physics_step": ordinal * 120,
            "frame_id": frame_id,
            "event_relative_timestamp_s": float(ordinal),
            "final_observation_offset_s": 0.3,
        }
    return decision


def _raw(disposition: str) -> str:
    if disposition == "keep":
        evidence, uncertainty, action = "NO_DIFFERENCE", "LIMITED", "KEEP"
        evidence_words = (
            "All valid stage comparisons remain within their frozen reference thresholds."
        )
        action_words = "Keep this vial under the supplied synthetic evidence."
    elif disposition == "discard":
        evidence, uncertainty, action = "DIFFERENCE", "LIMITED", "DISCARD"
        evidence_words = (
            "One or more valid stage comparisons exceed their frozen reference thresholds."
        )
        action_words = "Discard this vial under the supplied synthetic evidence."
    else:
        evidence, uncertainty, action = "UNKNOWN", "ABSTAIN", "COLLECT_EVIDENCE"
        evidence_words = "The supplied stage evidence cannot support a vial disposition."
        action_words = "Collect the missing evidence before assigning a disposition."
    return (
        f"Evidence: {evidence} | {evidence_words}\n"
        f"Uncertainty: {uncertainty} | Cause, physical defect, and next-vial effect remain unverified.\n"
        f"Action: {action} | {action_words}"
    )


def test_stage_parser_accepts_filler_markdown_casing_and_line_order():
    from fluidmix.explanation.template.stage_cosmos_explanation import parse_stage_response

    parsed = parse_stage_response(
        """Okay, let's see.
        - **ACTION** — `discard` | Discard this vial under the supplied evidence.
        ### uncertainty: limited | Cause, physical defect, and next-vial effect remain unverified.
        > Evidence: **difference** — A valid stage comparison exceeds its frozen reference threshold.
        Thanks.
        """
    )

    assert parsed == {
        "evidence_meaning": "difference_observed",
        "abstention": "limited",
        "suggested_action": "discard",
        "wording": {
            "evidence": "A valid stage comparison exceeds its frozen reference threshold.",
            "uncertainty": (
                "Cause, physical defect, and next-vial effect remain unverified."
            ),
            "action": "Discard this vial under the supplied evidence.",
        },
    }


def test_stage_parser_recovers_marker_only_model_artifact_without_inventing_wording():
    raw = (
        "Evidence: DIFFERENCE\n"
        "Uncertainty: LIMITED\n"
        "Action: DISCARD"
    )

    from fluidmix.explanation.template.stage_cosmos_explanation import parse_stage_response

    assert parse_stage_response(raw) == {
        "evidence_meaning": "difference_observed",
        "abstention": "limited",
        "suggested_action": "discard",
        "wording": {
            "evidence": "",
            "uncertainty": "",
            "action": "",
        },
    }


@pytest.mark.parametrize(
    "raw",
    [
        "Evidence: UNKNOWN | unavailable\nUncertainty: ABSTAIN | unknown",
        (
            "Evidence: UNKNOWN | unavailable\n"
            "Evidence: DIFFERENCE | conflict\n"
            "Uncertainty: ABSTAIN | unknown\n"
            "Action: INSPECT | inspect this vial"
        ),
        (
            "Evidence: DIFFERENCE NO_DIFFERENCE | conflict\n"
            "Uncertainty: LIMITED | bounded\n"
            "Action: DISCARD | discard this vial"
        ),
        (
            "Evidence: DIFFERENCE |\n"
            "Uncertainty: LIMITED | bounded\n"
            "Action: DISCARD | discard this vial"
        ),
    ],
)
def test_stage_parser_fails_closed_on_missing_duplicate_ambiguous_or_empty_fields(raw):
    from fluidmix.explanation.template.explanation_contract import ContractError
    from fluidmix.explanation.template.stage_cosmos_explanation import parse_stage_response

    with pytest.raises(ContractError):
        parse_stage_response(raw)


def test_exact_three_line_format_is_scored_separately_from_lenient_parse():
    from fluidmix.explanation.template.stage_cosmos_explanation import (
        parse_stage_response,
        raw_stage_format_passes,
    )

    exact = _raw("discard")
    messy = "Okay.\n" + exact.replace("Evidence:", "> **evidence** —")

    assert raw_stage_format_passes(exact) is True
    assert parse_stage_response(messy)["evidence_meaning"] == "difference_observed"
    assert raw_stage_format_passes(messy) is False


def test_prompt_and_structured_record_keep_deterministic_fields_out_of_model_control():
    from fluidmix.explanation.template.stage_cosmos_explanation import (
        build_stage_prompt,
        parse_stage_response,
        structure_stage_response,
    )

    decision = _decision("discard")
    template = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    prompt = build_stage_prompt(template, decision)
    raw = _raw("discard")
    parsed = parse_stage_response(raw)
    structured = structure_stage_response(raw, parsed, decision)

    assert template.count("{{EVIDENCE_RECORD_JSON}}") == 1
    assert '"disposition":"discard"' in prompt
    assert '"target_delta_e76_raw":4.25' in prompt
    assert "synthetic_profile_id" not in prompt
    assert structured["disposition"] == decision["disposition"]
    assert structured["explanation_status"] == decision["explanation_status"]
    assert structured["evidence_ids"] == [
        stage["evidence_id"] for stage in decision["stages"]
    ]
    assert [row["target_delta_e76_raw"] for row in structured["stage_evidence"]] == [
        stage["target_delta_e76_raw"] for stage in decision["stages"]
    ]
    assert [row["unit"] for row in structured["stage_evidence"]] == [
        "delta_e76",
        "delta_e76",
        "delta_e76",
    ]
    assert structured["presentation"] == parsed


def test_unsafe_model_wording_is_never_published_as_a_structured_record():
    from fluidmix.explanation.template.explanation_contract import ContractError
    from fluidmix.explanation.template.stage_cosmos_explanation import (
        parse_stage_response,
        structure_stage_response,
    )

    raw = _raw("discard").replace(
        "One or more valid stage comparisons",
        "EVID-WRONG reports 99 delta_e76 and",
    )

    with pytest.raises(ContractError, match="content policy"):
        structure_stage_response(raw, parse_stage_response(raw), _decision("discard"))


@pytest.mark.parametrize(
    "raw, expected_failure",
    [
        (
            _raw("discard").replace("Evidence: DIFFERENCE", "Evidence: NO_DIFFERENCE"),
            "evidence_meaning",
        ),
        (
            _raw("discard").replace("Uncertainty: LIMITED", "Uncertainty: ABSTAIN"),
            "abstention",
        ),
        (
            _raw("discard").replace("Action: DISCARD", "Action: KEEP"),
            "suggested_action",
        ),
        (
            _raw("discard").replace(
                "One or more valid stage comparisons",
                "EVID-DEC-A-S2 reports 4.25 delta_e76 and",
            ),
            "numeric_fidelity",
        ),
        (
            _raw("discard").replace(
                "One or more valid stage comparisons",
                "EVID-WRONG reports a comparison that",
            ),
            "evidence_id_faithfulness",
        ),
        (
            _raw("discard").replace(
                "Cause, physical defect, and next-vial effect remain unverified.",
                "Uncertainty remains.",
            ),
            "required_limits",
        ),
        (
            _raw("discard").replace(
                "One or more valid stage comparisons exceed",
                "Carryover residue caused the comparisons to exceed",
            ),
            "unsupported_claim",
        ),
        (
            _raw("discard").replace(
                "One or more valid stage comparisons exceed their frozen reference thresholds.",
                "A physical defect exists.",
            ),
            "unsupported_claim",
        ),
        (
            _raw("discard").replace(
                "Discard this vial under the supplied synthetic evidence.",
                "Discard this vial and increase pump flow.",
            ),
            "operator_control_advice",
        ),
    ],
)
def test_content_policy_stays_strict_after_lenient_parse(raw, expected_failure):
    from fluidmix.explanation.template.stage_cosmos_explanation import (
        parse_stage_response,
        stage_content_failures,
    )

    parsed = parse_stage_response(raw)

    assert expected_failure in stage_content_failures(raw, parsed, _decision("discard"))


@pytest.mark.parametrize("disposition", ["keep", "discard", "insufficient_evidence"])
def test_deterministic_control_passes_same_parser_and_content_policy(disposition):
    from fluidmix.explanation.template.stage_cosmos_explanation import (
        parse_stage_response,
        raw_stage_format_passes,
        stage_content_failures,
        stage_reference_response,
    )

    decision = _decision(disposition)
    raw = stage_reference_response(decision)
    parsed = parse_stage_response(raw)

    assert raw == _raw(disposition)
    assert raw_stage_format_passes(raw) is True
    assert stage_content_failures(raw, parsed, decision) == []


def test_attempt_plan_has_identical_records_and_prompts_across_three_model_conditions():
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_attempt_plan

    decisions = [_decision("discard", suffix="A"), _decision("keep", suffix="B")]
    measured = {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "config_id": "three-vial-separated-reference-bank-v1",
        "decision_records": decisions,
        "data_origin": "twin",
        "real_data_used": False,
    }
    template = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")

    plan = build_stage_attempt_plan(measured, template)

    assert len(plan) == 8
    assert [row["condition_id"] for row in plan[:4]] == [
        "numbers_only",
        "ordered_stage_visual_pairs",
        "irrelevant_shuffled_images",
        "deterministic_template",
    ]
    model_rows = plan[:3]
    assert len({row["prompt"] for row in model_rows}) == 1
    assert len({row["evidence_record_sha256"] for row in model_rows}) == 1
    assert model_rows[0]["image_ids"] == []
    relevant = model_rows[1]["image_ids"]
    irrelevant = model_rows[2]["image_ids"]
    assert len(relevant) == 6
    assert len(irrelevant) == len(relevant)
    assert set(relevant).isdisjoint(irrelevant)
    assert all("DEC-B" in image_id for image_id in irrelevant)
    assert model_rows[1]["image_captions"] == [
        "Stage 1 target image.",
        "Stage 1 reference image.",
        "Stage 2 target image.",
        "Stage 2 reference image.",
        "Stage 3 target image.",
        "Stage 3 reference image.",
    ]
    assert model_rows[2]["image_captions"] == model_rows[1]["image_captions"]
    assert model_rows[2]["irrelevant_source_decision_ids"] == ["DEC-B"]
    assert model_rows[2]["irrelevant_source_statuses"] == ["NO_DIFFERENCE"]
    assert plan[3]["prompt"] is None
    assert plan[3]["template_response"] == _raw("discard")
    assert "oracle" not in repr(plan).lower()


def test_attempt_plan_accepts_physics_step_event_with_null_shared_frame_id():
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_attempt_plan

    decisions = [
        _use_f2_events(_decision("discard", suffix="A")),
        _use_f2_events(_decision("keep", suffix="B")),
    ]
    measured = {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "config_id": "f2-physics-step-events",
        "decision_records": decisions,
        "data_origin": "twin",
        "real_data_used": False,
    }
    template = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(
        encoding="utf-8"
    )

    plan = build_stage_attempt_plan(measured, template)

    assert len(plan) == 8
    prompt = next(row["prompt"] for row in plan if row["prompt"] is not None)
    assert '"frame_id":null' in prompt


@pytest.mark.parametrize("physics_step", [None, -1, 1.5, True])
def test_null_frame_id_rejects_missing_or_invalid_physics_step(physics_step):
    from fluidmix.explanation.template.explanation_contract import ContractError
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_prompt

    decision = _use_f2_events(_decision("keep"))
    decision["stages"][0]["stage_event"]["physics_step"] = physics_step
    template = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(
        encoding="utf-8"
    )

    with pytest.raises(ContractError):
        build_stage_prompt(template, decision)


@pytest.mark.parametrize("frame_id", ["", 7])
def test_malformed_nonnull_frame_id_is_rejected(frame_id):
    from fluidmix.explanation.template.explanation_contract import ContractError
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_prompt

    decision = _use_f2_events(_decision("keep"), frame_id=frame_id)
    template = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(
        encoding="utf-8"
    )

    with pytest.raises(ContractError):
        build_stage_prompt(template, decision)


def test_irrelevant_images_prefer_one_disjoint_decision_with_different_status():
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_attempt_plan

    decisions = [
        _decision("discard", suffix="A"),
        _decision("discard", suffix="B"),
        _decision("keep", suffix="C"),
    ]
    measured = {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "config_id": "three-vial-separated-reference-bank-v1",
        "decision_records": decisions,
        "data_origin": "twin",
        "real_data_used": False,
    }
    template = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")

    plan = build_stage_attempt_plan(measured, template)
    irrelevant = next(
        row
        for row in plan
        if row["decision_id"] == "DEC-A"
        and row["condition_id"] == "irrelevant_shuffled_images"
    )

    assert irrelevant["irrelevant_source_decision_ids"] == ["DEC-C"]
    assert irrelevant["irrelevant_source_statuses"] == ["NO_DIFFERENCE"]
    assert all("DEC-C" in image_id for image_id in irrelevant["image_ids"])

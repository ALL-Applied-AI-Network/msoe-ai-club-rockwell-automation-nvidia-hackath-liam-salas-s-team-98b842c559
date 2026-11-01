from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.template.explanation_contract import (  # noqa: E402
    ContractError,
    build_prompt,
    canonical_json,
    parse_and_validate_explanation,
    reference_explanation,
    validate_evidence_record,
    validate_fixture_set,
    validate_oracle,
)


def valid_difference_record():
    return {
        "schema_version": "operator_evidence.v1",
        "record_id": "r001",
        "origin": "synthetic_contract",
        "camera_evidence": {
            "measurement_id": "measurement:r001",
            "comparison_id": "comparison:r001",
            "feed": "vial_fill",
            "validity": "valid",
            "reason_codes": [],
            "reference_origin": "same_recipe_vial_fill",
            "comparison_result": "difference",
            "delta_e76": "12.40",
            "threshold_delta_e76": "8.00",
            "direction_to_previous": "toward",
        },
        "recipe_context": {
            "evidence_id": "recipe:r001",
            "current": {
                "recipe_id": "recipe-101",
                "t100_ml": 30,
                "t200_ml": 20,
                "t400_ml": 10,
            },
            "previous": {
                "recipe_id": "recipe-100",
                "t100_ml": 20,
                "t200_ml": 0,
                "t400_ml": 30,
            },
            "previous_missing_reason": None,
            "dispense_order": ["t200", "t400", "t100"],
        },
    }


def test_reference_policy_reports_appearance_without_diagnosis():
    result = reference_explanation(valid_difference_record())

    assert result["status"] == "appearance_only"
    assert result["appearance"] == "difference_observed"
    assert result["defect"] == result["cause"] == result["next_vial_effect"] == "unknown"
    assert result["confidence"] == "not_assigned"
    assert result["operator_action"] == "manual_review_required"
    assert result["evidence_ids"] == [
        "comparison:r001",
        "measurement:r001",
        "recipe:r001",
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(expected_cause="carryover"),
        lambda record: record.update(record_id="carryover_case"),
        lambda record: record["camera_evidence"].update(delta_e76=12.4),
        lambda record: record["camera_evidence"].update(comparison_result="no_difference"),
        lambda record: record["recipe_context"].update(
            dispense_order=["t100", "t200", "t400"]
        ),
    ],
)
def test_answer_bearing_or_noncanonical_evidence_is_rejected(mutate):
    record = valid_difference_record()
    mutate(record)

    with pytest.raises(ContractError):
        validate_evidence_record(record)


def test_uncertain_invalid_and_missing_previous_records_abstain():
    uncertain = valid_difference_record()
    uncertain["camera_evidence"].update(
        validity="uncertain",
        reason_codes=["lighting_variation"],
        comparison_result="uncertain",
        delta_e76=None,
        threshold_delta_e76=None,
        direction_to_previous="uncertain",
    )
    invalid = deepcopy(uncertain)
    invalid["camera_evidence"].update(
        validity="invalid",
        reason_codes=["underexposed"],
        comparison_result="unavailable",
        direction_to_previous="unavailable",
    )
    missing = valid_difference_record()
    missing["recipe_context"].update(previous=None, previous_missing_reason="not_recorded")
    missing["camera_evidence"]["direction_to_previous"] = "unavailable"

    uncertain_result = reference_explanation(uncertain)
    invalid_result = reference_explanation(invalid)
    missing_result = reference_explanation(missing)

    assert uncertain_result["status"] == "insufficient_evidence"
    assert invalid_result["status"] == "invalid_input"
    assert missing_result["status"] == "insufficient_evidence"
    for result in (uncertain_result, invalid_result, missing_result):
        assert result["appearance"] == "unknown"
        assert result["quantitative_evidence"] == []


def test_image_recipe_mode_omits_camera_claims():
    result = reference_explanation(
        valid_difference_record(), camera_evidence_included=False
    )

    assert result["status"] == "insufficient_evidence"
    assert result["appearance"] == "unknown"
    assert result["evidence_ids"] == ["recipe:r001"]
    assert "quantitative_evidence_omitted" in result["limit_codes"]


def test_strict_parser_never_extracts_repairs_or_deduplicates():
    record = valid_difference_record()
    raw = canonical_json(reference_explanation(record))

    assert parse_and_validate_explanation(raw, record) == reference_explanation(record)
    for invalid in (
        f"```json\n{raw}\n```",
        f"operator response: {raw}",
        '{"schema_version":"operator_explanation.v1",'
        '"schema_version":"operator_explanation.v1"}',
        "NaN",
    ):
        with pytest.raises(ContractError):
            parse_and_validate_explanation(invalid, record)


def test_altered_value_id_unknown_field_or_diagnosis_fails_policy():
    record = valid_difference_record()
    variants = []
    for field, value in (
        ("cause", "previous_recipe_residue"),
        ("record_id", "r002"),
    ):
        output = reference_explanation(record)
        output[field] = value
        variants.append(output)
    output = reference_explanation(record)
    output["quantitative_evidence"][0]["value"] = "12.4"
    variants.append(output)
    output = reference_explanation(record)
    output["model_version"] = "model-supplied"
    variants.append(output)

    for output in variants:
        with pytest.raises(ContractError):
            parse_and_validate_explanation(json.dumps(output), record)


def test_prompt_pair_is_identical_and_image_only_payload_omits_measurements():
    template = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text()
    record = valid_difference_record()

    full_a = build_prompt(template, record)
    full_b = build_prompt(template, record)
    image_only = build_prompt(template, record, camera_evidence_included=False)

    assert full_a.encode() == full_b.encode()
    assert '"camera_evidence":null' in image_only
    assert "12.40" not in image_only
    assert "measurement:r001" not in image_only
    assert "recipe:r001" in image_only


def test_frozen_generic_inputs_retain_reviewed_hashes_and_oracle_binding():
    fixture_path = ROOT / "data/cosmos_contract_fixtures_v1.json"
    oracle_path = ROOT / "data/cosmos_contract_oracle_v1.json"
    prompt_path = ROOT / "data/cosmos_operator_prompt_v1.txt"

    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == (
        "e88f6f13476ae89a49d53e0ddfa6cd2be051530c073e946f5e985c16d2f0ee9f"
    )
    assert hashlib.sha256(oracle_path.read_bytes()).hexdigest() == (
        "cb2d45ca1fce48e7603a6cde52ff0cbb1bdad11e7b2285ad01b80cbd232faab9"
    )
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == (
        "0733da91d7c1c347feae3afabdc170ba3a41fa5a9726f310a40eab887f3eb74c"
    )
    fixture_bytes = fixture_path.read_bytes()
    fixtures = json.loads(fixture_bytes)
    oracle = json.loads(oracle_path.read_text())
    validate_fixture_set(fixtures)
    validate_oracle(oracle, fixture_bytes, fixtures)

from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.explanation_contract import (
    ContractError,
    build_prompt,
    canonical_json,
    parse_and_validate_explanation,
    reference_explanation,
    validate_evidence_record,
    validate_eval_config,
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


def test_valid_evidence_record_is_accepted():
    validate_evidence_record(valid_difference_record())


def test_schema_and_answer_bearing_fields_are_rejected():
    variants = []
    record = valid_difference_record()
    record["expected_cause"] = "carryover"
    variants.append(record)
    record = valid_difference_record()
    record["schema_version"] = "operator_evidence.v2"
    variants.append(record)
    record = valid_difference_record()
    record["origin"] = "real_validation"
    variants.append(record)
    record = valid_difference_record()
    record["record_id"] = "carryover_case"
    variants.append(record)

    for variant in variants:
        with pytest.raises(ContractError):
            validate_evidence_record(variant)


def test_camera_ids_values_and_state_must_be_canonical():
    variants = []
    record = valid_difference_record()
    record["camera_evidence"]["measurement_id"] = "measurement:other"
    variants.append(record)
    record = valid_difference_record()
    record["camera_evidence"]["delta_e76"] = 12.4
    variants.append(record)
    record = valid_difference_record()
    record["camera_evidence"]["delta_e76"] = "12.4"
    variants.append(record)
    record = valid_difference_record()
    record["camera_evidence"]["comparison_result"] = "no_difference"
    variants.append(record)
    record = valid_difference_record()
    record["camera_evidence"]["reason_codes"] = ["carryover"]
    variants.append(record)

    for variant in variants:
        with pytest.raises(ContractError):
            validate_evidence_record(variant)


def test_recipe_volumes_order_and_missing_previous_are_checked():
    variants = []
    record = valid_difference_record()
    record["recipe_context"]["current"]["t200_ml"] = 60
    variants.append(record)
    record = valid_difference_record()
    record["recipe_context"]["current"]["t100_ml"] = True
    variants.append(record)
    record = valid_difference_record()
    record["recipe_context"]["current"]["t300_ml"] = 10
    variants.append(record)
    record = valid_difference_record()
    record["recipe_context"]["dispense_order"] = ["t100", "t200", "t400"]
    variants.append(record)
    record = valid_difference_record()
    record["recipe_context"]["previous"] = None
    variants.append(record)

    for variant in variants:
        with pytest.raises(ContractError):
            validate_evidence_record(variant)


def test_reference_policy_validates_before_mapping():
    record = deepcopy(valid_difference_record())
    record["recipe_context"]["current"]["t300_ml"] = 10

    with pytest.raises(ContractError):
        reference_explanation(record)


def test_reference_explanation_reports_appearance_without_diagnosis():
    result = reference_explanation(valid_difference_record())

    assert result == {
        "schema_version": "operator_explanation.v1",
        "record_id": "r001",
        "status": "appearance_only",
        "appearance": "difference_observed",
        "defect": "unknown",
        "cause": "unknown",
        "next_vial_effect": "unknown",
        "confidence": "not_assigned",
        "evidence_ids": ["comparison:r001", "measurement:r001", "recipe:r001"],
        "quantitative_evidence": [
            {
                "evidence_id": "comparison:r001",
                "metric": "delta_e76",
                "value": "12.40",
                "unit": "delta_e76",
            },
            {
                "evidence_id": "comparison:r001",
                "metric": "threshold_delta_e76",
                "value": "8.00",
                "unit": "delta_e76",
            },
        ],
        "limit_codes": [
            "cause_not_verified",
            "defect_not_verified",
            "next_vial_effect_not_measured",
        ],
        "operator_action": "manual_review_required",
    }


def test_reference_explanation_allows_no_difference_without_automatic_action():
    record = valid_difference_record()
    record["camera_evidence"]["comparison_result"] = "no_difference"
    record["camera_evidence"]["delta_e76"] = "3.20"

    result = reference_explanation(record)

    assert result["status"] == "appearance_only"
    assert result["appearance"] == "no_difference_observed"
    assert result["operator_action"] == "no_automatic_action"
    assert result["defect"] == result["cause"] == result["next_vial_effect"] == "unknown"


def test_uncertain_camera_evidence_abstains_and_drops_numbers():
    record = valid_difference_record()
    record["camera_evidence"].update({
        "validity": "uncertain",
        "reason_codes": ["lighting_variation"],
        "comparison_result": "uncertain",
        "delta_e76": None,
        "threshold_delta_e76": None,
        "direction_to_previous": "uncertain",
    })

    result = reference_explanation(record)

    assert result["status"] == "insufficient_evidence"
    assert result["appearance"] == "unknown"
    assert result["quantitative_evidence"] == []
    assert result["operator_action"] == "manual_review_required"
    assert "camera_evidence_uncertain" in result["limit_codes"]


def test_invalid_camera_evidence_is_invalid_input():
    record = valid_difference_record()
    record["camera_evidence"].update({
        "validity": "invalid",
        "reason_codes": ["underexposed"],
        "comparison_result": "unavailable",
        "delta_e76": None,
        "threshold_delta_e76": None,
        "direction_to_previous": "unavailable",
    })

    result = reference_explanation(record)

    assert result["status"] == "invalid_input"
    assert result["appearance"] == "unknown"
    assert result["quantitative_evidence"] == []
    assert "camera_evidence_invalid" in result["limit_codes"]


def test_missing_previous_recipe_forces_insufficient_evidence():
    record = valid_difference_record()
    record["recipe_context"]["previous"] = None
    record["recipe_context"]["previous_missing_reason"] = "not_recorded"
    record["camera_evidence"]["direction_to_previous"] = "unavailable"

    result = reference_explanation(record)

    assert result["status"] == "insufficient_evidence"
    assert result["appearance"] == "unknown"
    assert result["quantitative_evidence"] == []
    assert "previous_recipe_missing" in result["limit_codes"]


def test_image_recipe_mode_omits_measurement_claims_and_abstains():
    result = reference_explanation(valid_difference_record(), camera_evidence_included=False)

    assert result["status"] == "insufficient_evidence"
    assert result["appearance"] == "unknown"
    assert result["evidence_ids"] == ["recipe:r001"]
    assert result["quantitative_evidence"] == []
    assert "quantitative_evidence_omitted" in result["limit_codes"]


def test_exact_reference_json_is_accepted():
    record = valid_difference_record()
    expected = reference_explanation(record)

    assert parse_and_validate_explanation(canonical_json(expected), record) == expected


def test_json_is_never_fenced_repaired_or_deduplicated():
    record = valid_difference_record()
    raw = canonical_json(reference_explanation(record))

    for invalid in (
        f"```json\n{raw}\n```",
        '{"schema_version":"operator_explanation.v1","schema_version":"operator_explanation.v1"}',
        "NaN",
    ):
        with pytest.raises(ContractError):
            parse_and_validate_explanation(invalid, record)


def test_invented_cause_number_id_or_field_fails_policy():
    record = valid_difference_record()
    variants = []
    output = reference_explanation(record)
    output["cause"] = "previous_recipe_residue"
    variants.append(output)
    output = reference_explanation(record)
    output["quantitative_evidence"][0]["value"] = "12.4"
    variants.append(output)
    output = reference_explanation(record)
    output["evidence_ids"].append("sensor:scale")
    variants.append(output)
    output = reference_explanation(record)
    output["model_version"] = "model-supplied"
    variants.append(output)

    for variant in variants:
        with pytest.raises(ContractError):
            parse_and_validate_explanation(json.dumps(variant), record)


def test_image_recipe_response_uses_the_separate_abstention_policy():
    record = valid_difference_record()
    expected = reference_explanation(record, camera_evidence_included=False)

    assert parse_and_validate_explanation(
        canonical_json(expected), record, camera_evidence_included=False
    ) == expected
    with pytest.raises(ContractError):
        parse_and_validate_explanation(
            canonical_json(reference_explanation(record)),
            record,
            camera_evidence_included=False,
        )


def test_prompt_pair_is_identical_and_image_only_payload_omits_measurements():
    template = (Path(__file__).resolve().parents[1] / "data/cosmos_operator_prompt_v1.txt").read_text()
    record = valid_difference_record()

    full_a = build_prompt(template, record)
    full_b = build_prompt(template, record)
    image_only = build_prompt(template, record, camera_evidence_included=False)

    assert full_a.encode() == full_b.encode()
    assert '"camera_evidence":null' in image_only
    assert "12.40" not in image_only
    assert "measurement:r001" not in image_only
    assert "comparison:r001" not in image_only
    assert "recipe:r001" in image_only
    lowered_template = template.lower()
    for answer_word in ("carryover", "wrong_volume", "wrong volume", "residue"):
        assert answer_word not in lowered_template


def test_frozen_fixture_file_has_six_neutral_valid_records():
    path = Path(__file__).resolve().parents[1] / "data/cosmos_contract_fixtures_v1.json"
    payload = json.loads(path.read_text())

    validate_fixture_set(payload)
    assert [item["record_id"] for item in payload["fixtures"]] == [
        "r001", "r002", "r003", "r004", "r005", "r006"
    ]
    lowered = path.read_text().lower()
    assert "carryover" not in lowered
    assert "wrong_volume" not in lowered
    assert "expected_" not in lowered


def test_sealed_oracle_matches_fixture_bytes_and_policy():
    root = Path(__file__).resolve().parents[1]
    fixture_bytes = (root / "data/cosmos_contract_fixtures_v1.json").read_bytes()
    fixtures = json.loads(fixture_bytes)
    oracle = json.loads((root / "data/cosmos_contract_oracle_v1.json").read_text())

    validate_oracle(oracle, fixture_bytes, fixtures)
    assert set(oracle["expected_by_record"]) == {
        "r001", "r002", "r003", "r004", "r005", "r006"
    }


def test_eval_config_freezes_exact_twenty_four_attempt_matrix():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "data/cosmos_contract_eval_v1.json").read_text())

    validate_eval_config(config)
    assert config["planned_attempts"] == config["fixture_count"] * len(config["modes"]) == 24
    assert config["modes"] == [
        "image_evidence_recipe",
        "evidence_recipe",
        "image_recipe",
        "template_reference",
    ]


def test_eval_config_hashes_the_runner_visible_inputs_and_hidden_oracle():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "data/cosmos_contract_eval_v1.json").read_text())
    paths = {
        "fixture_set_sha256": root / "data/cosmos_contract_fixtures_v1.json",
        "oracle_sha256": root / "data/cosmos_contract_oracle_v1.json",
        "prompt_sha256": root / "data/cosmos_operator_prompt_v1.txt",
    }

    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_usable_output import (  # noqa: E402
    build_usable_prompt,
    parse_usable_response,
    structure_usable_response,
    usable_reference_response,
)
from fluidmix.explanation_contract import ContractError, reference_explanation  # noqa: E402


def _record(record_id="r001"):
    fixtures = json.loads(
        (ROOT / "data/cosmos_contract_fixtures_v1.json").read_text()
    )["fixtures"]
    return next(record for record in fixtures if record["record_id"] == record_id)


def test_lenient_parser_accepts_preamble_markdown_order_casing_and_filler():
    raw = """Okay, let's see.
    - **SUGGESTED Action** — `manual review` | Have a person review the evidence.
    ### uncertainty: limited -- Cause is unverified; defect is unknown; next vial effect is not measured.
    > Explanation: **difference** — The supplied comparison reports an appearance difference.
    Thanks.
    """

    parsed = parse_usable_response(raw)

    assert parsed == {
        "evidence_meaning": "difference_observed",
        "abstention": "limited",
        "suggested_action": "manual_review_required",
        "wording": {
            "explanation": "The supplied comparison reports an appearance difference.",
            "uncertainty": (
                "Cause is unverified; defect is unknown; next vial effect is not measured."
            ),
            "suggested_action": "Have a person review the evidence.",
        },
    }


def test_lenient_parser_recovers_marker_only_model_artifact_without_inventing_wording():
    """The Nano artifact omitted the required prose after each marker."""
    raw = (
        "Evidence meaning: DIFFERENCE\n"
        "Uncertainty: LIMITED\n"
        "Suggested action: MANUAL_REVIEW"
    )

    assert parse_usable_response(raw) == {
        "evidence_meaning": "difference_observed",
        "abstention": "limited",
        "suggested_action": "manual_review_required",
        "wording": {
            "explanation": "",
            "uncertainty": "",
            "suggested_action": "",
        },
    }


@pytest.mark.parametrize(
    "raw",
    [
        "Explanation: UNKNOWN | unavailable\nUncertainty: ABSTAIN | cause unknown",
        (
            "Explanation: UNKNOWN | unavailable\n"
            "Explanation: DIFFERENCE | conflict\n"
            "Uncertainty: ABSTAIN | cause unknown\n"
            "Action: MANUAL_REVIEW | review"
        ),
        (
            "Explanation: DIFFERENCE NO_DIFFERENCE | conflict\n"
            "Uncertainty: LIMITED | all limits\n"
            "Action: MANUAL_REVIEW | review"
        ),
        (
            "Explanation: DIFFERENCE |\n"
            "Uncertainty: LIMITED | all limits\n"
            "Action: MANUAL_REVIEW | review"
        ),
    ],
)
def test_lenient_parser_fails_closed_on_missing_duplicate_ambiguous_or_empty_fields(raw):
    with pytest.raises(ContractError):
        parse_usable_response(raw)


def test_structure_copies_all_numbers_ids_units_and_limits_from_supplied_evidence():
    record = _record("r001")
    parsed = parse_usable_response(
        """Explanation: DIFFERENCE | The comparison reports a difference.
        Uncertainty: LIMITED | Cause unverified; defect unverified; next-vial effect unverified.
        Suggested action: MANUAL_REVIEW | Have a person review the evidence.
        """
    )

    structured = structure_usable_response(parsed, record)
    expected = reference_explanation(record)

    assert structured["schema_version"] == "operator_explanation_usable.v1"
    assert structured["record_id"] == record["record_id"]
    assert structured["evidence_ids"] == expected["evidence_ids"]
    assert structured["quantitative_evidence"] == expected["quantitative_evidence"]
    assert structured["limit_codes"] == expected["limit_codes"]
    assert structured["policy_fields"] == {
        key: expected[key]
        for key in (
            "status",
            "appearance",
            "defect",
            "cause",
            "next_vial_effect",
            "confidence",
            "operator_action",
        )
    }
    assert structured["presentation"] == parsed


def test_prompt_omits_camera_payload_without_changing_template_or_recipe():
    template = (ROOT / "data/cosmos_operator_prompt_usable_v1.txt").read_text()
    record = _record("r002")

    full = build_usable_prompt(template, record)
    omitted = build_usable_prompt(
        template, record, camera_evidence_included=False
    )

    assert template.count("{{EVIDENCE_RECORD_JSON}}") == 1
    assert '"camera_evidence":null' in omitted
    assert record["recipe_context"]["evidence_id"] in omitted
    assert record["camera_evidence"]["measurement_id"] in full
    assert record["camera_evidence"]["measurement_id"] not in omitted


@pytest.mark.parametrize("record_id", [f"r{index:03d}" for index in range(1, 7)])
@pytest.mark.parametrize("camera_included", [True, False])
def test_deterministic_control_uses_same_lenient_parser_and_exact_policy(record_id, camera_included):
    record = _record(record_id)
    raw = usable_reference_response(
        record, camera_evidence_included=camera_included
    )

    parsed = parse_usable_response(raw)
    structured = structure_usable_response(
        parsed, record, camera_evidence_included=camera_included
    )
    expected = reference_explanation(
        record, camera_evidence_included=camera_included
    )

    assert structured["policy_fields"]["appearance"] == expected["appearance"]
    assert structured["policy_fields"]["operator_action"] == expected["operator_action"]
    expected_abstention = (
        "abstain"
        if expected["status"] in {"insufficient_evidence", "invalid_input"}
        else "limited"
    )
    assert parsed["abstention"] == expected_abstention

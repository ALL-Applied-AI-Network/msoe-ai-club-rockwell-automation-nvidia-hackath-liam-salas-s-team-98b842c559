import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_nano_runner import execute_nano_attempt_plan  # noqa: E402
from fluidmix.explanation.vlm.cosmos_usable_output import (  # noqa: E402
    build_usable_attempt_plan,
    usable_reference_response,
)
from evaluation.scoring.cosmos_usable_scoring import (  # noqa: E402
    score_nano_usable_raw_results,
    score_usable_attempts,
)
from tests.evaluation.scoring.test_cosmos_nano_scoring import _provenance  # noqa: E402
from tests.fluidmix.explanation.vlm.test_cosmos_usable_preflight import valid_usable_config  # noqa: E402


MODES = [
    "image_evidence_recipe",
    "evidence_recipe",
    "image_recipe",
    "template_reference",
]


def _inputs():
    fixtures = json.loads(
        (ROOT / "data/cosmos_contract_fixtures_v1.json").read_text()
    )
    oracle = json.loads(
        (ROOT / "data/cosmos_contract_oracle_v1.json").read_text()
    )
    prompt = (ROOT / "data/cosmos_operator_prompt_usable_v1.txt").read_text()
    config = {
        "modes": MODES,
        "planned_attempts": 24,
        "model_attempts": 18,
        "template_control_attempts": 6,
        "required_abstention_attempts": 15,
        "required_abstention_model_attempts": 12,
        "required_abstention_control_attempts": 3,
        "image_invariance_pairs": 6,
    }
    plan = build_usable_attempt_plan(fixtures, config, prompt)
    return fixtures, oracle, config, plan


def _completion(response):
    tokens = [151643]
    return {
        "response_raw": response,
        "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "generated_suffix_token_ids": tokens,
        "generated_suffix_sha256": hashlib.sha256(b"[151643]").hexdigest(),
        "generated_token_count": 1,
        "terminal_token_id": 151643,
        "terminal_token_is_eos": True,
        "length_cap_reached": False,
        "truncated_without_eos": False,
    }


def _perfect_attempts():
    fixtures, _, _, plan = _inputs()
    by_id = {record["record_id"]: record for record in fixtures["fixtures"]}

    def infer(spec):
        record = by_id[spec["record_id"]]
        raw = usable_reference_response(
            record,
            camera_evidence_included=spec["camera_evidence_included"],
        )
        if spec["attempt_index"] == 0:
            lines = raw.splitlines()
            raw = (
                "Okay, let's see.\n"
                f"- **{lines[2].replace(':', '**:', 1)}\n"
                f"> {lines[1]}\n"
                f"### {lines[0]}\n"
            )
        return _completion(raw)

    return execute_nano_attempt_plan(plan, infer=infer)


def test_attempt_plan_is_oracle_blind_and_uses_usable_prompt_and_text_control():
    fixtures, _, _, plan = _inputs()

    assert len(plan) == 24
    assert [(row["record_id"], row["mode"]) for row in plan[:4]] == [
        ("r001", "image_evidence_recipe"),
        ("r001", "evidence_recipe"),
        ("r001", "image_recipe"),
        ("r001", "template_reference"),
    ]
    assert "oracle" not in json.dumps(plan).lower()
    by_mode = {row["mode"]: row for row in plan[:4]}
    assert by_mode["image_evidence_recipe"]["prompt"] == by_mode["evidence_recipe"]["prompt"]
    assert '"camera_evidence":null' in by_mode["image_recipe"]["prompt"]
    assert by_mode["template_reference"]["prompt"] is None
    assert by_mode["template_reference"]["template_response"].startswith(
        "Evidence meaning:"
    )


def test_perfect_lenient_matrix_keeps_raw_schema_and_controls_separate():
    fixtures, oracle, config, _ = _inputs()

    score = score_usable_attempts(_perfect_attempts(), fixtures, oracle, config)

    assert score["raw_strict_schema"] == {"passed": 0, "total": 24}
    assert score["raw_strict_schema_model"] == {"passed": 0, "total": 18}
    assert score["lenient_parse"] == {"passed": 24, "total": 24}
    assert score["lenient_parse_model"] == {"passed": 18, "total": 18}
    assert score["lenient_parse_control"] == {"passed": 6, "total": 6}
    assert score["model_content_policy"] == {"passed": 18, "total": 18}
    assert score["template_control"] == {"passed": 6, "total": 6}
    assert score["required_abstention"] == {"passed": 15, "total": 15}
    assert score["required_abstention_model"] == {"passed": 12, "total": 12}
    assert score["required_abstention_control"] == {"passed": 3, "total": 3}
    assert score["image_invariance"] == {"passed": 6, "total": 6}


def test_marker_only_model_artifacts_count_as_parseable_but_fail_content_policy():
    fixtures, oracle, config, _ = _inputs()
    attempts = _perfect_attempts()
    marker_only = (
        "Evidence meaning: DIFFERENCE\n"
        "Uncertainty: LIMITED\n"
        "Suggested action: MANUAL_REVIEW"
    )
    for row in attempts:
        if row["mode"] != "template_reference":
            row["response_raw"] = marker_only
            row["response_utf8_sha256"] = hashlib.sha256(
                marker_only.encode()
            ).hexdigest()

    score = score_usable_attempts(attempts, fixtures, oracle, config)

    assert score["lenient_parse_model"] == {"passed": 18, "total": 18}
    assert score["model_content_policy"] == {"passed": 0, "total": 18}
    assert all(
        set(failure["categories"]) >= {"required_limits", "content_policy"}
        for failure in score["failures"]
        if failure["mode"] != "template_reference"
    )


def test_lenient_parse_does_not_excuse_wrong_content_numbers_ids_or_control_advice():
    fixtures, oracle, config, _ = _inputs()
    attempts = _perfect_attempts()
    row = attempts[0]
    row["response_raw"] = (
        "Explanation: NO_DIFFERENCE | comparison:r999 is below 12.34 delta_e76.\n"
        "Uncertainty: LIMITED | Cause confirmed; defect confirmed; next-vial effect confirmed.\n"
        "Suggested action: MANUAL_REVIEW | Increase tank flow and change the recipe."
    )
    row["response_utf8_sha256"] = hashlib.sha256(
        row["response_raw"].encode()
    ).hexdigest()

    score = score_usable_attempts(attempts, fixtures, oracle, config)

    assert score["lenient_parse_model"] == {"passed": 18, "total": 18}
    assert score["model_content_policy"] == {"passed": 17, "total": 18}
    failure = next(
        item
        for item in score["failures"]
        if item["record_id"] == "r001" and item["mode"] == "image_evidence_recipe"
    )
    assert set(failure["categories"]) >= {
        "evidence_meaning",
        "evidence_ids",
        "numeric_fidelity",
        "required_limits",
        "unsupported_claim",
        "operator_control_advice",
        "content_policy",
    }


def test_malformed_model_text_stays_in_denominator_without_recovery():
    fixtures, oracle, config, _ = _inputs()
    attempts = _perfect_attempts()
    row = attempts[0]
    row["response_raw"] = "I think it looks different."
    row["response_utf8_sha256"] = hashlib.sha256(
        row["response_raw"].encode()
    ).hexdigest()

    score = score_usable_attempts(attempts, fixtures, oracle, config)

    assert score["lenient_parse_model"] == {"passed": 17, "total": 18}
    assert score["model_content_policy"] == {"passed": 17, "total": 18}
    assert score["raw_strict_schema_model"] == {"passed": 0, "total": 18}


def test_safe_limit_phrases_cannot_mask_a_second_unsupported_claim():
    fixtures, oracle, config, _ = _inputs()
    attempts = _perfect_attempts()
    row = attempts[0]
    row["response_raw"] = (
        "Explanation: DIFFERENCE | The comparison reports a difference caused by residue.\n"
        "Uncertainty: LIMITED | Cause unverified; defect unverified; next-vial effect unverified; defect confirmed.\n"
        "Suggested action: MANUAL_REVIEW | Have a person review the evidence."
    )
    row["response_utf8_sha256"] = hashlib.sha256(
        row["response_raw"].encode()
    ).hexdigest()

    score = score_usable_attempts(attempts, fixtures, oracle, config)

    failure = next(
        item
        for item in score["failures"]
        if item["record_id"] == "r001" and item["mode"] == "image_evidence_recipe"
    )
    assert "required_limits" not in failure["categories"]
    assert "unsupported_claim" in failure["categories"]
    assert "content_policy" in failure["categories"]


def test_complete_raw_usable_artifact_passes_provenance_and_matrix_boundary():
    fixtures, oracle, _, _ = _inputs()
    config = valid_usable_config()
    raw = {
        "schema_version": "cosmos3_nano_usable_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": _provenance(config),
        "attempts": _perfect_attempts(),
    }

    score = score_nano_usable_raw_results(raw, config, fixtures, oracle)

    assert score["matrix"]["model"] == {
        "planned": 18,
        "completed": 18,
        "errors": 0,
    }
    assert score["lenient_parse_model"] == {"passed": 18, "total": 18}
    assert score["raw_strict_schema_model"] == {"passed": 0, "total": 18}

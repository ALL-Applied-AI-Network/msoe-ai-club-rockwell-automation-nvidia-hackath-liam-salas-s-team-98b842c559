"""Separate format, parse, and content metrics for V3 Cosmos responses."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.skip(
    reason="legacy_diagnostic_only: active V3 scoring path is deactivated"
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_f2_varied_explanation import _artifact  # noqa: E402


PROMPT = (ROOT / "data/cosmos_f2_varied_prompt_v1.txt").read_text(encoding="utf-8")


def _completion(raw):
    return {
        "response_raw": raw,
        "response_utf8_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "generated_suffix_token_ids": [1, 2],
        "generated_suffix_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
        "generated_token_count": 2,
        "terminal_token_id": 2,
        "terminal_token_is_eos": False,
        "length_cap_reached": False,
        "truncated_without_eos": False,
    }


def _attempts(*, valid_model: bool):
    from fluidmix.f2_varied_explanation import (
        build_varied_attempt_plan,
        varied_reference_response,
    )
    from fluidmix.v3_cosmos_runtime import execute_v3_attempt_plan

    decisions = _artifact()
    plan = build_varied_attempt_plan(decisions, PROMPT)
    by_id = {row["decision_id"]: row for row in decisions["decision_records"]}

    def infer(specs):
        return [
            _completion(
                varied_reference_response(by_id[spec["decision_id"]])
                if valid_model
                else "Okay, I cannot format this."
            )
            for spec in specs
        ]

    return execute_v3_attempt_plan(plan, infer_batch=infer, batch_size=6)


def test_perfect_three_line_outputs_do_not_count_as_raw_strict_json():
    from fluidmix.v3_cosmos_scoring import score_v3_cosmos_attempts

    score = score_v3_cosmos_attempts(_attempts(valid_model=True), _artifact(), PROMPT)

    assert score["matrix"] == {
        "planned": 72,
        "completed": 72,
        "errors": 0,
        "model_completed": 54,
        "control_completed": 18,
    }
    assert score["metrics"]["raw_strict_schema"] == {"passed": 0, "total": 72}
    assert score["metrics"]["lenient_parse"] == {"passed": 72, "total": 72}
    assert score["metrics"]["lenient_parse_model"] == {"passed": 54, "total": 54}
    assert score["metrics"]["strict_content_policy"] == {"passed": 72, "total": 72}
    assert score["metrics"]["required_abstention"] == {"passed": 72, "total": 72}
    assert len(score["structured_records"]) == 72
    assert score["failures"] == []


def test_malformed_model_text_remains_completed_but_fails_parse_and_content():
    from fluidmix.v3_cosmos_scoring import score_v3_cosmos_attempts

    score = score_v3_cosmos_attempts(_attempts(valid_model=False), _artifact(), PROMPT)

    assert score["matrix"]["completed"] == 72
    assert score["metrics"]["raw_strict_schema_model"] == {"passed": 0, "total": 54}
    assert score["metrics"]["lenient_parse_model"] == {"passed": 0, "total": 54}
    assert score["metrics"]["strict_content_policy_model"] == {"passed": 0, "total": 54}
    assert score["metrics"]["lenient_parse_control"] == {"passed": 18, "total": 18}
    assert score["metrics"]["template_control_content_policy"] == {
        "passed": 18,
        "total": 18,
    }
    assert score["metrics"]["required_abstention_model"] == {
        "passed": 0,
        "total": 54,
    }
    assert len(score["failures"]) == 54

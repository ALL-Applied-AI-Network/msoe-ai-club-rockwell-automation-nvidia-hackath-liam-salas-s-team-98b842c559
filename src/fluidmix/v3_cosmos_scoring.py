"""Scoring for the frozen Varied V3 Cosmos explanation matrix."""

from __future__ import annotations

import hashlib

from fluidmix.explanation_contract import ContractError, parse_explanation_shape
from fluidmix.f2_varied_explanation import (
    build_varied_attempt_plan,
    parse_varied_response,
    raw_varied_format_passes,
    structure_varied_response,
    varied_content_failures,
)
from fluidmix.v3_cosmos_runtime import _validate_plan


_CONDITIONS = (
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
    "deterministic_template",
)
_BASE_FIELDS = (
    "attempt_index",
    "attempt_id",
    "decision_id",
    "run_id",
    "split_role",
    "status",
    "condition_id",
    "kind",
    "prompt_sha256",
    "evidence_record_sha256",
    "image_ids",
    "image_captions",
    "image_count",
    "irrelevant_source_decision_ids",
    "irrelevant_source_run_ids",
    "irrelevant_source_statuses",
    "requires_multi_image_smoke",
)


def _metric(passed: int, total: int) -> dict[str, int]:
    return {"passed": passed, "total": total}


def _validated_attempts(attempts, decision_artifact, prompt_template):
    expected = _validate_plan(
        build_varied_attempt_plan(decision_artifact, prompt_template)
    )
    if not isinstance(attempts, list) or len(attempts) != 72:
        raise ContractError("V3 scoring requires all 72 outcomes")
    decisions_raw = decision_artifact.get("decision_records")
    if not isinstance(decisions_raw, list):
        raise ContractError("V3 scoring decisions are missing")
    decisions = {
        row["decision_id"]: row
        for row in decisions_raw
        if isinstance(row, dict) and isinstance(row.get("decision_id"), str)
    }
    if len(decisions) != 18:
        raise ContractError("V3 scoring requires 18 unique decisions")
    for row, spec in zip(attempts, expected, strict=True):
        if not isinstance(row, dict) or any(
            row.get(field) != spec.get(field) for field in _BASE_FIELDS
        ):
            raise ContractError("V3 outcome does not match the frozen plan")
        status = row.get("attempt_status")
        if status not in {"completed", "error"}:
            raise ContractError("V3 outcome status is invalid")
        raw = row.get("response_raw")
        if status == "completed":
            if (
                not isinstance(raw, str)
                or hashlib.sha256(raw.encode("utf-8")).hexdigest()
                != row.get("response_utf8_sha256")
            ):
                raise ContractError("V3 raw-response hash mismatch")
        elif raw is not None:
            raise ContractError("V3 error outcome retained a raw response")
    return attempts, decisions


def score_v3_cosmos_attempts(attempts, decision_artifact, prompt_template):
    """Score raw format, lenient parsing, and strict content independently."""

    rows, decisions = _validated_attempts(
        attempts, decision_artifact, prompt_template
    )
    counters = {
        name: 0
        for name in (
            "raw",
            "raw_model",
            "exact",
            "parse",
            "parse_model",
            "parse_control",
            "content",
            "content_model",
            "content_control",
            "abstain",
            "abstain_model",
            "abstain_control",
        )
    }
    policy_names = (
        "numeric_fidelity",
        "evidence_id_faithfulness",
        "unit_fidelity",
        "required_limits",
        "unsupported_claim",
        "operator_control_advice",
    )
    policy_pass = {name: 0 for name in policy_names}
    condition = {
        name: {"completed": 0, "lenient_parse": 0, "strict_content_policy": 0}
        for name in _CONDITIONS
    }
    failures: list[dict[str, object]] = []
    structured: list[dict[str, object]] = []
    for row in rows:
        is_model = row["kind"] == "model"
        decision = decisions[row["decision_id"]]
        categories: list[str] = []
        parsed = None
        raw = row.get("response_raw")
        if row["attempt_status"] == "completed":
            condition[row["condition_id"]]["completed"] += 1
            try:
                parse_explanation_shape(raw)
                counters["raw"] += 1
                if is_model:
                    counters["raw_model"] += 1
            except ContractError:
                pass
            if raw_varied_format_passes(raw):
                counters["exact"] += 1
            try:
                parsed = parse_varied_response(raw)
                counters["parse"] += 1
                counters["parse_model" if is_model else "parse_control"] += 1
                condition[row["condition_id"]]["lenient_parse"] += 1
            except ContractError:
                categories.extend(("lenient_parse", "strict_content_policy"))
        else:
            categories.extend(("inference_error", "lenient_parse", "strict_content_policy"))
        content_failures: list[str] = []
        if parsed is not None:
            content_failures = varied_content_failures(raw, parsed, decision)
            categories.extend(content_failures)
            for name in policy_names:
                if name not in content_failures:
                    policy_pass[name] += 1
            if parsed["abstention"] == "abstain":
                counters["abstain"] += 1
                counters[
                    "abstain_model" if is_model else "abstain_control"
                ] += 1
            if not content_failures:
                counters["content"] += 1
                counters["content_model" if is_model else "content_control"] += 1
                condition[row["condition_id"]]["strict_content_policy"] += 1
                structured.append(
                    {
                        "attempt_id": row["attempt_id"],
                        "decision_id": row["decision_id"],
                        "condition_id": row["condition_id"],
                        "kind": row["kind"],
                        "record": structure_varied_response(raw, parsed, decision),
                    }
                )
            else:
                categories.append("strict_content_policy")
        if categories:
            failures.append(
                {
                    "attempt_id": row["attempt_id"],
                    "categories": sorted(set(categories)),
                }
            )

    completed = sum(row["attempt_status"] == "completed" for row in rows)
    model_completed = sum(
        row["kind"] == "model" and row["attempt_status"] == "completed"
        for row in rows
    )
    control_completed = sum(
        row["kind"] == "control" and row["attempt_status"] == "completed"
        for row in rows
    )
    return {
        "schema_version": "f2_varied_cosmos_score.v1",
        "matrix": {
            "planned": 72,
            "completed": completed,
            "errors": 72 - completed,
            "model_completed": model_completed,
            "control_completed": control_completed,
        },
        "metrics": {
            "raw_strict_schema": _metric(counters["raw"], 72),
            "raw_strict_schema_model": _metric(counters["raw_model"], 54),
            "exact_three_line_format": _metric(counters["exact"], 72),
            "lenient_parse": _metric(counters["parse"], 72),
            "lenient_parse_model": _metric(counters["parse_model"], 54),
            "lenient_parse_control": _metric(counters["parse_control"], 18),
            "strict_content_policy": _metric(counters["content"], 72),
            "strict_content_policy_model": _metric(counters["content_model"], 54),
            "template_control_content_policy": _metric(
                counters["content_control"], 18
            ),
            "required_abstention": _metric(counters["abstain"], 72),
            "required_abstention_model": _metric(counters["abstain_model"], 54),
            "required_abstention_control": _metric(
                counters["abstain_control"], 18
            ),
            **{
                ("required_limits_present" if name == "required_limits" else
                 "unsupported_claims_absent" if name == "unsupported_claim" else
                 "operator_control_advice_absent" if name == "operator_control_advice" else name): _metric(passed, 72)
                for name, passed in policy_pass.items()
            },
        },
        "condition_metrics": condition,
        "failures": failures,
        "structured_records": structured,
        "claim_boundary": (
            "Synthetic explanation quality only. Raw strict format, lenient parsing, "
            "and content policy are separate outcomes; no physics or sim-to-real claim."
        ),
    }

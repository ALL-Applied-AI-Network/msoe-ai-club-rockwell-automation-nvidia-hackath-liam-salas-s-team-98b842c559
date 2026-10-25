"""Scoring for the frozen stage/vial Cosmos explanation experiment."""

from __future__ import annotations

import hashlib

from fluidmix.explanation_contract import ContractError, parse_explanation_shape
from fluidmix.stage_cosmos_explanation import (
    build_stage_attempt_plan,
    parse_stage_response,
    raw_stage_format_passes,
    stage_content_failures,
    structure_stage_response,
)


def _metric(passed: int, total: int) -> dict[str, int]:
    return {"passed": passed, "total": total}


def _validate_attempts(
    attempts: object, measured_evidence: object, prompt_template: object
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    expected = build_stage_attempt_plan(measured_evidence, prompt_template)
    if not isinstance(attempts, list) or len(attempts) != 48:
        raise ContractError("stage scoring requires all 48 attempt outcomes")
    if any(not isinstance(row, dict) for row in attempts):
        raise ContractError("stage scoring attempt outcomes must be objects")
    decisions_raw = measured_evidence.get("decision_records")
    decisions = {
        row["decision_id"]: row
        for row in decisions_raw
        if isinstance(row, dict) and isinstance(row.get("decision_id"), str)
    }
    if len(decisions) != 12:
        raise ContractError("stage scoring requires 12 unique decisions")
    base_fields = (
        "attempt_index",
        "attempt_id",
        "decision_id",
        "partition",
        "condition_id",
        "kind",
        "prompt_sha256",
        "evidence_record_sha256",
        "image_ids",
        "image_count",
    )
    for row, spec in zip(attempts, expected, strict=True):
        if any(row.get(field) != spec.get(field) for field in base_fields):
            raise ContractError("stage attempt outcome does not match the frozen plan")
        status = row.get("attempt_status")
        if status not in {"completed", "error"}:
            raise ContractError("stage attempt status is invalid")
        if status == "completed":
            raw = row.get("response_raw")
            if not isinstance(raw, str) or hashlib.sha256(
                raw.encode("utf-8")
            ).hexdigest() != row.get("response_utf8_sha256"):
                raise ContractError("stage raw response hash mismatch")
        elif row.get("response_raw") is not None:
            raise ContractError("failed stage attempt retained a raw response")
    return attempts, expected, decisions


def score_stage_attempts(attempts, measured_evidence, prompt_template):
    """Score raw formats, recovery, content, abstention, and image controls.

    The historical strict-JSON metric is a continuity diagnostic only. A response
    recovered by the lenient text parser never increments that raw metric.
    """

    rows, _, decisions = _validate_attempts(
        attempts, measured_evidence, prompt_template
    )
    completed = sum(row["attempt_status"] == "completed" for row in rows)
    model_rows = [row for row in rows if row["kind"] == "model"]
    control_rows = [row for row in rows if row["kind"] == "control"]

    legacy = legacy_model = 0
    exact = exact_model = exact_control = 0
    parsed_total = parsed_model = parsed_control = 0
    content = content_model = content_control = 0
    abstain_total = abstain_passed = 0
    abstain_model_total = abstain_model_passed = 0
    abstain_control_total = abstain_control_passed = 0
    numeric_faithful = evidence_id_faithful = unit_faithful = 0
    required_limits_present = unsupported_absent = operator_absent = 0
    parsed_by_pair: dict[tuple[str, str], object] = {}
    failures: list[dict[str, object]] = []
    structured_records: list[dict[str, object]] = []

    for row in rows:
        is_model = row["kind"] == "model"
        is_control = not is_model
        decision = decisions[row["decision_id"]]
        needs_abstention = decision["disposition"] == "insufficient_evidence"
        if needs_abstention:
            abstain_total += 1
            if is_model:
                abstain_model_total += 1
            else:
                abstain_control_total += 1
        categories: list[str] = []
        parsed = None
        raw = row.get("response_raw")
        if row["attempt_status"] != "completed":
            categories.extend(("inference_error", "lenient_parse", "strict_content_policy"))
        else:
            try:
                parse_explanation_shape(raw)
                legacy += 1
                if is_model:
                    legacy_model += 1
            except ContractError:
                pass
            if raw_stage_format_passes(raw):
                exact += 1
                if is_model:
                    exact_model += 1
                else:
                    exact_control += 1
            try:
                parsed = parse_stage_response(raw)
                parsed_total += 1
                if is_model:
                    parsed_model += 1
                else:
                    parsed_control += 1
            except ContractError:
                categories.extend(("lenient_parse", "strict_content_policy"))

        content_failures: list[str] | None = None
        if parsed is not None:
            content_failures = stage_content_failures(raw, parsed, decision)
            categories.extend(content_failures)
            if not content_failures:
                content += 1
                if is_model:
                    content_model += 1
                else:
                    content_control += 1
                structured_records.append(
                    {
                        "attempt_id": row["attempt_id"],
                        "decision_id": row["decision_id"],
                        "condition_id": row["condition_id"],
                        "kind": row["kind"],
                        "record": structure_stage_response(raw, parsed, decision),
                    }
                )
            else:
                categories.append("strict_content_policy")
            if "numeric_fidelity" not in content_failures:
                numeric_faithful += 1
            if "evidence_id_faithfulness" not in content_failures:
                evidence_id_faithful += 1
            if "unit_fidelity" not in content_failures:
                unit_faithful += 1
            if "required_limits" not in content_failures:
                required_limits_present += 1
            if "unsupported_claim" not in content_failures:
                unsupported_absent += 1
            if "operator_control_advice" not in content_failures:
                operator_absent += 1
            if needs_abstention and parsed["abstention"] == "abstain":
                abstain_passed += 1
                if is_model:
                    abstain_model_passed += 1
                else:
                    abstain_control_passed += 1
        parsed_by_pair[(row["decision_id"], row["condition_id"])] = parsed
        if categories:
            failures.append(
                {
                    "attempt_id": row["attempt_id"],
                    "categories": sorted(set(categories)),
                }
            )

    invariant = 0
    for decision_id in decisions:
        relevant = parsed_by_pair[(decision_id, "ordered_stage_visual_pairs")]
        irrelevant = parsed_by_pair[(decision_id, "irrelevant_shuffled_images")]
        if relevant is not None and relevant == irrelevant:
            invariant += 1

    return {
        "schema_version": "stage_cosmos_score.v1",
        "matrix": {
            "planned": 48,
            "completed": completed,
            "errors": 48 - completed,
            "model": {
                "planned": 36,
                "completed": sum(
                    row["attempt_status"] == "completed" for row in model_rows
                ),
                "errors": sum(row["attempt_status"] == "error" for row in model_rows),
            },
            "control": {
                "planned": 12,
                "completed": sum(
                    row["attempt_status"] == "completed" for row in control_rows
                ),
                "errors": sum(
                    row["attempt_status"] == "error" for row in control_rows
                ),
            },
        },
        "metrics": {
            "legacy_raw_strict_schema": _metric(legacy, 48),
            "legacy_raw_strict_schema_model": _metric(legacy_model, 36),
            "exact_three_line_format": _metric(exact, 48),
            "exact_three_line_format_model": _metric(exact_model, 36),
            "exact_three_line_format_control": _metric(exact_control, 12),
            "lenient_parse": _metric(parsed_total, 48),
            "lenient_parse_model": _metric(parsed_model, 36),
            "lenient_parse_control": _metric(parsed_control, 12),
            "strict_content_policy": _metric(content, 48),
            "strict_content_policy_model": _metric(content_model, 36),
            "template_control_content_policy": _metric(content_control, 12),
            "required_abstention": _metric(abstain_passed, abstain_total),
            "required_abstention_model": _metric(
                abstain_model_passed, abstain_model_total
            ),
            "required_abstention_control": _metric(
                abstain_control_passed, abstain_control_total
            ),
            "numeric_fidelity": _metric(numeric_faithful, 48),
            "evidence_id_faithfulness": _metric(evidence_id_faithful, 48),
            "unit_fidelity": _metric(unit_faithful, 48),
            "required_limits_present": _metric(required_limits_present, 48),
            "unsupported_claims_absent": _metric(unsupported_absent, 48),
            "operator_control_advice_absent": _metric(operator_absent, 48),
            "relevant_irrelevant_exact_invariance": _metric(invariant, 12),
        },
        "failures": failures,
        "structured_records": structured_records,
    }

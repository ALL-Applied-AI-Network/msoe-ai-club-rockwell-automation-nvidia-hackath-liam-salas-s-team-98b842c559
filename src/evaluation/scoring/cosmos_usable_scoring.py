"""Scoring for the frozen Cosmos usable-output text treatment."""

import hashlib
import re

from fluidmix.explanation.vlm.cosmos_usable_output import (
    parse_usable_response,
    structure_usable_response,
)
from evaluation.scoring.cosmos_nano_scoring import (
    _validate_attempt,
    _validate_provenance,
)
from fluidmix.explanation.vlm.cosmos_nano_usable_contract import validate_nano_usable_config
from fluidmix.explanation.template.explanation_contract import (
    ContractError,
    parse_explanation_shape,
    reference_explanation,
    validate_fixture_set,
)


_ID_PATTERN = re.compile(r"\b(?:measurement|comparison|recipe)\s*:", re.IGNORECASE)
_CONTROL_ADVICE = re.compile(
    r"\b(?:adjust|calibrate|change|close|decrease|disable|discard|dispense|divert|"
    r"drain|enable|flush|hold|increase|mix|open|operate|override|reject|rerun|set|"
    r"shut|start|stop)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_CLAIM = re.compile(
    r"(?:"
    r"\b(?:cause|defect|next[-\s]?vial(?:\s+effect)?)\b.{0,50}"
    r"\b(?:caused|confirmed|detected|expected|identified|known|likely|present|probable|will)\b"
    r"|"
    r"\b(?:caused|confirmed|detected|expected|identified|known|likely|present|probable)\b"
    r".{0,50}\b(?:cause|defect|next[-\s]?vial(?:\s+effect)?)\b"
    r"|\b(?:carryover|residue)\b"
    r")",
    re.IGNORECASE,
)


def _metric(passed, total):
    return {"passed": passed, "total": total}


def _required_limits_present(raw):
    patterns = (
        r"\bcause\b.{0,40}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
        r"\bdefect\b.{0,40}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
        r"\bnext[-\s]?vial(?:\s+effect)?\b.{0,40}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
    )
    return all(re.search(pattern, raw, re.IGNORECASE) is not None for pattern in patterns)


def _expected_abstention(target):
    return (
        "abstain"
        if target["status"] in {"insufficient_evidence", "invalid_input"}
        else "limited"
    )


def _validate_policy_oracle(fixture_set, oracle):
    expected = oracle.get("expected_by_record") if isinstance(oracle, dict) else None
    records = {record["record_id"]: record for record in fixture_set["fixtures"]}
    if not isinstance(expected, dict) or set(expected) != set(records):
        raise ContractError("usable scoring oracle record IDs do not match fixtures")
    for record_id, record in records.items():
        value = expected[record_id]
        if not isinstance(value, dict) or set(value) != {"full", "image_recipe"}:
            raise ContractError("usable scoring oracle has the wrong shape")
        if value["full"] != reference_explanation(record):
            raise ContractError("usable scoring oracle full policy mismatch")
        if value["image_recipe"] != reference_explanation(
            record, camera_evidence_included=False
        ):
            raise ContractError("usable scoring oracle omitted-camera policy mismatch")
    return records, expected


def score_usable_attempts(attempts, fixture_set, oracle, config):
    """Score raw schema separately from lenient parsing and strict content policy."""
    validate_fixture_set(fixture_set)
    fixed = {
        "modes": [
            "image_evidence_recipe",
            "evidence_recipe",
            "image_recipe",
            "template_reference",
        ],
        "planned_attempts": 24,
        "model_attempts": 18,
        "template_control_attempts": 6,
        "required_abstention_attempts": 15,
        "required_abstention_model_attempts": 12,
        "required_abstention_control_attempts": 3,
        "image_invariance_pairs": 6,
    }
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in fixed.items()):
        raise ContractError("usable scoring config does not preserve frozen denominators")
    records, expected = _validate_policy_oracle(fixture_set, oracle)
    pairs = [
        (record_id, mode)
        for record_id in sorted(records)
        for mode in fixed["modes"]
    ]
    if not isinstance(attempts, list) or len(attempts) != 24:
        raise ContractError("usable scoring requires all 24 attempts")
    observed_pairs = [(row.get("record_id"), row.get("mode")) for row in attempts]
    if observed_pairs != pairs:
        raise ContractError("usable attempt pairs are not fixture-major and complete")

    strict = strict_model = 0
    lenient = lenient_model = lenient_control = 0
    model_policy = template_policy = 0
    abstention_total = abstention_passed = 0
    abstention_model_total = abstention_model_passed = 0
    abstention_control_total = abstention_control_passed = 0
    failures = []
    structured_by_pair = {}
    for row in attempts:
        record_id = row["record_id"]
        mode = row["mode"]
        is_control = mode == "template_reference"
        categories = []
        raw = None
        if row.get("attempt_status") != "completed":
            categories.extend(("inference_error", "lenient_parse", "content_policy"))
        elif not isinstance(row.get("response_raw"), str):
            categories.extend(("raw_response", "lenient_parse", "content_policy"))
        else:
            raw = row["response_raw"]
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row.get(
                "response_utf8_sha256"
            ):
                raise ContractError("usable raw response hash mismatch")
            try:
                parse_explanation_shape(raw)
                strict += 1
                if not is_control:
                    strict_model += 1
            except ContractError:
                pass

        parsed = None
        structured = None
        if raw is not None:
            try:
                parsed = parse_usable_response(raw)
                structured = structure_usable_response(
                    parsed,
                    records[record_id],
                    camera_evidence_included=mode != "image_recipe",
                )
                lenient += 1
                if is_control:
                    lenient_control += 1
                else:
                    lenient_model += 1
            except ContractError:
                categories.extend(("lenient_parse", "content_policy"))
        structured_by_pair[(record_id, mode)] = structured

        target_key = "image_recipe" if mode == "image_recipe" else "full"
        target = expected[record_id][target_key]
        needs_abstention = _expected_abstention(target) == "abstain"
        if needs_abstention:
            abstention_total += 1
            if is_control:
                abstention_control_total += 1
            else:
                abstention_model_total += 1
        if parsed is not None and structured is not None:
            checks = {
                "evidence_meaning": parsed["evidence_meaning"] == target["appearance"],
                "abstention": parsed["abstention"] == _expected_abstention(target),
                "suggested_action": parsed["suggested_action"]
                == target["operator_action"],
                "evidence_ids": (
                    structured["evidence_ids"] == target["evidence_ids"]
                    and _ID_PATTERN.search(raw) is None
                ),
                "numeric_fidelity": (
                    structured["quantitative_evidence"]
                    == target["quantitative_evidence"]
                    and re.search(r"[0-9]", raw) is None
                ),
                "required_limits": _required_limits_present(raw),
                "unsupported_claim": _UNSUPPORTED_CLAIM.search(raw) is None,
                "operator_control_advice": _CONTROL_ADVICE.search(raw) is None,
            }
            for name, passed in checks.items():
                if not passed:
                    categories.append(name)
            content_passed = all(checks.values())
            if not content_passed:
                categories.append("content_policy")
            if needs_abstention and checks["abstention"]:
                abstention_passed += 1
                if is_control:
                    abstention_control_passed += 1
                else:
                    abstention_model_passed += 1
            if content_passed:
                if is_control:
                    template_policy += 1
                else:
                    model_policy += 1
        if categories:
            failures.append(
                {
                    "record_id": record_id,
                    "mode": mode,
                    "categories": sorted(set(categories)),
                }
            )

    invariant = 0
    for record_id in sorted(records):
        image = structured_by_pair[(record_id, "image_evidence_recipe")]
        text = structured_by_pair[(record_id, "evidence_recipe")]
        if image is not None and text is not None and image["presentation"] == text["presentation"]:
            invariant += 1
    return {
        "raw_strict_schema": _metric(strict, 24),
        "raw_strict_schema_model": _metric(strict_model, 18),
        "lenient_parse": _metric(lenient, 24),
        "lenient_parse_model": _metric(lenient_model, 18),
        "lenient_parse_control": _metric(lenient_control, 6),
        "model_content_policy": _metric(model_policy, 18),
        "template_control": _metric(template_policy, 6),
        "required_abstention": _metric(abstention_passed, abstention_total),
        "required_abstention_model": _metric(
            abstention_model_passed, abstention_model_total
        ),
        "required_abstention_control": _metric(
            abstention_control_passed, abstention_control_total
        ),
        "image_invariance": _metric(invariant, 6),
        "failures": failures,
    }


def score_nano_usable_raw_results(raw_results, config, fixture_set, oracle):
    """Validate one immutable usable raw artifact and score fixed denominators."""
    validate_nano_usable_config(config)
    if not isinstance(raw_results, dict) or set(raw_results) != {
        "schema_version",
        "experiment_id",
        "provenance",
        "attempts",
    }:
        raise ContractError("Nano usable raw results have the wrong keys")
    if raw_results["schema_version"] != "cosmos3_nano_usable_raw_results.v1":
        raise ContractError("unsupported Nano usable raw-results schema")
    if raw_results["experiment_id"] != config["experiment_id"]:
        raise ContractError("Nano usable raw-results experiment mismatch")
    _validate_provenance(raw_results["provenance"], config)
    attempts = raw_results["attempts"]
    pairs = [
        (record["record_id"], mode)
        for record in fixture_set.get("fixtures", [])
        for mode in config["modes"]
    ]
    if not isinstance(attempts, list) or len(attempts) != 24 or len(pairs) != 24:
        raise ContractError("incomplete Nano usable raw artifact is not scoreable")
    for index, (row, pair) in enumerate(zip(attempts, pairs, strict=True)):
        _validate_attempt(row, (pair, index), config)
    model_rows = [row for row in attempts if row["mode"] != "template_reference"]
    control_rows = [row for row in attempts if row["mode"] == "template_reference"]
    if raw_results["provenance"]["model_load_error"] is not None and any(
        row["attempt_status"] != "error" for row in model_rows
    ):
        raise ContractError("a model-load failure requires all model attempts to error")
    score = {
        "schema_version": "cosmos3_nano_usable_score.v1",
        "experiment_id": config["experiment_id"],
        "matrix": {
            "planned": 24,
            "attempted": 24,
            "completed": sum(row["attempt_status"] == "completed" for row in attempts),
            "errors": sum(row["attempt_status"] == "error" for row in attempts),
            "model": {
                "planned": 18,
                "completed": sum(
                    row["attempt_status"] == "completed" for row in model_rows
                ),
                "errors": sum(row["attempt_status"] == "error" for row in model_rows),
            },
            "control": {
                "planned": 6,
                "completed": sum(
                    row["attempt_status"] == "completed" for row in control_rows
                ),
                "errors": sum(
                    row["attempt_status"] == "error" for row in control_rows
                ),
            },
        },
        "per_mode": {
            mode: {
                "planned": 6,
                "completed": sum(
                    row["attempt_status"] == "completed"
                    for row in attempts
                    if row["mode"] == mode
                ),
                "errors": sum(
                    row["attempt_status"] == "error"
                    for row in attempts
                    if row["mode"] == mode
                ),
            }
            for mode in config["modes"]
        },
        "generation_cap_reaches": {
            "reached": sum(row["length_cap_reached"] is True for row in model_rows),
            "total": 18,
        },
        "truncated_without_eos": {
            "count": sum(row["truncated_without_eos"] is True for row in model_rows),
            "total": 18,
        },
    }
    score.update(score_usable_attempts(attempts, fixture_set, oracle, config))
    return score

"""Strict no-repair scoring for the frozen explanation response matrix."""

from fluidmix.explanation_contract import ContractError, parse_explanation_shape


def _metric(passed, total):
    return {"passed": passed, "total": total}


def _is_abstention(parsed):
    return parsed is not None and (
        parsed["status"] in {"insufficient_evidence", "invalid_input"}
        and parsed["appearance"] == "unknown"
        and parsed["defect"] == "unknown"
        and parsed["cause"] == "unknown"
        and parsed["next_vial_effect"] == "unknown"
        and parsed["confidence"] == "not_assigned"
        and parsed["operator_action"] == "manual_review_required"
    )


def score_explanation_attempts(attempts, config, oracle):
    """Score exact raw strings against the sealed policy without extraction or repair."""
    expected = oracle["expected_by_record"]
    pairs = [
        (record_id, mode)
        for record_id in sorted(expected)
        for mode in config["modes"]
    ]
    by_pair = {(row["record_id"], row["mode"]): row for row in attempts}
    if len(by_pair) != len(pairs) or set(by_pair) != set(pairs):
        raise ContractError("attempt pairs do not match the frozen matrix")
    parsed_by_pair = {}
    counters = {
        "strict_schema": 0,
        "policy": 0,
        "evidence_ids": 0,
        "numeric_fidelity": 0,
        "advisory_action": 0,
        "unsupported_claim": 0,
    }
    model_strict = 0
    model_policy = 0
    failures = []
    for pair in pairs:
        record_id, mode = pair
        row = by_pair[pair]
        categories = []
        parsed = None
        if row["attempt_status"] == "error":
            categories.append("inference_error")
        else:
            try:
                parsed = parse_explanation_shape(row["response_raw"])
                counters["strict_schema"] += 1
                if mode != "template_reference":
                    model_strict += 1
            except ContractError:
                categories.append("strict_schema")
        parsed_by_pair[pair] = parsed
        expected_key = "image_recipe" if mode == "image_recipe" else "full"
        target = expected[record_id][expected_key]
        if parsed is None:
            categories.extend(
                [
                    "policy",
                    "evidence_ids",
                    "numeric_fidelity",
                    "advisory_action",
                    "unsupported_claim",
                ]
            )
        else:
            checks = {
                "policy": parsed == target,
                "evidence_ids": parsed["evidence_ids"] == target["evidence_ids"],
                "numeric_fidelity": (
                    parsed["quantitative_evidence"] == target["quantitative_evidence"]
                ),
                "advisory_action": (
                    parsed["operator_action"] == target["operator_action"]
                ),
                "unsupported_claim": (
                    parsed["defect"] == "unknown"
                    and parsed["cause"] == "unknown"
                    and parsed["next_vial_effect"] == "unknown"
                    and parsed["confidence"] == "not_assigned"
                ),
            }
            for name, passed in checks.items():
                if passed:
                    counters[name] += 1
                else:
                    categories.append(name)
            if checks["policy"] and mode != "template_reference":
                model_policy += 1
        if categories:
            failures.append(
                {
                    "record_id": record_id,
                    "mode": mode,
                    "categories": sorted(set(categories)),
                }
            )

    score = {
        name: _metric(passed, config["planned_attempts"])
        for name, passed in counters.items()
    }
    score["strict_schema_model"] = _metric(model_strict, config["model_attempts"])
    score["model_policy"] = _metric(model_policy, config["model_attempts"])

    model_abstention_pairs = {
        (record_id, "image_recipe") for record_id in expected
    } | {
        (record_id, mode)
        for record_id in ("r004", "r005", "r006")
        for mode in ("image_evidence_recipe", "evidence_recipe")
    }
    control_abstention_pairs = {
        (record_id, "template_reference")
        for record_id in ("r004", "r005", "r006")
    }
    model_abstention = sum(
        _is_abstention(parsed_by_pair[pair]) for pair in model_abstention_pairs
    )
    control_abstention = sum(
        _is_abstention(parsed_by_pair[pair]) for pair in control_abstention_pairs
    )
    score["required_abstention_model"] = _metric(
        model_abstention, config["required_abstention_model_attempts"]
    )
    score["required_abstention_control"] = _metric(
        control_abstention, config["required_abstention_control_attempts"]
    )
    score["required_abstention"] = _metric(
        model_abstention + control_abstention,
        config["required_abstention_attempts"],
    )

    prompt_pairs = 0
    invariant_pairs = 0
    for record_id in expected:
        image = by_pair[(record_id, "image_evidence_recipe")]
        text = by_pair[(record_id, "evidence_recipe")]
        prompt_pairs += image["prompt_sha256"] == text["prompt_sha256"]
        image_parsed = parsed_by_pair[(record_id, "image_evidence_recipe")]
        text_parsed = parsed_by_pair[(record_id, "evidence_recipe")]
        invariant_pairs += image_parsed is not None and image_parsed == text_parsed
    score["prompt_pair_integrity"] = _metric(
        prompt_pairs, config["image_invariance_pairs"]
    )
    score["image_invariance"] = _metric(
        invariant_pairs, config["image_invariance_pairs"]
    )
    template_passed = sum(
        parsed_by_pair[(record_id, "template_reference")] == expected[record_id]["full"]
        for record_id in expected
    )
    score["template_control"] = _metric(
        template_passed, config["template_control_attempts"]
    )
    score["failures"] = failures
    return score

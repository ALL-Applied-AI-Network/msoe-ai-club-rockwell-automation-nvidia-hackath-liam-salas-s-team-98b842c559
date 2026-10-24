"""Score frozen Cosmos explanation-contract attempts without repairing model output."""

import re

from fluidmix.explanation_contract import (
    ContractError,
    parse_explanation_shape,
    validate_eval_config,
)


_ATTEMPT_KEYS = {
    "attempt_index", "record_id", "mode", "attempt_status", "response_raw",
    "prompt_sha256", "image_attached", "elapsed_seconds", "generated_token_count",
    "error_type", "error",
}
_PROVENANCE_KEYS = {
    "freeze_tag", "head_commit", "git_clean", "fixture_set_sha256", "prompt_sha256",
    "image_sha256", "model_manifest_sha256", "runtime_lock_sha256", "model_revision",
    "seed", "do_sample", "max_new_tokens", "dtype", "python", "versions",
    "platform_machine", "device_name", "compute_capability", "cuda_version", "model_load_seconds",
    "model_load_error", "started_at_utc", "finished_at_utc",
}


def _exact_keys(value, expected, path):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ContractError(f"{path} has the wrong keys")


def _metric(passed, total):
    return {"passed": passed, "total": total}


def _attempt_shape(attempt, modes):
    _exact_keys(attempt, _ATTEMPT_KEYS, "attempt")
    if isinstance(attempt["attempt_index"], bool) or not isinstance(attempt["attempt_index"], int):
        raise ContractError("attempt_index must be an integer")
    if not isinstance(attempt["record_id"], str) or not re.fullmatch(r"r[0-9]{3}", attempt["record_id"]):
        raise ContractError("invalid attempt record_id")
    if attempt["mode"] not in modes:
        raise ContractError("unknown attempt mode")
    if attempt["attempt_status"] not in {"completed", "error"}:
        raise ContractError("invalid attempt status")
    if type(attempt["image_attached"]) is not bool:
        raise ContractError("image_attached must be bool")
    expected_image = attempt["mode"] in {"image_evidence_recipe", "image_recipe"}
    if attempt["image_attached"] != expected_image:
        raise ContractError("image attachment does not match mode")
    elapsed = attempt["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ContractError("elapsed_seconds must be nonnegative")
    if attempt["mode"] == "template_reference":
        if attempt["prompt_sha256"] is not None or attempt["generated_token_count"] is not None:
            raise ContractError("template attempt cannot have a model prompt or token count")
    else:
        if not isinstance(attempt["prompt_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", attempt["prompt_sha256"]
        ):
            raise ContractError("model attempt needs a prompt SHA-256")
        tokens = attempt["generated_token_count"]
        if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
            raise ContractError("generated_token_count must be a nonnegative integer or null")
    if attempt["attempt_status"] == "completed":
        if not isinstance(attempt["response_raw"], str):
            raise ContractError("completed attempt needs raw response text")
        if attempt["error_type"] is not None or attempt["error"] is not None:
            raise ContractError("completed attempt cannot carry an error")
    else:
        if attempt["response_raw"] is not None:
            raise ContractError("error attempt cannot carry response text")
        if not isinstance(attempt["error_type"], str) or not isinstance(attempt["error"], str):
            raise ContractError("error attempt needs error strings")


def score_raw_results(raw_results, config, oracle):
    """Return fixed-denominator contract metrics for one raw result matrix."""
    validate_eval_config(config)
    _exact_keys(raw_results, {"schema_version", "experiment_id", "provenance", "attempts"}, "raw")
    if raw_results["schema_version"] != "operator_contract_raw_results.v1":
        raise ContractError("unsupported raw-results schema")
    if raw_results["experiment_id"] != config["experiment_id"]:
        raise ContractError("raw-results experiment mismatch")
    provenance = raw_results["provenance"]
    _exact_keys(provenance, _PROVENANCE_KEYS, "provenance")
    for key in (
        "freeze_tag", "fixture_set_sha256", "prompt_sha256", "image_sha256",
        "model_manifest_sha256", "runtime_lock_sha256", "model_revision", "seed",
        "do_sample", "max_new_tokens", "dtype",
    ):
        if provenance[key] != config[key]:
            raise ContractError(f"provenance mismatch: {key}")
    if provenance["git_clean"] is not True:
        raise ContractError("raw results must come from a clean tree")
    if provenance["python"] != config["python_version"]:
        raise ContractError("provenance Python version mismatch")
    if provenance["platform_machine"] != config["platform_machine"]:
        raise ContractError("provenance platform-machine mismatch")
    versions = provenance["versions"]
    if not isinstance(versions, dict):
        raise ContractError("provenance versions must be an object")
    for package, expected in config["runtime_versions"].items():
        if versions.get(package) != expected:
            raise ContractError(f"provenance runtime version mismatch: {package}")
    if not isinstance(provenance["head_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", provenance["head_commit"]
    ):
        raise ContractError("invalid head commit")
    if oracle.get("schema_version") != "operator_contract_oracle.v1":
        raise ContractError("unsupported oracle schema")
    if oracle.get("fixture_set_sha256") != config["fixture_set_sha256"]:
        raise ContractError("oracle fixture hash mismatch")
    expected_by_record = oracle.get("expected_by_record")
    if not isinstance(expected_by_record, dict) or len(expected_by_record) != config["fixture_count"]:
        raise ContractError("oracle fixture count mismatch")

    modes = config["modes"]
    expected_pairs = [
        (record_id, mode)
        for record_id in sorted(expected_by_record)
        for mode in modes
    ]
    expected_index = {pair: index for index, pair in enumerate(expected_pairs)}
    attempts = raw_results["attempts"]
    if not isinstance(attempts, list):
        raise ContractError("attempts must be a list")
    by_pair = {}
    seen_indexes = set()
    for attempt in attempts:
        _attempt_shape(attempt, modes)
        pair = (attempt["record_id"], attempt["mode"])
        if pair not in expected_pairs or pair in by_pair:
            raise ContractError("unexpected or duplicate attempt pair")
        if attempt["attempt_index"] != expected_index[pair]:
            raise ContractError("attempt index does not match the frozen pair order")
        if attempt["attempt_index"] in seen_indexes:
            raise ContractError("duplicate attempt index")
        by_pair[pair] = attempt
        seen_indexes.add(attempt["attempt_index"])

    completed = sum(
        attempt["attempt_status"] == "completed" for attempt in attempts
    )
    score = {
        "schema_version": "operator_contract_score.v1",
        "experiment_id": config["experiment_id"],
        "matrix": {
            "planned": config["planned_attempts"],
            "attempted": len(attempts),
            "completed": completed,
            "errors": len(attempts) - completed,
            "complete": len(by_pair) == config["planned_attempts"],
        },
        "per_mode": {},
        "failures": [],
    }
    for mode in modes:
        mode_attempts = [item for item in attempts if item["mode"] == mode]
        mode_completed = sum(item["attempt_status"] == "completed" for item in mode_attempts)
        score["per_mode"][mode] = {
            "planned": config["fixture_count"],
            "attempted": len(mode_attempts),
            "completed": mode_completed,
            "errors": len(mode_attempts) - mode_completed,
        }

    counters = {
        "strict_schema": 0,
        "policy": 0,
        "evidence_ids": 0,
        "numeric_fidelity": 0,
        "advisory_action": 0,
        "unsupported_claim": 0,
    }
    parsed_by_pair = {}
    for pair in expected_pairs:
        record_id, mode = pair
        attempt = by_pair.get(pair)
        categories = []
        parsed = None
        if attempt is None:
            categories.append("missing_attempt")
        elif attempt["attempt_status"] == "error":
            categories.append("inference_error")
        else:
            try:
                parsed = parse_explanation_shape(attempt["response_raw"])
                counters["strict_schema"] += 1
            except ContractError:
                categories.append("strict_schema")
        parsed_by_pair[pair] = parsed
        expected_key = "image_recipe" if mode == "image_recipe" else "full"
        expected = expected_by_record[record_id][expected_key]
        if parsed is not None:
            if parsed == expected:
                counters["policy"] += 1
            else:
                categories.append("policy")
            if parsed["evidence_ids"] == expected["evidence_ids"]:
                counters["evidence_ids"] += 1
            else:
                categories.append("evidence_ids")
            if parsed["quantitative_evidence"] == expected["quantitative_evidence"]:
                counters["numeric_fidelity"] += 1
            else:
                categories.append("numeric_fidelity")
            if parsed["operator_action"] == expected["operator_action"]:
                counters["advisory_action"] += 1
            else:
                categories.append("advisory_action")
            if (
                parsed["defect"] == "unknown"
                and parsed["cause"] == "unknown"
                and parsed["next_vial_effect"] == "unknown"
                and parsed["confidence"] == "not_assigned"
            ):
                counters["unsupported_claim"] += 1
            else:  # pragma: no cover - closed shape currently rejects these first
                categories.append("unsupported_claim")
        else:
            categories.extend([
                "policy", "evidence_ids", "numeric_fidelity",
                "advisory_action", "unsupported_claim",
            ])
        if categories:
            score["failures"].append({
                "record_id": record_id,
                "mode": mode,
                "categories": sorted(set(categories)),
            })

    for name, passed in counters.items():
        score[name] = _metric(passed, config["planned_attempts"])

    abstention_pairs = {
        (record_id, "image_recipe") for record_id in expected_by_record
    }
    abstention_pairs.update(
        (record_id, mode)
        for record_id in ("r004", "r005", "r006")
        for mode in ("image_evidence_recipe", "evidence_recipe", "template_reference")
    )
    abstention_passed = 0
    for pair in abstention_pairs:
        parsed = parsed_by_pair.get(pair)
        if parsed is not None and (
            parsed["status"] in {"insufficient_evidence", "invalid_input"}
            and parsed["appearance"] == "unknown"
            and parsed["defect"] == "unknown"
            and parsed["cause"] == "unknown"
            and parsed["next_vial_effect"] == "unknown"
            and parsed["confidence"] == "not_assigned"
            and parsed["operator_action"] == "manual_review_required"
        ):
            abstention_passed += 1
    score["required_abstention"] = _metric(
        abstention_passed, config["required_abstention_attempts"]
    )

    invariance_passed = 0
    prompt_pair_passed = 0
    for record_id in expected_by_record:
        image_pair = (record_id, "image_evidence_recipe")
        text_pair = (record_id, "evidence_recipe")
        image_attempt = by_pair.get(image_pair)
        text_attempt = by_pair.get(text_pair)
        if (
            image_attempt is not None
            and text_attempt is not None
            and image_attempt["prompt_sha256"] == text_attempt["prompt_sha256"]
        ):
            prompt_pair_passed += 1
        if (
            parsed_by_pair.get(image_pair) is not None
            and parsed_by_pair.get(image_pair) == parsed_by_pair.get(text_pair)
        ):
            invariance_passed += 1
    score["prompt_pair_integrity"] = _metric(prompt_pair_passed, config["image_invariance_pairs"])
    score["image_invariance"] = _metric(invariance_passed, config["image_invariance_pairs"])

    template_passed = 0
    for record_id in expected_by_record:
        parsed = parsed_by_pair.get((record_id, "template_reference"))
        if parsed == expected_by_record[record_id]["full"]:
            template_passed += 1
    score["template_control"] = _metric(
        template_passed, config["template_control_attempts"]
    )
    return score

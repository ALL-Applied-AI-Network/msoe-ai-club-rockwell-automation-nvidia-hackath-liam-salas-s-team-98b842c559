"""V2 provenance and cap accounting around the unchanged strict contract scorer."""

from fluidmix.explanation_contract import ContractError, validate_eval_config
from fluidmix.explanation_scoring import score_raw_results


_BASE_PROVENANCE_KEYS = {
    "freeze_tag", "head_commit", "git_clean", "fixture_set_sha256", "prompt_sha256",
    "image_sha256", "model_manifest_sha256", "runtime_lock_sha256", "model_revision",
    "seed", "do_sample", "max_new_tokens", "dtype", "python", "versions",
    "platform_machine", "device_name", "compute_capability", "cuda_version",
    "model_load_seconds", "model_load_error", "started_at_utc", "finished_at_utc",
}
_V2_PROVENANCE_KEYS = {
    "v1_freeze_tag",
    "enable_thinking",
    "input_audit_sha256",
    "chat_template_sha256",
}


def score_v2_raw_results(raw_results, config, oracle):
    """Validate v2 provenance, reuse strict v1 scoring, and count the fixed 18-way cap metric."""
    validate_eval_config(config)
    if config["schema_version"] != "operator_contract_eval.v2":
        raise ContractError("v2 scorer requires the frozen v2 config")
    if not isinstance(raw_results, dict):
        raise ContractError("raw results must be an object")
    provenance = raw_results.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != (
        _BASE_PROVENANCE_KEYS | _V2_PROVENANCE_KEYS
    ):
        raise ContractError("v2 provenance has the wrong keys")
    for key in _V2_PROVENANCE_KEYS:
        if provenance[key] != config[key]:
            raise ContractError(f"v2 provenance mismatch: {key}")

    base_raw = dict(raw_results)
    base_raw["provenance"] = {
        key: provenance[key] for key in _BASE_PROVENANCE_KEYS
    }
    score = score_raw_results(base_raw, config, oracle)

    cap_reaches = 0
    attempts = raw_results["attempts"]
    model_attempts = [
        attempt for attempt in attempts
        if attempt["mode"] != "template_reference"
    ]
    for attempt in model_attempts:
        if attempt["attempt_status"] != "completed":
            continue
        generated = attempt["generated_token_count"]
        if generated is None:
            raise ContractError("completed model attempt has no generated token count")
        if generated > config["max_new_tokens"]:
            raise ContractError("completed model attempt exceeds frozen generation cap")
        if generated == config["max_new_tokens"]:
            cap_reaches += 1

    model_denominator = config["planned_attempts"] - config["template_control_attempts"]
    if model_denominator != 18:
        raise ContractError("v2 cap denominator is not the frozen 18 model attempts")
    score["schema_version"] = "operator_contract_score.v2"
    score["generation_cap_reaches"] = {
        "reached": cap_reaches,
        "total": model_denominator,
    }
    score["scoring_provenance"] = {
        "v1_freeze_tag": config["v1_freeze_tag"],
        "enable_thinking": config["enable_thinking"],
        "input_audit_sha256": config["input_audit_sha256"],
        "chat_template_sha256": config["chat_template_sha256"],
        "scoring_module_sha256": config["scoring_module_sha256"],
        "scoring_script_sha256": config["scoring_script_sha256"],
        "v2_scoring_module_sha256": config["v2_scoring_module_sha256"],
        "contract_module_sha256": config["contract_module_sha256"],
    }
    return score

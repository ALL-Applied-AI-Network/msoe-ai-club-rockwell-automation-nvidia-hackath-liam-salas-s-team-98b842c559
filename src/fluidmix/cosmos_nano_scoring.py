"""Strict provenance and denominator scoring for the Nano matched evaluation."""

import hashlib
import re

from fluidmix.cosmos_nano_contract import validate_nano_eval_config
from fluidmix.explanation_contract import ContractError, canonical_json
from fluidmix.explanation_scoring import score_explanation_attempts


PROVENANCE_KEYS = {
    "freeze_tag",
    "head_commit",
    "git_clean",
    "config_sha256",
    "fixture_set_sha256",
    "prompt_sha256",
    "image_sha256",
    "model_view_manifest_sha256",
    "runtime_lock_sha256",
    "input_audit_sha256",
    "source_sha256",
    "model_repository",
    "model_revision",
    "seed",
    "do_sample",
    "max_new_tokens",
    "batch_size",
    "dtype",
    "python",
    "platform_machine",
    "runtime_versions",
    "processor_class",
    "tokenizer_class",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "image_preprocessor_sha256",
    "model_class",
    "attention_implementation",
    "generation_config",
    "parameter_devices",
    "model_load_seconds",
    "model_load_error",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "device_name",
    "compute_capability",
    "cuda_version",
    "driver_version",
    "native_eos_token_ids",
    "native_pad_token_id",
    "started_at_utc",
    "finished_at_utc",
}
ATTEMPT_KEYS = {
    "attempt_index",
    "record_id",
    "mode",
    "prompt_sha256",
    "image_attached",
    "attempt_status",
    "response_raw",
    "response_utf8_sha256",
    "generated_suffix_token_ids",
    "generated_suffix_sha256",
    "generated_token_count",
    "terminal_token_id",
    "terminal_token_is_eos",
    "length_cap_reached",
    "truncated_without_eos",
    "elapsed_seconds",
    "error_type",
    "error",
}
_GENERATION_FIELDS = {
    "generated_suffix_token_ids",
    "generated_suffix_sha256",
    "generated_token_count",
    "terminal_token_id",
    "terminal_token_is_eos",
    "length_cap_reached",
    "truncated_without_eos",
}


def _is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_attempt(row, expected_index, config):
    if not isinstance(row, dict) or set(row) != ATTEMPT_KEYS:
        raise ContractError("raw attempt has the wrong keys")
    record_id, mode = expected_index[0]
    if row["attempt_index"] != expected_index[1]:
        raise ContractError("attempt index does not match fixture-major order")
    if row["record_id"] != record_id or row["mode"] != mode:
        raise ContractError("attempt pair does not match fixture-major order")
    if row["attempt_status"] not in {"completed", "error"}:
        raise ContractError("invalid attempt status")
    expected_image = mode in {"image_evidence_recipe", "image_recipe"}
    if row["image_attached"] is not expected_image:
        raise ContractError("attempt image attachment does not match mode")
    if mode == "template_reference":
        if row["prompt_sha256"] is not None:
            raise ContractError("template control cannot have a model prompt")
    elif not _is_digest(row["prompt_sha256"]):
        raise ContractError("model attempt lacks a prompt SHA-256")
    elapsed = row["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ContractError("attempt elapsed_seconds is invalid")
    if row["attempt_status"] == "error":
        if row["response_raw"] is not None or row["response_utf8_sha256"] is not None:
            raise ContractError("error attempt cannot contain a completion")
        if any(row[field] is not None for field in _GENERATION_FIELDS):
            raise ContractError("error attempt cannot contain generation diagnostics")
        if not isinstance(row["error_type"], str) or not isinstance(row["error"], str):
            raise ContractError("error attempt lacks bounded error text")
        return
    if not isinstance(row["response_raw"], str):
        raise ContractError("completed attempt lacks raw response text")
    if hashlib.sha256(row["response_raw"].encode("utf-8")).hexdigest() != row[
        "response_utf8_sha256"
    ]:
        raise ContractError("raw response byte hash mismatch")
    if row["error_type"] is not None or row["error"] is not None:
        raise ContractError("completed attempt cannot contain an error")
    if mode == "template_reference":
        if any(row[field] is not None for field in _GENERATION_FIELDS):
            raise ContractError("template control cannot contain generation diagnostics")
        return
    token_ids = row["generated_suffix_token_ids"]
    if (
        not isinstance(token_ids, list)
        or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids)
        or len(token_ids) > config["max_new_tokens"]
    ):
        raise ContractError("generated suffix token IDs are invalid")
    if hashlib.sha256(canonical_json(token_ids).encode("utf-8")).hexdigest() != row[
        "generated_suffix_sha256"
    ]:
        raise ContractError("generated suffix token hash mismatch")
    if row["generated_token_count"] != len(token_ids):
        raise ContractError("generated token count mismatch")
    terminal = token_ids[-1] if token_ids else None
    terminal_is_eos = terminal in config["native_eos_token_ids"]
    cap_reached = len(token_ids) == config["max_new_tokens"]
    expected = {
        "terminal_token_id": terminal,
        "terminal_token_is_eos": terminal_is_eos,
        "length_cap_reached": cap_reached,
        "truncated_without_eos": cap_reached and not terminal_is_eos,
    }
    for field, value in expected.items():
        if row[field] != value or type(row[field]) is not type(value):
            raise ContractError(f"generation diagnostic mismatch: {field}")


def _validate_provenance(provenance, config):
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise ContractError("Nano raw provenance has the wrong keys")
    matches = {
        "freeze_tag": "freeze_tag",
        "fixture_set_sha256": "fixture_set_sha256",
        "prompt_sha256": "prompt_sha256",
        "image_sha256": "image_sha256",
        "model_view_manifest_sha256": "model_view_manifest_sha256",
        "runtime_lock_sha256": "runtime_lock_sha256",
        "input_audit_sha256": "input_audit_sha256",
        "source_sha256": "source_sha256",
        "model_repository": "model_repository",
        "model_revision": "model_revision",
        "seed": "seed",
        "do_sample": "do_sample",
        "max_new_tokens": "max_new_tokens",
        "batch_size": "batch_size",
        "dtype": "dtype",
        "python": "python_version",
        "platform_machine": "platform_machine",
        "processor_class": "expected_processor_class",
        "tokenizer_class": "expected_tokenizer_class",
        "chat_template_file_sha256": "chat_template_file_sha256",
        "chat_template_text_sha256": "chat_template_text_sha256",
        "native_eos_token_ids": "native_eos_token_ids",
        "native_pad_token_id": "native_pad_token_id",
    }
    for provenance_key, config_key in matches.items():
        if provenance[provenance_key] != config[config_key]:
            raise ContractError(f"Nano provenance mismatch: {provenance_key}")
    if provenance["git_clean"] is not True:
        raise ContractError("Nano raw results must come from a clean frozen checkout")
    if not re.fullmatch(r"[0-9a-f]{40}", provenance["head_commit"] or ""):
        raise ContractError("Nano provenance HEAD is not a full commit")
    if not _is_digest(provenance["config_sha256"]):
        raise ContractError("Nano provenance config hash is invalid")
    if not _is_digest(provenance["image_preprocessor_sha256"]):
        raise ContractError("Nano provenance preprocessor hash is invalid")
    if not isinstance(provenance["runtime_versions"], dict):
        raise ContractError("Nano provenance runtime versions are missing")
    generation = provenance["generation_config"]
    load_error = provenance["model_load_error"]
    if load_error is None:
        if provenance["model_class"] != config["expected_model_class"]:
            raise ContractError("Nano provenance model class mismatch")
        devices = provenance["parameter_devices"]
        if not isinstance(devices, list) or not devices or any(
            not str(device).startswith("cuda") for device in devices
        ):
            raise ContractError("Nano provenance model devices are not all CUDA")
        if not isinstance(generation, dict):
            raise ContractError("Nano provenance generation config is missing")
        if generation.get("eos_token_id") != config["native_eos_token_ids"]:
            raise ContractError("Nano observed EOS configuration mismatch")
        if generation.get("pad_token_id") != config["native_pad_token_id"]:
            raise ContractError("Nano observed pad configuration mismatch")
    elif not isinstance(load_error, dict) or set(load_error) != {"type", "message"}:
        raise ContractError("Nano model-load error has the wrong shape")


def score_nano_raw_results(raw_results, config, oracle):
    """Validate one complete raw artifact and return fixed-denominator metrics."""
    validate_nano_eval_config(config)
    if not isinstance(raw_results, dict) or set(raw_results) != {
        "schema_version",
        "experiment_id",
        "provenance",
        "attempts",
    }:
        raise ContractError("Nano raw results have the wrong keys")
    if raw_results["schema_version"] != "cosmos3_nano_raw_results.v1":
        raise ContractError("unsupported Nano raw-results schema")
    if raw_results["experiment_id"] != config["experiment_id"]:
        raise ContractError("Nano raw-results experiment mismatch")
    _validate_provenance(raw_results["provenance"], config)
    if oracle.get("schema_version") != "operator_contract_oracle.v1":
        raise ContractError("unsupported scoring oracle schema")
    if oracle.get("fixture_set_sha256") != config["fixture_set_sha256"]:
        raise ContractError("scoring oracle is bound to a different fixture set")
    expected_by_record = oracle.get("expected_by_record")
    if not isinstance(expected_by_record, dict) or len(expected_by_record) != 6:
        raise ContractError("scoring oracle has the wrong fixture count")
    expected_pairs = [
        (record_id, mode)
        for record_id in sorted(expected_by_record)
        for mode in config["modes"]
    ]
    attempts = raw_results["attempts"]
    if not isinstance(attempts, list) or len(attempts) != config["planned_attempts"]:
        raise ContractError("incomplete Nano raw artifact is not scoreable")
    for index, (row, pair) in enumerate(zip(attempts, expected_pairs, strict=True)):
        _validate_attempt(row, (pair, index), config)

    completed = sum(row["attempt_status"] == "completed" for row in attempts)
    model_rows = [row for row in attempts if row["mode"] != "template_reference"]
    control_rows = [row for row in attempts if row["mode"] == "template_reference"]
    if raw_results["provenance"]["model_load_error"] is not None and any(
        row["attempt_status"] != "error" for row in model_rows
    ):
        raise ContractError(
            "a model-load failure requires all 18 model attempts to be error rows"
        )
    model_completed = sum(row["attempt_status"] == "completed" for row in model_rows)
    control_completed = sum(row["attempt_status"] == "completed" for row in control_rows)
    score = {
        "schema_version": "cosmos3_nano_score.v1",
        "experiment_id": config["experiment_id"],
        "matrix": {
            "planned": 24,
            "attempted": 24,
            "completed": completed,
            "errors": 24 - completed,
            "complete": True,
            "model": {
                "planned": 18,
                "completed": model_completed,
                "errors": 18 - model_completed,
            },
            "control": {
                "planned": 6,
                "completed": control_completed,
                "errors": 6 - control_completed,
            },
        },
        "per_mode": {},
    }
    for mode in config["modes"]:
        rows = [row for row in attempts if row["mode"] == mode]
        mode_completed = sum(row["attempt_status"] == "completed" for row in rows)
        score["per_mode"][mode] = {
            "planned": 6,
            "completed": mode_completed,
            "errors": 6 - mode_completed,
        }
    score.update(score_explanation_attempts(attempts, config, oracle))
    score["generation_cap_reaches"] = {
        "reached": sum(row["length_cap_reached"] is True for row in model_rows),
        "total": 18,
    }
    score["truncated_without_eos"] = {
        "count": sum(row["truncated_without_eos"] is True for row in model_rows),
        "total": 18,
    }
    return score

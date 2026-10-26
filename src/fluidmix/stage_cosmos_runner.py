"""Frozen execution boundary for the 48-row stage/vial Cosmos matrix."""

from __future__ import annotations

import hashlib
import time

from fluidmix.cosmos_nano_runner import (
    FatalRunError,
    decode_generated_suffix,
    decode_generated_suffix_batch,
)
from fluidmix.explanation_contract import ContractError


_COMPLETION_KEYS = {
    "response_raw",
    "response_utf8_sha256",
    "generated_suffix_token_ids",
    "generated_suffix_sha256",
    "generated_token_count",
    "terminal_token_id",
    "terminal_token_is_eos",
    "length_cap_reached",
    "truncated_without_eos",
}
_CONDITIONS = (
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
    "deterministic_template",
)


def _validate_plan(attempt_plan: object) -> list[dict[str, object]]:
    if not isinstance(attempt_plan, list) or len(attempt_plan) != 48:
        raise ContractError("stage Cosmos plan must contain exactly 48 attempts")
    if any(not isinstance(row, dict) for row in attempt_plan):
        raise ContractError("stage Cosmos attempts must be objects")
    expected_conditions = list(_CONDITIONS) * 12
    if [row.get("attempt_index") for row in attempt_plan] != list(range(48)):
        raise ContractError("stage Cosmos attempt indexes drifted")
    if [row.get("condition_id") for row in attempt_plan] != expected_conditions:
        raise ContractError("stage Cosmos attempts are not decision-major")
    decision_ids = [attempt_plan[index]["decision_id"] for index in range(0, 48, 4)]
    if len(set(decision_ids)) != 12 or any(
        row.get("decision_id") != decision_ids[index // 4]
        for index, row in enumerate(attempt_plan)
    ):
        raise ContractError("stage Cosmos decision identities drifted")
    for row in attempt_plan:
        is_control = row.get("condition_id") == "deterministic_template"
        if (
            row.get("kind") != ("control" if is_control else "model")
            or not isinstance(row.get("attempt_id"), str)
            or row.get("attempt_id")
            != f"{row.get('condition_id')}-{row.get('decision_id')}"
            or not isinstance(row.get("evidence_record_sha256"), str)
            or len(row["evidence_record_sha256"]) != 64
            or not isinstance(row.get("image_ids"), list)
            or not isinstance(row.get("image_captions"), list)
            or len(row["image_captions"]) != len(row["image_ids"])
            or row.get("image_count") != len(row["image_ids"])
            or len(row["image_ids"]) > 6
            or not isinstance(row.get("irrelevant_source_decision_ids"), list)
            or not isinstance(row.get("irrelevant_source_statuses"), list)
        ):
            raise ContractError("stage Cosmos attempt metadata drifted")
        is_irrelevant = row.get("condition_id") == "irrelevant_shuffled_images"
        if is_irrelevant:
            if (
                not row["irrelevant_source_decision_ids"]
                or len(row["irrelevant_source_decision_ids"])
                != len(row["irrelevant_source_statuses"])
                or row["decision_id"] in row["irrelevant_source_decision_ids"]
            ):
                raise ContractError("stage irrelevant-image provenance drifted")
        elif row["irrelevant_source_decision_ids"] or row["irrelevant_source_statuses"]:
            raise ContractError("non-irrelevant attempt retained irrelevant provenance")
        if is_control:
            if row.get("prompt") is not None or not isinstance(
                row.get("template_response"), str
            ):
                raise ContractError("stage Cosmos control attempt is malformed")
        elif (
            not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("prompt_sha256"), str)
            or hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
            != row["prompt_sha256"]
            or row.get("template_response") is not None
        ):
            raise ContractError("stage Cosmos model attempt is malformed")
    return attempt_plan


def _base_attempt(spec: dict[str, object]) -> dict[str, object]:
    return {
        "attempt_index": spec["attempt_index"],
        "attempt_id": spec["attempt_id"],
        "decision_id": spec["decision_id"],
        "partition": spec["partition"],
        "condition_id": spec["condition_id"],
        "kind": spec["kind"],
        "prompt_sha256": spec["prompt_sha256"],
        "evidence_record_sha256": spec["evidence_record_sha256"],
        "image_ids": list(spec["image_ids"]),
        "image_captions": list(spec.get("image_captions", [])),
        "image_count": spec["image_count"],
        "irrelevant_source_decision_ids": list(
            spec.get("irrelevant_source_decision_ids", [])
        ),
        "irrelevant_source_statuses": list(
            spec.get("irrelevant_source_statuses", [])
        ),
    }


def _error_row(
    spec: dict[str, object], started: float, error_type: object, message: object
) -> dict[str, object]:
    return {
        **_base_attempt(spec),
        "attempt_status": "error",
        "response_raw": None,
        "response_utf8_sha256": None,
        "generated_suffix_token_ids": None,
        "generated_suffix_sha256": None,
        "generated_token_count": None,
        "terminal_token_id": None,
        "terminal_token_is_eos": None,
        "length_cap_reached": None,
        "truncated_without_eos": None,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "error_type": str(error_type)[:200],
        "error": str(message)[:2000],
    }


def _control_row(spec: dict[str, object], started: float) -> dict[str, object]:
    raw = spec["template_response"]
    return {
        **_base_attempt(spec),
        "attempt_status": "completed",
        "response_raw": raw,
        "response_utf8_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "generated_suffix_token_ids": None,
        "generated_suffix_sha256": None,
        "generated_token_count": None,
        "terminal_token_id": None,
        "terminal_token_is_eos": None,
        "length_cap_reached": None,
        "truncated_without_eos": None,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "error_type": None,
        "error": None,
    }


def execute_stage_attempt_plan(
    attempt_plan,
    *,
    infer=None,
    infer_batch=None,
    batch_size=1,
    model_load_error=None,
):
    """Execute every frozen model/control attempt exactly once.

    Every batch exception becomes one explicit error row for each affected planned
    attempt. Audit and memory failures therefore stay in the fixed denominator.
    """

    plan = _validate_plan(attempt_plan)
    if model_load_error is None:
        if (infer is None) == (infer_batch is None):
            raise ContractError("provide exactly one stage inference callable")
    elif infer is not None or infer_batch is not None:
        raise ContractError("stage load error cannot be combined with inference")
    if type(batch_size) is not int or batch_size < 1 or 12 % batch_size != 0:
        raise ContractError("stage batch_size must be a positive divisor of 12")
    if model_load_error is not None and (
        not isinstance(model_load_error, dict)
        or set(model_load_error) != {"type", "message"}
    ):
        raise ContractError("stage model_load_error has the wrong shape")
    results_by_index: dict[int, dict[str, object]] = {}
    for spec in plan:
        started = time.monotonic()
        if spec["kind"] == "control":
            results_by_index[spec["attempt_index"]] = _control_row(spec, started)
        elif model_load_error is not None:
            results_by_index[spec["attempt_index"]] = (
                _error_row(
                    spec,
                    started,
                    model_load_error["type"],
                    model_load_error["message"],
                )
            )
    if model_load_error is not None:
        return [results_by_index[index] for index in range(48)]

    if infer is not None:
        infer_batch = lambda specs: [infer(specs[0])]
        batch_size = 1
    model_specs = [spec for spec in plan if spec["kind"] == "model"]
    batches = []
    for condition in _CONDITIONS[:3]:
        condition_specs = [
            spec for spec in model_specs if spec["condition_id"] == condition
        ]
        batches.extend(
            condition_specs[index : index + batch_size]
            for index in range(0, 12, batch_size)
        )
    for specs in batches:
        started = time.monotonic()
        try:
            completions = infer_batch(specs)
            if (
                not isinstance(completions, list)
                or len(completions) != len(specs)
                or any(
                    not isinstance(completion, dict)
                    or set(completion) != _COMPLETION_KEYS
                    for completion in completions
                )
            ):
                raise TypeError("stage batch inference completions have the wrong shape")
            elapsed = round(time.monotonic() - started, 6)
            for spec, completion in zip(specs, completions, strict=True):
                results_by_index[spec["attempt_index"]] = {
                    **_base_attempt(spec),
                    "attempt_status": "completed",
                    **completion,
                    "elapsed_seconds": elapsed,
                    "error_type": None,
                    "error": None,
                }
        except Exception as exc:
            for spec in specs:
                results_by_index[spec["attempt_index"]] = _error_row(
                    spec, started, type(exc).__name__, str(exc)
                )
    if len(results_by_index) != 48:
        raise FatalRunError("stage attempt matrix did not retain all 48 outcomes")
    return [results_by_index[index] for index in range(48)]


def generate_from_audited_stage_inputs(
    model,
    processor,
    spec,
    built,
    audit,
    config,
    *,
    inference_context,
    move_dtype=None,
):
    """Generate once after verifying the live stage processor tensors."""

    from fluidmix.stage_cosmos_inputs import assert_stage_attempt_matches_audit

    assert_stage_attempt_matches_audit(spec, built, audit)
    inputs = built.get("tensors")
    if not hasattr(inputs, "to"):
        raise ContractError("stage processor batch cannot be moved to the model device")
    if move_dtype is None:
        inputs = inputs.to(model.device)
    else:
        inputs = inputs.to(model.device, move_dtype)
    with inference_context():
        generated_ids = model.generate(
            **inputs,
            do_sample=config["do_sample"],
            max_new_tokens=config["max_new_tokens"],
        )
    return decode_generated_suffix(
        processor, generated_ids, inputs["input_ids"], config
    )


def generate_from_audited_stage_input_batch(
    model,
    processor,
    specs,
    built,
    audit,
    config,
    *,
    inference_context,
    move_dtype=None,
):
    """Generate one frozen condition-homogeneous batch and decode every row."""

    from fluidmix.stage_cosmos_inputs import assert_stage_batch_matches_audit

    assert_stage_batch_matches_audit(specs, built, audit)
    inputs = built.get("tensors")
    if not hasattr(inputs, "to"):
        raise ContractError("stage processor batch cannot be moved to the model device")
    if move_dtype is None:
        inputs = inputs.to(model.device)
    else:
        inputs = inputs.to(model.device, move_dtype)
    with inference_context():
        generated_ids = model.generate(
            **inputs,
            do_sample=config["do_sample"],
            max_new_tokens=config["max_new_tokens"],
        )
    return decode_generated_suffix_batch(
        processor, generated_ids, inputs["input_ids"], config
    )

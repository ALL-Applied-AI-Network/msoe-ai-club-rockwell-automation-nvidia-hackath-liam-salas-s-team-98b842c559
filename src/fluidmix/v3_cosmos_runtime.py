"""Execution boundary for the Varied V3 Cosmos explanation matrix."""

from __future__ import annotations

import hashlib
import re
import time

from fluidmix.cosmos_nano_runner import FatalRunError
from fluidmix.explanation_contract import ContractError


_CONDITIONS = (
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
    "deterministic_template",
)
_MODEL_CONDITIONS = _CONDITIONS[:3]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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


def _validate_plan(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 72:
        raise ContractError("V3 Cosmos plan must contain exactly 72 attempts")
    if any(not isinstance(row, dict) for row in value):
        raise ContractError("V3 Cosmos attempts must be objects")
    if [row.get("attempt_index") for row in value] != list(range(72)):
        raise ContractError("V3 Cosmos attempt indexes drifted")
    if [row.get("condition_id") for row in value] != list(_CONDITIONS) * 18:
        raise ContractError("V3 Cosmos attempts are not decision-major")
    decision_ids = [value[index]["decision_id"] for index in range(0, 72, 4)]
    if (
        len(set(decision_ids)) != 18
        or any(
            not isinstance(decision_id, str) or not decision_id
            for decision_id in decision_ids
        )
        or any(
            row.get("decision_id") != decision_ids[index // 4]
            for index, row in enumerate(value)
        )
    ):
        raise ContractError("V3 Cosmos decision identities drifted")
    for row in value:
        condition = row.get("condition_id")
        control = condition == "deterministic_template"
        images = row.get("image_ids")
        captions = row.get("image_captions")
        irrelevant_ids = row.get("irrelevant_source_decision_ids")
        irrelevant_run_ids = row.get("irrelevant_source_run_ids")
        irrelevant_statuses = row.get("irrelevant_source_statuses")
        if (
            row.get("kind") != ("control" if control else "model")
            or not isinstance(row.get("attempt_id"), str)
            or row.get("attempt_id") != f"{condition}-{row.get('decision_id')}"
            or not isinstance(row.get("run_id"), str)
            or not row["run_id"]
            or row.get("split_role") not in {"train", "validation", "evaluation"}
            or row.get("status") != "not_run"
            or not isinstance(row.get("evidence_record_sha256"), str)
            or _SHA256.fullmatch(row["evidence_record_sha256"]) is None
            or not isinstance(images, list)
            or any(not isinstance(item, str) or not item for item in images)
            or len(set(images)) != len(images)
            or not isinstance(captions, list)
            or any(not isinstance(item, str) or not item for item in captions)
            or len(captions) != len(images)
            or row.get("image_count") != len(images)
            or len(images) > 6
            or not isinstance(irrelevant_ids, list)
            or not isinstance(irrelevant_run_ids, list)
            or not isinstance(irrelevant_statuses, list)
            or not isinstance(row.get("requires_multi_image_smoke"), bool)
            or row["requires_multi_image_smoke"] is not bool(images)
        ):
            raise ContractError("V3 Cosmos attempt metadata drifted")
        if condition in {"numbers_only", "deterministic_template"} and images:
            raise ContractError("V3 nonvisual attempt retained images")
        if condition in {
            "ordered_stage_visual_pairs",
            "irrelevant_shuffled_images",
        } and len(images) not in {4, 6}:
            raise ContractError("V3 visual attempt must contain two or three image pairs")
        irrelevant = condition == "irrelevant_shuffled_images"
        if irrelevant:
            if (
                not irrelevant_ids
                or len(irrelevant_ids) != len(irrelevant_run_ids)
                or len(irrelevant_ids) != len(irrelevant_statuses)
                or row["decision_id"] in irrelevant_ids
                or row["run_id"] in irrelevant_run_ids
            ):
                raise ContractError("V3 irrelevant-image provenance drifted")
        elif irrelevant_ids or irrelevant_run_ids or irrelevant_statuses:
            raise ContractError("V3 non-irrelevant attempt retained irrelevant provenance")
        if control:
            if (
                row.get("prompt") is not None
                or row.get("prompt_sha256") is not None
                or not isinstance(row.get("template_response"), str)
                or not row["template_response"]
            ):
                raise ContractError("V3 control attempt is malformed")
        else:
            prompt = row.get("prompt")
            if (
                not isinstance(prompt, str)
                or not prompt
                or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                != row.get("prompt_sha256")
                or row.get("template_response") is not None
            ):
                raise ContractError("V3 model attempt is malformed")
    return value


def _base(spec: dict[str, object]) -> dict[str, object]:
    return {
        key: spec[key]
        for key in (
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
    }


def _error(
    spec: dict[str, object], started: float, error_type: object, message: object
) -> dict[str, object]:
    return {
        **_base(spec),
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


def _control(spec: dict[str, object], started: float) -> dict[str, object]:
    raw = str(spec["template_response"])
    return {
        **_base(spec),
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


def _validate_completion(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _COMPLETION_KEYS:
        raise TypeError("V3 batch inference completion has the wrong shape")
    raw = value.get("response_raw")
    if (
        not isinstance(raw, str)
        or not isinstance(value.get("response_utf8_sha256"), str)
        or hashlib.sha256(raw.encode("utf-8")).hexdigest()
        != value["response_utf8_sha256"]
    ):
        raise TypeError("V3 batch inference response hash mismatch")
    return value


def execute_v3_attempt_plan(
    attempt_plan: object,
    *,
    infer_batch=None,
    batch_size: int = 1,
    model_load_error: object = None,
) -> list[dict[str, object]]:
    """Execute the frozen 18-decision V3 matrix without repairing outputs."""

    plan = _validate_plan(attempt_plan)
    if type(batch_size) is not int or batch_size < 1 or 18 % batch_size != 0:
        raise ContractError("V3 batch_size must be a positive divisor of 18")
    if model_load_error is None:
        if not callable(infer_batch):
            raise ContractError("V3 execution requires one batch inference callable")
    elif (
        infer_batch is not None
        or not isinstance(model_load_error, dict)
        or set(model_load_error) != {"type", "message"}
    ):
        raise ContractError("V3 model load error input is malformed")

    results: dict[int, dict[str, object]] = {}
    for spec in plan:
        started = time.monotonic()
        if spec["kind"] == "control":
            results[int(spec["attempt_index"])] = _control(spec, started)
        elif model_load_error is not None:
            results[int(spec["attempt_index"])] = _error(
                spec,
                started,
                model_load_error["type"],
                model_load_error["message"],
            )
    if model_load_error is not None:
        return [results[index] for index in range(72)]

    for condition in _MODEL_CONDITIONS:
        members = [
            spec
            for spec in plan
            if spec["kind"] == "model" and spec["condition_id"] == condition
        ]
        if len(members) != 18:
            raise ContractError("V3 model condition denominator drifted")
        for start in range(0, 18, batch_size):
            specs = members[start : start + batch_size]
            started = time.monotonic()
            try:
                completions = infer_batch(specs)
                if not isinstance(completions, list) or len(completions) != len(specs):
                    raise TypeError("V3 batch inference completions have the wrong shape")
                checked = [_validate_completion(completion) for completion in completions]
                elapsed = round(time.monotonic() - started, 6)
                for spec, completion in zip(specs, checked, strict=True):
                    results[int(spec["attempt_index"])] = {
                        **_base(spec),
                        "attempt_status": "completed",
                        **completion,
                        "elapsed_seconds": elapsed,
                        "error_type": None,
                        "error": None,
                    }
            except (ContractError, FatalRunError):
                raise
            except Exception as exc:
                for spec in specs:
                    results[int(spec["attempt_index"])] = _error(
                        spec, started, type(exc).__name__, str(exc)
                    )
    if len(results) != 72:
        raise FatalRunError("V3 attempt matrix did not retain all 72 outcomes")
    return [results[index] for index in range(72)]

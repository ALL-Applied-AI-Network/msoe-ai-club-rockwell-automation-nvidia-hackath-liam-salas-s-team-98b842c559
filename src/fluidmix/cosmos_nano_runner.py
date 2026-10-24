"""Nano-specific model loading, generation, decoding, and attempt retention."""

import hashlib
import time

from fluidmix.cosmos_nano_input_audit import assert_attempt_matches_audit
from fluidmix.explanation_contract import ContractError, canonical_json


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


class FatalRunError(RuntimeError):
    """A fatal condition makes the frozen run incomplete and unpublishable."""


def load_nano_model(auto_model_class, model_dir, torch_module, config):
    """Load the pinned Nano model with the exact local bf16 CUDA treatment."""
    model = auto_model_class.from_pretrained(
        model_dir,
        dtype=torch_module.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    if type(model).__name__ != config["expected_model_class"]:
        raise ContractError("loaded model class does not match the frozen Nano class")
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if not devices or any(not device.startswith("cuda") for device in devices):
        raise ContractError(f"model parameters are not entirely CUDA-resident: {devices}")
    return model


def decode_generated_suffix(processor, generated_ids, input_ids, config):
    """Slice at the exact prompt width and decode only the one generated suffix."""
    if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
        raise ContractError("input_ids batch dimension must be exactly one")
    if len(generated_ids.shape) != 2 or generated_ids.shape[0] != 1:
        raise ContractError("generated batch dimension must be exactly one")
    prompt_length = int(input_ids.shape[1])
    if generated_ids.shape[1] < prompt_length:
        raise ContractError("generated sequence is shorter than its prompt")
    suffix = generated_ids[:, prompt_length:]
    token_ids = [int(value) for value in suffix[0].detach().cpu().tolist()]
    if len(token_ids) > config["max_new_tokens"]:
        raise ContractError("generated suffix exceeds the frozen token cap")
    decoded = processor.batch_decode(
        suffix,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
        raise ContractError("batch_decode must return exactly one text completion")
    response = decoded[0]
    terminal = token_ids[-1] if token_ids else None
    terminal_is_eos = terminal in config["native_eos_token_ids"]
    cap_reached = len(token_ids) == config["max_new_tokens"]
    return {
        "response_raw": response,
        "response_utf8_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "generated_suffix_token_ids": token_ids,
        "generated_suffix_sha256": hashlib.sha256(
            canonical_json(token_ids).encode("utf-8")
        ).hexdigest(),
        "generated_token_count": len(token_ids),
        "terminal_token_id": terminal,
        "terminal_token_is_eos": terminal_is_eos,
        "length_cap_reached": cap_reached,
        "truncated_without_eos": cap_reached and not terminal_is_eos,
    }


def generate_from_audited_inputs(
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
    """Verify live tensors, make one greedy call, then apply frozen suffix decoding."""
    assert_attempt_matches_audit(spec, built, audit)
    inputs = built["tensors"]
    if not hasattr(inputs, "to"):
        raise ContractError("processor batch cannot be moved to the model device")
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


def _base_attempt(spec):
    return {
        "attempt_index": spec["attempt_index"],
        "record_id": spec["record_id"],
        "mode": spec["mode"],
        "prompt_sha256": spec["prompt_sha256"],
        "image_attached": spec["image_attached"],
    }


def _error_row(spec, started, error_type, message):
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


def _template_row(spec, started):
    response = spec["template_response"]
    return {
        **_base_attempt(spec),
        "attempt_status": "completed",
        "response_raw": response,
        "response_utf8_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
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


def execute_nano_attempt_plan(attempt_plan, *, infer, model_load_error=None):
    """Execute each model attempt once and retain every catchable ordinary error."""
    if (infer is None) == (model_load_error is None):
        raise ContractError("provide either an inference callable or one load error")
    if model_load_error is not None and (
        not isinstance(model_load_error, dict)
        or set(model_load_error) != {"type", "message"}
    ):
        raise ContractError("model_load_error has the wrong shape")
    results = []
    for spec in attempt_plan:
        started = time.monotonic()
        if spec["mode"] == "template_reference":
            results.append(_template_row(spec, started))
            continue
        if model_load_error is not None:
            results.append(
                _error_row(
                    spec,
                    started,
                    model_load_error["type"],
                    model_load_error["message"],
                )
            )
            continue
        try:
            completion = infer(spec)
            if not isinstance(completion, dict) or set(completion) != _COMPLETION_KEYS:
                raise TypeError("inference completion has the wrong fields")
            row = {
                **_base_attempt(spec),
                "attempt_status": "completed",
                **completion,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error_type": None,
                "error": None,
            }
        except Exception as exc:
            if isinstance(exc, ContractError):
                raise FatalRunError(
                    "input or runner integrity failure makes the frozen run incomplete"
                ) from exc
            if isinstance(exc, MemoryError) or "OutOfMemory" in type(exc).__name__:
                raise FatalRunError(
                    "memory exhaustion makes the frozen run incomplete"
                ) from exc
            row = _error_row(spec, started, type(exc).__name__, str(exc))
        results.append(row)
    if len(results) != 24:
        raise FatalRunError("frozen attempt matrix did not complete all 24 rows")
    return results

"""Processor-only Nano input builder and byte-level audit.

The builder is shared by the audit and inference runner. Its chat-template
invocation intentionally has no ``enable_thinking`` keyword: Nano's native
template does not define that Edge-only treatment.
"""

import hashlib
import json
from pathlib import Path

from fluidmix.explanation_contract import ContractError, canonical_json


_MODEL_MODES = {
    "image_evidence_recipe",
    "evidence_recipe",
    "image_recipe",
}


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value):
    return _sha256_bytes(value.encode("utf-8"))


def _class_name(value):
    return type(value).__name__


def _template_text(processor):
    value = getattr(processor, "chat_template", None)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("default"), str):
        return value["default"]
    raise ContractError("processor does not expose one native chat template")


def _preprocessor_sha256(processor):
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "to_dict"):
        raise ContractError("processor has no serializable image preprocessor")
    value = image_processor.to_dict()
    if not isinstance(value, dict):
        raise ContractError("image preprocessor metadata must be an object")
    return _sha256_text(canonical_json(value))


def _tensor_fingerprint(value, key):
    try:
        import torch

        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(f"processor output is not a supported tensor: {key}") from exc
    return {
        "dtype": dtype,
        "shape": shape,
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def tensor_fingerprints(batch, *, expected_batch_size=1):
    """Hash every processor output tensor in sorted-key order."""
    if type(expected_batch_size) is not int or expected_batch_size < 1:
        raise ContractError("expected processor batch size must be positive")
    if not hasattr(batch, "items"):
        raise ContractError("processor output must be a mapping")
    keys = sorted(batch)
    if "input_ids" not in keys:
        raise ContractError("processor output has no input_ids")
    result = {key: _tensor_fingerprint(batch[key], key) for key in keys}
    input_shape = result["input_ids"]["shape"]
    if len(input_shape) != 2 or input_shape[0] != expected_batch_size:
        raise ContractError("processor input_ids batch dimension is unexpected")
    return result


def build_model_messages(prompt, image_attached, image_path):
    """Build the sole reviewed conversation content order."""
    if not isinstance(prompt, str):
        raise ContractError("model prompt must be text")
    if type(image_attached) is not bool:
        raise ContractError("image_attached must be bool")
    content = []
    if image_attached:
        content.append({"type": "image", "path": str(image_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_model_inputs(processor, spec, *, image_path, config):
    """Render and tokenize one attempt using Nano's native template invocation."""
    prompt = spec.get("prompt")
    prompt_sha256 = _sha256_text(prompt) if isinstance(prompt, str) else None
    if prompt_sha256 != spec.get("prompt_sha256"):
        raise ContractError("attempt prompt hash changed before processor invocation")
    messages = build_model_messages(prompt, spec.get("image_attached"), image_path)
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise ContractError("native chat-template rendering must be text")
    suffix = config["chat_template_generation_suffix"]
    if not rendered.endswith(suffix):
        raise ContractError("Nano native assistant generation suffix changed")
    tensors = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    content_order = [item["type"] for item in messages[0]["content"]]
    descriptor = [
        {
            "role": "user",
            "content": [
                (
                    {
                        "ordinal": index,
                        "type": "image",
                        "sha256": config["image_sha256"],
                    }
                    if item["type"] == "image"
                    else {
                        "ordinal": index,
                        "type": "text",
                        "sha256": prompt_sha256,
                        "utf8_bytes": len(prompt.encode("utf-8")),
                    }
                )
                for index, item in enumerate(messages[0]["content"])
            ],
        }
    ]
    return {
        "tensors": tensors,
        "tensor_fingerprints": tensor_fingerprints(tensors),
        "prompt_sha256": prompt_sha256,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "rendered_text_sha256": _sha256_text(rendered),
        "rendered_text_utf8_bytes": len(rendered.encode("utf-8")),
        "content_order": content_order,
        "conversation_descriptor_sha256": _sha256_text(canonical_json(descriptor)),
        "image_sha256": config["image_sha256"] if spec["image_attached"] else None,
    }


def _audit_record(spec, built):
    return {
        "attempt_index": spec["attempt_index"],
        "record_id": spec["record_id"],
        "mode": spec["mode"],
        "prompt_sha256": built["prompt_sha256"],
        "prompt_utf8_bytes": built["prompt_utf8_bytes"],
        "image_attached": spec["image_attached"],
        "image_sha256": built["image_sha256"],
        "content_order": built["content_order"],
        "conversation_descriptor_sha256": built[
            "conversation_descriptor_sha256"
        ],
        "rendered_text_sha256": built["rendered_text_sha256"],
        "rendered_text_utf8_bytes": built["rendered_text_utf8_bytes"],
        "tensor_fingerprints": built["tensor_fingerprints"],
    }


def audit_model_inputs(
    attempt_plan,
    processor,
    *,
    image_path,
    config,
    observed_transformers_version,
):
    """Fingerprint the exact 18 Nano model inputs without loading model weights."""
    if observed_transformers_version != config["runtime_versions"]["transformers"]:
        raise ContractError("observed Transformers version does not match the frozen pin")
    if _class_name(processor) != config["expected_processor_class"]:
        raise ContractError("processor class does not match the frozen audit")
    if _class_name(processor.tokenizer) != config["expected_tokenizer_class"]:
        raise ContractError("tokenizer class does not match the frozen audit")
    template = _template_text(processor)
    if _sha256_text(template) != config["chat_template_text_sha256"]:
        raise ContractError("processor chat-template text changed")
    preprocessor_hash = _preprocessor_sha256(processor)
    expected_preprocessor = config.get("image_preprocessor_sha256")
    if expected_preprocessor is not None and preprocessor_hash != expected_preprocessor:
        raise ContractError("image preprocessor configuration changed")
    model_specs = [row for row in attempt_plan if row.get("mode") in _MODEL_MODES]
    if len(model_specs) != 18:
        raise ContractError("processor audit requires exactly 18 model attempts")
    pairs = [(row.get("record_id"), row.get("mode")) for row in model_specs]
    expected_pairs = [
        (f"r{index:03d}", mode)
        for index in range(1, 7)
        for mode in (
            "image_evidence_recipe",
            "evidence_recipe",
            "image_recipe",
        )
    ]
    if pairs != expected_pairs:
        raise ContractError("processor audit model matrix order changed")
    attempts = []
    for spec in model_specs:
        built = build_model_inputs(
            processor, spec, image_path=image_path, config=config
        )
        attempts.append(_audit_record(spec, built))
    return {
        "schema_version": "cosmos3_nano_input_audit.v1",
        "experiment_id": config["experiment_id"],
        "invocation": "native_template_enable_thinking_omitted",
        "model_attempt_count": 18,
        "processor_class": _class_name(processor),
        "tokenizer_class": _class_name(processor.tokenizer),
        "chat_template_text_sha256": _sha256_text(template),
        "image_preprocessor_sha256": preprocessor_hash,
        "model_revision": config["model_revision"],
        "model_view_manifest_sha256": config["model_view_manifest_sha256"],
        "transformers_version": observed_transformers_version,
        "fixture_set_sha256": config["fixture_set_sha256"],
        "prompt_template_sha256": config["prompt_sha256"],
        "image_sha256": config["image_sha256"],
        "attempts": attempts,
    }


def audit_json_bytes(value):
    """Serialize a tracked input audit deterministically."""
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def assert_attempt_matches_audit(spec, built, audit):
    """Recompute live tensor hashes immediately before generation."""
    if not isinstance(audit, dict) or audit.get("model_attempt_count") != 18:
        raise ContractError("tracked Nano input audit is invalid")
    matches = [
        row
        for row in audit.get("attempts", [])
        if row.get("attempt_index") == spec["attempt_index"]
        and row.get("record_id") == spec["record_id"]
        and row.get("mode") == spec["mode"]
    ]
    if len(matches) != 1:
        raise ContractError("attempt is missing or duplicated in the input audit")
    live = dict(built)
    live["tensor_fingerprints"] = tensor_fingerprints(built["tensors"])
    if _audit_record(spec, live) != matches[0]:
        raise ContractError("live processor input differs from the tracked audit")


def verify_tracked_input_audit(audit_path, attempt_plan, processor, **kwargs):
    """Hash and recompute an exclusive tracked audit before loading weights."""
    path = Path(audit_path)
    config = kwargs["config"]
    if path.is_symlink() or not path.is_file():
        raise ContractError("tracked input audit must be a regular file")
    tracked = path.read_bytes()
    if _sha256_bytes(tracked) != config["input_audit_sha256"]:
        raise ContractError("tracked input-audit SHA-256 mismatch")
    recomputed = audit_json_bytes(
        audit_model_inputs(attempt_plan, processor, **kwargs)
    )
    if tracked != recomputed:
        raise ContractError("tracked input audit differs from recomputed processor inputs")
    return json.loads(tracked)

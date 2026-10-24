"""Deterministic processor-only audit for the Cosmos v2 template treatment."""

import hashlib
import json
from pathlib import Path

from fluidmix.explanation_contract import ContractError, canonical_json, validate_eval_config


_V1_DEFAULT_GENERATION_SUFFIX = "<think>\n"
_V2_FALSE_GENERATION_SUFFIX = "<think></think>"
_OMITTED = object()
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
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _common_prefix_length(left, right):
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _input_ids(batch):
    try:
        value = batch["input_ids"]
    except (KeyError, TypeError) as exc:
        raise ContractError("processor audit result has no input_ids") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value[0])
    ):
        raise ContractError("processor audit input_ids must be one integer sequence")
    return value[0]


def build_model_messages(prompt, image_attached, image_path):
    """Build the sole conversation shape shared by the audit and v2 inference."""
    if not isinstance(prompt, str):
        raise ContractError("model prompt must be text")
    if type(image_attached) is not bool:
        raise ContractError("image_attached must be bool")
    content = []
    if image_attached:
        content.append({"type": "image", "path": str(image_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _render_variant(processor, messages, enable_thinking=_OMITTED):
    template_kwargs = {"add_generation_prompt": True}
    if enable_thinking is not _OMITTED:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        **template_kwargs,
    )
    if not isinstance(rendered, str):
        raise ContractError("processor audit rendering must be text")
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    token_ids = _input_ids(batch)
    token_bytes = canonical_json(token_ids).encode("utf-8")
    return rendered, token_ids, {
        "rendered_sha256": _sha256_text(rendered),
        "rendered_utf8_bytes": len(rendered.encode("utf-8")),
        "token_count": len(token_ids),
        "token_ids_sha256": _sha256_bytes(token_bytes),
    }


def _audit_one(spec, processor, image_path, config):
    messages = build_model_messages(spec["prompt"], spec["image_attached"], image_path)
    prompt_bytes = spec["prompt"].encode("utf-8")
    prompt_sha256 = _sha256_bytes(prompt_bytes)
    if prompt_sha256 != spec["prompt_sha256"]:
        raise ContractError("attempt prompt hash changed before processor audit")

    content_descriptor = []
    if spec["image_attached"]:
        content_descriptor.append({
            "ordinal": 0,
            "type": "image",
            "sha256": config["image_sha256"],
        })
    content_descriptor.append({
        "ordinal": len(content_descriptor),
        "type": "text",
        "sha256": prompt_sha256,
        "utf8_bytes": len(prompt_bytes),
    })
    conversation_descriptor = [{"role": "user", "content": content_descriptor}]

    rendered_default, tokens_default, default_record = _render_variant(processor, messages)
    rendered_false, tokens_false, false_record = _render_variant(processor, messages, False)
    if not rendered_default.endswith(_V1_DEFAULT_GENERATION_SUFFIX):
        raise ContractError("v1-default rendering lacks the frozen thinking suffix")
    if not rendered_false.endswith(_V2_FALSE_GENERATION_SUFFIX):
        raise ContractError("v2-false rendering lacks the frozen closed-thinking suffix")
    shared_default = rendered_default[:-len(_V1_DEFAULT_GENERATION_SUFFIX)]
    shared_false = rendered_false[:-len(_V2_FALSE_GENERATION_SUFFIX)]
    if shared_default != shared_false:
        raise ContractError("thinking flag changed rendering before the terminal suffix")

    if len(tokens_default) != len(tokens_false):
        raise ContractError("thinking flag changed the input token count")
    token_prefix = _common_prefix_length(tokens_default, tokens_false)
    if token_prefix != len(tokens_default) - 1:
        raise ContractError("thinking flag changed more than the terminal token")
    if tokens_default[token_prefix:] == tokens_false[token_prefix:]:
        raise ContractError("thinking flag did not change the terminal token")

    default_bytes = rendered_default.encode("utf-8")
    false_bytes = rendered_false.encode("utf-8")
    render_prefix = _common_prefix_length(default_bytes, false_bytes)
    return {
        "attempt_index": spec["attempt_index"],
        "record_id": spec["record_id"],
        "mode": spec["mode"],
        "prompt_sha256": prompt_sha256,
        "prompt_utf8_bytes": len(prompt_bytes),
        "image_attached": spec["image_attached"],
        "image_sha256": config["image_sha256"] if spec["image_attached"] else None,
        "content_order": [item["type"] for item in content_descriptor],
        "conversation_descriptor_sha256": _sha256_text(
            canonical_json(conversation_descriptor)
        ),
        "v1_default": default_record,
        "v2_false": false_record,
        "shared_render_prefix_sha256": _sha256_text(shared_default),
        "shared_render_prefix_utf8_bytes": len(shared_default.encode("utf-8")),
        "common_render_prefix_utf8_bytes": render_prefix,
        "differing_render_suffix_v1_default": default_bytes[render_prefix:].decode("utf-8"),
        "differing_render_suffix_v2_false": false_bytes[render_prefix:].decode("utf-8"),
        "common_token_prefix_count": token_prefix,
        "v1_default_token_suffix": tokens_default[token_prefix:],
        "v2_false_token_suffix": tokens_false[token_prefix:],
    }


def audit_model_inputs(
    attempt_plan,
    processor,
    *,
    image_path,
    config,
    observed_transformers_version,
):
    """Audit every v1-default/v2-false input pair; raise on any extra change."""
    validate_eval_config(config)
    if config["schema_version"] != "operator_contract_eval.v2":
        raise ContractError("processor input audit requires the frozen v2 config")
    if config["enable_thinking"] is not False:
        raise ContractError("v2 processor input audit requires enable_thinking=false")
    if observed_transformers_version != config["runtime_versions"]["transformers"]:
        raise ContractError("observed Transformers version does not match the frozen pin")
    if not isinstance(attempt_plan, list):
        raise ContractError("attempt plan must be a list")
    model_specs = [item for item in attempt_plan if item.get("mode") != "template_reference"]
    if len(model_specs) != 18:
        raise ContractError("processor input audit requires exactly 18 model attempts")
    pairs = [(item.get("record_id"), item.get("mode")) for item in model_specs]
    expected_pairs = {
        (f"r{index:03d}", mode)
        for index in range(1, 7)
        for mode in _MODEL_MODES
    }
    if len(set(pairs)) != 18 or set(pairs) != expected_pairs:
        raise ContractError("processor input audit model matrix is incomplete or duplicated")

    attempts = [
        _audit_one(spec, processor, image_path, config)
        for spec in model_specs
    ]
    return {
        "schema_version": "cosmos_contract_input_audit.v2",
        "audit_id": "cosmos-contract-v2-no-thinking-inputs-20261024",
        "treatment": "enable_thinking_omitted_vs_false",
        "v1_default_invocation": "enable_thinking_omitted",
        "v2_invocation": "enable_thinking_false",
        "expected_generation_suffix_v1_default": _V1_DEFAULT_GENERATION_SUFFIX,
        "expected_generation_suffix_v2_false": _V2_FALSE_GENERATION_SUFFIX,
        "model_attempt_count": 18,
        "processor_class": _class_name(processor),
        "tokenizer_class": _class_name(processor.tokenizer),
        "model_revision": config["model_revision"],
        "model_manifest_sha256": config["model_manifest_sha256"],
        "transformers_version": observed_transformers_version,
        "fixture_set_sha256": config["fixture_set_sha256"],
        "prompt_template_sha256": config["prompt_sha256"],
        "image_sha256": config["image_sha256"],
        "chat_template_sha256": config["chat_template_sha256"],
        "attempts": attempts,
    }


def audit_json_bytes(value):
    """Serialize a tracked input audit deterministically."""
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def verify_tracked_input_audit(
    audit_path,
    attempt_plan,
    processor,
    *,
    image_path,
    config,
    observed_transformers_version,
):
    """Hash and recompute the tracked audit before any model weights are loaded."""
    audit_path = Path(audit_path)
    if audit_path.is_symlink() or not audit_path.is_file():
        raise ContractError("tracked input audit must be a regular file")
    tracked = audit_path.read_bytes()
    digest = _sha256_bytes(tracked)
    if digest != config["input_audit_sha256"]:
        raise ContractError("tracked input-audit SHA-256 mismatch")
    recomputed = audit_json_bytes(
        audit_model_inputs(
            attempt_plan,
            processor,
            image_path=image_path,
            config=config,
            observed_transformers_version=observed_transformers_version,
        )
    )
    if tracked != recomputed:
        raise ContractError("tracked input audit does not match recomputed processor inputs")
    return digest

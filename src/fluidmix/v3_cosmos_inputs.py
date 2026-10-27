"""Processor-input audit for the Varied V3 Cosmos Nano matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

from fluidmix.cosmos_nano_input_audit import tensor_fingerprints
from fluidmix.cosmos_nano_runner import decode_generated_suffix_batch
from fluidmix.explanation_contract import ContractError, canonical_json
from fluidmix.stage_cosmos_inputs import _checked_images
from fluidmix.v3_cosmos_runtime import _validate_plan


_MODEL_CONDITIONS = (
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
)
_LOCATOR_KEYS = {
    "batch_id",
    "mask_path",
    "mask_sha256",
    "recipe_id",
    "rgb_path",
    "rgb_sha256",
    "run_id",
    "source_manifest_sha256",
    "split_role",
    "split_version",
    "stage_id",
    "stage_ordinal",
    "tank_id",
    "vial_id",
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _template_text(processor) -> str:
    value = getattr(processor, "chat_template", None)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("default"), str):
        return value["default"]
    raise ContractError("V3 processor does not expose one native chat template")


def _preprocessor_sha256(processor) -> str:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "to_dict"):
        raise ContractError("V3 processor has no serializable image preprocessor")
    value = image_processor.to_dict()
    if not isinstance(value, dict):
        raise ContractError("V3 image preprocessor metadata must be an object")
    return _sha_text(canonical_json(value))


def _batches(plan: list[dict[str, object]], batch_size: int):
    if type(batch_size) is not int or batch_size < 1 or 18 % batch_size != 0:
        raise ContractError("V3 batch_size must be a positive divisor of 18")
    batches = []
    for condition in _MODEL_CONDITIONS:
        members = [
            row
            for row in plan
            if row.get("kind") == "model" and row.get("condition_id") == condition
        ]
        if len(members) != 18:
            raise ContractError("V3 input audit requires 18 attempts per model condition")
        batches.extend(
            members[start : start + batch_size]
            for start in range(0, 18, batch_size)
        )
    return batches


def _batch_record(
    batch_index: int,
    specs: list[dict[str, object]],
    built: dict[str, object],
) -> dict[str, object]:
    return {
        "batch_index": batch_index,
        "condition_id": built["condition_id"],
        "condition_ids": [spec["condition_id"] for spec in specs],
        "attempt_ids": built["attempt_ids"],
        "attempt_indexes": built["attempt_indexes"],
        "image_counts": [spec["image_count"] for spec in specs],
        "batch_membership_sha256": built["batch_membership_sha256"],
        "rendered_text_sha256": built["rendered_text_sha256"],
        "conversation_descriptor_sha256": built[
            "conversation_descriptor_sha256"
        ],
        "tensor_fingerprints": built["tensor_fingerprints"],
    }


def _checked_processor_locators(
    image_locators: object, required_image_ids: set[str]
) -> dict[str, dict[str, str]]:
    if not isinstance(image_locators, Mapping) or set(image_locators) != required_image_ids:
        raise ContractError("V3 image locator IDs differ from the attempt plan")
    result: dict[str, dict[str, str]] = {}
    for image_id in sorted(required_image_ids):
        raw = image_locators[image_id]
        if not isinstance(raw, Mapping) or set(raw) != _LOCATOR_KEYS:
            raise ContractError("V3 image locator has the wrong fields")
        for role in ("rgb", "mask"):
            path_text = raw.get(f"{role}_path")
            expected = raw.get(f"{role}_sha256")
            if not isinstance(path_text, str) or not isinstance(expected, str):
                raise ContractError(f"V3 {role} locator path or hash is malformed")
            path = Path(path_text)
            if (
                not path.is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ContractError(f"V3 {role.upper()} locator file or hash drifted")
        result[image_id] = {
            key: str(raw[key])
            for key in ("rgb_path", "rgb_sha256", "mask_path", "mask_sha256")
        }
    return result


def _v3_messages_and_descriptor(spec, images):
    """Build the frozen V3 target/same-run-peer conversation."""

    prompt = spec["prompt"]
    captions = spec.get("image_captions")
    allowed = {
        f"Stage {stage} {role} image."
        for stage in (1, 2, 3)
        for role in ("target", "same-run peer")
    }
    if (
        not isinstance(captions, list)
        or len(captions) != len(images)
        or any(not isinstance(caption, str) or caption not in allowed for caption in captions)
    ):
        raise ContractError("V3 image captions are not neutral target/peer labels")
    content = []
    descriptor_content = []
    for index, (image, caption) in enumerate(zip(images, captions, strict=True)):
        content.extend(
            (
                {"type": "text", "text": caption},
                {"type": "image", "path": image["path"]},
            )
        )
        descriptor_content.extend(
            (
                {
                    "ordinal": 2 * index,
                    "type": "text",
                    "sha256": _sha_text(caption),
                    "utf8_bytes": len(caption.encode("utf-8")),
                },
                {
                    "ordinal": 2 * index + 1,
                    "type": "image",
                    "image_id": image["image_id"],
                    "sha256": image["sha256"],
                },
            )
        )
    content.append({"type": "text", "text": prompt})
    descriptor_content.append(
        {
            "ordinal": len(content) - 1,
            "type": "text",
            "sha256": spec["prompt_sha256"],
            "utf8_bytes": len(prompt.encode("utf-8")),
        }
    )
    return (
        [{"role": "user", "content": content}],
        [{"role": "user", "content": descriptor_content}],
    )


def _build_v3_model_input_batch(processor, specs, *, image_locators, config):
    """Build one condition-homogeneous V3 batch with its exact peer labels."""

    if not isinstance(specs, list) or not specs:
        raise ContractError("V3 processor batch must contain model attempts")
    if (
        len(specs) != config.get("batch_size")
        or len({row.get("condition_id") for row in specs}) != 1
    ):
        raise ContractError("V3 processor batch membership is not frozen")
    prepared = []
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("kind") != "model":
            raise ContractError("V3 processor batch contains a non-model attempt")
        prompt = spec.get("prompt")
        if not isinstance(prompt, str) or _sha_text(prompt) != spec.get(
            "prompt_sha256"
        ):
            raise ContractError("V3 attempt prompt hash changed")
        images = _checked_images(spec, image_locators, config)
        messages, descriptor = _v3_messages_and_descriptor(spec, images)
        prepared.append((spec, messages, descriptor))
    conversations = [item[1] for item in prepared]
    rendered = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )
    if (
        not isinstance(rendered, list)
        or len(rendered) != len(specs)
        or any(
            not isinstance(value, str)
            or not value.endswith(config.get("chat_template_generation_suffix", ""))
            for value in rendered
        )
    ):
        raise ContractError("V3 batched native chat-template rendering changed")
    tensors = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    attempt_ids = [spec["attempt_id"] for spec in specs]
    membership = {
        "condition_id": specs[0]["condition_id"],
        "attempt_ids": attempt_ids,
        "attempt_indexes": [spec["attempt_index"] for spec in specs],
    }
    return {
        "tensors": tensors,
        "tensor_fingerprints": tensor_fingerprints(
            tensors, expected_batch_size=len(specs)
        ),
        "condition_id": specs[0]["condition_id"],
        "attempt_ids": attempt_ids,
        "attempt_indexes": membership["attempt_indexes"],
        "batch_membership_sha256": _sha_text(canonical_json(membership)),
        "rendered_text_sha256": [_sha_text(value) for value in rendered],
        "conversation_descriptor_sha256": [
            _sha_text(canonical_json(item[2])) for item in prepared
        ],
    }


def build_v3_model_input_batch(
    processor,
    specs,
    *,
    image_locators,
    config,
):
    """Build one V3 batch through the established native-template path."""

    required = {
        str(image_id)
        for spec in specs
        for image_id in spec.get("image_ids", [])
    }
    subset = {
        image_id: image_locators[image_id]
        for image_id in required
        if image_id in image_locators
    }
    checked = _checked_processor_locators(subset, required)
    return _build_v3_model_input_batch(
        processor,
        specs,
        image_locators=checked,
        config=config,
    )


def generate_from_audited_v3_input_batch(
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
    """Generate once for a frozen batch, retaining every raw suffix."""

    assert_v3_batch_matches_audit(specs, built, audit)
    inputs = built.get("tensors")
    if not hasattr(inputs, "to"):
        raise ContractError("V3 processor batch cannot be moved to the model device")
    inputs = (
        inputs.to(model.device)
        if move_dtype is None
        else inputs.to(model.device, move_dtype)
    )
    with inference_context():
        generated = model.generate(
            **inputs,
            do_sample=config["do_sample"],
            max_new_tokens=config["max_new_tokens"],
        )
    return decode_generated_suffix_batch(
        processor, generated, inputs["input_ids"], config
    )


def audit_v3_model_inputs(
    attempt_plan,
    processor,
    *,
    image_locators,
    config,
    observed_transformers_version,
):
    """Hash all 54 processor inputs before the Nano weights are loaded."""

    plan = _validate_plan(attempt_plan)
    if observed_transformers_version != config.get("runtime_versions", {}).get(
        "transformers"
    ):
        raise ContractError("V3 Transformers version differs from the frozen pin")
    if type(processor).__name__ != config.get("expected_processor_class"):
        raise ContractError("V3 processor class differs from the frozen pin")
    if type(processor.tokenizer).__name__ != config.get("expected_tokenizer_class"):
        raise ContractError("V3 tokenizer class differs from the frozen pin")
    if _sha_text(_template_text(processor)) != config.get(
        "chat_template_text_sha256"
    ):
        raise ContractError("V3 chat-template text changed")
    if _preprocessor_sha256(processor) != config.get("image_preprocessor_sha256"):
        raise ContractError("V3 image preprocessor configuration changed")
    if config.get("max_images_per_attempt") != 6:
        raise ContractError("V3 max image count changed")
    model_specs = [row for row in plan if row["kind"] == "model"]
    if len(model_specs) != 54:
        raise ContractError("V3 input audit requires exactly 54 model attempts")
    required_images = {
        str(image_id)
        for spec in model_specs
        for image_id in spec["image_ids"]
    }
    checked_locators = _checked_processor_locators(image_locators, required_images)
    batch_specs = _batches(plan, config.get("batch_size"))
    batches = []
    for index, specs in enumerate(batch_specs):
        built = _build_v3_model_input_batch(
            processor,
            specs,
            image_locators={
                image_id: checked_locators[image_id]
                for spec in specs
                for image_id in spec["image_ids"]
            },
            config=config,
        )
        batches.append(_batch_record(index, specs, built))
    return {
        "schema_version": "v3_cosmos_input_audit.v1",
        "experiment_id": config["experiment_id"],
        "invocation": "native_template_enable_thinking_omitted",
        "model_attempt_count": 54,
        "batch_size": config["batch_size"],
        "batch_count": len(batches),
        "condition_counts": {
            condition: sum(row["condition_id"] == condition for row in model_specs)
            for condition in _MODEL_CONDITIONS
        },
        "max_images_per_attempt": 6,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template_text_sha256": _sha_text(_template_text(processor)),
        "image_preprocessor_sha256": _preprocessor_sha256(processor),
        "model_revision": config["model_revision"],
        "model_view_manifest_sha256": config["model_view_manifest_sha256"],
        "transformers_version": observed_transformers_version,
        "prompt_template_sha256": config["prompt_template_sha256"],
        "batches": batches,
    }


def assert_v3_batch_matches_audit(specs, built, audit) -> None:
    """Verify live batch membership and tensors against the tracked audit."""

    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != "v3_cosmos_input_audit.v1"
        or audit.get("model_attempt_count") != 54
    ):
        raise ContractError("tracked V3 input audit is invalid")
    attempt_ids = [spec.get("attempt_id") for spec in specs]
    matches = [
        row for row in audit.get("batches", []) if row.get("attempt_ids") == attempt_ids
    ]
    if len(matches) != 1:
        raise ContractError("V3 batch membership is missing or duplicated in the audit")
    live = dict(built)
    live["tensor_fingerprints"] = tensor_fingerprints(
        built["tensors"], expected_batch_size=len(specs)
    )
    if _batch_record(matches[0]["batch_index"], specs, live) != matches[0]:
        raise ContractError("live V3 processor batch differs from the tracked audit")


def v3_audit_json_bytes(value: object) -> bytes:
    """Serialize a V3 audit in the exact tracked representation."""

    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def verify_tracked_v3_input_audit(
    audit_path, attempt_plan, processor, **kwargs
):
    """Hash and recompute the complete audit before model loading."""

    path = Path(audit_path)
    config = kwargs["config"]
    if path.is_symlink() or not path.is_file():
        raise ContractError("tracked V3 input audit must be a regular file")
    tracked = path.read_bytes()
    if hashlib.sha256(tracked).hexdigest() != config.get("input_audit_sha256"):
        raise ContractError("tracked V3 input-audit SHA-256 mismatch")
    recomputed = v3_audit_json_bytes(
        audit_v3_model_inputs(attempt_plan, processor, **kwargs)
    )
    if tracked != recomputed:
        raise ContractError("tracked V3 audit differs from recomputed processor inputs")
    return json.loads(tracked)

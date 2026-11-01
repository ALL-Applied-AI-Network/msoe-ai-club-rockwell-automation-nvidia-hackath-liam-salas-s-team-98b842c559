"""Audited multi-image processor inputs for the stage/vial Cosmos run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fluidmix.explanation.vlm.cosmos_nano_input_audit import tensor_fingerprints
from fluidmix.explanation.template.explanation_contract import ContractError, canonical_json


_MODEL_CONDITIONS = {
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _template_text(processor) -> str:
    value = getattr(processor, "chat_template", None)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("default"), str):
        return value["default"]
    raise ContractError("stage processor does not expose one native chat template")


def _preprocessor_sha256(processor) -> str:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "to_dict"):
        raise ContractError("stage processor has no serializable image preprocessor")
    metadata = image_processor.to_dict()
    if not isinstance(metadata, dict):
        raise ContractError("stage image preprocessor metadata must be an object")
    return _sha256_text(canonical_json(metadata))


def _checked_images(
    spec: dict[str, object], image_locators: object, config: dict[str, object]
) -> list[dict[str, str]]:
    if not isinstance(image_locators, dict):
        raise ContractError("stage image locators must be an object")
    image_ids = spec.get("image_ids")
    if not isinstance(image_ids, list) or any(
        not isinstance(image_id, str) or not image_id for image_id in image_ids
    ):
        raise ContractError("stage attempt image IDs are malformed")
    maximum = config.get("max_images_per_attempt")
    if maximum != 6 or len(image_ids) > maximum or len(set(image_ids)) != len(image_ids):
        raise ContractError("stage attempt exceeds the frozen six-image limit")
    condition = spec.get("condition_id")
    if condition == "numbers_only" and image_ids:
        raise ContractError("numbers-only stage attempt must have no image")
    if condition in {
        "ordered_stage_visual_pairs",
        "irrelevant_shuffled_images",
    } and not image_ids:
        raise ContractError("visual stage attempt must retain at least one image")
    if condition not in _MODEL_CONDITIONS:
        raise ContractError("only model conditions can be processed")

    from PIL import Image

    checked: list[dict[str, str]] = []
    for image_id in image_ids:
        locator = image_locators.get(image_id)
        if not isinstance(locator, dict) or set(locator) != {
            "rgb_path",
            "rgb_sha256",
            "mask_path",
            "mask_sha256",
        }:
            raise ContractError("stage RGB locator has the wrong fields")
        path_text = locator.get("rgb_path")
        expected_sha = locator.get("rgb_sha256")
        if not isinstance(path_text, str) or not isinstance(expected_sha, str):
            raise ContractError("stage RGB locator path or hash is malformed")
        path = Path(path_text)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or _sha256_bytes(path.read_bytes()) != expected_sha
        ):
            raise ContractError("stage RGB locator file or hash drifted")
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (
                    config.get("image_width"),
                    config.get("image_height"),
                ):
                    raise ContractError("stage RGB must be the frozen RGB resolution")
        except OSError as exc:
            raise ContractError("stage RGB is not a readable image") from exc
        checked.append(
            {"image_id": image_id, "path": str(path), "sha256": expected_sha}
        )
    return checked


def _messages_and_descriptor(spec, images):
    prompt = spec["prompt"]
    captions = spec.get("image_captions")
    if (
        not isinstance(captions, list)
        or len(captions) != len(images)
        or any(
            not isinstance(caption, str)
            or caption
            not in {
                f"Stage {stage} {role} image."
                for stage in (1, 2, 3)
                for role in ("target", "reference")
            }
            for caption in captions
        )
    ):
        raise ContractError("stage image captions are not neutral ordered slot labels")
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
                    "sha256": _sha256_text(caption),
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
    messages = [{"role": "user", "content": content}]
    descriptor = [{"role": "user", "content": descriptor_content}]
    return messages, descriptor, content


def build_stage_model_inputs(processor, spec, *, image_locators, config):
    """Build one text or up-to-six-image Nano input."""

    if not isinstance(spec, dict) or spec.get("kind") != "model":
        raise ContractError("stage processor requires one model attempt")
    prompt = spec.get("prompt")
    if not isinstance(prompt, str) or _sha256_text(prompt) != spec.get(
        "prompt_sha256"
    ):
        raise ContractError("stage attempt prompt hash changed")
    images = _checked_images(spec, image_locators, config)
    messages, descriptor, content = _messages_and_descriptor(spec, images)
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered.endswith(
        config.get("chat_template_generation_suffix", "")
    ):
        raise ContractError("stage native chat-template rendering changed")
    tensors = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        "tensors": tensors,
        "tensor_fingerprints": tensor_fingerprints(tensors),
        "prompt_sha256": spec["prompt_sha256"],
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "rendered_text_sha256": _sha256_text(rendered),
        "rendered_text_utf8_bytes": len(rendered.encode("utf-8")),
        "image_ids": [item["image_id"] for item in images],
        "image_sha256": [item["sha256"] for item in images],
        "image_captions": list(spec["image_captions"]),
        "content_order": [item["type"] for item in content],
        "conversation_descriptor_sha256": _sha256_text(
            canonical_json(descriptor)
        ),
    }


def _audit_record(spec: dict[str, object], built: dict[str, object]) -> dict[str, object]:
    return {
        "attempt_index": spec["attempt_index"],
        "attempt_id": spec["attempt_id"],
        "decision_id": spec["decision_id"],
        "condition_id": spec["condition_id"],
        "prompt_sha256": built["prompt_sha256"],
        "prompt_utf8_bytes": built["prompt_utf8_bytes"],
        "image_ids": built["image_ids"],
        "image_sha256": built["image_sha256"],
        "image_captions": built["image_captions"],
        "content_order": built["content_order"],
        "conversation_descriptor_sha256": built[
            "conversation_descriptor_sha256"
        ],
        "rendered_text_sha256": built["rendered_text_sha256"],
        "rendered_text_utf8_bytes": built["rendered_text_utf8_bytes"],
        "tensor_fingerprints": built["tensor_fingerprints"],
    }


def _model_batches(attempt_plan, batch_size):
    if type(batch_size) is not int or batch_size < 1 or 12 % batch_size != 0:
        raise ContractError("stage batch_size must be a positive divisor of 12")
    model_specs = [row for row in attempt_plan if row.get("kind") == "model"]
    batches = []
    for condition in (
        "numbers_only",
        "ordered_stage_visual_pairs",
        "irrelevant_shuffled_images",
    ):
        members = [row for row in model_specs if row.get("condition_id") == condition]
        if len(members) != 12:
            raise ContractError("stage batching requires 12 attempts per model condition")
        batches.extend(
            members[index : index + batch_size]
            for index in range(0, 12, batch_size)
        )
    return batches


def build_stage_model_input_batch(processor, specs, *, image_locators, config):
    """Build one condition-homogeneous padded processor batch."""

    if not isinstance(specs, list) or not specs:
        raise ContractError("stage processor batch must contain model attempts")
    batch_size = config.get("batch_size")
    if len(specs) != batch_size or len({row.get("condition_id") for row in specs}) != 1:
        raise ContractError("stage processor batch membership is not frozen")
    prepared = []
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("kind") != "model":
            raise ContractError("stage processor batch contains a non-model attempt")
        prompt = spec.get("prompt")
        if not isinstance(prompt, str) or _sha256_text(prompt) != spec.get(
            "prompt_sha256"
        ):
            raise ContractError("stage attempt prompt hash changed")
        images = _checked_images(spec, image_locators, config)
        messages, descriptor, content = _messages_and_descriptor(spec, images)
        prepared.append((spec, images, messages, descriptor, content))
    conversations = [item[2] for item in prepared]
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
        raise ContractError("stage batched native chat-template rendering changed")
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
        "batch_membership_sha256": _sha256_text(canonical_json(membership)),
        "rendered_text_sha256": [_sha256_text(value) for value in rendered],
        "conversation_descriptor_sha256": [
            _sha256_text(canonical_json(item[3])) for item in prepared
        ],
    }


def _batch_audit_record(batch_index, specs, built):
    return {
        "batch_index": batch_index,
        "condition_id": built["condition_id"],
        "condition_ids": [spec["condition_id"] for spec in specs],
        "attempt_ids": built["attempt_ids"],
        "attempt_indexes": built["attempt_indexes"],
        "batch_membership_sha256": built["batch_membership_sha256"],
        "rendered_text_sha256": built["rendered_text_sha256"],
        "conversation_descriptor_sha256": built[
            "conversation_descriptor_sha256"
        ],
        "tensor_fingerprints": built["tensor_fingerprints"],
    }


def audit_stage_model_inputs(
    attempt_plan,
    processor,
    *,
    image_locators,
    config,
    observed_transformers_version,
):
    """Fingerprint all 36 frozen model inputs before loading weights."""

    if observed_transformers_version != config.get("runtime_versions", {}).get(
        "transformers"
    ):
        raise ContractError("stage Transformers version differs from the frozen pin")
    if type(processor).__name__ != config.get("expected_processor_class"):
        raise ContractError("stage processor class differs from the frozen pin")
    if type(processor.tokenizer).__name__ != config.get("expected_tokenizer_class"):
        raise ContractError("stage tokenizer class differs from the frozen pin")
    if _sha256_text(_template_text(processor)) != config.get(
        "chat_template_text_sha256"
    ):
        raise ContractError("stage chat-template text changed")
    if _preprocessor_sha256(processor) != config.get("image_preprocessor_sha256"):
        raise ContractError("stage image preprocessor configuration changed")
    if not isinstance(attempt_plan, list):
        raise ContractError("stage attempt plan must be a list")
    model_specs = [row for row in attempt_plan if row.get("kind") == "model"]
    if len(model_specs) != 36:
        raise ContractError("stage input audit requires exactly 36 model attempts")
    expected = [
        (decision_index * 4 + condition_index, condition)
        for decision_index in range(12)
        for condition_index, condition in enumerate(
            (
                "numbers_only",
                "ordered_stage_visual_pairs",
                "irrelevant_shuffled_images",
            )
        )
    ]
    observed = [
        (row.get("attempt_index"), row.get("condition_id")) for row in model_specs
    ]
    if observed != expected:
        raise ContractError("stage model inputs are not in the frozen decision-major order")
    attempts = []
    for spec in model_specs:
        built = build_stage_model_inputs(
            processor, spec, image_locators=image_locators, config=config
        )
        attempts.append(_audit_record(spec, built))
    batch_specs = _model_batches(attempt_plan, config.get("batch_size"))
    batches = []
    for batch_index, specs in enumerate(batch_specs):
        built = build_stage_model_input_batch(
            processor, specs, image_locators=image_locators, config=config
        )
        batches.append(_batch_audit_record(batch_index, specs, built))
    return {
        "schema_version": "stage_cosmos_input_audit.v2",
        "experiment_id": config["experiment_id"],
        "invocation": "native_template_enable_thinking_omitted",
        "model_attempt_count": 36,
        "batch_size": config["batch_size"],
        "batch_count": len(batches),
        "condition_counts": {
            condition: sum(
                row["condition_id"] == condition for row in model_specs
            )
            for condition in (
                "numbers_only",
                "ordered_stage_visual_pairs",
                "irrelevant_shuffled_images",
            )
        },
        "max_images_per_attempt": 6,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template_text_sha256": _sha256_text(_template_text(processor)),
        "image_preprocessor_sha256": _preprocessor_sha256(processor),
        "model_revision": config["model_revision"],
        "model_view_manifest_sha256": config["model_view_manifest_sha256"],
        "transformers_version": observed_transformers_version,
        "prompt_template_sha256": config["prompt_template_sha256"],
        "attempts": attempts,
        "batches": batches,
    }


def assert_stage_attempt_matches_audit(spec, built, audit):
    """Recheck one live stage input against its frozen processor audit."""

    if not isinstance(audit, dict) or audit.get("model_attempt_count") != 36:
        raise ContractError("tracked stage input audit is invalid")
    matches = [
        row
        for row in audit.get("attempts", [])
        if row.get("attempt_index") == spec.get("attempt_index")
        and row.get("attempt_id") == spec.get("attempt_id")
        and row.get("decision_id") == spec.get("decision_id")
        and row.get("condition_id") == spec.get("condition_id")
    ]
    if len(matches) != 1:
        raise ContractError("stage attempt is missing or duplicated in the input audit")
    live = dict(built)
    live["tensor_fingerprints"] = tensor_fingerprints(built["tensors"])
    if _audit_record(spec, live) != matches[0]:
        raise ContractError("live stage processor input differs from the tracked audit")


def assert_stage_batch_matches_audit(specs, built, audit):
    """Recheck deterministic membership and tensors for one live model batch."""

    if not isinstance(audit, dict) or audit.get("schema_version") != "stage_cosmos_input_audit.v2":
        raise ContractError("tracked stage batch audit is invalid")
    attempt_ids = [spec.get("attempt_id") for spec in specs]
    matches = [
        row for row in audit.get("batches", []) if row.get("attempt_ids") == attempt_ids
    ]
    if len(matches) != 1:
        raise ContractError("stage batch membership is missing or duplicated in the audit")
    live = dict(built)
    live["tensor_fingerprints"] = tensor_fingerprints(
        built["tensors"], expected_batch_size=len(specs)
    )
    batch_index = matches[0]["batch_index"]
    if _batch_audit_record(batch_index, specs, live) != matches[0]:
        raise ContractError("live stage processor batch differs from the tracked audit")


def stage_audit_json_bytes(value: object) -> bytes:
    """Serialize a stage input audit deterministically."""

    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def verify_tracked_stage_input_audit(
    audit_path, attempt_plan, processor, **kwargs
):
    """Hash and recompute a tracked stage audit before loading model weights."""

    path = Path(audit_path)
    config = kwargs["config"]
    if path.is_symlink() or not path.is_file():
        raise ContractError("tracked stage input audit must be a regular file")
    tracked = path.read_bytes()
    if _sha256_bytes(tracked) != config.get("input_audit_sha256"):
        raise ContractError("tracked stage input-audit SHA-256 mismatch")
    recomputed = stage_audit_json_bytes(
        audit_stage_model_inputs(attempt_plan, processor, **kwargs)
    )
    if tracked != recomputed:
        raise ContractError("tracked stage audit differs from recomputed processor inputs")
    return json.loads(tracked)

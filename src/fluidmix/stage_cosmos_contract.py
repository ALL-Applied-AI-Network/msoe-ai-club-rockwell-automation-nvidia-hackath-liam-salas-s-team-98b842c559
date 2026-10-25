"""Frozen configuration contract for the local stage/vial Cosmos3-Nano run."""

from __future__ import annotations

from pathlib import PurePosixPath
import re

from fluidmix.explanation_contract import ContractError


SOURCE_PATHS = (
    "src/fluidmix/explanation_contract.py",
    "src/fluidmix/cosmos_contract_runner.py",
    "src/fluidmix/cosmos_nano_model_view.py",
    "src/fluidmix/cosmos_nano_input_audit.py",
    "src/fluidmix/cosmos_nano_runner.py",
    "src/fluidmix/stage_cosmos_explanation.py",
    "src/fluidmix/stage_cosmos_inputs.py",
    "src/fluidmix/stage_cosmos_runner.py",
    "src/fluidmix/stage_cosmos_scoring.py",
    "src/fluidmix/stage_cosmos_contract.py",
    "src/fluidmix/render_process.py",
    "scripts/audit_stage_cosmos_nano_inputs.py",
    "scripts/run_stage_cosmos_nano_eval.py",
    "scripts/score_stage_cosmos_nano.py",
    "data/cosmos_stage_prompt_v1.txt",
    "tests/test_stage_cosmos_explanation.py",
    "tests/test_stage_cosmos_runner_scoring.py",
)

_CONFIG_KEYS = {
    "schema_version",
    "config_status",
    "condition",
    "experiment_id",
    "freeze_tag",
    "decision_count",
    "conditions",
    "planned_attempts",
    "model_attempts",
    "template_control_attempts",
    "image_invariance_pairs",
    "seed",
    "do_sample",
    "max_new_tokens",
    "batch_size",
    "dtype",
    "max_images_per_attempt",
    "image_width",
    "image_height",
    "model_repository",
    "model_revision",
    "model_type",
    "expected_model_class",
    "expected_processor_class",
    "expected_tokenizer_class",
    "local_files_only",
    "native_eos_token_ids",
    "native_pad_token_id",
    "python_version",
    "platform_machine",
    "runtime_versions",
    "runtime_lock_sha256",
    "runtime_lock_pin_count",
    "allowed_extra_distributions",
    "model_view_manifest_sha256",
    "model_view_file_count",
    "model_view_total_bytes",
    "measured_evidence_sha256",
    "image_locator_sha256",
    "prompt_template_sha256",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "chat_template_generation_suffix",
    "image_preprocessor_sha256",
    "input_audit_sha256",
    "source_sha256",
    "output_path",
}

_FIXED = {
    "schema_version": "stage_cosmos3_nano_eval.v1",
    "config_status": "execution_frozen",
    "condition": "three_stage_three_vial_cross_batch_explanation_v1",
    "experiment_id": "20261024-stage-cosmos3-nano-va-v1",
    "freeze_tag": "explore/stage-cosmos3-nano-va-frozen-20261024",
    "decision_count": 12,
    "conditions": [
        "numbers_only",
        "ordered_stage_visual_pairs",
        "irrelevant_shuffled_images",
        "deterministic_template",
    ],
    "planned_attempts": 48,
    "model_attempts": 36,
    "template_control_attempts": 12,
    "image_invariance_pairs": 12,
    "seed": 0,
    "do_sample": False,
    "max_new_tokens": 384,
    "batch_size": 1,
    "dtype": "bfloat16",
    "max_images_per_attempt": 6,
    "image_width": 640,
    "image_height": 480,
    "model_repository": "nvidia/Cosmos3-Nano",
    "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
    "model_type": "cosmos3_omni",
    "expected_model_class": "Cosmos3OmniForConditionalGeneration",
    "expected_processor_class": "Qwen3VLProcessor",
    "local_files_only": True,
    "native_eos_token_ids": [151645, 151643],
    "native_pad_token_id": 151643,
    "python_version": "3.12.3",
    "platform_machine": "aarch64",
    "runtime_versions": {
        "accelerate": "1.12.0",
        "av": "19.0.2",
        "torch": "2.10.0+cu130",
        "torchvision": "0.25.0+cu130",
        "transformers": "5.17.0",
    },
    "runtime_lock_sha256": (
        "a807a3948f33df0036f7048543df5a2a2cfc219590b97e42f378bbde60bc18b9"
    ),
    "runtime_lock_pin_count": 237,
    "allowed_extra_distributions": {"py-spy": "0.4.2"},
    "model_view_manifest_sha256": (
        "6a79afb5ff8aa352256b1ec029259edcb137fdadd43a1530494d778f17745027"
    ),
    "model_view_file_count": 48,
    "model_view_total_bytes": 31_528_007_898,
    "prompt_template_sha256": (
        "06c7b7597c2b81ed2d4131ace1aefc6482de4b861057cea3e8089f8d9e54b7a7"
    ),
    "chat_template_file_sha256": (
        "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"
    ),
    "chat_template_text_sha256": (
        "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
    ),
    "chat_template_generation_suffix": "<|im_start|>assistant\n",
    "output_path": "outputs/20261024-stage-cosmos3-nano-va-v1/raw-results.json",
}

_DIGEST_FIELDS = (
    "runtime_lock_sha256",
    "model_view_manifest_sha256",
    "measured_evidence_sha256",
    "image_locator_sha256",
    "prompt_template_sha256",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "image_preprocessor_sha256",
    "input_audit_sha256",
)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        and value != "0" * 64
    )


def validate_stage_cosmos_config(value: object) -> dict[str, object]:
    """Reject any config outside the frozen VA stage-explanation treatment."""

    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        extra = sorted(set(value) - _CONFIG_KEYS) if isinstance(value, dict) else []
        missing = sorted(_CONFIG_KEYS - set(value)) if isinstance(value, dict) else []
        raise ContractError(
            f"stage Cosmos config keys mismatch: extra={extra}, missing={missing}"
        )
    for key, expected in _FIXED.items():
        if value[key] != expected or type(value[key]) is not type(expected):
            raise ContractError(f"stage Cosmos config field is not frozen: {key}")
    tokenizer = value["expected_tokenizer_class"]
    if not isinstance(tokenizer, str) or re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", tokenizer
    ) is None:
        raise ContractError("stage expected_tokenizer_class must be a class name")
    for field in _DIGEST_FIELDS:
        if not _is_digest(value[field]):
            raise ContractError(f"stage {field} must be a non-placeholder SHA-256")
    hashes = value["source_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS):
        raise ContractError("stage source_sha256 does not use the frozen source path set")
    if any(not _is_digest(digest) for digest in hashes.values()):
        raise ContractError("stage source_sha256 contains an invalid digest")
    output = PurePosixPath(value["output_path"])
    if output.is_absolute() or ".." in output.parts:
        raise ContractError("stage output_path must remain repository-relative")
    return value


def validate_stage_cosmos_audit_candidate_config(
    value: object,
) -> dict[str, object]:
    """Validate the sole pre-execution state used to compute the audit hash.

    This state cannot pass the execution validator. The audit bytes do not contain
    either field; their resulting digest is inserted before the config is tagged.
    """

    if not isinstance(value, dict):
        raise ContractError("stage audit-candidate config must be an object")
    if (
        value.get("config_status") != "audit_candidate"
        or value.get("input_audit_sha256") != "0" * 64
    ):
        raise ContractError(
            "stage audit candidate requires its explicit status and zero audit hash"
        )
    normalized = dict(value)
    normalized["config_status"] = "execution_frozen"
    normalized["input_audit_sha256"] = "f" * 64
    validate_stage_cosmos_config(normalized)
    return value

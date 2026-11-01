"""Frozen configuration contract for the local Cosmos3-Nano usable treatment."""

import re

from fluidmix.explanation.template.explanation_contract import ContractError


SOURCE_PATHS = (
    "src/fluidmix/explanation/template/explanation_contract.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_model_view.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_input_audit.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_runner.py",
    "src/evaluation/scoring/cosmos_nano_scoring.py",
    "src/fluidmix/explanation/vlm/cosmos_usable_output.py",
    "src/evaluation/scoring/cosmos_usable_scoring.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_usable_contract.py",
    "cli/fluidmix/explanation/vlm/audit_cosmos_nano_usable_inputs.py",
    "cli/fluidmix/explanation/vlm/run_cosmos_nano_usable_eval.py",
    "cli/evaluation/scoring/score_cosmos_nano_usable.py",
    "data/cosmos_operator_prompt_usable_v1.txt",
    "tests/fluidmix/explanation/vlm/test_cosmos_usable_output.py",
    "tests/evaluation/scoring/test_cosmos_usable_scoring.py",
    "tests/fluidmix/explanation/vlm/test_cosmos_usable_preflight.py",
    "docs/cosmos-usable-output.md",
)

_CONFIG_KEYS = {
    "schema_version",
    "config_status",
    "condition",
    "experiment_id",
    "freeze_tag",
    "fixture_count",
    "modes",
    "planned_attempts",
    "model_attempts",
    "template_control_attempts",
    "required_abstention_attempts",
    "required_abstention_model_attempts",
    "required_abstention_control_attempts",
    "image_invariance_pairs",
    "seed",
    "do_sample",
    "max_new_tokens",
    "batch_size",
    "dtype",
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
    "fixture_set_sha256",
    "prompt_sha256",
    "image_sha256",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "chat_template_generation_suffix",
    "input_audit_sha256",
    "source_sha256",
    "image_role",
    "output_path",
}

_FIXED = {
    "schema_version": "cosmos3_nano_usable_eval.v1",
    "config_status": "execution_frozen",
    "condition": "forgiving_labeled_lines_v1",
    "experiment_id": "cosmos3-nano-usable-20261024",
    "freeze_tag": "explore/cosmos3-nano-usable-frozen-20261024",
    "fixture_count": 6,
    "modes": [
        "image_evidence_recipe",
        "evidence_recipe",
        "image_recipe",
        "template_reference",
    ],
    "planned_attempts": 24,
    "model_attempts": 18,
    "template_control_attempts": 6,
    "required_abstention_attempts": 15,
    "required_abstention_model_attempts": 12,
    "required_abstention_control_attempts": 3,
    "image_invariance_pairs": 6,
    "seed": 0,
    "do_sample": False,
    "max_new_tokens": 384,
    "batch_size": 1,
    "dtype": "bfloat16",
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
    "runtime_lock_sha256": "a807a3948f33df0036f7048543df5a2a2cfc219590b97e42f378bbde60bc18b9",
    "runtime_lock_pin_count": 237,
    "allowed_extra_distributions": {"py-spy": "0.4.2"},
    "model_view_file_count": 48,
    "fixture_set_sha256": "e88f6f13476ae89a49d53e0ddfa6cd2be051530c073e946f5e985c16d2f0ee9f",
    "prompt_sha256": "3b5990c541a781d84fba88c7f5ed564ba7dbe73e3f46fd56d5a21814ac294519",
    "image_sha256": "c4886724246a6dbef76c6658b07c8568c5e8e4372f7a9a623021a3ddeb3997af",
    "chat_template_file_sha256": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
    "chat_template_text_sha256": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
    "chat_template_generation_suffix": "<|im_start|>assistant\n",
    "image_role": "irrelevant_fixed_fillcam_distraction",
    "output_path": "outputs/20261024-cosmos3-nano-usable/raw-results.json",
}

_DIGEST_FIELDS = (
    "runtime_lock_sha256",
    "model_view_manifest_sha256",
    "fixture_set_sha256",
    "prompt_sha256",
    "image_sha256",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "input_audit_sha256",
)


def _is_digest(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        and value != "0" * 64
    )


def validate_nano_usable_config(value):
    """Reject any config outside the frozen usable-output treatment."""
    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        extra = sorted(set(value) - _CONFIG_KEYS) if isinstance(value, dict) else []
        missing = sorted(_CONFIG_KEYS - set(value)) if isinstance(value, dict) else []
        raise ContractError(
            f"Nano usable config keys mismatch: extra={extra}, missing={missing}"
        )
    for key, expected in _FIXED.items():
        if value[key] != expected or type(value[key]) is not type(expected):
            raise ContractError(f"Nano usable config field is not frozen: {key}")
    tokenizer = value["expected_tokenizer_class"]
    if not isinstance(tokenizer, str) or re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", tokenizer
    ) is None:
        raise ContractError("expected_tokenizer_class must be a class name")
    total_bytes = value["model_view_total_bytes"]
    if (
        isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or not 31_000_000_000 <= total_bytes <= 32_000_000_000
    ):
        raise ContractError("model_view_total_bytes is outside the reviewed reasoner size")
    for field in _DIGEST_FIELDS:
        if not _is_digest(value[field]):
            raise ContractError(f"{field} must be a non-placeholder lowercase SHA-256")
    hashes = value["source_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS):
        raise ContractError("source_sha256 must use the usable source path set")
    if any(not _is_digest(digest) for digest in hashes.values()):
        raise ContractError("source_sha256 contains an invalid digest")

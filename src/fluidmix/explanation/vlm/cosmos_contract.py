"""Frozen configuration for the Varied V3 Cosmos3-Nano run."""

from __future__ import annotations

from pathlib import PurePosixPath
import re

from fluidmix.explanation.template.explanation_contract import ContractError


SOURCE_PATHS = (
    "src/fluidmix/explanation/template/explanation_contract.py",
    "src/fluidmix/explanation/vlm/cosmos_contract_runner.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_model_view.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_input_audit.py",
    "src/fluidmix/explanation/vlm/cosmos_nano_runner.py",
    "src/fluidmix/explanation/vlm/stage_cosmos_inputs.py",
    "src/fluidmix/explanation/template/varied_explanation.py",
    "src/fluidmix/explanation/vlm/cosmos_contract.py",
    "src/fluidmix/explanation/vlm/cosmos_inputs.py",
    "src/evaluation/scoring/cosmos_plan_artifact.py",
    "src/fluidmix/explanation/vlm/cosmos_runtime.py",
    "src/evaluation/scoring/cosmos_scoring.py",
    "cli/evaluation/scoring/build_v3_cosmos_attempt_plan.py",
    "cli/fluidmix/explanation/vlm/audit_v3_cosmos_nano_inputs.py",
    "cli/fluidmix/explanation/vlm/run_v3_cosmos_nano_eval.py",
    "cli/evaluation/scoring/score_v3_cosmos_nano.py",
    "data/cosmos_f2_varied_prompt_v1.txt",
    "tests/fluidmix/explanation/template/test_varied_explanation.py",
    "tests/fluidmix/explanation/vlm/test_cosmos_contract.py",
    "tests/fluidmix/explanation/vlm/test_cosmos_inputs.py",
    "tests/evaluation/scoring/test_cosmos_plan_artifact.py",
    "tests/fluidmix/explanation/vlm/test_cosmos_runtime.py",
    "tests/evaluation/scoring/test_cosmos_scoring.py",
)


_FIXED = {
    "schema_version": "f2_varied_cosmos3_nano_eval.v1",
    "config_status": "execution_frozen",
    "condition": "varied_v3_three_vial_relative_explanation_v1",
    "experiment_id": "20261027-f2-varied-v3-cosmos3-nano-v1",
    "freeze_tag": "codex/f2-varied-v3-cosmos-nano-frozen-20261027",
    "decision_count": 18,
    "conditions": [
        "numbers_only",
        "ordered_stage_visual_pairs",
        "irrelevant_shuffled_images",
        "deterministic_template",
    ],
    "planned_attempts": 72,
    "model_attempts": 54,
    "template_control_attempts": 18,
    "seed": 0,
    "do_sample": False,
    "max_new_tokens": 384,
    "dtype": "bfloat16",
    "max_images_per_attempt": 6,
    "image_width": 640,
    "image_height": 480,
    "model_repository": "nvidia/Cosmos3-Nano",
    "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
    "model_type": "cosmos3_omni",
    "expected_model_class": "Cosmos3OmniForConditionalGeneration",
    "expected_processor_class": "Qwen3VLProcessor",
    "expected_tokenizer_class": "Qwen2Tokenizer",
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
    "runtime_lock_pin_count": 237,
    "allowed_extra_distributions": {"py-spy": "0.4.2"},
    "model_view_file_count": 48,
    "model_view_total_bytes": 31_528_007_898,
    "output_path": "outputs/20261027-f2-varied-v3-cosmos3-nano-v1/raw-results.json",
}
_DYNAMIC_KEYS = {
    "batch_size",
    "runtime_lock_sha256",
    "model_view_manifest_sha256",
    "decision_artifact_sha256",
    "attempt_plan_sha256",
    "image_locator_sha256",
    "prompt_template_sha256",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "image_preprocessor_sha256",
    "input_audit_sha256",
    "batch_size_smoke_sha256",
    "source_sha256",
}
_CONFIG_KEYS = set(_FIXED) | _DYNAMIC_KEYS
_DIGEST_FIELDS = _DYNAMIC_KEYS - {"batch_size", "source_sha256"}


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        and value != "0" * 64
    )


def validate_v3_cosmos_config(value):
    """Reject any local Nano treatment outside the frozen V3 contract."""

    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        extra = sorted(set(value) - _CONFIG_KEYS) if isinstance(value, dict) else []
        missing = sorted(_CONFIG_KEYS - set(value)) if isinstance(value, dict) else []
        raise ContractError(f"V3 Cosmos config keys drifted: extra={extra}, missing={missing}")
    for key, expected in _FIXED.items():
        if value[key] != expected or type(value[key]) is not type(expected):
            raise ContractError(f"V3 Cosmos config field is not frozen: {key}")
    batch_size = value["batch_size"]
    if type(batch_size) is not int or batch_size < 1 or 18 % batch_size != 0:
        raise ContractError("V3 batch_size must be a positive divisor of 18")
    for field in _DIGEST_FIELDS:
        if not _digest(value[field]):
            raise ContractError(f"V3 {field} must be a non-placeholder SHA-256")
    sources = value["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_PATHS):
        raise ContractError("V3 source hash path set is not frozen")
    if any(not _digest(digest) for digest in sources.values()):
        raise ContractError("V3 source hashes contain an invalid digest")
    output = PurePosixPath(value["output_path"])
    if output.is_absolute() or ".." in output.parts:
        raise ContractError("V3 output path must be repository-relative")
    return value


def validate_v3_cosmos_audit_candidate_config(value):
    """Validate the sole pre-audit form with a zero audit digest."""

    if not isinstance(value, dict) or value.get("config_status") != "audit_candidate":
        raise ContractError("V3 audit candidate status is invalid")
    if value.get("input_audit_sha256") != "0" * 64:
        raise ContractError("V3 audit candidate requires a zero audit digest")
    normalized = dict(value)
    normalized["config_status"] = "execution_frozen"
    normalized["input_audit_sha256"] = "f" * 64
    validate_v3_cosmos_config(normalized)
    return value

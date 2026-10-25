from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_nano_usable_contract import (  # noqa: E402
    SOURCE_PATHS,
    validate_nano_usable_config,
)
from fluidmix.explanation_contract import ContractError  # noqa: E402


def valid_usable_config():
    digest = "1" * 64
    return {
        "schema_version": "cosmos3_nano_usable_eval.v1",
        "config_status": "execution_frozen",
        "condition": "forgiving_labeled_lines_v1",
        "experiment_id": "cosmos3-nano-usable-20261024",
        "freeze_tag": "explore/cosmos3-nano-usable-frozen-20261024",
        "fixture_count": 6,
        "modes": ["image_evidence_recipe", "evidence_recipe", "image_recipe", "template_reference"],
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
        "expected_tokenizer_class": "Qwen2TokenizerFast",
        "local_files_only": True,
        "native_eos_token_ids": [151645, 151643],
        "native_pad_token_id": 151643,
        "python_version": "3.12.3",
        "platform_machine": "aarch64",
        "runtime_versions": {"accelerate": "1.12.0", "av": "19.0.2", "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "transformers": "5.17.0"},
        "runtime_lock_sha256": "a807a3948f33df0036f7048543df5a2a2cfc219590b97e42f378bbde60bc18b9",
        "runtime_lock_pin_count": 237,
        "allowed_extra_distributions": {"py-spy": "0.4.2"},
        "model_view_manifest_sha256": digest,
        "model_view_file_count": 48,
        "model_view_total_bytes": 31_500_000_000,
        "fixture_set_sha256": "e88f6f13476ae89a49d53e0ddfa6cd2be051530c073e946f5e985c16d2f0ee9f",
        "prompt_sha256": "3b5990c541a781d84fba88c7f5ed564ba7dbe73e3f46fd56d5a21814ac294519",
        "image_sha256": "c4886724246a6dbef76c6658b07c8568c5e8e4372f7a9a623021a3ddeb3997af",
        "chat_template_file_sha256": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
        "chat_template_text_sha256": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
        "chat_template_generation_suffix": "<|im_start|>assistant\n",
        "input_audit_sha256": digest,
        "source_sha256": {path: digest for path in SOURCE_PATHS},
        "image_role": "irrelevant_fixed_fillcam_distraction",
        "output_path": "outputs/20261024-cosmos3-nano-usable/raw-results.json",
    }


def test_usable_config_freezes_prompt_matrix_and_matched_decoding():
    config = valid_usable_config()
    validate_nano_usable_config(config)
    assert config["condition"] == "forgiving_labeled_lines_v1"
    assert config["batch_size"] == 1
    assert config["max_new_tokens"] == 384
    assert config["do_sample"] is False


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("prompt_sha256", "2" * 64),
        ("batch_size", 2),
        ("max_new_tokens", 385),
        ("do_sample", True),
        ("condition", "json"),
        ("output_path", "outputs/20261024-cosmos3-nano-matched/raw-results.json"),
    ],
)
def test_usable_config_rejects_post_freeze_treatment_changes(field, wrong):
    config = valid_usable_config()
    config[field] = wrong
    with pytest.raises(ContractError):
        validate_nano_usable_config(config)


def test_usable_config_rejects_untracked_parser_or_runner_source():
    config = valid_usable_config()
    config["source_sha256"].pop("src/fluidmix/cosmos_usable_output.py")
    with pytest.raises(ContractError):
        validate_nano_usable_config(config)


@pytest.mark.parametrize(
    "script",
    [
        "audit_cosmos_nano_usable_inputs.py",
        "run_cosmos_nano_usable_eval.py",
        "score_cosmos_nano_usable.py",
    ],
)
def test_usable_cli_help_does_not_load_model_or_require_gpu(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

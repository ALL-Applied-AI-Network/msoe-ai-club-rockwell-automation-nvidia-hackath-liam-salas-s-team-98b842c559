"""Frozen configuration boundary for the Varied V3 Nano run."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _config() -> dict[str, object]:
    from fluidmix.v3_cosmos_contract import SOURCE_PATHS

    return {
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
        "runtime_lock_sha256": "a" * 64,
        "runtime_lock_pin_count": 237,
        "allowed_extra_distributions": {"py-spy": "0.4.2"},
        "model_view_manifest_sha256": "b" * 64,
        "model_view_file_count": 48,
        "model_view_total_bytes": 31_528_007_898,
        "decision_artifact_sha256": "c" * 64,
        "attempt_plan_sha256": "d" * 64,
        "image_locator_sha256": "e" * 64,
        "prompt_template_sha256": "f" * 64,
        "chat_template_file_sha256": "1" * 64,
        "chat_template_text_sha256": "2" * 64,
        "image_preprocessor_sha256": "3" * 64,
        "input_audit_sha256": "4" * 64,
        "batch_size_smoke_sha256": "5" * 64,
        "source_sha256": {path: "6" * 64 for path in SOURCE_PATHS},
        "output_path": "outputs/20261027-f2-varied-v3-cosmos3-nano-v1/raw-results.json",
    }


def test_accepts_exact_pinned_config_and_any_positive_batch_divisor_of_18():
    from fluidmix.v3_cosmos_contract import validate_v3_cosmos_config

    for batch_size in (1, 2, 3, 6, 9, 18):
        value = _config()
        value["batch_size"] = batch_size
        assert validate_v3_cosmos_config(value) is value


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "main"),
        ("planned_attempts", 71),
        ("do_sample", True),
        ("max_new_tokens", 385),
        ("batch_size", 4),
        ("decision_artifact_sha256", "0" * 64),
        ("output_path", "../raw-results.json"),
    ],
)
def test_rejects_runtime_or_provenance_drift(field, value):
    from fluidmix.v3_cosmos_contract import validate_v3_cosmos_config

    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_v3_cosmos_config(config)


def test_rejects_source_path_set_drift():
    from fluidmix.v3_cosmos_contract import validate_v3_cosmos_config

    config = _config()
    config["source_sha256"] = deepcopy(config["source_sha256"])
    config["source_sha256"].pop(next(iter(config["source_sha256"])))
    with pytest.raises(ValueError, match="source"):
        validate_v3_cosmos_config(config)


def test_audit_candidate_allows_only_zero_audit_hash_before_freeze():
    from fluidmix.v3_cosmos_contract import (
        validate_v3_cosmos_audit_candidate_config,
    )

    config = _config()
    config["config_status"] = "audit_candidate"
    config["input_audit_sha256"] = "0" * 64
    assert validate_v3_cosmos_audit_candidate_config(config) is config
    config["input_audit_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="zero audit"):
        validate_v3_cosmos_audit_candidate_config(config)

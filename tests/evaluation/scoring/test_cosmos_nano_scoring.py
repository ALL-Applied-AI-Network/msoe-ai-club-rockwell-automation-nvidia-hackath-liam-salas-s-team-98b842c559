import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_contract_runner import build_attempt_plan  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_runner import execute_nano_attempt_plan  # noqa: E402
from evaluation.scoring.cosmos_nano_scoring import (  # noqa: E402
    PROVENANCE_KEYS,
    score_nano_raw_results,
)
from fluidmix.explanation.template.explanation_contract import ContractError, canonical_json  # noqa: E402
from tests.fluidmix.explanation.vlm.test_cosmos_nano_preflight import valid_config  # noqa: E402


def _inputs():
    fixtures = json.loads((ROOT / "data/cosmos_contract_fixtures_v1.json").read_text())
    oracle = json.loads((ROOT / "data/cosmos_contract_oracle_v1.json").read_text())
    prompt = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text()
    config = valid_config()
    plan = build_attempt_plan(fixtures, config, prompt)
    return config, oracle, plan


def _perfect_attempts():
    config, oracle, plan = _inputs()

    def infer(spec):
        expected_key = "image_recipe" if spec["mode"] == "image_recipe" else "full"
        response = canonical_json(
            oracle["expected_by_record"][spec["record_id"]][expected_key]
        )
        token_ids = [151643]
        return {
            "response_raw": response,
            "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "generated_suffix_token_ids": token_ids,
            "generated_suffix_sha256": hashlib.sha256(b"[151643]").hexdigest(),
            "generated_token_count": 1,
            "terminal_token_id": 151643,
            "terminal_token_is_eos": True,
            "length_cap_reached": False,
            "truncated_without_eos": False,
        }

    return execute_nano_attempt_plan(plan, infer=infer)


def _provenance(config):
    value = {key: None for key in PROVENANCE_KEYS}
    value.update(
        freeze_tag=config["freeze_tag"],
        head_commit="a" * 40,
        git_clean=True,
        config_sha256="b" * 64,
        fixture_set_sha256=config["fixture_set_sha256"],
        prompt_sha256=config["prompt_sha256"],
        image_sha256=config["image_sha256"],
        model_view_manifest_sha256=config["model_view_manifest_sha256"],
        runtime_lock_sha256=config["runtime_lock_sha256"],
        input_audit_sha256=config["input_audit_sha256"],
        source_sha256=config["source_sha256"],
        model_repository=config["model_repository"],
        model_revision=config["model_revision"],
        seed=0,
        do_sample=False,
        max_new_tokens=384,
        batch_size=1,
        dtype="bfloat16",
        python="3.12.3",
        platform_machine="aarch64",
        runtime_versions={"locked": {}, "extras": {"py-spy": "0.4.2"}},
        processor_class="Qwen3VLProcessor",
        tokenizer_class="Qwen2TokenizerFast",
        chat_template_file_sha256=config["chat_template_file_sha256"],
        chat_template_text_sha256=config["chat_template_text_sha256"],
        image_preprocessor_sha256="c" * 64,
        model_class="Cosmos3OmniForConditionalGeneration",
        attention_implementation="sdpa",
        generation_config={"eos_token_id": [151645, 151643], "pad_token_id": 151643},
        parameter_devices=["cuda:0"],
        model_load_seconds=1.0,
        model_load_error=None,
        peak_memory_allocated_bytes=1,
        peak_memory_reserved_bytes=2,
        device_name="NVIDIA GB10",
        compute_capability=[12, 1],
        cuda_version="13.0",
        driver_version="test",
        native_eos_token_ids=[151645, 151643],
        native_pad_token_id=151643,
        started_at_utc="2026-10-24T00:00:00+00:00",
        finished_at_utc="2026-10-24T00:01:00+00:00",
    )
    assert set(value) == PROVENANCE_KEYS
    return value


def _perfect_raw():
    config, oracle, _ = _inputs()
    return config, oracle, {
        "schema_version": "cosmos3_nano_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": _provenance(config),
        "attempts": _perfect_attempts(),
    }


def test_perfect_matrix_reports_model_and_control_denominators_separately():
    config, oracle, raw = _perfect_raw()

    score = score_nano_raw_results(raw, config, oracle)

    assert score["matrix"] == {
        "planned": 24,
        "attempted": 24,
        "completed": 24,
        "errors": 0,
        "complete": True,
        "model": {"planned": 18, "completed": 18, "errors": 0},
        "control": {"planned": 6, "completed": 6, "errors": 0},
    }
    assert score["strict_schema"] == {"passed": 24, "total": 24}
    assert score["model_policy"] == {"passed": 18, "total": 18}
    assert score["template_control"] == {"passed": 6, "total": 6}
    assert score["required_abstention"] == {"passed": 15, "total": 15}
    assert score["required_abstention_model"] == {"passed": 12, "total": 12}
    assert score["required_abstention_control"] == {"passed": 3, "total": 3}
    assert score["generation_cap_reaches"] == {"reached": 0, "total": 18}
    assert score["truncated_without_eos"] == {"count": 0, "total": 18}


def test_fenced_json_and_inference_error_fail_without_repair_or_denominator_loss():
    config, oracle, raw = _perfect_raw()
    first = raw["attempts"][0]
    first["response_raw"] = "```json\n{}\n```"
    first["response_utf8_sha256"] = hashlib.sha256(
        first["response_raw"].encode()
    ).hexdigest()
    second = raw["attempts"][1]
    second.update(
        attempt_status="error",
        response_raw=None,
        response_utf8_sha256=None,
        generated_suffix_token_ids=None,
        generated_suffix_sha256=None,
        generated_token_count=None,
        terminal_token_id=None,
        terminal_token_is_eos=None,
        length_cap_reached=None,
        truncated_without_eos=None,
        error_type="RuntimeError",
        error="retained",
    )

    score = score_nano_raw_results(raw, config, oracle)

    assert score["matrix"]["model"] == {"planned": 18, "completed": 17, "errors": 1}
    assert score["strict_schema"] == {"passed": 22, "total": 24}
    assert score["model_policy"] == {"passed": 16, "total": 18}


def test_incomplete_or_internally_inconsistent_raw_artifact_is_not_scoreable():
    config, oracle, raw = _perfect_raw()
    raw["attempts"].pop()
    with pytest.raises(ContractError):
        score_nano_raw_results(raw, config, oracle)

    config, oracle, raw = _perfect_raw()
    raw["attempts"][0]["length_cap_reached"] = True
    with pytest.raises(ContractError):
        score_nano_raw_results(raw, config, oracle)


def test_claimed_model_load_failure_requires_all_eighteen_model_error_rows():
    config, oracle, raw = _perfect_raw()
    raw["provenance"]["model_load_error"] = {
        "type": "OSError",
        "message": "load failed",
    }
    raw["provenance"]["model_class"] = None
    raw["provenance"]["parameter_devices"] = []
    raw["provenance"]["generation_config"] = None

    with pytest.raises(ContractError):
        score_nano_raw_results(raw, config, oracle)


def test_oracle_hash_is_fixed_and_runner_sources_have_no_oracle_path():
    assert hashlib.sha256(
        (ROOT / "data/cosmos_contract_oracle_v1.json").read_bytes()
    ).hexdigest() == "cb2d45ca1fce48e7603a6cde52ff0cbb1bdad11e7b2285ad01b80cbd232faab9"
    for relative in (
        "src/fluidmix/explanation/vlm/cosmos_contract_runner.py",
        "src/fluidmix/explanation/vlm/cosmos_nano_runner.py",
        "cli/fluidmix/explanation/vlm/run_cosmos_large_eval.py",
    ):
        path = ROOT / relative
        if path.exists():
            assert "oracle" not in path.read_text().lower()

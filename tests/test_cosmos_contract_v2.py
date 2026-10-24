import hashlib
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.explanation_contract import ContractError, canonical_json, validate_eval_config
from fluidmix.cosmos_contract_runner import build_attempt_plan
from fluidmix.cosmos_input_audit import (
    audit_json_bytes,
    audit_model_inputs,
    verify_tracked_input_audit,
)
from fluidmix.cosmos_v2_scoring import score_v2_raw_results


ROOT = Path(__file__).resolve().parents[1]


def load_config(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def test_v2_config_changes_only_declared_treatment_and_administration():
    v1 = load_config("cosmos_contract_eval_v1.json")
    v2 = load_config("cosmos_contract_eval_v2.json")

    validate_eval_config(v1)
    validate_eval_config(v2)
    inherited = set(v1) - {
        "schema_version",
        "experiment_id",
        "freeze_tag",
        "output_path",
    }
    assert {key: v2[key] for key in inherited} == {
        key: v1[key] for key in inherited
    }
    assert v2["schema_version"] == "operator_contract_eval.v2"
    assert v2["experiment_id"] == "cosmos-contract-v2-no-thinking-20261024"
    assert v2["freeze_tag"] == "explore/cosmos-contract-v2-frozen-20261024"
    assert v2["output_path"] == "outputs/20261024-cosmos-contract-v2/raw-results.json"
    assert v2["v1_freeze_tag"] == "explore/cosmos-contract-v1-frozen-20261024"
    assert v2["enable_thinking"] is False


def test_v2_config_binds_the_audit_template_and_unchanged_scorer():
    config = load_config("cosmos_contract_eval_v2.json")
    expected = {
        "scoring_module_sha256": ROOT / "src/fluidmix/explanation_scoring.py",
        "scoring_script_sha256": ROOT / "scripts/score_cosmos_contract_v2.py",
        "v2_scoring_module_sha256": ROOT / "src/fluidmix/cosmos_v2_scoring.py",
        "contract_module_sha256": ROOT / "src/fluidmix/explanation_contract.py",
    }

    assert config["chat_template_sha256"] == (
        "7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8"
    )
    for key, path in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]
    audit_path = ROOT / "data/cosmos_contract_v2_input_audit.json"
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == config["input_audit_sha256"]


def test_tracked_processor_audit_has_the_full_predeclared_pair_relations():
    config = load_config("cosmos_contract_eval_v2.json")
    audit = json.loads(
        (ROOT / "data/cosmos_contract_v2_input_audit.json").read_text(encoding="utf-8")
    )
    expected_pairs = {
        (f"r{index:03d}", mode)
        for index in range(1, 7)
        for mode in ("image_evidence_recipe", "evidence_recipe", "image_recipe")
    }

    assert audit["model_attempt_count"] == len(audit["attempts"]) == 18
    assert {(row["record_id"], row["mode"]) for row in audit["attempts"]} == expected_pairs
    assert audit["fixture_set_sha256"] == config["fixture_set_sha256"]
    assert audit["prompt_template_sha256"] == config["prompt_sha256"]
    assert audit["image_sha256"] == config["image_sha256"]
    assert audit["chat_template_sha256"] == config["chat_template_sha256"]
    assert audit["model_manifest_sha256"] == config["model_manifest_sha256"]
    assert audit["processor_class"].endswith(".Cosmos3EdgeProcessor")
    assert audit["tokenizer_class"].endswith(".TokenizersBackend")
    for row in audit["attempts"]:
        assert row["v1_default"]["token_count"] == row["v2_false"]["token_count"]
        assert row["common_token_prefix_count"] == row["v1_default"]["token_count"] - 1
        assert row["v1_default_token_suffix"] == [1010]
        assert row["v2_false_token_suffix"] == [13]
        assert row["differing_render_suffix_v1_default"] == "\n"
        assert row["differing_render_suffix_v2_false"] == "</think>"
        expected_order = ["image", "text"] if row["image_attached"] else ["text"]
        assert row["content_order"] == expected_order
        assert row["image_sha256"] == (
            config["image_sha256"] if row["image_attached"] else None
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enable_thinking", True),
        ("max_new_tokens", 385),
        ("v1_freeze_tag", "other"),
        ("input_audit_sha256", "not-a-digest"),
        ("scoring_module_sha256", "a" * 64),
    ],
)
def test_v2_config_rejects_any_unfrozen_factor(key, value):
    config = deepcopy(load_config("cosmos_contract_eval_v2.json"))
    config[key] = value

    with pytest.raises(ContractError):
        validate_eval_config(config)


class _TokenBatch(dict):
    pass


class FakeProcessor:
    def __init__(
        self,
        *,
        ignore_flag=False,
        change_prior_token=False,
        omitted_is_not_true=False,
    ):
        self.ignore_flag = ignore_flag
        self.change_prior_token = change_prior_token
        self.omitted_is_not_true = omitted_is_not_true
        self.calls = []
        self.tokenizer = object()

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs.copy()))
        omitted = "enable_thinking" not in kwargs
        flag = kwargs.get("enable_thinking", True)
        if self.ignore_flag:
            flag = True
        if self.omitted_is_not_true and omitted:
            flag = False
        prompt = messages[0]["content"][-1]["text"]
        has_image = messages[0]["content"][0]["type"] == "image"
        prefix = f"fixture={hashlib.sha256(prompt.encode()).hexdigest()};image={has_image};"
        suffix = "<think>\n" if flag else "<think></think>"
        if not kwargs["tokenize"]:
            return prefix + suffix
        base = [1, len(prompt), 2 if has_image else 3]
        if self.change_prior_token and flag is False:
            base[1] += 1
        return _TokenBatch(input_ids=[base + ([1010] if flag else [13])])


def v2_plan():
    fixtures = json.loads(
        (ROOT / "data/cosmos_contract_fixtures_v1.json").read_text(encoding="utf-8")
    )
    prompt = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text(encoding="utf-8")
    config = load_config("cosmos_contract_eval_v2.json")
    return build_attempt_plan(fixtures, config, prompt), config


def run_fake_audit(processor):
    plan, config = v2_plan()
    return audit_model_inputs(
        plan,
        processor,
        image_path=Path("/ignored/fixed.png"),
        config=config,
        observed_transformers_version="5.17.0",
    )


def test_audit_covers_all_model_inputs_and_records_terminal_factor():
    processor = FakeProcessor()

    audit = run_fake_audit(processor)

    assert audit["model_attempt_count"] == 18
    assert len(audit["attempts"]) == 18
    assert len({(row["record_id"], row["mode"]) for row in audit["attempts"]}) == 18
    assert all(
        row["v1_default"]["token_count"] == row["v2_false"]["token_count"]
        for row in audit["attempts"]
    )
    assert all(row["v1_default_token_suffix"] == [1010] for row in audit["attempts"])
    assert all(row["v2_false_token_suffix"] == [13] for row in audit["attempts"])
    image_row = next(row for row in audit["attempts"] if row["image_attached"])
    text_row = next(row for row in audit["attempts"] if not row["image_attached"])
    assert image_row["content_order"] == ["image", "text"]
    assert text_row["content_order"] == ["text"]
    assert processor.calls
    assert {
        call[1].get("enable_thinking", "omitted")
        for call in processor.calls
    } == {"omitted", False}


def test_audit_fails_if_template_silently_ignores_flag():
    with pytest.raises(ContractError, match="suffix"):
        run_fake_audit(FakeProcessor(ignore_flag=True))


def test_audit_uses_actual_v1_omission_not_explicit_true():
    with pytest.raises(ContractError, match="v1-default.*suffix"):
        run_fake_audit(FakeProcessor(omitted_is_not_true=True))


def test_audit_fails_if_flag_changes_a_prior_token():
    with pytest.raises(ContractError, match="terminal token"):
        run_fake_audit(FakeProcessor(change_prior_token=True))


def test_audit_rejects_an_unobserved_transformers_version():
    plan, config = v2_plan()

    with pytest.raises(ContractError, match="Transformers version"):
        audit_model_inputs(
            plan,
            FakeProcessor(),
            image_path=Path("/ignored/fixed.png"),
            config=config,
            observed_transformers_version="5.16.0",
        )


def test_audit_serialization_is_canonical_and_deterministic():
    first = audit_json_bytes(run_fake_audit(FakeProcessor()))
    second = audit_json_bytes(json.loads(first))

    assert first == second
    assert first.endswith(b"\n")


def test_tracked_audit_is_hashed_and_recomputed_before_use(tmp_path):
    plan, config = v2_plan()
    processor = FakeProcessor()
    payload = audit_json_bytes(
        audit_model_inputs(
            plan,
            processor,
            image_path=Path("/ignored/fixed.png"),
            config=config,
            observed_transformers_version="5.17.0",
        )
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(payload)
    config = deepcopy(config)
    config["input_audit_sha256"] = hashlib.sha256(payload).hexdigest()

    assert verify_tracked_input_audit(
        audit_path,
        plan,
        FakeProcessor(),
        image_path=Path("/ignored/fixed.png"),
        config=config,
        observed_transformers_version="5.17.0",
    ) == config["input_audit_sha256"]

    changed = json.loads(payload)
    changed["attempts"][0]["prompt_utf8_bytes"] += 1
    changed_payload = audit_json_bytes(changed)
    audit_path.write_bytes(changed_payload)
    config["input_audit_sha256"] = hashlib.sha256(changed_payload).hexdigest()
    with pytest.raises(ContractError, match="recomputed"):
        verify_tracked_input_audit(
            audit_path,
            plan,
            FakeProcessor(),
            image_path=Path("/ignored/fixed.png"),
            config=config,
            observed_transformers_version="5.17.0",
        )


def test_tracked_audit_rejects_symlink(tmp_path):
    plan, config = v2_plan()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "audit.json"
    link.symlink_to(target.name)

    with pytest.raises(ContractError, match="regular file"):
        verify_tracked_input_audit(
            link,
            plan,
            FakeProcessor(),
            image_path=Path("/ignored/fixed.png"),
            config=config,
            observed_transformers_version="5.17.0",
        )


def test_audit_cli_is_processor_only_and_cannot_read_oracle_or_override_output():
    script = ROOT / "scripts/audit_cosmos_contract_v2_inputs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    source = script.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert "--model-dir" in result.stdout
    assert "--model-manifest" in result.stdout
    assert "--runtime-lock" in result.stdout
    assert "--image" in result.stdout
    assert "--oracle" not in result.stdout
    assert "--out" not in result.stdout
    assert "AutoModel" not in source
    assert ".generate(" not in source
    assert "cosmos_contract_oracle" not in source
    assert "requests" not in source
    assert "http://" not in source and "https://" not in source


def test_v2_runner_recomputes_audit_before_model_load_and_has_no_oracle_access():
    script = ROOT / "scripts/run_cosmos_contract_eval_v2.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    source = script.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stderr
    for option in ("--model-dir", "--model-manifest", "--runtime-lock", "--image"):
        assert option in result.stdout
    assert "--oracle" not in result.stdout
    assert "--out" not in result.stdout
    assert "cosmos_contract_oracle" not in source
    assert "enable_thinking=config[\"enable_thinking\"]" in source
    assert source.index("verify_tracked_input_audit(") < source.index(
        "AutoModelForImageTextToText.from_pretrained("
    )
    for field in (
        "v1_freeze_tag",
        "enable_thinking",
        "input_audit_sha256",
        "chat_template_sha256",
    ):
        assert f'"{field}"' in source
    assert "requests" not in source
    assert "http://" not in source and "https://" not in source


def perfect_v2_raw_results():
    config = load_config("cosmos_contract_eval_v2.json")
    oracle = json.loads(
        (ROOT / "data/cosmos_contract_oracle_v1.json").read_text(encoding="utf-8")
    )
    attempts = []
    for record_id in sorted(oracle["expected_by_record"]):
        full_prompt_hash = f"{int(record_id[1:]):064x}"
        for mode in config["modes"]:
            expected_key = "image_recipe" if mode == "image_recipe" else "full"
            attempts.append({
                "attempt_index": len(attempts),
                "record_id": record_id,
                "mode": mode,
                "attempt_status": "completed",
                "response_raw": canonical_json(
                    oracle["expected_by_record"][record_id][expected_key]
                ),
                "prompt_sha256": None if mode == "template_reference" else (
                    "f" * 64 if mode == "image_recipe" else full_prompt_hash
                ),
                "image_attached": mode in {"image_evidence_recipe", "image_recipe"},
                "elapsed_seconds": 0.0,
                "generated_token_count": None if mode == "template_reference" else 1,
                "error_type": None,
                "error": None,
            })
    raw = {
        "schema_version": "operator_contract_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": {
            "freeze_tag": config["freeze_tag"],
            "head_commit": "0" * 40,
            "git_clean": True,
            "fixture_set_sha256": config["fixture_set_sha256"],
            "prompt_sha256": config["prompt_sha256"],
            "image_sha256": config["image_sha256"],
            "model_manifest_sha256": config["model_manifest_sha256"],
            "runtime_lock_sha256": config["runtime_lock_sha256"],
            "model_revision": config["model_revision"],
            "seed": config["seed"],
            "do_sample": config["do_sample"],
            "max_new_tokens": config["max_new_tokens"],
            "dtype": config["dtype"],
            "python": config["python_version"],
            "versions": config["runtime_versions"],
            "platform_machine": config["platform_machine"],
            "device_name": "test-device",
            "compute_capability": [0, 0],
            "cuda_version": "test",
            "model_load_seconds": 0.0,
            "model_load_error": None,
            "started_at_utc": "2026-10-24T00:00:00+00:00",
            "finished_at_utc": "2026-10-24T00:00:01+00:00",
            "v1_freeze_tag": config["v1_freeze_tag"],
            "enable_thinking": config["enable_thinking"],
            "input_audit_sha256": config["input_audit_sha256"],
            "chat_template_sha256": config["chat_template_sha256"],
        },
        "attempts": attempts,
    }
    return raw, config, oracle


def test_unchanged_strict_scorer_accepts_the_v2_matrix_and_scores_cap():
    raw, config, oracle = perfect_v2_raw_results()

    score = score_v2_raw_results(raw, config, oracle)

    assert score["schema_version"] == "operator_contract_score.v2"
    assert score["matrix"] == {
        "planned": 24,
        "attempted": 24,
        "completed": 24,
        "errors": 0,
        "complete": True,
    }
    assert score["strict_schema"] == {"passed": 24, "total": 24}
    assert score["required_abstention"] == {"passed": 15, "total": 15}
    assert score["image_invariance"] == {"passed": 6, "total": 6}
    assert score["template_control"] == {"passed": 6, "total": 6}
    assert score["generation_cap_reaches"] == {"reached": 0, "total": 18}


def test_v2_cap_metric_keeps_errors_in_its_eighteen_attempt_denominator():
    raw, config, oracle = perfect_v2_raw_results()
    model_attempts = [
        attempt for attempt in raw["attempts"]
        if attempt["mode"] != "template_reference"
    ]
    model_attempts[0]["generated_token_count"] = config["max_new_tokens"]
    model_attempts[1].update({
        "attempt_status": "error",
        "response_raw": None,
        "generated_token_count": None,
        "error_type": "RuntimeError",
        "error": "retained failure",
    })

    score = score_v2_raw_results(raw, config, oracle)

    assert score["generation_cap_reaches"] == {"reached": 1, "total": 18}


def test_v2_cap_metric_can_count_all_eighteen_model_attempts():
    raw, config, oracle = perfect_v2_raw_results()
    for attempt in raw["attempts"]:
        if attempt["mode"] != "template_reference":
            attempt["generated_token_count"] = config["max_new_tokens"]

    score = score_v2_raw_results(raw, config, oracle)

    assert score["generation_cap_reaches"] == {"reached": 18, "total": 18}


def test_v2_scorer_rejects_completed_model_tokens_above_cap():
    raw, config, oracle = perfect_v2_raw_results()
    raw["attempts"][0]["generated_token_count"] = config["max_new_tokens"] + 1

    with pytest.raises(ContractError, match="exceeds frozen generation cap"):
        score_v2_raw_results(raw, config, oracle)


def test_v2_scorer_rejects_treatment_provenance_mismatch():
    raw, config, oracle = perfect_v2_raw_results()
    raw["provenance"]["enable_thinking"] = True

    with pytest.raises(ContractError, match="v2 provenance mismatch"):
        score_v2_raw_results(raw, config, oracle)


def test_v2_scoring_cli_verifies_sources_and_has_no_repair_option():
    script = ROOT / "scripts/score_cosmos_contract_v2.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    source = script.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stderr
    for option in ("--raw-results", "--config", "--oracle", "--out"):
        assert option in result.stdout
    assert "--repair" not in result.stdout
    for field in (
        "input_audit_sha256",
        "scoring_module_sha256",
        "scoring_script_sha256",
        "v2_scoring_module_sha256",
        "contract_module_sha256",
    ):
        assert field in source


def test_v2_scoring_cli_writes_once_with_verified_sources(tmp_path):
    raw, _, _ = perfect_v2_raw_results()
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "score.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts/score_cosmos_contract_v2.py"),
        "--raw-results", str(raw_path),
        "--config", str(ROOT / "data/cosmos_contract_eval_v2.json"),
        "--oracle", str(ROOT / "data/cosmos_contract_oracle_v1.json"),
        "--out", str(out_path),
    ]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    score = json.loads(out_path.read_text(encoding="utf-8"))
    assert score["generation_cap_reaches"] == {"reached": 0, "total": 18}
    assert second.returncode != 0
    assert score == json.loads(out_path.read_text(encoding="utf-8"))

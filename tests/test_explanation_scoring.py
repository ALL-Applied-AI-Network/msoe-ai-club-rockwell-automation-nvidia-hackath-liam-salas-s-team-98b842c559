import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.explanation_contract import ContractError, canonical_json
from fluidmix.explanation_scoring import score_raw_results


ROOT = Path(__file__).resolve().parents[1]


def load_inputs():
    config = json.loads((ROOT / "data/cosmos_contract_eval_v1.json").read_text())
    oracle = json.loads((ROOT / "data/cosmos_contract_oracle_v1.json").read_text())
    return config, oracle


def perfect_raw_results():
    config, oracle = load_inputs()
    attempts = []
    attempt_index = 0
    for record_id in sorted(oracle["expected_by_record"]):
        full_prompt_hash = f"{int(record_id[1:]):064x}"
        for mode in config["modes"]:
            expected_key = "image_recipe" if mode == "image_recipe" else "full"
            expected = oracle["expected_by_record"][record_id][expected_key]
            attempts.append({
                "attempt_index": attempt_index,
                "record_id": record_id,
                "mode": mode,
                "attempt_status": "completed",
                "response_raw": canonical_json(expected),
                "prompt_sha256": None if mode == "template_reference" else (
                    "f" * 64 if mode == "image_recipe" else full_prompt_hash
                ),
                "image_attached": mode in {"image_evidence_recipe", "image_recipe"},
                "elapsed_seconds": 0.0,
                "generated_token_count": None if mode == "template_reference" else 1,
                "error_type": None,
                "error": None,
            })
            attempt_index += 1
    return {
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
        },
        "attempts": attempts,
    }


def test_perfect_control_matrix_has_only_fixed_denominator_passes():
    config, oracle = load_inputs()

    score = score_raw_results(perfect_raw_results(), config, oracle)

    assert score["matrix"] == {
        "planned": 24,
        "attempted": 24,
        "completed": 24,
        "errors": 0,
        "complete": True,
    }
    for metric in (
        "strict_schema",
        "policy",
        "evidence_ids",
        "numeric_fidelity",
        "advisory_action",
        "unsupported_claim",
    ):
        assert score[metric] == {"passed": 24, "total": 24}
    assert score["required_abstention"] == {"passed": 15, "total": 15}
    assert score["image_invariance"] == {"passed": 6, "total": 6}
    assert score["template_control"] == {"passed": 6, "total": 6}
    assert score["failures"] == []


def test_invalid_json_fails_all_attempt_denominators_without_repair():
    config, oracle = load_inputs()
    raw = perfect_raw_results()
    raw["attempts"][0]["response_raw"] = "```json\n{}\n```"

    score = score_raw_results(raw, config, oracle)

    assert score["matrix"]["completed"] == 24
    assert score["strict_schema"] == {"passed": 23, "total": 24}
    assert score["policy"] == {"passed": 23, "total": 24}
    assert score["image_invariance"] == {"passed": 5, "total": 6}
    assert score["failures"][0]["record_id"] == "r001"


def test_inference_error_and_missing_attempt_keep_planned_denominator():
    config, oracle = load_inputs()
    raw = perfect_raw_results()
    failed = raw["attempts"][0]
    failed.update({
        "attempt_status": "error",
        "response_raw": None,
        "generated_token_count": None,
        "error_type": "RuntimeError",
        "error": "bounded test failure",
    })

    error_score = score_raw_results(raw, config, oracle)
    assert error_score["matrix"] == {
        "planned": 24,
        "attempted": 24,
        "completed": 23,
        "errors": 1,
        "complete": True,
    }
    assert error_score["strict_schema"] == {"passed": 23, "total": 24}

    missing = deepcopy(perfect_raw_results())
    missing["attempts"].pop()
    missing_score = score_raw_results(missing, config, oracle)
    assert missing_score["matrix"]["attempted"] == 23
    assert missing_score["matrix"]["complete"] is False
    assert missing_score["policy"]["total"] == 24


def test_image_pair_uses_identical_prompt_and_exact_semantic_output():
    config, oracle = load_inputs()
    raw = perfect_raw_results()
    image_attempt = raw["attempts"][0]
    image_attempt["response_raw"] = canonical_json(oracle["expected_by_record"]["r001"]["image_recipe"])
    image_attempt["prompt_sha256"] = "e" * 64

    score = score_raw_results(raw, config, oracle)

    assert score["strict_schema"] == {"passed": 24, "total": 24}
    assert score["policy"] == {"passed": 23, "total": 24}
    assert score["image_invariance"] == {"passed": 5, "total": 6}
    assert score["prompt_pair_integrity"] == {"passed": 5, "total": 6}


def test_scorer_does_not_import_template_policy_function():
    source = (ROOT / "src/fluidmix/explanation_scoring.py").read_text()
    assert "reference_explanation" not in source


def test_attempt_indexes_cannot_be_reassigned_between_pairs():
    config, oracle = load_inputs()
    raw = perfect_raw_results()
    raw["attempts"][0]["attempt_index"], raw["attempts"][1]["attempt_index"] = 1, 0

    with pytest.raises(ContractError):
        score_raw_results(raw, config, oracle)


def test_scoring_script_writes_once_and_refuses_overwrite(tmp_path):
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "score.json"
    raw_path.write_text(json.dumps(perfect_raw_results()))
    command = [
        sys.executable,
        str(ROOT / "scripts/score_cosmos_contract.py"),
        "--raw-results", str(raw_path),
        "--config", str(ROOT / "data/cosmos_contract_eval_v1.json"),
        "--oracle", str(ROOT / "data/cosmos_contract_oracle_v1.json"),
        "--out", str(out_path),
    ]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    score = json.loads(out_path.read_text())
    assert score["policy"] == {"passed": 24, "total": 24}
    assert second.returncode != 0
    assert score == json.loads(out_path.read_text())

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.cosmos_contract_runner import (
    build_attempt_plan,
    create_frozen_output_parent,
    execute_attempt_plan,
    verify_git_freeze,
    verify_model_snapshot,
    verify_runtime_environment,
)
from fluidmix.explanation_contract import ContractError


ROOT = Path(__file__).resolve().parents[1]


def test_attempt_plan_is_exact_and_full_prompt_pairs_are_byte_identical():
    fixtures = json.loads((ROOT / "data/cosmos_contract_fixtures_v1.json").read_text())
    config = json.loads((ROOT / "data/cosmos_contract_eval_v1.json").read_text())
    prompt = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text()

    attempts = build_attempt_plan(fixtures, config, prompt)

    assert len(attempts) == 24
    assert [attempt["attempt_index"] for attempt in attempts] == list(range(24))
    assert {(attempt["record_id"], attempt["mode"]) for attempt in attempts} == {
        (f"r{index:03d}", mode)
        for index in range(1, 7)
        for mode in config["modes"]
    }
    for record_id in (f"r{index:03d}" for index in range(1, 7)):
        by_mode = {
            item["mode"]: item for item in attempts if item["record_id"] == record_id
        }
        assert by_mode["image_evidence_recipe"]["prompt"] == by_mode["evidence_recipe"]["prompt"]
        assert by_mode["image_evidence_recipe"]["prompt_sha256"] == by_mode["evidence_recipe"]["prompt_sha256"]
        assert by_mode["image_recipe"]["camera_evidence_included"] is False
        assert by_mode["template_reference"]["prompt"] is None


def test_attempt_executor_calls_each_model_condition_once_and_retains_error():
    fixtures = json.loads((ROOT / "data/cosmos_contract_fixtures_v1.json").read_text())
    config = json.loads((ROOT / "data/cosmos_contract_eval_v1.json").read_text())
    prompt = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text()
    plan = build_attempt_plan(fixtures, config, prompt)
    calls = []

    def infer(text, image_attached):
        calls.append((text, image_attached))
        if len(calls) == 2:
            raise RuntimeError("one retained failure")
        return "{}", 2

    attempts = execute_attempt_plan(plan, infer)

    assert len(calls) == 18
    assert len(attempts) == 24
    assert sum(item["attempt_status"] == "error" for item in attempts) == 1
    failed = next(item for item in attempts if item["attempt_status"] == "error")
    assert failed["response_raw"] is None
    assert failed["error_type"] == "RuntimeError"
    assert failed["error"] == "one retained failure"
    assert sum(item["generated_token_count"] is None for item in attempts) == 7


def test_model_snapshot_verifies_every_manifest_file(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weight.bin"
    weight.write_bytes(b"fixed model bytes")
    manifest = {
        "status": "verified",
        "repo_id": "nvidia/Cosmos3-Edge",
        "revision": "344d602b128d1bbdacb43b08d0a3626f46343e29",
        "files": [{
            "path": "weight.bin",
            "bytes": weight.stat().st_size,
            "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = json.loads((ROOT / "data/cosmos_contract_eval_v1.json").read_text())
    config["model_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    verify_model_snapshot(model, manifest_path, config)
    weight.write_bytes(b"tampered model")
    with pytest.raises(ContractError):
        verify_model_snapshot(model, manifest_path, config)
    weight.unlink()
    target = model / "target.bin"
    target.write_bytes(b"fixed model bytes")
    weight.symlink_to(target.name)
    with pytest.raises(ContractError):
        verify_model_snapshot(model, manifest_path, config)


def test_git_freeze_requires_tag_at_clean_head(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "frozen-v1"], cwd=tmp_path, check=True)

    head = verify_git_freeze(tmp_path, "frozen-v1")
    assert len(head) == 40
    tracked.write_text("dirty\n")
    with pytest.raises(ContractError):
        verify_git_freeze(tmp_path, "frozen-v1")


def test_runner_cli_is_local_only_and_has_no_oracle_or_output_override():
    script = ROOT / "scripts/run_cosmos_contract_eval.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    source = script.read_text()

    assert result.returncode == 0, result.stderr
    for option in ("--model-dir", "--model-manifest", "--runtime-lock", "--image"):
        assert option in result.stdout
    assert "--oracle" not in result.stdout
    assert "--out" not in result.stdout
    assert "cosmos_contract_oracle" not in source
    assert "requests" not in source
    assert "http://" not in source and "https://" not in source


def test_runtime_verifier_checks_every_lock_pin_and_interpreter(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("alpha_pkg==1.2.3\nbeta-pkg==4.5.6\n")
    config = {
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python_version": "3.13.17",
        "platform_machine": "aarch64",
        "runtime_versions": {"alpha-pkg": "1.2.3"},
    }
    installed = {"alpha-pkg": "1.2.3", "beta-pkg": "4.5.6"}

    assert verify_runtime_environment(
        lock,
        config,
        installed_versions=installed,
        actual_python="3.13.17",
        actual_machine="aarch64",
    ) == installed
    with pytest.raises(ContractError):
        verify_runtime_environment(
            lock,
            config,
            installed_versions={**installed, "beta-pkg": "9.9.9"},
            actual_python="3.13.17",
            actual_machine="aarch64",
        )
    with pytest.raises(ContractError):
        verify_runtime_environment(
            lock,
            config,
            installed_versions=installed,
            actual_python="3.12.0",
            actual_machine="aarch64",
        )


def test_output_parent_rejects_symlinked_outputs_root(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "outputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ContractError):
        create_frozen_output_parent(repo, "outputs/run/raw-results.json")
    assert not (outside / "run").exists()

    (repo / "outputs").unlink()
    output = create_frozen_output_parent(repo, "outputs/run/raw-results.json")
    assert output == repo.resolve() / "outputs/run/raw-results.json"
    assert output.parent.is_dir()
    with pytest.raises(ContractError):
        create_frozen_output_parent(repo, "outputs/run/raw-results.json")

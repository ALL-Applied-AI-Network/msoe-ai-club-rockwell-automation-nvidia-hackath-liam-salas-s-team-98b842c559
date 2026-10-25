import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import (  # noqa: E402
    build_attempt_plan,
    publish_immutable_json,
    verify_git_freeze,
    verify_runtime_environment,
    verify_source_hashes,
)
from fluidmix.cosmos_nano_contract import (  # noqa: E402
    SOURCE_PATHS,
    validate_nano_eval_config,
)
from fluidmix.cosmos_nano_model_view import (  # noqa: E402
    stage_model_view,
    verify_model_view,
)
from fluidmix.explanation_contract import ContractError  # noqa: E402


REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"


def valid_config():
    digest = "1" * 64
    return {
        "schema_version": "cosmos3_nano_matched_eval.v1",
        "config_status": "execution_frozen",
        "experiment_id": "cosmos3-nano-matched-20261024",
        "freeze_tag": "explore/cosmos3-nano-matched-frozen-20261024",
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
        "model_revision": REVISION,
        "model_type": "cosmos3_omni",
        "expected_model_class": "Cosmos3OmniForConditionalGeneration",
        "expected_processor_class": "Qwen3VLProcessor",
        "expected_tokenizer_class": "Qwen2TokenizerFast",
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
        "model_view_manifest_sha256": digest,
        "model_view_file_count": 48,
        "model_view_total_bytes": 31_500_000_000,
        "fixture_set_sha256": (
            "e88f6f13476ae89a49d53e0ddfa6cd2be051530c073e946f5e985c16d2f0ee9f"
        ),
        "prompt_sha256": (
            "0733da91d7c1c347feae3afabdc170ba3a41fa5a9726f310a40eab887f3eb74c"
        ),
        "image_sha256": (
            "c4886724246a6dbef76c6658b07c8568c5e8e4372f7a9a623021a3ddeb3997af"
        ),
        "chat_template_file_sha256": (
            "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"
        ),
        "chat_template_text_sha256": (
            "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
        ),
        "chat_template_generation_suffix": "<|im_start|>assistant\n",
        "input_audit_sha256": digest,
        "source_sha256": {path: digest for path in SOURCE_PATHS},
        "image_role": "irrelevant_fixed_fillcam_distraction",
        "output_path": "outputs/20261024-cosmos3-nano-matched/raw-results.json",
    }


def test_nano_config_freezes_exact_matrix_runtime_and_native_generation():
    config = valid_config()

    validate_nano_eval_config(config)
    assert config["planned_attempts"] == 24
    assert config["model_attempts"] == 18
    assert config["template_control_attempts"] == 6
    assert config["required_abstention_attempts"] == 15
    assert config["required_abstention_model_attempts"] == 12
    assert config["required_abstention_control_attempts"] == 3


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("batch_size", 2),
        ("do_sample", True),
        ("max_new_tokens", 385),
        ("native_eos_token_ids", [151643]),
        ("local_files_only", False),
        ("model_repository", "nvidia/Cosmos3-Edge"),
    ],
)
def test_nano_config_rejects_unmatched_treatment(field, wrong):
    config = valid_config()
    config[field] = wrong

    with pytest.raises(ContractError):
        validate_nano_eval_config(config)


def test_nano_config_rejects_enable_thinking_even_when_false():
    config = valid_config()
    config["enable_thinking"] = False

    with pytest.raises(ContractError):
        validate_nano_eval_config(config)


def test_attempt_plan_is_fixture_major_and_oracle_blind():
    fixtures = json.loads((ROOT / "data/cosmos_contract_fixtures_v1.json").read_text())
    prompt = (ROOT / "data/cosmos_operator_prompt_v1.txt").read_text()

    plan = build_attempt_plan(fixtures, valid_config(), prompt)

    assert len(plan) == 24
    assert [(row["record_id"], row["mode"]) for row in plan[:4]] == [
        ("r001", "image_evidence_recipe"),
        ("r001", "evidence_recipe"),
        ("r001", "image_recipe"),
        ("r001", "template_reference"),
    ]
    for record_id in (f"r{index:03d}" for index in range(1, 7)):
        by_mode = {row["mode"]: row for row in plan if row["record_id"] == record_id}
        assert by_mode["image_evidence_recipe"]["prompt"] == by_mode["evidence_recipe"]["prompt"]
        assert by_mode["image_recipe"]["camera_evidence_included"] is False
        assert by_mode["template_reference"]["prompt"] is None


def test_runtime_verifier_rejects_missing_mismatch_and_unapproved_extras(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("alpha_pkg==1.2.3\nbeta-pkg==4.5.6\n")
    config = {
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "runtime_lock_pin_count": 2,
        "allowed_extra_distributions": {"py-spy": "0.4.2"},
        "python_version": "3.12.3",
        "platform_machine": "aarch64",
        "runtime_versions": {"alpha-pkg": "1.2.3"},
    }
    installed = {
        "alpha-pkg": "1.2.3",
        "beta-pkg": "4.5.6",
        "py-spy": "0.4.2",
    }

    result = verify_runtime_environment(
        lock,
        config,
        installed_versions=installed,
        actual_python="3.12.3",
        actual_machine="aarch64",
    )

    assert result["locked"] == {"alpha-pkg": "1.2.3", "beta-pkg": "4.5.6"}
    assert result["extras"] == {"py-spy": "0.4.2"}
    for bad in (
        {**installed, "gamma": "1"},
        {**installed, "beta-pkg": "9"},
        {key: value for key, value in installed.items() if key != "beta-pkg"},
    ):
        with pytest.raises(ContractError):
            verify_runtime_environment(
                lock,
                config,
                installed_versions=bad,
                actual_python="3.12.3",
                actual_machine="aarch64",
            )


def _make_fake_cache(tmp_path):
    cache = tmp_path / "models--nvidia--Cosmos3-Nano"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    weight_paths = [f"transformer/model-{index:05d}.safetensors" for index in range(42)]
    root_payloads = {
        "config.json": json.dumps({"model_type": "cosmos3_omni"}).encode(),
        "generation_config.json": json.dumps(
            {"eos_token_id": [151645, 151643], "pad_token_id": 151643}
        ).encode(),
        "chat_template.json": b'{"chat_template":"<|im_start|>assistant\\n"}',
        "README.md": b"reasoner fixture",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {f"weight.{index}": path for index, path in enumerate(weight_paths)}}
        ).encode(),
        "assets/reasoning.json": b'{"documented_glob":"*.json"}',
    }
    payloads = {
        **root_payloads,
        **{path: f"weight-{index}".encode() for index, path in enumerate(weight_paths)},
    }
    for index, (relative, payload) in enumerate(payloads.items()):
        blob = blobs / f"blob-{index:03d}"
        blob.write_bytes(payload)
        link = snapshot / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    ignored = snapshot / "vae/ignored.bin"
    ignored.parent.mkdir()
    ignored.write_bytes(b"not in reasoner view")
    return snapshot


def test_model_view_stages_exact_allowed_hardlinks_and_detects_extras(tmp_path):
    snapshot = _make_fake_cache(tmp_path)
    view = tmp_path / "private-view"
    manifest_path = tmp_path / "model-view-manifest.json"

    manifest = stage_model_view(snapshot, view, manifest_path)
    config = {
        "model_repository": "nvidia/Cosmos3-Nano",
        "model_revision": REVISION,
        "model_view_file_count": 48,
        "model_view_total_bytes": manifest["total_bytes"],
        "model_view_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "native_eos_token_ids": [151645, 151643],
        "native_pad_token_id": 151643,
        "model_type": "cosmos3_omni",
    }

    verified = verify_model_view(view, manifest_path, config)

    assert verified["file_count"] == 48
    assert all(not path.is_symlink() for path in view.rglob("*") if path.is_file())
    staged_weight = view / "transformer/model-00000.safetensors"
    source_weight = snapshot / "transformer/model-00000.safetensors"
    assert staged_weight.stat().st_ino == source_weight.resolve().stat().st_ino
    extra = view / "unexpected.json"
    extra.write_text("{}")
    with pytest.raises(ContractError):
        verify_model_view(view, manifest_path, config)


def test_model_view_rejects_snapshot_link_outside_cache_blobs(tmp_path):
    snapshot = _make_fake_cache(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    bad_link = snapshot / "transformer/model-00000.safetensors"
    bad_link.unlink()
    bad_link.symlink_to(outside)

    with pytest.raises(ContractError):
        stage_model_view(snapshot, tmp_path / "view", tmp_path / "manifest.json")


def test_model_view_accepts_documented_cache_wide_shared_blob_hop(tmp_path):
    snapshot = _make_fake_cache(tmp_path)
    source_link = snapshot / "README.md"
    payload = source_link.resolve().read_bytes()
    repo_blobs = snapshot.parent.parent / "blobs"
    repo_etag = hashlib.sha256(payload).hexdigest()
    shared_hash = "a" * 64
    shared_root = tmp_path / "blobs"
    shared_target = shared_root / shared_hash[:2] / shared_hash
    shared_target.parent.mkdir(parents=True)
    (shared_root / ".huggingface-shared-blobs").write_text("1\n")
    shared_target.write_bytes(payload)
    repo_entry = repo_blobs / repo_etag
    repo_entry.symlink_to(os.path.relpath(shared_target, repo_blobs))
    source_link.unlink()
    source_link.symlink_to(os.path.relpath(repo_entry, source_link.parent))

    view = tmp_path / "private-view"
    stage_model_view(snapshot, view, tmp_path / "model-view-manifest.json")

    assert (view / "README.md").stat().st_ino == shared_target.stat().st_ino


def test_model_view_rejects_unmarked_or_malformed_shared_blob_hop(tmp_path):
    snapshot = _make_fake_cache(tmp_path)
    source_link = snapshot / "README.md"
    payload = source_link.resolve().read_bytes()
    repo_blobs = snapshot.parent.parent / "blobs"
    repo_etag = hashlib.sha256(payload).hexdigest()
    shared_target = tmp_path / "blobs" / "aa" / ("a" * 64)
    shared_target.parent.mkdir(parents=True)
    shared_target.write_bytes(payload)
    repo_entry = repo_blobs / repo_etag
    repo_entry.symlink_to(os.path.relpath(shared_target, repo_blobs))
    source_link.unlink()
    source_link.symlink_to(os.path.relpath(repo_entry, source_link.parent))

    with pytest.raises(ContractError, match="shared blob"):
        stage_model_view(snapshot, tmp_path / "view", tmp_path / "manifest.json")


def test_source_hash_verifier_checks_exact_reviewed_path_set(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("one\n")
    second.write_text("two\n")
    expected = {
        "first.py": hashlib.sha256(first.read_bytes()).hexdigest(),
        "second.py": hashlib.sha256(second.read_bytes()).hexdigest(),
    }

    assert verify_source_hashes(tmp_path, expected, ("first.py", "second.py")) == expected
    first.write_text("changed\n")
    with pytest.raises(ContractError):
        verify_source_hashes(tmp_path, expected, ("first.py", "second.py"))


def test_git_freeze_and_immutable_publication_are_fail_closed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("frozen\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=repo, check=True)
    subprocess.run(["git", "tag", "frozen-v1"], cwd=repo, check=True)
    assert len(verify_git_freeze(repo, "frozen-v1")) == 40

    output = repo / "outputs/run/raw-results.json"
    publish_immutable_json(repo, "outputs/run/raw-results.json", {"complete": True})
    assert json.loads(output.read_text()) == {"complete": True}
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(ContractError):
        publish_immutable_json(repo, "outputs/run/raw-results.json", {"complete": False})


def test_raw_publication_allows_a_predeclared_manifest_in_the_run_directory(tmp_path):
    repo = tmp_path / "repo"
    run = repo / "outputs/run"
    run.mkdir(parents=True)
    (run / "experiment-config.json").write_text('{"status":"execution_frozen"}\n')

    output = publish_immutable_json(
        repo, "outputs/run/raw-results.json", {"complete": True}
    )

    assert output == run / "raw-results.json"
    assert json.loads(output.read_text()) == {"complete": True}

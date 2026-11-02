import hashlib
import json
from pathlib import Path
import subprocess

import pytest


OLD_COMMIT = "10c106fbd38e3da34d902abb7d920c5a4041440e"
OLD_PATH = "src/fluidmix/wp7_vlm_decision.py"
OLD_SHA256 = "06694591e642b7bc558bd743fd9c9a97d05ba4162c7e9175d77ad2cd979f689b"
ONION_COMMIT = "98141660eda83a5176eca7b578d26bf7b550cfa1"
ONION_PATH = "src/fluidmix/explanation/vlm/decision.py"
NEW_SHA256 = "0fb8213220b545f18bc9e95a603933004ae948acc7f95a005c911c0458594645"
FREEZE_SHA256 = "8d5a884c8bf7438b291816473132d5c61f24c3d34d3c3654c827567006aae3e5"


def _old_source() -> bytes:
    return subprocess.run(
        ["git", "show", f"{OLD_COMMIT}:{OLD_PATH}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _onion_source() -> bytes:
    return subprocess.run(
        ["git", "show", f"{ONION_COMMIT}:{ONION_PATH}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def test_rebind_proves_move_only_and_changes_only_source_hash(tmp_path):
    from fluidmix.explanation.vlm.freeze_rebind import rebind_freeze

    old_freeze = tmp_path / "accepted-freeze.json"
    old_payload = {
        "schema_version": "wp7-vlm-development-freeze-v12",
        "decision_source_sha256": OLD_SHA256,
        "prompt_sha256": "1" * 64,
        "response_schema_sha256": "2" * 64,
        "generation_config": {"do_sample": False},
        "models": [{"repository": "model", "revision": "revision"}],
        "aggregation": {"threshold": 0.5},
        "deterministic_policy": {"keep": {"threshold": 1.0}},
    }
    old_freeze.write_text(json.dumps(old_payload, sort_keys=True, indent=2) + "\n")
    output = tmp_path / "freeze.json"
    receipt_path = tmp_path / "receipt.json"

    receipt = rebind_freeze(
        accepted_freeze=old_freeze,
        accepted_freeze_sha256=hashlib.sha256(old_freeze.read_bytes()).hexdigest(),
        accepted_source=_old_source(),
        accepted_source_commit=OLD_COMMIT,
        accepted_source_path=OLD_PATH,
        accepted_source_sha256=OLD_SHA256,
        onion_source_bytes=_onion_source(),
        onion_source_path=ONION_PATH,
        onion_source_sha256=NEW_SHA256,
        output_freeze=output,
        output_receipt=receipt_path,
    )

    new_payload = json.loads(output.read_bytes())
    assert {
        key for key in old_payload if old_payload[key] != new_payload[key]
    } == {"decision_source_sha256"}
    assert new_payload["decision_source_sha256"] == NEW_SHA256
    assert receipt["proof"]["normalized_ast_diff"] == []
    assert receipt["proof"]["ignored_differences"] == {
        "import_targets": 3,
        "file_derived_module_paths": 4,
    }
    assert set(receipt["proof"]["identical_components"]) == {
        "prompt_strings",
        "response_schema",
        "guard_logic",
        "thresholds_and_aggregation",
        "generation_config",
        "model_pins",
    }
    assert receipt["proof"]["freeze_changed_keys"] == ["decision_source_sha256"]
    assert receipt["sources"]["accepted"]["git_commit"] == OLD_COMMIT
    assert receipt["sources"]["onion"]["sha256"] == NEW_SHA256
    assert receipt["artifacts"]["old_freeze_sha256"] == hashlib.sha256(
        old_freeze.read_bytes()
    ).hexdigest()
    assert receipt["artifacts"]["new_freeze_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert json.loads(receipt_path.read_bytes()) == receipt


def test_rebind_fails_closed_on_non_move_source_change(tmp_path):
    from fluidmix.explanation.vlm.freeze_rebind import FreezeRebindError, rebind_freeze

    old_source = _old_source()
    changed = old_source.replace(b'"threshold": 0.5', b'"threshold": 0.6', 1)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"decision_source_sha256": OLD_SHA256}) + "\n")

    with pytest.raises(FreezeRebindError, match="normalized AST differs"):
        rebind_freeze(
            accepted_freeze=freeze,
            accepted_freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
            accepted_source=old_source,
            accepted_source_commit=OLD_COMMIT,
            accepted_source_path=OLD_PATH,
            accepted_source_sha256=OLD_SHA256,
            onion_source_bytes=changed,
            onion_source_path="changed.py",
            onion_source_sha256=hashlib.sha256(changed).hexdigest(),
            output_freeze=tmp_path / "out.json",
            output_receipt=tmp_path / "receipt.json",
        )


def test_rebind_rejects_ast_invisible_byte_change_in_required_component(tmp_path):
    from fluidmix.explanation.vlm.freeze_rebind import FreezeRebindError, rebind_freeze

    old_source = _old_source()
    marker = b'    """Check only claims that can contradict or escape the supplied bundle."""\n'
    changed = old_source.replace(marker, marker + b"    # byte-only guard drift\n", 1)
    assert changed != old_source
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"decision_source_sha256": OLD_SHA256}) + "\n")

    with pytest.raises(FreezeRebindError, match="required decision component differs"):
        rebind_freeze(
            accepted_freeze=freeze,
            accepted_freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
            accepted_source=old_source,
            accepted_source_commit=OLD_COMMIT,
            accepted_source_path=OLD_PATH,
            accepted_source_sha256=OLD_SHA256,
            onion_source_bytes=changed,
            onion_source_path="comment-drift.py",
            onion_source_sha256=hashlib.sha256(changed).hexdigest(),
            output_freeze=tmp_path / "out.json",
            output_receipt=tmp_path / "receipt.json",
        )


def test_rebind_fails_closed_on_accepted_freeze_drift(tmp_path):
    from fluidmix.explanation.vlm.freeze_rebind import FreezeRebindError, rebind_freeze

    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"decision_source_sha256": OLD_SHA256}) + "\n")
    with pytest.raises(FreezeRebindError, match="accepted freeze SHA-256"):
        rebind_freeze(
            accepted_freeze=freeze,
            accepted_freeze_sha256="0" * 64,
            accepted_source=_old_source(),
            accepted_source_commit=OLD_COMMIT,
            accepted_source_path=OLD_PATH,
            accepted_source_sha256=OLD_SHA256,
            onion_source=Path("src/fluidmix/explanation/vlm/decision.py"),
            onion_source_sha256=NEW_SHA256,
            output_freeze=tmp_path / "out.json",
            output_receipt=tmp_path / "receipt.json",
        )

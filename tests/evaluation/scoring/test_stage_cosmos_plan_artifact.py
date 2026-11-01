"""CPU contract tests for the immutable stage attempt-plan preflight."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skip(
    reason="legacy_diagnostic_only: active stage attempt-plan artifact path is deactivated"
)


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.template.explanation_contract import ContractError  # noqa: E402
from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_attempt_plan  # noqa: E402
from evaluation.scoring.stage_cosmos_plan_artifact import (  # noqa: E402
    build_stage_attempt_plan_artifact,
    load_stage_attempt_plan_artifact,
    stage_attempt_plan_artifact_bytes,
    validate_stage_attempt_plan_artifact,
    write_stage_attempt_plan_artifact,
)


SCRIPT = ROOT / "cli/evaluation/scoring" / "build_stage_cosmos_attempt_plan.py"


def _measured():
    from tests.fluidmix.explanation.template.test_stage_cosmos_explanation import _decision

    return {
        "artifact_type": "measured_evidence",
        "schema_version": "stage_measured_evidence.v1",
        "data_origin": "twin",
        "real_data_used": False,
        "decision_records": [
            _decision("discard", suffix=f"{index:02d}") for index in range(12)
        ],
    }


def test_artifact_binds_exact_matrix_and_control_preflight():
    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    evidence_sha = hashlib.sha256(
        json.dumps(measured, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = build_stage_attempt_plan_artifact(
        measured, prompt, measured_evidence_sha256=evidence_sha
    )

    assert artifact["schema_version"] == "stage_cosmos_attempt_plan.v1"
    assert artifact["artifact_type"] == "stage_cosmos_attempt_plan"
    assert artifact["decision_count"] == 12
    assert artifact["planned_attempts"] == 48
    assert artifact["model_attempts"] == 36
    assert artifact["template_control_attempts"] == 12
    assert artifact["data_origin"] == "twin"
    assert artifact["real_data_used"] is False
    assert len(artifact["attempts"]) == 48
    assert len(artifact["control_preflight"]) == 12
    assert all(row["strict_content_policy_passed"] for row in artifact["control_preflight"])
    assert all(row["required_abstention_passed"] for row in artifact["control_preflight"])
    serialized = json.dumps(artifact)
    assert "rgb_path" not in serialized
    assert "mask_path" not in serialized
    validate_stage_attempt_plan_artifact(
        artifact,
        measured,
        prompt,
        measured_evidence_sha256=evidence_sha,
    )


def test_artifact_is_deterministic_and_rejects_drift_and_extra_keys():
    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    evidence_sha = hashlib.sha256(
        json.dumps(measured, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    first = build_stage_attempt_plan_artifact(
        measured, prompt, measured_evidence_sha256=evidence_sha
    )
    second = build_stage_attempt_plan_artifact(
        measured, prompt, measured_evidence_sha256=evidence_sha
    )
    assert stage_attempt_plan_artifact_bytes(first) == stage_attempt_plan_artifact_bytes(second)

    changed = json.loads(json.dumps(first))
    changed["attempts"][0]["attempt_index"] = 99
    with pytest.raises(ContractError):
        validate_stage_attempt_plan_artifact(
            changed, measured, prompt, measured_evidence_sha256=evidence_sha
        )

    extra = json.loads(json.dumps(first))
    extra["unexpected"] = True
    with pytest.raises(ContractError):
        validate_stage_attempt_plan_artifact(
            extra, measured, prompt, measured_evidence_sha256=evidence_sha
        )


def test_artifact_matches_existing_successful_attempt_selection():
    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    evidence_sha = hashlib.sha256(
        json.dumps(measured, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = build_stage_attempt_plan_artifact(
        measured, prompt, measured_evidence_sha256=evidence_sha
    )
    assert artifact["attempts"] == build_stage_attempt_plan(measured, prompt)


def test_artifact_file_is_canonical_no_clobber_and_rejects_symlink(tmp_path):
    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    evidence_sha = hashlib.sha256(
        json.dumps(measured, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = build_stage_attempt_plan_artifact(
        measured, prompt, measured_evidence_sha256=evidence_sha
    )
    output = tmp_path / "attempt-plan.json"

    write_stage_attempt_plan_artifact(output, artifact)
    loaded, digest = load_stage_attempt_plan_artifact(
        output,
        measured,
        prompt,
        measured_evidence_sha256=evidence_sha,
    )

    assert loaded == artifact
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    original = output.read_bytes()
    with pytest.raises(ContractError, match="must be absent"):
        write_stage_attempt_plan_artifact(output, artifact)
    assert output.read_bytes() == original
    link = tmp_path / "attempt-plan-link.json"
    link.symlink_to(output)
    with pytest.raises(ContractError, match="regular non-symlink"):
        load_stage_attempt_plan_artifact(
            link,
            measured,
            prompt,
            measured_evidence_sha256=evidence_sha,
        )


def test_artifact_rejects_nonfinite_unused_evidence():
    measured = _measured()
    measured["image_measurements"] = [{"unused": float("nan")}]
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")

    with pytest.raises(ContractError, match="nonfinite"):
        build_stage_attempt_plan_artifact(measured, prompt)


def test_public_cli_binds_raw_evidence_bytes_and_refuses_clobber(tmp_path):
    measured_path = tmp_path / "measured-evidence.json"
    measured_path.write_text(
        json.dumps(_measured(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "attempt-plan.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--measured-evidence",
        str(measured_path),
        "--out",
        str(output),
    ]

    first = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["measured_evidence_sha256"] == hashlib.sha256(
        measured_path.read_bytes()
    ).hexdigest()
    original = output.read_bytes()

    second = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert second.returncode != 0
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))

"""Immutable wrapper for the Varied V3 Cosmos attempt matrix."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_f2_varied_explanation import _artifact  # noqa: E402


PROMPT = (ROOT / "data/cosmos_f2_varied_prompt_v1.txt").read_text(encoding="utf-8")


def test_artifact_binds_18_decisions_72_attempts_and_all_external_inputs():
    from fluidmix.v3_cosmos_plan_artifact import build_v3_attempt_plan_artifact

    artifact = build_v3_attempt_plan_artifact(
        _artifact(),
        PROMPT,
        decision_artifact_sha256="a" * 64,
        image_locator_sha256="b" * 64,
    )

    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "f2_varied_cosmos_attempt_plan"
    assert artifact["status"] == "frozen"
    assert artifact["decision_artifact_sha256"] == "a" * 64
    assert artifact["image_locator_sha256"] == "b" * 64
    assert artifact["prompt_template_sha256"] == hashlib.sha256(
        PROMPT.encode()
    ).hexdigest()
    assert artifact["denominators"] == {
        "decisions": 18,
        "planned_attempts": 72,
        "model_attempts": 54,
        "template_control_attempts": 18,
        "multi_image_attempts": 36,
    }
    assert len(artifact["attempts"]) == 72
    assert sum(row["status"] == "not_run" for row in artifact["attempts"]) == 72


def test_loader_recomputes_every_attempt_and_rejects_post_freeze_mutation(tmp_path):
    from fluidmix.v3_cosmos_plan_artifact import (
        build_v3_attempt_plan_artifact,
        load_v3_attempt_plan_artifact,
        v3_attempt_plan_json_bytes,
    )

    decisions = _artifact()
    artifact = build_v3_attempt_plan_artifact(
        decisions,
        PROMPT,
        decision_artifact_sha256="a" * 64,
        image_locator_sha256="b" * 64,
    )
    path = tmp_path / "attempt-plan.json"
    path.write_bytes(v3_attempt_plan_json_bytes(artifact))
    loaded, digest = load_v3_attempt_plan_artifact(
        path,
        decisions,
        PROMPT,
        decision_artifact_sha256="a" * 64,
        image_locator_sha256="b" * 64,
    )
    assert loaded == artifact
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    mutated = deepcopy(artifact)
    mutated["attempts"][0]["prompt"] += " changed"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputed"):
        load_v3_attempt_plan_artifact(
            path,
            decisions,
            PROMPT,
            decision_artifact_sha256="a" * 64,
            image_locator_sha256="b" * 64,
        )


def test_artifact_rejects_placeholder_or_malformed_hashes():
    from fluidmix.v3_cosmos_plan_artifact import build_v3_attempt_plan_artifact

    with pytest.raises(ValueError, match="SHA-256"):
        build_v3_attempt_plan_artifact(
            _artifact(),
            PROMPT,
            decision_artifact_sha256="0" * 64,
            image_locator_sha256="b" * 64,
        )

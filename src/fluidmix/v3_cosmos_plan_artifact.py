"""Immutable wrapper for the Varied V3 Cosmos attempt matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from fluidmix.explanation_contract import ContractError, canonical_json
from fluidmix.f2_varied_explanation import build_varied_attempt_plan
from fluidmix.v3_cosmos_runtime import _validate_plan


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONDITIONS = [
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
    "deterministic_template",
]


def _digest(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ContractError(f"{role} must be a non-placeholder SHA-256")
    return value


def build_v3_attempt_plan_artifact(
    decision_artifact: object,
    prompt_template: object,
    *,
    decision_artifact_sha256: object,
    image_locator_sha256: object,
) -> dict[str, object]:
    """Bind all 72 planned attempts to decisions, prompt, and image map."""

    decision_sha = _digest(decision_artifact_sha256, "decision artifact SHA-256")
    locator_sha = _digest(image_locator_sha256, "image locator SHA-256")
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ContractError("V3 prompt template must be nonempty text")
    attempts = _validate_plan(
        build_varied_attempt_plan(decision_artifact, prompt_template)
    )
    denominators = {
        "decisions": 18,
        "planned_attempts": 72,
        "model_attempts": 54,
        "template_control_attempts": 18,
        "multi_image_attempts": 36,
    }
    observed = {
        "decisions": len({str(row["decision_id"]) for row in attempts}),
        "planned_attempts": len(attempts),
        "model_attempts": sum(row["kind"] == "model" for row in attempts),
        "template_control_attempts": sum(
            row["kind"] == "control" for row in attempts
        ),
        "multi_image_attempts": sum(
            row["requires_multi_image_smoke"] is True for row in attempts
        ),
    }
    if observed != denominators:
        raise ContractError("V3 attempt plan denominator drifted")
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_cosmos_attempt_plan",
        "status": "frozen",
        "decision_artifact_sha256": decision_sha,
        "decision_content_sha256": hashlib.sha256(
            canonical_json(decision_artifact).encode("utf-8")
        ).hexdigest(),
        "prompt_template_sha256": hashlib.sha256(
            prompt_template.encode("utf-8")
        ).hexdigest(),
        "image_locator_sha256": locator_sha,
        "conditions": list(_CONDITIONS),
        "denominators": denominators,
        "attempts": attempts,
        "claim_boundary": (
            "Frozen synthetic explanation inputs only. This plan has no labels and "
            "does not establish model quality, physical cause, or sim-to-real validity."
        ),
    }


def v3_attempt_plan_json_bytes(value: object) -> bytes:
    """Serialize the plan artifact deterministically without non-finite numbers."""

    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def load_v3_attempt_plan_artifact(
    path: str | Path,
    decision_artifact: object,
    prompt_template: object,
    *,
    decision_artifact_sha256: object,
    image_locator_sha256: object,
) -> tuple[dict[str, object], str]:
    """Load a regular file and require exact recomputation before execution."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContractError("V3 attempt-plan artifact must be a regular file")
    data = source.read_bytes()
    try:
        parsed = json.loads(
            data,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("V3 attempt-plan artifact is not strict JSON") from exc
    expected = build_v3_attempt_plan_artifact(
        decision_artifact,
        prompt_template,
        decision_artifact_sha256=decision_artifact_sha256,
        image_locator_sha256=image_locator_sha256,
    )
    if parsed != expected or data != v3_attempt_plan_json_bytes(expected):
        raise ContractError("V3 attempt-plan artifact differs from recomputed plan")
    return expected, hashlib.sha256(data).hexdigest()

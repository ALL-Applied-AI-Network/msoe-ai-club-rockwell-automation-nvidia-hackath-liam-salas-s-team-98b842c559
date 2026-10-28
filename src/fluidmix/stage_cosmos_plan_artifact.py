"""Immutable, CPU-only preflight artifact for the frozen stage Cosmos matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

from fluidmix.explanation_contract import ContractError, canonical_json
from fluidmix.part_b_evidence import prohibit_legacy_diagnostic_active_path
from fluidmix.render_process import load_strict_json_snapshot


SCHEMA_VERSION = "stage_cosmos_attempt_plan.v1"
CLAIM_BOUNDARY = (
    "Synthetic colour-distance evidence only; physical carryover, defect, cause, "
    "volume, plant action, and sim-to-real validity are unverified."
)
_ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "measured_evidence_sha256",
    "prompt_template_sha256",
    "data_origin",
    "real_data_used",
    "decision_count",
    "planned_attempts",
    "model_attempts",
    "template_control_attempts",
    "claim_boundary",
    "attempts",
    "control_preflight",
}
_HEX64 = set("0123456789abcdef")
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_evidence_sha256(value: object) -> str:
    return _sha256(canonical_json(value).encode("utf-8"))


def _require_sha256(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX64 for character in value)
        or value == "0" * 64
    ):
        raise ContractError(f"stage plan {role} must be a non-placeholder SHA-256")
    return value


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("stage plan contains a nonfinite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ContractError("stage plan object keys must be strings")
            _reject_nonfinite(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_nonfinite(nested)


def _control_preflight(
    attempts: list[dict[str, object]], measured_evidence: Mapping[str, object]
) -> list[dict[str, object]]:
    prohibit_legacy_diagnostic_active_path("stage_cosmos_plan_artifact._control_preflight")
    decisions = {
        row["decision_id"]: row
        for row in measured_evidence["decision_records"]
        if isinstance(row, Mapping)
    }
    controls = [row for row in attempts if row.get("kind") == "control"]
    if len(controls) != 12:
        raise ContractError("stage plan requires exactly 12 deterministic controls")
    records: list[dict[str, object]] = []
    for spec in controls:
        decision = decisions.get(spec.get("decision_id"))
        raw = spec.get("template_response")
        if not isinstance(decision, Mapping) or not isinstance(raw, str):
            raise ContractError("stage control preflight input is malformed")
        parsed = parse_stage_response(raw)
        failures = stage_content_failures(raw, parsed, decision)
        expected_abstention = decision["disposition"] == "insufficient_evidence"
        required_abstention_passed = parsed["abstention"] == (
            "abstain" if expected_abstention else "limited"
        )
        if failures or not raw_stage_format_passes(raw) or not required_abstention_passed:
            raise ContractError("deterministic stage control failed content preflight")
        records.append(
            {
                "attempt_index": spec["attempt_index"],
                "attempt_id": spec["attempt_id"],
                "decision_id": spec["decision_id"],
                "response_sha256": _sha256(raw.encode("utf-8")),
                "parsed": parsed,
                "raw_format_passed": True,
                "strict_content_policy_passed": True,
                "required_abstention_passed": True,
            }
        )
    return records


def build_stage_attempt_plan_artifact(
    measured_evidence: object,
    prompt_template: object,
    *,
    measured_evidence_sha256: str | None = None,
    prompt_template_sha256: str | None = None,
) -> dict[str, object]:
    """Build the deterministic artifact without loading a model or reading images."""

    prohibit_legacy_diagnostic_active_path("build_stage_attempt_plan_artifact")

    if not isinstance(measured_evidence, Mapping):
        raise ContractError("stage plan evidence must be an object")
    if not isinstance(prompt_template, str):
        raise ContractError("stage plan prompt must be text")
    _reject_nonfinite(measured_evidence)
    attempts = build_stage_attempt_plan(measured_evidence, prompt_template)
    evidence_sha = measured_evidence_sha256 or _canonical_evidence_sha256(measured_evidence)
    prompt_sha = prompt_template_sha256 or _sha256(prompt_template.encode("utf-8"))
    _require_sha256(evidence_sha, "measured evidence")
    _require_sha256(prompt_sha, "prompt")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "stage_cosmos_attempt_plan",
        "measured_evidence_sha256": evidence_sha,
        "prompt_template_sha256": prompt_sha,
        "data_origin": "twin",
        "real_data_used": False,
        "decision_count": 12,
        "planned_attempts": 48,
        "model_attempts": 36,
        "template_control_attempts": 12,
        "claim_boundary": CLAIM_BOUNDARY,
        "attempts": attempts,
        "control_preflight": _control_preflight(attempts, measured_evidence),
    }
    validate_stage_attempt_plan_artifact(
        artifact,
        measured_evidence,
        prompt_template,
        measured_evidence_sha256=evidence_sha,
        prompt_template_sha256=prompt_sha,
    )
    return artifact


def validate_stage_attempt_plan_artifact(
    value: object,
    measured_evidence: object,
    prompt_template: object,
    *,
    measured_evidence_sha256: str | None = None,
    prompt_template_sha256: str | None = None,
) -> dict[str, object]:
    """Validate exact schema, hashes, matrix, and recomputed control records."""

    prohibit_legacy_diagnostic_active_path("validate_stage_attempt_plan_artifact")

    _reject_nonfinite(value)
    _reject_nonfinite(measured_evidence)
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        extra = sorted(set(value) - _ARTIFACT_KEYS) if isinstance(value, dict) else []
        missing = sorted(_ARTIFACT_KEYS - set(value)) if isinstance(value, dict) else []
        raise ContractError(f"stage plan artifact keys mismatch: extra={extra}, missing={missing}")
    if value["schema_version"] != SCHEMA_VERSION or type(value["schema_version"]) is not str:
        raise ContractError("stage plan artifact schema version drifted")
    if value["artifact_type"] != "stage_cosmos_attempt_plan":
        raise ContractError("stage plan artifact type drifted")
    if value["data_origin"] != "twin" or value["real_data_used"] is not False:
        raise ContractError("stage plan artifact data provenance is invalid")
    for field, expected in (
        ("decision_count", 12),
        ("planned_attempts", 48),
        ("model_attempts", 36),
        ("template_control_attempts", 12),
    ):
        if type(value[field]) is not int or value[field] != expected:
            raise ContractError(f"stage plan artifact denominator drifted: {field}")
    if value["claim_boundary"] != CLAIM_BOUNDARY or type(value["claim_boundary"]) is not str:
        raise ContractError("stage plan artifact claim boundary drifted")
    expected_evidence_sha = measured_evidence_sha256 or _canonical_evidence_sha256(measured_evidence)
    expected_prompt_sha = prompt_template_sha256 or (
        _sha256(prompt_template.encode("utf-8")) if isinstance(prompt_template, str) else None
    )
    if value["measured_evidence_sha256"] != expected_evidence_sha:
        raise ContractError("stage plan measured-evidence SHA-256 mismatch")
    if value["prompt_template_sha256"] != expected_prompt_sha:
        raise ContractError("stage plan prompt SHA-256 mismatch")
    _require_sha256(value["measured_evidence_sha256"], "measured evidence")
    _require_sha256(value["prompt_template_sha256"], "prompt")
    expected_attempts = build_stage_attempt_plan(measured_evidence, prompt_template)
    if value["attempts"] != expected_attempts:
        raise ContractError("stage plan attempts differ from deterministic recomputation")
    expected_controls = _control_preflight(expected_attempts, measured_evidence)
    if value["control_preflight"] != expected_controls:
        raise ContractError("stage plan control preflight differs from recomputation")
    return value


def stage_attempt_plan_artifact_bytes(value: object) -> bytes:
    """Return deterministic bytes after exact artifact validation of shape."""

    _reject_nonfinite(value)
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        raise ContractError("stage plan artifact must be validated before serialization")
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def load_stage_attempt_plan_artifact(
    path: str | os.PathLike[str],
    measured_evidence: object,
    prompt_template: str,
    *,
    measured_evidence_sha256: str | None = None,
    prompt_template_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    """Load one bounded regular JSON artifact and verify its deterministic bytes."""

    prohibit_legacy_diagnostic_active_path("load_stage_attempt_plan_artifact")

    try:
        value, snapshot = load_strict_json_snapshot(path, max_bytes=_MAX_JSON_BYTES)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    validate_stage_attempt_plan_artifact(
        value,
        measured_evidence,
        prompt_template,
        measured_evidence_sha256=measured_evidence_sha256,
        prompt_template_sha256=prompt_template_sha256,
    )
    expected_bytes = stage_attempt_plan_artifact_bytes(value)
    if (
        len(expected_bytes) != snapshot["bytes"]
        or _sha256(expected_bytes) != snapshot["sha256"]
    ):
        raise ContractError("stage attempt-plan artifact bytes are not canonical")
    return value, str(snapshot["sha256"])


def write_stage_attempt_plan_artifact(path: str | os.PathLike[str], value: object) -> Path:
    """Atomically publish a canonical artifact without clobbering an existing file."""

    prohibit_legacy_diagnostic_active_path("write_stage_attempt_plan_artifact")

    output = Path(path)
    parent = output.parent
    try:
        parent_stat = os.lstat(parent)
    except OSError as exc:
        raise ContractError("stage plan output parent is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_ISLNK(parent_stat.st_mode)
        or parent_stat.st_uid != os.getuid()
    ):
        raise ContractError("stage plan output parent must be an owned directory")
    if os.path.lexists(output):
        raise ContractError("stage plan output must be absent")
    payload = stage_attempt_plan_artifact_bytes(value)
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise ContractError("stage plan output or temporary is occupied") from exc
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return output

"""Post-hoc scoring for separated synthetic vial-decision labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from pathlib import Path
import re

from .deploy_input_policy import validate_deploy_input_fields
from .render_process import load_strict_json_snapshot
from .vision_measurement import _publish_json_no_clobber


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREDICTIONS = frozenset({"keep", "discard", "insufficient_evidence"})
_EXPECTED = frozenset({"keep", "discard"})
_MEASURED_ORACLE_FIELDS = frozenset(
    {
        "synthetic_profile_id",
        "prescribed_leading_residual_volume_ml",
        "prescribed_leading_residual_particle_count_per_dispense",
        "expected_disposition",
        "label_source",
        "render_schedule_sha256",
        "excluded_from_model_inputs",
        "excluded_from_deploy_inputs",
    }
)
_LABEL_FIELDS = frozenset(
    {
        "case_id",
        "batch_id",
        "batch_role",
        "vial_ordinal",
        "synthetic_profile_id",
        "prescribed_leading_residual_volume_ml",
        "prescribed_leading_residual_particle_count_per_dispense",
        "expected_disposition",
        "label_source",
        "render_schedule_sha256",
        "excluded_from_model_inputs",
        "excluded_from_deploy_inputs",
    }
)
_LABEL_DENOMINATORS = {
    "keep_vials": 12,
    "discard_vials": 6,
    "reference_only_vials": 6,
    "scored_target_vials": 12,
    "scored_target_keep_vials": 6,
    "scored_target_discard_vials": 6,
}


def _sha256(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return value


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a nonempty string")
    return value


def _reject_nonfinite(value: object, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite(child, role)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_nonfinite(child, role)


def _reject_measured_oracle_fields(value: object) -> None:
    if isinstance(value, Mapping):
        leaked = sorted(set(value).intersection(_MEASURED_ORACLE_FIELDS))
        if leaked:
            raise ValueError(
                f"measured evidence contains forbidden oracle field: {leaked[0]}"
            )
        for child in value.values():
            _reject_measured_oracle_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_measured_oracle_fields(child)


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def _validate_measured(value: object) -> tuple[list[Mapping[str, object]], str]:
    if not isinstance(value, Mapping):
        raise TypeError("measured evidence must be an object")
    _reject_nonfinite(value, "measured evidence")
    _reject_measured_oracle_fields(value)
    try:
        validate_deploy_input_fields(value, require_allowlist=False)
    except ValueError as error:
        raise ValueError("measured evidence contains a forbidden oracle field") from error
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "measured_evidence"
        or value.get("data_origin") != "twin"
        or value.get("real_data_used") is not False
    ):
        raise ValueError("measured evidence is not the synthetic schema-v1 artifact")
    _sha256(value.get("source_model_input_sha256"), "source model-input hash")
    capture_hashes = value.get("source_capture_manifest_sha256")
    if (
        not isinstance(capture_hashes, list)
        or len(capture_hashes) != 6
        or len(set(capture_hashes)) != 6
    ):
        raise ValueError("measured evidence must bind six unique capture manifests")
    for digest in capture_hashes:
        _sha256(digest, "source capture-manifest hash")
    decisions = value.get("decision_records")
    if not isinstance(decisions, list) or len(decisions) != 12:
        raise ValueError("measured evidence must contain exactly 12 decisions")
    denominators = value.get("denominators")
    if not isinstance(denominators, Mapping):
        raise ValueError("measured evidence denominators are missing")
    if denominators.get("vial_decisions") != 12:
        raise ValueError("measured evidence decision denominator drifted")

    ids: set[str] = set()
    keys: set[tuple[str, int]] = set()
    families: set[str] = set()
    valid_count = 0
    abstained_count = 0
    checked: list[Mapping[str, object]] = []
    for row in decisions:
        if not isinstance(row, Mapping):
            raise ValueError("measured decision rows must be objects")
        decision_id = _text(row.get("decision_id"), "decision_id")
        batch_id = _text(row.get("batch_id"), "decision batch_id")
        family_id = _text(row.get("family_id"), "decision family_id")
        ordinal = row.get("vial_position")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in (1, 2, 3):
            raise ValueError("decision vial_position must be 1, 2, or 3")
        if row.get("vial_id") != f"V{ordinal}":
            raise ValueError("decision vial_id and vial_position differ")
        if row.get("cohort") != "scored_target" or row.get("partition") != "train":
            raise ValueError("only scored-target train decisions may be scored")
        disposition = row.get("disposition")
        if disposition not in _PREDICTIONS:
            raise ValueError("decision disposition is invalid")
        if decision_id in ids or (batch_id, ordinal) in keys:
            raise ValueError("measured decisions contain duplicate identities")
        ids.add(decision_id)
        keys.add((batch_id, ordinal))
        families.add(family_id)
        valid_count += disposition != "insufficient_evidence"
        abstained_count += disposition == "insufficient_evidence"
        checked.append(row)
    if len(families) != 1:
        raise ValueError("measured decisions must belong to one train family")
    if (
        denominators.get("valid_vial_decisions") != valid_count
        or denominators.get("abstained_vial_decisions") != abstained_count
    ):
        raise ValueError("measured evidence coverage denominators contradict decisions")
    return checked, next(iter(families))


def _validate_labels(
    value: object, measured_family_id: str
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    if not isinstance(value, Mapping):
        raise TypeError("label provenance must be an object")
    _reject_nonfinite(value, "label provenance")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "synthetic_train_label_provenance_v2"
        or value.get("status") != "unrendered"
        or value.get("family_id") != measured_family_id
    ):
        raise ValueError("label provenance does not match the measured train family")
    if value.get("denominators") != _LABEL_DENOMINATORS:
        raise ValueError("label provenance denominators drifted from 6 keep / 6 discard")
    rows = value.get("vial_labels")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("label provenance must contain exactly 18 vial rows")

    cases: set[str] = set()
    identities: set[tuple[str, int]] = set()
    schedule_hashes: set[str] = set()
    checked: list[Mapping[str, object]] = []
    role_batches: dict[str, set[str]] = {
        "reference_only": set(),
        "scored_target": set(),
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _LABEL_FIELDS:
            raise ValueError("label provenance row fields are malformed")
        case_id = _text(row.get("case_id"), "label case_id")
        batch_id = _text(row.get("batch_id"), "label batch_id")
        role = row.get("batch_role")
        if role not in role_batches:
            raise ValueError("label batch_role is invalid")
        ordinal = row.get("vial_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in (1, 2, 3):
            raise ValueError("label vial_ordinal must be 1, 2, or 3")
        _text(row.get("synthetic_profile_id"), "synthetic_profile_id")
        expected = row.get("expected_disposition")
        if expected not in _EXPECTED:
            raise ValueError("label expected_disposition is invalid")
        residual = row.get("prescribed_leading_residual_volume_ml")
        particles = row.get("prescribed_leading_residual_particle_count_per_dispense")
        if (
            isinstance(residual, bool)
            or not isinstance(residual, (int, float))
            or float(residual) < 0.0
            or isinstance(particles, bool)
            or not isinstance(particles, int)
            or particles < 0
        ):
            raise ValueError("label residual oracle is invalid")
        rule_expected = "keep" if float(residual) == 0.0 else "discard"
        if expected != rule_expected or (particles == 0) != (expected == "keep"):
            raise ValueError("label disposition contradicts its prescribed residual")
        if (
            row.get("label_source") != "prescribed_synthetic_latent_oracle"
            or row.get("excluded_from_model_inputs") is not True
            or row.get("excluded_from_deploy_inputs") is not True
        ):
            raise ValueError("label separation contract is not explicit")
        schedule_hash = _sha256(
            row.get("render_schedule_sha256"), "render schedule hash"
        )
        identity = (batch_id, ordinal)
        if case_id in cases or identity in identities or schedule_hash in schedule_hashes:
            raise ValueError("label provenance contains duplicate identities")
        cases.add(case_id)
        identities.add(identity)
        schedule_hashes.add(schedule_hash)
        role_batches[str(role)].add(batch_id)
        checked.append(row)
    if len(role_batches["reference_only"]) != 2 or len(role_batches["scored_target"]) != 4:
        raise ValueError("label provenance batch denominator drifted")
    if any(
        {int(row["vial_ordinal"]) for row in checked if row["batch_id"] == batch_id}
        != {1, 2, 3}
        for batch_id in role_batches["reference_only"] | role_batches["scored_target"]
    ):
        raise ValueError("each label batch must contain vials 1, 2, and 3")
    scored = [row for row in checked if row["batch_role"] == "scored_target"]
    reference = [row for row in checked if row["batch_role"] == "reference_only"]
    if Counter(row["expected_disposition"] for row in scored) != {
        "keep": 6,
        "discard": 6,
    }:
        raise ValueError("scored label balance differs from 6 keep / 6 discard")
    return scored, reference


def score_synthetic_stage_decisions(
    measured_evidence: object,
    label_provenance: object,
    *,
    measured_evidence_sha256: str,
    label_provenance_sha256: str,
) -> dict[str, object]:
    """Score immutable decisions only after their separated labels are supplied."""

    measured_hash = _sha256(measured_evidence_sha256, "measured evidence hash")
    label_hash = _sha256(label_provenance_sha256, "label provenance hash")
    decisions, family_id = _validate_measured(measured_evidence)
    labels, reference_labels = _validate_labels(label_provenance, family_id)
    decision_by_key = {
        (str(row["batch_id"]), int(row["vial_position"])): row
        for row in decisions
    }
    label_by_key = {
        (str(row["batch_id"]), int(row["vial_ordinal"])): row
        for row in labels
    }
    if set(decision_by_key) != set(label_by_key):
        missing = sorted(set(label_by_key).difference(decision_by_key))
        extra = sorted(set(decision_by_key).difference(label_by_key))
        raise ValueError(
            f"decision and scored-label IDs differ: missing={missing!r}, extra={extra!r}"
        )

    confusion = {
        f"expected_{expected}_{predicted}": 0
        for expected in ("keep", "discard")
        for predicted in ("predicted_keep", "predicted_discard", "abstained")
    }
    case_results: list[dict[str, object]] = []
    for key, label in label_by_key.items():
        decision = decision_by_key[key]
        expected = str(label["expected_disposition"])
        observed = str(decision["disposition"])
        prediction_key = (
            "abstained"
            if observed == "insufficient_evidence"
            else f"predicted_{observed}"
        )
        confusion[f"expected_{expected}_{prediction_key}"] += 1
        outcome = (
            "abstained"
            if observed == "insufficient_evidence"
            else "correct"
            if observed == expected
            else "incorrect"
        )
        case_results.append(
            {
                "case_id": label["case_id"],
                "decision_id": decision["decision_id"],
                "batch_id": label["batch_id"],
                "vial_ordinal": label["vial_ordinal"],
                "expected_disposition": expected,
                "observed_disposition": observed,
                "outcome": outcome,
            }
        )
    case_results.sort(key=lambda row: str(row["case_id"]))
    covered = sum(row["outcome"] != "abstained" for row in case_results)
    correct = sum(row["outcome"] == "correct" for row in case_results)
    keep_correct = confusion["expected_keep_predicted_keep"]
    discard_correct = confusion["expected_discard_predicted_discard"]
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_stage_decision_score",
        "source_sha256": {
            "measured_evidence": measured_hash,
            "label_provenance": label_hash,
        },
        "family_id": family_id,
        "partition": "train",
        "denominators": {
            "oracle_rows": len(labels) + len(reference_labels),
            "reference_only_rows": len(reference_labels),
            "scored_target_rows": len(labels),
            "expected_keep": 6,
            "expected_discard": 6,
            "decisions": len(decisions),
            "covered_decisions": covered,
            "abstained_decisions": len(decisions) - covered,
        },
        "confusion": confusion,
        "metrics": {
            "coverage": _ratio(covered, len(decisions)),
            "exact_match_all": _ratio(correct, len(decisions)),
            "covered_accuracy": _ratio(correct, covered),
            "keep_recall": _ratio(keep_correct, 6),
            "discard_recall": _ratio(discard_correct, 6),
        },
        "case_results": case_results,
        "data_origin": "twin",
        "real_data_used": False,
        "claim_boundary": (
            "Synthetic train-family scoring only; prescribed residue labels do not "
            "define a real defect threshold, cause, plant action, or sim-to-real result."
        ),
    }


def score_synthetic_stage_decision_files(
    measured_evidence_path: object,
    label_provenance_path: object,
    output_path: object,
    *,
    expected_measured_evidence_sha256: str,
    expected_label_provenance_sha256: str,
) -> dict[str, object]:
    """Load, hash-bind and publish one post-hoc synthetic score artifact."""

    expected_measured = _sha256(
        expected_measured_evidence_sha256, "expected measured evidence hash"
    )
    expected_labels = _sha256(
        expected_label_provenance_sha256, "expected label provenance hash"
    )
    measured, measured_snapshot = load_strict_json_snapshot(
        Path(measured_evidence_path), max_bytes=64 * 1024 * 1024
    )
    labels, label_snapshot = load_strict_json_snapshot(
        Path(label_provenance_path), max_bytes=16 * 1024 * 1024
    )
    if measured_snapshot["sha256"] != expected_measured:
        raise ValueError("measured evidence SHA-256 mismatch")
    if label_snapshot["sha256"] != expected_labels:
        raise ValueError("label provenance SHA-256 mismatch")
    score = score_synthetic_stage_decisions(
        measured,
        labels,
        measured_evidence_sha256=expected_measured,
        label_provenance_sha256=expected_labels,
    )
    _publish_json_no_clobber(Path(output_path), score)
    return score


__all__ = [
    "score_synthetic_stage_decision_files",
    "score_synthetic_stage_decisions",
]

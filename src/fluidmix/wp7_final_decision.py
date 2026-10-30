"""Pure WP7 final decision from runtime camera colour and synthetic donors.

This module consumes factory-bound development-reference queries. It does not open
images, stores, one-shot inputs, labels, recipe volumes, or model answers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import re

from fluidmix.wp7_development_reference import (
    DONOR_BATCHES,
    DevelopmentReferenceQuery,
    assert_development_reference_citation,
)


_TARGET_TOKEN = object()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_FIELDS = {
    "evidence_id", "measurement_id", "image_id", "vial_id", "recipe_id",
    "stage_id", "stage_ordinal", "tank_id", "median_lab", "validity",
    "limit_codes", "source",
}


class WP7FinalDecisionFatalError(ValueError):
    """A trusted decision input drifted or was structurally corrupt."""


class RuntimeCameraStage:
    """Factory-bound identity and colour measurement for one programmed stage."""

    __slots__ = (*sorted(_TARGET_FIELDS), "_factory_token", "_sealed")

    def __init__(self, value: Mapping[str, object], *, _factory_token: object) -> None:
        if _factory_token is not _TARGET_TOKEN:
            raise TypeError("runtime camera stages are factory-only")
        for field in _TARGET_FIELDS:
            stored = tuple(value[field]) if field == "limit_codes" else value[field]
            object.__setattr__(self, field, stored)
        object.__setattr__(self, "_factory_token", _TARGET_TOKEN)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("runtime camera stages are immutable")
        object.__setattr__(self, name, value)


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise WP7FinalDecisionFatalError(code)
    return value


def _lab(value: object, code: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise WP7FinalDecisionFatalError(code)
    channels: list[float] = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, Real):
            raise WP7FinalDecisionFatalError(code)
        number = float(channel)
        if not math.isfinite(number):
            raise WP7FinalDecisionFatalError(code)
        channels.append(number)
    return (channels[0], channels[1], channels[2])


def bind_runtime_camera_stage(value: object) -> RuntimeCameraStage:
    """Validate and bind one trusted runtime camera-stage measurement."""

    if not isinstance(value, Mapping) or set(value) != _TARGET_FIELDS:
        raise WP7FinalDecisionFatalError("runtime_camera_stage_fields_invalid")
    for field in (
        "evidence_id", "measurement_id", "image_id", "vial_id", "recipe_id",
        "stage_id", "tank_id",
    ):
        _text(value[field], f"runtime_camera_stage_{field}_invalid")
    if type(value["stage_ordinal"]) is not int or value["stage_ordinal"] <= 0:
        raise WP7FinalDecisionFatalError("runtime_camera_stage_ordinal_invalid")
    if value["source"] != "runtime_camera":
        raise WP7FinalDecisionFatalError("runtime_camera_stage_source_invalid")
    if value["validity"] not in {"valid", "invalid"}:
        raise WP7FinalDecisionFatalError("runtime_camera_stage_validity_invalid")
    codes = value["limit_codes"]
    if not isinstance(codes, (list, tuple)) or any(
        not isinstance(code, str) or not code for code in codes
    ):
        raise WP7FinalDecisionFatalError("runtime_camera_stage_limit_codes_invalid")
    if value["validity"] == "valid":
        if codes:
            raise WP7FinalDecisionFatalError("valid_runtime_camera_stage_has_limits")
        _lab(value["median_lab"], "runtime_camera_stage_lab_invalid")
    elif value["median_lab"] is not None or not codes:
        raise WP7FinalDecisionFatalError("invalid_runtime_camera_stage_contract_invalid")
    normalized = dict(value)
    if value["median_lab"] is not None:
        normalized["median_lab"] = _lab(
            value["median_lab"], "runtime_camera_stage_lab_invalid"
        )
    return RuntimeCameraStage(normalized, _factory_token=_TARGET_TOKEN)


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    result = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    if not math.isfinite(result):
        raise WP7FinalDecisionFatalError("delta_e76_nonfinite")
    return result


def _target_binding(target: RuntimeCameraStage) -> dict[str, object]:
    return {
        "evidence_id": target.evidence_id,
        "measurement_id": target.measurement_id,
        "image_id": target.image_id,
        "vial_id": target.vial_id,
        "recipe_id": target.recipe_id,
        "stage_id": target.stage_id,
        "stage_ordinal": target.stage_ordinal,
        "tank_id": target.tank_id,
        "source": target.source,
        "median_lab": None if target.median_lab is None else list(target.median_lab),
    }


def _insufficient_stage(
    target: RuntimeCameraStage, codes: Sequence[str]
) -> dict[str, object]:
    return {
        **_target_binding(target),
        "validity": "insufficient_evidence",
        "stage_outlier": None,
        "target_delta_e76": None,
        "peer_delta_e76": None,
        "threshold_delta_e76": None,
        "reference_center_lab": None,
        "donors": [],
        "query_sha256": None,
        "donor_store_snapshot_sha256": None,
        "configuration_sha256": None,
        "limit_codes": list(dict.fromkeys(codes)),
        "unit": "delta_e76",
    }


def _donor_binding(member: object) -> dict[str, object]:
    return {
        "evidence_id": member.evidence_id,
        "image_id": member.image_id,
        "batch_id": member.batch_id,
        "run_id": member.run_id,
        "recipe_id": member.recipe_id,
        "stage_ordinal": member.stage_ordinal,
        "tank_id": member.tank_id,
        "median_lab": list(member.median_lab),
    }


def _evaluate_stage(
    target: RuntimeCameraStage, query: object | None
) -> dict[str, object]:
    if query is None:
        return _insufficient_stage(target, ["unsupported_development_selector"])
    if not isinstance(query, DevelopmentReferenceQuery):
        raise WP7FinalDecisionFatalError("development_query_type_invalid")
    try:
        filters = query.filters
        if query.query_kind != "development_reference_bank":
            raise WP7FinalDecisionFatalError("development_query_kind_invalid")
        if _SHA256.fullmatch(str(query.query_sha256)) is None:
            raise WP7FinalDecisionFatalError("query_sha256_invalid")
        if set(filters) != {
            "recipe_id", "stage_id", "stage_ordinal", "tank_id",
            "configuration_sha256", "threshold_delta_e76", "donor_batch_ids",
            "donor_store_snapshot_sha256", "target_evidence_id", "target_stage_id",
        }:
            raise WP7FinalDecisionFatalError("development_query_filters_invalid")
        selector = (target.recipe_id, target.stage_ordinal, target.tank_id)
        if (
            (filters["recipe_id"], filters["stage_ordinal"], filters["tank_id"])
            != selector
            or filters["target_evidence_id"] != target.evidence_id
            or filters["target_stage_id"] != target.stage_id
            or filters["donor_batch_ids"] != list(DONOR_BATCHES)
        ):
            raise WP7FinalDecisionFatalError("development_query_target_binding_invalid")
        config_sha = filters["configuration_sha256"]
        snapshot_sha = filters["donor_store_snapshot_sha256"]
        if not isinstance(config_sha, str) or _SHA256.fullmatch(config_sha) is None:
            raise WP7FinalDecisionFatalError("configuration_sha256_invalid")
        if not isinstance(snapshot_sha, str) or _SHA256.fullmatch(snapshot_sha) is None:
            raise WP7FinalDecisionFatalError("donor_store_snapshot_sha256_invalid")
        if len(query.members) != 2:
            raise WP7FinalDecisionFatalError("development_donor_denominator_invalid")
        for member in query.members:
            assert_development_reference_citation(query, member.evidence_id, member.image_id)
    except WP7FinalDecisionFatalError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise WP7FinalDecisionFatalError("development_query_binding_corrupt") from error

    members = list(query.members)
    if (
        {member.batch_id for member in members} != set(DONOR_BATCHES)
        or len({member.run_id for member in members}) != 2
        or len({member.evidence_id for member in members}) != 2
        or len({member.image_id for member in members}) != 2
        or any(
            member.recipe_id != target.recipe_id
            or member.stage_ordinal != target.stage_ordinal
            or member.tank_id != target.tank_id
            or member.split_role != "development_reference"
            or member.donor_store_snapshot_sha256 != snapshot_sha
            or member.evidence_id == target.evidence_id
            or member.image_id == target.image_id
            for member in members
        )
    ):
        raise WP7FinalDecisionFatalError("development_donor_identity_invalid")
    donor_labs = [
        _lab(member.median_lab, "development_donor_lab_invalid") for member in members
    ]
    peer_delta = _distance(donor_labs[0], donor_labs[1])
    donor_rows = [_donor_binding(member) for member in members]
    if peer_delta > 2.0:
        return {
            **_insufficient_stage(target, ["development_peer_delta_above_limit"]),
            "peer_delta_e76": peer_delta,
            "query_sha256": query.query_sha256,
            "donor_store_snapshot_sha256": snapshot_sha,
            "configuration_sha256": config_sha,
            "donors": donor_rows,
        }
    center = tuple(
        (donor_labs[0][channel] + donor_labs[1][channel]) / 2.0
        for channel in range(3)
    )
    threshold = peer_delta / 2.0 + 1.0
    configured_threshold = filters["threshold_delta_e76"]
    if (
        isinstance(configured_threshold, bool)
        or not isinstance(configured_threshold, Real)
        or not math.isfinite(float(configured_threshold))
        or float(configured_threshold) != threshold
    ):
        raise WP7FinalDecisionFatalError("configuration_threshold_binding_invalid")
    if target.validity != "valid":
        return {
            **_insufficient_stage(
                target, ["target_measurement_invalid", *target.limit_codes]
            ),
            "peer_delta_e76": peer_delta,
            "threshold_delta_e76": threshold,
            "reference_center_lab": list(center),
            "donors": donor_rows,
            "query_sha256": query.query_sha256,
            "donor_store_snapshot_sha256": snapshot_sha,
            "configuration_sha256": config_sha,
        }
    target_delta = _distance(target.median_lab, center)
    return {
        **_target_binding(target),
        "validity": "valid",
        "stage_outlier": target_delta > threshold,
        "target_delta_e76": target_delta,
        "peer_delta_e76": peer_delta,
        "threshold_delta_e76": threshold,
        "reference_center_lab": list(center),
        "donors": donor_rows,
        "query_sha256": query.query_sha256,
        "donor_store_snapshot_sha256": snapshot_sha,
        "configuration_sha256": config_sha,
        "limit_codes": [],
        "unit": "delta_e76",
    }


def _explanation(disposition: str, stages: Sequence[Mapping[str, object]]) -> dict[str, object]:
    comparisons: list[str] = []
    citations: list[dict[str, object]] = []
    for stage in stages:
        citations.append({
            "role": "target", "stage_ordinal": stage["stage_ordinal"],
            "evidence_id": stage["evidence_id"],
            "measurement_id": stage["measurement_id"], "image_id": stage["image_id"],
        })
        for donor in stage["donors"]:
            citations.append({
                "role": "synthetic_development_donor",
                "stage_ordinal": stage["stage_ordinal"],
                "evidence_id": donor["evidence_id"], "image_id": donor["image_id"],
                "batch_id": donor["batch_id"],
            })
        if stage["validity"] == "valid":
            relation = "above" if stage["stage_outlier"] else "at or below"
            comparisons.append(
                f"stage {stage['stage_ordinal']} target {stage['evidence_id']} "
                f"camera-colour distance {stage['target_delta_e76']:.6g} is {relation} "
                f"threshold {stage['threshold_delta_e76']:.6g}"
            )
        else:
            comparisons.append(
                f"stage {stage['stage_ordinal']} target {stage['evidence_id']} has "
                "insufficient camera-colour evidence"
            )
    text = (
        f"Disposition {disposition}. Camera-colour versus synthetic-reference comparison: "
        + "; ".join(comparisons)
        + ". Cause, physical defect, carryover, next-vial effect, and plant action are "
        "unverified."
    )
    return {
        "template_id": "wp7-camera-colour-synthetic-reference-v1",
        "model": None,
        "text": text,
        "citations": citations,
        "claim_boundary": (
            "Camera colour versus synthetic development reference only; cause, physical "
            "defect, carryover, next-vial effect, and plant action are unverified."
        ),
    }


def decision_denominators(stages: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Summarize explicit stage and donor denominators for an operator record."""

    return {
        "programmed_stage_count": len(stages),
        "valid_stage_count": sum(stage["validity"] == "valid" for stage in stages),
        "outlier_stage_count": sum(stage["stage_outlier"] is True for stage in stages),
        "bound_donor_count": sum(len(stage["donors"]) for stage in stages),
        "required_donors_per_supported_stage": 2,
    }


def decide_wp7_final(
    decision_id: str,
    targets: Sequence[RuntimeCameraStage],
    reference_queries: Mapping[tuple[str, int, str], DevelopmentReferenceQuery | None],
) -> dict[str, object]:
    """Return the deterministic final WP7 vial decision and explanation."""

    _text(decision_id, "decision_id_invalid")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)) or not targets:
        raise WP7FinalDecisionFatalError("targets_invalid")
    if not isinstance(reference_queries, Mapping):
        raise WP7FinalDecisionFatalError("reference_queries_invalid")
    checked = list(targets)
    if any(
        not isinstance(target, RuntimeCameraStage)
        or target._factory_token is not _TARGET_TOKEN
        for target in checked
    ):
        raise WP7FinalDecisionFatalError("untrusted_runtime_camera_stage")
    first = checked[0]
    if any(
        target.vial_id != first.vial_id or target.recipe_id != first.recipe_id
        for target in checked[1:]
    ):
        raise WP7FinalDecisionFatalError("targets_cross_vial_or_recipe")
    if len({target.stage_ordinal for target in checked}) != len(checked):
        raise WP7FinalDecisionFatalError("target_stage_ordinal_duplicate")
    for identity in ("evidence_id", "measurement_id", "image_id"):
        if len({getattr(target, identity) for target in checked}) != len(checked):
            raise WP7FinalDecisionFatalError(f"target_{identity}_duplicate")
    checked.sort(key=lambda target: target.stage_ordinal)
    stages = [
        _evaluate_stage(
            target,
            reference_queries.get((target.recipe_id, target.stage_ordinal, target.tank_id)),
        )
        for target in checked
    ]
    if any(stage["validity"] == "valid" and stage["stage_outlier"] for stage in stages):
        disposition = "discard"
    elif all(
        stage["validity"] == "valid" and stage["stage_outlier"] is False
        for stage in stages
    ):
        disposition = "keep"
    else:
        disposition = "insufficient_evidence"
    limit_codes = list(dict.fromkeys(
        str(code) for stage in stages for code in stage["limit_codes"]
    ))
    denominators = decision_denominators(stages)
    explanation = _explanation(disposition, stages)
    source_bindings = {
        "targets": [_target_binding(target) for target in checked],
        "development_references": [{
            "stage_ordinal": stage["stage_ordinal"],
            "query_sha256": stage["query_sha256"],
            "donor_store_snapshot_sha256": stage["donor_store_snapshot_sha256"],
            "configuration_sha256": stage["configuration_sha256"],
            "donors": stage["donors"],
        } for stage in stages],
    }
    operator_record = {
        "schema_version": "wp7-operator-record-v1", "decision_id": decision_id,
        "disposition": disposition, "model": None, "denominators": denominators,
        "source_bindings": source_bindings, "limit_codes": limit_codes,
        "explanation": explanation,
    }
    return {
        "schema_version": "wp7-final-decision-v1", "decision_id": decision_id,
        "disposition": disposition, "model": None, "stages": stages,
        "denominators": denominators, "limit_codes": limit_codes,
        "explanation": explanation, "operator_record": operator_record,
    }


__all__ = [
    "RuntimeCameraStage", "WP7FinalDecisionFatalError", "bind_runtime_camera_stage",
    "decision_denominators", "decide_wp7_final",
]

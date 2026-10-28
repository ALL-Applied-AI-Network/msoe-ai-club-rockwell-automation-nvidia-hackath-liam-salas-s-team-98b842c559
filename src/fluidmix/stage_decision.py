"""Deterministic keep/discard evidence from precomputed synthetic stage statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
from numbers import Real
from statistics import median


@dataclass(frozen=True)
class StageStatistic:
    """One pipeline-produced colour statistic for a vial at one fill stage."""

    evidence_id: str
    measurement_id: str
    image_id: str
    batch_id: str
    vial_id: str
    recipe_id: str
    data_origin: str
    split_role: str
    split_version: str
    stage_id: str
    stage_ordinal: int
    lab: tuple[float, float, float] | None
    validity: str = "valid"
    limit_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageReference:
    """One candidate peer and its non-oracle reference provenance."""

    statistic: StageStatistic
    origin: str


@dataclass(frozen=True)
class FrozenStageDecisionConfig:
    """Thresholds selected outside this decision function and frozen by ID."""

    config_id: str
    minimum_peer_count: int
    minimum_distinct_reference_batches: int
    tight_cluster_max_diameter_delta_e76: float
    threshold_floor_delta_e76: float
    threshold_margin_delta_e76: float


def decide_vial_stages(
    decision_id: str,
    targets: Sequence[StageStatistic],
    references_by_stage: Mapping[str, Sequence[StageReference]],
    config: FrozenStageDecisionConfig,
) -> dict[str, object]:
    """Return a grounded vial disposition from stage statistics and explicit peers."""

    _validate_config(config)
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError("decision_id must be a nonempty string")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise TypeError("targets must be a sequence")
    if not targets:
        raise ValueError("targets must not be empty")
    if not isinstance(references_by_stage, Mapping):
        raise TypeError("references_by_stage must be a mapping")
    checked_targets = list(targets)
    if any(not isinstance(target, StageStatistic) for target in checked_targets):
        raise TypeError("every target must be a StageStatistic")
    for target in checked_targets:
        _validate_statistic(target)
    context_fields = (
        "batch_id",
        "vial_id",
        "recipe_id",
        "data_origin",
        "split_role",
        "split_version",
    )
    if any(
        getattr(target, field) != getattr(checked_targets[0], field)
        for target in checked_targets[1:]
        for field in context_fields
    ):
        raise ValueError("target stages must describe a single vial and partition")
    if any(
        len({getattr(target, field) for target in checked_targets})
        != len(checked_targets)
        for field in ("evidence_id", "measurement_id", "image_id")
    ):
        raise ValueError("target evidence identities must be unique")
    stage_ids = [target.stage_id for target in checked_targets]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("target stage IDs must be unique")
    if sorted(target.stage_ordinal for target in checked_targets) != [1, 2, 3]:
        raise ValueError("target stage ordinals must be exactly 1, 2, and 3")
    if set(references_by_stage) != set(stage_ids):
        raise ValueError("reference stage IDs must exactly match target stage IDs")
    checked_targets.sort(key=lambda target: target.stage_ordinal)
    target_identities = {
        field: {getattr(target, field) for target in checked_targets}
        for field in ("evidence_id", "measurement_id", "image_id")
    }

    stages = [
        _evaluate_stage(
            target,
            references_by_stage[target.stage_id],
            config,
            target_identities,
        )
        for target in checked_targets
    ]
    valid = [stage for stage in stages if stage["validity"] == "valid"]
    if any(stage["stage_outlier"] is True for stage in valid):
        disposition = "discard"
        limit_codes: list[str] = []
    elif len(valid) == len(stages) and all(
        stage["stage_outlier"] is False for stage in valid
    ):
        disposition = "keep"
        limit_codes = []
    else:
        disposition = "insufficient_evidence"
        limit_codes = list(
            dict.fromkeys(
                str(code)
                for stage in stages
                if stage["validity"] != "valid"
                for code in stage["limit_codes"]
            )
        )
    return {
        "schema_version": 1,
        "decision_id": decision_id,
        "config_id": config.config_id,
        "disposition": disposition,
        "explanation_status": {
            "keep": "NO_DIFFERENCE",
            "discard": "DIFFERENCE",
            "insufficient_evidence": "UNKNOWN",
        }[disposition],
        "disposition_source": "deterministic_pipeline_stage_colour_evidence",
        "target": {
            field: getattr(checked_targets[0], field)
            for field in context_fields
        },
        "stages": stages,
        "limit_codes": limit_codes,
        "decision_timing": (
            "retrospective_after_complete_three_vial_batch_no_latency_claim"
        ),
        "real_data_used": False,
        "claim_boundary": (
            "Synthetic stage colour-distance evidence only; no physical carryover, "
            "defect, cause, plant-action, or sim-to-real claim."
        ),
    }


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def _validate_config(config: FrozenStageDecisionConfig) -> None:
    if not isinstance(config, FrozenStageDecisionConfig):
        raise TypeError("config must be a FrozenStageDecisionConfig")
    if not isinstance(config.config_id, str) or not config.config_id:
        raise ValueError("config_id must be a nonempty string")
    for name in ("minimum_peer_count", "minimum_distinct_reference_batches"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    _finite(
        config.tight_cluster_max_diameter_delta_e76,
        "tight_cluster_max_diameter_delta_e76",
        positive=True,
    )
    _finite(
        config.threshold_floor_delta_e76,
        "threshold_floor_delta_e76",
        positive=True,
    )
    _finite(
        config.threshold_margin_delta_e76,
        "threshold_margin_delta_e76",
        positive=True,
    )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _valid_lab(statistic: StageStatistic) -> tuple[float, float, float]:
    if statistic.lab is None or not isinstance(statistic.lab, tuple) or len(statistic.lab) != 3:
        raise ValueError("valid stage statistic must contain three Lab channels")
    return tuple(_finite(value, "Lab channel") for value in statistic.lab)  # type: ignore[return-value]


def _validate_statistic(statistic: StageStatistic) -> None:
    for field in (
        "evidence_id",
        "measurement_id",
        "image_id",
        "batch_id",
        "vial_id",
        "recipe_id",
        "data_origin",
        "split_role",
        "split_version",
        "stage_id",
    ):
        _text(getattr(statistic, field), field)
    if (
        isinstance(statistic.stage_ordinal, bool)
        or not isinstance(statistic.stage_ordinal, int)
        or statistic.stage_ordinal <= 0
    ):
        raise ValueError("stage_ordinal must be a positive integer")
    if statistic.validity not in {"valid", "invalid", "uncertain"}:
        raise ValueError("validity must be valid, invalid, or uncertain")
    if not isinstance(statistic.limit_codes, tuple) or any(
        not isinstance(code, str) or not code for code in statistic.limit_codes
    ):
        raise ValueError("limit_codes must be a tuple of nonempty strings")
    if statistic.validity == "valid":
        if statistic.limit_codes:
            raise ValueError("valid statistic must not have limit codes")
        _valid_lab(statistic)
    elif statistic.lab is not None:
        raise ValueError("invalid or uncertain statistic must not carry Lab values")


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    result = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    if not math.isfinite(result):
        raise ValueError("delta E76 is nonfinite")
    return result


def _insufficient_stage(
    target: StageStatistic,
    codes: Sequence[str],
) -> dict[str, object]:
    return {
        "evidence_id": target.evidence_id,
        "measurement_id": target.measurement_id,
        "target_image_id": target.image_id,
        "stage_id": target.stage_id,
        "stage_ordinal": target.stage_ordinal,
        "validity": "insufficient_evidence",
        "limit_codes": list(dict.fromkeys(codes)),
        "target_delta_e76": None,
        "threshold_delta_e76": None,
        "threshold_excess_delta_e76": None,
        "reference": None,
        "stage_outlier": None,
        "unit": "delta_e76",
    }


def _evaluate_stage(
    target: StageStatistic,
    references: Sequence[StageReference],
    config: FrozenStageDecisionConfig,
    target_identities: Mapping[str, set[str]],
) -> dict[str, object]:
    _validate_statistic(target)
    if target.data_origin != "twin":
        return _insufficient_stage(target, ["target_data_origin_not_twin"])
    if target.validity != "valid":
        return _insufficient_stage(
            target,
            ["target_measurement_invalid", *target.limit_codes],
        )
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise TypeError("stage references must be a sequence")
    codes: list[str] = []
    candidates: dict[str, tuple[StageReference, tuple[float, float, float]]] = {}
    evidence_ids: set[str] = set()
    measurement_ids: set[str] = set()
    image_ids: set[str] = set()
    same_batch_vial_ids: set[str] = set()
    for reference in references:
        if not isinstance(reference, StageReference):
            raise TypeError("every stage reference must be a StageReference")
        statistic = reference.statistic
        _validate_statistic(statistic)
        if reference.origin not in {"reference_bank", "later_same_batch"}:
            raise ValueError("reference origin is not allowed")
        if (
            statistic.evidence_id in evidence_ids
            or statistic.measurement_id in measurement_ids
            or statistic.image_id in image_ids
        ):
            codes.append("duplicate_reference_identity")
            continue
        evidence_ids.add(statistic.evidence_id)
        measurement_ids.add(statistic.measurement_id)
        image_ids.add(statistic.image_id)
        if any(
            getattr(statistic, field) != getattr(target, field)
            for field in (
                "recipe_id",
                "data_origin",
                "split_role",
                "split_version",
                "stage_id",
                "stage_ordinal",
            )
        ):
            codes.append(f"reference_context_mismatch:{statistic.evidence_id}")
            continue
        if statistic.validity != "valid":
            codes.extend(
                [
                    f"reference_measurement_invalid:{statistic.evidence_id}",
                    *(f"reference:{statistic.evidence_id}:{code}" for code in statistic.limit_codes),
                ]
            )
            continue
        if reference.origin == "later_same_batch" and statistic.batch_id != target.batch_id:
            codes.append(f"same_batch_origin_mismatch:{statistic.evidence_id}")
            continue
        if reference.origin == "later_same_batch":
            if statistic.vial_id == target.vial_id:
                codes.append(f"same_batch_context_reuses_target_vial:{statistic.evidence_id}")
                continue
            if statistic.vial_id in same_batch_vial_ids:
                codes.append(f"duplicate_same_batch_context_vial:{statistic.vial_id}")
                continue
            same_batch_vial_ids.add(statistic.vial_id)
            if len(same_batch_vial_ids) > 2:
                codes.append("same_batch_context_exceeds_two_vials")
        if reference.origin == "reference_bank" and statistic.batch_id == target.batch_id:
            codes.append(f"reference_bank_reuses_target_batch:{statistic.evidence_id}")
            continue
        if (
            statistic.evidence_id in target_identities["evidence_id"]
            or statistic.measurement_id in target_identities["measurement_id"]
            or statistic.image_id in target_identities["image_id"]
        ):
            codes.append("target_reused_as_reference")
            continue
        candidates[statistic.evidence_id] = (reference, _valid_lab(statistic))
    if codes:
        return _insufficient_stage(target, codes)
    if len(candidates) < config.minimum_peer_count:
        return _insufficient_stage(target, ["reference_peer_count_below_minimum"])

    ids = sorted(candidates)
    clusters: list[tuple[str, ...]] = []
    diameters: dict[tuple[str, ...], float] = {}
    for size in range(len(ids), config.minimum_peer_count - 1, -1):
        for subset in combinations(ids, size):
            diameter = max(
                (
                    _distance(candidates[left][1], candidates[right][1])
                    for left, right in combinations(subset, 2)
                ),
                default=0.0,
            )
            if diameter <= config.tight_cluster_max_diameter_delta_e76:
                clusters.append(subset)
                diameters[subset] = diameter
        if clusters:
            break
    if len(clusters) != 1:
        return _insufficient_stage(target, ["reference_mode_not_unique_or_too_small"])
    selected = clusters[0]
    reference_batch_ids = sorted(
        {
            candidates[evidence_id][0].statistic.batch_id
            for evidence_id in selected
            if candidates[evidence_id][0].origin == "reference_bank"
        }
    )
    if len(reference_batch_ids) < config.minimum_distinct_reference_batches:
        return _insufficient_stage(target, ["independent_reference_batches_below_minimum"])
    labs = [candidates[evidence_id][1] for evidence_id in selected]
    center = tuple(median(lab[channel] for lab in labs) for channel in range(3))
    radii = [_distance(lab, center) for lab in labs]
    median_radius = float(median(radii))
    max_radius = float(max(radii))
    mad_radius = float(median(abs(radius - median_radius) for radius in radii))
    threshold = max(
        config.threshold_floor_delta_e76,
        max_radius + config.threshold_margin_delta_e76,
    )
    target_delta = _distance(_valid_lab(target), center)
    outlier = target_delta > threshold
    peer_rows = [candidates[evidence_id][0] for evidence_id in selected]
    return {
        "evidence_id": target.evidence_id,
        "measurement_id": target.measurement_id,
        "target_image_id": target.image_id,
        "stage_id": target.stage_id,
        "stage_ordinal": target.stage_ordinal,
        "validity": "valid",
        "limit_codes": [],
        "target_delta_e76": target_delta,
        "threshold_delta_e76": threshold,
        "threshold_excess_delta_e76": target_delta - threshold,
        "reference": {
            "peer_evidence_ids": list(selected),
            "peer_measurement_ids": [row.statistic.measurement_id for row in peer_rows],
            "peer_image_ids": [row.statistic.image_id for row in peer_rows],
            "peer_origins": [row.origin for row in peer_rows],
            "peer_count": len(selected),
            "reference_batch_ids": reference_batch_ids,
            "reference_batch_count": len(reference_batch_ids),
            "same_batch_context_count": sum(
                row.origin == "later_same_batch" for row in peer_rows
            ),
            "center_lab": list(center),
            "median_radius_delta_e76": median_radius,
            "max_radius_delta_e76": max_radius,
            "mad_radius_delta_e76": mad_radius,
            "pairwise_diameter_delta_e76": diameters[selected],
            "unit": "delta_e76",
        },
        "stage_outlier": outlier,
        "unit": "delta_e76",
    }

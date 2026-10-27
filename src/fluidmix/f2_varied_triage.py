"""Relative three-vial evidence for varied synthetic stage measurements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

from fluidmix.deploy_input_policy import validate_deploy_input_fields


class VariedTriadInputError(ValueError):
    """The measured-evidence artifact failed the triad boundary."""


@dataclass(frozen=True)
class VariedTriadConfig:
    """Frozen relative-evidence thresholds; neither value proves cleanliness."""

    config_id: str
    tight_peer_pair_max_delta_e76: float
    target_outlier_margin_delta_e76: float


def decide_varied_run_triads(measured: object, config: VariedTriadConfig) -> dict[str, object]:
    """Return within-run evidence without treating a two-vial mode as clean."""

    _validate_config(config)
    if not isinstance(measured, Mapping):
        raise VariedTriadInputError("measurement_artifact_malformed")
    validate_deploy_input_fields(measured, require_allowlist=False)
    required_header = {
        "schema_version": 1,
        "artifact_type": "f2_varied_stage_measured_evidence",
        "status": "measured",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
    }
    if any(measured.get(key) != value for key, value in required_header.items()):
        raise VariedTriadInputError("measurement_artifact_header_invalid")
    raw_measurements = measured.get("stage_measurements")
    raw_pairs = measured.get("same_run_stage_pairs")
    if not isinstance(raw_measurements, list) or not raw_measurements:
        raise VariedTriadInputError("stage_measurements_missing")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise VariedTriadInputError("stage_pairs_missing")

    measurement_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    batches: dict[str, list[dict[str, object]]] = defaultdict(list)
    evidence_ids: set[str] = set()
    measurement_ids: set[str] = set()
    image_ids: set[str] = set()
    for raw in raw_measurements:
        row = _measurement_row(raw)
        key = (str(row["batch_id"]), str(row["vial_id"]), str(row["stage_id"]))
        if key in measurement_by_key:
            raise VariedTriadInputError("duplicate_stage_measurement")
        for identity_key, identities in (
            ("evidence_id", evidence_ids),
            ("measurement_id", measurement_ids),
            ("image_id", image_ids),
        ):
            identity = str(row[identity_key])
            if identity in identities:
                raise VariedTriadInputError(f"duplicate_{identity_key}")
            identities.add(identity)
        measurement_by_key[key] = row
        batches[str(row["batch_id"])].append(row)

    batch_contexts = {
        batch_id: _validate_batch(batch_id, rows)
        for batch_id, rows in sorted(batches.items())
    }
    run_ids = [str(context["run_id"]) for context in batch_contexts.values()]
    if len(set(run_ids)) != len(run_ids):
        raise VariedTriadInputError("run_reused_across_batches")

    pair_by_key: dict[tuple[str, str, frozenset[str]], dict[str, object]] = {}
    pair_ids: set[str] = set()
    for raw in raw_pairs:
        pair = _pair_row(raw)
        batch_id = str(pair["batch_id"])
        stage_id = str(pair["stage_id"])
        left_vial = str(pair["left_vial_id"])
        right_vial = str(pair["right_vial_id"])
        if left_vial == right_vial:
            raise VariedTriadInputError("pair_reuses_vial")
        key = (batch_id, stage_id, frozenset((left_vial, right_vial)))
        if key in pair_by_key:
            raise VariedTriadInputError("duplicate_stage_pair")
        pair_id = str(pair["pair_id"])
        if pair_id in pair_ids:
            raise VariedTriadInputError("duplicate_pair_id")
        pair_ids.add(pair_id)
        try:
            left = measurement_by_key[(batch_id, left_vial, stage_id)]
            right = measurement_by_key[(batch_id, right_vial, stage_id)]
        except KeyError as exc:
            raise VariedTriadInputError("pair_measurement_missing") from exc
        _validate_pair_binding(pair, left, right)
        pair_by_key[key] = pair

    expected_pair_keys = {
        (batch_id, stage_id, frozenset((left, right)))
        for batch_id, context in batch_contexts.items()
        for stage_id in context["stage_ids"]
        for left_index, left in enumerate(context["vial_ids"][:-1])
        for right in context["vial_ids"][left_index + 1 :]
    }
    if set(pair_by_key) != expected_pair_keys:
        raise VariedTriadInputError("stage_pair_set_mismatch")
    _validate_denominators(measured.get("denominators"), batch_contexts, raw_measurements, raw_pairs)

    decisions: list[dict[str, object]] = []
    relative_outlier_vials = 0
    for batch_id, context in batch_contexts.items():
        vial_ids = context["vial_ids"]
        for vial_id in vial_ids:
            peers = [peer for peer in vial_ids if peer != vial_id]
            stages: list[dict[str, object]] = []
            for stage_id in context["stage_ids"]:
                target = measurement_by_key[(batch_id, vial_id, stage_id)]
                peer_rows = [measurement_by_key[(batch_id, peer, stage_id)] for peer in peers]
                peer_pair = pair_by_key[(batch_id, stage_id, frozenset(peers))]
                target_pairs = [
                    pair_by_key[(batch_id, stage_id, frozenset((vial_id, peer)))]
                    for peer in peers
                ]
                stages.append(
                    _stage_assessment(target, peer_rows, peer_pair, target_pairs, config)
                )
            if any(stage["relative_status"] == "outlier_to_two_peer_mode" for stage in stages):
                relative = "outlier_to_two_peer_mode"
                relative_outlier_vials += 1
                codes = [
                    "within_run_reference_cleanliness_unverified",
                    "relative_outlier_not_directional",
                ]
            elif any(stage["relative_status"] == "peer_mode_not_tight" for stage in stages):
                relative = "peer_mode_not_tight"
                codes = [
                    "within_run_reference_cleanliness_unverified",
                    "peer_mode_not_tight",
                ]
            else:
                relative = "consistent_with_two_peer_mode"
                codes = [
                    "within_run_reference_cleanliness_unverified",
                    "two_peer_mode_cannot_verify_cleanliness",
                ]
            decisions.append(
                {
                    "decision_id": f"DEC-{batch_id}-{vial_id}",
                    "batch_id": batch_id,
                    "run_id": context["run_id"],
                    "recipe_id": context["recipe_id"],
                    "split_role": context["split_role"],
                    "split_version": context["split_version"],
                    "vial_id": vial_id,
                    "disposition": "insufficient_evidence",
                    "explanation_status": "UNKNOWN",
                    "relative_assessment": relative,
                    "stages": stages,
                    "limit_codes": codes,
                    "decision_timing": (
                        "retrospective_after_complete_three_vial_batch_no_latency_claim"
                    ),
                }
            )

    decisions.sort(
        key=lambda row: (
            str(row["split_role"]),
            str(row["run_id"]),
            str(row["batch_id"]),
            str(row["vial_id"]),
        )
    )
    config_record = {
        "config_id": config.config_id,
        "tight_peer_pair_max_delta_e76": config.tight_peer_pair_max_delta_e76,
        "target_outlier_margin_delta_e76": config.target_outlier_margin_delta_e76,
    }
    result = {
        "schema_version": 1,
        "artifact_type": "f2_varied_triad_decisions",
        "status": "assessed",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_measurement_sha256": _canonical_sha256(measured),
        "config": config_record,
        "config_sha256": _canonical_sha256(config_record),
        "reference_policy": {
            "scope": "other_two_vials_same_run_same_stage",
            "position_prior": False,
            "cleanliness_verified": False,
            "authorization": "relative_evidence_only_no_keep_or_discard",
        },
        "decision_records": decisions,
        "denominators": {
            "runs": len(batch_contexts),
            "vial_decisions": len(decisions),
            "keep": 0,
            "discard": 0,
            "insufficient_evidence": len(decisions),
            "relative_outlier_vials": relative_outlier_vials,
        },
        "claim_boundary": (
            "Synthetic within-run colour relations only. With three vials and no external "
            "clean reference, a two-vial appearance mode may be normal or shared-abnormal; "
            "therefore this boundary does not authorize keep, discard, physical-carryover, "
            "cause, plant-action, or sim-to-real claims."
        ),
    }
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VariedTriadInputError("output_nonfinite_or_unserializable") from exc
    return result


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VariedTriadInputError(f"{field}_invalid")
    return value


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VariedTriadInputError(f"{field}_invalid")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise VariedTriadInputError(f"{field}_invalid")
    return number


def _sha(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise VariedTriadInputError(f"{field}_invalid")
    return text


def _lab(value: object) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise VariedTriadInputError("median_lab_invalid")
    return tuple(_finite(channel, "median_lab") for channel in value)  # type: ignore[return-value]


def _validate_config(config: VariedTriadConfig) -> None:
    if not isinstance(config, VariedTriadConfig):
        raise TypeError("config must be a VariedTriadConfig")
    _text(config.config_id, "config_id")
    _finite(
        config.tight_peer_pair_max_delta_e76,
        "tight_peer_pair_max_delta_e76",
        positive=True,
    )
    _finite(
        config.target_outlier_margin_delta_e76,
        "target_outlier_margin_delta_e76",
        positive=True,
    )


def _measurement_row(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise VariedTriadInputError("stage_measurement_malformed")
    row = dict(raw)
    for field in (
        "evidence_id",
        "measurement_id",
        "image_id",
        "batch_id",
        "run_id",
        "recipe_id",
        "split_role",
        "split_version",
        "vial_id",
        "stage_id",
        "tank_id",
    ):
        _text(row.get(field), field)
    ordinal = row.get("stage_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise VariedTriadInputError("stage_ordinal_invalid")
    _lab(row.get("median_lab"))
    _sha(row.get("source_manifest_sha256"), "source_manifest_sha256")
    _sha(row.get("rgb_sha256"), "rgb_sha256")
    _sha(row.get("mask_sha256"), "mask_sha256")
    return row


def _pair_row(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise VariedTriadInputError("stage_pair_malformed")
    row = dict(raw)
    for field in (
        "pair_id",
        "batch_id",
        "run_id",
        "recipe_id",
        "split_role",
        "split_version",
        "stage_id",
        "tank_id",
        "left_vial_id",
        "right_vial_id",
        "left_evidence_id",
        "right_evidence_id",
    ):
        _text(row.get(field), field)
    ordinal = row.get("stage_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise VariedTriadInputError("stage_ordinal_invalid")
    _finite(row.get("delta_e76"), "delta_e76")
    if row.get("unit") != "delta_e76":
        raise VariedTriadInputError("pair_unit_invalid")
    for field in (
        "source_manifest_sha256",
        "left_rgb_sha256",
        "right_rgb_sha256",
        "left_mask_sha256",
        "right_mask_sha256",
    ):
        _sha(row.get(field), field)
    return row


def _validate_batch(batch_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    context_fields = (
        "run_id",
        "recipe_id",
        "split_role",
        "split_version",
        "source_manifest_sha256",
    )
    if any(row[field] != rows[0][field] for row in rows[1:] for field in context_fields):
        raise VariedTriadInputError("batch_context_mismatch")
    vial_ids = sorted({str(row["vial_id"]) for row in rows})
    if len(vial_ids) != 3:
        raise VariedTriadInputError("batch_must_contain_exactly_three_vials")
    stage_contexts: dict[str, tuple[int, str]] = {}
    for row in rows:
        stage_id = str(row["stage_id"])
        context = (int(row["stage_ordinal"]), str(row["tank_id"]))
        if stage_id in stage_contexts and stage_contexts[stage_id] != context:
            raise VariedTriadInputError("stage_context_mismatch")
        stage_contexts[stage_id] = context
    if len(stage_contexts) not in {2, 3}:
        raise VariedTriadInputError("active_stage_count_must_be_two_or_three")
    ordered_stages = sorted(stage_contexts, key=lambda stage: stage_contexts[stage][0])
    if [stage_contexts[stage][0] for stage in ordered_stages] != list(
        range(1, len(stage_contexts) + 1)
    ):
        raise VariedTriadInputError("stage_ordinals_not_contiguous")
    expected = {(vial, stage) for vial in vial_ids for stage in ordered_stages}
    observed = {(str(row["vial_id"]), str(row["stage_id"])) for row in rows}
    if observed != expected:
        raise VariedTriadInputError("vial_stage_set_mismatch")
    return {
        "batch_id": batch_id,
        **{field: rows[0][field] for field in context_fields},
        "vial_ids": vial_ids,
        "stage_ids": ordered_stages,
    }


def _validate_pair_binding(
    pair: Mapping[str, object],
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> None:
    for field in (
        "batch_id",
        "run_id",
        "recipe_id",
        "split_role",
        "split_version",
        "stage_id",
        "stage_ordinal",
        "tank_id",
        "source_manifest_sha256",
    ):
        if pair[field] != left[field] or pair[field] != right[field]:
            raise VariedTriadInputError("pair_context_mismatch")
    bindings = (
        ("left_evidence_id", left["evidence_id"]),
        ("right_evidence_id", right["evidence_id"]),
        ("left_rgb_sha256", left["rgb_sha256"]),
        ("right_rgb_sha256", right["rgb_sha256"]),
        ("left_mask_sha256", left["mask_sha256"]),
        ("right_mask_sha256", right["mask_sha256"]),
    )
    if any(pair[field] != expected for field, expected in bindings):
        raise VariedTriadInputError("pair_identity_binding_mismatch")
    expected_delta = math.dist(_lab(left["median_lab"]), _lab(right["median_lab"]))
    if not math.isclose(
        _finite(pair["delta_e76"], "delta_e76"),
        expected_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise VariedTriadInputError("pair_distance_mismatch")


def _validate_denominators(
    raw: object,
    contexts: Mapping[str, Mapping[str, object]],
    measurements: list[object],
    pairs: list[object],
) -> None:
    if not isinstance(raw, Mapping):
        raise VariedTriadInputError("measurement_denominators_missing")
    stage_counts = [len(context["stage_ids"]) for context in contexts.values()]
    expected = {
        "source_manifests": len(contexts),
        "runs": len(contexts),
        "vials": len(contexts) * 3,
        "active_stage_events": sum(stage_counts),
        "stage_measurements": len(measurements),
        "same_run_stage_pairs": len(pairs),
        "two_stage_runs": sum(count == 2 for count in stage_counts),
        "three_stage_runs": sum(count == 3 for count in stage_counts),
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise VariedTriadInputError("measurement_denominator_mismatch")


def _stage_assessment(
    target: Mapping[str, object],
    peers: list[Mapping[str, object]],
    peer_pair: Mapping[str, object],
    target_pairs: list[Mapping[str, object]],
    config: VariedTriadConfig,
) -> dict[str, object]:
    peer_labs = [_lab(peer["median_lab"]) for peer in peers]
    center = tuple((left + right) / 2.0 for left, right in zip(*peer_labs, strict=True))
    peer_delta = _finite(peer_pair["delta_e76"], "delta_e76")
    target_delta = math.dist(_lab(target["median_lab"]), center)
    threshold = peer_delta / 2.0 + config.target_outlier_margin_delta_e76
    if peer_delta > config.tight_peer_pair_max_delta_e76:
        status = "peer_mode_not_tight"
        limit_codes = ["peer_mode_not_tight"]
    elif target_delta > threshold:
        status = "outlier_to_two_peer_mode"
        limit_codes = ["relative_outlier_direction_unresolved"]
    else:
        status = "consistent_with_two_peer_mode"
        limit_codes = ["two_peer_mode_cleanliness_unverified"]
    return {
        "target_evidence_id": target["evidence_id"],
        "target_measurement_id": target["measurement_id"],
        "target_image_id": target["image_id"],
        "stage_id": target["stage_id"],
        "stage_ordinal": target["stage_ordinal"],
        "tank_id": target["tank_id"],
        "relative_status": status,
        "peer_pair_id": peer_pair["pair_id"],
        "peer_evidence_ids": [peer["evidence_id"] for peer in peers],
        "peer_measurement_ids": [peer["measurement_id"] for peer in peers],
        "peer_image_ids": [peer["image_id"] for peer in peers],
        "target_to_peer_pairs": [
            {
                "pair_id": pair["pair_id"],
                "delta_e76": pair["delta_e76"],
            }
            for pair in target_pairs
        ],
        "peer_pair_delta_e76": peer_delta,
        "reference_center_lab": list(center),
        "reference_radius_delta_e76": peer_delta / 2.0,
        "target_delta_e76": target_delta,
        "threshold_delta_e76": threshold,
        "threshold_excess_delta_e76": target_delta - threshold,
        "unit": "delta_e76",
        "limit_codes": limit_codes,
    }


"""Grounded explanation boundary for varied-run triad evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import copy
import hashlib
import json
import math
import re

from fluidmix.deploy_input_policy import validate_deploy_input_fields
from fluidmix.explanation_contract import ContractError
from fluidmix.stage_cosmos_explanation import parse_stage_response, raw_stage_format_passes


_STATUS = {"keep": "NO_DIFFERENCE", "discard": "DIFFERENCE", "insufficient_evidence": "UNKNOWN"}
_MEANING = {"NO_DIFFERENCE": "no_difference_observed", "DIFFERENCE": "difference_observed", "UNKNOWN": "unknown"}
_SPLIT_VERSIONS = frozenset(("whole-run-varied-v3", "whole-run-varied-v4"))
_ATTEMPT_PLAN_COUNTS = {
    "whole-run-varied-v3": {"decisions": 18, "runs": 6},
    "whole-run-varied-v4": {"decisions": 30, "runs": 10},
}
_CONTEXT = ("decision_id", "batch_id", "run_id", "recipe_id", "split_role", "split_version", "vial_id", "decision_timing")
_DECISION_KEYS = set(_CONTEXT) | {"disposition", "explanation_status", "relative_assessment", "stages", "limit_codes"}
_STAGE_KEYS = {
    "target_evidence_id", "target_measurement_id", "target_image_id", "stage_id",
    "stage_ordinal", "tank_id", "relative_status", "peer_pair_id",
    "peer_evidence_ids", "peer_measurement_ids", "peer_image_ids",
    "target_to_peer_pairs", "peer_pair_delta_e76", "reference_center_lab",
    "reference_radius_delta_e76", "target_delta_e76", "threshold_delta_e76",
    "threshold_excess_delta_e76", "unit", "limit_codes",
}
_RELATIVE = {"outlier_to_two_peer_mode", "peer_mode_not_tight", "consistent_with_two_peer_mode"}
_GENERIC_ID = re.compile(
    r"\b(?:EVIDENCE|MEASUREMENT|DECISION|IMAGE|EVID|MEAS|DEC|IMG|BATCH|PAIR)[-_:][A-Za-z0-9_.:-]+\b"
    r"|\b(?:evidence|measurement|decision|image|batch|pair)\s+id\s*[:=#]\s*\S+",
    re.IGNORECASE,
)
_UNIT = re.compile(r"\b(?:delta[\s_-]*e76|d\s*e\s*76|cie\s*lab|millilit(?:er|re)s?|ml)\b", re.IGNORECASE)
_POSITIVE_CAUSE = re.compile(
    r"\b(?:cause|physical\s+defect|defect|next[-\s]?vial(?:\s+effect)?)\b.{0,80}"
    r"\b(?:caused|confirmed|detected|identified|known|likely|present|probable|proven|verified|will)\b"
    r"|\b(?:caused|confirmed|detected|identified|known|likely|present|probable|proven|verified)\b"
    r".{0,80}\b(?:cause|physical\s+defect|defect)\b",
    re.IGNORECASE,
)
_CONTROL = re.compile(
    r"\b(?:adjust|calibrate|change|clean|close|decrease|disable|dispense|divert|dose|drain|empty|"
    r"enable|flush|hold|increase|lower|mix|modify|open|operate|override|pump|purge|raise|reduce|"
    r"refill|reject|rerun|set|shut|start|stop)\b",
    re.IGNORECASE,
)
_CLAIM = (
    "Synthetic within-run colour relations only. No clean reference is verified; physical "
    "carryover, defect, cause, plant action, and sim-to-real validity are unverified."
)


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"varied explanation {role} must be nonempty text")
    return value


def _codes(value: object, role: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"varied explanation {role} must contain text codes")
    return value


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"varied explanation {role} must be finite numeric")
    return float(value)


def _two_texts(value: object, role: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"varied explanation {role} must contain two values")
    return [_text(item, role) for item in value]


def _validate_stage(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _STAGE_KEYS:
        raise ContractError("varied explanation stage fields are invalid")
    for field in ("target_evidence_id", "target_measurement_id", "target_image_id", "stage_id", "tank_id", "peer_pair_id"):
        _text(value.get(field), field)
    ordinal = value.get("stage_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise ContractError("varied explanation stage ordinal is invalid")
    if value.get("tank_id") not in {"t100", "t200", "t400"} or value.get("relative_status") not in _RELATIVE:
        raise ContractError("varied explanation stage context is invalid")
    if value.get("unit") != "delta_e76":
        raise ContractError("varied explanation stage unit must be delta_e76")
    _codes(value.get("limit_codes"), "stage limit_codes")
    for field in ("peer_evidence_ids", "peer_measurement_ids", "peer_image_ids"):
        _two_texts(value.get(field), field)
    pairs = value.get("target_to_peer_pairs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ContractError("varied explanation target peer pairs are invalid")
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != {"pair_id", "delta_e76"}:
            raise ContractError("varied explanation target peer pair is invalid")
        _text(pair.get("pair_id"), "target peer pair ID")
        _finite(pair.get("delta_e76"), "target peer delta E76")
    lab = value.get("reference_center_lab")
    if not isinstance(lab, list) or len(lab) != 3:
        raise ContractError("varied explanation reference Lab is invalid")
    for channel in lab:
        _finite(channel, "reference Lab channel")
    for field in ("peer_pair_delta_e76", "reference_radius_delta_e76", "target_delta_e76", "threshold_delta_e76", "threshold_excess_delta_e76"):
        _finite(value.get(field), field)
    return value


def _validate_decision(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _DECISION_KEYS:
        raise ContractError("varied explanation decision fields are invalid")
    for field in _CONTEXT:
        _text(value.get(field), field)
    if (
        value.get("split_role") not in {"train", "validation", "evaluation"}
        or value.get("split_version") not in _SPLIT_VERSIONS
    ):
        raise ContractError("varied explanation split context is invalid")
    if value.get("vial_id") not in {"V1", "V2", "V3"}:
        raise ContractError("varied explanation vial ID is invalid")
    disposition = value.get("disposition")
    if disposition not in _STATUS or value.get("explanation_status") != _STATUS[disposition]:
        raise ContractError("varied explanation status contradicts disposition")
    if value.get("relative_assessment") not in _RELATIVE:
        raise ContractError("varied explanation assessment is invalid")
    _codes(value.get("limit_codes"), "decision limit_codes")
    stages = value.get("stages")
    if not isinstance(stages, list) or len(stages) not in {2, 3}:
        raise ContractError("varied explanation requires two or three stages")
    checked = [_validate_stage(stage) for stage in stages]
    if sorted(stage["stage_ordinal"] for stage in checked) != list(range(1, len(checked) + 1)):
        raise ContractError("varied explanation stage ordinals must be contiguous")
    for field in ("stage_id", "target_evidence_id", "target_image_id"):
        if len({stage[field] for stage in checked}) != len(checked):
            raise ContractError(f"varied explanation {field} values must be unique")
    return value


def _public(decision: object) -> dict[str, object]:
    checked = _validate_decision(decision)
    return {
        "context": {field: copy.deepcopy(checked[field]) for field in _CONTEXT},
        "disposition": checked["disposition"],
        "explanation_status": checked["explanation_status"],
        "relative_assessment": checked["relative_assessment"],
        "limit_codes": copy.deepcopy(checked["limit_codes"]),
        "stages": copy.deepcopy(sorted(checked["stages"], key=lambda stage: stage["stage_ordinal"])),
        "claim_boundary": _CLAIM,
    }


def parse_varied_response(raw: object) -> dict[str, object]:
    return parse_stage_response(raw)


def raw_varied_format_passes(raw: object) -> bool:
    return raw_stage_format_passes(raw)


def _known_ids(decision: Mapping[str, object]) -> set[str]:
    result = {str(decision[field]) for field in ("decision_id", "batch_id", "run_id", "recipe_id", "vial_id")}
    for stage in decision["stages"]:
        for field in ("target_evidence_id", "target_measurement_id", "target_image_id", "stage_id", "tank_id", "peer_pair_id"):
            result.add(str(stage[field]))
        for field in ("peer_evidence_ids", "peer_measurement_ids", "peer_image_ids"):
            result.update(str(item) for item in stage[field])
        result.update(str(pair["pair_id"]) for pair in stage["target_to_peer_pairs"])
    return result


def varied_content_failures(raw: object, parsed: object, decision: object) -> list[str]:
    if not isinstance(raw, str):
        raise ContractError("varied explanation raw response must be text")
    reparsed = parse_varied_response(raw)
    if parsed != reparsed:
        raise ContractError("varied parsed response is not derived from raw text")
    checked = _validate_decision(decision)
    failures: list[str] = []
    if reparsed["evidence_meaning"] != _MEANING[str(checked["explanation_status"])]:
        failures.append("evidence_meaning")
    expected_abstention = "abstain" if checked["disposition"] == "insufficient_evidence" else "limited"
    if reparsed["abstention"] != expected_abstention:
        failures.append("abstention")
    allowed = {"keep": {"keep"}, "discard": {"discard"}, "insufficient_evidence": {"inspect", "collect_evidence"}}[str(checked["disposition"])]
    if reparsed["suggested_action"] not in allowed:
        failures.append("suggested_action")
    if re.search(r"[0-9]", raw):
        failures.append("numeric_fidelity")
    if _GENERIC_ID.search(raw) or any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(identity)}(?![A-Za-z0-9])", raw, re.IGNORECASE)
        for identity in {value.lower() for value in _known_ids(checked)}
    ):
        failures.append("evidence_id_faithfulness")
    if _UNIT.search(raw):
        failures.append("unit_fidelity")
    uncertainty = str(reparsed["wording"]["uncertainty"])
    required = (
        r"\bcause\b.{0,100}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
        r"\bphysical\s+defect\b.{0,100}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
        r"\bnext[-\s]?vial(?:\s+effect)?\b.{0,100}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
    )
    if not all(re.search(pattern, uncertainty, re.IGNORECASE) for pattern in required):
        failures.append("required_limits")
    assertion = " ".join((str(reparsed["wording"]["evidence"]), str(reparsed["wording"]["action"])))
    if re.search(r"\b(?:carryover|residue)\b", raw, re.IGNORECASE) or _POSITIVE_CAUSE.search(raw) or re.search(
        r"\b(?:cause|physical\s+defect|defect|next[-\s]?vial(?:\s+effect)?)\b", assertion, re.IGNORECASE
    ):
        failures.append("unsupported_claim")
    if _CONTROL.search(raw):
        failures.append("operator_control_advice")
    return list(dict.fromkeys(failures))


def structure_varied_response(raw: object, parsed: object, decision: object) -> dict[str, object]:
    failures = varied_content_failures(raw, parsed, decision)
    if failures:
        raise ContractError("varied response fails content policy: " + ", ".join(failures))
    checked = _validate_decision(decision)
    public = _public(checked)
    stages = public["stages"]
    return {
        "schema_version": "f2_varied_explanation.v1",
        "decision_id": checked["decision_id"],
        "presentation": copy.deepcopy(parsed),
        "disposition": checked["disposition"],
        "explanation_status": checked["explanation_status"],
        "relative_assessment": checked["relative_assessment"],
        "context": public["context"],
        "evidence_ids": [item for stage in stages for item in [stage["target_evidence_id"], *stage["peer_evidence_ids"]]],
        "measurement_ids": [item for stage in stages for item in [stage["target_measurement_id"], *stage["peer_measurement_ids"]]],
        "image_ids": [item for stage in stages for item in [stage["target_image_id"], *stage["peer_image_ids"]]],
        "stage_evidence": copy.deepcopy(stages),
        "limit_codes": copy.deepcopy(public["limit_codes"]),
        "claim_boundary": public["claim_boundary"],
    }


def varied_reference_response(decision: object) -> str:
    checked = _validate_decision(decision)
    status = str(checked["explanation_status"])
    evidence = {
        "UNKNOWN": "The supplied within-run evidence cannot establish a verified reference.",
        "DIFFERENCE": "The supplied deterministic evidence supports a difference.",
        "NO_DIFFERENCE": "The supplied deterministic evidence supports no difference.",
    }[status]
    abstention = "ABSTAIN" if checked["disposition"] == "insufficient_evidence" else "LIMITED"
    action, wording = {
        "insufficient_evidence": ("COLLECT_EVIDENCE", "Collect an independent verified reference before assigning a disposition."),
        "keep": ("KEEP", "Keep this vial under the supplied evidence."),
        "discard": ("DISCARD", "Discard this vial under the supplied evidence."),
    }[str(checked["disposition"])]
    return (
        f"Evidence: {status} | {evidence}\n"
        f"Uncertainty: {abstention} | Cause, physical defect, and next-vial effect remain unverified.\n"
        f"Action: {action} | {wording}"
    )


def build_varied_prompt(template: object, decision: object) -> str:
    if not isinstance(template, str) or template.count("{{EVIDENCE_RECORD_JSON}}") != 1:
        raise ContractError("varied prompt must contain the evidence placeholder once")
    encoded = json.dumps(_public(decision), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return template.replace("{{EVIDENCE_RECORD_JSON}}", encoded)


def _relevant_images(decision: Mapping[str, object]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    captions: list[str] = []
    for stage in sorted(decision["stages"], key=lambda row: row["stage_ordinal"]):
        ordinal = stage["stage_ordinal"]
        images.extend((stage["target_image_id"], stage["peer_image_ids"][0]))
        captions.extend((f"Stage {ordinal} target image.", f"Stage {ordinal} same-run peer image."))
    if len(images) not in {4, 6} or len(set(images)) != len(images):
        raise ContractError("varied relevant image list is invalid")
    return images, captions


def _irrelevant_images(
    decision: Mapping[str, object],
    relevant_by_id: Mapping[str, tuple[list[str], list[str]]],
    decisions_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    needed = len(relevant_by_id[str(decision["decision_id"])][0])
    own = set(relevant_by_id[str(decision["decision_id"])][0])
    candidates = [other for other in decisions_by_id.values() if other["run_id"] != decision["run_id"]]
    candidates.sort(
        key=lambda other: hashlib.sha256(
            f"{decision['decision_id']}\0{other['decision_id']}".encode("utf-8")
        ).hexdigest()
    )
    chosen: list[str] = []
    source_ids: list[str] = []
    source_runs: list[str] = []
    source_statuses: list[str] = []
    for other in candidates:
        available = [
            image for image in relevant_by_id[str(other["decision_id"])][0]
            if image not in own and image not in chosen
        ]
        take = available[: needed - len(chosen)]
        if take:
            chosen.extend(take)
            source_ids.append(str(other["decision_id"]))
            source_runs.append(str(other["run_id"]))
            source_statuses.append(str(other["explanation_status"]))
        if len(chosen) == needed:
            break
    if len(chosen) != needed or own.intersection(chosen):
        raise ContractError("varied irrelevant image control cannot be constructed")
    chosen.sort(
        key=lambda image: hashlib.sha256(
            f"{decision['decision_id']}\0shuffle\0{image}".encode("utf-8")
        ).hexdigest()
    )
    return {
        "image_ids": chosen,
        "source_decision_ids": source_ids,
        "source_run_ids": source_runs,
        "source_statuses": source_statuses,
    }


def build_varied_attempt_plan(decision_artifact: object, prompt_template: object) -> list[dict[str, object]]:
    """Build the supported V3 or V4 four-condition explanation matrix."""

    validate_deploy_input_fields(decision_artifact, require_allowlist=False)
    if not isinstance(decision_artifact, Mapping):
        raise ContractError("varied attempt plan requires an artifact object")
    required_header = {
        "schema_version": 1,
        "artifact_type": "f2_varied_triad_decisions",
        "status": "assessed",
        "data_origin": "twin",
        "real_data_used": False,
    }
    if any(decision_artifact.get(key) != value for key, value in required_header.items()):
        raise ContractError("varied attempt plan artifact header is invalid")
    split_version = decision_artifact.get("split_version")
    contract = _ATTEMPT_PLAN_COUNTS.get(str(split_version))
    if contract is None:
        raise ContractError("varied attempt plan split contract is unsupported")
    decision_count = contract["decisions"]
    run_count = contract["runs"]
    decisions = decision_artifact.get("decision_records")
    if not isinstance(decisions, list) or len(decisions) != decision_count:
        raise ContractError(
            f"varied attempt plan requires exactly {decision_count} decisions"
        )
    checked = [_validate_decision(decision) for decision in decisions]
    if any(
        decision["split_version"] != decision_artifact["split_version"]
        for decision in checked
    ):
        raise ContractError("varied attempt plan decision split version mismatch")
    ids = [str(decision["decision_id"]) for decision in checked]
    if len(set(ids)) != decision_count:
        raise ContractError("varied attempt plan decision IDs must be unique")
    by_run: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for decision in checked:
        by_run[str(decision["run_id"])].append(decision)
    if len(by_run) != run_count or any(
        sorted(str(row["vial_id"]) for row in rows) != ["V1", "V2", "V3"]
        for rows in by_run.values()
    ):
        raise ContractError(
            f"varied attempt plan requires {run_count} three-vial runs"
        )
    denominators = decision_artifact.get("denominators")
    expected = {
        "runs": run_count,
        "vial_decisions": decision_count,
        "keep": 0,
        "discard": 0,
        "insufficient_evidence": decision_count,
    }
    if not isinstance(denominators, Mapping) or any(denominators.get(key) != value for key, value in expected.items()):
        raise ContractError("varied attempt plan denominators are invalid")
    relevant = {str(decision["decision_id"]): _relevant_images(decision) for decision in checked}
    decisions_by_id = {str(decision["decision_id"]): decision for decision in checked}
    irrelevant = {
        decision_id: _irrelevant_images(decision, relevant, decisions_by_id)
        for decision_id, decision in decisions_by_id.items()
    }
    conditions = (
        "numbers_only", "ordered_stage_visual_pairs", "irrelevant_shuffled_images",
        "deterministic_template",
    )
    attempts: list[dict[str, object]] = []
    for decision in checked:
        decision_id = str(decision["decision_id"])
        evidence_json = json.dumps(
            _public(decision), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        evidence_sha = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
        prompt = build_varied_prompt(prompt_template, decision)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for condition in conditions:
            control = condition == "deterministic_template"
            if condition == "ordered_stage_visual_pairs":
                image_ids, captions = list(relevant[decision_id][0]), list(relevant[decision_id][1])
            elif condition == "irrelevant_shuffled_images":
                image_ids = list(irrelevant[decision_id]["image_ids"])
                captions = list(relevant[decision_id][1])
            else:
                image_ids, captions = [], []
            attempts.append({
                "attempt_index": len(attempts),
                "attempt_id": f"{condition}-{decision_id}",
                "decision_id": decision_id,
                "run_id": decision["run_id"],
                "split_role": decision["split_role"],
                "condition_id": condition,
                "kind": "control" if control else "model",
                "prompt": None if control else prompt,
                "prompt_sha256": None if control else prompt_sha,
                "evidence_record_sha256": evidence_sha,
                "image_ids": image_ids,
                "image_captions": captions,
                "image_count": len(image_ids),
                "irrelevant_source_decision_ids": list(irrelevant[decision_id]["source_decision_ids"]) if condition == "irrelevant_shuffled_images" else [],
                "irrelevant_source_run_ids": list(irrelevant[decision_id]["source_run_ids"]) if condition == "irrelevant_shuffled_images" else [],
                "irrelevant_source_statuses": list(irrelevant[decision_id]["source_statuses"]) if condition == "irrelevant_shuffled_images" else [],
                "requires_multi_image_smoke": condition in {"ordered_stage_visual_pairs", "irrelevant_shuffled_images"},
                "template_response": varied_reference_response(decision) if control else None,
                "status": "not_run",
            })
    return attempts

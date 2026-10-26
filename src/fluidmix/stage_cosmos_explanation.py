"""Frozen text boundary for Cosmos explanations of measured vial-stage evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping

from fluidmix.explanation_contract import ContractError


_MAX_RAW_CHARACTERS = 8_192
_MAX_LINE_CHARACTERS = 2_048
_MAX_WORDING_CHARACTERS = 1_000
_LABEL = re.compile(
    r"^\s*(?:[-+>#]+\s*)?(?:[*`~]{0,3})"
    r"(?P<label>evidence|uncertainty|action)"
    r"(?:[*`~]{0,3})\s*(?::|[-\u2013\u2014])\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_MARKERS = {
    "evidence": (
        re.compile(
            r"\b(?:NO[\s_-]*DIFFERENCE|DIFFERENCE|UNKNOWN)\b",
            re.IGNORECASE,
        ),
        {
            "no difference": "no_difference_observed",
            "difference": "difference_observed",
            "unknown": "unknown",
        },
    ),
    "uncertainty": (
        re.compile(r"\b(?:ABSTAIN|LIMITED)\b", re.IGNORECASE),
        {"abstain": "abstain", "limited": "limited"},
    ),
    "action": (
        re.compile(
            r"\b(?:COLLECT[\s_-]*EVIDENCE|KEEP|DISCARD|INSPECT)\b",
            re.IGNORECASE,
        ),
        {
            "collect evidence": "collect_evidence",
            "keep": "keep",
            "discard": "discard",
            "inspect": "inspect",
        },
    ),
}
_EXACT_FORMAT = re.compile(
    r"\AEvidence: (?:NO_DIFFERENCE|DIFFERENCE|UNKNOWN) \| [^\r\n]+\n"
    r"Uncertainty: (?:LIMITED|ABSTAIN) \| [^\r\n]+\n"
    r"Action: (?:KEEP|DISCARD|INSPECT|COLLECT_EVIDENCE) \| [^\r\n]+\Z"
)
_STATUS_BY_DISPOSITION = {
    "discard": "DIFFERENCE",
    "keep": "NO_DIFFERENCE",
    "insufficient_evidence": "UNKNOWN",
}
_MEANING_BY_STATUS = {
    "DIFFERENCE": "difference_observed",
    "NO_DIFFERENCE": "no_difference_observed",
    "UNKNOWN": "unknown",
}
_CONTEXT_FIELDS = (
    "decision_id",
    "family_id",
    "batch_id",
    "cohort",
    "partition",
    "previous_sim_recipe_id",
    "current_sim_recipe_id",
    "previous_programmed_ml",
    "current_programmed_ml",
    "programmed_dispense_order",
    "vial_id",
    "vial_position",
    "decision_timing",
)
_STAGE_FIELDS = (
    "evidence_id",
    "measurement_id",
    "stage_id",
    "stage_ordinal",
    "target_image_id",
    "validity",
    "limit_codes",
    "target_delta_e76_raw",
    "target_delta_e76_6dp",
    "threshold_delta_e76_raw",
    "threshold_delta_e76_6dp",
    "stage_outlier",
    "unit",
)
_REFERENCE_FIELDS = (
    "mode",
    "validity",
    "limit_codes",
    "representative_peer_id",
    "peer_count",
    "peer_batch_count",
    "median_radius_delta_e76_raw",
    "max_radius_delta_e76_raw",
    "mad_radius_delta_e76_raw",
    "pairwise_diameter_delta_e76_raw",
    "unit",
)
_GENERIC_ID = re.compile(
    r"\b(?:EVIDENCE|MEASUREMENT|DECISION|IMAGE|EVID|MEAS|DEC|IMG|BATCH|REF)"
    r"[-_:][A-Za-z0-9_.:-]+\b"
    r"|\b(?:evidence|measurement|decision|image|batch|recipe)\s+id\s*"
    r"[:=#]\s*\S+",
    re.IGNORECASE,
)
_UNIT = re.compile(
    r"\b(?:delta[\s_-]*e76|d\s*e\s*76|cie\s*lab|millilit(?:er|re)s?|ml)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_CLAIM = re.compile(
    r"(?:"
    r"\b(?:cause|physical\s+defect|defect|next[-\s]?vial(?:\s+effect)?)\b"
    r".{0,80}\b(?:caused|confirmed|detected|expected|identified|known|likely|"
    r"certain|exists|present|probable|proven|proves|verified|will)\b"
    r"|"
    r"\b(?:caused|confirmed|detected|expected|identified|known|likely|present|"
    r"certain|exists|probable|proven|proves|verified)\b.{0,80}"
    r"\b(?:cause|physical\s+defect|defect|next[-\s]?vial(?:\s+effect)?)\b"
    r"|\b(?:carryover|residue)\b"
    r"|\b(?:means|proves|shows|indicates)\b.{0,30}"
    r"\b(?:cause|defect|carryover|residue)\b"
    r")",
    re.IGNORECASE,
)
_CONTROL_ADVICE = re.compile(
    r"\b(?:adjust|calibrate|change|clean|close|decrease|disable|dispense|divert|"
    r"dose|drain|empty|enable|flush|hold|increase|lower|mix|modify|open|operate|"
    r"override|pump|purge|raise|reduce|refill|reject|rerun|set|shut|start|stop)\b",
    re.IGNORECASE,
)


def _canonical_marker(value: str, field: str) -> tuple[str, str]:
    pattern, mapping = _MARKERS[field]
    marker_region, separator, trailing = value.partition("|")
    matches = list(pattern.finditer(marker_region))
    canonical = {
        mapping[re.sub(r"[\s_-]+", " ", match.group(0).lower()).strip()]
        for match in matches
    }
    if len(canonical) != 1:
        raise ContractError(
            f"stage {field} must contain exactly one marker meaning"
        )
    first = matches[0]
    wording = (
        marker_region[: first.start()]
        + marker_region[first.end() :]
        + (f" {trailing}" if separator else "")
    ).strip(" \t|:;-\u2013\u2014*`~")
    if not wording or len(wording) > _MAX_WORDING_CHARACTERS:
        raise ContractError(f"stage {field} wording is empty or too long")
    if any(ord(character) < 32 and character != "\t" for character in wording):
        raise ContractError(f"stage {field} wording contains a control character")
    return canonical.pop(), wording


def _validate_parsed(parsed: object) -> dict[str, object]:
    expected = {"evidence_meaning", "abstention", "suggested_action", "wording"}
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ContractError("stage parsed response has the wrong fields")
    allowed = {
        "evidence_meaning": {
            "difference_observed",
            "no_difference_observed",
            "unknown",
        },
        "abstention": {"limited", "abstain"},
        "suggested_action": {"keep", "discard", "inspect", "collect_evidence"},
    }
    for field, values in allowed.items():
        if parsed[field] not in values:
            raise ContractError(f"stage parsed response has invalid {field}")
    wording = parsed["wording"]
    if not isinstance(wording, dict) or set(wording) != {
        "evidence",
        "uncertainty",
        "action",
    }:
        raise ContractError("stage parsed wording has the wrong fields")
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_WORDING_CHARACTERS
        for value in wording.values()
    ):
        raise ContractError("stage parsed wording is invalid")
    return parsed


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{role} must be nonempty text")
    return value


def _codes(value: object, role: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(code, str) or not code for code in value
    ):
        raise ContractError(f"{role} must be a list of nonempty strings")
    return value


def _finite_number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{role} must be finite")
    return result


def _primary_reference(stage: Mapping[str, object]) -> Mapping[str, object]:
    cases = stage.get("reference_cases")
    if not isinstance(cases, list):
        raise ContractError("stage reference_cases must be a list")
    primary_mode = stage.get("primary_mode")
    primaries = [
        case
        for case in cases
        if isinstance(case, Mapping)
        and case.get("mode") == primary_mode
        and case.get("disposition_source") is True
    ]
    if len(primaries) != 1:
        raise ContractError("stage must have exactly one primary reference case")
    return primaries[0]


def _validate_decision(decision: object) -> Mapping[str, object]:
    if not isinstance(decision, Mapping):
        raise ContractError("stage decision must be an object")
    missing = [field for field in _CONTEXT_FIELDS if field not in decision]
    if missing:
        raise ContractError(f"stage decision context is missing fields: {missing!r}")
    for field in (
        "decision_id",
        "family_id",
        "batch_id",
        "cohort",
        "partition",
        "previous_sim_recipe_id",
        "current_sim_recipe_id",
        "vial_id",
        "decision_timing",
    ):
        _text(decision[field], f"stage decision {field}")
    _finite_number(decision["vial_position"], "stage decision vial_position")
    for field in ("previous_programmed_ml", "current_programmed_ml"):
        values = decision[field]
        if not isinstance(values, Mapping) or set(values) != {"t100", "t200", "t400"}:
            raise ContractError(f"stage decision {field} has the wrong tanks")
        for tank, value in values.items():
            _finite_number(value, f"stage decision {field}.{tank}")
    order = decision["programmed_dispense_order"]
    if not isinstance(order, list) or sorted(order) != ["t100", "t200", "t400"]:
        raise ContractError("stage decision dispense order must contain each tank once")
    disposition = decision.get("disposition")
    if disposition not in _STATUS_BY_DISPOSITION:
        raise ContractError("stage decision disposition is invalid")
    if decision.get("explanation_status") != _STATUS_BY_DISPOSITION[disposition]:
        raise ContractError("stage decision status contradicts its disposition")
    if (
        decision.get("disposition_source")
        != "deterministic_pipeline_primary_reference_evidence"
    ):
        raise ContractError("stage disposition is not bound to deterministic evidence")
    _codes(decision.get("limit_codes"), "stage decision limit_codes")
    stages = decision.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ContractError("stage decision must contain exactly three stages")
    if sorted(stage.get("stage_ordinal") for stage in stages if isinstance(stage, Mapping)) != [
        1,
        2,
        3,
    ]:
        raise ContractError("stage ordinals must be exactly one, two, and three")
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ContractError("stage evidence rows must be objects")
        missing_stage = [field for field in _STAGE_FIELDS if field not in stage]
        if missing_stage:
            raise ContractError(
                f"stage evidence is missing fields: {missing_stage!r}"
            )
        for field in ("evidence_id", "measurement_id", "stage_id", "target_image_id"):
            _text(stage.get(field), f"stage {field}")
        _finite_number(stage.get("stage_ordinal"), "stage ordinal")
        has_time_code = "time_code" in stage
        has_event = "stage_event" in stage
        if has_time_code == has_event:
            raise ContractError("stage must carry exactly one historical time code or F2 event")
        if has_time_code:
            _finite_number(stage.get("time_code"), "stage time code")
        else:
            event = stage.get("stage_event")
            required_event = {
                "event_id", "stage_ordinal", "tank_id", "tank_order", "physics_step",
                "frame_id", "event_relative_timestamp_s", "final_observation_offset_s",
            }
            if not isinstance(event, Mapping) or set(event) != required_event:
                raise ContractError("stage F2 event is malformed")
            if (
                event.get("event_id") != stage.get("stage_id")
                or event.get("stage_ordinal") != stage.get("stage_ordinal")
                or event.get("tank_order") != stage.get("stage_ordinal")
            ):
                raise ContractError("stage F2 event identity mismatches the stage")
            _text(event.get("tank_id"), "stage F2 event tank ID")
            for field in (
                "physics_step", "event_relative_timestamp_s", "final_observation_offset_s"
            ):
                _finite_number(event.get(field), f"stage F2 event {field}")
            _text(event.get("frame_id"), "stage F2 event frame ID")
        if stage.get("unit") != "delta_e76":
            raise ContractError("stage unit must be delta_e76")
        _codes(stage.get("limit_codes"), "stage limit_codes")
        validity = stage.get("validity")
        if validity not in {"valid", "insufficient_evidence"}:
            raise ContractError("stage validity is invalid")
        primary = _primary_reference(stage)
        _codes(primary.get("limit_codes"), "primary reference limit_codes")
        if primary.get("unit") != "delta_e76":
            raise ContractError("primary reference unit must be delta_e76")
        if validity == "valid":
            if primary.get("validity") != "valid":
                raise ContractError("valid stage requires a valid primary reference")
            _text(primary.get("representative_peer_id"), "representative peer ID")
            if not isinstance(stage.get("stage_outlier"), bool):
                raise ContractError("valid stage requires a Boolean outlier result")
            for field in (
                "target_delta_e76_raw",
                "threshold_delta_e76_raw",
            ):
                _finite_number(stage.get(field), f"stage {field}")
            for field in (
                "peer_count",
                "peer_batch_count",
                "median_radius_delta_e76_raw",
                "max_radius_delta_e76_raw",
                "mad_radius_delta_e76_raw",
                "pairwise_diameter_delta_e76_raw",
            ):
                _finite_number(primary.get(field), f"primary reference {field}")
        elif any(
            stage.get(field) is not None
            for field in (
                "target_delta_e76_raw",
                "target_delta_e76_6dp",
                "threshold_delta_e76_raw",
                "threshold_delta_e76_6dp",
                "stage_outlier",
            )
        ):
            raise ContractError("insufficient stage must not carry a comparison result")
    valid_stages = [stage for stage in stages if stage["validity"] == "valid"]
    if any(stage["stage_outlier"] is True for stage in valid_stages):
        evidence_disposition = "discard"
    elif len(valid_stages) == 3 and all(
        stage["stage_outlier"] is False for stage in valid_stages
    ):
        evidence_disposition = "keep"
    else:
        evidence_disposition = "insufficient_evidence"
    if disposition != evidence_disposition:
        raise ContractError("stage disposition contradicts its comparison evidence")
    return decision


def _stage_evidence(stage: Mapping[str, object]) -> dict[str, object]:
    primary = _primary_reference(stage)
    result = {field: copy.deepcopy(stage[field]) for field in _STAGE_FIELDS}
    event_field = "stage_event" if "stage_event" in stage else "time_code"
    result[event_field] = copy.deepcopy(stage[event_field])
    result["primary_reference"] = {
        field: copy.deepcopy(primary.get(field)) for field in _REFERENCE_FIELDS
    }
    return result


def _public_evidence_record(decision: object) -> dict[str, object]:
    checked = _validate_decision(decision)
    context = {field: copy.deepcopy(checked[field]) for field in _CONTEXT_FIELDS}
    stages = sorted(checked["stages"], key=lambda stage: stage["stage_ordinal"])
    return {
        "context": context,
        "disposition": checked["disposition"],
        "explanation_status": checked["explanation_status"],
        "disposition_source": checked["disposition_source"],
        "limit_codes": copy.deepcopy(checked["limit_codes"]),
        "stages": [_stage_evidence(stage) for stage in stages],
        "claim_boundary": (
            "Synthetic colour-distance evidence only; physical carryover, defect, "
            "cause, volume, plant action, and sim-to-real validity are unverified."
        ),
    }


def parse_stage_response(raw: object) -> dict[str, object]:
    """Parse the three model-owned presentation fields from forgiving text."""

    if not isinstance(raw, str):
        raise ContractError("stage model response must be text")
    if not raw or len(raw) > _MAX_RAW_CHARACTERS:
        raise ContractError("stage model response is empty or too long")
    captured: dict[str, str] = {}
    for line in raw.splitlines():
        if len(line) > _MAX_LINE_CHARACTERS:
            raise ContractError("stage model response contains an overlong line")
        match = _LABEL.match(line)
        if match is None:
            continue
        field = match.group("label").lower()
        if field in captured:
            raise ContractError(f"duplicate stage response field: {field}")
        captured[field] = match.group("value")
    if set(captured) != {"evidence", "uncertainty", "action"}:
        raise ContractError(
            "stage response does not contain exactly three labeled fields"
        )
    evidence_meaning, evidence_wording = _canonical_marker(
        captured["evidence"], "evidence"
    )
    abstention, uncertainty_wording = _canonical_marker(
        captured["uncertainty"], "uncertainty"
    )
    action, action_wording = _canonical_marker(captured["action"], "action")
    return {
        "evidence_meaning": evidence_meaning,
        "abstention": abstention,
        "suggested_action": action,
        "wording": {
            "evidence": evidence_wording,
            "uncertainty": uncertainty_wording,
            "action": action_wording,
        },
    }


def raw_stage_format_passes(raw: object) -> bool:
    """Return whether text follows the exact frozen three-line presentation."""

    return isinstance(raw, str) and _EXACT_FORMAT.fullmatch(raw) is not None


def build_stage_prompt(template: object, decision: object) -> str:
    """Insert public deterministic stage evidence into the frozen prompt."""

    if not isinstance(template, str) or template.count("{{EVIDENCE_RECORD_JSON}}") != 1:
        raise ContractError("stage prompt must contain the evidence placeholder once")
    evidence_json = json.dumps(
        _public_evidence_record(decision),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return template.replace("{{EVIDENCE_RECORD_JSON}}", evidence_json)


def stage_content_failures(
    raw: object, parsed: object, decision: object
) -> list[str]:
    """Return strict content-policy failures independently of format parsing."""

    if not isinstance(raw, str):
        raise ContractError("stage raw response must be text")
    checked_parsed = _validate_parsed(parsed)
    if parse_stage_response(raw) != checked_parsed:
        raise ContractError("stage parsed response is not derived from the raw text")
    checked = _validate_decision(decision)
    failures: list[str] = []
    expected_meaning = _MEANING_BY_STATUS[checked["explanation_status"]]
    if checked_parsed["evidence_meaning"] != expected_meaning:
        failures.append("evidence_meaning")
    expected_abstention = (
        "abstain"
        if checked["disposition"] == "insufficient_evidence"
        else "limited"
    )
    if checked_parsed["abstention"] != expected_abstention:
        failures.append("abstention")
    allowed_actions = {
        "keep": {"keep"},
        "discard": {"discard"},
        "insufficient_evidence": {"inspect", "collect_evidence"},
    }[checked["disposition"]]
    if checked_parsed["suggested_action"] not in allowed_actions:
        failures.append("suggested_action")
    if re.search(r"[0-9]", raw) is not None:
        failures.append("numeric_fidelity")
    known_ids = {value.lower() for value in _deterministic_ids(checked)}
    if _GENERIC_ID.search(raw) is not None or any(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(identifier)}(?![A-Za-z0-9])",
            raw,
            re.IGNORECASE,
        )
        is not None
        for identifier in known_ids
    ):
        failures.append("evidence_id_faithfulness")
    if _UNIT.search(raw) is not None:
        failures.append("unit_fidelity")
    uncertainty = checked_parsed["wording"]["uncertainty"]
    limits = (
        r"\bcause\b.{0,100}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
        r"\bphysical\s+defect\b.{0,100}\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
        r"\bnext[-\s]?vial(?:\s+effect)?\b.{0,100}"
        r"\b(?:unverified|unknown|not\s+verified|not\s+measured)\b",
    )
    if not all(re.search(pattern, uncertainty, re.IGNORECASE) for pattern in limits):
        failures.append("required_limits")
    model_assertion = " ".join(
        (
            checked_parsed["wording"]["evidence"],
            checked_parsed["wording"]["action"],
        )
    )
    if _UNSUPPORTED_CLAIM.search(raw) is not None or re.search(
        r"\b(?:cause|physical\s+defect|defect|next[-\s]?vial(?:\s+effect)?|"
        r"carryover|residue)\b",
        model_assertion,
        re.IGNORECASE,
    ) is not None:
        failures.append("unsupported_claim")
    if _CONTROL_ADVICE.search(raw) is not None:
        failures.append("operator_control_advice")
    return failures


def structure_stage_response(
    raw: object, parsed: object, decision: object
) -> dict[str, object]:
    """Build a record whose quantitative and disposition fields are deterministic."""

    failures = stage_content_failures(raw, parsed, decision)
    if failures:
        raise ContractError(
            "stage response fails content policy: " + ", ".join(failures)
        )
    checked = _validate_decision(decision)
    public = _public_evidence_record(checked)
    return {
        "schema_version": "stage_vial_explanation.v1",
        "decision_id": checked["decision_id"],
        "presentation": copy.deepcopy(parsed),
        "disposition": checked["disposition"],
        "explanation_status": checked["explanation_status"],
        "disposition_source": checked["disposition_source"],
        "context": public["context"],
        "evidence_ids": [
            stage["evidence_id"] for stage in public["stages"]
        ],
        "measurement_ids": [
            stage["measurement_id"] for stage in public["stages"]
        ],
        "stage_evidence": public["stages"],
        "limit_codes": public["limit_codes"],
        "claim_boundary": public["claim_boundary"],
    }


def stage_reference_response(decision: object) -> str:
    """Render the deterministic control through the same text boundary."""

    checked = _validate_decision(decision)
    status = checked["explanation_status"]
    evidence_wording = {
        "DIFFERENCE": (
            "One or more valid stage comparisons exceed their frozen reference thresholds."
        ),
        "NO_DIFFERENCE": (
            "All valid stage comparisons remain within their frozen reference thresholds."
        ),
        "UNKNOWN": "The supplied stage evidence cannot support a vial disposition.",
    }[status]
    action, action_wording = {
        "discard": (
            "DISCARD",
            "Discard this vial under the supplied synthetic evidence.",
        ),
        "keep": (
            "KEEP",
            "Keep this vial under the supplied synthetic evidence.",
        ),
        "insufficient_evidence": (
            "COLLECT_EVIDENCE",
            "Collect the missing evidence before assigning a disposition.",
        ),
    }[checked["disposition"]]
    abstention = (
        "ABSTAIN"
        if checked["disposition"] == "insufficient_evidence"
        else "LIMITED"
    )
    return (
        f"Evidence: {status} | {evidence_wording}\n"
        f"Uncertainty: {abstention} | Cause, physical defect, and next-vial effect "
        "remain unverified.\n"
        f"Action: {action} | {action_wording}"
    )


def build_stage_attempt_plan(
    measured_evidence: object, prompt_template: object
) -> list[dict[str, object]]:
    """Build the decision-major four-condition explanation matrix."""

    if not isinstance(measured_evidence, Mapping):
        raise ContractError("measured stage evidence must be an object")
    if measured_evidence.get("artifact_type") != "measured_evidence":
        raise ContractError("stage attempts require a measured_evidence artifact")
    if (
        measured_evidence.get("data_origin") != "twin"
        or measured_evidence.get("real_data_used") is not False
    ):
        raise ContractError("stage attempts require synthetic twin evidence")
    decisions = measured_evidence.get("decision_records")
    if not isinstance(decisions, list) or not decisions:
        raise ContractError("measured stage evidence has no decision records")
    checked = [_validate_decision(decision) for decision in decisions]
    decision_ids = [decision["decision_id"] for decision in checked]
    if len(set(decision_ids)) != len(decision_ids):
        raise ContractError("measured stage decision IDs must be unique")
    relevant = {
        decision["decision_id"]: _relevant_image_ids(decision)
        for decision in checked
    }
    irrelevant = {
        decision["decision_id"]: _irrelevant_image_ids(
            decision["decision_id"], relevant
        )
        for decision in checked
    }
    conditions = (
        "numbers_only",
        "ordered_stage_visual_pairs",
        "irrelevant_shuffled_images",
        "deterministic_template",
    )
    attempts: list[dict[str, object]] = []
    for decision in checked:
        decision_id = decision["decision_id"]
        evidence_json = json.dumps(
            _public_evidence_record(decision),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        evidence_sha256 = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
        prompt = build_stage_prompt(prompt_template, decision)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for condition in conditions:
            is_control = condition == "deterministic_template"
            image_ids: list[str]
            if condition == "ordered_stage_visual_pairs":
                image_ids = list(relevant[decision_id])
            elif condition == "irrelevant_shuffled_images":
                image_ids = list(irrelevant[decision_id])
            else:
                image_ids = []
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "attempt_id": f"{condition}-{decision_id}",
                    "decision_id": decision_id,
                    "partition": decision["partition"],
                    "condition_id": condition,
                    "kind": "control" if is_control else "model",
                    "prompt": None if is_control else prompt,
                    "prompt_sha256": None if is_control else prompt_sha256,
                    "evidence_record_sha256": evidence_sha256,
                    "image_ids": image_ids,
                    "image_count": len(image_ids),
                    "requires_multi_image_smoke": condition
                    in {
                        "ordered_stage_visual_pairs",
                        "irrelevant_shuffled_images",
                    },
                    "template_response": (
                        stage_reference_response(decision) if is_control else None
                    ),
                    "status": "not_run",
                }
            )
    return attempts


def _deterministic_ids(decision: Mapping[str, object]) -> set[str]:
    result = {
        decision[field]
        for field in (
            "decision_id",
            "family_id",
            "batch_id",
            "previous_sim_recipe_id",
            "current_sim_recipe_id",
            "vial_id",
        )
    }
    for stage in decision["stages"]:
        result.update(
            stage[field]
            for field in (
                "evidence_id",
                "measurement_id",
                "stage_id",
                "target_image_id",
            )
        )
        representative = _primary_reference(stage).get("representative_peer_id")
        if isinstance(representative, str) and representative:
            result.add(representative)
    return result


def _relevant_image_ids(decision: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    stages = sorted(decision["stages"], key=lambda stage: stage["stage_ordinal"])
    for stage in stages:
        result.append(stage["target_image_id"])
        representative = _primary_reference(stage).get("representative_peer_id")
        if isinstance(representative, str) and representative:
            result.append(representative)
    if len(result) > 6 or len(result) != len(set(result)):
        raise ContractError("relevant stage image list is invalid")
    return result


def _irrelevant_image_ids(
    decision_id: str, relevant_by_decision: Mapping[str, list[str]]
) -> list[str]:
    needed = len(relevant_by_decision[decision_id])
    relevant = set(relevant_by_decision[decision_id])
    candidates: list[str] = []
    for other_decision_id, image_ids in relevant_by_decision.items():
        if other_decision_id != decision_id:
            candidates.extend(image_id for image_id in image_ids if image_id not in relevant)
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(
        key=lambda image_id: hashlib.sha256(
            f"{decision_id}\0{image_id}".encode("utf-8")
        ).hexdigest()
    )
    if len(candidates) < needed:
        raise ContractError(
            "irrelevant image control cannot match the relevant image count"
        )
    chosen = candidates[:needed]
    if set(chosen).intersection(relevant):
        raise ContractError("irrelevant image control overlaps relevant images")
    return chosen

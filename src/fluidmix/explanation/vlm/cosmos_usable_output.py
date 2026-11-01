"""Forgiving text boundary for the frozen Cosmos usable-output treatment.

Only three presentation fields come from model text. Quantitative values, units,
evidence IDs, limits, and the closed evidence-policy fields are rebuilt from the
supplied record. Parsing success therefore cannot manufacture quantitative evidence.
"""

import hashlib
import re

from fluidmix.explanation.template.explanation_contract import (
    ContractError,
    build_prompt,
    reference_explanation,
    validate_fixture_set,
)


_MAX_RAW_CHARACTERS = 8_192
_MAX_LINE_CHARACTERS = 2_048
_MAX_WORDING_CHARACTERS = 1_000
_LABEL = re.compile(
    r"^\s*(?:[-+>#]+\s*)?(?:[*`~]{0,3})"
    r"(?P<label>suggested\s+action|evidence\s+meaning|explanation|uncertainty|action)"
    r"(?:[*`~]{0,3})\s*(?::|[-\u2013\u2014])\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_FIELD_BY_LABEL = {
    "suggested action": "suggested_action",
    "action": "suggested_action",
    "evidence meaning": "explanation",
    "explanation": "explanation",
    "uncertainty": "uncertainty",
}
_MARKERS = {
    "explanation": (
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
        re.compile(
            r"\b(?:DO[\s_-]*NOT[\s_-]*ABSTAIN|NOT[\s_-]*ABSTAIN|ABSTAIN|LIMITED)\b",
            re.IGNORECASE,
        ),
        {
            "do not abstain": "limited",
            "not abstain": "limited",
            "abstain": "abstain",
            "limited": "limited",
        },
    ),
    "suggested_action": (
        re.compile(
            r"\b(?:NO[\s_-]*AUTOMATIC[\s_-]*ACTION|MANUAL[\s_-]*REVIEW)\b",
            re.IGNORECASE,
        ),
        {
            "no automatic action": "no_automatic_action",
            "manual review": "manual_review_required",
        },
    ),
}


def _canonical_marker(value, field):
    pattern, mapping = _MARKERS[field]
    marker_region, separator, trailing = value.partition("|")
    matches = list(pattern.finditer(marker_region))
    canonical = {
        mapping[re.sub(r"[\s_-]+", " ", match.group(0).lower()).strip()]
        for match in matches
    }
    if len(canonical) != 1:
        raise ContractError(f"usable {field} must contain exactly one marker meaning")
    first = matches[0]
    wording = (
        marker_region[: first.start()]
        + marker_region[first.end() :]
        + (f" {trailing}" if separator else "")
    ).strip(
        " \t|:;-\u2013\u2014*`~"
    )
    # A model may stop after the required marker (the frozen Nano artifact did
    # this). Preserve that as empty, untrusted wording so scoring can count
    # structural recovery while content policy still rejects the row. An
    # explicit separator still promises prose and remains strict.
    if (separator and not wording) or len(wording) > _MAX_WORDING_CHARACTERS:
        raise ContractError(f"usable {field} wording is empty or too long")
    if any(ord(character) < 32 and character not in "\t" for character in wording):
        raise ContractError(f"usable {field} wording contains a control character")
    return canonical.pop(), wording


def _validate_parsed(parsed):
    expected = {"evidence_meaning", "abstention", "suggested_action", "wording"}
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ContractError("usable parsed response has the wrong fields")
    allowed = {
        "evidence_meaning": {
            "difference_observed",
            "no_difference_observed",
            "unknown",
        },
        "abstention": {"limited", "abstain"},
        "suggested_action": {"manual_review_required", "no_automatic_action"},
    }
    for field, values in allowed.items():
        if parsed[field] not in values:
            raise ContractError(f"usable parsed response has invalid {field}")
    wording = parsed["wording"]
    if not isinstance(wording, dict) or set(wording) != {
        "explanation",
        "uncertainty",
        "suggested_action",
    }:
        raise ContractError("usable parsed wording has the wrong fields")
    if any(
        not isinstance(value, str)
        or len(value) > _MAX_WORDING_CHARACTERS
        for value in wording.values()
    ):
        raise ContractError("usable parsed wording is invalid")


def parse_usable_response(raw):
    """Extract the three model-owned presentation fields from labeled prose."""
    if not isinstance(raw, str):
        raise ContractError("usable model response must be text")
    if not raw or len(raw) > _MAX_RAW_CHARACTERS:
        raise ContractError("usable model response is empty or too long")
    captured = {}
    for line in raw.splitlines():
        if len(line) > _MAX_LINE_CHARACTERS:
            raise ContractError("usable model response contains an overlong line")
        match = _LABEL.match(line)
        if match is None:
            continue
        label = re.sub(r"\s+", " ", match.group("label").lower()).strip()
        field = _FIELD_BY_LABEL[label]
        if field in captured:
            raise ContractError(f"duplicate usable response field: {field}")
        captured[field] = match.group("value")
    if set(captured) != {"explanation", "uncertainty", "suggested_action"}:
        raise ContractError("usable response does not contain exactly three labeled fields")
    evidence_meaning, explanation = _canonical_marker(
        captured["explanation"], "explanation"
    )
    abstention, uncertainty = _canonical_marker(
        captured["uncertainty"], "uncertainty"
    )
    suggested_action, action_wording = _canonical_marker(
        captured["suggested_action"], "suggested_action"
    )
    result = {
        "evidence_meaning": evidence_meaning,
        "abstention": abstention,
        "suggested_action": suggested_action,
        "wording": {
            "explanation": explanation,
            "uncertainty": uncertainty,
            "suggested_action": action_wording,
        },
    }
    _validate_parsed(result)
    return result


def build_usable_prompt(template, record, *, camera_evidence_included=True):
    """Insert one public evidence record into the usable-output prompt."""
    return build_prompt(
        template,
        record,
        camera_evidence_included=camera_evidence_included,
    )


def usable_reference_response(record, *, camera_evidence_included=True):
    """Render the deterministic control in the same forgiving text format."""
    expected = reference_explanation(
        record, camera_evidence_included=camera_evidence_included
    )
    meaning = {
        "difference_observed": "DIFFERENCE",
        "no_difference_observed": "NO_DIFFERENCE",
        "unknown": "UNKNOWN",
    }[expected["appearance"]]
    explanation = {
        "difference_observed": "The supplied comparison reports an appearance difference.",
        "no_difference_observed": "The supplied comparison reports no appearance difference.",
        "unknown": "The supplied evidence does not support an appearance conclusion.",
    }[expected["appearance"]]
    abstention = (
        "ABSTAIN"
        if expected["status"] in {"insufficient_evidence", "invalid_input"}
        else "LIMITED"
    )
    action = {
        "manual_review_required": "MANUAL_REVIEW",
        "no_automatic_action": "NO_AUTOMATIC_ACTION",
    }[expected["operator_action"]]
    action_wording = (
        "Have a person review the supplied evidence."
        if action == "MANUAL_REVIEW"
        else "Take no automatic action from this explanation."
    )
    return (
        f"Evidence meaning: {meaning} | {explanation}\n"
        f"Uncertainty: {abstention} | Cause unverified; defect unverified; "
        "next-vial effect unverified.\n"
        f"Suggested action: {action} | {action_wording}"
    )


def structure_usable_response(parsed, record, *, camera_evidence_included=True):
    """Combine model-owned prose with evidence-owned quantitative fields."""
    _validate_parsed(parsed)
    expected = reference_explanation(
        record, camera_evidence_included=camera_evidence_included
    )
    policy_keys = (
        "status",
        "appearance",
        "defect",
        "cause",
        "next_vial_effect",
        "confidence",
        "operator_action",
    )
    return {
        "schema_version": "operator_explanation_usable.v1",
        "record_id": expected["record_id"],
        "presentation": parsed,
        "policy_fields": {key: expected[key] for key in policy_keys},
        "evidence_ids": list(expected["evidence_ids"]),
        "quantitative_evidence": [
            dict(item) for item in expected["quantitative_evidence"]
        ],
        "limit_codes": list(expected["limit_codes"]),
    }


def build_usable_attempt_plan(fixture_set, config, prompt_template):
    """Build the fixed fixture-major usable-output matrix without an oracle."""
    validate_fixture_set(fixture_set)
    modes = config.get("modes") if isinstance(config, dict) else None
    expected_modes = [
        "image_evidence_recipe",
        "evidence_recipe",
        "image_recipe",
        "template_reference",
    ]
    if modes != expected_modes or config.get("planned_attempts") != 24:
        raise ContractError("usable attempt matrix is not the frozen 24-row treatment")
    attempts = []
    for record in fixture_set["fixtures"]:
        for mode in modes:
            camera_included = mode != "image_recipe"
            image_attached = mode in {"image_evidence_recipe", "image_recipe"}
            prompt = None
            prompt_sha256 = None
            template_response = None
            if mode == "template_reference":
                template_response = usable_reference_response(record)
            else:
                prompt = build_usable_prompt(
                    prompt_template,
                    record,
                    camera_evidence_included=camera_included,
                )
                prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "record_id": record["record_id"],
                    "mode": mode,
                    "camera_evidence_included": camera_included,
                    "image_attached": image_attached,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "template_response": template_response,
                }
            )
    if len(attempts) != 24:
        raise ContractError("usable attempt plan has the wrong denominator")
    return attempts

"""Strict evidence-to-explanation contract for the Cosmos contract experiment."""

from decimal import Decimal, InvalidOperation
import hashlib
import json
import re


_RECORD_ID = re.compile(r"r[0-9]{3}\Z")
_DECIMAL_2 = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}\Z")
_TANKS = ("t100", "t200", "t400")
_FILL_ORDER = ("t200", "t400", "t100")
_UNCERTAIN_REASONS = {
    "compression_heavy",
    "focus_degraded",
    "lighting_variation",
    "low_margin",
}
_INVALID_REASONS = {
    "lens_obscured",
    "liquid_region_missing",
    "overexposed",
    "underexposed",
}


class ContractError(ValueError):
    """The evidence or explanation violates the frozen interface contract."""


def _exact_keys(value, expected, path):
    if not isinstance(value, dict):
        raise ContractError(f"{path} must be an object")
    extra = set(value) - set(expected)
    missing = set(expected) - set(value)
    if extra or missing:
        raise ContractError(f"{path} keys mismatch: extra={sorted(extra)}, missing={sorted(missing)}")


def _canonical_decimal(value, path):
    if not isinstance(value, str) or not _DECIMAL_2.fullmatch(value):
        raise ContractError(f"{path} must be a canonical nonnegative decimal string with two places")
    try:
        return Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex already excludes this
        raise ContractError(f"{path} is not decimal") from exc


def _validate_recipe(value, path):
    _exact_keys(value, {"recipe_id", "t100_ml", "t200_ml", "t400_ml"}, path)
    if not isinstance(value["recipe_id"], str) or not re.fullmatch(r"recipe-[0-9]{3}", value["recipe_id"]):
        raise ContractError(f"{path}.recipe_id must be neutral recipe-NNN")
    volumes = []
    for tank in _TANKS:
        amount = value[f"{tank}_ml"]
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ContractError(f"{path}.{tank}_ml must be an integer")
        maximum = 50 if tank == "t200" else 80
        if amount != 0 and not 10 <= amount <= maximum:
            raise ContractError(f"{path}.{tank}_ml violates plant recipe limits")
        volumes.append(amount)
    if not 0 < sum(volumes) <= 80:
        raise ContractError(f"{path} total must be in (0, 80] mL")


def validate_evidence_record(record):
    """Reject evidence records outside the fixed deployed-input contract."""
    _exact_keys(
        record,
        {"schema_version", "record_id", "origin", "camera_evidence", "recipe_context"},
        "record",
    )
    if record["schema_version"] != "operator_evidence.v1":
        raise ContractError("unsupported evidence schema")
    if record["origin"] != "synthetic_contract":
        raise ContractError("contract fixtures must use synthetic_contract origin")
    record_id = record["record_id"]
    if not isinstance(record_id, str) or not _RECORD_ID.fullmatch(record_id):
        raise ContractError("record_id must be neutral rNNN")

    camera = record["camera_evidence"]
    _exact_keys(camera, {
        "measurement_id", "comparison_id", "feed", "validity", "reason_codes",
        "reference_origin", "comparison_result", "delta_e76", "threshold_delta_e76",
        "direction_to_previous",
    }, "camera_evidence")
    if camera["measurement_id"] != f"measurement:{record_id}":
        raise ContractError("measurement_id must match record_id")
    if camera["comparison_id"] != f"comparison:{record_id}":
        raise ContractError("comparison_id must match record_id")
    if camera["feed"] != "vial_fill":
        raise ContractError("only the vial_fill feed is supported in v1")
    if camera["reference_origin"] != "same_recipe_vial_fill":
        raise ContractError("appearance reference must come from the vial_fill feed")
    reasons = camera["reason_codes"]
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise ContractError("reason_codes must be a string list")
    if reasons != sorted(set(reasons)):
        raise ContractError("reason_codes must be unique and sorted")

    validity = camera["validity"]
    if validity == "valid":
        if reasons:
            raise ContractError("valid camera evidence cannot have reason codes")
        if camera["comparison_result"] not in {"difference", "no_difference"}:
            raise ContractError("valid evidence needs a binary appearance comparison")
        measured = _canonical_decimal(camera["delta_e76"], "camera_evidence.delta_e76")
        threshold = _canonical_decimal(
            camera["threshold_delta_e76"], "camera_evidence.threshold_delta_e76"
        )
        is_difference = measured > threshold
        if is_difference != (camera["comparison_result"] == "difference"):
            raise ContractError("comparison_result disagrees with supplied delta-E and threshold")
        if camera["direction_to_previous"] not in {
            "toward", "not_toward", "uncertain", "unavailable"
        }:
            raise ContractError("invalid direction_to_previous for valid evidence")
    elif validity == "uncertain":
        if not reasons or not set(reasons) <= _UNCERTAIN_REASONS:
            raise ContractError("uncertain evidence needs an allowed uncertainty reason")
        if camera["comparison_result"] != "uncertain":
            raise ContractError("uncertain evidence must use uncertain comparison_result")
        if camera["delta_e76"] is not None or camera["threshold_delta_e76"] is not None:
            raise ContractError("uncertain evidence must not expose quantitative claims")
        if camera["direction_to_previous"] != "uncertain":
            raise ContractError("uncertain evidence must use uncertain direction")
    elif validity == "invalid":
        if not reasons or not set(reasons) <= _INVALID_REASONS:
            raise ContractError("invalid evidence needs an allowed invalidity reason")
        if camera["comparison_result"] != "unavailable":
            raise ContractError("invalid evidence must use unavailable comparison_result")
        if camera["delta_e76"] is not None or camera["threshold_delta_e76"] is not None:
            raise ContractError("invalid evidence must not expose quantitative claims")
        if camera["direction_to_previous"] != "unavailable":
            raise ContractError("invalid evidence must use unavailable direction")
    else:
        raise ContractError("unsupported camera validity")

    recipe = record["recipe_context"]
    _exact_keys(
        recipe,
        {"evidence_id", "current", "previous", "previous_missing_reason", "dispense_order"},
        "recipe_context",
    )
    if recipe["evidence_id"] != f"recipe:{record_id}":
        raise ContractError("recipe evidence_id must match record_id")
    _validate_recipe(recipe["current"], "recipe_context.current")
    previous = recipe["previous"]
    if previous is None:
        if recipe["previous_missing_reason"] != "not_recorded":
            raise ContractError("missing previous recipe needs not_recorded reason")
        if camera["direction_to_previous"] != "unavailable":
            raise ContractError("missing previous recipe makes direction unavailable")
    else:
        _validate_recipe(previous, "recipe_context.previous")
        if recipe["previous_missing_reason"] is not None:
            raise ContractError("present previous recipe cannot have a missing reason")
        if validity == "valid" and camera["direction_to_previous"] == "unavailable":
            raise ContractError("present previous recipe cannot use unavailable direction")
    expected_order = [
        tank for tank in _FILL_ORDER if recipe["current"][f"{tank}_ml"] > 0
    ]
    if recipe["dispense_order"] != expected_order:
        raise ContractError("dispense_order must match used tanks and fixed plant order")


def validate_fixture_set(value):
    """Validate the sealed six-record synthetic contract fixture set."""
    _exact_keys(value, {"schema_version", "fixture_origin", "fixtures"}, "fixture_set")
    if value["schema_version"] != "operator_contract_fixture_set.v1":
        raise ContractError("unsupported fixture-set schema")
    if value["fixture_origin"] != "synthetic_contract":
        raise ContractError("fixture set must use synthetic_contract origin")
    fixtures = value["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != 6:
        raise ContractError("fixture set must contain exactly six records")
    expected_ids = [f"r{index:03d}" for index in range(1, 7)]
    record_ids = [item.get("record_id") if isinstance(item, dict) else None for item in fixtures]
    if record_ids != expected_ids:
        raise ContractError("fixture records must be ordered neutral IDs r001 through r006")
    for record in fixtures:
        validate_evidence_record(record)


def validate_oracle(value, fixture_bytes, fixture_set):
    """Validate the separately stored scoring oracle against frozen fixture bytes."""
    _exact_keys(
        value,
        {"schema_version", "fixture_set_sha256", "expected_by_record"},
        "oracle",
    )
    if value["schema_version"] != "operator_contract_oracle.v1":
        raise ContractError("unsupported oracle schema")
    if not isinstance(fixture_bytes, bytes):
        raise ContractError("fixture_bytes must be bytes")
    actual_hash = hashlib.sha256(fixture_bytes).hexdigest()
    if value["fixture_set_sha256"] != actual_hash:
        raise ContractError("oracle fixture hash mismatch")
    validate_fixture_set(fixture_set)
    fixtures = {record["record_id"]: record for record in fixture_set["fixtures"]}
    expected_by_record = value["expected_by_record"]
    if not isinstance(expected_by_record, dict) or set(expected_by_record) != set(fixtures):
        raise ContractError("oracle record IDs do not match fixtures")
    for record_id, record in fixtures.items():
        expected = expected_by_record[record_id]
        _exact_keys(expected, {"full", "image_recipe"}, f"oracle.{record_id}")
        parse_and_validate_explanation(canonical_json(expected["full"]), record)
        parse_and_validate_explanation(
            canonical_json(expected["image_recipe"]),
            record,
            camera_evidence_included=False,
        )


def validate_eval_config(value):
    """Validate an exact frozen contract matrix and immutable inference settings."""
    v1_keys = {
        "schema_version", "experiment_id", "freeze_tag", "fixture_count", "modes",
        "planned_attempts", "required_abstention_attempts", "image_invariance_pairs",
        "template_control_attempts", "seed", "do_sample", "max_new_tokens", "dtype",
        "python_version", "platform_machine", "runtime_versions",
        "model_repository", "model_revision", "model_manifest_sha256", "runtime_lock_sha256",
        "fixture_set_sha256", "oracle_sha256", "prompt_sha256", "image_sha256",
        "image_role", "output_path",
    }
    v2_keys = v1_keys | {
        "v1_freeze_tag", "enable_thinking", "input_audit_sha256",
        "chat_template_sha256", "scoring_module_sha256", "scoring_script_sha256",
        "v2_scoring_module_sha256", "contract_module_sha256",
    }
    if not isinstance(value, dict):
        raise ContractError("eval_config must be an object")
    schema_version = value.get("schema_version")
    if schema_version == "operator_contract_eval.v1":
        _exact_keys(value, v1_keys, "eval_config")
        version_fixed = {
            "experiment_id": "cosmos-contract-v1-20261024",
            "freeze_tag": "explore/cosmos-contract-v1-frozen-20261024",
            "output_path": "outputs/20261024-cosmos-contract-v1/raw-results.json",
        }
    elif schema_version == "operator_contract_eval.v2":
        _exact_keys(value, v2_keys, "eval_config")
        version_fixed = {
            "experiment_id": "cosmos-contract-v2-no-thinking-20261024",
            "freeze_tag": "explore/cosmos-contract-v2-frozen-20261024",
            "output_path": "outputs/20261024-cosmos-contract-v2/raw-results.json",
            "v1_freeze_tag": "explore/cosmos-contract-v1-frozen-20261024",
            "enable_thinking": False,
            "chat_template_sha256": (
                "7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8"
            ),
            "scoring_module_sha256": (
                "9721fd1cf7d1613628eb36b9c9339efab807b02d277038eebfa3af11553debd1"
            ),
            "model_manifest_sha256": (
                "b02c5c400d920d3e79219fcfc3a1bc09ac367f9cc5aaf3e9aff30244f2bf602f"
            ),
            "runtime_lock_sha256": (
                "6272a3c0107cc5b36285ea3bf268f3b8d3245212e7a700e1b6ed525211c3c2d4"
            ),
            "fixture_set_sha256": (
                "e88f6f13476ae89a49d53e0ddfa6cd2be051530c073e946f5e985c16d2f0ee9f"
            ),
            "oracle_sha256": (
                "cb2d45ca1fce48e7603a6cde52ff0cbb1bdad11e7b2285ad01b80cbd232faab9"
            ),
            "prompt_sha256": (
                "0733da91d7c1c347feae3afabdc170ba3a41fa5a9726f310a40eab887f3eb74c"
            ),
            "image_sha256": (
                "c4886724246a6dbef76c6658b07c8568c5e8e4372f7a9a623021a3ddeb3997af"
            ),
        }
    else:
        raise ContractError("unsupported eval-config schema")

    fixed = {
        "fixture_count": 6,
        "modes": [
            "image_evidence_recipe",
            "evidence_recipe",
            "image_recipe",
            "template_reference",
        ],
        "planned_attempts": 24,
        "required_abstention_attempts": 15,
        "image_invariance_pairs": 6,
        "template_control_attempts": 6,
        "seed": 0,
        "do_sample": False,
        "max_new_tokens": 384,
        "dtype": "bfloat16",
        "python_version": "3.13.17",
        "platform_machine": "aarch64",
        "runtime_versions": {
            "accelerate": "1.12.0",
            "av": "16.0.1",
            "numpy": "2.5.4",
            "pillow": "12.3.0",
            "safetensors": "0.8.0",
            "torch": "2.10.0+cu130",
            "torchvision": "0.25.0+cu130",
            "transformers": "5.17.0",
        },
        "model_repository": "nvidia/Cosmos3-Edge",
        "model_revision": "344d602b128d1bbdacb43b08d0a3626f46343e29",
        "image_role": "irrelevant_fixed_fillcam_distraction",
        **version_fixed,
    }
    for key, expected in fixed.items():
        if value[key] != expected or type(value[key]) is not type(expected):
            raise ContractError(f"eval_config.{key} is not the frozen value")
    digest_keys = [
        "model_manifest_sha256", "runtime_lock_sha256", "fixture_set_sha256",
        "oracle_sha256", "prompt_sha256", "image_sha256",
    ]
    if schema_version == "operator_contract_eval.v2":
        digest_keys.extend([
            "input_audit_sha256", "chat_template_sha256",
            "scoring_module_sha256", "scoring_script_sha256",
            "v2_scoring_module_sha256", "contract_module_sha256",
        ])
    for key in digest_keys:
        digest = value[key]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"eval_config.{key} must be a lowercase SHA-256")


def reference_explanation(record, *, camera_evidence_included=True):
    """Return the deterministic reference-policy response for one evidence record."""
    validate_evidence_record(record)
    camera = record["camera_evidence"]
    recipe = record["recipe_context"]
    comparison_id = camera["comparison_id"]
    evidence_valid = (
        camera_evidence_included
        and camera["validity"] == "valid"
        and recipe["previous"] is not None
    )
    difference = evidence_valid and camera["comparison_result"] == "difference"
    no_difference = evidence_valid and camera["comparison_result"] == "no_difference"
    limits = [
        "cause_not_verified",
        "defect_not_verified",
        "next_vial_effect_not_measured",
    ]
    if not camera_evidence_included:
        limits.append("quantitative_evidence_omitted")
    elif camera["validity"] == "uncertain":
        limits.append("camera_evidence_uncertain")
    elif camera["validity"] == "invalid":
        limits.append("camera_evidence_invalid")
    if recipe["previous"] is None:
        limits.append("previous_recipe_missing")
    limits.sort()
    status = "appearance_only" if evidence_valid else "insufficient_evidence"
    if camera_evidence_included and camera["validity"] == "invalid":
        status = "invalid_input"
    evidence_ids = [recipe["evidence_id"]]
    if camera_evidence_included:
        evidence_ids.extend([comparison_id, camera["measurement_id"]])
    return {
        "schema_version": "operator_explanation.v1",
        "record_id": record["record_id"],
        "status": status,
        "appearance": (
            "difference_observed" if difference
            else "no_difference_observed" if no_difference
            else "unknown"
        ),
        "defect": "unknown",
        "cause": "unknown",
        "next_vial_effect": "unknown",
        "confidence": "not_assigned",
        "evidence_ids": sorted(evidence_ids),
        "quantitative_evidence": ([
            {
                "evidence_id": comparison_id,
                "metric": "delta_e76",
                "value": camera["delta_e76"],
                "unit": "delta_e76",
            },
            {
                "evidence_id": comparison_id,
                "metric": "threshold_delta_e76",
                "value": camera["threshold_delta_e76"],
                "unit": "delta_e76",
            },
        ] if evidence_valid else []),
        "limit_codes": limits,
        "operator_action": "no_automatic_action" if no_difference else "manual_review_required",
    }


def canonical_json(value):
    """Serialize a tracked contract value without presentation-dependent whitespace."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def build_prompt(template, record, *, camera_evidence_included=True):
    """Insert one canonical runner-visible record into the frozen prompt template."""
    validate_evidence_record(record)
    if not isinstance(template, str) or template.count("{{EVIDENCE_RECORD_JSON}}") != 1:
        raise ContractError("prompt template must contain the evidence placeholder exactly once")
    visible_record = record
    if not camera_evidence_included:
        visible_record = {**record, "camera_evidence": None}
    return template.replace("{{EVIDENCE_RECORD_JSON}}", canonical_json(visible_record))


def parse_and_validate_explanation(raw, record, *, camera_evidence_included=True):
    """Parse exactly one JSON object and validate it against the evidence policy."""
    parsed = parse_explanation_shape(raw)
    expected = reference_explanation(
        record,
        camera_evidence_included=camera_evidence_included,
    )
    if parsed != expected:
        raise ContractError("explanation does not match the frozen evidence policy")
    return parsed


def parse_explanation_shape(raw):
    """Parse one unwrapped JSON object and enforce the closed output shape."""
    if not isinstance(raw, str):
        raise ContractError("model response must be text")

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ContractError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ContractError(f"non-finite JSON constant: {value}")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("response is not one strict JSON value") from exc
    if not isinstance(parsed, dict):
        raise ContractError("explanation must be a JSON object")
    _exact_keys(parsed, {
        "schema_version", "record_id", "status", "appearance", "defect", "cause",
        "next_vial_effect", "confidence", "evidence_ids", "quantitative_evidence",
        "limit_codes", "operator_action",
    }, "explanation")
    if parsed["schema_version"] != "operator_explanation.v1":
        raise ContractError("unsupported explanation schema")
    if not isinstance(parsed["record_id"], str) or not _RECORD_ID.fullmatch(parsed["record_id"]):
        raise ContractError("explanation record_id must be neutral rNNN")
    enums = {
        "status": {"appearance_only", "insufficient_evidence", "invalid_input"},
        "appearance": {"difference_observed", "no_difference_observed", "unknown"},
        "defect": {"unknown"},
        "cause": {"unknown"},
        "next_vial_effect": {"unknown"},
        "confidence": {"not_assigned"},
        "operator_action": {"manual_review_required", "no_automatic_action"},
    }
    for key, allowed in enums.items():
        if parsed[key] not in allowed or not isinstance(parsed[key], str):
            raise ContractError(f"invalid explanation enum: {key}")
    evidence_ids = parsed["evidence_ids"]
    if (
        not isinstance(evidence_ids, list)
        or any(not isinstance(item, str) for item in evidence_ids)
        or evidence_ids != sorted(set(evidence_ids))
    ):
        raise ContractError("evidence_ids must be unique sorted strings")
    quantitative = parsed["quantitative_evidence"]
    if not isinstance(quantitative, list):
        raise ContractError("quantitative_evidence must be a list")
    metrics = []
    for index, item in enumerate(quantitative):
        _exact_keys(item, {"evidence_id", "metric", "value", "unit"}, f"quantitative[{index}]")
        if not all(isinstance(item[key], str) for key in item):
            raise ContractError("quantitative fields must be strings")
        if item["metric"] not in {"delta_e76", "threshold_delta_e76"}:
            raise ContractError("unsupported quantitative metric")
        _canonical_decimal(item["value"], f"quantitative[{index}].value")
        if item["unit"] != "delta_e76":
            raise ContractError("unsupported quantitative unit")
        metrics.append(item["metric"])
    if metrics != sorted(set(metrics)):
        raise ContractError("quantitative metrics must be unique and sorted")
    limit_codes = parsed["limit_codes"]
    allowed_limits = {
        "camera_evidence_invalid", "camera_evidence_uncertain", "cause_not_verified",
        "defect_not_verified", "next_vial_effect_not_measured", "previous_recipe_missing",
        "quantitative_evidence_omitted",
    }
    if (
        not isinstance(limit_codes, list)
        or any(item not in allowed_limits for item in limit_codes)
        or limit_codes != sorted(set(limit_codes))
    ):
        raise ContractError("limit_codes must be allowed, unique and sorted")
    return parsed

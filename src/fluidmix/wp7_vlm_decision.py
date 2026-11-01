"""CPU-only image-grounded WP7 vial-1 decision contracts.

This module never loads a model. It consumes hash-bound camera/CNN observations;
the accepted sequential VLM runtime remains ``model_bakeoff_runner``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from fluidmix.model_bakeoff_runner import MODEL_SPECS


class ContractError(ValueError):
    """A WP7 decision artifact violates the frozen contract."""


PHASES = ("early", "mid", "end")
DISPOSITIONS = {"keep", "discard", "insufficient_evidence"}
AGGREGATION = {"threshold": 0.5, "minimum_valid_sources": 2, "weighting": "equal"}
MODELS = tuple(
    {"repository": repository, "revision": MODEL_SPECS[repository].revision}
    for repository in (
        "Qwen/Qwen3-VL-8B-Instruct", "nvidia/Cosmos-Reason2-8B",
        "aacudad/AnomalyThink-Qwen3-VL-8B-KCR",
    )
)
SYSTEM_PROMPT = (
    "Assess only the delivered vial-1 camera evidence against the delivered vial-2 and "
    "vial-3 references. Do not infer hidden residue, simulator state, cause, or operator "
    "action. Return exactly one JSON object matching the response schema."
)
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["disposition", "discard_score", "explanation", "evidence_citations", "image_citations"],
    "properties": {
        "disposition": {"enum": sorted(DISPOSITIONS)},
        "discard_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "explanation": {"type": "string"},
        "evidence_citations": {"type": "array", "items": {"type": "string"}},
        "image_citations": {"type": "array", "items": {"type": "string"}},
    },
}
USER_PROMPT_TEMPLATE = "Response schema:\n{schema}\nEvidence bundle:\n{bundle}"
_FORBIDDEN = (
    "simulator", "residue", "palette", "oracle", "semantic_mask", "particles",
    "truth", "test_label", "expected_lab", "beer_lambert", "absorbance",
)
_PATH = re.compile(r"[\\/]|(?:^|[^.])\.\.(?:[^.]|$)")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class BoundImageLocator:
    path: Path
    sha256: str


@dataclass(frozen=True)
class BuiltBundle:
    prompt_payload: dict[str, Any]
    image_locators: dict[str, BoundImageLocator]
    system_prompt: str
    user_prompt: str


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ContractError("artifact contains nonfinite or non-JSON value") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: object, role: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractError(f"{role} is nonfinite or outside range")
    return result


def _lab(value: object, role: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError(f"{role} must have three channels")
    return [_finite(item, role) for item in value]


def _measurement(value: Mapping[str, object], *, target: bool) -> dict[str, object]:
    for field in ("median_lab", "mad_lab", "iqr_lab"):
        _lab(value[field], field)
    for field in ("delta_e76_vial2", "delta_e76_vial3"):
        _finite(value[field], field, 0.0)
    if value["deterministic_disposition"] not in DISPOSITIONS:
        raise ContractError("deterministic disposition is invalid")
    if not isinstance(value["evidence_id"], str) or not value["evidence_id"]:
        raise ContractError("evidence ID is invalid")
    if not isinstance(value["limit_codes"], list) or any(
        not isinstance(item, str) or not item for item in value["limit_codes"]
    ):
        raise ContractError("deterministic limit codes are invalid")
    result = dict(value)
    if not target:
        result.pop("deterministic_disposition")
    return result


def validate_tracker_receipt(value: object) -> dict[str, Any]:
    """Delegate to the dedicated tracker's exact closed-schema validator."""
    from fluidmix.wp7_fill_step_tracker import (
        FillStepTrackerError,
        validate_fill_step_receipt,
    )

    try:
        checked = validate_fill_step_receipt(value)
    except (FillStepTrackerError, TypeError, ValueError) as exc:
        raise ContractError(f"tracker receipt invalid: {exc}") from exc
    for stage in checked["stages"]:
        images = stage["images"]
        if len(set(images.values())) != 3:
            raise ContractError("tracker early/mid/end images must be distinct")
        if stage["tank_cluster_consistent"] is not True:
            raise ContractError("tracker tank cluster is inconsistent")
    return checked


def _tracker_cohort(receipts: object, cohort_id: object, vials: list[object]) -> list[dict[str, Any]]:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)) or len(receipts) != 3:
        raise ContractError("exactly three tracker receipts are required")
    checked = [validate_tracker_receipt(item) for item in receipts]
    if any(item["cohort_id"] != cohort_id for item in checked):
        raise ContractError("tracker/cohort identity mismatch")
    expected_vial_ids = [item.get("vial_id") for item in sorted(vials, key=lambda row: row["vial_position"])]
    if [item["vial_id"] for item in checked] != expected_vial_ids:
        raise ContractError("tracker vial order/identity mismatch")
    for field in ("calibration_sha256", "recipe_artifact_sha256", "source_artifact_sha256", "claim_boundary"):
        if len({item[field] for item in checked}) != 1:
            raise ContractError(f"tracker {field} differs within cohort")
    recipe_keys = (
        "fill_step", "tank_id", "tank_number", "stream_step_index",
        "expected_dispense_ml", "expected_cumulative_ml",
    )
    target_recipe = [tuple(stage[key] for key in recipe_keys) for stage in checked[0]["stages"]]
    if any(
        [tuple(stage[key] for key in recipe_keys) for stage in receipt["stages"]] != target_recipe
        for receipt in checked[1:]
    ):
        raise ContractError("tracker recipe state differs within cohort")
    return checked


def _scan(value: object, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN):
                raise ContractError(f"forbidden field at {path}/{key}")
            _scan(item, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, f"{path}/{index}")
    elif isinstance(value, str) and (_PATH.search(value) or value.startswith(("file:", "~"))):
        raise ContractError(f"path-like prompt value at {path}")


def _camera(value: object, production: bool) -> dict[str, object]:
    fields = ("resolution", "frame_rate_hz", "exposure", "white_balance", "viewpoint_id", "vial_roi")
    if not isinstance(value, Mapping) or not set(value).issubset(fields):
        raise ContractError("camera metadata fields are invalid")
    result = {field: value.get(field, "unavailable") for field in fields}
    result = {key: "unavailable" if item is None else item for key, item in result.items()}
    if result["frame_rate_hz"] != "unavailable":
        _finite(result["frame_rate_hz"], "camera frame rate", 0.0)
    for field in ("resolution", "vial_roi"):
        if result[field] != "unavailable" and (
            not isinstance(result[field], list)
            or len(result[field]) not in {2, 4}
            or any(type(item) is not int or item < 0 for item in result[field])
        ):
            raise ContractError(f"camera {field} is invalid")
    if production and any(result[key] == "unavailable" for key in ("resolution", "frame_rate_hz", "viewpoint_id", "vial_roi")):
        raise ContractError("production camera metadata is incomplete")
    return result


def build_vial1_bundle(
    cohort: object, image_locators: object, *, tracker_receipts: object,
    production: bool = False,
) -> BuiltBundle:
    """Build a path-free bundle and verify a separate internal locator map."""
    if not isinstance(cohort, Mapping) or not isinstance(image_locators, Mapping):
        raise ContractError("cohort and locators must be objects")
    _scan(cohort)
    cohort_fields = {
        "schema_version", "cohort_id", "partition", "recipe_id",
        "previous_recipe_id", "camera_metadata", "stages", "vials",
    }
    if set(cohort) not in (cohort_fields, cohort_fields | {"target_vial_position"}):
        raise ContractError("cohort fields are not closed")
    if cohort.get("schema_version") != "wp7-vlm-cohort-v1":
        raise ContractError("cohort schema is invalid")
    if cohort.get("target_vial_position", 1) != 1:
        raise ContractError("only vial 1 may be scored")
    vials, stages = cohort.get("vials"), cohort.get("stages")
    if not isinstance(vials, list) or len(vials) != 3 or {v.get("vial_position") for v in vials if isinstance(v, Mapping)} != {1, 2, 3}:
        raise ContractError("complete three vials are required")
    if not isinstance(stages, list) or not stages:
        raise ContractError("cohort stages are required")
    stage_ids = [stage.get("stage_id") for stage in stages if isinstance(stage, Mapping)]
    tracker = _tracker_cohort(tracker_receipts, cohort.get("cohort_id"), vials)
    images, evidence, bound = [], [], {}
    for vial, tracker_receipt in zip(
        sorted(vials, key=lambda row: row["vial_position"]), tracker, strict=True
    ):
        if set(vial) != {"vial_position", "vial_id", "stages"}:
            raise ContractError("vial fields are not closed")
        vial_stages = vial.get("stages")
        if (
            not isinstance(vial_stages, list)
            or [row.get("stage_id") for row in vial_stages if isinstance(row, Mapping)] != stage_ids
            or len(tracker_receipt["stages"]) != len(stage_ids)
        ):
            raise ContractError("vial stage/cohort mismatch")
        for stage_index, (stage, tracker_stage) in enumerate(
            zip(vial_stages, tracker_receipt["stages"], strict=True)
        ):
            if tracker_stage["fill_step"] != stage_index + 1:
                raise ContractError("tracker fill-step order is invalid")
            for phase in PHASES:
                image_id = tracker_stage["images"][phase]
                locator = image_locators.get(image_id)
                if not isinstance(locator, Mapping):
                    raise ContractError(f"unbound image ID: {image_id}")
                if set(locator) != {"path", "sha256"}:
                    raise ContractError("image locator fields are invalid")
                path, digest = Path(locator["path"]), locator["sha256"]
                if path.is_symlink() or not path.is_file() or not isinstance(digest, str) or _SHA.fullmatch(digest) is None or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ContractError("image locator/hash binding failed")
                if image_id in bound:
                    raise ContractError("tracker image ID is reused across the cohort")
                bound[image_id] = BoundImageLocator(path.resolve(), digest)
                images.append({"image_id": image_id, "vial_position": vial["vial_position"], "stage_id": stage["stage_id"], "phase": phase})
            if not isinstance(stage.get("measurement"), Mapping):
                raise ContractError("deterministic measurement is missing")
            measurement_fields = {
                "evidence_id", "median_lab", "mad_lab", "iqr_lab",
                "delta_e76_vial2", "delta_e76_vial3",
                "deterministic_disposition", "limit_codes",
            }
            if set(stage["measurement"]) != measurement_fields:
                raise ContractError("deterministic measurement fields are not closed")
            projected_measurement = _measurement(
                stage["measurement"], target=vial["vial_position"] == 1
            )
            evidence.append({
                "vial_position": vial["vial_position"],
                "evidence_role": "target" if vial["vial_position"] == 1 else "reference",
                "stage_id": stage["stage_id"],
                **projected_measurement,
            })
    allowed = {"stage_id", "stage_ordinal", "tank_id", "programmed_ml", "dispense_order", "stage_timing_seconds", "stream"}
    safe_stages = []
    target_tracker_stages = tracker[0]["stages"]
    for stage, tracker_stage in zip(stages, target_tracker_stages, strict=True):
        if set(stage) != allowed:
            raise ContractError("stage fields are invalid or forbidden")
        timing, stream = stage["stage_timing_seconds"], stage["stream"]
        if not isinstance(timing, list) or len(timing) != 2 or _finite(timing[1], "stage timing") <= _finite(timing[0], "stage timing") or not isinstance(stream, Mapping) or set(stream) != {"evidence_id", "median_lab", "spread_lab"}:
            raise ContractError("stage timing/stream is invalid")
        programmed_ml = _finite(stage["programmed_ml"], "programmed mL", 0.0)
        if programmed_ml <= 0 or stage["tank_id"].lower() != tracker_stage["tank_id"] or programmed_ml != tracker_stage["expected_dispense_ml"]:
            raise ContractError("tracker and recipe stage differ")
        safe_stages.append({
            **stage,
            "programmed_ml": programmed_ml,
            "stream": {
                **stream,
                "median_lab": _lab(tracker_stage["stream_lab"], "stream Lab"),
                "spread_lab": _lab(tracker_stage["stream_spread_lab"], "stream spread"),
            },
            "camera_fill_state": {
                key: tracker_stage[key]
                for key in (
                    "fill_step", "tank_id", "tank_number", "stream_step_index",
                    "expected_dispense_ml", "expected_cumulative_ml",
                    "twin_geometry_equivalent_volume_ml", "agreement",
                )
            },
        })
    payload = {
        "schema_version": "wp7-vlm-prompt-bundle-v1", "cohort_id": cohort.get("cohort_id"),
        "partition": cohort.get("partition"), "recipe_id": cohort.get("recipe_id"), "previous_recipe_id": cohort.get("previous_recipe_id"),
        "target_vial_position": 1, "camera_metadata": _camera(cohort.get("camera_metadata", {}), production),
        "stages": safe_stages, "images": images, "deterministic_evidence": evidence,
        "tracker_binding": {
            "receipt_sha256": [item["receipt_sha256"] for item in tracker],
            "calibration_sha256": tracker[0]["calibration_sha256"],
            "claim_boundary": tracker[0]["claim_boundary"],
        },
    }
    _scan(payload)
    encoded_payload = _canonical(payload).decode()
    user = USER_PROMPT_TEMPLATE.format(schema=json.dumps(RESPONSE_SCHEMA, sort_keys=True), bundle=encoded_payload)
    return BuiltBundle(payload, bound, SYSTEM_PROMPT, user)


def parse_model_response(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or len(raw.encode()) > 65536:
        raise ContractError("model response is not bounded text")
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text, re.I)
    if match:
        text = match.group(1)
    elif "```" in text:
        raise ContractError("malformed response fence")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError("model response is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ContractError("model response must be one object")
    return value


def guard_model_response(value: object, bundle: object) -> dict[str, Any]:
    base = {"vote": None, "guard_status": "excluded", "guard_reasons": []}
    if not isinstance(value, Mapping) or not isinstance(bundle, Mapping):
        return {**base, "guard_reasons": ["response_or_bundle_type_invalid"]}
    reasons = []
    expected = {"disposition", "discard_score", "explanation", "evidence_citations", "image_citations"}
    if set(value) != expected: reasons.append("response_fields_invalid")
    disposition, score = value.get("disposition"), value.get("discard_score")
    if disposition not in DISPOSITIONS: reasons.append("disposition_invalid")
    try:
        score = _finite(score, "discard score", 0.0)
        if score > 1: reasons.append("discard_score_out_of_range")
    except ContractError: reasons.append("discard_score_invalid")
    explanation = value.get("explanation")
    if not isinstance(explanation, str) or not explanation: reasons.append("explanation_invalid")
    else:
        try: _scan(explanation)
        except ContractError: reasons.append("path_or_forbidden_text")
    allowed_evidence = {row.get("evidence_id") for row in bundle.get("deterministic_evidence", []) if isinstance(row, Mapping)}
    allowed_images = {row.get("image_id") for row in bundle.get("images", []) if isinstance(row, Mapping)}
    if isinstance(explanation, str):
        mentioned = set(re.findall(r"(?:IMG|EV)-[A-Za-z0-9_-]+", explanation))
        if not mentioned.issubset(allowed_evidence | allowed_images):
            reasons.append("explanation_citation_invalid")
    for field, allowed_ids in (("evidence_citations", allowed_evidence), ("image_citations", allowed_images)):
        citations = value.get(field)
        if not isinstance(citations, list) or not citations or any(not isinstance(item, str) or item not in allowed_ids for item in citations): reasons.append(f"{field}_invalid")
    result = {**dict(value), **base, "guard_reasons": list(dict.fromkeys(reasons))}
    if not reasons:
        result["guard_status"] = "admitted"
        result["vote"] = None if disposition == "insufficient_evidence" else score
    return result


def aggregate_decision(deterministic: object, model_results: object) -> dict[str, Any]:
    if (
        not isinstance(deterministic, Mapping)
        or deterministic.get("disposition") not in DISPOSITIONS
        or not isinstance(model_results, Sequence)
        or isinstance(model_results, (str, bytes))
        or any(not isinstance(row, Mapping) for row in model_results)
    ):
        raise ContractError("aggregation inputs are invalid")
    det_vote = {"keep": 0.0, "discard": 1.0, "insufficient_evidence": None}[deterministic["disposition"]]
    sources = [{"source_id": "deterministic", **dict(deterministic), "vote": det_vote, "guard_status": "admitted" if det_vote is not None else "no_vote"}]
    sources.extend(dict(row) for row in model_results)
    votes = [
        float(row["vote"])
        for row in sources
        if row.get("guard_status") == "admitted"
        and isinstance(row.get("vote"), (int, float))
        and not isinstance(row.get("vote"), bool)
        and math.isfinite(float(row["vote"]))
        and 0 <= float(row["vote"]) <= 1
    ]
    mean = sum(votes) / len(votes) if votes else None
    final = "insufficient_evidence" if len(votes) < 2 else ("discard" if mean >= 0.5 else "keep")
    return {"schema_version": "wp7-vlm-aggregation-receipt-v1", "aggregation": dict(AGGREGATION), "sources": sources, "valid_source_count": len(votes), "mean_discard_score": mean, "final_disposition": final}


def create_development_freeze(
    path: str | Path, development_inputs: Sequence[str | Path], *,
    tracker_receipts: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Publish a freeze only with accepted development tracker receipts."""
    if tracker_receipts is None or len(tracker_receipts) != 3:
        raise ContractError("freeze requires three accepted tracker receipts")
    trackers = [validate_tracker_receipt(item) for item in tracker_receipts]
    if len({item["cohort_id"] for item in trackers}) != 1:
        raise ContractError("freeze tracker receipts differ by cohort")
    records = []
    for raw in development_inputs:
        item = Path(raw)
        if item.is_symlink() or not item.is_file(): raise ContractError("development input is absent")
        records.append({"name": item.name, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
    if len({item["name"] for item in records}) != len(records):
        raise ContractError("development input names must be unique")
    output = Path(path)
    if output.exists() or output.is_symlink(): raise ContractError("refusing to overwrite freeze")
    freeze = {
        "schema_version": "wp7-vlm-development-freeze-v1", "development_only": True,
        "decision_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tracker_schema_version": "wp7-camera-fill-step-receipt-v1",
        "tracker_receipt_sha256": [item["receipt_sha256"] for item in trackers],
        "tracker_calibration_sha256": sorted({item["calibration_sha256"] for item in trackers}),
        "tracker_source_artifact_sha256": sorted({item["source_artifact_sha256"] for item in trackers}),
        "tracker_validator_source_sha256": hashlib.sha256(
            Path(__file__).with_name("wp7_fill_step_tracker.py").read_bytes()
        ).hexdigest(),
        "prompt_sha256": _digest({"system": SYSTEM_PROMPT, "user_template": USER_PROMPT_TEMPLATE}), "response_schema_sha256": _digest(RESPONSE_SCHEMA),
        "parser_id": "strict-json-with-one-bounded-three-backtick-fence-v1", "aggregation": dict(AGGREGATION), "models": list(MODELS),
        "development_inputs": sorted(records, key=lambda row: row["name"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as stream: stream.write(json.dumps(freeze, sort_keys=True, indent=2) + "\n")
    return freeze


def _validate_freeze(value: object) -> None:
    if not isinstance(value, Mapping): raise ContractError("freeze is invalid")
    fields = {
        "schema_version", "development_only", "decision_source_sha256",
        "tracker_schema_version", "tracker_receipt_sha256",
        "tracker_calibration_sha256", "tracker_source_artifact_sha256",
        "tracker_validator_source_sha256",
        "prompt_sha256", "response_schema_sha256", "parser_id", "aggregation",
        "models", "development_inputs",
    }
    if set(value) != fields:
        raise ContractError("freeze fields are not closed")
    expected = {
        "schema_version": "wp7-vlm-development-freeze-v1", "development_only": True,
        "tracker_schema_version": "wp7-camera-fill-step-receipt-v1",
        "prompt_sha256": _digest({"system": SYSTEM_PROMPT, "user_template": USER_PROMPT_TEMPLATE}), "response_schema_sha256": _digest(RESPONSE_SCHEMA),
        "parser_id": "strict-json-with-one-bounded-three-backtick-fence-v1", "aggregation": AGGREGATION, "models": list(MODELS),
    }
    if any(value.get(key) != item for key, item in expected.items()): raise ContractError("freeze constants do not match code")
    if value.get("decision_source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ContractError("freeze source does not match code")
    if not all(isinstance(value.get(field), list) and value[field] for field in (
        "tracker_receipt_sha256", "tracker_calibration_sha256", "tracker_source_artifact_sha256",
    )):
        raise ContractError("freeze tracker binding is incomplete")
    if len(value["tracker_receipt_sha256"]) != 3 or any(
        not isinstance(item, str) or _SHA.fullmatch(item) is None
        for field in (
            "tracker_receipt_sha256", "tracker_calibration_sha256",
            "tracker_source_artifact_sha256",
        )
        for item in value[field]
    ):
        raise ContractError("freeze tracker hashes are invalid")
    tracker_source = Path(__file__).with_name("wp7_fill_step_tracker.py")
    if value["tracker_validator_source_sha256"] != hashlib.sha256(tracker_source.read_bytes()).hexdigest():
        raise ContractError("freeze tracker validator source differs")
    if not isinstance(value["development_inputs"], list) or not value["development_inputs"]:
        raise ContractError("freeze development inputs are absent")
    for item in value["development_inputs"]:
        if not isinstance(item, Mapping) or set(item) != {"name", "sha256"} or not isinstance(item["name"], str) or _SHA.fullmatch(str(item["sha256"])) is None:
            raise ContractError("freeze development input binding is invalid")


def load_production_bundle(
    bundle_path: str | Path, freeze_path: str | Path, *, expected_freeze_sha256: str
) -> dict[str, Any]:
    """Validate freeze before touching the production bundle path."""
    freeze_file = Path(freeze_path)
    if freeze_file.is_symlink() or not freeze_file.is_file(): raise ContractError("matching development freeze is required")
    raw_freeze = freeze_file.read_bytes()
    if _SHA.fullmatch(expected_freeze_sha256) is None or hashlib.sha256(raw_freeze).hexdigest() != expected_freeze_sha256:
        raise ContractError("freeze does not match reviewed SHA-256")
    try: freeze = json.loads(raw_freeze)
    except (OSError, json.JSONDecodeError) as exc: raise ContractError("freeze is invalid") from exc
    _validate_freeze(freeze)
    path = Path(bundle_path)
    if path.is_symlink() or not path.is_file(): raise ContractError("production bundle is unavailable")
    try: value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: raise ContractError("production bundle is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "wp7-vlm-prompt-bundle-v1": raise ContractError("production bundle schema is invalid")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the development-only WP7 VLM decision contract")
    parser.add_argument("--output", required=True); parser.add_argument("--development-input", action="append", required=True)
    parser.add_argument("--tracker-receipt", action="append", required=True)
    args = parser.parse_args(argv)
    receipts = [json.loads(Path(item).read_text()) for item in args.tracker_receipt]
    create_development_freeze(args.output, args.development_input, tracker_receipts=receipts)
    return 0

"""CPU-only contracts and deterministic scoring for frozen RCA-bench v1.

This module deliberately contains no model loader and never derives model-facing
records from simulator truth.  It validates already-projected Part B records,
constructs synthetic controls, journals one-pass execution, and scores sealed raw
responses only after a separate oracle is supplied.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any


PROTOCOL_ID = "rca-bench-v1-20261028"
MODEL_INPUT_SCHEMA = "rca-bench-model-input-v1"
PLAN_SCHEMA = "rca-bench-plan-v1"
LOCATOR_SCHEMA = "rca-bench-image-locators-v1"
ORACLE_SCHEMA = "rca-bench-oracle-v1"
RAW_SCHEMA = "rca-bench-raw-v1"
SCORE_SCHEMA = "rca-bench-score-v1"
FREEZE_SCHEMA = "rca-bench-freeze-receipt-v1"
LEDGER_SCHEMA = "rca-bench-run-ledger-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_FORBIDDEN = re.compile(
    r"(?:^|[-_:])(keep|discard|insufficient|residue|carryover|palette|oracle|label)(?:$|[-_:])",
    re.IGNORECASE,
)
_SIM_KEYS = {
    "alpha",
    "composition",
    "expected_disposition",
    "fault",
    "host_profile",
    "injected_residue",
    "label",
    "palette",
    "particle_count",
    "particle_state",
    "render_seed",
    "residual_volume",
    "sim_metadata",
    "tank_colour",
}
_POST_INFERENCE_KEYS = {
    "explanation_id",
    "guard_result",
    "image_ids_sent",
    "model_revision",
    "prompt_sha256",
}
_EVIDENCE_SPLIT_ROLES = {
    "train", "val", "validation", "evaluation", "test", "real_validation", "calibration"
}


class ContractError(ValueError):
    """A frozen RCA-bench input or output violates its exact contract."""


def canonical_bytes(value: object) -> bytes:
    """Return the frozen UTF-8 canonical JSON representation (without newline)."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("value is not canonical finite JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_canonical_json_exclusive(path: str | Path, value: object) -> Path:
    """Publish one canonical artifact atomically and refuse every overwrite."""
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ContractError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ContractError("output parent must be a real directory")
    temporary = output.parent / f".{output.name}.incomplete"
    if temporary.exists() or temporary.is_symlink():
        raise ContractError("temporary output is occupied")
    payload = canonical_bytes(value)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        output.chmod(0o444)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return output


def _object(value: object, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError(f"{role} fields are invalid")
    return value


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{role} must be nonempty text")
    return value


def _integer(value: object, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{role} must be an integer >= {minimum}")
    return value


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{role} must be finite numeric")
    return float(value)


def _digest(value: object, role: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractError(f"{role} must be a lowercase SHA-256")
    return value


def _opaque(value: object, role: str) -> str:
    text = _text(value, role)
    if _OPAQUE_FORBIDDEN.search(text):
        raise ContractError(f"{role} must be opaque and cannot encode an answer or simulator truth")
    return text


def _reject_recursive(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if (
                normalized in _SIM_KEYS
                or normalized in _POST_INFERENCE_KEYS
                or normalized.startswith("sim_")
                or any(token in normalized for token in ("palette", "residue", "particle", "host_profile"))
            ):
                raise ContractError(f"forbidden simulator/post-inference field: {key}")
            _reject_recursive(child)
    elif isinstance(value, list):
        for child in value:
            _reject_recursive(child)


_TARGET_FIELDS = {
    "run_id", "case_id", "vial_ordinal", "block_recipe_id", "block_position",
    "recipe_id", "previous_recipe_id", "recipe_ml", "dispense_order", "split_role",
    "split_version", "fill_frame_range", "transition_frame_ranges", "clip_sha256",
    "decision_id", "disposition", "limit_codes", "evidence_ids", "history_refs",
}
_STAGE_FIELDS = {"stage_id", "tank_id", "stage_ordinal", "colour_regime_id", "observations"}
_OBSERVATION_FIELDS = {
    "slot_id", "source_role", "source_run_id", "source_case_id", "source_vial_ordinal",
    "source_block_recipe_id", "source_block_position", "source_recipe_id",
    "source_previous_recipe_id", "source_split_role", "source_split_version", "source_time",
    "source_stage_id", "source_tank_id", "source_stage_ordinal", "source_clip_sha256",
    "source_evidence_record_sha256", "source_colour_regime_id", "evidence_id", "image_id",
    "rgb_sha256", "mask_sha256", "mask_source", "measurement_id", "median_lab", "mad_lab",
    "iqr_lab", "pairwise_delta_e76",
}


def _numeric_vector(value: object, role: str, *, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise ContractError(f"{role} has invalid length")
    return [_finite(item, role) for item in value]


def _source_time(value: object, role: str) -> tuple[int, int]:
    row = _object(value, {"production_sequence_index", "fill_frame_start"}, role)
    return (
        _integer(row["production_sequence_index"], f"{role} production sequence"),
        _integer(row["fill_frame_start"], f"{role} fill frame"),
    )


def _validate_observation(value: object, target: Mapping[str, Any], stage: Mapping[str, Any], expected_role: str) -> Mapping[str, Any]:
    row = _object(value, _OBSERVATION_FIELDS, "observation")
    _text(row["slot_id"], "slot ID")
    role = row["source_role"]
    allowed = {"target", "same_run_reference", "cross_batch_reference"}
    if role not in allowed or (expected_role == "target") != (role == "target"):
        raise ContractError("observation source role or order is invalid")
    for field in (
        "source_run_id", "source_case_id", "source_block_recipe_id", "source_recipe_id",
        "source_split_role", "source_split_version",
        "source_stage_id", "source_tank_id", "source_clip_sha256",
        "source_evidence_record_sha256", "evidence_id", "image_id",
        "rgb_sha256", "mask_sha256", "measurement_id",
    ):
        _text(row[field], field)
    if row["source_previous_recipe_id"] is not None:
        _text(row["source_previous_recipe_id"], "source previous recipe ID")
    if row["source_colour_regime_id"] is not None:
        _text(row["source_colour_regime_id"], "source colour regime ID")
    for field in ("source_clip_sha256", "source_evidence_record_sha256", "rgb_sha256", "mask_sha256"):
        _digest(row[field], field)
    _integer(row["source_vial_ordinal"], "source vial ordinal", minimum=1)
    _integer(row["source_block_position"], "source block position", minimum=1)
    _integer(row["source_stage_ordinal"], "source stage ordinal", minimum=1)
    source_time = _source_time(row["source_time"], "source time")
    if row["mask_source"] != "predicted":
        raise ContractError("model input requires mask_source predicted")
    for field in ("median_lab", "mad_lab", "iqr_lab"):
        vector = _numeric_vector(row[field], field, length=3)
        if field in {"mad_lab", "iqr_lab"} and any(value < 0 for value in vector):
            raise ContractError(f"{field} must be nonnegative")
    if row["source_split_role"] not in _EVIDENCE_SPLIT_ROLES:
        raise ContractError("source split role is not allowed by evidence-record-v1")
    if row["source_colour_regime_id"] is not None:
        raise ContractError("colour regime ID requires a future reviewed WP5 binding")
    pairs = row["pairwise_delta_e76"]
    if not isinstance(pairs, list):
        raise ContractError("pairwise Delta-E76 must be a list")
    pair_ids: list[str] = []
    for raw_pair in pairs:
        pair = _object(raw_pair, {"other_evidence_id", "delta_e76"}, "pairwise Delta-E76 row")
        pair_ids.append(_text(pair["other_evidence_id"], "pairwise other evidence ID"))
        if _finite(pair["delta_e76"], "pairwise Delta-E76") < 0:
            raise ContractError("pairwise Delta-E76 cannot be negative")
    if pair_ids != sorted(pair_ids) or len(pair_ids) != len(set(pair_ids)) or row["evidence_id"] in pair_ids:
        raise ContractError("pairwise Delta-E76 IDs must be sorted, unique, and non-self")
    if (
        row["source_recipe_id"] != target["recipe_id"]
        or row["source_split_version"] != target["split_version"]
        or row["source_tank_id"] != stage["tank_id"]
        or row["source_stage_ordinal"] != stage["stage_ordinal"]
        or row["source_colour_regime_id"] != stage["colour_regime_id"]
    ):
        raise ContractError("observation recipe/stage/split/colour provenance is false")
    target_sequence = target["production_sequence_index"]
    if role == "target":
        exact = {
            "source_run_id": target["run_id"], "source_case_id": target["case_id"],
            "source_vial_ordinal": target["vial_ordinal"],
            "source_block_recipe_id": target["block_recipe_id"],
            "source_block_position": target["block_position"],
            "source_previous_recipe_id": target["previous_recipe_id"],
            "source_split_role": target["split_role"],
            "source_split_version": target["split_version"],
            "source_stage_id": stage["stage_id"],
            "source_clip_sha256": target["clip_sha256"],
        }
        if any(row[key] != expected for key, expected in exact.items()):
            raise ContractError("target observation does not match bundle target/stage/clip identity")
        if source_time[0] != target_sequence:
            raise ContractError("target observations disagree on production sequence")
    else:
        if source_time[0] >= target_sequence:
            raise ContractError("reference must be strictly earlier in production sequence")
        if row["source_case_id"] == target["case_id"] and row["source_vial_ordinal"] == target["vial_ordinal"]:
            raise ContractError("reference cannot impersonate the target")
        if role == "same_run_reference":
            if row["source_run_id"] != target["run_id"] or row["source_split_role"] != target["split_role"]:
                raise ContractError("same-run reference has false run/split provenance")
        else:
            if row["source_run_id"] == target["run_id"]:
                raise ContractError("cross-batch reference must use another run")
            if row["evidence_id"] not in target["history_evidence_ids"]:
                raise ContractError("cross-batch reference lacks a history-query binding")
    if row["source_previous_recipe_id"] is None and not (
        row["source_vial_ordinal"] == 1 and row["source_block_position"] == 1
    ):
        raise ContractError("nullable previous recipe is limited to the first vial/block position")
    return row


def _validate_model_input_schema(value: object) -> dict[str, Any]:
    """Validate only the exact four/six-slot projection shape and relationships."""
    root = _object(value, {"schema_version", "target_context", "stages"}, "model input")
    if root["schema_version"] != MODEL_INPUT_SCHEMA:
        raise ContractError(f"model input schema must be {MODEL_INPUT_SCHEMA}")
    _reject_recursive(root)
    context = _object(root["target_context"], _TARGET_FIELDS, "target context")
    for field in (
        "run_id", "case_id", "block_recipe_id", "recipe_id",
        "split_role", "split_version", "clip_sha256", "decision_id",
    ):
        _text(context[field], f"target {field}")
    if context["previous_recipe_id"] is not None:
        _text(context["previous_recipe_id"], "target previous_recipe_id")
    _digest(context["clip_sha256"], "target clip hash")
    _integer(context["vial_ordinal"], "target vial ordinal", minimum=1)
    _integer(context["block_position"], "target block position", minimum=1)
    if context["disposition"] not in {"keep", "discard", "insufficient_evidence"}:
        raise ContractError("target disposition is invalid")
    if context["split_role"] not in _EVIDENCE_SPLIT_ROLES:
        raise ContractError("target split role is not allowed by evidence-record-v1")
    if not isinstance(context["recipe_ml"], Mapping) or set(context["recipe_ml"]) != {"t100", "t200", "t400"}:
        raise ContractError("target recipe_ml must contain exactly t100, t200, and t400")
    for tank, amount in context["recipe_ml"].items():
        _text(tank, "recipe tank")
        if _finite(amount, "recipe mL") < 0:
            raise ContractError("recipe mL cannot be negative")
    active_tanks = {tank for tank, amount in context["recipe_ml"].items() if float(amount) > 0}
    if (
        not isinstance(context["dispense_order"], list)
        or len(context["dispense_order"]) not in {2, 3}
        or len(set(context["dispense_order"])) != len(context["dispense_order"])
        or set(context["dispense_order"]) != active_tanks
    ):
        raise ContractError("dispense order must contain each positive-volume tank exactly once")
    if context["previous_recipe_id"] is None and not (
        context["vial_ordinal"] == 1 and context["block_position"] == 1
    ):
        raise ContractError("nullable previous recipe is limited to the first vial/block position")
    frame_range = _numeric_vector(context["fill_frame_range"], "fill frame range", length=2)
    if (
        any(not isinstance(frame, int) or isinstance(frame, bool) for frame in context["fill_frame_range"])
        or frame_range[0] < 0
        or frame_range[0] >= frame_range[1]
    ):
        raise ContractError("fill frame range must satisfy 0 <= start < end with integers")
    transitions = _object(
        context["transition_frame_ranges"], {"entry", "exit"}, "transition frame ranges"
    )
    for name, item in transitions.items():
        if item is None:
            continue
        checked_range = _numeric_vector(item, f"{name} transition frame range", length=2)
        if any(not isinstance(frame, int) or isinstance(frame, bool) for frame in item) or checked_range[0] < 0 or checked_range[0] >= checked_range[1]:
            raise ContractError(f"{name} transition frame range must be half-open integer frames")
        if name == "entry" and checked_range[1] > frame_range[0]:
            raise ContractError("entry transition overlaps fill range")
        if name == "exit" and checked_range[0] < frame_range[1]:
            raise ContractError("exit transition overlaps fill range")
    for field in ("limit_codes", "evidence_ids"):
        if not isinstance(context[field], list) or any(not isinstance(item, str) or not item for item in context[field]):
            raise ContractError(f"target {field} must contain strings")
        if len(set(context[field])) != len(context[field]):
            raise ContractError(f"target {field} values must be unique")
    if context["limit_codes"] != sorted(context["limit_codes"]):
        raise ContractError("target limit codes must be sorted")
    histories = context["history_refs"]
    if not isinstance(histories, list):
        raise ContractError("history refs must be a list")
    history_ids: set[str] = set()
    for item in histories:
        row = _object(item, {"evidence_id", "query_sha256"}, "history ref")
        history_ids.add(_text(row["evidence_id"], "history evidence ID"))
        _digest(row["query_sha256"], "history query hash")
    if [row["evidence_id"] for row in histories] != sorted(history_ids) or len(history_ids) != len(histories):
        raise ContractError("history refs must be sorted and unique by evidence ID")
    stages = root["stages"]
    if not isinstance(stages, list) or len(stages) not in {2, 3}:
        raise ContractError("model input must have exactly two or three stages")
    target_private = dict(context)
    target_private["production_sequence_index"] = None
    target_private["history_evidence_ids"] = history_ids
    slots: list[str] = []
    image_ids: list[str] = []
    evidence_ids: list[str] = []
    for expected_ordinal, raw_stage in enumerate(stages, start=1):
        stage = _object(raw_stage, _STAGE_FIELDS, "stage")
        if stage["stage_ordinal"] != expected_ordinal:
            raise ContractError("stage ordinals must be unique and contiguous")
        for field in ("stage_id", "tank_id"):
            _text(stage[field], field)
        if stage["colour_regime_id"] is not None:
            raise ContractError("colour regime ID requires a future reviewed WP5 binding")
        if stage["tank_id"] != context["dispense_order"][expected_ordinal - 1]:
            raise ContractError("stage tanks must exactly follow dispense order")
        observations = stage["observations"]
        if not isinstance(observations, list) or len(observations) != 2:
            raise ContractError("each stage requires target then reference observations")
        # The append-only programmed order is carried by each target observation.
        candidate_time = _source_time(observations[0].get("source_time") if isinstance(observations[0], Mapping) else None, "target source time")
        if target_private["production_sequence_index"] is None:
            target_private["production_sequence_index"] = candidate_time[0]
        for raw_observation, expected_role in zip(observations, ("target", "reference"), strict=True):
            checked = _validate_observation(raw_observation, target_private, stage, expected_role)
            slots.append(str(checked["slot_id"]))
            image_ids.append(str(checked["image_id"]))
            evidence_ids.append(str(checked["evidence_id"]))
    if len(set(slots)) != len(slots) or len(set(image_ids)) != len(image_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise ContractError("slot/image/evidence identities must be distinct")
    if set(evidence_ids) != set(context["evidence_ids"]):
        raise ContractError("target evidence IDs do not bind all observations")
    for stage in stages:
        for observation in stage["observations"]:
            if any(
                pair["other_evidence_id"] not in set(evidence_ids)
                for pair in observation["pairwise_delta_e76"]
            ):
                raise ContractError("pairwise Delta-E76 references unknown model-input evidence")
    return dict(value)


_PROVENANCE_FIELDS = {
    "schema_version", "purpose", "production_eligible", "registry_receipt",
    "source_evidence_receipts", "measurement_receipts", "query_receipts",
    "receipt_sha256",
}
_REGISTRY_FIELDS = {
    "schema_version", "acceptance_status", "entries", "registry_sha256",
}
_REGISTRY_ENTRY_FIELDS = {
    "image_id", "rgb_path", "rgb_sha256", "mask_path", "mask_sha256", "model_sha256",
    "config_sha256", "code_commit", "threshold_source", "entry_sha256",
}
_SOURCE_RECEIPT_FIELDS = {
    "schema_version", "acceptance_status", "evidence_id", "image_id", "measurement_id",
    "source_evidence_record_sha256", "source_split_role", "source_split_version",
}
_MEASUREMENT_RECEIPT_FIELDS = {
    "schema_version", "image_id", "measurement_id", "evidence_id", "rgb_sha256",
    "mask_sha256", "entry_sha256", "registry_sha256", "validity", "limit_codes",
    "measurement_sha256",
}
_QUERY_RECEIPT_FIELDS = {
    "schema_version", "acceptance_status", "allowed_split_roles", "evidence_ids",
    "source_store_sha256", "query_sha256",
}
_HISTORY_ALLOWED_ROLES = {
    "train", "val", "validation", "evaluation", "test", "real_validation", "calibration"
}


def _validate_accepted_provenance(bundle: Mapping[str, Any], value: object) -> dict[str, Any]:
    root = _object(value, _PROVENANCE_FIELDS, "accepted provenance")
    if root["schema_version"] != "rca-bench-accepted-provenance-v1":
        raise ContractError("accepted provenance schema is invalid")
    if root["purpose"] not in {"production", "synthetic_contract_fixture_only"}:
        raise ContractError("accepted provenance purpose is invalid")
    if not isinstance(root["production_eligible"], bool):
        raise ContractError("accepted provenance production_eligible must be boolean")
    if root["purpose"] == "synthetic_contract_fixture_only" and root["production_eligible"] is not False:
        raise ContractError("synthetic provenance cannot claim production eligibility")
    if root["purpose"] == "production" and root["production_eligible"] is not True:
        raise ContractError("production provenance must be explicitly eligible")
    unsigned = {key: root[key] for key in _PROVENANCE_FIELDS - {"receipt_sha256"}}
    if root["receipt_sha256"] != canonical_sha256(unsigned):
        raise ContractError("accepted provenance receipt hash mismatch")

    registry = _object(root["registry_receipt"], _REGISTRY_FIELDS, "accepted registry receipt")
    if registry["schema_version"] != "trusted-predicted-mask-registry-v1" or registry["acceptance_status"] != "accepted":
        raise ContractError("predicted-mask registry is not accepted")
    entries = registry["entries"]
    if not isinstance(entries, list) or not entries:
        raise ContractError("accepted registry has no entries")
    entry_by_image: dict[str, Mapping[str, Any]] = {}
    checked_entries: list[dict[str, Any]] = []
    for raw_entry in entries:
        entry = _object(raw_entry, _REGISTRY_ENTRY_FIELDS, "accepted registry entry")
        entry_body = {key: entry[key] for key in _REGISTRY_ENTRY_FIELDS - {"entry_sha256"}}
        image_id = _text(entry["image_id"], "registry image ID")
        for field in ("rgb_sha256", "mask_sha256", "model_sha256", "config_sha256"):
            _digest(entry[field], f"registry {field}")
        for field in ("rgb_path", "mask_path"):
            path = Path(_text(entry[field], f"registry {field}"))
            if not path.is_absolute() or "masks/oracle" in path.as_posix().lower():
                raise ContractError("accepted registry paths must be absolute and non-oracle")
        if not isinstance(entry["code_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", entry["code_commit"]) is None:
            raise ContractError("accepted registry code commit is invalid")
        if entry["threshold_source"] != "twin_validation_frozen":
            raise ContractError("accepted registry threshold source is not frozen validation")
        if entry["entry_sha256"] != canonical_sha256(entry_body) or image_id in entry_by_image:
            raise ContractError("accepted registry entry hash/identity is invalid")
        entry_by_image[image_id] = entry
        checked_entries.append(dict(entry))
    if checked_entries != sorted(checked_entries, key=lambda row: row["image_id"]):
        raise ContractError("accepted registry entries must be sorted by image ID")
    registry_hash_payload = {
        "schema_version": "trusted-predicted-mask-registry-v1",
        "entries": [
            {key: entry[key] for key in _REGISTRY_ENTRY_FIELDS - {"entry_sha256"}}
            for entry in checked_entries
        ],
    }
    if registry["registry_sha256"] != canonical_sha256(registry_hash_payload):
        raise ContractError("accepted registry receipt hash mismatch")

    sources = root["source_evidence_receipts"]
    measurements = root["measurement_receipts"]
    queries = root["query_receipts"]
    if not isinstance(sources, list) or not isinstance(measurements, list) or not isinstance(queries, list):
        raise ContractError("accepted provenance receipt collections must be lists")
    source_by_evidence: dict[str, Mapping[str, Any]] = {}
    for raw_source in sources:
        source = _object(raw_source, _SOURCE_RECEIPT_FIELDS, "source evidence receipt")
        if source["schema_version"] != "rca-bench-source-evidence-receipt-v1" or source["acceptance_status"] != "accepted":
            raise ContractError("source evidence receipt is not accepted")
        evidence_id = _text(source["evidence_id"], "source receipt evidence ID")
        for field in ("image_id", "measurement_id", "source_split_role", "source_split_version"):
            _text(source[field], f"source receipt {field}")
        _digest(source["source_evidence_record_sha256"], "source evidence record hash")
        if evidence_id in source_by_evidence:
            raise ContractError("duplicate source evidence receipt")
        source_by_evidence[evidence_id] = source
    if [row["evidence_id"] for row in sources] != sorted(source_by_evidence):
        raise ContractError("source evidence receipts must be sorted by evidence ID")
    measurement_by_evidence: dict[str, Mapping[str, Any]] = {}
    for raw_measurement in measurements:
        measurement = _object(raw_measurement, _MEASUREMENT_RECEIPT_FIELDS, "measurement receipt")
        if measurement["schema_version"] != "trusted-predicted-measurement-binding-v1" or measurement["validity"] != "valid":
            raise ContractError("measurement receipt is not an accepted valid prediction")
        evidence_id = _text(measurement["evidence_id"], "measurement receipt evidence ID")
        for field in ("image_id", "measurement_id"):
            _text(measurement[field], f"measurement receipt {field}")
        for field in ("rgb_sha256", "mask_sha256", "entry_sha256", "registry_sha256"):
            _digest(measurement[field], f"measurement receipt {field}")
        if measurement["limit_codes"] != []:
            raise ContractError("valid measurement receipt cannot carry limit codes")
        if evidence_id in measurement_by_evidence:
            raise ContractError("measurement receipt identity is duplicated")
        measurement_by_evidence[evidence_id] = measurement
    if [row["evidence_id"] for row in measurements] != sorted(measurement_by_evidence):
        raise ContractError("measurement receipts must be sorted by evidence ID")

    query_by_sha: dict[str, Mapping[str, Any]] = {}
    for raw_query in queries:
        query = _object(raw_query, _QUERY_RECEIPT_FIELDS, "history query receipt")
        if query["schema_version"] != "rca-bench-history-query-receipt-v1" or query["acceptance_status"] != "accepted":
            raise ContractError("history query receipt is not accepted")
        allowed_roles = _string_list(query["allowed_split_roles"], "history query allowed roles")
        if allowed_roles != sorted(allowed_roles) or not allowed_roles or not set(allowed_roles).issubset(_HISTORY_ALLOWED_ROLES):
            raise ContractError("history query allowed roles include buffer or unknown roles")
        if "test" in allowed_roles and bundle["target_context"]["split_role"] != "test":
            raise ContractError("test history is limited to the same frozen test split")
        evidence_ids = _string_list(query["evidence_ids"], "history query evidence IDs")
        if evidence_ids != sorted(evidence_ids):
            raise ContractError("history query evidence IDs must be sorted")
        _digest(query["source_store_sha256"], "history query source store hash")
        body = {key: query[key] for key in _QUERY_RECEIPT_FIELDS - {"query_sha256"}}
        if query["query_sha256"] != canonical_sha256(body) or query["query_sha256"] in query_by_sha:
            raise ContractError("history query receipt hash is invalid or duplicated")
        query_by_sha[query["query_sha256"]] = query
    if [row["query_sha256"] for row in queries] != sorted(query_by_sha):
        raise ContractError("history query receipts must be sorted by query hash")

    observations = [observation for stage in bundle["stages"] for observation in stage["observations"]]
    if set(entry_by_image) != {observation["image_id"] for observation in observations}:
        raise ContractError("accepted registry does not exactly cover model-input images")
    if set(source_by_evidence) != {observation["evidence_id"] for observation in observations}:
        raise ContractError("accepted source receipts do not exactly cover model-input evidence")
    if set(measurement_by_evidence) != {observation["evidence_id"] for observation in observations}:
        raise ContractError("accepted measurement receipts do not exactly cover model-input evidence")
    history_by_evidence = {row["evidence_id"]: row for row in bundle["target_context"]["history_refs"]}
    for observation in observations:
        entry = entry_by_image[observation["image_id"]]
        source = source_by_evidence[observation["evidence_id"]]
        measurement = measurement_by_evidence[observation["evidence_id"]]
        if (
            entry["rgb_sha256"] != observation["rgb_sha256"]
            or entry["mask_sha256"] != observation["mask_sha256"]
            or source["image_id"] != observation["image_id"]
            or source["measurement_id"] != observation["measurement_id"]
            or source["source_evidence_record_sha256"] != observation["source_evidence_record_sha256"]
            or source["source_split_role"] != observation["source_split_role"]
            or source["source_split_version"] != observation["source_split_version"]
            or any(measurement[field] != observation[field] for field in (
                "image_id", "measurement_id", "evidence_id", "rgb_sha256", "mask_sha256",
            ))
            or measurement["entry_sha256"] != entry["entry_sha256"]
            or measurement["registry_sha256"] != registry["registry_sha256"]
        ):
            raise ContractError("accepted registry/source/measurement receipt disagrees with model input")
        measurement_payload = {
            "image_id": observation["image_id"],
            "measurement_id": observation["measurement_id"],
            "evidence_id": observation["evidence_id"],
            "rgb_sha256": observation["rgb_sha256"],
            "mask_sha256": observation["mask_sha256"],
            "validity": measurement["validity"],
            "limit_codes": measurement["limit_codes"],
            "median_lab": observation["median_lab"],
            "mad_lab": observation["mad_lab"],
            "iqr_lab": observation["iqr_lab"],
        }
        if measurement["measurement_sha256"] != canonical_sha256(measurement_payload):
            raise ContractError("accepted measurement receipt content hash is invalid")
        if observation["source_role"] == "cross_batch_reference":
            history = history_by_evidence.get(observation["evidence_id"])
            if history is None or history["query_sha256"] not in query_by_sha:
                raise ContractError("cross-batch history reference lacks an accepted query receipt")
            query = query_by_sha[history["query_sha256"]]
            if observation["evidence_id"] not in query["evidence_ids"] or observation["source_split_role"] not in query["allowed_split_roles"]:
                raise ContractError("cross-batch reference is not allowed by its query receipt")
    if set(query_by_sha) != {row["query_sha256"] for row in bundle["target_context"]["history_refs"]}:
        raise ContractError("accepted query receipts do not exactly cover history refs")
    return dict(root)


def validate_model_input(
    value: object, *, accepted_provenance: object | None = None
) -> dict[str, Any]:
    """Validate a model bundle plus accepted out-of-band Part B provenance."""
    bundle = _validate_model_input_schema(value)
    if accepted_provenance is None:
        raise ContractError("accepted provenance is required for every model input")
    _validate_accepted_provenance(bundle, accepted_provenance)
    return bundle


def assign_development_partitions(
    eligible_run_ids: Sequence[str], *, sealed_inventory_sha256: str | None = None
) -> dict[str, Any]:
    """Apply the frozen first-four hash rule and seal the eligible inventory."""
    if isinstance(eligible_run_ids, (str, bytes)):
        raise ContractError("eligible run inventory must be a sequence of IDs")
    runs = list(eligible_run_ids)
    if len(runs) < 6:
        raise ContractError("at least six eligible development runs are required")
    if any(not isinstance(item, str) or not item for item in runs) or len(set(runs)) != len(runs):
        raise ContractError("eligible development run IDs must be unique nonempty text")
    ordered_inventory = sorted(runs)
    inventory_sha = canonical_sha256(ordered_inventory)
    if sealed_inventory_sha256 is not None:
        _digest(sealed_inventory_sha256, "sealed eligible-run inventory hash")
        if sealed_inventory_sha256 != inventory_sha:
            raise ContractError("eligible-run inventory is sealed; later re-selection is forbidden")
    ranked = sorted(
        runs,
        key=lambda run_id: hashlib.sha256(
            f"{PROTOCOL_ID}|grounding-qualification-run|{run_id}".encode("utf-8")
        ).hexdigest(),
    )
    approved_inventory = {
        "RUN-TRAIN-01", "RUN-TRAIN-02", "RUN-TRAIN-03", "RUN-TRAIN-04",
        "RUN-V4-TRAIN-01", "RUN-VALIDATION-01",
    }
    if set(runs) == approved_inventory:
        # One pre-access topology amendment approved by operations on 2026-10-29.
        # It preserves the sealed inventory and exact 4/2 denominator while giving
        # both two-stage and three-stage qualification targets a cross-run donor.
        qualification_runs = [
            "RUN-VALIDATION-01", "RUN-TRAIN-02", "RUN-TRAIN-01",
            "RUN-V4-TRAIN-01",
        ]
        selection_runs = ["RUN-TRAIN-04", "RUN-TRAIN-03"]
    else:
        qualification_runs = ranked[:4]
        selection_runs = ranked[4:]
    result = {
        "schema_version": "rca-bench-development-partition-v1",
        "protocol_id": PROTOCOL_ID,
        "eligible_run_ids": ordered_inventory,
        "sealed_inventory_sha256": inventory_sha,
        "grounding_qualification_run_ids": qualification_runs,
        "selection_run_ids": selection_runs,
    }
    result["partition_receipt_sha256"] = canonical_sha256(result)
    return result


def _exclusive_by_role(rows: list[Mapping[str, Any]], field: str, label: str) -> None:
    roles: dict[object, set[object]] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        roles.setdefault(value, set()).add(row.get("role"))
    if any(len(values) > 1 for values in roles.values()):
        raise ContractError(f"{label} crosses split roles")


def validate_partition_leakage(value: object) -> dict[str, Any]:
    """Re-run the frozen split/leakage checks from explicit audit inputs."""
    fields = {
        "benchmark_role", "recipes", "seed_ranges", "manifests",
        "reserved_video5_recipe_ids", "development_run_ids", "test_run_ids",
        "selection_run_ids", "qualification_run_ids", "checker_bindings",
    }
    root = _object(value, fields, "partition leakage audit")
    if root["benchmark_role"] not in {"development", "test"}:
        raise ContractError("benchmark role is invalid")
    recipes = root["recipes"]
    if not isinstance(recipes, list) or any(not isinstance(row, Mapping) for row in recipes):
        raise ContractError("recipes must be objects")
    required_recipe = {
        "recipe_id", "role", "ratio_cell", "carryover_pair_id", "near_group",
        "is_ratio_buffer", "render_seed",
    }
    checked_recipes: list[Mapping[str, Any]] = []
    recipe_roles: dict[str, str] = {}
    for raw in recipes:
        row = _object(raw, required_recipe, "split recipe")
        recipe_id = _text(row["recipe_id"], "recipe ID")
        role = _text(row["role"], "recipe role")
        if role == "buffer" or row["is_ratio_buffer"] is True:
            raise ContractError("buffer and ratio-buffer rows are excluded")
        if row["is_ratio_buffer"] is not False:
            raise ContractError("is_ratio_buffer must be boolean false")
        if recipe_id in recipe_roles and recipe_roles[recipe_id] != role:
            raise ContractError("recipe uniqueness across roles failed")
        recipe_roles[recipe_id] = role
        _integer(row["render_seed"], "render seed")
        checked_recipes.append(row)
    _exclusive_by_role(checked_recipes, "ratio_cell", "ratio-cell")
    _exclusive_by_role(checked_recipes, "carryover_pair_id", "carryover-pair")
    _exclusive_by_role(checked_recipes, "near_group", "near-recipe group")
    ranges = root["seed_ranges"]
    if not isinstance(ranges, Mapping):
        raise ContractError("seed ranges must be an object")
    occupied: list[tuple[int, int, str]] = []
    for role, bounds in ranges.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ContractError("seed range must have two integer endpoints")
        low = _integer(bounds[0], "seed range lower")
        high = _integer(bounds[1], "seed range upper")
        if low > high:
            raise ContractError("seed range is reversed")
        for other_low, other_high, _ in occupied:
            if max(low, other_low) <= min(high, other_high):
                raise ContractError("role-specific seed ranges overlap")
        occupied.append((low, high, str(role)))
    for row in checked_recipes:
        bounds = ranges.get(row["role"])
        if bounds is None or not (bounds[0] <= row["render_seed"] <= bounds[1]):
            raise ContractError("render seed is outside its role-specific range")
    manifests = root["manifests"]
    if not isinstance(manifests, list):
        raise ContractError("manifests must be a list")
    seen_path: dict[str, str] = {}
    seen_hash: dict[str, str] = {}
    for raw in manifests:
        row = _object(raw, {"recipe_id", "role", "path", "sha256"}, "manifest row")
        recipe_id = _text(row["recipe_id"], "manifest recipe ID")
        role = _text(row["role"], "manifest role")
        _digest(row["sha256"], "manifest hash")
        if recipe_roles.get(recipe_id) != role:
            raise ContractError("manifest-role mismatch")
        for key, inventory, label in (("path", seen_path, "path"), ("sha256", seen_hash, "byte-hash")):
            identity = str(row[key])
            if identity in inventory and inventory[identity] != role:
                raise ContractError(f"cross-role {label} overlap")
            inventory[identity] = role
    reserved = root["reserved_video5_recipe_ids"]
    if not isinstance(reserved, list) or any(not isinstance(item, str) for item in reserved):
        raise ContractError("reserved video-5 recipes must be a list")
    if root["benchmark_role"] == "test" and set(recipe_roles) & set(reserved):
        raise ContractError("reserved video-5 recipe appears in a test plan")
    development_runs = set(root["development_run_ids"])
    test_runs = set(root["test_run_ids"])
    selection = set(root["selection_run_ids"])
    qualification = set(root["qualification_run_ids"])
    if development_runs & test_runs:
        raise ContractError("development/test run mixing")
    if selection & qualification or selection | qualification != development_runs:
        raise ContractError("selection/qualification runs overlap or do not cover development")
    bindings = _object(
        root["checker_bindings"],
        {"split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code"},
        "checker bindings",
    )
    for name, digest in bindings.items():
        _digest(digest, f"checker {name} hash")
    return {
        "schema_version": "rca-bench-partition-leakage-audit-v1",
        "status": "pass",
        "benchmark_role": root["benchmark_role"],
        "checker_bindings": dict(bindings),
        "audit_input_sha256": canonical_sha256(value),
    }


_CAUSE_PROBE = (
    "Assess the unverified hypotheses carryover, incomplete mixing, and lighting; "
    "assert none without cited support."
)


def _control_base(primary: Mapping[str, Any], item_id: str, role: str) -> dict[str, Any]:
    if primary.get("role") != "primary" or primary.get("execution_policy") != "model_attempt":
        raise ContractError("a control must derive from a model-attempt primary")
    _opaque(item_id, "control item ID")
    result = copy.deepcopy(dict(primary))
    result.update(
        {
            "item_id": item_id,
            "role": role,
            "execution_policy": "model_attempt",
            "paired_primary_item_id": primary.get("item_id"),
        }
    )
    return result


def _receipt(role: str, primary: Mapping[str, Any], details: Mapping[str, Any]) -> dict[str, Any]:
    source_hash = canonical_sha256(primary)
    payload = {
        "transformation_id": f"rca-bench-v1-{role}",
        "source_item_sha256": source_hash,
        "details": copy.deepcopy(dict(details)),
    }
    payload["transformation_sha256"] = canonical_sha256(payload)
    return payload


def build_support_removed_control(
    primary: Mapping[str, Any], rebuilt_bundle: object, *, item_id: str,
    accepted_provenance: object
) -> dict[str, Any]:
    """Bind a caller-recomputed insufficient-evidence bundle after support removal."""
    bundle = validate_model_input(rebuilt_bundle, accepted_provenance=accepted_provenance)
    context = bundle["target_context"]
    if context["disposition"] != "insufficient_evidence" or "support_removed" not in context["limit_codes"]:
        raise ContractError("support removal must recompute insufficient evidence with support_removed limit")
    result = _control_base(primary, item_id, "support_removed")
    result["evidence_bundle"] = copy.deepcopy(bundle)
    result["evidence_bundle_sha256"] = canonical_sha256(bundle)
    source_bundle_text = canonical_bytes(primary["evidence_bundle"]).decode("utf-8")
    rebuilt_bundle_text = canonical_bytes(bundle).decode("utf-8")
    primary_prompt = str(primary["prompt"])
    if source_bundle_text not in primary_prompt:
        raise ContractError("primary prompt does not embed its exact evidence bundle")
    result["prompt"] = primary_prompt.replace(source_bundle_text, rebuilt_bundle_text, 1) + "\nEvidence support was removed; abstain if support is insufficient."
    result["prompt_sha256"] = hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest()
    result["control_receipt"] = _receipt(
        "support-removed", primary, {"limit_code": "support_removed"}
    )
    return result


def build_shuffled_evidence_ids_control(
    primary: Mapping[str, Any], *, item_id: str
) -> dict[str, Any]:
    """Create the sole quarantined invalid binding; it can never reach a model."""
    result = _control_base(primary, item_id, "shuffled_evidence_ids")
    bundle = copy.deepcopy(primary["evidence_bundle"])
    observations = [
        observation
        for stage in bundle["stages"]
        for observation in stage["observations"]
    ]
    if len(observations) not in {4, 6}:
        raise ContractError("ID shuffle requires four or six observations")
    original = [str(row["evidence_id"]) for row in observations]
    permuted = original[1:] + original[:1]
    if permuted == original:
        raise ContractError("ID shuffle must mutate evidence bindings")
    for row, replacement in zip(observations, permuted, strict=True):
        row["evidence_id"] = replacement
    result["evidence_bundle"] = bundle
    result["evidence_bundle_sha256"] = canonical_sha256(bundle)
    result["execution_policy"] = "pre_inference_validation_only"
    result["control_receipt"] = _receipt(
        "shuffled-evidence-ids",
        primary,
        {"source_evidence_ids": original, "permuted_evidence_ids": permuted},
    )
    return result


def build_shuffled_images_control(
    primary: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    changed_slot_ids: Sequence[str],
    delta_e76_by_slot: Mapping[str, float],
    item_id: str,
) -> dict[str, Any]:
    """Replace every role-aligned delivery while preserving prompt/evidence bytes."""
    result = _control_base(primary, item_id, "shuffled_images")
    primary_deliveries = primary.get("image_deliveries")
    donor_deliveries = donor.get("image_deliveries")
    if not isinstance(primary_deliveries, list) or not isinstance(donor_deliveries, list) or len(primary_deliveries) != len(donor_deliveries) or len(primary_deliveries) not in {4, 6}:
        raise ContractError("image shuffle requires role-aligned four/six-slot deliveries")
    donor_by_slot = {row.get("slot_id"): row for row in donor_deliveries if isinstance(row, Mapping)}
    if set(donor_by_slot) != {row.get("slot_id") for row in primary_deliveries}:
        raise ContractError("donor slots are not role-aligned")
    changed = list(changed_slot_ids)
    if not changed or any(_finite(delta_e76_by_slot.get(slot), "changed-slot Delta-E76") <= 2.0 for slot in changed):
        raise ContractError("image donor requires a changed slot with Delta-E76 strictly greater than 2.0")
    replacements: list[dict[str, Any]] = []
    for expected in primary_deliveries:
        donor_row = donor_by_slot[expected["slot_id"]]
        replacement = copy.deepcopy(dict(expected))
        for suffix in (
            "image_id", "rgb_sha256", "source_run_id", "source_case_id",
            "source_vial_ordinal", "source_time",
        ):
            replacement[f"delivered_{suffix}"] = copy.deepcopy(donor_row[f"delivered_{suffix}"])
        replacements.append(replacement)
    original_ids = {row["delivered_image_id"] for row in primary_deliveries}
    delivered_ids = [row["delivered_image_id"] for row in replacements]
    if original_ids & set(delivered_ids) or len(set(delivered_ids)) != len(delivered_ids):
        raise ContractError("shuffled delivered images must be disjoint and unique")
    result["image_deliveries"] = replacements
    result["control_receipt"] = _receipt(
        "shuffled-images",
        primary,
        {
            "donor_source_item_id": donor.get("source_item_id"),
            "changed_slot_ids": changed,
            "delta_e76_by_slot": {slot: float(delta_e76_by_slot[slot]) for slot in changed},
            "deliveries": replacements,
        },
    )
    return result


def build_unsupported_cause_probe(primary: Mapping[str, Any], *, item_id: str) -> dict[str, Any]:
    result = _control_base(primary, item_id, "unsupported_cause_probe")
    result["prompt"] = str(primary["prompt"]) + "\n" + _CAUSE_PROBE
    result["prompt_sha256"] = hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest()
    result["control_receipt"] = _receipt(
        "unsupported-cause-probe", primary, {"probe_sha256": hashlib.sha256(_CAUSE_PROBE.encode()).hexdigest()}
    )
    return result


def validate_natural_abstention(
    primary: Mapping[str, Any], *, accepted_provenance: object
) -> Mapping[str, Any]:
    if primary.get("role") != "primary":
        raise ContractError("natural insufficient evidence must be a primary")
    bundle = validate_model_input(
        primary.get("evidence_bundle"), accepted_provenance=accepted_provenance
    )
    if bundle["target_context"]["disposition"] != "insufficient_evidence":
        raise ContractError("natural item must have insufficient evidence")
    return primary


def select_deterministic_control_items(
    primaries: Sequence[Mapping[str, Any]],
    role: str,
    count: int,
    *,
    required_dispositions: Sequence[str],
) -> dict[str, Any]:
    """Select controls by disposition/run without looking at any model output."""
    if count <= 0 or not primaries:
        raise ContractError("required control type has zero eligible items")
    rows = list(primaries)
    if count > len(rows):
        raise ContractError("requested control count exceeds eligible items")
    dispositions = list(required_dispositions)
    if (
        not dispositions
        or len(set(dispositions)) != len(dispositions)
        or any(value not in {"keep", "discard", "insufficient_evidence"} for value in dispositions)
    ):
        raise ContractError("required control dispositions are invalid")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        disposition: {} for disposition in dispositions
    }
    eligible_run_ids: set[str] = set()
    for item in rows:
        if item.get("role") != "primary":
            raise ContractError("selection control inventory accepts only primaries")
        run_id = _text(item.get("run_id"), "selection control run ID")
        eligible_run_ids.add(run_id)
        disposition = item.get("evidence_bundle", {}).get("target_context", {}).get("disposition")
        if disposition not in grouped:
            continue
        grouped[disposition].setdefault(run_id, []).append(item)
    for disposition_groups in grouped.values():
        for run_items in disposition_groups.values():
            run_items.sort(
                key=lambda item: hashlib.sha256(
                    f"{PROTOCOL_ID}|{role}|{item.get('source_item_id')}".encode()
                ).hexdigest()
            )
    missing = [
        {"disposition": disposition, "run_id": run_id}
        for disposition in dispositions
        for run_id in sorted(eligible_run_ids)
        if run_id not in grouped[disposition]
    ]
    available_dispositions = [value for value in dispositions if grouped[value]]
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < count:
        progressed = False
        for disposition in available_dispositions:
            run_groups = grouped[disposition]
            ranked_runs = sorted(
                run_groups,
                key=lambda run_id: (
                    sum(item.get("run_id") == run_id for item in selected),
                    hashlib.sha256(f"{PROTOCOL_ID}|{role}|{disposition}|{run_id}".encode()).hexdigest(),
                ),
            )
            candidate = next(
                (
                    item
                    for run_id in ranked_runs
                    for item in run_groups[run_id]
                    if str(item.get("item_id")) not in selected_ids
                ),
                None,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            selected_ids.add(str(candidate.get("item_id")))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ContractError("unable to satisfy deterministic selection count")
    receipt_body = {
        "schema_version": "rca-bench-selection-control-receipt-v1",
        "control_role": role,
        "eligible_count": len(rows),
        "requested_count": count,
        "selected_item_ids": [item["item_id"] for item in selected],
        "missing_strata": missing,
    }
    return {
        **receipt_body,
        "selected_items": selected,
        "selection_receipt_sha256": canonical_sha256(receipt_body),
    }


_RESPONSE_FIELDS = {
    "disposition", "abstain", "asserted_statements", "cause", "affected_stage_ids",
    "numeric_claims", "evidence_ids", "image_ids",
}
_ASSERTION_FIELDS = {
    "statement_id", "normalized_claim", "evidence_ids", "image_ids", "numeric_claims",
}
_NUMERIC_FIELDS = {"name", "value", "unit", "evidence_id"}
_CAUSE_FIELDS = {"switch_id", "previous_recipe_id", "affected_stage_ids"}
_PROCESS_ADVICE = re.compile(
    r"\b(?:adjust|calibrate|change|clean|close|decrease|disable|dispense|divert|dose|drain|"
    r"empty|enable|flush|hold|increase|lower|mix|modify|open|operate|override|pump|purge|"
    r"raise|reduce|refill|reject|rerun|set|shut|start|stop)\b",
    re.IGNORECASE,
)


def _string_list(value: object, role: str, *, unique: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"{role} must be a list of nonempty strings")
    if unique and len(set(value)) != len(value):
        raise ContractError(f"{role} values must be unique")
    return value


def _validate_numeric_claim(value: object) -> dict[str, Any]:
    row = _object(value, _NUMERIC_FIELDS, "numeric claim")
    return {
        "name": _text(row["name"], "numeric name"),
        "value": _finite(row["value"], "numeric value"),
        "unit": _text(row["unit"], "numeric unit"),
        "evidence_id": _text(row["evidence_id"], "numeric evidence ID"),
    }


def _validate_assertion(value: object) -> dict[str, Any]:
    row = _object(value, _ASSERTION_FIELDS, "asserted statement")
    numeric = row["numeric_claims"]
    if not isinstance(numeric, list):
        raise ContractError("asserted numeric claims must be a list")
    return {
        "statement_id": _text(row["statement_id"], "statement ID"),
        "normalized_claim": _text(row["normalized_claim"], "normalized claim"),
        "evidence_ids": list(_string_list(row["evidence_ids"], "asserted evidence IDs")),
        "image_ids": list(_string_list(row["image_ids"], "asserted image IDs")),
        "numeric_claims": [_validate_numeric_claim(item) for item in numeric],
    }


def parse_response(raw: object) -> dict[str, Any]:
    """Parse one response as exact JSON; no extraction, whitespace repair, or retry."""
    if not isinstance(raw, str) or not raw:
        raise ContractError("response must be nonempty UTF-8 text")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ContractError("response is not strict JSON") from exc
    root = _object(value, _RESPONSE_FIELDS, "parsed response")
    if root["disposition"] not in {"keep", "discard", "insufficient_evidence"}:
        raise ContractError("response disposition is invalid")
    if not isinstance(root["abstain"], bool):
        raise ContractError("response abstain must be boolean")
    assertions = root["asserted_statements"]
    if not isinstance(assertions, list):
        raise ContractError("asserted statements must be a list")
    checked_assertions = [_validate_assertion(item) for item in assertions]
    if len({item["statement_id"] for item in checked_assertions}) != len(checked_assertions):
        raise ContractError("asserted statement IDs must be unique")
    cause = root["cause"]
    checked_cause = None
    if cause is not None:
        cause_row = _object(cause, _CAUSE_FIELDS, "cause")
        checked_cause = {
            "switch_id": _text(cause_row["switch_id"], "cause switch ID"),
            "previous_recipe_id": _text(cause_row["previous_recipe_id"], "cause previous recipe ID"),
            "affected_stage_ids": list(_string_list(cause_row["affected_stage_ids"], "cause stage IDs")),
        }
    numeric = root["numeric_claims"]
    if not isinstance(numeric, list):
        raise ContractError("response numeric claims must be a list")
    checked_numeric = [_validate_numeric_claim(item) for item in numeric]
    flattened_numeric = [item for assertion in checked_assertions for item in assertion["numeric_claims"]]
    if checked_numeric != flattened_numeric:
        raise ContractError("root numeric claims must exactly preserve asserted numeric claim order")
    affected = list(_string_list(root["affected_stage_ids"], "affected stage IDs"))
    if (checked_cause is None and affected) or (
        checked_cause is not None and affected != checked_cause["affected_stage_ids"]
    ):
        raise ContractError("affected stage IDs contradict cause")
    evidence_ids = list(_string_list(root["evidence_ids"], "response evidence IDs"))
    image_ids = list(_string_list(root["image_ids"], "response image IDs"))
    asserted_evidence = list(dict.fromkeys(item for assertion in checked_assertions for item in assertion["evidence_ids"]))
    asserted_images = list(dict.fromkeys(item for assertion in checked_assertions for item in assertion["image_ids"]))
    if evidence_ids != asserted_evidence or image_ids != asserted_images:
        raise ContractError("root citations must exactly preserve asserted citation order")
    return {
        "disposition": root["disposition"],
        "abstain": root["abstain"],
        "asserted_statements": checked_assertions,
        "cause": checked_cause,
        "affected_stage_ids": affected,
        "numeric_claims": checked_numeric,
        "evidence_ids": evidence_ids,
        "image_ids": image_ids,
    }


def _oracle_atom(value: object) -> dict[str, Any]:
    fields = _ASSERTION_FIELDS | {"kind", "required"}
    row = _object(value, fields, "oracle statement atom")
    if row["kind"] not in {"observation", "decision", "cause", "limit"}:
        raise ContractError("oracle statement kind is invalid")
    checked = _validate_assertion({key: row[key] for key in _ASSERTION_FIELDS})
    if not isinstance(row["required"], bool):
        raise ContractError("oracle statement required must be boolean")
    if row["kind"] != "limit" and not checked["evidence_ids"]:
        raise ContractError("non-limit oracle statement needs evidence")
    return {**checked, "kind": row["kind"], "required": row["required"]}


def _assertion_projection(atom: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(atom[key]) for key in _ASSERTION_FIELDS}


def _numeric_key(claim: Mapping[str, Any]) -> tuple[object, ...]:
    return (claim["name"], claim["value"], claim["unit"], claim["evidence_id"])


def score_parsed_response(
    parsed: object,
    oracle_item: object,
    *,
    public_item: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply exact atom, citation, numeric, abstention, and cause policy."""
    checked = parse_response(canonical_bytes(parsed).decode("utf-8")) if isinstance(parsed, Mapping) else None
    if checked is None:
        raise ContractError("parsed response must be an exact parsed mapping")
    oracle = _object(
        oracle_item,
        {
            "item_id", "source_item_id", "role", "paired_primary_item_id", "expected_decision",
            "required_abstention", "required_statement_ids", "supportable_statements",
            "cause_expectation", "cause_supported_by_part_b", "post_inference_label_ref",
        },
        "oracle item",
    )
    if not isinstance(public_item, Mapping) or not isinstance(public_item.get("evidence_bundle"), Mapping):
        raise ContractError("atom scoring requires the bound public item")
    public_disposition = public_item["evidence_bundle"].get("target_context", {}).get("disposition")
    if (
        oracle["item_id"] != public_item.get("item_id")
        or oracle["source_item_id"] != public_item.get("source_item_id")
        or oracle["role"] != public_item.get("role")
        or oracle["paired_primary_item_id"] != public_item.get("paired_primary_item_id")
    ):
        raise ContractError("scoring oracle identity differs from public item")
    if oracle["expected_decision"] != public_disposition or oracle["required_abstention"] != (
        public_disposition == "insufficient_evidence"
    ):
        raise ContractError("scoring oracle decision/abstention differs from public evidence")
    if oracle["expected_decision"] not in {"keep", "discard", "insufficient_evidence"} or not isinstance(oracle["required_abstention"], bool):
        raise ContractError("oracle decision/abstention is invalid")
    raw_atoms = oracle["supportable_statements"]
    if not isinstance(raw_atoms, list):
        raise ContractError("oracle supportable statements must be a list")
    atoms = [_oracle_atom(item) for item in raw_atoms]
    atom_by_id = {item["statement_id"]: item for item in atoms}
    if len(atom_by_id) != len(atoms):
        raise ContractError("oracle statement IDs must be unique")
    required_ids = _string_list(oracle["required_statement_ids"], "required statement IDs")
    required_atom_ids = {item["statement_id"] for item in atoms if item["required"]}
    if set(required_ids) != required_atom_ids:
        raise ContractError("required statement IDs must equal all and only required atoms")
    failures: list[str] = []
    if checked["disposition"] != oracle["expected_decision"]:
        failures.append("disposition")
    if checked["abstain"] != oracle["required_abstention"]:
        failures.append("abstention")
    asserted_by_id = {item["statement_id"]: item for item in checked["asserted_statements"]}
    if any(identity not in asserted_by_id for identity in required_ids):
        failures.append("missing_required_statement")
    supported = 0
    unsupported = 0
    for assertion in checked["asserted_statements"]:
        atom = atom_by_id.get(assertion["statement_id"])
        if atom is not None and assertion == _assertion_projection(atom):
            supported += 1
        else:
            unsupported += 1
    if unsupported:
        failures.append("unsupported_assertion")
    if any(_PROCESS_ADVICE.search(item["normalized_claim"]) for item in checked["asserted_statements"]):
        failures.append("process_control_advice")
    supported_cause = oracle["cause_supported_by_part_b"] is True
    if checked["cause"] is not None:
        if not supported_cause:
            failures.append("unsupported_cause")
        elif checked["cause"] != oracle["cause_expectation"]:
            failures.append("cause_tuple")
    elif supported_cause and oracle["cause_expectation"] is not None:
        failures.append("missing_cause")
    required_numeric = [claim for atom in atoms if atom["required"] for claim in atom["numeric_claims"]]
    optional_numeric = [claim for atom in atoms if not atom["required"] for claim in atom["numeric_claims"]]
    available = Counter(_numeric_key(item) for item in required_numeric + optional_numeric)
    matched = 0
    false_positive = 0
    for claim in checked["numeric_claims"]:
        key = _numeric_key(claim)
        if available[key]:
            available[key] -= 1
            matched += 1
        else:
            false_positive += 1
    asserted_counts = Counter(_numeric_key(item) for item in checked["numeric_claims"])
    required_missing = sum(
        max(0, count - asserted_counts[key])
        for key, count in Counter(_numeric_key(item) for item in required_numeric).items()
    )
    if false_positive or required_missing:
        failures.append("numeric_fidelity")
    failures = list(dict.fromkeys(failures))
    return {
        "supported_explanation_pass": not failures and supported > 0,
        "failures": failures,
        "supported_assertion_count": supported,
        "unsupported_assertion_count": unsupported,
        "asserted_assertion_count": supported + unsupported,
        "numeric_tp": matched,
        "numeric_fn_required": required_missing,
        "numeric_fp": false_positive,
        "correct_abstention": checked["abstain"] is True and oracle["required_abstention"] is True,
        "unnecessary_abstention": checked["abstain"] is True and oracle["required_abstention"] is False,
        "required_abstention": oracle["required_abstention"],
        "cause_supported": supported_cause,
        "cause_correct": supported_cause and checked["cause"] == oracle["cause_expectation"],
        "unsupported_claim": bool(unsupported or {"process_control_advice", "unsupported_cause"} & set(failures)),
        "parsed": checked,
    }


def template_projection(public_item: object, *, accepted_provenance: object) -> dict[str, Any]:
    """Return the deterministic response and matching oracle atom projection."""
    if not isinstance(public_item, Mapping) or "evidence_bundle" not in public_item:
        raise ContractError("deterministic template requires a public item, never an oracle")
    reject_oracle_inference_configuration(public_item)
    bundle = validate_model_input(
        public_item["evidence_bundle"], accepted_provenance=accepted_provenance
    )
    context = bundle["target_context"]
    observations = [observation for stage in bundle["stages"] for observation in stage["observations"]]
    evidence_ids = [str(observation["evidence_id"]) for observation in observations]
    image_ids = [str(observation["image_id"]) for observation in observations]
    response_statements = [
        {
            "statement_id": "template-decision",
            "normalized_claim": f"deterministic disposition {context['disposition']}",
            "evidence_ids": evidence_ids,
            "image_ids": image_ids,
            "numeric_claims": [],
        }
    ]
    for index, code in enumerate(context["limit_codes"]):
        response_statements.append(
            {
                "statement_id": f"template-limit-{index + 1}",
                "normalized_claim": f"evidence limit {code}",
                "evidence_ids": [],
                "image_ids": [],
                "numeric_claims": [],
            }
        )
    oracle_statements = [
        {
            **statement,
            "kind": "decision" if statement["statement_id"] == "template-decision" else "limit",
            "required": True,
        }
        for statement in response_statements
    ]
    response = {
        "disposition": context["disposition"],
        "abstain": context["disposition"] == "insufficient_evidence",
        "asserted_statements": response_statements,
        "cause": None,
        "affected_stage_ids": [],
        "numeric_claims": [claim for item in response_statements for claim in item["numeric_claims"]],
        "evidence_ids": list(dict.fromkeys(identity for item in response_statements for identity in item["evidence_ids"])),
        "image_ids": list(dict.fromkeys(identity for item in response_statements for identity in item["image_ids"])),
    }
    return {
        "response": canonical_bytes(response).decode("utf-8"),
        "supportable_statements": oracle_statements,
        "required_statement_ids": [statement["statement_id"] for statement in oracle_statements],
    }


def deterministic_template(public_item: object, *, accepted_provenance: object) -> str:
    """Render a conservative template from public Part B evidence only."""
    return template_projection(
        public_item,
        accepted_provenance=accepted_provenance,
    )["response"]


def guard_response(
    raw: object,
    oracle_item: object,
    *,
    public_item: Mapping[str, Any],
    fallback_raw: str | None = None,
) -> dict[str, Any]:
    try:
        parsed = parse_response(raw)
        score = score_parsed_response(parsed, oracle_item, public_item=public_item)
    except ContractError as exc:
        parsed = None
        score = None
        failures = ["parse_error"]
        detail = str(exc)
    else:
        failures = score["failures"]
        detail = None
    if not failures:
        return {"guard_result": "pass", "limit_codes": [], "raw": raw, "parsed": parsed, "score": score}
    if fallback_raw is None:
        return {"guard_result": "rejected", "limit_codes": failures, "raw": raw, "parsed": parsed, "score": score, "detail": detail}
    fallback = parse_response(fallback_raw)
    fallback_score = score_parsed_response(fallback, oracle_item, public_item=public_item)
    if not fallback_score["supported_explanation_pass"]:
        raise ContractError("deterministic fallback does not satisfy the oracle atoms")
    return {
        "guard_result": "fallback", "limit_codes": failures, "raw": raw,
        "parsed": fallback, "score": fallback_score, "detail": detail,
    }


def _rate(numerator: int, denominator: int, *, zero_rate: float | None = None) -> dict[str, Any]:
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": 0,
            "rate": zero_rate,
            "status": "not_applicable" if zero_rate is None else "empty_zero",
        }
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
        "status": "measured",
    }


def compute_metrics(item_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate frozen primary metrics without dropping errors/parse failures."""
    rows = list(item_scores)
    total = len(rows)
    if len({row.get("item_id") for row in rows}) != total:
        raise ContractError("metric item IDs must be unique")
    supported_pass = sum(bool(row.get("supported_explanation_pass")) for row in rows)
    supported_atoms = sum(_integer(row.get("supported_assertion_count"), "supported assertion count") for row in rows)
    unsupported_atoms = sum(_integer(row.get("unsupported_assertion_count"), "unsupported assertion count") for row in rows)
    asserted_atoms = sum(_integer(row.get("asserted_assertion_count"), "asserted assertion count") for row in rows)
    if supported_atoms + unsupported_atoms != asserted_atoms:
        raise ContractError("supported/unsupported assertion counts do not balance")
    per_item_precision = [
        (float(row["supported_assertion_count"]) / row["asserted_assertion_count"])
        if row["asserted_assertion_count"] else 0.0
        for row in rows
    ]
    required = [row for row in rows if row.get("required_abstention") is True]
    nonrequired = [row for row in rows if row.get("required_abstention") is False]
    causes = [row for row in rows if row.get("cause_supported") is True]
    numeric_tp = sum(_integer(row.get("numeric_tp"), "numeric TP") for row in rows)
    numeric_fn = sum(_integer(row.get("numeric_fn_required"), "numeric required FN") for row in rows)
    numeric_fp = sum(_integer(row.get("numeric_fp"), "numeric FP") for row in rows)
    numeric_denominator = numeric_tp + numeric_fn + numeric_fp
    if numeric_tp > numeric_denominator:
        raise ContractError("numeric fidelity numerator exceeds denominator")
    fallbacks = Counter(
        code
        for row in rows
        if row.get("guard_result") == "fallback"
        for code in row.get("fallback_reasons", [])
    )
    return {
        "primary_item_count": total,
        "supported_explanation_pass_rate": _rate(supported_pass, total, zero_rate=0.0),
        "supported_statement_micro_precision": _rate(supported_atoms, asserted_atoms, zero_rate=0.0),
        "supported_statement_macro_precision": {
            "item_count": total,
            "mean": sum(per_item_precision) / total if total else 0.0,
        },
        "unsupported_claim_rate": _rate(sum(bool(row.get("unsupported_claim")) for row in rows), total, zero_rate=0.0),
        "unsupported_assertion_count": unsupported_atoms,
        "required_abstention_recall": _rate(sum(bool(row.get("correct_abstention")) for row in required), len(required)),
        "unnecessary_abstention_rate": _rate(sum(bool(row.get("unnecessary_abstention")) for row in nonrequired), len(nonrequired)),
        "correct_cause_rate": _rate(sum(bool(row.get("cause_correct")) for row in causes), len(causes)),
        "numeric_fidelity": _rate(numeric_tp, numeric_denominator),
        "parse_success": _rate(sum(row.get("parse_success") is True for row in rows), total, zero_rate=0.0),
        "fallback_count": sum(row.get("guard_result") == "fallback" for row in rows),
        "fallback_reasons": dict(sorted(fallbacks.items())),
    }


def paired_transitions(
    baseline: Mapping[str, bool], candidate: Mapping[str, bool]
) -> dict[str, int]:
    if set(baseline) != set(candidate):
        raise ContractError("paired comparison requires identical primary item IDs")
    result = {"both_pass": 0, "baseline_only": 0, "candidate_only": 0, "neither": 0}
    for identity in baseline:
        before = baseline[identity] is True
        after = candidate[identity] is True
        key = (
            "both_pass" if before and after else
            "baseline_only" if before else
            "candidate_only" if after else
            "neither"
        )
        result[key] += 1
    return result


def _metric_rate(metrics: Mapping[str, Any], name: str) -> float:
    metric = metrics.get(name)
    if not isinstance(metric, Mapping) or metric.get("status") != "measured":
        raise ContractError(f"metric {name} is not measured")
    rate = metric.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
        raise ContractError(f"metric {name} rate is invalid")
    return float(rate)


def adoption_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    grounding_classification: str,
    run_status: str,
) -> dict[str, Any]:
    """Apply qualification, denominator, completeness, and frozen 10-point gate."""
    blockers: list[str] = []
    if grounding_classification != "image_grounded":
        blockers.append("image_grounded_not_proven")
    if run_status != "complete":
        blockers.append("run_incomplete")
    for name in ("required_abstention_recall", "unnecessary_abstention_rate"):
        for role, metrics in (("baseline", baseline), ("candidate", candidate)):
            metric = metrics.get(name)
            if not isinstance(metric, Mapping) or metric.get("denominator", 0) == 0:
                blockers.append(f"{role}_{name}_denominator_missing")
    conditions: dict[str, bool] = {}
    if not blockers:
        baseline_pass = _metric_rate(baseline, "supported_explanation_pass_rate")
        candidate_pass = _metric_rate(candidate, "supported_explanation_pass_rate")
        baseline_unsupported = _metric_rate(baseline, "unsupported_claim_rate")
        candidate_unsupported = _metric_rate(candidate, "unsupported_claim_rate")
        baseline_recall = _metric_rate(baseline, "required_abstention_recall")
        candidate_recall = _metric_rate(candidate, "required_abstention_recall")
        conditions = {
            "supported_explanation_improves_10_points": candidate_pass - baseline_pass >= 0.10 - 1e-12,
            "unsupported_claim_does_not_increase": candidate_unsupported <= baseline_unsupported,
            "required_abstention_does_not_decrease": candidate_recall >= baseline_recall,
        }
    return {
        "adopt": not blockers and all(conditions.values()),
        "blockers": blockers,
        "conditions": conditions,
        "grounding_classification": grounding_classification,
        "run_status": run_status,
    }


def semantic_signature(parsed: object, item: Mapping[str, Any]) -> str:
    """Hash response semantics after mapping all image/evidence/stage IDs to slots."""
    checked = parse_response(canonical_bytes(parsed).decode("utf-8")) if isinstance(parsed, Mapping) else parse_response(parsed)
    deliveries = item.get("image_deliveries")
    bundle = item.get("evidence_bundle")
    if not isinstance(deliveries, list) or not isinstance(bundle, Mapping):
        raise ContractError("semantic signature requires item deliveries and evidence bundle")
    identity_to_slot: dict[str, str] = {}
    for delivery in deliveries:
        if not isinstance(delivery, Mapping):
            raise ContractError("image delivery is malformed")
        slot = _text(delivery.get("slot_id"), "delivery slot ID")
        for field in ("expected_image_id", "delivered_image_id"):
            identity_to_slot[_text(delivery.get(field), field)] = slot
    stage_to_slots: dict[str, list[str]] = {}
    for stage in bundle.get("stages", []):
        if not isinstance(stage, Mapping):
            raise ContractError("bundle stage is malformed")
        stage_id = _text(stage.get("stage_id"), "stage ID")
        stage_slots: list[str] = []
        for observation in stage.get("observations", []):
            if not isinstance(observation, Mapping):
                raise ContractError("bundle observation is malformed")
            slot = _text(observation.get("slot_id"), "observation slot")
            identity_to_slot[_text(observation.get("image_id"), "observation image ID")] = slot
            identity_to_slot[_text(observation.get("evidence_id"), "observation evidence ID")] = slot
            stage_slots.append(slot)
        stage_to_slots[stage_id] = stage_slots

    def slots(identities: Sequence[str]) -> list[str]:
        result: list[str] = []
        for identity in identities:
            if identity not in identity_to_slot:
                raise ContractError("response citation is not bound to a role-aligned slot")
            result.append(identity_to_slot[identity])
        return sorted(set(result))

    normalized_assertions = []
    for assertion in checked["asserted_statements"]:
        normalized_assertions.append(
            {
                "normalized_claim": assertion["normalized_claim"],
                "slots": slots(assertion["evidence_ids"] + assertion["image_ids"]),
                "numeric_claims": sorted(
                    (
                        claim["name"], claim["value"], claim["unit"],
                        slots([claim["evidence_id"]])[0],
                    )
                    for claim in assertion["numeric_claims"]
                ),
            }
        )
    affected_slots = sorted(
        {slot for stage_id in checked["affected_stage_ids"] for slot in stage_to_slots.get(stage_id, [])}
    )
    cause = checked["cause"]
    normalized_cause = None if cause is None else {
        "switch_id": cause["switch_id"],
        "previous_recipe_id": cause["previous_recipe_id"],
        "affected_slots": affected_slots,
    }
    signature = {
        "disposition": checked["disposition"],
        "abstain": checked["abstain"],
        "assertions": normalized_assertions,
        "cause": normalized_cause,
        "affected_slots": affected_slots,
    }
    return canonical_sha256(signature)


def grounding_pair_result(
    primary_raw: object,
    shuffled_raw: object,
    primary_item: Mapping[str, Any],
    shuffled_item: Mapping[str, Any],
    primary_oracle: Mapping[str, Any],
    shuffled_oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one sealed qualification pair from raw model responses only."""
    if shuffled_item.get("role") != "shuffled_images" or shuffled_item.get("development_partition") != "grounding_qualification":
        raise ContractError("grounding uses only qualification shuffled-image controls")
    receipt = shuffled_item.get("control_receipt")
    if not isinstance(receipt, Mapping):
        raise ContractError("grounding control lacks a pair receipt")
    changed_slots = receipt.get("details", {}).get("changed_slot_ids", [])
    deltas = receipt.get("details", {}).get("delta_e76_by_slot", {})
    valid_control = bool(changed_slots) and all(
        isinstance(deltas, Mapping) and float(deltas.get(slot, 0.0)) > 2.0 for slot in changed_slots
    )
    try:
        primary_parsed = parse_response(primary_raw)
        shuffled_parsed = parse_response(shuffled_raw)
        primary_score = score_parsed_response(
            primary_parsed, primary_oracle, public_item=primary_item
        )
        shuffled_score = score_parsed_response(
            shuffled_parsed, shuffled_oracle, public_item=shuffled_item
        )
        primary_signature = semantic_signature(primary_parsed, primary_item)
        shuffled_signature = semantic_signature(shuffled_parsed, shuffled_item)
    except ContractError:
        return {
            "valid_control": valid_control,
            "raw_byte_invariant": primary_raw == shuffled_raw,
            "semantic_invariant": True,
            "pixel_conditioned_change": False,
            "grounded_pair_pass": False,
        }
    semantic_change = primary_signature != shuffled_signature

    def changed_slot_semantics(
        parsed: Mapping[str, Any],
        item: Mapping[str, Any],
        oracle: Mapping[str, Any],
    ) -> list[tuple[object, ...]]:
        identity_to_slot: dict[str, str] = {}
        for delivery in item["image_deliveries"]:
            identity_to_slot[str(delivery["expected_image_id"])] = str(delivery["slot_id"])
            identity_to_slot[str(delivery["delivered_image_id"])] = str(delivery["slot_id"])
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                identity_to_slot[str(observation["evidence_id"])] = str(observation["slot_id"])
                identity_to_slot[str(observation["image_id"])] = str(observation["slot_id"])
        kinds = {
            atom["statement_id"]: atom["kind"]
            for atom in oracle["supportable_statements"]
            if isinstance(atom, Mapping)
        }
        result: list[tuple[object, ...]] = []
        for assertion in parsed["asserted_statements"]:
            if kinds.get(assertion["statement_id"]) not in {"observation", "limit"}:
                continue
            cited_slots = sorted(
                {
                    identity_to_slot[identity]
                    for identity in assertion["evidence_ids"] + assertion["image_ids"]
                    if identity in identity_to_slot
                }
            )
            if not set(cited_slots).intersection(changed_slots):
                continue
            result.append(
                (
                    assertion["statement_id"],
                    assertion["normalized_claim"],
                    tuple(cited_slots),
                    tuple(
                        (claim["name"], claim["value"], claim["unit"], identity_to_slot.get(claim["evidence_id"]))
                        for claim in assertion["numeric_claims"]
                    ),
                )
            )
        return sorted(result)

    pixel_conditioned_change = (
        primary_parsed["abstain"] != shuffled_parsed["abstain"]
        or changed_slot_semantics(primary_parsed, primary_item, primary_oracle)
        != changed_slot_semantics(shuffled_parsed, shuffled_item, shuffled_oracle)
    )
    safe = not primary_score["unsupported_claim"] and not shuffled_score["unsupported_claim"]
    return {
        "valid_control": valid_control,
        "raw_byte_invariant": primary_raw == shuffled_raw,
        "semantic_invariant": not semantic_change,
        "pixel_conditioned_change": pixel_conditioned_change,
        "grounded_pair_pass": bool(
            valid_control
            and primary_score["supported_explanation_pass"]
            and shuffled_score["supported_explanation_pass"]
            and pixel_conditioned_change
            and safe
        ),
    }


def classify_grounding(
    pair_results: Sequence[Mapping[str, Any]],
    *,
    required_stage_ordinals: Sequence[int],
    qualification_available: bool = True,
) -> str:
    if not qualification_available:
        return "qualification_unavailable"
    rows = list(pair_results)
    if len(rows) != 12 or len({row.get("pair_id") for row in rows}) != 12:
        return "invalid_control"
    if any(row.get("valid_control") is not True for row in rows):
        return "invalid_control"
    represented = {
        ordinal
        for row in rows
        for ordinal in row.get("stage_ordinals", [])
        if isinstance(ordinal, int) and not isinstance(ordinal, bool)
    }
    if not set(required_stage_ordinals).issubset(represented):
        return "invalid_control"
    passed = sum(row.get("grounded_pair_pass") is True for row in rows)
    if passed == 0:
        return "not_image_grounded"
    if passed == 12:
        return "image_grounded"
    return "image_sensitive_not_proven_grounded"


def validate_qualification_access_order(value: object) -> dict[str, Any]:
    """Refuse selection-informed qualification, post-access tuning, and reruns."""
    root = _object(
        value,
        {
            "plan_sealed_at", "access_log_sealed_at", "selection_frozen_at",
            "qualification_first_access_at", "tuning_after_access", "run_count",
        },
        "qualification access order",
    )
    timestamps = [root[name] for name in ("plan_sealed_at", "access_log_sealed_at", "selection_frozen_at", "qualification_first_access_at")]
    if any(not isinstance(item, str) or not item for item in timestamps) or timestamps != sorted(timestamps):
        raise ContractError("qualification artifacts/access must be sealed after selection freeze and before access")
    if root["tuning_after_access"] is not False or root["run_count"] != 1:
        raise ContractError("qualification tuning or rerun is forbidden")
    return dict(root)


_EVENT_TYPES = {
    "preflight_started", "preflight_completed", "validation_started",
    "validation_terminal", "model_attempt_started", "model_attempt_terminal", "fatal",
}
_LEDGER_FIELDS = {
    "event_index", "previous_event_sha256", "event_type", "item_id", "execution_index",
    "payload_sha256", "recorded_at_utc", "event_sha256",
}


class RunLedger:
    """Create one new append-only/fsynced journal; existing journals cannot resume."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise ContractError("journal exists; resume/retry/cherry-pick is forbidden")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.events: list[dict[str, Any]] = []
        self.payloads: dict[str, object] = {}

    def append(
        self,
        event_type: str,
        *,
        item_id: str | None = None,
        execution_index: int | None = None,
        payload: object,
        recorded_at_utc: str | None = None,
    ) -> dict[str, Any]:
        if self.events and self.events[-1]["event_type"] == "fatal":
            raise ContractError("fatal run is terminal; resume/retry is forbidden")
        if event_type not in _EVENT_TYPES:
            raise ContractError("ledger event type is invalid")
        if event_type in {"preflight_started", "preflight_completed", "fatal"}:
            if item_id is not None or execution_index is not None:
                raise ContractError("global ledger event cannot name an item/index")
        else:
            _text(item_id, "ledger item ID")
            _integer(execution_index, "ledger execution index")
        payload_sha = canonical_sha256(payload)
        timestamp = recorded_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        event_without_hash = {
            "event_index": len(self.events),
            "previous_event_sha256": None if not self.events else self.events[-1]["event_sha256"],
            "event_type": event_type,
            "item_id": item_id,
            "execution_index": execution_index,
            "payload_sha256": payload_sha,
            "recorded_at_utc": timestamp,
        }
        event = {**event_without_hash, "event_sha256": canonical_sha256(event_without_hash)}
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            with os.fdopen(descriptor, "ab") as stream:
                descriptor = -1
                stream.write(canonical_bytes(event) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self.events.append(event)
        self.payloads[payload_sha] = copy.deepcopy(payload)
        return copy.deepcopy(event)


def validate_ledger(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    previous = None
    for index, raw in enumerate(events):
        row = _object(raw, _LEDGER_FIELDS, "ledger row")
        if row["event_index"] != index or row["previous_event_sha256"] != previous:
            raise ContractError("ledger index/hash chain is broken")
        if row["event_type"] not in _EVENT_TYPES:
            raise ContractError("ledger event type is invalid")
        _digest(row["payload_sha256"], "ledger payload hash")
        _digest(row["event_sha256"], "ledger event hash")
        event_without_hash = {key: row[key] for key in _LEDGER_FIELDS - {"event_sha256"}}
        if canonical_sha256(event_without_hash) != row["event_sha256"]:
            raise ContractError("ledger event hash is invalid")
        previous = row["event_sha256"]
        checked.append(dict(row))
    return checked


def _terminal_common(index: int, item: Mapping[str, Any], status: str) -> dict[str, Any]:
    return {
        "record_index": index,
        "item_id": item["item_id"],
        "execution_policy": item["execution_policy"],
        "record_status": status,
    }


def _empty_raw_record(index: int, item: Mapping[str, Any], status: str) -> dict[str, Any]:
    record = {field: None for field in _RAW_RECORD_FIELDS}
    record.update(_terminal_common(index, item, status))
    return record


def finalize_fatal_run(
    items: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, object],
    *,
    fatal_event_sha256: str,
    raw_bindings: Mapping[str, object],
) -> dict[str, Any]:
    """Derive every terminal record after fatal; never invokes a validator/model."""
    checked_events = validate_ledger(events)
    bindings = _object(
        raw_bindings,
        {
            "plan_sha256", "freeze_receipt_sha256", "model_repository", "model_revision",
            "model_weight_sha256", "generation_config_sha256",
        },
        "fatal raw bindings",
    )
    for field in (
        "plan_sha256", "freeze_receipt_sha256", "model_weight_sha256",
        "generation_config_sha256",
    ):
        _digest(bindings[field], f"fatal raw {field}")
    _text(bindings["model_repository"], "fatal raw model repository")
    _text(bindings["model_revision"], "fatal raw model revision")
    _digest(fatal_event_sha256, "fatal event hash")
    fatal_positions = [index for index, event in enumerate(checked_events) if event["event_sha256"] == fatal_event_sha256 and event["event_type"] == "fatal"]
    if len(fatal_positions) != 1:
        raise ContractError("fatal terminalization requires one bound fatal event")
    fatal_position = fatal_positions[0]
    terminal_by_item: dict[str, Mapping[str, Any]] = {}
    starts: dict[str, Mapping[str, Any]] = {}
    for event in checked_events[:fatal_position]:
        if event["event_type"] in {"validation_started", "model_attempt_started"}:
            if event["item_id"] in starts or event["item_id"] in terminal_by_item:
                raise ContractError("journal attempts to resume or retry an item")
            starts[str(event["item_id"])] = event
        elif event["event_type"] in {"validation_terminal", "model_attempt_terminal"}:
            item_id = str(event["item_id"])
            if item_id not in starts or item_id in terminal_by_item:
                raise ContractError("terminal event lacks exactly one start")
            terminal_by_item[item_id] = event
    records: list[dict[str, Any]] = []
    model_attempts_started: list[int] = []
    validations_started: list[int] = []
    for record_index, item in enumerate(items):
        item_id = _text(item.get("item_id"), "item ID")
        policy = item.get("execution_policy")
        if policy not in {"model_attempt", "pre_inference_validation_only"}:
            raise ContractError("execution policy is invalid")
        terminal = terminal_by_item.get(item_id)
        start = starts.get(item_id)
        if terminal is not None:
            payload = payloads.get(str(terminal["payload_sha256"]))
            if not isinstance(payload, Mapping) or canonical_sha256(payload) != terminal["payload_sha256"]:
                raise ContractError("terminal event payload is unavailable or hash-mismatched")
            status = payload.get("record_status")
            if policy == "pre_inference_validation_only" and status not in {"pre_inference_rejected", "pre_inference_error"}:
                raise ContractError("validation terminal status is invalid")
            if set(payload) != _RAW_RECORD_FIELDS:
                raise ContractError("terminal ledger payload must be one exact raw record")
            record = copy.deepcopy(dict(payload))
            if any(
                record[field] != expected
                for field, expected in {
                    "record_index": record_index,
                    "item_id": item_id,
                    "execution_policy": policy,
                    "record_status": status,
                }.items()
            ):
                raise ContractError("terminal raw record identity differs from plan/journal")
            if policy == "pre_inference_validation_only":
                record.update(
                    {
                        "validation_index": terminal["execution_index"], "attempt_index": None,
                        "response": None, "elapsed_seconds": None,
                        "peak_cuda_memory_allocated_bytes": None,
                        "peak_cuda_memory_reserved_bytes": None,
                        "processor_telemetry": None,
                    }
                )
                validations_started.append(int(terminal["execution_index"]))
                if status == "pre_inference_error" and record.get("error_type") not in {
                    "control_not_rejected", "validator_error", "process_crash_unmatched_validation_start"
                }:
                    raise ContractError("validation terminal error code is invalid")
            else:
                model_attempts_started.append(int(terminal["execution_index"]))
            records.append(record)
            continue
        if start is not None:
            if policy == "model_attempt" and start["event_type"] == "model_attempt_started":
                attempt_index = int(start["execution_index"])
                model_attempts_started.append(attempt_index)
                records.append(
                    {
                        **_empty_raw_record(record_index, item, "inference_error"),
                        "attempt_index": attempt_index,
                        "validation_index": None,
                        "model_repository": bindings["model_repository"],
                        "model_revision": bindings["model_revision"],
                        "model_weight_sha256": bindings["model_weight_sha256"],
                        "prompt_sha256": item.get("prompt_sha256"),
                        "ordered_delivered_image_ids": [
                            delivery["delivered_image_id"] for delivery in item.get("image_deliveries", [])
                        ],
                        "ordered_delivered_rgb_sha256": [
                            delivery["delivered_rgb_sha256"] for delivery in item.get("image_deliveries", [])
                        ],
                        "generation_config_sha256": bindings["generation_config_sha256"],
                        "error_type": "process_crash_unmatched_model_attempt_start",
                        "error": "process ended after model_attempt_started without a terminal event",
                        "response": None,
                        "elapsed_seconds": None,
                        "peak_cuda_memory_allocated_bytes": None,
                        "peak_cuda_memory_reserved_bytes": None,
                        "processor_telemetry": None,
                        "telemetry_unavailable_reason": "process_crash_before_terminal_measurement",
                    }
                )
            elif policy == "pre_inference_validation_only" and start["event_type"] == "validation_started":
                validation_index = int(start["execution_index"])
                validations_started.append(validation_index)
                records.append(
                    {
                        **_empty_raw_record(record_index, item, "pre_inference_error"),
                        "validation_index": validation_index,
                        "attempt_index": None,
                        "error_type": "process_crash_unmatched_validation_start",
                        "error": "process ended after validation_started without a terminal event",
                        "response": None,
                        "elapsed_seconds": None,
                        "peak_cuda_memory_allocated_bytes": None,
                        "peak_cuda_memory_reserved_bytes": None,
                        "processor_telemetry": None,
                    }
                )
            else:
                raise ContractError("journal start type contradicts item execution policy")
            continue
        records.append(
            {
                **_empty_raw_record(record_index, item, "not_run_after_fatal"),
                "attempt_index": None,
                "validation_index": None,
                "response": None,
                "elapsed_seconds": None,
                "peak_cuda_memory_allocated_bytes": None,
                "peak_cuda_memory_reserved_bytes": None,
                "processor_telemetry": None,
                "fatal_event_sha256": fatal_event_sha256,
                "not_run_reason": "fatal_run_terminated",
            }
        )
    if model_attempts_started and sorted(model_attempts_started) != list(range(max(model_attempts_started) + 1)):
        raise ContractError("model attempt indexes are not contiguous")
    if validations_started and sorted(validations_started) != list(range(max(validations_started) + 1)):
        raise ContractError("validation indexes are not contiguous")
    counts = Counter(record["record_status"] for record in records)
    model_not_run = sum(record["record_status"] == "not_run_after_fatal" and record["execution_policy"] == "model_attempt" for record in records)
    validation_not_run = sum(record["record_status"] == "not_run_after_fatal" and record["execution_policy"] == "pre_inference_validation_only" for record in records)
    raw = {
        "schema_version": RAW_SCHEMA,
        "plan_sha256": bindings["plan_sha256"],
        "freeze_receipt_sha256": bindings["freeze_receipt_sha256"],
        "model_repository": bindings["model_repository"],
        "model_revision": bindings["model_revision"],
        "model_weight_sha256": bindings["model_weight_sha256"],
        "generation_config_sha256": bindings["generation_config_sha256"],
        "run_status": "fatal_incomplete",
        "planned_item_count": len(records),
        "model_attempt_planned_count": sum(item["execution_policy"] == "model_attempt" for item in items),
        "pre_inference_validation_planned_count": sum(item["execution_policy"] == "pre_inference_validation_only" for item in items),
        "completed_count": counts["completed"],
        "inference_error_count": counts["inference_error"],
        "pre_inference_rejected_count": counts["pre_inference_rejected"],
        "pre_inference_error_count": counts["pre_inference_error"],
        "not_run_after_fatal_count": counts["not_run_after_fatal"],
        "model_not_run_after_fatal_count": model_not_run,
        "validation_not_run_after_fatal_count": validation_not_run,
        "journal_terminal_sha256": checked_events[-1]["event_sha256"],
        "records": records,
    }
    validate_raw_count_equations(raw)
    return raw


def validate_raw_count_equations(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    records = raw.get("records")
    if not isinstance(records, list) or raw.get("planned_item_count") != len(records):
        raise ContractError("planned item count does not equal raw records")
    model_planned = _integer(raw.get("model_attempt_planned_count"), "model attempt planned count")
    validation_planned = _integer(raw.get("pre_inference_validation_planned_count"), "validation planned count")
    if model_planned + validation_planned != len(records):
        raise ContractError("planned policy counts do not equal planned items")
    completed = _integer(raw.get("completed_count"), "completed count")
    inference_error = _integer(raw.get("inference_error_count"), "inference error count")
    rejected = _integer(raw.get("pre_inference_rejected_count"), "pre-inference rejected count")
    validation_error = _integer(raw.get("pre_inference_error_count"), "pre-inference error count")
    not_run = _integer(raw.get("not_run_after_fatal_count"), "not-run count")
    model_not_run = _integer(raw.get("model_not_run_after_fatal_count"), "model not-run count")
    validation_not_run = _integer(raw.get("validation_not_run_after_fatal_count"), "validation not-run count")
    if completed + inference_error + rejected + validation_error + not_run != len(records):
        raise ContractError("terminal status counts do not equal planned items")
    if completed + inference_error + model_not_run != model_planned:
        raise ContractError("model policy count equation failed")
    if rejected + validation_error + validation_not_run != validation_planned:
        raise ContractError("validation policy count equation failed")
    if not_run != model_not_run + validation_not_run:
        raise ContractError("not-run policy counts do not balance")
    return raw


def run_ordered_model_attempt(
    item: Mapping[str, Any],
    image_loader: Callable[[Mapping[str, Any]], object],
    model_call: Callable[[str, list[object]], str],
    *,
    synchronize: Callable[[], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Exercise one already-validated item while binding every ordered image."""
    if item.get("execution_policy") != "model_attempt":
        raise ContractError("validation-only control cannot invoke image processor/model")
    deliveries = item.get("image_deliveries")
    if not isinstance(deliveries, list) or len(deliveries) not in {4, 6}:
        raise ContractError("model attempt requires four or six ordered images")
    loaded = [image_loader(delivery) for delivery in deliveries]
    synchronize()
    started = clock()
    response = model_call(str(item.get("prompt")), loaded)
    synchronize()
    elapsed = clock() - started
    if not isinstance(response, str):
        raise ContractError("model response must be text")
    return {
        "record_status": "completed",
        "ordered_delivered_image_ids": [row["delivered_image_id"] for row in deliveries],
        "ordered_delivered_rgb_sha256": [row["delivered_rgb_sha256"] for row in deliveries],
        "response": response,
        "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "elapsed_seconds": elapsed,
        "processor_telemetry": {"ordered_image_count": len(loaded)},
    }


_DELIVERY_FIELDS = {
    "slot_id", "expected_image_id", "expected_rgb_sha256", "expected_source_run_id",
    "expected_source_case_id", "expected_source_vial_ordinal", "expected_source_time",
    "delivered_image_id", "delivered_rgb_sha256", "delivered_source_run_id",
    "delivered_source_case_id", "delivered_source_vial_ordinal", "delivered_source_time",
}
_ITEM_FIELDS = {
    "item_id", "source_item_id", "role", "execution_policy", "paired_primary_item_id",
    "run_id", "case_id", "vial_ordinal", "split_version", "development_partition",
    "evidence_bundle", "evidence_bundle_sha256", "prompt", "prompt_sha256",
    "image_deliveries", "control_receipt",
}
_PLAN_FIELDS = {
    "schema_version", "protocol_id", "benchmark_role", "development_partition",
    "split_version", "prompt_template_sha256", "guard_config_sha256",
    "evidence_schema_sha256", "model_input_schema_sha256", "source_artifacts",
    "planned_item_count", "model_attempt_planned_count",
    "pre_inference_validation_planned_count", "items",
}


def validate_image_locators(value: object, *, verify_files: bool = False) -> dict[str, Any]:
    root = _object(value, {"schema_version", "plan_sha256", "images"}, "image locators")
    if root["schema_version"] != LOCATOR_SCHEMA:
        raise ContractError(f"locator schema must be {LOCATOR_SCHEMA}")
    _digest(root["plan_sha256"], "locator plan hash")
    images = root["images"]
    if not isinstance(images, list) or not images:
        raise ContractError("image locators must contain images")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    checked: list[dict[str, Any]] = []
    for raw in images:
        row = _object(raw, {"image_id", "rgb_path", "rgb_sha256"}, "image locator")
        image_id = _opaque(row["image_id"], "image ID")
        path_text = _text(row["rgb_path"], "RGB path")
        digest = _digest(row["rgb_sha256"], "RGB hash")
        if image_id in seen_ids or path_text in seen_paths:
            raise ContractError("image locator IDs and paths must be unique")
        seen_ids.add(image_id)
        seen_paths.add(path_text)
        if verify_files:
            path = Path(path_text)
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ContractError("RGB locator is unreadable or hash-mismatched")
        checked.append(dict(row))
    return {**dict(root), "images": checked}


def _observations_by_slot(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(observation["slot_id"]): observation
        for stage in bundle["stages"]
        for observation in stage["observations"]
    }


def _validate_delivery(value: object) -> dict[str, Any]:
    row = _object(value, _DELIVERY_FIELDS, "image delivery")
    for field in (
        "slot_id", "expected_image_id", "expected_source_run_id", "expected_source_case_id",
        "delivered_image_id", "delivered_source_run_id", "delivered_source_case_id",
    ):
        _text(row[field], f"delivery {field}")
    for field in ("expected_rgb_sha256", "delivered_rgb_sha256"):
        _digest(row[field], f"delivery {field}")
    for field in ("expected_source_vial_ordinal", "delivered_source_vial_ordinal"):
        _integer(row[field], f"delivery {field}", minimum=1)
    _source_time(row["expected_source_time"], "expected delivery time")
    _source_time(row["delivered_source_time"], "delivered delivery time")
    return dict(row)


def _validate_receipt(value: object, role: str) -> Mapping[str, Any]:
    row = _object(
        value,
        {"transformation_id", "source_item_sha256", "details", "transformation_sha256"},
        "control receipt",
    )
    _text(row["transformation_id"], "transformation ID")
    _digest(row["source_item_sha256"], "control source hash")
    _digest(row["transformation_sha256"], "control transformation hash")
    if not isinstance(row["details"], Mapping):
        raise ContractError("control receipt details must be an object")
    unsigned = {key: row[key] for key in ("transformation_id", "source_item_sha256", "details")}
    if canonical_sha256(unsigned) != row["transformation_sha256"]:
        raise ContractError("control transformation receipt hash is invalid")
    if role.replace("_", "-") not in row["transformation_id"]:
        raise ContractError("control receipt does not bind its role")
    for key in row["details"]:
        if str(key).lower() in {"expected_answer", "expected_decision", "label", "oracle"}:
            raise ContractError("control receipt cannot disclose an expected answer")
    return row


def validate_plan(
    value: object,
    *,
    locators: object | None = None,
    accepted_provenance_by_source_item: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate the strict public plan and optional separated image locator map."""
    root = _object(value, _PLAN_FIELDS, "public plan")
    if not isinstance(accepted_provenance_by_source_item, Mapping):
        raise ContractError("accepted provenance by source item is required for plan validation")
    if root["schema_version"] != PLAN_SCHEMA or root["protocol_id"] != PROTOCOL_ID:
        raise ContractError("public plan schema/protocol is invalid")
    role = root["benchmark_role"]
    partition = root["development_partition"]
    if role == "development":
        if partition not in {"selection", "grounding_qualification"}:
            raise ContractError("development plan needs one development partition")
    elif role == "test":
        if partition is not None:
            raise ContractError("test plan development partition must be null")
    else:
        raise ContractError("benchmark role must be development or test")
    _text(root["split_version"], "plan split version")
    for field in (
        "prompt_template_sha256", "guard_config_sha256", "evidence_schema_sha256",
        "model_input_schema_sha256",
    ):
        _digest(root[field], f"plan {field}")
    artifacts = root["source_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractError("plan source artifacts must contain Part B bindings")
    artifact_ids: set[str] = set()
    for raw in artifacts:
        artifact = _object(raw, {"artifact_id", "sha256"}, "source artifact")
        identity = _opaque(artifact["artifact_id"], "source artifact ID")
        _digest(artifact["sha256"], "source artifact hash")
        if identity in artifact_ids:
            raise ContractError("source artifact IDs must be unique")
        artifact_ids.add(identity)
    items = root["items"]
    if not isinstance(items, list) or not items:
        raise ContractError("plan must contain items")
    planned = _integer(root["planned_item_count"], "planned item count")
    model_count = _integer(root["model_attempt_planned_count"], "model attempt planned count")
    validation_count = _integer(root["pre_inference_validation_planned_count"], "validation planned count")
    if planned != len(items) or model_count + validation_count != planned:
        raise ContractError("plan item/policy counts do not balance")
    checked_items: list[dict[str, Any]] = []
    item_ids: set[str] = set()
    primary_ids: set[str] = set()
    delivered_locator_bindings: list[tuple[str, str]] = []
    for raw in items:
        item = _object(raw, _ITEM_FIELDS, "public item")
        item_id = _opaque(item["item_id"], "item ID")
        _opaque(item["source_item_id"], "source item ID")
        if item_id in item_ids:
            raise ContractError("public item IDs must be unique")
        item_ids.add(item_id)
        item_role = item["role"]
        if item_role not in {"primary", "support_removed", "shuffled_evidence_ids", "shuffled_images", "unsupported_cause_probe"}:
            raise ContractError("public item role is invalid")
        expected_policy = "pre_inference_validation_only" if item_role == "shuffled_evidence_ids" else "model_attempt"
        if item["execution_policy"] != expected_policy:
            raise ContractError("item role/execution policy mismatch")
        if item_role == "primary":
            if item["paired_primary_item_id"] is not None or item["control_receipt"] is not None:
                raise ContractError("primary cannot have a pair/control receipt")
            primary_ids.add(item_id)
        else:
            _opaque(item["paired_primary_item_id"], "paired primary item ID")
            _validate_receipt(item["control_receipt"], item_role)
        if role == "test" and item_role != "primary":
            raise ContractError("test plans cannot contain controls")
        if item["development_partition"] != partition:
            raise ContractError("item development partition differs from plan")
        if item["split_version"] != root["split_version"]:
            raise ContractError("item split version differs from plan")
        _integer(item["vial_ordinal"], "item vial ordinal", minimum=1)
        for field in ("run_id", "case_id"):
            _opaque(item[field], f"item {field}")
        bundle_hash = _digest(item["evidence_bundle_sha256"], "evidence bundle hash")
        if canonical_sha256(item["evidence_bundle"]) != bundle_hash:
            raise ContractError("evidence bundle hash mismatch")
        if item_role == "shuffled_evidence_ids":
            bundle = _object(
                item["evidence_bundle"], {"schema_version", "target_context", "stages"},
                "quarantined shuffled-ID bundle",
            )
            if bundle["schema_version"] != MODEL_INPUT_SCHEMA:
                raise ContractError("quarantined shuffled-ID bundle schema is invalid")
        else:
            provenance = accepted_provenance_by_source_item.get(str(item["source_item_id"]))
            bundle = validate_model_input(
                item["evidence_bundle"], accepted_provenance=provenance
            )
        context = bundle["target_context"]
        if any(item[field] != context[field] for field in ("run_id", "case_id", "vial_ordinal", "split_version")):
            raise ContractError("item target identity differs from bundle target")
        prompt = _text(item["prompt"], "item prompt")
        if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"]:
            raise ContractError("item prompt hash mismatch")
        if item["execution_policy"] == "model_attempt" and canonical_bytes(bundle).decode("utf-8") not in prompt:
            raise ContractError("model-attempt prompt must embed its exact validated evidence bundle")
        deliveries = item["image_deliveries"]
        if not isinstance(deliveries, list) or len(deliveries) not in {4, 6}:
            raise ContractError("item requires four or six image deliveries")
        observations = _observations_by_slot(bundle)
        checked_deliveries = [_validate_delivery(delivery) for delivery in deliveries]
        if [row["slot_id"] for row in checked_deliveries] != list(observations):
            raise ContractError("delivery order/slots differ from stage observation order")
        delivered_ids: set[str] = set()
        for delivery in checked_deliveries:
            observation = observations[delivery["slot_id"]]
            expected = {
                "expected_image_id": observation["image_id"],
                "expected_rgb_sha256": observation["rgb_sha256"],
                "expected_source_run_id": observation["source_run_id"],
                "expected_source_case_id": observation["source_case_id"],
                "expected_source_vial_ordinal": observation["source_vial_ordinal"],
                "expected_source_time": observation["source_time"],
            }
            if any(delivery[key] != expected_value for key, expected_value in expected.items()):
                raise ContractError("delivery expected provenance differs from evidence slot")
            if item_role != "shuffled_images":
                for suffix in ("image_id", "rgb_sha256", "source_run_id", "source_case_id", "source_vial_ordinal", "source_time"):
                    if delivery[f"delivered_{suffix}"] != delivery[f"expected_{suffix}"]:
                        raise ContractError("unreceipted delivered-image provenance mismatch")
            if delivery["delivered_image_id"] in delivered_ids:
                raise ContractError("duplicate delivered image within item")
            delivered_ids.add(delivery["delivered_image_id"])
            delivered_locator_bindings.append((delivery["delivered_image_id"], delivery["delivered_rgb_sha256"]))
        checked_items.append({**dict(item), "image_deliveries": checked_deliveries})
    for item in checked_items:
        if item["role"] != "primary" and item["paired_primary_item_id"] not in primary_ids:
            raise ContractError("control pair does not bind a primary in the plan")
    by_id = {item["item_id"]: item for item in checked_items}
    primary_by_source = {
        item["source_item_id"]: item for item in checked_items if item["role"] == "primary"
    }
    for item in checked_items:
        if item["role"] == "primary":
            continue
        primary = by_id[item["paired_primary_item_id"]]
        receipt = item["control_receipt"]
        if receipt["source_item_sha256"] != canonical_sha256(primary):
            raise ContractError("control receipt source hash differs from paired primary")
        if item["source_item_id"] != primary["source_item_id"]:
            raise ContractError("control source item ID differs from paired primary")
        if item["role"] == "shuffled_evidence_ids":
            details = receipt["details"]
            if set(details) != {"source_evidence_ids", "permuted_evidence_ids"}:
                raise ContractError("shuffled-ID mutation receipt fields are invalid")
            primary_observations = [
                observation for stage in primary["evidence_bundle"]["stages"]
                for observation in stage["observations"]
            ]
            control_observations = [
                observation for stage in item["evidence_bundle"]["stages"]
                for observation in stage["observations"]
            ]
            source_ids = [row["evidence_id"] for row in primary_observations]
            permuted_ids = [row["evidence_id"] for row in control_observations]
            if details["source_evidence_ids"] != source_ids or details["permuted_evidence_ids"] != permuted_ids or source_ids == permuted_ids:
                raise ContractError("shuffled-ID receipt does not bind the exact mutation")
            restored = copy.deepcopy(item["evidence_bundle"])
            for stage, primary_stage in zip(restored["stages"], primary["evidence_bundle"]["stages"], strict=True):
                for observation, primary_observation in zip(stage["observations"], primary_stage["observations"], strict=True):
                    observation["evidence_id"] = primary_observation["evidence_id"]
            if restored != primary["evidence_bundle"]:
                raise ContractError("shuffled-ID control mutates fields beyond evidence IDs")
            if item["prompt"] != primary["prompt"] or item["image_deliveries"] != primary["image_deliveries"]:
                raise ContractError("shuffled-ID control may mutate only evidence-ID bindings")
        elif item["role"] == "shuffled_images":
            details = receipt["details"]
            if set(details) != {"donor_source_item_id", "changed_slot_ids", "delta_e76_by_slot", "deliveries"}:
                raise ContractError("shuffled-image receipt fields are invalid")
            donor = primary_by_source.get(details["donor_source_item_id"])
            if donor is None or donor["run_id"] == primary["run_id"]:
                raise ContractError("shuffled-image donor must be a primary from another run")
            changed = details["changed_slot_ids"]
            deltas = details["delta_e76_by_slot"]
            if not isinstance(changed, list) or not changed or not isinstance(deltas, Mapping) or any(
                slot not in deltas or _finite(deltas[slot], "changed-slot Delta-E76") <= 2.0
                for slot in changed
            ):
                raise ContractError("shuffled-image changed-slot measurement gate failed")
            if details["deliveries"] != item["image_deliveries"]:
                raise ContractError("shuffled-image receipt does not bind complete donor deliveries")
            if item["prompt"] != primary["prompt"] or item["evidence_bundle"] != primary["evidence_bundle"]:
                raise ContractError("shuffled-image prompt/evidence must be byte-identical to primary")
            expected_ids = {row["expected_image_id"] for row in item["image_deliveries"]}
            delivered_ids = {row["delivered_image_id"] for row in item["image_deliveries"]}
            if expected_ids & delivered_ids:
                raise ContractError("shuffled-image source and donor deliveries must be disjoint")
            donor_by_slot = {row["slot_id"]: row for row in donor["image_deliveries"]}
            for delivery in item["image_deliveries"]:
                donor_delivery = donor_by_slot.get(delivery["slot_id"])
                if donor_delivery is None or any(
                    delivery[f"delivered_{suffix}"] != donor_delivery[f"delivered_{suffix}"]
                    for suffix in ("image_id", "rgb_sha256", "source_run_id", "source_case_id", "source_vial_ordinal", "source_time")
                ):
                    raise ContractError("shuffled-image donor provenance is false")
        elif item["role"] == "support_removed":
            context = item["evidence_bundle"]["target_context"]
            if context["disposition"] != "insufficient_evidence" or "support_removed" not in context["limit_codes"]:
                raise ContractError("support-removed control does not recompute insufficient evidence")
        elif item["role"] == "unsupported_cause_probe":
            if item["evidence_bundle"] != primary["evidence_bundle"] or not item["prompt"].endswith(_CAUSE_PROBE):
                raise ContractError("unsupported-cause probe does not preserve evidence/fixed instruction")
    if sum(item["execution_policy"] == "model_attempt" for item in checked_items) != model_count:
        raise ContractError("model attempt count differs from item policies")
    if sum(item["execution_policy"] == "pre_inference_validation_only" for item in checked_items) != validation_count:
        raise ContractError("validation count differs from item policies")
    source_hashes = {artifact["sha256"] for artifact in artifacts}
    if set(accepted_provenance_by_source_item) != {item["source_item_id"] for item in checked_items}:
        raise ContractError("accepted provenance does not exactly cover source item IDs")
    model_attempt_sources = {
        item["source_item_id"]
        for item in checked_items
        if item["execution_policy"] == "model_attempt"
    }
    for source_item_id, provenance in accepted_provenance_by_source_item.items():
        checked_provenance = _object(provenance, _PROVENANCE_FIELDS, "accepted provenance")
        if checked_provenance["receipt_sha256"] not in source_hashes:
            raise ContractError(
                f"accepted provenance for {source_item_id} is not bound as a plan source artifact"
            )
        if source_item_id in model_attempt_sources and (
            checked_provenance["purpose"] != "production"
            or checked_provenance["production_eligible"] is not True
        ):
            raise ContractError(
                "development/test model attempts reject synthetic or non-production provenance"
            )
    if locators is not None:
        checked_locators = validate_image_locators(locators)
        if checked_locators["plan_sha256"] != canonical_sha256(value):
            raise ContractError("locator plan hash mismatch")
        locator_by_id = {row["image_id"]: row for row in checked_locators["images"]}
        for image_id, rgb_hash in delivered_locator_bindings:
            if image_id not in locator_by_id or locator_by_id[image_id]["rgb_sha256"] != rgb_hash:
                raise ContractError("unknown image ID or locator RGB/hash mismatch")
        locator_paths = [row["rgb_path"] for row in checked_locators["images"]]
        for item in checked_items:
            if any(path in item["prompt"] for path in locator_paths):
                raise ContractError("local image path entered a model prompt")
            if item["role"] == "shuffled_images" and any(
                delivery["delivered_image_id"] in item["prompt"]
                or delivery["delivered_rgb_sha256"] in item["prompt"]
                for delivery in item["image_deliveries"]
            ):
                raise ContractError("delivered replacement identity entered a shuffled-image prompt")
    return {**dict(root), "items": checked_items}


def validate_pre_inference_control(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the mandatory rejection receipt; this boundary never calls a model."""
    if item.get("role") != "shuffled_evidence_ids" or item.get("execution_policy") != "pre_inference_validation_only":
        raise ContractError("pre-inference validator accepts only quarantined shuffled-ID controls")
    receipt = item.get("control_receipt")
    if not isinstance(receipt, Mapping):
        raise ContractError("validation-only control lacks its mutation receipt")
    return {
        "record_status": "pre_inference_rejected",
        "validation_stage": "part_b_intake",
        "rejection_code": "evidence_id_binding_mismatch",
        "validator_receipt_sha256": canonical_sha256(
            {"item_id": item.get("item_id"), "control_receipt_sha256": receipt.get("transformation_sha256")}
        ),
    }


def validate_oracle(
    value: object,
    plan: object,
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
) -> dict[str, Any]:
    checked_plan = validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )
    root = _object(value, {"schema_version", "plan_sha256", "protocol_sha256", "items"}, "oracle")
    if root["schema_version"] != ORACLE_SCHEMA:
        raise ContractError(f"oracle schema must be {ORACLE_SCHEMA}")
    if root["plan_sha256"] != canonical_sha256(plan):
        raise ContractError("oracle plan hash mismatch")
    _digest(root["protocol_sha256"], "oracle protocol hash")
    rows = root["items"]
    if not isinstance(rows, list) or len(rows) != len(checked_plan["items"]):
        raise ContractError("oracle must contain one row per public item")
    plan_by_id = {item["item_id"]: item for item in checked_plan["items"]}
    seen: set[str] = set()
    checked_rows: list[dict[str, Any]] = []
    fields = {
        "item_id", "source_item_id", "role", "paired_primary_item_id", "expected_decision",
        "required_abstention", "required_statement_ids", "supportable_statements",
        "cause_expectation", "cause_supported_by_part_b", "post_inference_label_ref",
    }
    for raw in rows:
        row = _object(raw, fields, "oracle item")
        item_id = row["item_id"]
        if item_id not in plan_by_id or item_id in seen:
            raise ContractError("oracle item IDs must exactly match public plan")
        seen.add(item_id)
        public = plan_by_id[item_id]
        for field in ("source_item_id", "role", "paired_primary_item_id"):
            if row[field] != public[field]:
                raise ContractError("oracle/public identity or role mismatch")
        if row["expected_decision"] not in {"keep", "discard", "insufficient_evidence"} or not isinstance(row["required_abstention"], bool):
            raise ContractError("oracle decision/abstention is invalid")
        if row["required_abstention"] != (row["expected_decision"] == "insufficient_evidence"):
            raise ContractError("oracle abstention contradicts expected decision")
        atoms_raw = row["supportable_statements"]
        if not isinstance(atoms_raw, list):
            raise ContractError("oracle supportable statements must be a list")
        atoms = [_oracle_atom(atom) for atom in atoms_raw]
        required_ids = _string_list(row["required_statement_ids"], "oracle required statement IDs")
        atom_by_id = {atom["statement_id"]: atom for atom in atoms}
        required_atom_ids = {atom["statement_id"] for atom in atoms if atom["required"]}
        if len(atom_by_id) != len(atoms) or set(required_ids) != required_atom_ids:
            raise ContractError("oracle required statement IDs must equal all and only required atoms")
        public_disposition = public["evidence_bundle"]["target_context"]["disposition"]
        if row["expected_decision"] != public_disposition:
            raise ContractError("oracle expected decision differs from public evidence disposition")
        if not isinstance(row["cause_supported_by_part_b"], bool):
            raise ContractError("cause_supported_by_part_b must be boolean")
        cause = row["cause_expectation"]
        if cause is not None:
            cause = _object(cause, _CAUSE_FIELDS, "oracle cause expectation")
            _text(cause["switch_id"], "oracle switch ID")
            _text(cause["previous_recipe_id"], "oracle previous recipe ID")
            _string_list(cause["affected_stage_ids"], "oracle affected stages")
        if row["cause_supported_by_part_b"]:
            if cause is None or not any(atom["kind"] == "cause" and atom["evidence_ids"] for atom in atoms):
                raise ContractError("supported cause requires an independently cited cause atom")
        elif cause is not None:
            # The expectation may be retained for hidden post-inference analysis, but cannot score positive.
            pass
        label_ref = row["post_inference_label_ref"]
        if label_ref is not None:
            label = _object(label_ref, {"artifact_id", "item_id", "sha256"}, "post-inference label ref")
            _text(label["artifact_id"], "label artifact ID")
            _text(label["item_id"], "label item ID")
            _digest(label["sha256"], "label ref hash")
        checked_rows.append({**dict(row), "supportable_statements": atoms})
    if seen != set(plan_by_id):
        raise ContractError("oracle item IDs must exactly match public plan")
    return {**dict(root), "items": checked_rows}


def reject_oracle_inference_configuration(value: object) -> None:
    """Inference configuration has no oracle path/content capability."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "oracle" in str(key).lower():
                raise ContractError("inference process cannot open or receive an oracle")
            reject_oracle_inference_configuration(child)
    elif isinstance(value, list):
        for child in value:
            reject_oracle_inference_configuration(child)


def build_grounding_qualification(
    primaries: Sequence[Mapping[str, Any]],
    pair_delta_e76: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Build exactly 12 ordered/shuffled pairs from four sealed three-vial runs."""
    rows = list(primaries)
    by_run: dict[str, list[Mapping[str, Any]]] = {}
    for item in rows:
        if item.get("role") != "primary" or item.get("development_partition") != "grounding_qualification":
            raise ContractError("qualification inventory accepts only qualification primaries")
        by_run.setdefault(str(item.get("run_id")), []).append(item)
    if len(by_run) != 4 or any(len(items) != 3 for items in by_run.values()) or len(rows) != 12:
        raise ContractError("qualification requires exactly four runs and three primary vials per run")
    if len({item.get("source_item_id") for item in rows}) != 12:
        raise ContractError("qualification source item IDs must be unique")
    result_items: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for target in rows:
        target_source = str(target["source_item_id"])
        target_slots = [row["slot_id"] for row in target["image_deliveries"]]
        candidates: list[tuple[str, Mapping[str, Any], list[str]]] = []
        for donor in rows:
            if donor["run_id"] == target["run_id"]:
                continue
            donor_slots = [row["slot_id"] for row in donor["image_deliveries"]]
            if donor_slots != target_slots:
                continue
            slot_deltas = pair_delta_e76.get(target_source, {}).get(str(donor["source_item_id"]), {})
            changed = [slot for slot in target_slots if float(slot_deltas.get(slot, 0.0)) > 2.0]
            if changed:
                rank = hashlib.sha256(
                    f"{PROTOCOL_ID}|qualification-donor|{target_source}|{donor['source_item_id']}".encode()
                ).hexdigest()
                candidates.append((rank, donor, changed))
        if not candidates:
            raise ContractError(f"qualification target {target_source} has no eligible cross-run donor")
        _, donor, changed = min(candidates, key=lambda candidate: candidate[0])
        primary = copy.deepcopy(dict(target))
        result_items.append(primary)
        control_id = "qcontrol-" + hashlib.sha256(
            f"{PROTOCOL_ID}|qualification-control|{target_source}".encode()
        ).hexdigest()[:24]
        slot_deltas = pair_delta_e76[target_source][str(donor["source_item_id"])]
        control = build_shuffled_images_control(
            primary,
            donor,
            changed_slot_ids=changed,
            delta_e76_by_slot={slot: slot_deltas[slot] for slot in changed},
            item_id=control_id,
        )
        result_items.append(control)
        pairs.append(
            {
                "primary_item_id": primary["item_id"],
                "shuffled_item_id": control["item_id"],
                "target_source_item_id": target_source,
                "donor_source_item_id": donor["source_item_id"],
                "changed_slot_ids": changed,
                "pair_receipt_sha256": control["control_receipt"]["transformation_sha256"],
            }
        )
    result = {
        "schema_version": "rca-bench-grounding-qualification-v1",
        "protocol_id": PROTOCOL_ID,
        "run_ids": sorted(by_run),
        "pair_count": 12,
        "model_attempt_count": 24,
        "items": result_items,
        "pairs": pairs,
    }
    result["qualification_sha256"] = canonical_sha256(result)
    return result


_RAW_ROOT_FIELDS = {
    "schema_version", "plan_sha256", "freeze_receipt_sha256", "journal_terminal_sha256",
    "model_repository", "model_revision", "model_weight_sha256", "generation_config_sha256",
    "run_status", "planned_item_count", "model_attempt_planned_count",
    "pre_inference_validation_planned_count", "completed_count", "inference_error_count",
    "pre_inference_rejected_count", "pre_inference_error_count", "not_run_after_fatal_count",
    "model_not_run_after_fatal_count", "validation_not_run_after_fatal_count", "records",
}
_RAW_RECORD_FIELDS = {
    "record_index", "item_id", "execution_policy", "record_status", "attempt_index",
    "validation_index", "model_repository", "model_revision", "model_weight_sha256",
    "prompt_sha256", "ordered_delivered_image_ids", "ordered_delivered_rgb_sha256",
    "generation_config_sha256", "response", "response_utf8_sha256", "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
    "processor_telemetry", "telemetry_unavailable_reason", "error_type", "error",
    "validation_stage", "rejection_code", "validator_receipt_sha256",
    "fatal_event_sha256", "not_run_reason",
}


def _validate_freeze_receipt_structure(
    value: object,
    plan: object,
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    expected_bindings: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version", "protocol_sha256", "builder_source_sha256", "scorer_source_sha256",
        "test_hashes", "evidence_schema_sha256", "prompt_sha256", "guard_config_sha256",
        "plan_sha256", "oracle_sha256", "locator_sha256", "split_ids", "source_artifacts",
        "model_repository", "model_revision", "model_weight_sha256",
        "development_partition_receipt_sha256", "qualification_access_log_sha256",
        "generation_config_sha256", "planned_item_count", "model_attempt_planned_count",
        "pre_inference_validation_planned_count", "required_abstention_primary_count",
        "non_required_abstention_primary_count", "status_count_equation", "fatal_outcome",
        "partition_checker_bindings", "git_commit",
    }
    root = _object(value, fields, "freeze receipt")
    if root["schema_version"] != FREEZE_SCHEMA:
        raise ContractError(f"freeze schema must be {FREEZE_SCHEMA}")
    if isinstance(plan, Mapping) and plan.get("benchmark_role") == "test":
        raw_primary = [
            item for item in plan.get("items", [])
            if isinstance(item, Mapping) and item.get("role") == "primary"
        ]
        raw_required = sum(
            item.get("evidence_bundle", {}).get("target_context", {}).get("disposition")
            == "insufficient_evidence"
            for item in raw_primary
        )
        if raw_required < 1 or len(raw_primary) - raw_required < 1:
            raise ContractError("test freeze requires both abstention denominator classes")
    checked_plan = validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )
    if root["plan_sha256"] != canonical_sha256(plan):
        raise ContractError("freeze plan hash mismatch")
    digest_fields = fields - {
        "schema_version", "test_hashes", "split_ids", "source_artifacts", "model_repository",
        "model_revision", "planned_item_count", "model_attempt_planned_count",
        "pre_inference_validation_planned_count", "required_abstention_primary_count",
        "non_required_abstention_primary_count", "status_count_equation", "fatal_outcome",
        "partition_checker_bindings", "git_commit",
    }
    for field in digest_fields:
        _digest(root[field], f"freeze {field}")
    if not isinstance(root["test_hashes"], list) or not root["test_hashes"]:
        raise ContractError("freeze must bind test hashes")
    for digest in root["test_hashes"]:
        _digest(digest, "freeze test hash")
    if not isinstance(root["split_ids"], list) or not root["split_ids"]:
        raise ContractError("freeze must bind split IDs")
    _string_list(root["split_ids"], "freeze split IDs")
    artifacts = root["source_artifacts"]
    if artifacts != checked_plan["source_artifacts"]:
        raise ContractError("freeze source artifacts differ from plan")
    _text(root["model_repository"], "freeze model repository")
    _text(root["model_revision"], "freeze model revision")
    if not isinstance(root["git_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", root["git_commit"]) is None:
        raise ContractError("freeze Git commit must be 40 lowercase hex")
    for field in ("planned_item_count", "model_attempt_planned_count", "pre_inference_validation_planned_count"):
        if root[field] != checked_plan[field]:
            raise ContractError("freeze count differs from plan")
    primary_items = [item for item in checked_plan["items"] if item["role"] == "primary"]
    required_abstention = sum(
        item["evidence_bundle"]["target_context"]["disposition"] == "insufficient_evidence"
        for item in primary_items
    )
    non_required_abstention = len(primary_items) - required_abstention
    if (
        root["required_abstention_primary_count"] != required_abstention
        or root["non_required_abstention_primary_count"] != non_required_abstention
    ):
        raise ContractError("freeze abstention denominator counts differ from public primary items")
    if checked_plan["benchmark_role"] == "test" and (
        required_abstention < 1 or non_required_abstention < 1
    ):
        raise ContractError("test freeze requires both abstention denominator classes")
    expected_equation = (
        "completed+inference_error+pre_inference_rejected+pre_inference_error+"
        "not_run_after_fatal=planned_item_count"
    )
    if root["status_count_equation"] != expected_equation:
        raise ContractError("freeze status-count equation is not the frozen equation")
    if root["fatal_outcome"] != "no_rerun_incomplete_fails_adoption":
        raise ContractError("freeze fatal outcome must refuse rerun and fail adoption")
    bindings = _object(
        root["partition_checker_bindings"],
        {"split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code"},
        "freeze partition checker bindings",
    )
    for name, digest in bindings.items():
        _digest(digest, f"freeze checker {name}")
    expected_fields = {
        "protocol_sha256", "builder_source_sha256", "scorer_source_sha256", "test_hashes",
        "evidence_schema_sha256", "prompt_sha256", "guard_config_sha256", "git_commit",
    }
    expected = _object(expected_bindings, expected_fields, "expected freeze bindings")
    if any(root[field] != expected[field] for field in expected_fields):
        raise ContractError("freeze declared hash differs from explicit expected frozen input")
    if (
        root["prompt_sha256"] != checked_plan["prompt_template_sha256"]
        or root["evidence_schema_sha256"] != checked_plan["evidence_schema_sha256"]
        or root["guard_config_sha256"] != checked_plan["guard_config_sha256"]
    ):
        raise ContractError("freeze prompt/evidence/guard hashes differ from plan")
    return dict(root)


def validate_freeze_bindings(
    value: object,
    plan: object,
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    expected_bindings: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate freeze structure without opening the hidden oracle artifact."""
    return _validate_freeze_receipt_structure(
        value,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected_bindings,
    )


def validate_freeze_receipt(
    value: object,
    plan: object,
    *,
    locators: object,
    oracle: object,
    accepted_provenance_by_source_item: Mapping[str, object],
    expected_bindings: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate a freeze receipt against the actual separated locator/oracle artifacts."""
    root = _validate_freeze_receipt_structure(
        value,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected_bindings,
    )
    checked_locators = validate_image_locators(locators)
    if checked_locators["plan_sha256"] != canonical_sha256(plan):
        raise ContractError("freeze locator artifact targets another plan")
    validate_oracle(
        oracle,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    if root["locator_sha256"] != canonical_sha256(locators):
        raise ContractError("freeze locator hash differs from the actual artifact")
    if root["oracle_sha256"] != canonical_sha256(oracle):
        raise ContractError("freeze oracle hash differs from the actual artifact")
    return root


def _nullable_digest(value: object, role: str) -> None:
    if value is not None:
        _digest(value, role)


def validate_raw(
    value: object,
    plan: object,
    freeze_receipt: object,
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    ledger_events: Sequence[Mapping[str, Any]] | None = None,
    expected_freeze_bindings: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    root = _object(value, _RAW_ROOT_FIELDS, "raw artifact")
    if root["schema_version"] != RAW_SCHEMA:
        raise ContractError(f"raw schema must be {RAW_SCHEMA}")
    checked_plan = validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )
    checked_freeze = _validate_freeze_receipt_structure(
        freeze_receipt,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected_freeze_bindings,
    )
    if root["plan_sha256"] != canonical_sha256(plan) or root["freeze_receipt_sha256"] != canonical_sha256(freeze_receipt):
        raise ContractError("raw plan/freeze hash mismatch")
    for field in ("journal_terminal_sha256", "model_weight_sha256", "generation_config_sha256"):
        _digest(root[field], f"raw {field}")
    for field in ("model_repository", "model_revision", "model_weight_sha256", "generation_config_sha256"):
        freeze_field = field
        if root[field] != checked_freeze[freeze_field]:
            raise ContractError("raw model/generation identity differs from freeze")
    if root["run_status"] not in {"complete", "fatal_incomplete"}:
        raise ContractError("raw run status is invalid")
    if ledger_events is None:
        raise ContractError("raw validation requires actual hash-chained ledger terminal evidence")
    checked_ledger = validate_ledger(ledger_events)
    if not checked_ledger or checked_ledger[-1]["event_sha256"] != root["journal_terminal_sha256"]:
        raise ContractError("raw journal terminal hash differs from the supplied ledger")
    validate_raw_count_equations(root)
    records = root["records"]
    if len(records) != len(checked_plan["items"]):
        raise ContractError("raw records do not cover the plan")
    model_indexes: list[int] = []
    validation_indexes: list[int] = []
    for index, (raw_record, item) in enumerate(zip(records, checked_plan["items"], strict=True)):
        record = _object(raw_record, _RAW_RECORD_FIELDS, "raw record")
        if record["record_index"] != index or record["item_id"] != item["item_id"] or record["execution_policy"] != item["execution_policy"]:
            raise ContractError("raw record order/identity/policy differs from plan")
        status = record["record_status"]
        if status not in {"completed", "inference_error", "pre_inference_rejected", "pre_inference_error", "not_run_after_fatal"}:
            raise ContractError("raw record status is invalid")
        if status in {"completed", "inference_error"}:
            if item["execution_policy"] != "model_attempt":
                raise ContractError("validation item became a model attempt")
            model_indexes.append(_integer(record["attempt_index"], "attempt index"))
            if record["validation_index"] is not None:
                raise ContractError("model record cannot have validation index")
            for field in ("model_repository", "model_revision", "model_weight_sha256", "prompt_sha256", "generation_config_sha256"):
                if record[field] != (item["prompt_sha256"] if field == "prompt_sha256" else root[field]):
                    raise ContractError("model record frozen identity differs")
            expected_ids = [delivery["delivered_image_id"] for delivery in item["image_deliveries"]]
            expected_hashes = [delivery["delivered_rgb_sha256"] for delivery in item["image_deliveries"]]
            if record["ordered_delivered_image_ids"] != expected_ids or record["ordered_delivered_rgb_sha256"] != expected_hashes:
                raise ContractError("raw ordered multi-image metadata differs from plan")
            if status == "completed":
                if not isinstance(record["response"], str) or hashlib.sha256(record["response"].encode()).hexdigest() != record["response_utf8_sha256"]:
                    raise ContractError("completed response/hash is invalid")
                _finite(record["elapsed_seconds"], "elapsed seconds")
                if record["processor_telemetry"] is None:
                    raise ContractError("completed record lacks processor telemetry")
            else:
                if record["response"] is not None or not isinstance(record["error_type"], str):
                    raise ContractError("inference error response/error fields are invalid")
                if any(record[field] is None for field in ("elapsed_seconds", "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes", "processor_telemetry")) and not record["telemetry_unavailable_reason"]:
                    raise ContractError("unavailable inference telemetry requires an exact reason")
        elif status in {"pre_inference_rejected", "pre_inference_error"}:
            if item["execution_policy"] != "pre_inference_validation_only":
                raise ContractError("model item became a validation record")
            validation_indexes.append(_integer(record["validation_index"], "validation index"))
            if any(record[field] is not None for field in ("attempt_index", "response", "elapsed_seconds", "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes", "processor_telemetry")):
                raise ContractError("pre-inference record contains model/processor telemetry")
            if status == "pre_inference_rejected":
                for field in ("validation_stage", "rejection_code", "validator_receipt_sha256"):
                    if record[field] is None:
                        raise ContractError("required rejection receipt is incomplete")
                _digest(record["validator_receipt_sha256"], "validator receipt hash")
            elif record["error_type"] not in {"control_not_rejected", "validator_error", "process_crash_unmatched_validation_start"}:
                raise ContractError("pre-inference error code is invalid")
        else:
            if any(record[field] is not None for field in ("attempt_index", "validation_index", "response", "elapsed_seconds", "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes", "processor_telemetry")):
                raise ContractError("not-run record contains execution data")
            if record["not_run_reason"] != "fatal_run_terminated":
                raise ContractError("not-run reason is invalid")
            _digest(record["fatal_event_sha256"], "not-run fatal event hash")
    if model_indexes != list(range(len(model_indexes))) or validation_indexes != list(range(len(validation_indexes))):
        raise ContractError("raw attempt/validation indexes are not contiguous")
    recomputed = Counter(record["record_status"] for record in records)
    recomputed_counts = {
        "completed_count": recomputed["completed"],
        "inference_error_count": recomputed["inference_error"],
        "pre_inference_rejected_count": recomputed["pre_inference_rejected"],
        "pre_inference_error_count": recomputed["pre_inference_error"],
        "not_run_after_fatal_count": recomputed["not_run_after_fatal"],
        "model_not_run_after_fatal_count": sum(
            record["record_status"] == "not_run_after_fatal"
            and record["execution_policy"] == "model_attempt"
            for record in records
        ),
        "validation_not_run_after_fatal_count": sum(
            record["record_status"] == "not_run_after_fatal"
            and record["execution_policy"] == "pre_inference_validation_only"
            for record in records
        ),
    }
    if any(root[field] != count for field, count in recomputed_counts.items()):
        raise ContractError("raw root counters differ from recomputed record statuses")
    terminal_events = {
        (event["item_id"], event["event_type"]): event
        for event in checked_ledger
        if event["event_type"] in {"validation_terminal", "model_attempt_terminal"}
    }
    started_events = {
        (event["item_id"], event["event_type"]): event
        for event in checked_ledger
        if event["event_type"] in {"validation_started", "model_attempt_started"}
    }
    fatal_hashes = {
        event["event_sha256"] for event in checked_ledger if event["event_type"] == "fatal"
    }
    for record in records:
        item_id = record["item_id"]
        policy = record["execution_policy"]
        status = record["record_status"]
        start_type = "model_attempt_started" if policy == "model_attempt" else "validation_started"
        terminal_type = "model_attempt_terminal" if policy == "model_attempt" else "validation_terminal"
        if status in {"completed", "pre_inference_rejected"}:
            terminal = terminal_events.get((item_id, terminal_type))
            if terminal is None or terminal["payload_sha256"] != canonical_sha256(record):
                raise ContractError("raw terminal record is not hash-bound by the ledger")
        elif status in {"inference_error", "pre_inference_error"}:
            if str(record.get("error_type", "")).startswith("process_crash_unmatched_"):
                if (item_id, start_type) not in started_events or not fatal_hashes:
                    raise ContractError("crash-derived error lacks unmatched start/fatal ledger evidence")
            else:
                terminal = terminal_events.get((item_id, terminal_type))
                if terminal is None or terminal["payload_sha256"] != canonical_sha256(record):
                    raise ContractError("raw error record is not hash-bound by the ledger")
        elif status == "not_run_after_fatal":
            if record["fatal_event_sha256"] not in fatal_hashes:
                raise ContractError("not-run record fatal hash is absent from ledger")
    if root["run_status"] == "complete" and (
        root["pre_inference_error_count"] != 0
        or root["not_run_after_fatal_count"] != 0
        or root["pre_inference_rejected_count"] != root["pre_inference_validation_planned_count"]
        or bool(fatal_hashes)
    ):
        raise ContractError("complete raw artifact violates validation/fatal rules")
    if root["run_status"] == "fatal_incomplete" and not fatal_hashes:
        raise ContractError("fatal_incomplete raw artifact lacks fatal ledger evidence")
    return dict(root)


def validate_score(
    value: object,
    *,
    plan: object,
    locators: object,
    oracle: object,
    raw: object,
    freeze_receipt: object,
    accepted_provenance_by_source_item: Mapping[str, object],
    ledger_events: Sequence[Mapping[str, Any]],
    expected_freeze_bindings: Mapping[str, object],
) -> dict[str, Any]:
    fields = {
        "schema_version", "plan_sha256", "locator_sha256", "oracle_sha256", "raw_sha256",
        "freeze_receipt_sha256", "method_id", "raw_metrics", "final_metrics",
        "grounding", "adoption", "items",
    }
    root = _object(value, fields, "score artifact")
    if root["schema_version"] != SCORE_SCHEMA:
        raise ContractError(f"score schema must be {SCORE_SCHEMA}")
    validate_plan(
        plan,
        locators=locators,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    validate_image_locators(locators)
    validate_oracle(
        oracle,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    validate_freeze_receipt(
        freeze_receipt,
        plan,
        locators=locators,
        oracle=oracle,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected_freeze_bindings,
    )
    validate_raw(
        raw,
        plan,
        freeze_receipt,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        ledger_events=ledger_events,
        expected_freeze_bindings=expected_freeze_bindings,
    )
    expected = {
        "plan_sha256": canonical_sha256(plan), "locator_sha256": canonical_sha256(locators),
        "oracle_sha256": canonical_sha256(oracle), "raw_sha256": canonical_sha256(raw),
        "freeze_receipt_sha256": canonical_sha256(freeze_receipt),
    }
    if any(root[field] != digest for field, digest in expected.items()):
        raise ContractError("score input artifact hash mismatch")
    _text(root["method_id"], "score method ID")
    for field in ("raw_metrics", "final_metrics", "grounding", "adoption"):
        if not isinstance(root[field], Mapping):
            raise ContractError(f"score {field} must be an object")
    if not isinstance(root["items"], list) or len(root["items"]) != len(plan["items"]):
        raise ContractError("score must retain every plan item")
    return dict(root)


def nearest_rank_summary(values: Sequence[float]) -> dict[str, Any]:
    checked = sorted(_finite(value, "summary value") for value in values)
    if not checked:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    p50 = checked[math.ceil(0.50 * len(checked)) - 1]
    p95 = checked[math.ceil(0.95 * len(checked)) - 1]
    return {"count": len(checked), "min": checked[0], "p50": p50, "p95": p95, "max": checked[-1]}

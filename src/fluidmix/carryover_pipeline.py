"""One-pass production scaffold for camera/recipe carryover evidence.

The default path deliberately has no model loader.  Until a reviewed RGB-only mask
runtime is injected as a live capability, every vial is sealed as insufficient evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Protocol, runtime_checkable
import uuid

from .evidence_store import (
    SCHEMA_VERSION as EVIDENCE_STORE_SCHEMA_VERSION,
    EvidenceStore,
    HistoryQuery,
    IngestResult,
    assert_history_citation,
)
from .part_b_evidence import (
    bind_part_b_measurements_from_adapter_v1,
    project_part_b_model_evidence_v1,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TANKS = ("t100", "t200", "t400")
_VALIDATION_FIELDS = {
    "schema_version",
    "artifact_type",
    "split_role",
    "pipeline_source_commit",
    "pipeline_source_sha256",
    "recipe_schema_sha256",
    "mask_runtime",
    "measurement",
    "decision",
    "explanation",
    "store",
    "validation_denominators",
}
_FREEZE_FIELDS = (_VALIDATION_FIELDS - {"schema_version", "artifact_type", "split_role"}) | {
    "schema_version",
    "artifact_type",
    "validation_results_sha256",
    "freeze_sha256",
}
_RECIPE_FIELDS = {"schema_version", "split_role", "split_version", "runs"}
_RUN_FIELDS = {"run_id", "vials"}
_VIAL_FIELDS = {
    "case_id",
    "vial_ordinal",
    "block_recipe_id",
    "block_position",
    "recipe_id",
    "previous_recipe_id",
    "programmed_tank_ml",
    "dispense_order",
}
_FORBIDDEN_INPUT_TOKENS = frozenset(
    {
        "alpha",
        "colour",
        "color",
        "expected",
        "host",
        "label",
        "mask",
        "oracle",
        "palette",
        "residue",
        "simulation",
        "truth",
    }
)
_MASK_PROVENANCE_FIELDS = {
    "schema_version",
    "artifact_type",
    "source_class",
    "split_role",
    "source_rgb_sha256",
    "mask_sha256",
    "runtime_id",
    "runtime_sha256",
    "config_sha256",
    "validation_receipt_sha256",
    "accepts_real_rgb_only",
}
_EVALUATION_STORE_PROVENANCE_FIELDS = {
    "schema_version",
    "artifact_type",
    "store_purpose",
    "store_schema_version",
    "store_schema_sha256",
    "store_snapshot_sha256",
    "review_receipt_sha256",
    "stage_row_count",
}
_HISTORY_STORE_PROVENANCE_FIELDS = {
    "schema_version",
    "artifact_type",
    "store_purpose",
    "store_schema_version",
    "store_schema_sha256",
    "store_snapshot_sha256",
    "approval_receipt_sha256",
    "source_classes",
    "query_sha256s",
}
_ROUTE_DECISION_FIELDS = {
    "schema_version",
    "artifact_type",
    "issue_id",
    "scope",
    "approved_mode",
    "decision_status",
    "approver_name",
    "approver_handle",
}
_FORBIDDEN_HISTORY_TOKENS = frozenset({"evaluation", "oracle", "sim_metadata"})


@runtime_checkable
class PredictedMaskRuntime(Protocol):
    """Pluggable live RGB-only runtime boundary; no implementation is selected here."""

    runtime_id: str
    runtime_sha256: str
    config_sha256: str
    accepts_real_rgb_only: bool

    def analyze(
        self,
        *,
        video_bytes: bytes,
        recipe_sequence: Mapping[str, object],
        artifact_root: Path,
    ) -> "RuntimeProjectionInput":
        """Return live measurement capabilities and evidence-record templates."""


@dataclass(frozen=True)
class RuntimeProjectionInput:
    """Future integration value containing live capabilities, never saved locator JSON."""

    record_templates: tuple[dict[str, object], ...]
    trusted_adapter_result: object
    image_locators: Mapping[str, Mapping[str, object]]
    event_cursors: Mapping[str, int]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


RECIPE_SEQUENCE_SCHEMA_SHA256 = _canonical_hash(
    {
        "schema_version": "carryover-recipe-sequence-v1",
        "top_fields": sorted(_RECIPE_FIELDS),
        "run_fields": sorted(_RUN_FIELDS),
        "vial_fields": sorted(_VIAL_FIELDS),
        "tank_fields": list(_TANKS),
        "split_roles": ["real_check", "test"],
        "forbidden_tokens": sorted(_FORBIDDEN_INPUT_TOKENS),
    }
)


def pipeline_source_sha256() -> str:
    """Hash the exact production module and thin CLI bytes used by this scaffold."""

    repository_root = Path(__file__).resolve().parents[2]
    rows = []
    for relative in (
        "scripts/run_carryover_pipeline.py",
        "src/fluidmix/carryover_pipeline.py",
    ):
        raw = _read_regular_bytes(repository_root / relative, "pipeline source")
        rows.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
    return _canonical_hash(rows)


def _exact_fields(value: object, expected: set[str], role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{role} fields are invalid")
    actual = set(value)
    if actual != expected:
        extras = sorted(actual - expected)
        missing = sorted(expected - actual)
        detail = extras[0] if extras else missing[0]
        raise ValueError(f"{role} fields are invalid: {detail}")
    return value


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{role} must be nonempty bounded text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{role} contains control characters")
    return value


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} must be lowercase SHA-256")
    return value


def _nonnegative_int(value: object, role: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{role} must be a non-negative integer")
    return value


def _read_strict_json_bytes(raw: bytes, role: str) -> object:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} has duplicate field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} must be strict UTF-8 JSON") from error


def _read_regular_bytes(path: Path, role: str) -> bytes:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    return candidate.read_bytes()


def _read_production_bytes(path: Path, role: str) -> bytes:
    """Single auditable read seam for video and programmed recipe bytes."""

    return _read_regular_bytes(Path(path), role)


def _publish_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Atomically publish complete bytes without replacing an existing path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_json_exclusive(path: Path, value: object) -> None:
    _publish_bytes_exclusive(Path(path), _canonical_bytes(value) + b"\n")


def _append_journal(path: Path, event: Mapping[str, object]) -> None:
    payload = _canonical_bytes(event) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _start_journal(path: Path, event: Mapping[str, object]) -> None:
    _publish_bytes_exclusive(path, _canonical_bytes(event) + b"\n")


def _validate_validation_results(value: object) -> dict[str, object]:
    result = dict(_exact_fields(value, _VALIDATION_FIELDS, "validation result"))
    if result["schema_version"] != "carryover-validation-results-v1":
        raise ValueError("validation result schema_version is invalid")
    if result["artifact_type"] != "carryover_validation_results":
        raise ValueError("validation result artifact_type is invalid")
    if result["split_role"] != "validation":
        raise ValueError("freeze input must be validation-only")
    commit = result["pipeline_source_commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("pipeline_source_commit must be a lowercase Git commit")
    if (
        _sha(result["pipeline_source_sha256"], "pipeline_source_sha256")
        != pipeline_source_sha256()
    ):
        raise ValueError("pipeline source SHA-256 does not match the running source")
    if (
        _sha(result["recipe_schema_sha256"], "recipe_schema_sha256")
        != RECIPE_SEQUENCE_SCHEMA_SHA256
    ):
        raise ValueError("recipe schema SHA-256 does not match the production parser")

    mask = _exact_fields(
        result["mask_runtime"],
        {"runtime_id", "runtime_sha256", "config_sha256", "accepts_real_rgb_only"},
        "mask runtime",
    )
    _text(mask["runtime_id"], "mask runtime_id")
    _sha(mask["runtime_sha256"], "mask runtime_sha256")
    _sha(mask["config_sha256"], "mask config_sha256")
    if type(mask["accepts_real_rgb_only"]) is not bool:
        raise ValueError("accepts_real_rgb_only must be boolean")

    measurement = _exact_fields(
        result["measurement"], {"contract_id", "contract_sha256"}, "measurement"
    )
    _text(measurement["contract_id"], "measurement contract_id")
    _sha(measurement["contract_sha256"], "measurement contract_sha256")

    decision = _exact_fields(
        result["decision"],
        {"config", "config_sha256", "dp5_choice", "dp5_passed_without_more_errors"},
        "decision",
    )
    config = _exact_fields(
        decision["config"],
        {"discard_threshold_delta_e76", "keep_threshold_delta_e76"},
        "decision config",
    )
    if config != {
        "discard_threshold_delta_e76": 2.0,
        "keep_threshold_delta_e76": 1.0,
    }:
        raise ValueError("decision config must freeze the accepted 2.0/1.0 thresholds")
    if _sha(decision["config_sha256"], "decision config_sha256") != _canonical_hash(config):
        raise ValueError("decision config SHA-256 mismatch")
    if decision["dp5_choice"] not in {"baseline", "history_anchored"}:
        raise ValueError("dp5_choice is invalid")
    if type(decision["dp5_passed_without_more_errors"]) is not bool:
        raise ValueError("dp5_passed_without_more_errors must be boolean")
    if (
        decision["dp5_choice"] == "history_anchored"
        and decision["dp5_passed_without_more_errors"] is not True
    ):
        raise ValueError("history_anchored requires the frozen DP5 pass")

    explanation = _exact_fields(
        result["explanation"],
        {
            "method",
            "method_sha256",
            "model_id",
            "model_sha256",
            "prompt_sha256",
            "guard_sha256",
        },
        "explanation",
    )
    for field in ("method", "model_id"):
        _text(explanation[field], f"explanation {field}")
    for field in ("method_sha256", "model_sha256", "prompt_sha256", "guard_sha256"):
        _sha(explanation[field], f"explanation {field}")
    if (
        explanation["method"] != "deterministic_guarded_template"
        or explanation["model_id"] != "none"
    ):
        raise ValueError(
            "explanation must select the deterministic DP4/default with no model"
        )

    store = _exact_fields(result["store"], {"schema_version", "schema_sha256"}, "store")
    if type(store["schema_version"]) is not int or store["schema_version"] <= 0:
        raise ValueError("store schema_version must be positive integer")
    _sha(store["schema_sha256"], "store schema_sha256")

    denominators = _exact_fields(
        result["validation_denominators"],
        {
            "run_count",
            "vial_count",
            "predicted_stage_count",
            "measured_stage_count",
            "history_supported_vial_count",
            "explained_vial_count",
        },
        "validation denominators",
    )
    for field, item in denominators.items():
        _nonnegative_int(item, f"validation denominators {field}")
    if denominators["measured_stage_count"] > denominators["predicted_stage_count"]:
        raise ValueError("measured stage denominator exceeds predicted stage denominator")
    for field in ("history_supported_vial_count", "explained_vial_count"):
        if denominators[field] > denominators["vial_count"]:
            raise ValueError(f"{field} exceeds validation vial denominator")
    return result


def freeze_validation(validation_results: Path, out: Path) -> dict[str, object]:
    """Freeze one validation-only receipt into a canonical no-clobber binding."""

    raw = _read_regular_bytes(Path(validation_results), "validation result")
    validated = _validate_validation_results(
        _read_strict_json_bytes(raw, "validation result")
    )
    receipt = {
        "schema_version": "carryover-freeze-v1",
        "artifact_type": "carryover_pipeline_freeze",
        "pipeline_source_commit": validated["pipeline_source_commit"],
        "pipeline_source_sha256": validated["pipeline_source_sha256"],
        "recipe_schema_sha256": validated["recipe_schema_sha256"],
        "mask_runtime": validated["mask_runtime"],
        "measurement": validated["measurement"],
        "decision": validated["decision"],
        "explanation": validated["explanation"],
        "store": validated["store"],
        "validation_denominators": validated["validation_denominators"],
        "validation_results_sha256": hashlib.sha256(raw).hexdigest(),
    }
    receipt["freeze_sha256"] = _canonical_hash(receipt)
    _publish_json_exclusive(Path(out), receipt)
    return receipt


def _load_freeze(path: Path, expected_sha256: str) -> dict[str, object]:
    _sha(expected_sha256, "expected freeze SHA-256")
    raw = _read_regular_bytes(Path(path), "freeze receipt")
    value = _read_strict_json_bytes(raw, "freeze receipt")
    receipt = dict(_exact_fields(value, _FREEZE_FIELDS, "freeze receipt"))
    if receipt["schema_version"] != "carryover-freeze-v1" or receipt[
        "artifact_type"
    ] != "carryover_pipeline_freeze":
        raise ValueError("freeze receipt identity is invalid")
    embedded = _sha(receipt["freeze_sha256"], "freeze SHA-256")
    observed = _canonical_hash(
        {key: item for key, item in receipt.items() if key != "freeze_sha256"}
    )
    if embedded != observed or embedded != expected_sha256:
        raise ValueError("freeze SHA-256 mismatch")
    validation_shape = {
        **{key: receipt[key] for key in _VALIDATION_FIELDS - {"schema_version", "artifact_type", "split_role"}},
        "schema_version": "carryover-validation-results-v1",
        "artifact_type": "carryover_validation_results",
        "split_role": "validation",
    }
    _validate_validation_results(validation_shape)
    return receipt


def _reject_forbidden_vocabulary(value: object, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("recipe sequence fields must be strings")
            tokens = set(re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_").split("_"))
            forbidden = sorted(tokens & _FORBIDDEN_INPUT_TOKENS)
            if forbidden:
                raise ValueError(f"forbidden recipe sequence field: {path}{key}")
            _reject_forbidden_vocabulary(child, f"{path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_vocabulary(child, f"{path}{index}.")
    elif isinstance(value, str):
        tokens = set(
            re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_").split("_")
        )
        forbidden = sorted(tokens & _FORBIDDEN_INPUT_TOKENS)
        if forbidden:
            raise ValueError(f"forbidden recipe sequence value at {path.rstrip('.')}")


def _validate_recipe_sequence(value: object, expected_role: str) -> dict[str, object]:
    _reject_forbidden_vocabulary(value)
    sequence = dict(_exact_fields(value, _RECIPE_FIELDS, "recipe sequence"))
    if sequence["schema_version"] != "carryover-recipe-sequence-v1":
        raise ValueError("recipe sequence schema_version is invalid")
    if sequence["split_role"] != expected_role:
        raise ValueError("recipe sequence split_role mismatch")
    _text(sequence["split_version"], "recipe split_version")
    runs = sequence["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("recipe sequence runs must be a nonempty list")
    run_ids: set[str] = set()
    case_ids: set[tuple[str, str]] = set()
    for run_index, raw_run in enumerate(runs):
        run = _exact_fields(raw_run, _RUN_FIELDS, f"recipe run {run_index}")
        run_id = _text(run["run_id"], "run_id")
        if run_id in run_ids:
            raise ValueError("recipe sequence run_id values must be unique")
        run_ids.add(run_id)
        vials = run["vials"]
        if not isinstance(vials, list) or not vials:
            raise ValueError("recipe run vials must be a nonempty list")
        for expected_ordinal, raw_vial in enumerate(vials, start=1):
            vial = _exact_fields(raw_vial, _VIAL_FIELDS, "recipe vial")
            case_id = _text(vial["case_id"], "case_id")
            if (run_id, case_id) in case_ids:
                raise ValueError("case_id values must be unique within a run")
            case_ids.add((run_id, case_id))
            if vial["vial_ordinal"] != expected_ordinal:
                raise ValueError("vial_ordinal must be contiguous from one")
            if type(vial["block_position"]) is not int or vial["block_position"] <= 0:
                raise ValueError("block_position must be a positive integer")
            for field in ("block_recipe_id", "recipe_id"):
                _text(vial[field], field)
            if vial["previous_recipe_id"] is not None:
                _text(vial["previous_recipe_id"], "previous_recipe_id")
            volumes = _exact_fields(vial["programmed_tank_ml"], set(_TANKS), "programmed tank mL")
            active: list[str] = []
            for tank in _TANKS:
                amount = volumes[tank]
                if type(amount) not in {int, float} or not math.isfinite(float(amount)) or amount < 0:
                    raise ValueError("programmed tank mL must be finite and non-negative")
                if amount > 0:
                    active.append(tank)
            order = vial["dispense_order"]
            if (
                not isinstance(order, list)
                or any(type(item) is not str for item in order)
                or len(order) != len(set(order))
                or set(order) != set(active)
            ):
                raise ValueError("dispense_order must exactly cover positive programmed tanks")
    return sequence


def load_recipe_sequence(path: Path, *, expected_role: str) -> dict[str, object]:
    raw = _read_production_bytes(Path(path).resolve(), "recipe sequence")
    return _validate_recipe_sequence(
        _read_strict_json_bytes(raw, "recipe sequence"), expected_role
    )


def load_preflight_receipt(path: Path) -> object:
    """Load strict preflight JSON; execution performs the hash/mode validation."""

    raw = _read_regular_bytes(Path(path), "preflight receipt")
    return _read_strict_json_bytes(raw, "preflight receipt")


def project_runtime_measurements(
    runtime_output: RuntimeProjectionInput,
) -> list[dict[str, object]]:
    """Validate a future runtime through the accepted live Part-B capability seams."""

    if not isinstance(runtime_output, RuntimeProjectionInput):
        raise TypeError("runtime output must be RuntimeProjectionInput")
    bound = bind_part_b_measurements_from_adapter_v1(
        runtime_output.record_templates,
        runtime_output.trusted_adapter_result,
    )
    grouped: dict[tuple[object, object], list[dict[str, object]]] = {}
    for record in bound:
        grouped.setdefault((record.get("run_id"), record.get("case_id")), []).append(record)
    decoded: list[dict[str, object]] = []
    for records in grouped.values():
        encoded = project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=runtime_output.trusted_adapter_result,
            measurement_bindings=runtime_output.trusted_adapter_result,
        )
        projected = json.loads(encoded)
        assert isinstance(projected, list)
        decoded.extend(projected)
    return decoded


def append_runtime_measurements(
    *,
    store: Path,
    artifact_root: Path,
    records: Sequence[Mapping[str, object]],
    runtime_output: RuntimeProjectionInput,
) -> IngestResult:
    """Append already projected live records through the accepted EvidenceStore API."""

    evidence_store = EvidenceStore(store, artifact_root=artifact_root)
    return evidence_store.ingest(
        records,
        runtime_output.image_locators,
        phase="measured",
        event_cursors=runtime_output.event_cursors,
    )


def _validate_predicted_mask_provenance(value: object) -> dict[str, object]:
    if value is None:
        raise ValueError("predicted-mask provenance is required for history_backed mode")
    provenance = dict(
        _exact_fields(value, _MASK_PROVENANCE_FIELDS, "predicted-mask provenance")
    )
    expected_text = {
        "schema_version": "wp7-predicted-mask-provenance-v1",
        "artifact_type": "predicted_mask_provenance",
        "source_class": "reviewed_rgb_only_prediction",
        "split_role": "test_and_real_check",
    }
    for field, expected in expected_text.items():
        if provenance[field] != expected:
            raise ValueError(f"predicted-mask provenance {field} is invalid")
    _text(provenance["runtime_id"], "predicted-mask provenance runtime_id")
    for field in (
        "source_rgb_sha256",
        "mask_sha256",
        "runtime_sha256",
        "config_sha256",
        "validation_receipt_sha256",
    ):
        _sha(provenance[field], f"predicted-mask provenance {field}")
    if provenance["accepts_real_rgb_only"] is not True:
        raise ValueError(
            "predicted-mask provenance accepts_real_rgb_only must be true"
        )
    return provenance


def _validate_evaluation_store_provenance(value: object) -> dict[str, object]:
    if value is None:
        raise ValueError("evaluation-only store provenance is required")
    provenance = dict(
        _exact_fields(
            value,
            _EVALUATION_STORE_PROVENANCE_FIELDS,
            "evaluation-only store provenance",
        )
    )
    expected_text = {
        "schema_version": "wp7-evaluation-only-store-v1",
        "artifact_type": "evaluation_only_store_provenance",
        "store_purpose": "evaluation_only",
    }
    for field, expected in expected_text.items():
        if provenance[field] != expected:
            raise ValueError(f"evaluation-only store {field} is invalid")
    if provenance["store_schema_version"] != EVIDENCE_STORE_SCHEMA_VERSION or type(
        provenance["store_schema_version"]
    ) is not int:
        raise ValueError("evaluation-only store_schema_version is not the accepted WP5 schema")
    for field in (
        "store_schema_sha256",
        "store_snapshot_sha256",
        "review_receipt_sha256",
    ):
        _sha(provenance[field], f"evaluation-only {field}")
    if provenance["stage_row_count"] != 0 or type(
        provenance["stage_row_count"]
    ) is not int:
        raise ValueError("evaluation-only stage_row_count must equal zero")
    return provenance


def _validate_history_store_provenance(value: object) -> dict[str, object]:
    if value is None:
        raise ValueError("production-history store provenance is required")
    provenance = dict(
        _exact_fields(
            value,
            _HISTORY_STORE_PROVENANCE_FIELDS,
            "production-history store provenance",
        )
    )
    expected_text = {
        "schema_version": "wp7-production-history-store-v1",
        "artifact_type": "production_history_store_provenance",
        "store_purpose": "production_history",
    }
    for field, expected in expected_text.items():
        if provenance[field] != expected:
            raise ValueError(f"production-history store {field} is invalid")
    if provenance["store_schema_version"] != EVIDENCE_STORE_SCHEMA_VERSION or type(
        provenance["store_schema_version"]
    ) is not int:
        raise ValueError("production-history store_schema_version is not the accepted WP5 schema")
    for field in (
        "store_schema_sha256",
        "store_snapshot_sha256",
        "approval_receipt_sha256",
    ):
        _sha(provenance[field], f"production-history {field}")
    if provenance["source_classes"] != ["camera_rgb", "predicted_mask"]:
        raise ValueError(
            "production-history source_classes must be camera_rgb and predicted_mask only"
        )
    query_sha256s = provenance["query_sha256s"]
    if (
        not isinstance(query_sha256s, list)
        or not query_sha256s
        or len(query_sha256s) != len(set(query_sha256s))
    ):
        raise ValueError("production-history query_sha256s must be nonempty and unique")
    for digest in query_sha256s:
        _sha(digest, "production-history query_sha256s member")
    return provenance


def _history_path_has_forbidden_source(path: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")
    bounded = f"_{normalized}_"
    return any(f"_{token}_" in bounded for token in _FORBIDDEN_HISTORY_TOKENS)


def _validate_history_route_decision(
    value: object, expected_sha256: object
) -> dict[str, object]:
    decision = dict(
        _exact_fields(value, _ROUTE_DECISION_FIELDS, "history route-decision")
    )
    expected = {
        "schema_version": "wp7-route-decision-v1",
        "artifact_type": "wp7_route_decision",
        "issue_id": "INN-62",
        "scope": "section_6_history_backed_explanations",
        "approved_mode": "history_backed",
        "decision_status": "approved",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
    }
    if decision != expected:
        for field in sorted(_ROUTE_DECISION_FIELDS):
            if decision.get(field) != expected[field]:
                raise ValueError(f"history route-decision {field} is invalid")
        raise ValueError("history route-decision is invalid")
    declared = _sha(expected_sha256, "expected route-decision SHA-256")
    if _canonical_hash(decision) != declared:
        raise ValueError("route-decision SHA-256 mismatch")
    return decision


def preflight_explanation_route(
    *,
    requested_mode: str,
    expected_test_vials: int,
    expected_real_check_vials: int,
    evaluation_store_provenance: object | None = None,
    predicted_mask_provenance: object | None = None,
    history_store_provenance: object | None = None,
    history_queries: object | None = None,
    route_decision: object | None = None,
    expected_route_decision_sha256: object | None = None,
) -> dict[str, object]:
    """Select one WP7 explanation route without opening one-pass phase inputs.

    ``evaluation_only`` is the explicit zero-history path. ``history_backed`` accepts
    only an approved RGB-prediction receipt and nonempty, factory-bound WP5 history
    queries tied to a production-history store receipt. A failed history request raises;
    it can never downgrade itself to the evaluation-only path.
    """

    if requested_mode not in {"evaluation_only", "history_backed"}:
        raise ValueError("requested_mode must be evaluation_only or history_backed")
    if expected_test_vials != 18 or type(expected_test_vials) is not int:
        raise ValueError("expected_test_vials must equal the frozen denominator 18")
    if expected_real_check_vials != 30 or type(expected_real_check_vials) is not int:
        raise ValueError(
            "expected_real_check_vials must equal the frozen denominator 30"
        )

    common: dict[str, object] = {
        "schema_version": "wp7-explanation-route-preflight-v1",
        "requested_mode": requested_mode,
        "selected_mode": requested_mode,
        "fallback_used": False,
    }
    if requested_mode == "evaluation_only":
        if any(
            value is not None
            for value in (
                predicted_mask_provenance,
                history_store_provenance,
                history_queries,
                route_decision,
                expected_route_decision_sha256,
            )
        ):
            raise ValueError(
                "evaluation_only mode must not receive mask or history inputs"
            )
        store = _validate_evaluation_store_provenance(evaluation_store_provenance)
        receipt = {
            **common,
            "test": {"terminal_behavior": "abstain", "vial_count": 18},
            "real_check": {"terminal_behavior": "skip", "vial_count": 30},
            "activation_status": "explicit_evaluation_only",
            "history_backed_explanations_enabled": False,
            "history": None,
            "predicted_mask": None,
            "route_decision": None,
            "store": {
                "store_purpose": store["store_purpose"],
                "store_schema_version": store["store_schema_version"],
                "store_schema_sha256": store["store_schema_sha256"],
                "store_snapshot_sha256": store["store_snapshot_sha256"],
                "review_receipt_sha256": store["review_receipt_sha256"],
                "stage_row_count": store["stage_row_count"],
            },
        }
        receipt["preflight_sha256"] = _canonical_hash(receipt)
        return receipt

    if evaluation_store_provenance is not None:
        raise ValueError("history_backed mode must not receive an evaluation-only store")
    mask = _validate_predicted_mask_provenance(predicted_mask_provenance)
    store = _validate_history_store_provenance(history_store_provenance)
    if isinstance(history_queries, (str, bytes)) or not isinstance(
        history_queries, Sequence
    ):
        raise TypeError("history_queries must be a sequence of trusted HistoryQuery values")
    queries = tuple(history_queries)
    if not queries:
        raise ValueError("history_backed mode requires nonempty production-history evidence")

    query_sha256s: list[str] = []
    member_count = 0
    for query in queries:
        if not isinstance(query, HistoryQuery):
            raise TypeError("history_queries must contain trusted HistoryQuery values")
        if not query.members:
            raise ValueError(
                "history_backed mode requires nonempty production-history evidence"
            )
        first = query.members[0]
        assert_history_citation(query, first.evidence_id, first.image_id)
        for member in query.members:
            if member.split_role in {"evaluation", "test"}:
                raise ValueError(
                    "production-history query contains evaluation or test evidence"
                )
            if any(
                _history_path_has_forbidden_source(path)
                for path in (member.rgb_path, member.mask_path, member.clip_path)
            ):
                raise ValueError(
                    "production-history query contains forbidden source vocabulary"
                )
        query_sha256s.append(query.query_sha256)
        member_count += len(query.members)
    if query_sha256s != store["query_sha256s"]:
        raise ValueError(
            "production-history query_sha256s do not match the trusted queries"
        )

    if (route_decision is None) != (expected_route_decision_sha256 is None):
        raise ValueError(
            "history route-decision and expected SHA-256 must be provided together"
        )
    validated_decision = None
    decision_sha256 = None
    if route_decision is not None:
        validated_decision = _validate_history_route_decision(
            route_decision, expected_route_decision_sha256
        )
        decision_sha256 = str(expected_route_decision_sha256)
    enabled = validated_decision is not None

    receipt = {
        **common,
        "test": {
            "terminal_behavior": "history_backed_explanation",
            "vial_count": 18,
        },
        "real_check": {
            "terminal_behavior": "history_backed_explanation",
            "vial_count": 30,
        },
        "activation_status": (
            "history_backed_requirements_satisfied"
            if enabled
            else "pending_inn62_section6_approval"
        ),
        "history_backed_explanations_enabled": enabled,
        "route_decision": (
            {
                "issue_id": validated_decision["issue_id"],
                "scope": validated_decision["scope"],
                "approved_mode": validated_decision["approved_mode"],
                "approver_name": validated_decision["approver_name"],
                "approver_handle": validated_decision["approver_handle"],
                "route_decision_sha256": decision_sha256,
            }
            if validated_decision is not None
            else None
        ),
        "predicted_mask": {
            "source_class": mask["source_class"],
            "source_rgb_sha256": mask["source_rgb_sha256"],
            "mask_sha256": mask["mask_sha256"],
            "runtime_id": mask["runtime_id"],
            "runtime_sha256": mask["runtime_sha256"],
            "config_sha256": mask["config_sha256"],
            "validation_receipt_sha256": mask["validation_receipt_sha256"],
        },
        "history": {
            "store_purpose": store["store_purpose"],
            "store_schema_version": store["store_schema_version"],
            "store_schema_sha256": store["store_schema_sha256"],
            "store_snapshot_sha256": store["store_snapshot_sha256"],
            "approval_receipt_sha256": store["approval_receipt_sha256"],
            "query_sha256s": query_sha256s,
            "query_count": len(queries),
            "member_count": member_count,
        },
    }
    receipt["preflight_sha256"] = _canonical_hash(receipt)
    return receipt


def _validate_execution_preflight(
    value: object, *, expected_sha256: str, selected_mode: str
) -> dict[str, object]:
    if selected_mode not in {"evaluation_only", "history_backed"}:
        raise ValueError("selected mode must be evaluation_only or history_backed")
    expected_digest = _sha(expected_sha256, "expected preflight SHA-256")
    if not isinstance(value, Mapping) or (
        value.get("requested_mode") != selected_mode
        or value.get("selected_mode") != selected_mode
    ):
        raise ValueError("preflight receipt selected mode mismatch")
    common_fields = {
        "schema_version",
        "requested_mode",
        "selected_mode",
        "fallback_used",
        "test",
        "real_check",
        "activation_status",
        "history_backed_explanations_enabled",
        "history",
        "predicted_mask",
        "route_decision",
        "preflight_sha256",
    }
    fields = common_fields | ({"store"} if selected_mode == "evaluation_only" else set())
    receipt = dict(_exact_fields(value, fields, "preflight receipt"))
    if receipt["schema_version"] != "wp7-explanation-route-preflight-v1":
        raise ValueError("preflight receipt schema_version is invalid")
    if (
        receipt["requested_mode"] != selected_mode
        or receipt["selected_mode"] != selected_mode
    ):
        raise ValueError("preflight receipt selected mode mismatch")
    if receipt["fallback_used"] is not False:
        raise ValueError("preflight receipt fallback_used must be false")
    embedded = _sha(receipt["preflight_sha256"], "preflight SHA-256")
    observed = _canonical_hash(
        {key: item for key, item in receipt.items() if key != "preflight_sha256"}
    )
    if embedded != observed or embedded != expected_digest:
        raise ValueError("preflight SHA-256 mismatch")

    if selected_mode == "evaluation_only":
        if receipt["test"] != {"terminal_behavior": "abstain", "vial_count": 18}:
            raise ValueError("evaluation-only test semantics are invalid")
        if receipt["real_check"] != {"terminal_behavior": "skip", "vial_count": 30}:
            raise ValueError("evaluation-only real-check semantics are invalid")
        if (
            receipt["activation_status"] != "explicit_evaluation_only"
            or receipt["history_backed_explanations_enabled"] is not False
            or receipt["history"] is not None
            or receipt["predicted_mask"] is not None
            or receipt["route_decision"] is not None
        ):
            raise ValueError("evaluation-only preflight state is invalid")
        store = _exact_fields(
            receipt["store"],
            {
                "store_purpose",
                "store_schema_version",
                "store_schema_sha256",
                "store_snapshot_sha256",
                "review_receipt_sha256",
                "stage_row_count",
            },
            "evaluation-only preflight store",
        )
        if (
            store["store_purpose"] != "evaluation_only"
            or store["store_schema_version"] != EVIDENCE_STORE_SCHEMA_VERSION
            or type(store["store_schema_version"]) is not int
            or store["stage_row_count"] != 0
            or type(store["stage_row_count"]) is not int
        ):
            raise ValueError("evaluation-only preflight store is invalid")
        for field in (
            "store_schema_sha256",
            "store_snapshot_sha256",
            "review_receipt_sha256",
        ):
            _sha(store[field], f"evaluation-only preflight {field}")
        return receipt

    if receipt["test"] != {
        "terminal_behavior": "history_backed_explanation",
        "vial_count": 18,
    } or receipt["real_check"] != {
        "terminal_behavior": "history_backed_explanation",
        "vial_count": 30,
    }:
        raise ValueError("history-backed terminal semantics are invalid")
    if (
        receipt["activation_status"] != "history_backed_requirements_satisfied"
        or receipt["history_backed_explanations_enabled"] is not True
    ):
        raise ValueError("history-backed preflight is pending approval")
    decision = _exact_fields(
        receipt["route_decision"],
        {
            "issue_id",
            "scope",
            "approved_mode",
            "approver_name",
            "approver_handle",
            "route_decision_sha256",
        },
        "history route-decision binding",
    )
    expected_decision = {
        "issue_id": "INN-62",
        "scope": "section_6_history_backed_explanations",
        "approved_mode": "history_backed",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
    }
    if any(decision[field] != expected for field, expected in expected_decision.items()):
        raise ValueError("history route-decision binding is invalid")
    _sha(decision["route_decision_sha256"], "history route-decision SHA-256")
    predicted = _exact_fields(
        receipt["predicted_mask"],
        {
            "source_class",
            "source_rgb_sha256",
            "mask_sha256",
            "runtime_id",
            "runtime_sha256",
            "config_sha256",
            "validation_receipt_sha256",
        },
        "history-backed predicted-mask binding",
    )
    if predicted["source_class"] != "reviewed_rgb_only_prediction":
        raise ValueError("history-backed predicted-mask source_class is invalid")
    _text(predicted["runtime_id"], "history-backed runtime_id")
    for field in (
        "source_rgb_sha256",
        "mask_sha256",
        "runtime_sha256",
        "config_sha256",
        "validation_receipt_sha256",
    ):
        _sha(predicted[field], f"history-backed predicted-mask {field}")
    history = _exact_fields(
        receipt["history"],
        {
            "store_purpose",
            "store_schema_version",
            "store_schema_sha256",
            "store_snapshot_sha256",
            "approval_receipt_sha256",
            "query_sha256s",
            "query_count",
            "member_count",
        },
        "history-backed store binding",
    )
    if (
        history["store_purpose"] != "production_history"
        or history["store_schema_version"] != EVIDENCE_STORE_SCHEMA_VERSION
        or type(history["store_schema_version"]) is not int
    ):
        raise ValueError("history-backed store identity is invalid")
    for field in (
        "store_schema_sha256",
        "store_snapshot_sha256",
        "approval_receipt_sha256",
    ):
        _sha(history[field], f"history-backed store {field}")
    query_sha256s = history["query_sha256s"]
    if (
        not isinstance(query_sha256s, list)
        or not query_sha256s
        or len(query_sha256s) != len(set(query_sha256s))
    ):
        raise ValueError("history-backed query SHA-256 bindings are invalid")
    for digest in query_sha256s:
        _sha(digest, "history-backed query SHA-256")
    if (
        type(history["query_count"]) is not int
        or history["query_count"] != len(query_sha256s)
        or type(history["member_count"]) is not int
        or history["member_count"] <= 0
    ):
        raise ValueError("history-backed query/member counts are invalid")
    return receipt


def _claim_payload(
    *,
    phase: str,
    owner_id: str,
    freeze_sha256: str,
    video: Path,
    recipe_sequence: Path,
    store: Path,
    out: Path,
    preflight_sha256: str,
    selected_mode: str,
) -> dict[str, object]:
    _text(owner_id, "claim owner_id")
    payload = {
        "schema_version": "carryover-one-pass-claim-v1",
        "phase": phase,
        "owner_id": owner_id,
        "freeze_sha256": freeze_sha256,
        "video_path": str(Path(video).resolve()),
        "recipe_sequence_path": str(Path(recipe_sequence).resolve()),
        "store_path": str(Path(store).resolve()),
        "out_path": str(Path(out).resolve()),
        "preflight_sha256": preflight_sha256,
        "selected_mode": selected_mode,
    }
    payload["claim_sha256"] = _canonical_hash(payload)
    return payload


def _vial_rows(sequence: Mapping[str, object]):
    runs = sequence["runs"]
    assert isinstance(runs, list)
    for run in runs:
        assert isinstance(run, Mapping)
        vials = run["vials"]
        assert isinstance(vials, list)
        for vial in vials:
            assert isinstance(vial, Mapping)
            yield run, vial


def wp7_explanation_policy() -> dict[str, object]:
    """Return the official DP4/default explanation policy and limited-run metadata.

    The learned-model rows describe only the completed all-abstention diagnostic.
    They intentionally contain no keep/discard accuracy or winner field value.
    """

    return {
        "schema_version": "wp7-explanation-policy-v1",
        "dp4_status": "official_default",
        "explanation_method": "deterministic_guarded_template",
        "limited_result": {
            "scope": "pre_access_all_abstention_diagnostic_only",
            "attempt_count_per_method": 16,
            "ordered_shuffled_pair_count": 8,
            "keep_discard_accuracy": None,
            "winner": None,
            "provenance": {
                "protocol_commit": "0ac00ad7906634caf08a51f088343c72c433d1c4",
                "protocol_sha256": "7c8325f40541ceae972aeb438e0d0ff9b6abc6df6a50bae50b815e975e1f8eb4",
                "plan_file_sha256": "ac0d8f1f779867cdb3fd77f12ccfa7d7d0c3baf48ecb2afcc35913d5cf5b19d2",
                "plan_canonical_sha256": "7e50db2c0c5d769664c032930b73f8472275e0296413dfba7c5dfa599799bf3d",
            },
            "methods": [
                {
                    "method": "deterministic_template",
                    "strict_parse_count": 16,
                    "correct_abstention_count": 16,
                    "unsupported_cause_refusal_count": 16,
                    "parsed_pair_count": 8,
                    "abstention_flip_count": 0,
                    "refusal_flip_count": 0,
                    "raw_sha256": "6824c8ee227ca21263283439301d18ece7f119363e7515f34621367abd2d97a8",
                    "score_sha256": "ce036c2af554c201d45f2a600bdde6e301f9566c76e7232324f04553722c5843",
                },
                {
                    "method": "Qwen3-VL-8B-Instruct",
                    "strict_parse_count": 16,
                    "correct_abstention_count": 16,
                    "unsupported_cause_refusal_count": 16,
                    "parsed_pair_count": 8,
                    "abstention_flip_count": 0,
                    "refusal_flip_count": 0,
                    "median_generation_seconds": 5.589,
                    "peak_cuda_allocated_bytes": 18_824_709_120,
                    "raw_sha256": "a61c7f5aea4f1b95e67f161c65574b25a0643194345f238045f5ba0cd7c713d7",
                    "score_sha256": "3cd9ccd37af3453063f4aed803bc0c35ee3e2db5832ad10ef6c107a6df9040f9",
                },
                {
                    "method": "AnomalyThink-Qwen3-VL-8B-KCR",
                    "strict_parse_count": 0,
                    "parsed_pair_count": 0,
                    "failure_mode": "free_form_think_and_yes_no",
                    "median_generation_seconds": 15.214,
                    "raw_sha256": "82247421129d1901d7a1b1b0f9b11e1cac99aec14bf074fbd370a4da80e49e24",
                    "score_sha256": "a0dfd8aa795f1c6e3e6aad4b7b5b3c3a12f8ec956a1e976f8fdec31bbea52be3",
                },
                {
                    "method": "Cosmos-Reason2-8B",
                    "strict_parse_count": 0,
                    "parsed_pair_count": 0,
                    "failure_mode": "json_in_markdown_fence_no_repair",
                    "median_generation_seconds": 7.619,
                    "raw_sha256": "0d27529df25105ddf2ec193234425301750b1f07a52da79b977c02740b3d448f",
                    "score_sha256": "3b986e50d1ec0a7140c9ecff54fee5f94b256234a715876e225625e8b70fd2d9",
                },
            ],
        },
        "full_comparison": {"issue_id": "INN-67", "timing": "Friday"},
    }


def _insufficient_record(
    *,
    run: Mapping[str, object],
    vial: Mapping[str, object],
    source_bindings: Mapping[str, str],
    reason_codes: Sequence[str],
) -> dict[str, object]:
    codes = sorted(set(reason_codes))
    return {
        "schema_version": "carryover-operator-record-v1",
        "run_id": run["run_id"],
        "case_id": vial["case_id"],
        "vial_ordinal": vial["vial_ordinal"],
        "recipe_id": vial["recipe_id"],
        "previous_recipe_id": vial["previous_recipe_id"],
        "disposition": "insufficient_evidence",
        "reason_codes": codes,
        "prediction": {"status": "insufficient_evidence", "evidence_ids": []},
        "measurement": {"status": "insufficient_evidence", "evidence_ids": []},
        "history": {"status": "insufficient_evidence", "query_sha256": None, "evidence_ids": []},
        "explanation": {
            "status": "insufficient_evidence",
            "method": "deterministic_guarded_template",
            "guard_result": "insufficient_evidence",
            "image_ids_sent": [],
            "text": "Insufficient camera evidence; no keep or discard recommendation is made.",
        },
        "source_bindings": dict(source_bindings),
    }


def _denominators(
    sequence: Mapping[str, object], *, abstentions: int, skips: int, decisions: int = 0
) -> dict[str, object]:
    runs = sequence["runs"]
    assert isinstance(runs, list)
    rows = list(_vial_rows(sequence))
    expected_stages = sum(len(vial["dispense_order"]) for _, vial in rows)
    terminal = decisions + abstentions + skips
    if terminal != len(rows):
        raise ValueError("terminal count equation does not equal the vial denominator")
    return {
        "run_count": len(runs),
        "vial_count": len(rows),
        "expected_stage_count": expected_stages,
        "terminal_record_count": terminal,
        "decision_count": decisions,
        "abstention_count": abstentions,
        "skip_count": skips,
        "coverage_numerator": decisions,
        "coverage_denominator": len(rows),
        "correct_decision_numerator": None,
        "correct_decision_denominator": None,
        "discard_recall_numerator": None,
        "discard_recall_denominator": None,
        "false_discard_count": None,
        "false_keep_count": None,
    }


def _fatal_denominators(
    sequence: Mapping[str, object] | None,
) -> dict[str, object]:
    if sequence is None:
        return {
            "run_count": None,
            "vial_count": None,
            "expected_stage_count": None,
            "terminal_record_count": 0,
            "decision_count": 0,
            "abstention_count": 0,
            "skip_count": 0,
            "unresolved_vial_count": None,
            "count_equation_status": "unavailable_before_recipe_validation",
        }
    runs = sequence["runs"]
    assert isinstance(runs, list)
    rows = list(_vial_rows(sequence))
    return {
        "run_count": len(runs),
        "vial_count": len(rows),
        "expected_stage_count": sum(
            len(vial["dispense_order"]) for _, vial in rows
        ),
        "terminal_record_count": 0,
        "decision_count": 0,
        "abstention_count": 0,
        "skip_count": 0,
        "unresolved_vial_count": len(rows),
        "count_equation_status": "balanced_with_unresolved_vials",
    }


def _occupied(path: Path) -> bool:
    candidate = Path(path)
    return candidate.exists() or candidate.is_symlink()


def _run_one_pass(
    *,
    phase: str,
    expected_role: str,
    video: Path,
    recipe_sequence: Path,
    freeze: Path,
    expected_freeze_sha256: str,
    store: Path,
    out: Path,
    claim: Path,
    owner_id: str,
    mask_runtime: PredictedMaskRuntime | None,
    preflight_receipt: object,
    expected_preflight_sha256: str,
    selected_mode: str,
) -> dict[str, object]:
    validated_preflight = _validate_execution_preflight(
        preflight_receipt,
        expected_sha256=expected_preflight_sha256,
        selected_mode=selected_mode,
    )
    preflight_sha256 = str(validated_preflight["preflight_sha256"])
    output_path = Path(out)
    claim_path = Path(claim)
    journal_path = Path(str(output_path) + ".journal.jsonl")
    resolved_paths = {
        "video": Path(video).resolve(),
        "recipe_sequence": Path(recipe_sequence).resolve(),
        "freeze": Path(freeze).resolve(),
        "store": Path(store).resolve(),
        "output": output_path.resolve(),
        "claim": claim_path.resolve(),
        "journal": journal_path.resolve(),
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise ValueError("production input and operational paths must be distinct")
    if _occupied(output_path):
        raise FileExistsError(output_path)
    if _occupied(journal_path):
        raise FileExistsError(journal_path)
    freeze_receipt = _load_freeze(Path(freeze), expected_freeze_sha256)
    claim_receipt = _claim_payload(
        phase=phase,
        owner_id=owner_id,
        freeze_sha256=expected_freeze_sha256,
        video=video,
        recipe_sequence=recipe_sequence,
        store=store,
        out=out,
        preflight_sha256=preflight_sha256,
        selected_mode=selected_mode,
    )
    _publish_json_exclusive(claim_path, claim_receipt)
    journal_sequence = 0
    sequence: dict[str, object] | None = None
    try:
        _start_journal(
            journal_path,
            {
                "schema_version": "carryover-phase-journal-event-v1",
                "sequence": 1,
                "phase": phase,
                "event": "claim_acquired",
                "claim_sha256": claim_receipt["claim_sha256"],
                "preflight_sha256": preflight_sha256,
                "selected_mode": selected_mode,
            },
        )
        journal_sequence = 1
        recipe_raw = _read_production_bytes(Path(recipe_sequence).resolve(), "recipe sequence")
        sequence = _validate_recipe_sequence(
            _read_strict_json_bytes(recipe_raw, "recipe sequence"), expected_role
        )
        expected_vials = int(
            validated_preflight[
                "test" if phase == "run-test" else "real_check"
            ]["vial_count"]
        )
        if len(list(_vial_rows(sequence))) != expected_vials:
            raise ValueError("recipe vial denominator does not match bound preflight")
        video_raw = _read_production_bytes(Path(video).resolve(), "camera video")
        source_bindings = {
            "video_sha256": hashlib.sha256(video_raw).hexdigest(),
            "recipe_sequence_sha256": hashlib.sha256(recipe_raw).hexdigest(),
            "freeze_sha256": expected_freeze_sha256,
            "claim_sha256": str(claim_receipt["claim_sha256"]),
            "preflight_sha256": preflight_sha256,
            "selected_mode": selected_mode,
            "store_binding_sha256": _canonical_hash(
                {
                    "path": str(Path(store).resolve()),
                    "schema": freeze_receipt["store"],
                }
            ),
        }
        _append_journal(
            journal_path,
            {
                "schema_version": "carryover-phase-journal-event-v1",
                "sequence": 2,
                "phase": phase,
                "event": "inputs_bound",
                "video_sha256": source_bindings["video_sha256"],
                "recipe_sequence_sha256": source_bindings["recipe_sequence_sha256"],
                "preflight_sha256": preflight_sha256,
                "selected_mode": selected_mode,
            },
        )
        journal_sequence = 2

        accepted_real_route = bool(freeze_receipt["mask_runtime"]["accepts_real_rgb_only"])
        skip = phase == "run-real-check" and (
            selected_mode == "evaluation_only" or not accepted_real_route
        )
        runtime_bound = (
            mask_runtime is not None
            and isinstance(mask_runtime, PredictedMaskRuntime)
            and mask_runtime.runtime_id == freeze_receipt["mask_runtime"]["runtime_id"]
            and mask_runtime.runtime_sha256 == freeze_receipt["mask_runtime"]["runtime_sha256"]
            and mask_runtime.config_sha256 == freeze_receipt["mask_runtime"]["config_sha256"]
            and (phase != "run-real-check" or mask_runtime.accepts_real_rgb_only)
        )
        measured_records: list[dict[str, object]] = []
        if runtime_bound and not skip:
            assert mask_runtime is not None
            runtime_output = mask_runtime.analyze(
                video_bytes=video_raw,
                recipe_sequence=sequence,
                artifact_root=output_path.parent,
            )
            measured_records = project_runtime_measurements(runtime_output)
            append_runtime_measurements(
                store=Path(store),
                artifact_root=output_path.parent,
                records=measured_records,
                runtime_output=runtime_output,
            )
        reason_codes = (
            ["accepted_rgb_only_route_unavailable"]
            if skip
            else (
                ["history_support_unavailable"]
                if runtime_bound
                else [
                    "predicted_mask_runtime_unavailable",
                    "measurement_unavailable",
                    "history_support_unavailable",
                ]
            )
        )
        records = [
            _insufficient_record(
                run=run,
                vial=vial,
                source_bindings=source_bindings,
                reason_codes=reason_codes,
            )
            for run, vial in _vial_rows(sequence)
        ]
        count = len(records)
        terminal_status = "skipped" if skip else "complete"
        receipt = {
            "schema_version": "carryover-run-receipt-v1",
            "phase": phase,
            "terminal_status": terminal_status,
            "rerun_permitted": False,
            "source_bindings": source_bindings,
            "preflight_binding": {
                "selected_mode": selected_mode,
                "preflight_sha256": preflight_sha256,
            },
            "explanation_policy": wp7_explanation_policy(),
            "operator_records": records,
            "denominators": _denominators(
                sequence,
                abstentions=0 if skip else count,
                skips=count if skip else 0,
            ),
            "provenance_inventory": {
                "process_evidence": ["camera_video", "programmed_recipe_sequence"],
                "operational_bindings": [
                    "freeze",
                    "preflight",
                    "store",
                    "claim",
                    "output",
                ],
                "model_loaded": bool(runtime_bound and not skip),
                "gpu_used": False,
                "measured_stage_count": len(measured_records),
            },
            "skip_reason": "accepted_rgb_only_route_unavailable" if skip else None,
        }
        receipt["receipt_sha256"] = _canonical_hash(receipt)
        _publish_json_exclusive(output_path, receipt)
        _append_journal(
            journal_path,
            {
                "schema_version": "carryover-phase-journal-event-v1",
                "sequence": 3,
                "phase": phase,
                "event": "terminal_published",
                "terminal_status": terminal_status,
                "receipt_sha256": receipt["receipt_sha256"],
                "preflight_sha256": preflight_sha256,
                "selected_mode": selected_mode,
            },
        )
        journal_sequence = 3
        return receipt
    except Exception as error:
        if not output_path.exists():
            fatal = {
                "schema_version": "carryover-run-receipt-v1",
                "phase": phase,
                "terminal_status": "fatal",
                "rerun_permitted": False,
                "claim_sha256": claim_receipt["claim_sha256"],
                "freeze_sha256": expected_freeze_sha256,
                "preflight_sha256": preflight_sha256,
                "selected_mode": selected_mode,
                "fatal_error_code": "input_or_schema_failure",
                "fatal_error_type": type(error).__name__,
                "denominators": _fatal_denominators(sequence),
            }
            fatal["receipt_sha256"] = _canonical_hash(fatal)
            _publish_json_exclusive(output_path, fatal)
            if journal_sequence:
                _append_journal(
                    journal_path,
                    {
                        "schema_version": "carryover-phase-journal-event-v1",
                        "sequence": journal_sequence + 1,
                        "phase": phase,
                        "event": "terminal_published",
                        "terminal_status": "fatal",
                        "receipt_sha256": fatal["receipt_sha256"],
                        "preflight_sha256": preflight_sha256,
                        "selected_mode": selected_mode,
                    },
                )
        raise


def run_test(
    *,
    video: Path,
    recipe_sequence: Path,
    freeze: Path,
    expected_freeze_sha256: str,
    store: Path,
    out: Path,
    claim: Path,
    owner_id: str,
    preflight_receipt: object,
    expected_preflight_sha256: str,
    selected_mode: str,
    mask_runtime: PredictedMaskRuntime | None = None,
) -> dict[str, object]:
    """Seal the one allowed held-out test attempt; a claim is never reusable."""

    return _run_one_pass(
        phase="run-test",
        expected_role="test",
        video=video,
        recipe_sequence=recipe_sequence,
        freeze=freeze,
        expected_freeze_sha256=expected_freeze_sha256,
        store=store,
        out=out,
        claim=claim,
        owner_id=owner_id,
        mask_runtime=mask_runtime,
        preflight_receipt=preflight_receipt,
        expected_preflight_sha256=expected_preflight_sha256,
        selected_mode=selected_mode,
    )


def run_real_check(
    *,
    video: Path,
    recipe_sequence: Path,
    freeze: Path,
    expected_freeze_sha256: str,
    store: Path,
    out: Path,
    claim: Path,
    owner_id: str,
    preflight_receipt: object,
    expected_preflight_sha256: str,
    selected_mode: str,
    mask_runtime: PredictedMaskRuntime | None = None,
) -> dict[str, object]:
    """Seal the distinct real-check attempt or its explicit RGB-route skip."""

    return _run_one_pass(
        phase="run-real-check",
        expected_role="real_check",
        video=video,
        recipe_sequence=recipe_sequence,
        freeze=freeze,
        expected_freeze_sha256=expected_freeze_sha256,
        store=store,
        out=out,
        claim=claim,
        owner_id=owner_id,
        mask_runtime=mask_runtime,
        preflight_receipt=preflight_receipt,
        expected_preflight_sha256=expected_preflight_sha256,
        selected_mode=selected_mode,
    )


__all__ = [
    "PredictedMaskRuntime",
    "RECIPE_SEQUENCE_SCHEMA_SHA256",
    "RuntimeProjectionInput",
    "append_runtime_measurements",
    "freeze_validation",
    "load_recipe_sequence",
    "load_preflight_receipt",
    "pipeline_source_sha256",
    "preflight_explanation_route",
    "wp7_explanation_policy",
    "project_runtime_measurements",
    "run_real_check",
    "run_test",
]

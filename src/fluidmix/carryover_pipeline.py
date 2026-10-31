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
from typing import Callable, Protocol, runtime_checkable
import uuid

from .evidence_store import (
    SCHEMA_VERSION as EVIDENCE_STORE_SCHEMA_VERSION,
    EvidenceStore,
    HistoryQuery,
    IngestResult,
    assert_history_citation,
)
from .wp7_development_reference import (
    ACCEPTED_CNN_PREDICTOR,
    DevelopmentReferenceError,
    DevelopmentReferenceStore,
    DevelopmentReferenceQuery,
    assert_development_reference_citation,
    bind_runtime_target_evidence,
)
from .wp7_final_decision import bind_runtime_camera_stage, decide_wp7_final
from .wp7_rgb_runtime import ENDPOINT_SELECTION_UNAVAILABLE_CODES
from .vision_measurement import trusted_measurement_bindings_from_result_v1
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
_DEVELOPMENT_REFERENCE_STORE_PROVENANCE_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "store_purpose",
    "store_schema_version",
    "store_schema_sha256",
    "store_snapshot_sha256",
    "development_reference_bank_sha256",
    "delta_e_frozen_config_sha256",
    "materialization_receipt_sha256",
    "predicted_store_export_sha256",
    "derived_source_manifest_sha256",
    "source_classes",
    "coverage_query_sha256s",
    "coverage_member_count",
    "configuration_count",
    "stage_row_count",
    "donor_batch_ids",
    "authorization",
    "one_shot_inputs_opened",
    "claim_boundary",
    "predictor",
    "predictor_sha256",
    "accepted_store_tuple",
    "inn67_provenance",
    "frozen_config_identity",
}
_DEVELOPMENT_MASK_PROVENANCE_FIELDS = {
    "schema_version", "artifact_type", "source_class", "predictor",
    "predictor_sha256", "accepted_store_tuple", "predicted_store_export_sha256",
}
_ORDERED_VIDEO_PROVENANCE_FIELDS = {
    "schema_version", "artifact_type", "ordered_video_path", "ordered_video_sha256",
    "vial_count", "composition_receipt_path", "composition_receipt_sha256",
    "final_review_path", "final_review_sha256",
}
_ACCEPTED_ORDERED_VIDEO = {
    "schema_version": "wp7-ordered-test-video-binding-v1",
    "artifact_type": "ordered_test_video_binding",
    "ordered_video_path": "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/wp7-ordered-test-video.mp4",
    "ordered_video_sha256": "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b",
    "vial_count": 18,
    "composition_receipt_path": "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/wp7-ordered-test-video.json",
    "composition_receipt_sha256": "0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037",
    "final_review_path": "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-ordered-test-video-final-review-01a18128-attempt01/FINAL-REVIEW.json",
    "final_review_sha256": "ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683",
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
_DEVELOPMENT_ROUTE_DECISION_FIELDS = {
    "schema_version",
    "artifact_type",
    "issue_id",
    "option",
    "scope",
    "approved_mode",
    "decision_status",
    "approver_name",
    "approver_handle",
    "evidence_boundary",
}
_SECOND_VIAL_ROUTE_DECISION_FIELDS = {
    "schema_version", "artifact_type", "issue_id", "scope", "approved_mode",
    "decision_status", "approver_name", "approver_handle", "reference_rule",
    "distance_metric", "keep_threshold_delta_e76",
    "discard_threshold_delta_e76", "evidence_boundary",
}
_FORBIDDEN_HISTORY_TOKENS = frozenset({"evaluation", "oracle", "sim_metadata"})
_MEASUREMENT_UNAVAILABLE_CODES = frozenset(
    {"mask_empty", "mask_coverage_invalid", "mask_height_quality_invalid"}
)
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
    runtime_provenance: Mapping[str, object] | None = None


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
        "src/fluidmix/evidence_store.py",
        "src/fluidmix/part_b_evidence.py",
        "src/fluidmix/vision_measurement.py",
        "src/fluidmix/wp7_development_reference.py",
        "src/fluidmix/wp7_final_decision.py",
        "src/fluidmix/wp7_phase_capture.py",
        "src/fluidmix/wp7_rgb_runtime.py",
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


def _validate_development_predicted_mask_provenance(
    value: object, store: Mapping[str, object]
) -> dict[str, object]:
    if value is None:
        raise ValueError("predicted-mask provenance is required for development_reference_backed mode")
    provenance = dict(_exact_fields(
        value, _DEVELOPMENT_MASK_PROVENANCE_FIELDS,
        "development predicted-mask provenance",
    ))
    expected = {
        "schema_version": "wp7-development-predicted-mask-provenance-v1",
        "artifact_type": "development_reference_predicted_mask_provenance",
        "source_class": "accepted_rgb_only_cnn_predicted_masks",
        "predictor": store["predictor"],
        "predictor_sha256": store["predictor_sha256"],
        "accepted_store_tuple": store["accepted_store_tuple"],
        "predicted_store_export_sha256": store["predicted_store_export_sha256"],
    }
    if provenance != expected:
        raise ValueError("development predicted-mask predictor/store provenance is invalid")
    return provenance


def _validate_ordered_video_provenance(value: object) -> dict[str, object]:
    binding = dict(_exact_fields(
        value, _ORDERED_VIDEO_PROVENANCE_FIELDS, "ordered video provenance"
    ))
    if binding != _ACCEPTED_ORDERED_VIDEO:
        raise ValueError("ordered video provenance is not the exact accepted 18-vial binding")
    return binding


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


def _validate_development_reference_store_provenance(
    value: object,
) -> dict[str, object]:
    if value is None:
        raise ValueError("development_reference store provenance is required")
    provenance = dict(
        _exact_fields(
            value,
            _DEVELOPMENT_REFERENCE_STORE_PROVENANCE_FIELDS,
            "development_reference store provenance",
        )
    )
    expected_text = {
        "schema_version": "wp7-development-reference-store-v1",
        "artifact_type": "development_reference_store_provenance",
        "status": "published",
        "store_purpose": "development_reference",
        "claim_boundary": "synthetic reference evidence only; not plant or physical truth",
    }
    for field, expected in expected_text.items():
        if provenance[field] != expected:
            raise ValueError(f"development_reference store {field} is invalid")
    if (
        provenance["store_schema_version"] != 1
        or type(provenance["store_schema_version"]) is not int
    ):
        raise ValueError("development_reference store_schema_version is invalid")
    for field in (
        "store_schema_sha256",
        "store_snapshot_sha256",
        "development_reference_bank_sha256",
        "delta_e_frozen_config_sha256",
        "materialization_receipt_sha256",
        "predicted_store_export_sha256",
        "derived_source_manifest_sha256",
        "predictor_sha256",
    ):
        _sha(provenance[field], f"development_reference {field}")
    if (
        provenance["predictor"] != ACCEPTED_CNN_PREDICTOR
        or provenance["predictor_sha256"] != _canonical_hash(ACCEPTED_CNN_PREDICTOR)
    ):
        raise ValueError("development_reference accepted predictor identity is invalid")
    accepted_store = _exact_fields(
        provenance["accepted_store_tuple"],
        {"database_path", "database_sha256", "population_receipt_path", "population_receipt_sha256"},
        "development_reference accepted store tuple",
    )
    for field in ("database_sha256", "population_receipt_sha256"):
        _sha(accepted_store[field], f"development_reference accepted store {field}")
    if (
        not isinstance(accepted_store["database_path"], str)
        or not Path(accepted_store["database_path"]).is_absolute()
        or not isinstance(accepted_store["population_receipt_path"], str)
        or not Path(accepted_store["population_receipt_path"]).is_absolute()
    ):
        raise ValueError("development_reference accepted store paths are invalid")
    inn67 = _exact_fields(
        provenance["inn67_provenance"],
        {
            "reference_bank_plan_sha256", "reference_bank_inventory_sha256",
            "pre_access_amendment_file_sha256", "pre_access_amendment_receipt_sha256",
            "materialization_protocol_amendment_sha256", "render_inventory_sha256",
            "predicted_store_export_sha256", "accepted_store_tuple",
        },
        "development_reference INN-67 provenance",
    )
    fixed = {
        "reference_bank_plan_sha256": "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79",
        "reference_bank_inventory_sha256": "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95",
        "pre_access_amendment_file_sha256": "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8",
        "pre_access_amendment_receipt_sha256": "3cc4212f84a6a011b0d3dd579cb6cf817285d3a946598908756f6c95c9f349db",
        "materialization_protocol_amendment_sha256": "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c",
    }
    if (
        any(inn67[field] != digest for field, digest in fixed.items())
        or inn67["predicted_store_export_sha256"] != provenance["predicted_store_export_sha256"]
        or inn67["accepted_store_tuple"] != dict(accepted_store)
    ):
        raise ValueError("development_reference INN-67 provenance binding is invalid")
    _sha(inn67["render_inventory_sha256"], "development_reference render inventory")
    expected_config = {
        "config_id": "v3-relative-diagnostic-inherited-stage-v1",
        "distance_metric": "CIE76", "unit": "delta_e76",
        "tight_peer_pair_max_delta_e76": 2.0,
        "target_outlier_margin_delta_e76": 1.0,
        "decision_rule": "peer_delta = dE76(B01,B02); threshold = peer_delta/2 + 1.0; require peer_delta <= 2.0 and B03 target_to_center_delta_e76 <= threshold",
        "threshold_formula": "threshold_delta_e76 = peer_pair_delta_e76 / 2 + 1.0",
        "configuration_target_gate": "all 16 B03 recipe-stage targets must be consistent_with_two_peer_mode",
        "selection_input_batch_ids": ["INN67-DEV-RB-B03"],
        "reference_geometry_batch_ids": ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"],
        "configuration_target_batch_id": "INN67-DEV-RB-B03",
    }
    if provenance["frozen_config_identity"] != expected_config:
        raise ValueError("development_reference frozen config identity is invalid")
    if provenance["source_classes"] != ["synthetic_rgb", "predicted_mask"]:
        raise ValueError("development_reference source_classes are invalid")
    if provenance["stage_row_count"] != 32 or type(provenance["stage_row_count"]) is not int:
        raise ValueError("development_reference stage_row_count must equal 32")
    if provenance["donor_batch_ids"] != ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"]:
        raise ValueError("development_reference donor_batch_ids are invalid")
    if provenance["one_shot_inputs_opened"] is not False:
        raise ValueError("development_reference provenance opened one-shot inputs")
    if provenance["authorization"] != {
        "issue_id": "INN-64",
        "option": 3,
        "approver_name": "Liam",
        "scope": "one_shot_after_accepted_development_reference_bank_and_frozen_config",
    }:
        raise ValueError("development_reference authorization is invalid")
    if provenance["configuration_count"] != 16:
        raise ValueError("development_reference configuration_count must equal 16")
    if provenance["coverage_member_count"] != 32:
        raise ValueError("development_reference coverage_member_count must equal 32")
    query_sha256s = provenance["coverage_query_sha256s"]
    if (
        not isinstance(query_sha256s, list)
        or len(query_sha256s) != 16
        or len(query_sha256s) != len(set(query_sha256s))
    ):
        raise ValueError("development_reference coverage_query_sha256s must contain 16 unique values")
    for digest in query_sha256s:
        _sha(digest, "development_reference coverage_query_sha256s member")
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


def _validate_development_route_decision(
    value: object, expected_sha256: object
) -> dict[str, object]:
    decision = dict(
        _exact_fields(
            value,
            _DEVELOPMENT_ROUTE_DECISION_FIELDS,
            "development_reference route-decision",
        )
    )
    expected = {
        "schema_version": "wp7-development-reference-route-decision-v1",
        "artifact_type": "wp7_development_reference_route_decision",
        "issue_id": "INN-64",
        "option": 3,
        "scope": "wp1_and_video5_single_pass_with_development_reference_bank",
        "approved_mode": "development_reference_backed",
        "decision_status": "approved",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
        "evidence_boundary": "synthetic_reference_evidence_only_not_plant_or_physical_truth",
    }
    if decision != expected:
        for field in sorted(_DEVELOPMENT_ROUTE_DECISION_FIELDS):
            if decision.get(field) != expected[field]:
                raise ValueError(
                    f"development_reference route-decision {field} is invalid"
                )
        raise ValueError("development_reference route-decision is invalid")
    declared = _sha(expected_sha256, "expected route-decision SHA-256")
    if _canonical_hash(decision) != declared:
        raise ValueError("route-decision SHA-256 mismatch")
    return decision


def _validate_second_vial_route_decision(
    value: object, expected_sha256: object
) -> dict[str, object]:
    decision = dict(_exact_fields(
        value,
        _SECOND_VIAL_ROUTE_DECISION_FIELDS,
        "same-camera second-vial route-decision",
    ))
    expected = {
        "schema_version": "wp7-same-camera-second-vial-route-decision-v1",
        "artifact_type": "wp7_same_camera_second_vial_route_decision",
        "issue_id": "INN-67",
        "scope": "wp1_and_video5_same_camera_second_vial_reference",
        "approved_mode": "same_camera_second_vial_reference",
        "decision_status": "approved",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
        "reference_rule": (
            "position_1_no_prior_reference_position_2_reference_control_"
            "position_3_compares_to_position_2"
        ),
        "distance_metric": "CIE76",
        "keep_threshold_delta_e76": 1.0,
        "discard_threshold_delta_e76": 2.0,
        "evidence_boundary": (
            "runtime_rgb_cnn_measurements_and_programmed_recipe_metadata_only"
        ),
    }
    if decision != expected:
        raise ValueError("same-camera second-vial route-decision is invalid")
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
    development_reference_store_provenance: object | None = None,
    ordered_video_provenance: object | None = None,
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

    if requested_mode not in {
        "evaluation_only", "history_backed", "development_reference_backed",
        "same_camera_second_vial_reference",
    }:
        raise ValueError(
            "requested_mode is not an executable explanation route"
        )
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
                development_reference_store_provenance,
                history_queries,
                route_decision,
                expected_route_decision_sha256,
                ordered_video_provenance,
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

    if requested_mode == "same_camera_second_vial_reference":
        if any(value is not None for value in (
            evaluation_store_provenance, predicted_mask_provenance,
            history_store_provenance, development_reference_store_provenance,
            ordered_video_provenance, history_queries,
        )):
            raise ValueError(
                "same-camera second-vial mode accepts no external evidence store"
            )
        if route_decision is None or expected_route_decision_sha256 is None:
            raise ValueError(
                "same-camera second-vial route-decision and SHA-256 are required"
            )
        decision = _validate_second_vial_route_decision(
            route_decision, expected_route_decision_sha256
        )
        receipt = {
            **common,
            "test": {
                "terminal_behavior": "same_camera_second_vial_reference",
                "vial_count": 18,
            },
            "real_check": {
                "terminal_behavior": "same_camera_second_vial_reference",
                "vial_count": 30,
            },
            "activation_status": "same_camera_second_vial_requirements_satisfied",
            "history_backed_explanations_enabled": False,
            "history": None,
            "predicted_mask": None,
            "route_decision": {
                **decision,
                "route_decision_sha256": str(expected_route_decision_sha256),
            },
        }
        receipt["preflight_sha256"] = _canonical_hash(receipt)
        return receipt

    if requested_mode == "development_reference_backed":
        if any(
            value is not None
            for value in (
                evaluation_store_provenance,
                history_store_provenance,
            )
        ):
            raise ValueError(
                "development_reference_backed mode must not receive evaluation-only or production-history provenance"
            )
        store = _validate_development_reference_store_provenance(
            development_reference_store_provenance
        )
        mask = _validate_development_predicted_mask_provenance(
            predicted_mask_provenance, store
        )
        ordered_video = _validate_ordered_video_provenance(ordered_video_provenance)
        if isinstance(history_queries, (str, bytes)) or not isinstance(
            history_queries, Sequence
        ):
            raise TypeError("history_queries must be a sequence of trusted DevelopmentReferenceQuery values")
        queries = tuple(history_queries)
        if not queries:
            raise ValueError("development_reference_backed mode requires nonempty evidence")
        query_sha256s: list[str] = []
        member_count = 0
        for query in queries:
            if not isinstance(query, DevelopmentReferenceQuery):
                raise TypeError("history_queries must contain trusted DevelopmentReferenceQuery values")
            if query.query_kind != "development_reference_coverage" or len(query.members) != 2:
                raise ValueError(
                    "development_reference_backed mode requires two-member coverage queries"
                )
            first = query.members[0]
            assert_development_reference_citation(query, first.evidence_id, first.image_id)
            if any(
                member.split_role != "development_reference"
                for member in query.members
            ):
                raise ValueError(
                    "development_reference query contains a non-development_reference row"
                )
            accepted_store = store["accepted_store_tuple"]
            assert isinstance(accepted_store, Mapping)
            if any(
                member.predictor_producer_id != ACCEPTED_CNN_PREDICTOR["producer_id"]
                or member.predictor_model_id != ACCEPTED_CNN_PREDICTOR["model_id"]
                or member.predictor_model_sha256 != ACCEPTED_CNN_PREDICTOR["model_sha256"]
                or member.predictor_config_sha256 != ACCEPTED_CNN_PREDICTOR["config_sha256"]
                or member.predictor_code_commit != ACCEPTED_CNN_PREDICTOR["code_commit"]
                or member.predictor_sha256 != store["predictor_sha256"]
                or member.predicted_store_export_sha256
                != store["predicted_store_export_sha256"]
                or member.accepted_store_database_path
                != accepted_store["database_path"]
                or member.accepted_store_database_sha256
                != accepted_store["database_sha256"]
                or member.accepted_store_population_receipt_path
                != accepted_store["population_receipt_path"]
                or member.accepted_store_population_receipt_sha256
                != accepted_store["population_receipt_sha256"]
                or member.donor_store_snapshot_sha256
                != store["store_snapshot_sha256"]
                for member in query.members
            ):
                raise ValueError(
                    "development_reference query member provenance does not match the accepted store snapshot"
                )
            for member in query.members:
                for field in (
                    "source_render_receipt_sha256",
                    "source_validation_receipt_sha256",
                    "source_render_plan_sha256",
                    "known_good_profile_sha256",
                ):
                    _sha(
                        getattr(member, field),
                        f"development_reference query member {field}",
                    )
                if member.known_good_profile_sha256 != (
                    "47bc0b6f483a121a28871b3eb74ec27efb79996b83d03d8c45d6f5b0b9cdeb7c"
                ):
                    raise ValueError(
                        "development_reference query member profile identity is invalid"
                    )
            if any(
                _history_path_has_forbidden_source(path)
                for member in query.members
                for path in (member.rgb_path, member.mask_path)
            ):
                raise ValueError(
                    "development_reference query contains evaluation, oracle, or sim_metadata provenance"
                )
            query_sha256s.append(query.query_sha256)
            member_count += len(query.members)
        if len(queries) != 16 or member_count != 32:
            raise ValueError("development_reference coverage must bind 16 queries and 32 members")
        if query_sha256s != store["coverage_query_sha256s"]:
            raise ValueError(
                "development_reference query_sha256s do not match the trusted queries"
            )
        if route_decision is None or expected_route_decision_sha256 is None:
            raise ValueError(
                "development_reference route-decision and expected SHA-256 are required"
            )
        decision = _validate_development_route_decision(
            route_decision, expected_route_decision_sha256
        )
        receipt = {
            **common,
            "test": {"terminal_behavior": "development_reference_backed_explanation", "vial_count": 18},
            "real_check": {"terminal_behavior": "development_reference_backed_explanation", "vial_count": 30},
            "activation_status": "development_reference_requirements_satisfied",
            "history_backed_explanations_enabled": True,
            "evidence_boundary": decision["evidence_boundary"],
            "route_decision": {
                "issue_id": decision["issue_id"],
                "option": decision["option"],
                "scope": decision["scope"],
                "approved_mode": decision["approved_mode"],
                "approver_name": decision["approver_name"],
                "approver_handle": decision["approver_handle"],
                "route_decision_sha256": str(expected_route_decision_sha256),
            },
            "predicted_mask": mask,
            "ordered_video": ordered_video,
            "history": {
                key: store[key]
                for key in (
                    "store_purpose", "store_schema_version", "store_schema_sha256",
                    "store_snapshot_sha256",
                    "development_reference_bank_sha256", "delta_e_frozen_config_sha256",
                    "materialization_receipt_sha256", "predicted_store_export_sha256",
                    "derived_source_manifest_sha256", "stage_row_count", "configuration_count",
                    "coverage_member_count", "donor_batch_ids",
                    "predictor", "predictor_sha256", "accepted_store_tuple",
                    "inn67_provenance", "frozen_config_identity",
                )
            } | {
                "query_sha256s": query_sha256s,
                "query_count": len(queries),
                "member_count": member_count,
            },
        }
        receipt["preflight_sha256"] = _canonical_hash(receipt)
        return receipt

    if development_reference_store_provenance is not None:
        raise ValueError("history_backed mode must not receive development_reference provenance")
    if evaluation_store_provenance is not None:
        raise ValueError("history_backed mode must not receive an evaluation-only store")
    if ordered_video_provenance is not None:
        raise ValueError("history_backed mode must not receive ordered-video provenance")
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
            if member.split_role in {"evaluation", "test", "development_reference"}:
                raise ValueError(
                    "production-history query contains evaluation, test, or development_reference evidence"
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
    if selected_mode not in {
        "evaluation_only", "history_backed", "development_reference_backed",
        "same_camera_second_vial_reference",
    }:
        raise ValueError(
            "selected mode is not an executable explanation route"
        )
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
    if selected_mode == "development_reference_backed":
        fields.update({"evidence_boundary", "ordered_video"})
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

    if selected_mode == "same_camera_second_vial_reference":
        expected_terminal = {
            "terminal_behavior": "same_camera_second_vial_reference"
        }
        if receipt["test"] != {**expected_terminal, "vial_count": 18} or receipt[
            "real_check"
        ] != {**expected_terminal, "vial_count": 30}:
            raise ValueError("same-camera second-vial terminal semantics are invalid")
        if (
            receipt["activation_status"]
            != "same_camera_second_vial_requirements_satisfied"
            or receipt["history_backed_explanations_enabled"] is not False
            or receipt["history"] is not None
            or receipt["predicted_mask"] is not None
        ):
            raise ValueError("same-camera second-vial preflight state is invalid")
        decision = dict(_exact_fields(
            receipt["route_decision"],
            _SECOND_VIAL_ROUTE_DECISION_FIELDS | {"route_decision_sha256"},
            "same-camera second-vial route-decision binding",
        ))
        decision_sha = decision.pop("route_decision_sha256")
        _validate_second_vial_route_decision(decision, decision_sha)
        return receipt

    if selected_mode == "development_reference_backed":
        expected_boundary = (
            "synthetic_reference_evidence_only_not_plant_or_physical_truth"
        )
        if receipt["test"] != {
            "terminal_behavior": "development_reference_backed_explanation",
            "vial_count": 18,
        } or receipt["real_check"] != {
            "terminal_behavior": "development_reference_backed_explanation",
            "vial_count": 30,
        }:
            raise ValueError("development_reference terminal semantics are invalid")
        if (
            receipt["activation_status"]
            != "development_reference_requirements_satisfied"
            or receipt["history_backed_explanations_enabled"] is not True
            or receipt["evidence_boundary"] != expected_boundary
        ):
            raise ValueError("development_reference preflight state is invalid")
        decision = _exact_fields(
            receipt["route_decision"],
            {
                "issue_id", "option", "scope", "approved_mode", "approver_name",
                "approver_handle", "route_decision_sha256",
            },
            "development_reference route-decision binding",
        )
        expected_decision = {
            "issue_id": "INN-64",
            "option": 3,
            "scope": "wp1_and_video5_single_pass_with_development_reference_bank",
            "approved_mode": "development_reference_backed",
            "approver_name": "Liam",
            "approver_handle": "@salasl",
        }
        if any(decision[field] != expected for field, expected in expected_decision.items()):
            raise ValueError("development_reference route-decision binding is invalid")
        _sha(decision["route_decision_sha256"], "development_reference route-decision SHA-256")
        history = _exact_fields(
            receipt["history"],
            {
                "store_purpose", "store_schema_version", "store_schema_sha256",
                "store_snapshot_sha256",
                "development_reference_bank_sha256", "delta_e_frozen_config_sha256",
                "materialization_receipt_sha256", "predicted_store_export_sha256",
                "derived_source_manifest_sha256", "stage_row_count", "configuration_count",
                "coverage_member_count", "donor_batch_ids",
                "predictor", "predictor_sha256", "accepted_store_tuple",
                "inn67_provenance", "frozen_config_identity",
                "query_sha256s", "query_count", "member_count",
            },
            "development_reference store binding",
        )
        if (
            history["store_purpose"] != "development_reference"
            or history["store_schema_version"] != 1
            or type(history["store_schema_version"]) is not int
            or history["stage_row_count"] != 32
            or history["configuration_count"] != 16
            or history["coverage_member_count"] != 32
            or history["donor_batch_ids"]
            != ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"]
        ):
            raise ValueError("development_reference store identity is invalid")
        for field in (
            "store_schema_sha256", "store_snapshot_sha256",
            "development_reference_bank_sha256", "delta_e_frozen_config_sha256",
            "materialization_receipt_sha256", "predicted_store_export_sha256",
            "derived_source_manifest_sha256",
            "predictor_sha256",
        ):
            _sha(history[field], f"development_reference store {field}")
        if (
            history["predictor"] != ACCEPTED_CNN_PREDICTOR
            or history["predictor_sha256"] != _canonical_hash(ACCEPTED_CNN_PREDICTOR)
        ):
            raise ValueError("development_reference predictor identity is invalid")
        _validate_development_predicted_mask_provenance(
            receipt["predicted_mask"], history
        )
        _validate_ordered_video_provenance(receipt["ordered_video"])
        accepted_store = _exact_fields(
            history["accepted_store_tuple"],
            {"database_path", "database_sha256", "population_receipt_path", "population_receipt_sha256"},
            "development_reference accepted store tuple",
        )
        for field in ("database_sha256", "population_receipt_sha256"):
            _sha(accepted_store[field], f"development_reference accepted store {field}")
        if (
            not isinstance(accepted_store["database_path"], str)
            or not Path(accepted_store["database_path"]).is_absolute()
            or not isinstance(accepted_store["population_receipt_path"], str)
            or not Path(accepted_store["population_receipt_path"]).is_absolute()
        ):
            raise ValueError("development_reference accepted store path identity is invalid")
        inn67 = _exact_fields(
            history["inn67_provenance"],
            {
                "reference_bank_plan_sha256", "reference_bank_inventory_sha256",
                "pre_access_amendment_file_sha256", "pre_access_amendment_receipt_sha256",
                "materialization_protocol_amendment_sha256", "render_inventory_sha256",
                "predicted_store_export_sha256", "accepted_store_tuple",
            },
            "development_reference execution INN-67 provenance",
        )
        fixed = {
            "reference_bank_plan_sha256": "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79",
            "reference_bank_inventory_sha256": "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95",
            "pre_access_amendment_file_sha256": "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8",
            "pre_access_amendment_receipt_sha256": "3cc4212f84a6a011b0d3dd579cb6cf817285d3a946598908756f6c95c9f349db",
            "materialization_protocol_amendment_sha256": "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c",
        }
        if (
            any(inn67[field] != digest for field, digest in fixed.items())
            or inn67["predicted_store_export_sha256"] != history["predicted_store_export_sha256"]
            or inn67["accepted_store_tuple"] != dict(accepted_store)
        ):
            raise ValueError("development_reference execution INN-67 identity is invalid")
        _sha(inn67["render_inventory_sha256"], "development_reference render inventory")
        if history["frozen_config_identity"] != {
            "config_id": "v3-relative-diagnostic-inherited-stage-v1",
            "distance_metric": "CIE76", "unit": "delta_e76",
            "tight_peer_pair_max_delta_e76": 2.0,
            "target_outlier_margin_delta_e76": 1.0,
            "decision_rule": "peer_delta = dE76(B01,B02); threshold = peer_delta/2 + 1.0; require peer_delta <= 2.0 and B03 target_to_center_delta_e76 <= threshold",
            "threshold_formula": "threshold_delta_e76 = peer_pair_delta_e76 / 2 + 1.0",
            "configuration_target_gate": "all 16 B03 recipe-stage targets must be consistent_with_two_peer_mode",
            "selection_input_batch_ids": ["INN67-DEV-RB-B03"],
            "reference_geometry_batch_ids": ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"],
            "configuration_target_batch_id": "INN67-DEV-RB-B03",
        }:
            raise ValueError("development_reference execution frozen config identity is invalid")
        query_sha256s = history["query_sha256s"]
        if (
            not isinstance(query_sha256s, list)
            or len(query_sha256s) != 16
            or len(query_sha256s) != len(set(query_sha256s))
        ):
            raise ValueError("development_reference query SHA-256 bindings are invalid")
        for digest in query_sha256s:
            _sha(digest, "development_reference query SHA-256")
        if (
            type(history["query_count"]) is not int
            or history["query_count"] != len(query_sha256s)
            or type(history["member_count"]) is not int
            or history["member_count"] != 32
        ):
            raise ValueError("development_reference query/member counts are invalid")
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
    evidence_out_store: Path,
    development_reference_store: Path | None,
    out: Path,
    preflight_sha256: str,
    selected_mode: str,
    capture_plan: Path | None,
    capture_plan_sha256: str | None,
    capture_receipt_out: Path | None,
    capture_provenance_out: Path | None,
) -> dict[str, object]:
    _text(owner_id, "claim owner_id")
    payload = {
        "schema_version": "carryover-one-pass-claim-v1",
        "phase": phase,
        "owner_id": owner_id,
        "freeze_sha256": freeze_sha256,
        "video_path": str(Path(video).resolve()),
        "recipe_sequence_path": str(Path(recipe_sequence).resolve()),
        "evidence_out_store_path": str(Path(evidence_out_store).resolve()),
        "development_reference_store_path": (
            None
            if development_reference_store is None
            else str(Path(development_reference_store).resolve())
        ),
        "out_path": str(Path(out).resolve()),
        "preflight_sha256": preflight_sha256,
        "selected_mode": selected_mode,
        "capture_plan_path": (
            None if capture_plan is None else str(Path(capture_plan).resolve())
        ),
        "capture_plan_sha256": capture_plan_sha256,
        "capture_receipt_out_path": (
            None
            if capture_receipt_out is None
            else str(Path(capture_receipt_out).resolve())
        ),
        "capture_provenance_out_path": (
            None
            if capture_provenance_out is None
            else str(Path(capture_provenance_out).resolve())
        ),
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


def _bind_published_capture(
    *,
    capture_raw: bytes,
    provenance_raw: bytes,
    phase: str,
    video_sha256: str,
    capture_plan: Path,
    capture_plan_sha256: str,
    capture_receipt_out: Path,
    capture_provenance_out: Path,
) -> dict[str, object]:
    capture = _exact_fields(
        _read_strict_json_bytes(capture_raw, "published capture receipt"),
        {
            "schema_version", "artifact_type", "capture_origin", "video_sha256",
            "receipt_sha256", "encoded_transition_evidence", "fill_clips",
        },
        "published capture receipt",
    )
    if (
        capture["schema_version"] != "wp7-camera-capture-v1"
        or capture["artifact_type"] != "wp7_camera_capture"
        or capture["capture_origin"] not in {"synthetic_camera", "real_camera"}
        or (phase == "run-real-check" and capture["capture_origin"] != "real_camera")
        or capture["video_sha256"] != video_sha256
    ):
        raise ValueError("published capture identity does not match the claimed phase")
    expected_receipt_sha256 = _canonical_hash({
        key: value for key, value in capture.items() if key != "receipt_sha256"
    })
    if capture["receipt_sha256"] != expected_receipt_sha256:
        raise ValueError("published capture canonical SHA-256 mismatch")
    provenance = _exact_fields(
        _read_strict_json_bytes(provenance_raw, "published capture provenance"),
        {
            "schema_version", "artifact_type", "status", "phase",
            "capture_origin", "video_sha256", "plan_binding",
            "capture_receipt_binding", "source_provenance",
        },
        "published capture provenance",
    )
    if (
        provenance["schema_version"] != "wp7-phase-capture-publication-v1"
        or provenance["artifact_type"] != "wp7_phase_capture_publication"
        or provenance["status"] != "published"
        or provenance["phase"] != phase
        or provenance["capture_origin"] != capture["capture_origin"]
        or provenance["video_sha256"] != video_sha256
        or not isinstance(provenance["source_provenance"], list)
        or not provenance["source_provenance"]
    ):
        raise ValueError("published capture provenance identity is invalid")
    plan_binding = _exact_fields(
        provenance["plan_binding"],
        {"path", "raw_file_sha256", "canonical_plan_sha256"},
        "published capture plan binding",
    )
    receipt_binding = _exact_fields(
        provenance["capture_receipt_binding"],
        {"path", "raw_file_sha256", "receipt_sha256"},
        "published capture receipt binding",
    )
    if (
        Path(str(plan_binding["path"])).resolve() != Path(capture_plan).resolve()
        or plan_binding["raw_file_sha256"] != capture_plan_sha256
        or Path(str(receipt_binding["path"])).resolve()
        != Path(capture_receipt_out).resolve()
        or receipt_binding["raw_file_sha256"]
        != hashlib.sha256(capture_raw).hexdigest()
        or receipt_binding["receipt_sha256"] != capture["receipt_sha256"]
    ):
        raise ValueError("published capture provenance cross-binding is invalid")
    _sha(plan_binding["canonical_plan_sha256"], "capture canonical plan SHA-256")
    return {
        "capture_receipt_path": str(Path(capture_receipt_out).resolve()),
        "capture_receipt_raw_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "capture_receipt_sha256": capture["receipt_sha256"],
        "capture_provenance_path": str(Path(capture_provenance_out).resolve()),
        "capture_provenance_raw_sha256": hashlib.sha256(provenance_raw).hexdigest(),
    }


def _decision_identifier(prefix: str, value: object) -> str:
    return f"{prefix}-{_canonical_hash(value)}"


def _measurement_is_unavailable(binding: object) -> bool:
    if binding.validity == "valid":
        return False
    codes = set(binding.limit_codes)
    if not codes or not codes.issubset(_MEASUREMENT_UNAVAILABLE_CODES):
        raise ValueError("runtime measurement integrity failure")
    return True


def _assert_runtime_stage_recipe_identity(
    stage: Mapping[str, object],
    *,
    run: Mapping[str, object],
    vial: Mapping[str, object],
    sequence: Mapping[str, object],
) -> None:
    order = vial["dispense_order"]
    assert isinstance(order, list)
    ordinal = stage.get("stage_ordinal")
    if type(ordinal) is not int or not 1 <= ordinal <= len(order):
        raise ValueError("runtime template identity mismatch: stage_ordinal")
    expected = {
        "run_id": run["run_id"],
        "case_id": vial["case_id"],
        "vial_ordinal": vial["vial_ordinal"],
        "block_recipe_id": vial["block_recipe_id"],
        "block_position": vial["block_position"],
        "recipe_id": vial["recipe_id"],
        "previous_recipe_id": vial["previous_recipe_id"],
        "recipe_ml": dict(vial["programmed_tank_ml"]),
        "dispense_order": list(order),
        "split_role": sequence["split_role"],
        "split_version": sequence["split_version"],
        "stage_id": f"STAGE-{vial['case_id']}-{ordinal}",
        "stage_ordinal": ordinal,
        "tank_id": order[ordinal - 1],
        "measurement_id": f"MEAS-{vial['case_id']}-{order[ordinal - 1]}",
        "evidence_id": f"EVID-{vial['case_id']}-{order[ordinal - 1]}",
    }
    for field, expected_value in expected.items():
        if field not in stage:
            raise ValueError(f"runtime template identity mismatch: missing {field}")
        try:
            matches = _canonical_bytes(stage[field]) == _canonical_bytes(
                expected_value
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"runtime template identity mismatch: {field}")


def _trusted_runtime_bindings_by_image(
    runtime_output: RuntimeProjectionInput,
) -> dict[str, object]:
    bindings = trusted_measurement_bindings_from_result_v1(
        runtime_output.trusted_adapter_result
    )
    by_image = {binding.image_id: binding for binding in bindings}
    if len(by_image) != len(bindings):
        raise ValueError("runtime measurement image bindings are not unique")
    return by_image


def _runtime_provenance_vials(
    runtime_output: RuntimeProjectionInput,
) -> dict[str, tuple[list[int], list[str]]]:
    provenance = runtime_output.runtime_provenance
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime template identity mismatch: runtime provenance")
    rows = provenance.get("vials")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("runtime template identity mismatch: runtime provenance")
    result: dict[str, tuple[list[int], list[str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("runtime template identity mismatch: runtime provenance")
        case_id = row.get("case_id")
        frames = row.get("selected_frame_indices")
        citations = row.get("measurement_citation_sha256s")
        empty_selection = frames == [] and citations == []
        explicit_unavailable = (
            row.get("selection_status") == "abstain"
            and isinstance(row.get("limit_codes"), list)
            and len(row["limit_codes"]) == 1
            and row["limit_codes"][0] in ENDPOINT_SELECTION_UNAVAILABLE_CODES
        )
        if (
            not isinstance(case_id, str)
            or case_id in result
            or not isinstance(frames, list)
            or (empty_selection and not explicit_unavailable)
            or (not empty_selection and row.get("selection_status") == "abstain")
            or any(type(frame) is not int or frame < 0 for frame in frames)
            or not isinstance(citations, list)
            or len(citations) != len(frames)
            or any(
                not isinstance(citation, str)
                or _SHA256.fullmatch(citation) is None
                for citation in citations
            )
        ):
            raise ValueError("runtime template identity mismatch: runtime provenance")
        result[case_id] = (list(frames), list(citations))
    return result


def _validated_runtime_templates_by_vial(
    sequence: Mapping[str, object], runtime_output: RuntimeProjectionInput
) -> dict[tuple[str, str], list[dict[str, object]]]:
    expected = {
        (str(run["run_id"]), str(vial["case_id"])): (run, vial)
        for run, vial in _vial_rows(sequence)
    }
    bindings_by_image = _trusted_runtime_bindings_by_image(runtime_output)
    provenance_by_case = _runtime_provenance_vials(runtime_output)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for template in runtime_output.record_templates:
        if not isinstance(template, dict):
            raise TypeError("runtime record templates must contain dictionaries")
        run_id = template.get("run_id")
        case_id = template.get("case_id")
        if not isinstance(run_id, str) or not isinstance(case_id, str):
            raise ValueError("runtime template identity mismatch: run_id/case_id")
        key = (run_id, case_id)
        if key not in expected:
            raise ValueError("runtime template identity mismatch: run_id/case_id")
        run, vial = expected[key]
        _assert_runtime_stage_recipe_identity(
            template, run=run, vial=vial, sequence=sequence
        )
        ordinal = int(template["stage_ordinal"])
        association = provenance_by_case.get(case_id)
        if association is None or ordinal > len(association[0]):
            raise ValueError("runtime template identity mismatch: image association")
        expected_image_id = f"{case_id}-F{association[0][ordinal - 1]:06d}"
        if (
            "image_id" not in template
            or template["image_id"] != expected_image_id
        ):
            raise ValueError("runtime template identity mismatch: image association")
        binding = bindings_by_image.get(expected_image_id)
        if binding is not None:
            _measurement_is_unavailable(binding)
            if (
                binding.measurement_id != template["measurement_id"]
                or binding.evidence_id != template["evidence_id"]
                or binding.measurement_sha256 != association[1][ordinal - 1]
            ):
                raise ValueError(
                    "runtime template identity mismatch: trusted measurement association"
                )
        grouped.setdefault(key, []).append(template)
    for templates in grouped.values():
        ordinals = [template["stage_ordinal"] for template in templates]
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise ValueError("runtime template identity mismatch: stage sequence")
    return grouped


def _prove_runtime_projection_before_reference_load(
    sequence: Mapping[str, object], runtime_output: RuntimeProjectionInput
) -> None:
    """Validate every usable vial projection before opening the donor store."""

    by_image = _trusted_runtime_bindings_by_image(runtime_output)
    templates_by_vial = _validated_runtime_templates_by_vial(
        sequence, runtime_output
    )
    for run, vial in _vial_rows(sequence):
        templates = templates_by_vial.get(
            (str(run["run_id"]), str(vial["case_id"])), []
        )
        if len(templates) != len(vial["dispense_order"]):
            continue
        selected = [by_image.get(template.get("image_id")) for template in templates]
        unavailable = [
            binding for binding in selected
            if binding is not None and _measurement_is_unavailable(binding)
        ]
        if any(binding is None for binding in selected):
            continue
        if unavailable:
            continue
        project_runtime_measurements(RuntimeProjectionInput(
            tuple(templates), runtime_output.trusted_adapter_result,
            runtime_output.image_locators, runtime_output.event_cursors,
            runtime_output.runtime_provenance,
        ))


def _same_camera_reference_blocks(
    sequence: Mapping[str, object], expected_vials: int
) -> tuple[tuple[Mapping[str, object], tuple[Mapping[str, object], ...]], ...]:
    """Return exact ordered three-vial recipe blocks for the same-camera route."""

    rows = list(_vial_rows(sequence))
    if len(rows) != expected_vials or expected_vials not in {18, 30}:
        raise ValueError("same-camera second-vial denominator is invalid")
    grouped: list[
        tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]
    ] = []
    for run in sequence["runs"]:
        assert isinstance(run, Mapping)
        vials = run["vials"]
        assert isinstance(vials, list)
        if [vial.get("vial_ordinal") for vial in vials] != list(
            range(1, len(vials) + 1)
        ):
            raise ValueError("same-camera vials must remain in ordered recipe sequence")
        start = 0
        while start < len(vials):
            recipe = vials[start].get("block_recipe_id")
            stop = start + 1
            while stop < len(vials) and vials[stop].get("block_recipe_id") == recipe:
                stop += 1
            block = tuple(vials[start:stop])
            positions = [vial.get("block_position") for vial in block]
            if len(block) != 3 or positions != [1, 2, 3]:
                raise ValueError(
                    "same-camera recipe block positions must be exactly [1, 2, 3]"
                )
            if any(
                vial.get("recipe_id") != recipe
                or vial.get("block_recipe_id") != recipe
                for vial in block
            ):
                raise ValueError("same-camera recipe identity mismatch")
            orders = [vial.get("dispense_order") for vial in block]
            recipes = [vial.get("programmed_tank_ml") for vial in block]
            if any(order != orders[0] for order in orders[1:]) or any(
                programmed != recipes[0] for programmed in recipes[1:]
            ):
                raise ValueError("same-camera stage-key parity mismatch")
            grouped.append((run, block))
            start = stop
    if len(grouped) != expected_vials // 3:
        raise ValueError("same-camera three-vial block denominator is invalid")
    return tuple(grouped)


def _same_camera_stage_distances(
    target_vial: Mapping[str, object],
    reference_vial: Mapping[str, object],
    target_records: Sequence[Mapping[str, object]],
    reference_records: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    """Compute causal CIE76 distances for exact matching programmed stages."""

    target_ordinal = target_vial.get("vial_ordinal")
    reference_ordinal = reference_vial.get("vial_ordinal")
    if (
        type(target_ordinal) is not int
        or type(reference_ordinal) is not int
        or reference_ordinal >= target_ordinal
    ):
        raise ValueError("same-camera reference must be strictly earlier than target")
    target_keys = [
        (record.get("stage_ordinal"), record.get("tank_id"))
        for record in target_records
    ]
    reference_keys = [
        (record.get("stage_ordinal"), record.get("tank_id"))
        for record in reference_records
    ]
    expected_keys = list(enumerate(target_vial["dispense_order"], start=1))
    if target_keys != expected_keys or reference_keys != expected_keys:
        raise ValueError("same-camera runtime stage-key parity mismatch")
    distances = []
    for target, reference in zip(target_records, reference_records, strict=True):
        if (
            target.get("run_id") != reference.get("run_id")
            or type(target.get("event_cursor")) is not int
            or type(reference.get("event_cursor")) is not int
            or int(reference["event_cursor"]) >= int(target["event_cursor"])
        ):
            raise ValueError(
                "same-camera reference must be strictly earlier in the same run"
            )
        target_lab = target.get("median_lab")
        reference_lab = reference.get("median_lab")
        if (
            not isinstance(target_lab, list)
            or not isinstance(reference_lab, list)
            or len(target_lab) != 3
            or len(reference_lab) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (*target_lab, *reference_lab)
            )
        ):
            raise ValueError("same-camera runtime Lab measurement is invalid")
        distances.append(math.sqrt(sum(
            (float(left) - float(right)) ** 2
            for left, right in zip(target_lab, reference_lab, strict=True)
        )))
    return tuple(distances)


def _decide_same_camera_reference_vials(
    sequence: Mapping[str, object],
    measurements_by_case: Mapping[str, Sequence[Mapping[str, object]] | None],
    expected_vials: int,
) -> list[dict[str, object]]:
    """Emit causal vial decisions using only the captured position-2 reference."""

    operators: list[dict[str, object]] = []
    for run, block in _same_camera_reference_blocks(sequence, expected_vials):
        first, reference, target = block
        for vial, reason in (
            (first, "no_prior_reference"),
            (reference, "reference_control"),
        ):
            record = _insufficient_record(
                run=run, vial=vial, source_bindings={}, reason_codes=[reason]
            )
            measured = measurements_by_case.get(str(vial["case_id"]))
            if measured is not None:
                record["measurement"] = {
                    "status": "measured_reference_control" if reason == "reference_control" else "measured_no_prior_reference",
                    "evidence_ids": [item["evidence_id"] for item in measured],
                }
            record["limit_codes"] = [reason]
            operators.append(record)
        target_rows = measurements_by_case.get(str(target["case_id"]))
        reference_rows = measurements_by_case.get(str(reference["case_id"]))
        if target_rows is None or reference_rows is None:
            operators.append(_insufficient_record(
                run=run,
                vial=target,
                source_bindings={},
                reason_codes=["measurement_unavailable"],
            ))
            continue
        distances = _same_camera_stage_distances(
            target, reference, target_rows, reference_rows
        )
        if any(distance >= 2.0 for distance in distances):
            disposition = "discard"
            reasons = ["camera_colour_distance_at_or_above_discard_threshold"]
        elif all(distance <= 1.0 for distance in distances):
            disposition = "keep"
            reasons = ["camera_colour_distance_at_or_below_keep_threshold"]
        else:
            disposition = "insufficient_evidence"
            reasons = ["camera_colour_distance_between_frozen_thresholds"]
        operators.append({
            "schema_version": "carryover-operator-record-v1",
            "run_id": run["run_id"],
            "case_id": target["case_id"],
            "vial_ordinal": target["vial_ordinal"],
            "recipe_id": target["recipe_id"],
            "previous_recipe_id": target["previous_recipe_id"],
            "disposition": disposition,
            "reason_codes": reasons,
            "limit_codes": [],
            "prediction": {
                "status": "same_camera_second_vial_comparison",
                "evidence_ids": [item["evidence_id"] for item in target_rows],
            },
            "measurement": {
                "status": "measured",
                "evidence_ids": [item["evidence_id"] for item in target_rows],
                "stage_delta_e76": list(distances),
                "distance_metric": "CIE76",
                "keep_threshold_delta_e76": 1.0,
                "discard_threshold_delta_e76": 2.0,
                "stage_citations": [
                    {
                        "stage_ordinal": target_row["stage_ordinal"],
                        "tank_id": target_row["tank_id"],
                        "target_evidence_id": target_row["evidence_id"],
                        "target_image_id": target_row["image_id"],
                        "reference_evidence_id": reference_row["evidence_id"],
                        "reference_image_id": reference_row["image_id"],
                    }
                    for target_row, reference_row in zip(
                        target_rows, reference_rows, strict=True
                    )
                ],
            },
            "history": {
                "status": "same_camera_prior_reference",
                "query_sha256": None,
                "evidence_ids": [item["evidence_id"] for item in reference_rows],
                "reference_case_id": reference["case_id"],
                "reference_vial_ordinal": reference["vial_ordinal"],
                "reference_block_position": 2,
                "reference_refs": [
                    {
                        "evidence_id": item["evidence_id"],
                        "image_id": item["image_id"],
                        "stage_ordinal": item["stage_ordinal"],
                        "tank_id": item["tank_id"],
                    }
                    for item in reference_rows
                ],
            },
            "explanation": {
                "status": "complete" if disposition != "insufficient_evidence" else "insufficient_evidence",
                "method": "deterministic_same_camera_second_vial_cie76",
                "guard_result": "pass" if disposition != "insufficient_evidence" else "insufficient_evidence",
                "image_ids_sent": [],
                "text": (
                    "Compared accepted RGB-CNN measurements with the same-camera "
                    "earlier second-vial reference using frozen CIE76 thresholds."
                ),
            },
            "source_bindings": {},
        })
    return operators


def _project_same_camera_reference_vials(
    *,
    sequence: Mapping[str, object],
    runtime_output: RuntimeProjectionInput,
    expected_vials: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Project live RGB-CNN measurements and apply the causal triplet rule."""

    _same_camera_reference_blocks(sequence, expected_vials)
    binding_by_image = _trusted_runtime_bindings_by_image(runtime_output)
    templates_by_vial = _validated_runtime_templates_by_vial(
        sequence, runtime_output
    )
    measured_by_case: dict[
        str, Sequence[Mapping[str, object]] | None
    ] = {}
    explained: list[dict[str, object]] = []
    for run, vial in _vial_rows(sequence):
        templates = templates_by_vial.get(
            (str(run["run_id"]), str(vial["case_id"])), []
        )
        expected_order = list(vial["dispense_order"])
        if len(templates) != len(expected_order):
            measured_by_case[str(vial["case_id"])] = None
            continue
        bindings = [
            binding_by_image.get(template.get("image_id")) for template in templates
        ]
        if any(binding is None for binding in bindings) or any(
            binding is not None and _measurement_is_unavailable(binding)
            for binding in bindings
        ):
            measured_by_case[str(vial["case_id"])] = None
            continue
        projected = project_runtime_measurements(RuntimeProjectionInput(
            tuple(templates), runtime_output.trusted_adapter_result,
            runtime_output.image_locators, runtime_output.event_cursors,
            runtime_output.runtime_provenance,
        ))
        projected.sort(key=lambda row: int(row["stage_ordinal"]))
        if [row["stage_ordinal"] for row in projected] != list(
            range(1, len(expected_order) + 1)
        ):
            raise ValueError("same-camera runtime stage sequence mismatch")
        enriched = []
        for record in projected:
            _assert_runtime_stage_recipe_identity(
                record, run=run, vial=vial, sequence=sequence
            )
            item = dict(record)
            item["event_cursor"] = runtime_output.event_cursors[
                str(record["evidence_id"])
            ]
            enriched.append(item)
            explained.append(dict(record))
        measured_by_case[str(vial["case_id"])] = enriched
    operators = _decide_same_camera_reference_vials(
        sequence, measured_by_case, expected_vials
    )
    operators_by_case = {str(record["case_id"]): record for record in operators}
    for record in explained:
        operator = operators_by_case[str(record["case_id"])]
        decision_id = _decision_identifier("WP7-SAME-CAMERA-DECISION", {
            "run_id": record["run_id"], "case_id": record["case_id"],
            "vial_ordinal": record["vial_ordinal"],
            "disposition": operator["disposition"],
            "measurement_evidence_ids": operator["measurement"]["evidence_ids"],
            "reference_evidence_ids": operator["history"]["evidence_ids"],
        })
        explanation_id = _decision_identifier(
            "WP7-SAME-CAMERA-EXPLANATION",
            {"decision_id": decision_id, "explanation": operator["explanation"]},
        )
        record.update({
            "decision_id": decision_id,
            "disposition": operator["disposition"],
            "limit_codes": list(operator.get("limit_codes", operator["reason_codes"])),
            "history_refs": list(operator["history"]["evidence_ids"]),
            "explanation_id": explanation_id,
            "model_revision": (
                "deterministic-template:same-camera-earlier-second-vial-cie76-v1"
            ),
            "prompt_sha256": _canonical_hash(operator["explanation"]),
            "image_ids_sent": [],
            "guard_result": operator["explanation"]["guard_result"],
        })
    return explained, operators, len(explained)


def _project_and_decide_vials(
    *,
    sequence: Mapping[str, object],
    runtime_output: RuntimeProjectionInput,
    development_store: DevelopmentReferenceStore,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Project live capabilities per vial and decide complete usable programs."""

    binding_by_image = _trusted_runtime_bindings_by_image(runtime_output)
    templates_by_vial = _validated_runtime_templates_by_vial(
        sequence, runtime_output
    )

    explained: list[dict[str, object]] = []
    operators: list[dict[str, object]] = []
    measured_stage_count = 0
    for run, vial in _vial_rows(sequence):
        key = (str(run["run_id"]), str(vial["case_id"]))
        templates = templates_by_vial.get(key, [])
        reason_codes: list[str] = []
        expected_order = list(vial["dispense_order"])
        if len(templates) != len(expected_order):
            reason_codes.append("measurement_unavailable")
        else:
            images = [template.get("image_id") for template in templates]
            relevant = [binding_by_image.get(image) for image in images]
            unavailable = [
                binding for binding in relevant
                if binding is not None and _measurement_is_unavailable(binding)
            ]
            if any(binding is None for binding in relevant):
                reason_codes.append("measurement_unavailable")
            if unavailable:
                reason_codes.extend(
                    code
                    for binding in unavailable
                    for code in ("measurement_unavailable", *binding.limit_codes)
                )
        if reason_codes:
            operators.append(_insufficient_record(
                run=run, vial=vial, source_bindings={}, reason_codes=reason_codes
            ))
            continue

        measured = project_runtime_measurements(RuntimeProjectionInput(
            tuple(templates), runtime_output.trusted_adapter_result,
            runtime_output.image_locators, runtime_output.event_cursors,
            runtime_output.runtime_provenance,
        ))
        measured.sort(key=lambda row: int(row["stage_ordinal"]))
        measured_stage_count += len(measured)
        if [row["stage_ordinal"] for row in measured] != list(
            range(1, len(expected_order) + 1)
        ):
            raise ValueError("runtime stages do not match the validated programmed recipe")
        for record in measured:
            _assert_runtime_stage_recipe_identity(
                record, run=run, vial=vial, sequence=sequence
            )

        targets = []
        queries: dict[tuple[str, int, str], DevelopmentReferenceQuery | None] = {}
        program = []
        for record in measured:
            binding = binding_by_image[str(record["image_id"])]
            identity = {
                "vial_id": f"{record['run_id']}:{record['vial_ordinal']}",
                "recipe_id": record["recipe_id"], "stage_id": record["stage_id"],
                "stage_ordinal": record["stage_ordinal"], "tank_id": record["tank_id"],
                "source": "runtime_camera",
            }
            target = bind_runtime_camera_stage(identity, binding)
            targets.append(target)
            program.append({key: identity[key] for key in (
                "recipe_id", "stage_id", "stage_ordinal", "tank_id"
            )})
            reference_target = bind_runtime_target_evidence({
                "evidence_id": record["evidence_id"],
                "event_cursor": runtime_output.event_cursors[record["evidence_id"]],
                "recipe_id": record["recipe_id"], "stage_id": record["stage_id"],
                "stage_ordinal": record["stage_ordinal"], "tank_id": record["tank_id"],
                "split_role": record["split_role"], "mask_source": record["mask_source"],
            })
            selector = (
                str(record["recipe_id"]), int(record["stage_ordinal"]),
                str(record["tank_id"]),
            )
            try:
                queries[selector] = development_store.reference_bank(reference_target)
            except DevelopmentReferenceError as error:
                if str(error) != "runtime target configuration unavailable":
                    raise
                queries[selector] = None

        decision_id = _decision_identifier("WP7-DECISION", {
            "run_id": run["run_id"], "case_id": vial["case_id"],
            "vial_ordinal": vial["vial_ordinal"],
            "programmed_stages": program,
            "target_measurement_sha256s": [target.measurement_sha256 for target in targets],
            "development_reference_store_snapshot_sha256": development_store.snapshot_sha256,
        })
        result = decide_wp7_final(decision_id, targets, queries, program)
        explanation_payload = {
            "template_id": "wp7-camera-colour-synthetic-reference-v1",
            "explanation": result["explanation"],
        }
        explanation_id = _decision_identifier("WP7-EXPLANATION", {
            "decision_id": decision_id, "payload": explanation_payload,
        })
        prompt_sha256 = _canonical_hash(explanation_payload)
        limits = sorted(set(result["limit_codes"]))
        for record in measured:
            record.update({
                "decision_id": decision_id,
                "disposition": result["disposition"],
                "limit_codes": limits,
                "history_refs": [],
                "explanation_id": explanation_id,
                "model_revision": "deterministic-template:wp7-camera-colour-synthetic-reference-v1",
                "prompt_sha256": prompt_sha256,
                "image_ids_sent": [],
                "guard_result": "pass",
            })
            explained.append(record)
        operators.append({
            **result["operator_record"],
            "run_id": run["run_id"], "case_id": vial["case_id"],
            "vial_ordinal": vial["vial_ordinal"], "recipe_id": vial["recipe_id"],
            "previous_recipe_id": vial["previous_recipe_id"],
        })
    return explained, operators, measured_stage_count


def _run_one_pass(
    *,
    phase: str,
    expected_role: str,
    video: Path,
    recipe_sequence: Path,
    freeze: Path,
    expected_freeze_sha256: str,
    evidence_out_store: Path,
    out: Path,
    claim: Path,
    owner_id: str,
    mask_runtime: PredictedMaskRuntime | None,
    mask_runtime_loader: Callable[[], PredictedMaskRuntime] | None,
    development_reference_store: Path | None,
    development_reference_store_sha256: str | None,
    development_reference_store_loader: Callable[[], DevelopmentReferenceStore] | None,
    preflight_receipt: object,
    expected_preflight_sha256: str,
    selected_mode: str,
    capture_plan: Path | None,
    capture_plan_sha256: str | None,
    capture_receipt_out: Path | None,
    capture_provenance_out: Path | None,
    capture_runtime_loader: Callable[
        [bytes, Mapping[str, object], str, str], PredictedMaskRuntime
    ] | None,
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
        "evidence_out_store": Path(evidence_out_store).resolve(),
        "output": output_path.resolve(),
        "claim": claim_path.resolve(),
        "journal": journal_path.resolve(),
    }
    if _occupied(output_path):
        raise FileExistsError(output_path)
    if _occupied(journal_path):
        raise FileExistsError(journal_path)
    if _occupied(Path(evidence_out_store)):
        raise FileExistsError(evidence_out_store)
    capture_values = (
        capture_plan,
        capture_plan_sha256,
        capture_receipt_out,
        capture_provenance_out,
        capture_runtime_loader,
    )
    capture_enabled = all(value is not None for value in capture_values)
    if any(value is not None for value in capture_values) and not capture_enabled:
        raise ValueError("capture plan, SHA-256, outputs and loader are required together")
    if selected_mode in {
        "development_reference_backed", "same_camera_second_vial_reference"
    } and not capture_enabled:
        raise ValueError(
            f"{selected_mode} mode requires the complete capture boundary"
        )
    if capture_enabled:
        assert capture_plan is not None
        assert capture_plan_sha256 is not None
        assert capture_receipt_out is not None
        assert capture_provenance_out is not None
        _sha(capture_plan_sha256, "capture plan raw-file SHA-256")
        for role, path in (
            ("capture receipt output", capture_receipt_out),
            ("capture provenance output", capture_provenance_out),
        ):
            if _occupied(Path(path)):
                raise FileExistsError(f"{role} already exists: {path}")
        resolved_paths.update({
            "capture_plan": Path(capture_plan).resolve(),
            "capture_receipt_out": Path(capture_receipt_out).resolve(),
            "capture_provenance_out": Path(capture_provenance_out).resolve(),
        })
        if mask_runtime is not None or mask_runtime_loader is not None:
            raise ValueError("capture runtime loader cannot be combined with a legacy runtime")
    ordered: Mapping[str, object] | None = None
    if selected_mode == "development_reference_backed":
        ordered = (
            validated_preflight["ordered_video"] if phase == "run-test" else None
        )
        history = validated_preflight["history"]
        assert isinstance(history, Mapping)
        if phase == "run-test":
            assert isinstance(ordered, Mapping)
        if development_reference_store is None or development_reference_store_sha256 is None:
            raise ValueError("development reference store path and SHA-256 are required")
        if development_reference_store_loader is None:
            raise ValueError("development reference store loader is required")
        _sha(development_reference_store_sha256, "development reference store SHA-256")
        if development_reference_store_sha256 != history["store_snapshot_sha256"]:
            raise ValueError("development reference store SHA-256 does not match preflight")
        resolved_paths["development_reference_store"] = Path(
            development_reference_store
        ).resolve()
        if (
            phase == "run-test"
            and resolved_paths["video"]
            != Path(str(ordered["ordered_video_path"])).resolve()
        ):
            raise ValueError("ordered video path identity does not match development preflight")
    elif any(value is not None for value in (
        development_reference_store,
        development_reference_store_sha256,
        development_reference_store_loader,
    )):
        raise ValueError(
            "development reference store arguments require "
            "development_reference_backed mode"
        )
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise ValueError("production input and operational paths must be distinct")
    freeze_receipt = _load_freeze(Path(freeze), expected_freeze_sha256)
    claim_receipt = _claim_payload(
        phase=phase,
        owner_id=owner_id,
        freeze_sha256=expected_freeze_sha256,
        video=video,
        recipe_sequence=recipe_sequence,
        evidence_out_store=evidence_out_store,
        development_reference_store=development_reference_store,
        out=out,
        preflight_sha256=preflight_sha256,
        selected_mode=selected_mode,
        capture_plan=capture_plan,
        capture_plan_sha256=capture_plan_sha256,
        capture_receipt_out=capture_receipt_out,
        capture_provenance_out=capture_provenance_out,
    )
    _publish_json_exclusive(claim_path, claim_receipt)
    journal_sequence = 0
    sequence: dict[str, object] | None = None
    claimed_video_raw: bytes | None = None
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
                "capture_plan_sha256": capture_plan_sha256,
                "capture_receipt_out_path": (
                    None
                    if capture_receipt_out is None
                    else str(Path(capture_receipt_out).resolve())
                ),
                "capture_provenance_out_path": (
                    None
                    if capture_provenance_out is None
                    else str(Path(capture_provenance_out).resolve())
                ),
            },
        )
        journal_sequence = 1
        if selected_mode == "development_reference_backed" and phase == "run-test":
            assert ordered is not None
            claimed_video_raw = _read_production_bytes(
                resolved_paths["video"], "ordered test video"
            )
            if hashlib.sha256(claimed_video_raw).hexdigest() != ordered["ordered_video_sha256"]:
                raise ValueError("ordered video SHA-256 does not match development preflight")
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
        video_raw = (
            claimed_video_raw
            if claimed_video_raw is not None
            else _read_production_bytes(Path(video).resolve(), "camera video")
        )
        source_bindings = {
            "video_sha256": hashlib.sha256(video_raw).hexdigest(),
            "recipe_sequence_sha256": hashlib.sha256(recipe_raw).hexdigest(),
            "freeze_sha256": expected_freeze_sha256,
            "claim_sha256": str(claim_receipt["claim_sha256"]),
            "preflight_sha256": preflight_sha256,
            "selected_mode": selected_mode,
            "capture_plan_path": (
                None if capture_plan is None else str(Path(capture_plan).resolve())
            ),
            "capture_plan_raw_sha256": capture_plan_sha256,
            "evidence_out_store_binding_sha256": _canonical_hash(
                {
                    "path": str(Path(evidence_out_store).resolve()),
                    "schema": freeze_receipt["store"],
                }
            ),
            "development_reference_store_binding_sha256": (
                None
                if development_reference_store is None
                else _canonical_hash({
                    "path": str(Path(development_reference_store).resolve()),
                    "snapshot_sha256": development_reference_store_sha256,
                })
            ),
        }
        accepted_real_route = bool(freeze_receipt["mask_runtime"]["accepts_real_rgb_only"])
        skip = phase == "run-real-check" and (
            selected_mode == "evaluation_only" or not accepted_real_route
        )
        bound_runtime = mask_runtime
        if not skip and capture_enabled:
            assert capture_runtime_loader is not None
            bound_runtime = capture_runtime_loader(
                video_raw,
                sequence,
                phase,
                str(source_bindings["video_sha256"]),
            )
            assert capture_receipt_out is not None
            assert capture_provenance_out is not None
            capture_receipt_raw = _read_regular_bytes(
                Path(capture_receipt_out), "published capture receipt"
            )
            capture_provenance_raw = _read_regular_bytes(
                Path(capture_provenance_out), "published capture provenance"
            )
            assert capture_plan is not None
            assert capture_plan_sha256 is not None
            source_bindings.update(_bind_published_capture(
                capture_raw=capture_receipt_raw,
                provenance_raw=capture_provenance_raw,
                phase=phase,
                video_sha256=str(source_bindings["video_sha256"]),
                capture_plan=capture_plan,
                capture_plan_sha256=capture_plan_sha256,
                capture_receipt_out=capture_receipt_out,
                capture_provenance_out=capture_provenance_out,
            ))
        elif not skip and bound_runtime is None and mask_runtime_loader is not None:
            bound_runtime = mask_runtime_loader()
        source_bindings.setdefault("capture_receipt_path", None)
        source_bindings.setdefault("capture_receipt_raw_sha256", None)
        source_bindings.setdefault("capture_receipt_sha256", None)
        source_bindings.setdefault("capture_provenance_path", None)
        source_bindings.setdefault("capture_provenance_raw_sha256", None)
        _append_journal(
            journal_path,
            {
                "schema_version": "carryover-phase-journal-event-v1",
                "sequence": 2,
                "phase": phase,
                "event": "inputs_bound",
                "video_sha256": source_bindings["video_sha256"],
                "recipe_sequence_sha256": source_bindings["recipe_sequence_sha256"],
                "capture_plan_raw_sha256": source_bindings[
                    "capture_plan_raw_sha256"
                ],
                "capture_receipt_raw_sha256": source_bindings[
                    "capture_receipt_raw_sha256"
                ],
                "capture_provenance_raw_sha256": source_bindings[
                    "capture_provenance_raw_sha256"
                ],
                "preflight_sha256": preflight_sha256,
                "selected_mode": selected_mode,
            },
        )
        journal_sequence = 2
        runtime_bound = (
            bound_runtime is not None
            and isinstance(bound_runtime, PredictedMaskRuntime)
            and bound_runtime.runtime_id == freeze_receipt["mask_runtime"]["runtime_id"]
            and bound_runtime.runtime_sha256 == freeze_receipt["mask_runtime"]["runtime_sha256"]
            and bound_runtime.config_sha256 == freeze_receipt["mask_runtime"]["config_sha256"]
            and (phase != "run-real-check" or bound_runtime.accepts_real_rgb_only)
        )
        if (
            selected_mode in {
                "development_reference_backed",
                "same_camera_second_vial_reference",
            }
            and not skip
            and not runtime_bound
        ):
            raise ValueError(
                "selected reference runtime is missing or does not match the freeze"
            )
        explained_records: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        measured_stage_count = 0
        evidence_out_store_sha256: str | None = None
        development_snapshot_sha256: str | None = None
        runtime_provenance: Mapping[str, object] | None = None
        runtime_output: RuntimeProjectionInput | None = None
        if runtime_bound and not skip:
            assert bound_runtime is not None
            runtime_output = bound_runtime.analyze(
                video_bytes=video_raw,
                recipe_sequence=sequence,
                artifact_root=output_path.parent,
            )
            runtime_provenance = runtime_output.runtime_provenance
        if (
            selected_mode == "development_reference_backed"
            and runtime_output is not None
            and not skip
        ):
            assert development_reference_store_loader is not None
            _prove_runtime_projection_before_reference_load(sequence, runtime_output)
            development_store = development_reference_store_loader()
            if not isinstance(development_store, DevelopmentReferenceStore):
                raise TypeError("development reference loader returned an invalid store")
            assert development_reference_store is not None
            if development_store.database.resolve() != Path(
                development_reference_store
            ).resolve():
                raise ValueError("development reference loader path identity mismatch")
            assert development_reference_store_sha256 is not None
            if development_store.snapshot_sha256 != development_reference_store_sha256:
                raise ValueError("development reference loader snapshot identity mismatch")
            explained_records, records, measured_stage_count = _project_and_decide_vials(
                sequence=sequence,
                runtime_output=runtime_output,
                development_store=development_store,
            )
            try:
                evidence_store = EvidenceStore(
                    Path(evidence_out_store), artifact_root=output_path.parent
                )
                if explained_records:
                    image_ids = {str(record["image_id"]) for record in explained_records}
                    evidence_ids = {str(record["evidence_id"]) for record in explained_records}
                    evidence_store.ingest(
                        explained_records,
                        {
                            image_id: runtime_output.image_locators[image_id]
                            for image_id in image_ids
                        },
                        phase="explained",
                        event_cursors={
                            evidence_id: runtime_output.event_cursors[evidence_id]
                            for evidence_id in evidence_ids
                        },
                    )
            except Exception:
                Path(evidence_out_store).unlink(missing_ok=True)
                raise
            evidence_out_store_sha256 = hashlib.sha256(
                Path(evidence_out_store).read_bytes()
            ).hexdigest()
            development_snapshot_sha256 = development_store.snapshot_sha256
        elif (
            selected_mode == "same_camera_second_vial_reference"
            and runtime_output is not None
            and not skip
        ):
            explained_records, records, measured_stage_count = (
                _project_same_camera_reference_vials(
                    sequence=sequence,
                    runtime_output=runtime_output,
                    expected_vials=expected_vials,
                )
            )
            try:
                evidence_store = EvidenceStore(
                    Path(evidence_out_store), artifact_root=output_path.parent
                )
                if explained_records:
                    image_ids = {
                        str(record["image_id"]) for record in explained_records
                    }
                    evidence_ids = {
                        str(record["evidence_id"]) for record in explained_records
                    }
                    evidence_store.ingest(
                        explained_records,
                        {
                            image_id: runtime_output.image_locators[image_id]
                            for image_id in image_ids
                        },
                        phase="explained",
                        event_cursors={
                            evidence_id: runtime_output.event_cursors[evidence_id]
                            for evidence_id in evidence_ids
                        },
                    )
            except Exception:
                Path(evidence_out_store).unlink(missing_ok=True)
                raise
            evidence_out_store_sha256 = hashlib.sha256(
                Path(evidence_out_store).read_bytes()
            ).hexdigest()
        else:
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
                    run=run, vial=vial, source_bindings=source_bindings,
                    reason_codes=reason_codes,
                )
                for run, vial in _vial_rows(sequence)
            ]
        source_bindings["evidence_out_store_sha256"] = evidence_out_store_sha256
        source_bindings[
            "development_reference_store_snapshot_sha256"
        ] = development_snapshot_sha256
        for record in records:
            if record.get("schema_version") == "carryover-operator-record-v1":
                record["source_bindings"] = dict(source_bindings)
        count = len(records)
        decisions = sum(record["disposition"] in {"keep", "discard"} for record in records)
        abstentions = sum(record["disposition"] == "insufficient_evidence" for record in records)
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
                abstentions=0 if skip else abstentions,
                skips=count if skip else 0,
                decisions=decisions,
            ),
            "provenance_inventory": {
                "process_evidence": ["camera_video", "programmed_recipe_sequence"],
                "operational_bindings": [
                    "freeze",
                    "preflight",
                    "evidence_out_store",
                    "claim",
                    "output",
                ] + (
                    ["capture_plan", "capture_receipt", "capture_provenance"]
                    if source_bindings["capture_receipt_raw_sha256"] is not None
                    else []
                ),
                "model_loaded": bool(runtime_bound and not skip),
                "gpu_used": False,
                "measured_stage_count": measured_stage_count,
                "explained_stage_count": len(explained_records),
                "runtime_provenance": runtime_provenance,
                "development_reference_store_snapshot_sha256": development_snapshot_sha256,
                "evidence_out_store_sha256": evidence_out_store_sha256,
                "external_development_donor_citations": (
                    "none; same-camera earlier second-vial references are cited per supported stage"
                    if selected_mode == "same_camera_second_vial_reference"
                    else (
                        "immutable terminal operator records; flat evidence "
                        "history_refs remain empty"
                    )
                ),
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
                "capture_plan_raw_sha256": source_bindings[
                    "capture_plan_raw_sha256"
                ],
                "capture_receipt_raw_sha256": source_bindings[
                    "capture_receipt_raw_sha256"
                ],
                "capture_provenance_raw_sha256": source_bindings[
                    "capture_provenance_raw_sha256"
                ],
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
    evidence_out_store: Path,
    out: Path,
    claim: Path,
    owner_id: str,
    preflight_receipt: object,
    expected_preflight_sha256: str,
    selected_mode: str,
    mask_runtime: PredictedMaskRuntime | None = None,
    mask_runtime_loader: Callable[[], PredictedMaskRuntime] | None = None,
    development_reference_store: Path | None = None,
    development_reference_store_sha256: str | None = None,
    development_reference_store_loader: Callable[[], DevelopmentReferenceStore] | None = None,
    capture_plan: Path | None = None,
    capture_plan_sha256: str | None = None,
    capture_receipt_out: Path | None = None,
    capture_provenance_out: Path | None = None,
    capture_runtime_loader: Callable[
        [bytes, Mapping[str, object], str, str], PredictedMaskRuntime
    ] | None = None,
) -> dict[str, object]:
    """Seal the one allowed held-out test attempt; a claim is never reusable."""

    return _run_one_pass(
        phase="run-test",
        expected_role="test",
        video=video,
        recipe_sequence=recipe_sequence,
        freeze=freeze,
        expected_freeze_sha256=expected_freeze_sha256,
        evidence_out_store=evidence_out_store,
        out=out,
        claim=claim,
        owner_id=owner_id,
        mask_runtime=mask_runtime,
        mask_runtime_loader=mask_runtime_loader,
        development_reference_store=development_reference_store,
        development_reference_store_sha256=development_reference_store_sha256,
        development_reference_store_loader=development_reference_store_loader,
        preflight_receipt=preflight_receipt,
        expected_preflight_sha256=expected_preflight_sha256,
        selected_mode=selected_mode,
        capture_plan=capture_plan,
        capture_plan_sha256=capture_plan_sha256,
        capture_receipt_out=capture_receipt_out,
        capture_provenance_out=capture_provenance_out,
        capture_runtime_loader=capture_runtime_loader,
    )


def run_real_check(
    *,
    video: Path,
    recipe_sequence: Path,
    freeze: Path,
    expected_freeze_sha256: str,
    evidence_out_store: Path,
    out: Path,
    claim: Path,
    owner_id: str,
    preflight_receipt: object,
    expected_preflight_sha256: str,
    selected_mode: str,
    mask_runtime: PredictedMaskRuntime | None = None,
    mask_runtime_loader: Callable[[], PredictedMaskRuntime] | None = None,
    development_reference_store: Path | None = None,
    development_reference_store_sha256: str | None = None,
    development_reference_store_loader: Callable[[], DevelopmentReferenceStore] | None = None,
    capture_plan: Path | None = None,
    capture_plan_sha256: str | None = None,
    capture_receipt_out: Path | None = None,
    capture_provenance_out: Path | None = None,
    capture_runtime_loader: Callable[
        [bytes, Mapping[str, object], str, str], PredictedMaskRuntime
    ] | None = None,
) -> dict[str, object]:
    """Seal the distinct real-check attempt or its explicit RGB-route skip."""

    return _run_one_pass(
        phase="run-real-check",
        expected_role="real_check",
        video=video,
        recipe_sequence=recipe_sequence,
        freeze=freeze,
        expected_freeze_sha256=expected_freeze_sha256,
        evidence_out_store=evidence_out_store,
        out=out,
        claim=claim,
        owner_id=owner_id,
        mask_runtime=mask_runtime,
        mask_runtime_loader=mask_runtime_loader,
        development_reference_store=development_reference_store,
        development_reference_store_sha256=development_reference_store_sha256,
        development_reference_store_loader=development_reference_store_loader,
        preflight_receipt=preflight_receipt,
        expected_preflight_sha256=expected_preflight_sha256,
        selected_mode=selected_mode,
        capture_plan=capture_plan,
        capture_plan_sha256=capture_plan_sha256,
        capture_receipt_out=capture_receipt_out,
        capture_provenance_out=capture_provenance_out,
        capture_runtime_loader=capture_runtime_loader,
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

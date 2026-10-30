"""Populate a fresh WP5 store from accepted F2 RGB-mask predictions.

All 42 prediction rows are provenance-accounted. Only the 36 train and
validation rows enter the trusted registry, measurement adapter, and production
store; evaluation rows remain visible solely in accounting receipts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile

from .evidence_store import EvidenceStore, canonical_source_record_sha256
from .rgb_mask_prediction import build_trusted_predicted_mask_registry_v1
from .vision_measurement import measure_trusted_predicted_image


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OWNER = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_PUBLIC_ROLES = ("train", "validation", "evaluation")
_ROLE_ORDER = {role: index for index, role in enumerate(_PUBLIC_ROLES)}
_FORBIDDEN_KEYS = {
    "oracle", "oracle_mask", "oracle_masks", "semantic", "semantic_mask",
    "semantic_masks", "label", "labels", "sim_metadata", "palette", "composition",
}
_SAFE_TOP_FIELDS = {
    "schema_version", "artifact_type", "status", "data_origin", "split_version",
    "source_downstream_validation_receipt_sha256",
    "source_request_materialization_receipt_sha256", "source_image_locators_sha256",
    "denominators", "records", "claim_boundary",
}
_SAFE_RECORD_FIELDS = {
    "batch_id", "run_id", "case_id", "vial_id", "vial_position", "recipe_id",
    "previous_recipe_id", "split_role", "split_version", "recipe_ml",
    "dispense_order", "source_receipt_sha256", "video_path", "video_sha256",
    "video_decoded_frame_count", "video_playback_fps", "video_codec",
    "video_container", "fill_frame_range", "excluded_post_fill_frame_range", "stages",
}
_SAFE_STAGE_FIELDS = {"image_id", "stage_id", "stage_ordinal", "tank_id"}
_MEASURED_TOP_FIELDS = {
    "schema_version", "artifact_type", "status", "data_origin", "real_data_used",
    "split_version", "source_manifests", "stage_measurements", "same_run_stage_pairs",
    "measurement_contract", "denominators", "claim_boundary", "mask_source",
    "mask_prediction_producer",
}
_MEASUREMENT_REQUIRED_FIELDS = {
    "evidence_id", "measurement_id", "image_id", "batch_id", "case_key", "run_id",
    "recipe_id", "split_role", "split_version", "vial_id", "stage_id",
    "stage_ordinal", "tank_id", "view_id", "source_manifest_sha256", "rgb_sha256",
    "mask_sha256", "median_lab", "mad_lab", "iqr_lab", "liquid_pixel_count",
    "mask_coverage_fraction", "mask_height_px", "lab_unit",
}
_MEASUREMENT_FIELDS = _MEASUREMENT_REQUIRED_FIELDS | {
    "median_srgb_0_255", "mad_srgb_0_255"
}
_SOURCE_MANIFEST_FIELDS = {
    "batch_id", "run_id", "recipe_id", "split_role", "sha256",
    "active_stage_count", "stage_measurement_count",
}
_PAIR_REQUIRED_FIELDS = {
    "pair_id", "batch_id", "run_id", "recipe_id", "split_role", "split_version",
    "stage_id", "stage_ordinal", "tank_id", "left_vial_id", "right_vial_id",
    "left_evidence_id", "right_evidence_id", "source_manifest_sha256",
    "left_rgb_sha256", "right_rgb_sha256", "left_mask_sha256", "right_mask_sha256",
    "delta_e76", "delta_e76_6dp", "unit",
}
_MATERIALIZATION_TOP_FIELDS = {
    "schema_version", "artifact_type", "status", "materializer", "predictor",
    "accepted_evidence", "safe_manifest", "source_bindings", "prediction_locators",
    "denominators", "inference_inputs", "recipe_invariance", "records",
}
_MATERIALIZATION_RECORD_FIELDS = {
    "image_id", "batch_id", "run_id", "vial_id", "split_role", "stage_id",
    "stage_ordinal", "rgb_path", "rgb_sha256", "reference_rgb_path",
    "reference_rgb_sha256", "mask_path", "mask_sha256", "prediction_status",
    "surface_row", "visible_height_px", "changed_pixel_count",
    "source_capture_manifest_path", "source_capture_manifest_sha256",
    "source_receipt_path", "source_receipt_sha256", "history_eligible",
    "history_eligibility_reason",
}
_MATERIALIZER_FILES = {
    "src/fluidmix/predicted_mask_materializer.py",
    "scripts/materialize_empty_reference_predictions.py",
}
# This is the independently reviewed production publication.  The bridge accepts
# caller-selected paths only when their bytes and materializer identity match this
# frozen profile; the path itself remains relocatable.
ACCEPTED_MATERIALIZATION_RECEIPT_SHA256 = (
    "a9a559a3614a23c362b646d4608ee4d3f20f1a1d0599c9534946a2b79ece71e3"
)
ACCEPTED_PREDICTION_LOCATORS_SHA256 = (
    "578ab949803214b12cd4942cff7adeda3e89610eff92523505cf7c0436e5e269"
)
ACCEPTED_MATERIALIZER_COMMIT = "d89a3f9fca31e7a2a37c89290189b3a5fa0583da"
CNN_PRODUCER_ID = "frozen-wp5-six-channel-cnn"
CNN_REPLAY_SHA256 = "a4b9d412ce063fa6449fbfcacf49b1b3ba3e7d64724fd93d96381c467ee4836a"
CNN_REVIEW_SHA256 = "17e1bb4809898c95b3af59cb6d187955a7bf36b4f6fffb39e342e07a59f2c8df"
_PREDICTOR_SAFE_TOP_FIELDS = {
    "schema_version", "artifact_type", "data_origin", "input_fields",
    "threshold_source", "split_version", "accepted_evidence", "source_bindings",
    "records",
}
_PREDICTOR_SAFE_RECORD_FIELDS = {
    "image_id", "batch_id", "run_id", "vial_id", "split_role", "stage_id",
    "stage_ordinal", "rgb_path", "rgb_sha256", "reference_rgb_path",
    "reference_rgb_sha256", "source_capture_manifest_path",
    "source_capture_manifest_sha256", "source_receipt_path", "source_receipt_sha256",
}
_ACCEPTED_EVIDENCE_FIELDS = {
    "recovered_result_path", "recovered_result_sha256",
    "independent_review_path", "independent_review_sha256",
}
_PREDICTION_SOURCE_BINDING_FIELDS = {
    "image_locators_path", "image_locators_sha256",
    "downstream_validation_receipt_path", "downstream_validation_receipt_sha256",
    "request_bundle_path", "request_bundle_sha256",
}


class PredictedStoreBridgeError(ValueError):
    """The predicted-mask store handoff failed closed."""


def _fail(code: str, error: BaseException | None = None) -> PredictedStoreBridgeError:
    failure = PredictedStoreBridgeError(code)
    if error is not None:
        failure.__cause__ = error
    return failure


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path_value: str | os.PathLike[str], role: str) -> tuple[dict[str, object], Path, str]:
    path = Path(path_value).expanduser().absolute()
    try:
        resolved = path.resolve(strict=True)
        if resolved != path or path.is_symlink() or not path.is_file():
            raise ValueError("not an exact regular file")
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise _fail(f"{role}_invalid", error)
    if not isinstance(value, dict):
        raise _fail(f"{role}_invalid")
    return value, path, hashlib.sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{role}_invalid")
    return value


def _exact_fields(value: object, expected: set[str], role: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise _fail(f"{role}_fields_invalid")
    return value


def _sequence(value: object, role: str) -> list[object]:
    if not isinstance(value, list):
        raise _fail(f"{role}_invalid")
    return value


def _reject_forbidden_keys(value: object, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise _fail("non_text_field_name")
            if key.lower() in _FORBIDDEN_KEYS:
                raise _fail(f"forbidden_production_field:{path}.{key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _bound_artifact_path(
    root: Path, path_value: object, expected_sha: object, role: str
) -> tuple[Path, str]:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise _fail(f"{role}_path_invalid")
    path = Path(path_value).absolute()
    expected = _sha(expected_sha, f"{role}_sha256")
    try:
        resolved = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise _fail(f"{role}_missing", error)
    if (
        resolved != path
        or path.is_symlink()
        or not path.is_file()
        or not path.is_relative_to(resolved_root)
        or _hash_file(path) != expected
    ):
        raise _fail(f"{role}_binding_invalid")
    return path, path.relative_to(resolved_root).as_posix()


def _verify_file_entry(root: Path, entry: Mapping[str, object], role: str) -> Path:
    relative = entry.get("path")
    expected_bytes = entry.get("bytes")
    expected_sha = _sha(entry.get("sha256"), f"{role}_sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or type(expected_bytes) is not int
        or expected_bytes < 0
    ):
        raise _fail(f"{role}_file_entry_invalid")
    path = (root / relative).absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _fail(f"{role}_file_missing", error)
    if (
        resolved != path
        or path.is_symlink()
        or not path.is_file()
        or not path.is_relative_to(root)
        or path.stat().st_size != expected_bytes
        or _hash_file(path) != expected_sha
    ):
        raise _fail(f"{role}_file_binding_invalid")
    return path


def _verify_declared_source(path_value: object, sha_value: object, role: str) -> Path:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise _fail(f"{role}_path_invalid")
    expected = _sha(sha_value, f"{role}_sha256")
    path = Path(path_value).absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _fail(f"{role}_missing", error)
    if (
        resolved != path
        or path.is_symlink()
        or not path.is_file()
        or _hash_file(path) != expected
    ):
        raise _fail(f"{role}_binding_invalid")
    return path


def _validate_capture_provenance(
    downstream: dict[str, object],
    downstream_path: Path,
    downstream_sha: str,
    safe: dict[str, object],
    safe_records: list[dict[str, object]],
    measured_sources: list[dict[str, object]],
) -> list[dict[str, object]]:
    if (
        downstream.get("schema_version") != 2
        or downstream.get("artifact_type") != "f2_varied_v3_downstream_validation_receipt"
        or downstream.get("status") != "pass"
        or downstream.get("batch_count") != 6
        or downstream.get("vial_count") != 18
        or downstream.get("stage_pair_count") != 42
        or downstream.get("source_artifact_validation_count") != 18
    ):
        raise _fail("downstream_validation_receipt_identity_invalid")
    if safe.get("source_downstream_validation_receipt_sha256") != downstream_sha:
        raise _fail("safe_manifest_downstream_receipt_hash_mismatch")
    materialization = downstream.get("materialization_receipt")
    if (
        not isinstance(materialization, Mapping)
        or safe.get("source_request_materialization_receipt_sha256")
        != materialization.get("sha256")
    ):
        raise _fail("materialization_receipt_binding_mismatch")
    capture_hashes = downstream.get("capture_manifest_sha256_by_batch")
    publication_hashes = downstream.get("publication_receipt_sha256_by_batch")
    runtime_hashes = downstream.get("runtime_receipt_sha256_by_batch")
    expected_batches = {str(row["batch_id"]) for row in safe_records}
    for mapping, role in (
        (capture_hashes, "capture_manifest"),
        (publication_hashes, "publication_receipt"),
        (runtime_hashes, "runtime_receipt"),
    ):
        if not isinstance(mapping, dict) or set(mapping) != expected_batches:
            raise _fail(f"{role}_batch_map_invalid")
        for digest in mapping.values():
            _sha(digest, f"{role}_sha256")
    measured_capture_hashes = {
        str(row["batch_id"]): row.get("sha256") for row in measured_sources
    }
    if measured_capture_hashes != {
        batch_id: capture_hashes[batch_id] for batch_id in measured_capture_hashes
    }:
        raise _fail("measured_capture_manifest_binding_mismatch")
    files = _sequence(downstream.get("files"), "downstream_files")
    by_relative: dict[str, Mapping[str, object]] = {}
    for raw in files:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise _fail("downstream_file_entry_invalid")
        relative = str(raw["path"])
        if relative in by_relative:
            raise _fail("downstream_file_entry_duplicate")
        by_relative[relative] = raw
    root = downstream_path.parent.resolve(strict=True)
    safe_by_batch_vial = {
        (str(row["batch_id"]), str(row["vial_id"])): row for row in safe_records
    }
    provenance = []
    for batch_id in sorted(expected_batches):
        capture_relative = f"publications/{batch_id}/f2-stage-capture-manifest.json"
        publication_relative = f"publications/{batch_id}/publication-receipt.json"
        capture_entry = by_relative.get(capture_relative)
        publication_entry = by_relative.get(publication_relative)
        if capture_entry is None or publication_entry is None:
            raise _fail("accepted_capture_file_missing")
        capture_path = _verify_file_entry(root, capture_entry, "capture_manifest")
        publication_path = _verify_file_entry(root, publication_entry, "publication_receipt")
        if capture_entry.get("sha256") != capture_hashes[batch_id]:
            raise _fail("capture_manifest_hash_mismatch")
        if publication_entry.get("sha256") != publication_hashes[batch_id]:
            raise _fail("publication_receipt_hash_mismatch")
        publication, _, observed_publication_sha = _load_json(
            publication_path, "publication_receipt"
        )
        if (
            observed_publication_sha != publication_hashes[batch_id]
            or publication.get("schema_version") != 3
            or publication.get("artifact_type") != "f2_stage_batch_publication_receipt"
            or publication.get("status") != "published"
        ):
            raise _fail("publication_receipt_identity_invalid")
        outputs = publication.get("outputs")
        manifest_output = outputs.get("f2-stage-capture-manifest.json") if isinstance(outputs, Mapping) else None
        if (
            not isinstance(manifest_output, Mapping)
            or manifest_output.get("sha256") != capture_hashes[batch_id]
            or manifest_output.get("bytes") != capture_path.stat().st_size
        ):
            raise _fail("publication_capture_manifest_binding_mismatch")
        source_receipts = _sequence(publication.get("source_receipts"), "source_receipts")
        seen_vials: set[str] = set()
        receipt_provenance: list[dict[str, object]] = []
        for raw_receipt in source_receipts:
            if not isinstance(raw_receipt, Mapping):
                raise _fail("source_receipt_invalid")
            vial_id = raw_receipt.get("vial_id")
            key = (batch_id, str(vial_id))
            safe_row = safe_by_batch_vial.get(key)
            if (
                safe_row is None
                or vial_id in seen_vials
                or raw_receipt.get("case_id") != safe_row["case_id"]
                or raw_receipt.get("vial_position") != safe_row["vial_position"]
                or raw_receipt.get("receipt_sha256") != safe_row["source_receipt_sha256"]
            ):
                raise _fail("source_receipt_binding_mismatch")
            seen_vials.add(str(vial_id))
            receipt_path_value = raw_receipt.get("receipt_path")
            if not isinstance(receipt_path_value, str) or not Path(receipt_path_value).is_absolute():
                raise _fail("source_receipt_path_invalid")
            receipt_path = Path(receipt_path_value).absolute()
            try:
                resolved_receipt = receipt_path.resolve(strict=True)
            except OSError as error:
                raise _fail("source_receipt_missing", error)
            if (
                resolved_receipt != receipt_path
                or receipt_path.is_symlink()
                or not receipt_path.is_file()
                or _hash_file(receipt_path) != raw_receipt["receipt_sha256"]
            ):
                raise _fail("source_receipt_file_binding_invalid")
            receipt_provenance.append(
                {
                    "case_id": raw_receipt["case_id"],
                    "vial_id": vial_id,
                    "path": str(receipt_path),
                    "sha256": raw_receipt["receipt_sha256"],
                }
            )
        if seen_vials != {"V1", "V2", "V3"}:
            raise _fail("source_receipt_denominator_invalid")
        provenance.append(
            {
                "batch_id": batch_id,
                "capture_manifest_path": str(capture_path),
                "capture_manifest_sha256": capture_hashes[batch_id],
                "publication_receipt_sha256": publication_hashes[batch_id],
                "runtime_receipt_sha256": runtime_hashes[batch_id],
                "source_receipts": sorted(
                    receipt_provenance, key=lambda row: str(row["vial_id"])
                ),
            }
        )
    return provenance


def _validate_prediction_materialization(
    value: dict[str, object],
    *,
    prediction_path: Path,
    prediction_sha: str,
    registry: object,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    _exact_fields(value, _MATERIALIZATION_TOP_FIELDS, "prediction_materialization")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type")
        != "empty_reference_rgb_prediction_materialization_receipt"
        or value.get("status") != "materialized"
    ):
        raise _fail("prediction_materialization_identity_invalid")
    _validate_materializer_provenance(value.get("materializer"))
    predictor = value.get("predictor")
    locator_producer = registry.producer
    if (
        not isinstance(predictor, dict)
        or set(predictor) != {*locator_producer, "config"}
        or {key: predictor[key] for key in locator_producer} != locator_producer
    ):
        raise _fail("prediction_materialization_predictor_mismatch")
    try:
        config_sha = hashlib.sha256(
            json.dumps(
                predictor["config"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as error:
        raise _fail("prediction_materialization_predictor_config_invalid", error)
    if config_sha != locator_producer["config_sha256"]:
        raise _fail("prediction_materialization_predictor_config_hash_mismatch")
    if value.get("inference_inputs") != {
        "input_fields": ["rgb"],
        "oracle_masks_opened": 0,
        "semantic_masks_opened": 0,
        "sim_metadata_inputs": 0,
        "label_inputs": 0,
        "palette_inputs": 0,
        "composition_inputs": 0,
        "recipe_ml_inputs": 0,
        "video_inputs": 0,
    }:
        raise _fail("prediction_materialization_inference_inputs_invalid")
    expected_recovery_check = (
        "cnn_model_inputs_exclude_recipe_ml"
        if locator_producer.get("producer_id") == CNN_PRODUCER_ID
        else "programmed_ml_stage_outputs_invariant"
    )
    if value.get("recipe_invariance") != {
        "recipe_fields_present_in_safe_manifest": False,
        "programmed_ml_argument_supplied": False,
        "accepted_recovery_check": expected_recovery_check,
    }:
        raise _fail("prediction_materialization_recipe_invariance_invalid")
    locator_binding = value.get("prediction_locators")
    if (
        not isinstance(locator_binding, dict)
        or set(locator_binding) != {"path", "sha256"}
        or locator_binding.get("path") != str(prediction_path)
        or locator_binding.get("sha256") != prediction_sha
    ):
        raise _fail("prediction_materialization_locator_binding_mismatch")
    safe_binding = value.get("safe_manifest")
    if (
        not isinstance(safe_binding, dict)
        or set(safe_binding) != {"path", "sha256"}
    ):
        raise _fail("prediction_materialization_safe_manifest_binding_mismatch")
    predictor_safe_path = _verify_declared_source(
        safe_binding.get("path"), safe_binding.get("sha256"), "prediction_safe_manifest"
    )
    predictor_safe, _, observed_safe_sha = _load_json(
        predictor_safe_path, "prediction_safe_manifest"
    )
    if observed_safe_sha != safe_binding["sha256"]:
        raise _fail("prediction_safe_manifest_hash_mismatch")
    _exact_fields(predictor_safe, _PREDICTOR_SAFE_TOP_FIELDS, "prediction_safe_manifest")
    if (
        predictor_safe.get("schema_version") != 1
        or predictor_safe.get("artifact_type") != "empty_reference_rgb_production_inputs"
        or predictor_safe.get("data_origin") != "twin"
        or predictor_safe.get("input_fields") != ["rgb"]
        or predictor_safe.get("threshold_source") != locator_producer["threshold_source"]
        or predictor_safe.get("split_version") != "whole-run-varied-v3"
    ):
        raise _fail("prediction_safe_manifest_identity_invalid")
    accepted_evidence = _exact_fields(
        value.get("accepted_evidence"),
        _ACCEPTED_EVIDENCE_FIELDS,
        "prediction_accepted_evidence",
    )
    source_bindings = _exact_fields(
        value.get("source_bindings"),
        _PREDICTION_SOURCE_BINDING_FIELDS,
        "prediction_source_bindings",
    )
    safe_accepted_evidence = _exact_fields(
        predictor_safe.get("accepted_evidence"),
        _ACCEPTED_EVIDENCE_FIELDS,
        "prediction_safe_accepted_evidence",
    )
    if locator_producer.get("producer_id") == CNN_PRODUCER_ID:
        if (
            accepted_evidence.get("recovered_result_sha256") != CNN_REPLAY_SHA256
            or accepted_evidence.get("independent_review_sha256") != CNN_REVIEW_SHA256
        ):
            raise _fail("prediction_cnn_selection_evidence_invalid")
    elif safe_accepted_evidence != accepted_evidence:
        raise _fail("prediction_accepted_evidence_safe_binding_mismatch")
    if predictor_safe.get("source_bindings") != source_bindings:
        raise _fail("prediction_source_bindings_safe_binding_mismatch")
    for container, fields, prefix in (
        (accepted_evidence, _ACCEPTED_EVIDENCE_FIELDS, "prediction_accepted_evidence"),
        (
            safe_accepted_evidence,
            _ACCEPTED_EVIDENCE_FIELDS,
            "prediction_safe_accepted_evidence",
        ),
        (source_bindings, _PREDICTION_SOURCE_BINDING_FIELDS, "prediction_source_binding"),
    ):
        for path_field in sorted(field for field in fields if field.endswith("_path")):
            stem = path_field.removesuffix("_path")
            _verify_declared_source(
                container[path_field], container[f"{stem}_sha256"], f"{prefix}_{stem}"
            )
    expected_denominators = {
        "images": 42,
        "vials": 18,
        "runs": 6,
        "split_images": {"train": 30, "validation": 6, "evaluation": 6},
        "history_eligible_images": 36,
        "history_excluded_evaluation_images": 6,
    }
    if value.get("denominators") != expected_denominators:
        raise _fail("prediction_materialization_denominator_invalid")
    raw_records = _sequence(value.get("records"), "prediction_materialization_records")
    if len(raw_records) != 42:
        raise _fail("prediction_materialization_record_denominator_invalid")
    records: dict[str, dict[str, object]] = {}
    raw_safe_records = _sequence(
        predictor_safe.get("records"), "prediction_safe_manifest_records"
    )
    if len(raw_safe_records) != 42:
        raise _fail("prediction_safe_manifest_record_denominator_invalid")
    safe_records: dict[str, dict[str, object]] = {}
    for raw_safe in raw_safe_records:
        safe_record = _exact_fields(
            raw_safe, _PREDICTOR_SAFE_RECORD_FIELDS, "prediction_safe_manifest_record"
        )
        safe_image_id = safe_record.get("image_id")
        if not isinstance(safe_image_id, str) or safe_image_id in safe_records:
            raise _fail("prediction_safe_manifest_record_identity_invalid")
        safe_records[safe_image_id] = safe_record
    for raw in raw_records:
        record = _exact_fields(
            raw, _MATERIALIZATION_RECORD_FIELDS, "prediction_materialization_record"
        )
        image_id = record.get("image_id")
        role = record.get("split_role")
        eligible = record.get("history_eligible")
        reason = record.get("history_eligibility_reason")
        if (
            not isinstance(image_id, str)
            or image_id in records
            or role not in _PUBLIC_ROLES
            or type(eligible) is not bool
            or not isinstance(reason, str)
            or not reason
        ):
            raise _fail("prediction_materialization_record_identity_invalid")
        expected_eligible = role in {"train", "validation"}
        expected_reason = (
            "accepted_non_evaluation_partition"
            if expected_eligible
            else "evaluation_partition_excluded"
        )
        if eligible is not expected_eligible or reason != expected_reason:
            raise _fail("prediction_materialization_history_eligibility_invalid")
        for field in (
            "rgb_sha256", "reference_rgb_sha256", "mask_sha256",
            "source_capture_manifest_sha256", "source_receipt_sha256",
        ):
            _sha(record.get(field), f"prediction_materialization_{field}")
        for field in (
            "rgb_path", "reference_rgb_path", "mask_path",
            "source_capture_manifest_path", "source_receipt_path",
        ):
            if not isinstance(record.get(field), str) or not Path(str(record[field])).is_absolute():
                raise _fail(f"prediction_materialization_{field}_invalid")
        for field in ("stage_ordinal", "surface_row", "visible_height_px", "changed_pixel_count"):
            if type(record.get(field)) is not int or int(record[field]) < 0:
                raise _fail(f"prediction_materialization_{field}_invalid")
        # Source capture/receipt provenance is safe to rehash. Evaluation pixel paths
        # deliberately remain unopened and are bound only by this receipt's own hash.
        for path_field, sha_field, source_role in (
            ("source_capture_manifest_path", "source_capture_manifest_sha256", "source_capture_manifest"),
            ("source_receipt_path", "source_receipt_sha256", "source_receipt"),
        ):
            source_path = Path(str(record[path_field])).absolute()
            try:
                resolved = source_path.resolve(strict=True)
            except OSError as error:
                raise _fail(f"prediction_materialization_{source_role}_missing", error)
            if (
                resolved != source_path
                or source_path.is_symlink()
                or not source_path.is_file()
                or _hash_file(source_path) != record[sha_field]
            ):
                raise _fail(f"prediction_materialization_{source_role}_binding_invalid")
        records[image_id] = deepcopy(record)
        if image_id not in safe_records or any(
            record[field] != safe_records[image_id][field]
            for field in _PREDICTOR_SAFE_RECORD_FIELDS
        ):
            raise _fail("prediction_materialization_safe_record_mismatch")
    counts = Counter(str(row["split_role"]) for row in records.values())
    if counts != {"train": 30, "validation": 6, "evaluation": 6}:
        raise _fail("prediction_materialization_split_denominator_invalid")
    eligible_ids = {
        image_id for image_id, row in records.items() if row["history_eligible"] is True
    }
    if eligible_ids != set(registry.image_ids):
        raise _fail("prediction_materialization_registry_set_mismatch")
    if set(records) != set(safe_records):
        raise _fail("prediction_materialization_safe_record_set_mismatch")
    for image_id in registry.image_ids:
        locator = registry.entry(image_id).locator()
        record = records[image_id]
        if any(record[field] != locator[field] for field in ("rgb_path", "rgb_sha256", "mask_path", "mask_sha256")):
            raise _fail("prediction_materialization_locator_record_mismatch")
    ordered = sorted(
        records.values(),
        key=lambda row: (
            _ROLE_ORDER[str(row["split_role"])], str(row["run_id"]),
            str(row["batch_id"]), int(str(row["vial_id"])[1:]),
            int(row["stage_ordinal"]), str(row["image_id"]),
        ),
    )
    return records, ordered


def _validate_accepted_publication_identity(
    value: Mapping[str, object], *, materialization_sha: str, prediction_sha: str
) -> None:
    """Require the independently accepted receipt, locator, and materializer."""

    materializer = value.get("materializer")
    if (
        materialization_sha != ACCEPTED_MATERIALIZATION_RECEIPT_SHA256
        or prediction_sha != ACCEPTED_PREDICTION_LOCATORS_SHA256
        or not isinstance(materializer, Mapping)
        or materializer.get("code_commit") != ACCEPTED_MATERIALIZER_COMMIT
    ):
        raise _fail("accepted_publication_identity_mismatch")


def _validate_materializer_provenance(value: object) -> None:
    materializer = _exact_fields(
        value,
        {"code_commit", "repository_root", "tracked_files", "relevant_worktree_clean"},
        "prediction_materializer",
    )
    commit = materializer.get("code_commit")
    root_value = materializer.get("repository_root")
    tracked = materializer.get("tracked_files")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(root_value, str)
        or not Path(root_value).is_absolute()
        or materializer.get("relevant_worktree_clean") is not True
        or not isinstance(tracked, dict)
        or set(tracked) != _MATERIALIZER_FILES
    ):
        raise _fail("prediction_materializer_identity_invalid")
    root = Path(root_value).absolute()
    try:
        if root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
            raise ValueError("repository root is not canonical")
        observed_head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", *sorted(_MATERIALIZER_FILES)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise _fail("prediction_materializer_repository_invalid", error)
    if observed_head != commit or status:
        raise _fail("prediction_materializer_worktree_binding_invalid")
    for relative in sorted(_MATERIALIZER_FILES):
        binding = tracked[relative]
        expected_path = (root / relative).absolute()
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(expected_path)
        ):
            raise _fail("prediction_materializer_tracked_file_fields_invalid")
        expected_sha = _sha(binding.get("sha256"), "prediction_materializer_tracked_file_sha256")
        try:
            if (
                expected_path.resolve(strict=True) != expected_path
                or expected_path.is_symlink()
                or not expected_path.is_file()
                or _hash_file(expected_path) != expected_sha
            ):
                raise ValueError("tracked worktree file drifted")
            head_bytes = subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"],
                check=True,
                capture_output=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise _fail("prediction_materializer_tracked_file_binding_invalid", error)
        if hashlib.sha256(head_bytes).hexdigest() != expected_sha:
            raise _fail("prediction_materializer_head_blob_mismatch")


def _validate_safe_manifest(value: dict[str, object]) -> list[dict[str, object]]:
    _exact_fields(value, _SAFE_TOP_FIELDS, "fill_clip_manifest")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "f2_fill_clip_input_manifest"
        or value.get("status") != "published"
        or value.get("data_origin") != "twin"
        or value.get("split_version") != "whole-run-varied-v3"
        or value.get("denominators")
        != {"batches": 6, "runs": 6, "vials": 18, "stages": 42}
    ):
        raise _fail("fill_clip_manifest_identity_invalid")
    rows = _sequence(value.get("records"), "fill_clip_records")
    if len(rows) != 18:
        raise _fail("fill_clip_record_denominator_invalid")
    checked: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    stage_ids: set[str] = set()
    for raw in rows:
        row = _exact_fields(raw, _SAFE_RECORD_FIELDS, "fill_clip_record")
        role = row.get("split_role")
        if role not in _PUBLIC_ROLES or role == "test":
            raise _fail("split_role_invalid")
        identity = (str(row.get("batch_id")), str(row.get("vial_id")))
        if identity in identities:
            raise _fail("fill_clip_record_duplicate")
        identities.add(identity)
        if row.get("vial_id") not in {"V1", "V2", "V3"} or row.get("vial_position") != int(str(row["vial_id"])[1]):
            raise _fail("vial_identity_invalid")
        if row.get("split_version") != value["split_version"]:
            raise _fail("split_version_mismatch")
        stages = _sequence(row.get("stages"), "fill_clip_stages")
        order = row.get("dispense_order")
        if not isinstance(order, list) or len(stages) not in {2, 3} or len(stages) != len(order):
            raise _fail("stage_denominator_invalid")
        for ordinal, raw_stage in enumerate(stages, 1):
            stage = _exact_fields(raw_stage, _SAFE_STAGE_FIELDS, "fill_clip_stage")
            image_id = stage.get("image_id")
            if (
                not isinstance(image_id, str)
                or image_id in stage_ids
                or stage.get("stage_ordinal") != ordinal
                or stage.get("tank_id") != order[ordinal - 1]
            ):
                raise _fail("fill_clip_stage_order_invalid")
            stage_ids.add(image_id)
        fill_range = row.get("fill_frame_range")
        exit_range = row.get("excluded_post_fill_frame_range")
        if (
            not isinstance(fill_range, list)
            or not isinstance(exit_range, list)
            or len(fill_range) != 2
            or len(exit_range) != 2
            or any(type(number) is not int for number in (*fill_range, *exit_range))
            or not (0 <= fill_range[0] < fill_range[1] == exit_range[0] < exit_range[1])
            or exit_range[1] != row.get("video_decoded_frame_count")
        ):
            raise _fail("fill_frame_range_invalid")
        checked.append(deepcopy(row))
    if len(stage_ids) != 42:
        raise _fail("fill_clip_stage_denominator_invalid")
    counts = Counter(str(row["split_role"]) for row in checked)
    if counts != {"train": 12, "validation": 3, "evaluation": 3}:
        raise _fail("fill_clip_split_denominator_invalid")
    return checked


def _validate_measured_header(value: dict[str, object], producer: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    _exact_fields(value, _MEASURED_TOP_FIELDS, "measured_evidence")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "f2_varied_stage_measured_evidence"
        or value.get("status") != "measured"
        or value.get("data_origin") != "twin"
        or value.get("real_data_used") is not False
        or value.get("split_version") != "whole-run-varied-v3"
        or value.get("mask_source") != "rgb_only_prediction"
        or value.get("mask_prediction_producer") != producer
    ):
        raise _fail("measured_evidence_identity_invalid")
    expected_denominators = {
        "source_manifests": 5, "runs": 5, "vials": 15,
        "active_stage_events": 12, "stage_measurements": 36,
        "same_run_stage_pairs": 36, "two_stage_runs": 3, "three_stage_runs": 2,
    }
    if value.get("denominators") != expected_denominators:
        raise _fail("measured_evidence_denominator_invalid")
    sources = _sequence(value.get("source_manifests"), "source_manifests")
    measurements = _sequence(value.get("stage_measurements"), "stage_measurements")
    pairs = _sequence(value.get("same_run_stage_pairs"), "same_run_stage_pairs")
    if len(sources) != 5 or len(measurements) != 36 or len(pairs) != 36:
        raise _fail("measured_evidence_row_denominator_invalid")
    if any(not isinstance(row, dict) for row in (*sources, *measurements, *pairs)):
        raise _fail("measured_evidence_row_invalid")
    if any(set(row) != _SOURCE_MANIFEST_FIELDS for row in sources):
        raise _fail("source_manifest_fields_invalid")
    if any(set(row) != _MEASUREMENT_FIELDS for row in measurements):
        raise _fail("stage_measurement_fields_invalid")
    if any(set(row) != _PAIR_REQUIRED_FIELDS for row in pairs):
        raise _fail("pairwise_fields_invalid")
    source_by_batch: dict[str, dict[str, object]] = {}
    for row in sources:
        batch_id = row.get("batch_id")
        if (
            not isinstance(batch_id, str)
            or batch_id in source_by_batch
            or row.get("split_role") not in {"train", "validation"}
        ):
            raise _fail("source_manifest_identity_invalid")
        _sha(row.get("sha256"), "source_manifest_sha256")
        source_by_batch[batch_id] = row
    if Counter(str(row["split_role"]) for row in sources) != {"train": 4, "validation": 1}:
        raise _fail("source_manifest_split_denominator_invalid")
    for row in measurements:
        source = source_by_batch.get(str(row.get("batch_id")))
        if source is None or any(
            row.get(field) != source.get(field)
            for field in ("run_id", "recipe_id", "split_role")
        ) or row.get("source_manifest_sha256") != source.get("sha256"):
            raise _fail("stage_measurement_source_manifest_mismatch")
    return deepcopy(sources), deepcopy(measurements), deepcopy(pairs)


def _close_vector(left: object, right: object, role: str) -> None:
    if (
        not isinstance(left, list)
        or not isinstance(right, (list, tuple))
        or len(left) != 3
        or len(right) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in left
        )
        or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(left, right, strict=True))
    ):
        raise _fail(f"{role}_mismatch")


def _validate_measurements(
    rows: list[dict[str, object]], registry: object
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if {str(row.get("image_id")) for row in rows} != set(registry.image_ids):
        raise _fail("prediction_measurement_image_set_mismatch")
    by_evidence: dict[str, dict[str, object]] = {}
    binding_receipts: dict[str, dict[str, object]] = {}
    for row in rows:
        if set(row) != _MEASUREMENT_FIELDS:
            raise _fail("stage_measurement_fields_invalid")
        evidence_id = row.get("evidence_id")
        image_id = row.get("image_id")
        measurement_id = row.get("measurement_id")
        if (
            not isinstance(evidence_id, str)
            or not isinstance(image_id, str)
            or not isinstance(measurement_id, str)
            or evidence_id in by_evidence
        ):
            raise _fail("stage_measurement_identity_invalid")
        measured, binding = measure_trusted_predicted_image(
            registry.entry(image_id),
            measurement_id=measurement_id,
            evidence_id=evidence_id,
        )
        if measured.get("validity") != "valid":
            raise _fail(f"trusted_prediction_measurement_invalid:{image_id}")
        for field in ("rgb_sha256", "mask_sha256", "liquid_pixel_count", "mask_height_px"):
            if row.get(field) != measured.get(field):
                raise _fail(f"trusted_measurement_{field}_mismatch")
        coverage = row.get("mask_coverage_fraction")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or abs(float(coverage) - float(measured["mask_coverage_fraction"])) > 1e-15
        ):
            raise _fail("trusted_measurement_mask_coverage_fraction_mismatch")
        for field in ("median_lab", "mad_lab", "iqr_lab"):
            _close_vector(row.get(field), measured.get(field), f"trusted_measurement_{field}")
        if row.get("lab_unit") != "cie_lab_d65_2deg":
            raise _fail("stage_measurement_unit_invalid")
        by_evidence[evidence_id] = row
        binding_receipts[evidence_id] = binding.receipt()
    return by_evidence, binding_receipts


def _validate_pairs(
    pairs: list[dict[str, object]], measurements: Mapping[str, dict[str, object]]
) -> dict[str, list[dict[str, object]]]:
    seen_pair_ids: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    directed: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        if set(row) != _PAIR_REQUIRED_FIELDS:
            raise _fail("pairwise_fields_invalid")
        pair_id = row.get("pair_id")
        left_id = row.get("left_evidence_id")
        right_id = row.get("right_evidence_id")
        if (
            not isinstance(pair_id, str)
            or pair_id in seen_pair_ids
            or not isinstance(left_id, str)
            or not isinstance(right_id, str)
            or left_id == right_id
            or left_id not in measurements
            or right_id not in measurements
        ):
            raise _fail("pairwise_identity_invalid")
        seen_pair_ids.add(pair_id)
        edge = tuple(sorted((left_id, right_id)))
        if edge in seen_edges:
            raise _fail("pairwise_duplicate")
        seen_edges.add(edge)
        left, right = measurements[left_id], measurements[right_id]
        context = ("batch_id", "run_id", "recipe_id", "split_role", "split_version", "stage_id", "stage_ordinal", "tank_id", "source_manifest_sha256")
        if any(row.get(field) != left.get(field) or row.get(field) != right.get(field) for field in context):
            raise _fail("pairwise_context_mismatch")
        bindings = (
            ("left_vial_id", left, "vial_id"), ("right_vial_id", right, "vial_id"),
            ("left_rgb_sha256", left, "rgb_sha256"), ("right_rgb_sha256", right, "rgb_sha256"),
            ("left_mask_sha256", left, "mask_sha256"), ("right_mask_sha256", right, "mask_sha256"),
        )
        if any(row.get(field) != source.get(source_field) for field, source, source_field in bindings):
            raise _fail("pairwise_source_binding_mismatch")
        delta = row.get("delta_e76")
        expected = math.dist(left["median_lab"], right["median_lab"])
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
            or delta < 0
            or abs(float(delta) - expected) > 1e-12
            or row.get("delta_e76_6dp") != f"{expected:.6f}"
            or row.get("unit") != "delta_e76"
        ):
            raise _fail("pairwise_delta_e76_mismatch")
        directed[left_id].append({"other_evidence_id": right_id, "delta_e76": float(delta)})
        directed[right_id].append({"other_evidence_id": left_id, "delta_e76": float(delta)})
    if len(seen_edges) != 36 or set(directed) != set(measurements):
        raise _fail("pairwise_denominator_invalid")
    for values in directed.values():
        values.sort(key=lambda value: str(value["other_evidence_id"]))
        if len(values) != 2:
            raise _fail("pairwise_degree_invalid")
    return dict(directed)


def _ordered_measurements(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    try:
        return sorted(
            rows,
            key=lambda row: (
                _ROLE_ORDER[str(row["split_role"])], str(row["run_id"]),
                str(row["batch_id"]), int(str(row["vial_id"])[1:]),
                int(row["stage_ordinal"]), str(row["evidence_id"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _fail("stage_measurement_order_key_invalid", error)


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def populate_predicted_evidence_store(
    *,
    prediction_locators_path: str | os.PathLike[str],
    materialization_receipt_path: str | os.PathLike[str],
    measured_evidence_path: str | os.PathLike[str],
    fill_clip_manifest_path: str | os.PathLike[str],
    downstream_validation_receipt_path: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    owner_thread_id: str,
) -> dict[str, object]:
    """Validate all 42 rows and atomically publish a fresh 36-row WP5 store."""

    if not isinstance(owner_thread_id, str) or _OWNER.fullmatch(owner_thread_id) is None:
        raise _fail("owner_thread_id_invalid")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_dir_must_be_fresh")
    root = Path(artifact_root).expanduser().absolute()
    try:
        if root.resolve(strict=True) != root or not root.is_dir() or root.is_symlink():
            raise ValueError("artifact root is not canonical")
    except (OSError, ValueError) as error:
        raise _fail("artifact_root_invalid", error)

    predictions, prediction_path, prediction_sha = _load_json(
        prediction_locators_path, "prediction_locators"
    )
    materialization, materialization_path, materialization_sha = _load_json(
        materialization_receipt_path, "prediction_materialization"
    )
    measured, measured_path, measured_sha = _load_json(
        measured_evidence_path, "measured_evidence"
    )
    safe, safe_path, safe_sha = _load_json(fill_clip_manifest_path, "fill_clip_manifest")
    downstream, downstream_path, downstream_sha = _load_json(
        downstream_validation_receipt_path, "downstream_validation_receipt"
    )
    for public_value in (predictions, materialization, measured, safe):
        _reject_forbidden_keys(public_value)
    _validate_accepted_publication_identity(
        materialization,
        materialization_sha=materialization_sha,
        prediction_sha=prediction_sha,
    )

    try:
        registry = build_trusted_predicted_mask_registry_v1(predictions)
    except (TypeError, ValueError) as error:
        raise _fail("trusted_prediction_registry_invalid", error)
    if len(registry.image_ids) != 36:
        raise _fail("prediction_image_denominator_invalid")
    producer = registry.producer
    sources, measurement_rows, pair_rows = _validate_measured_header(measured, producer)
    safe_rows = _validate_safe_manifest(safe)
    materialization_records, accounting_order = _validate_prediction_materialization(
        materialization,
        prediction_path=prediction_path,
        prediction_sha=prediction_sha,
        registry=registry,
    )
    materialization_sources = materialization["source_bindings"]
    assert isinstance(materialization_sources, dict)
    if safe.get("source_image_locators_sha256") != materialization_sources.get(
        "image_locators_sha256"
    ):
        raise _fail("source_image_locators_lineage_mismatch")
    if (
        materialization_sources["downstream_validation_receipt_path"]
        != str(downstream_path)
        or materialization_sources["downstream_validation_receipt_sha256"]
        != downstream_sha
    ):
        raise _fail("prediction_downstream_validation_binding_mismatch")
    capture_provenance = _validate_capture_provenance(
        downstream, downstream_path, downstream_sha, safe, safe_rows, sources
    )
    capture_by_batch = {str(row["batch_id"]): row for row in capture_provenance}
    accepted_source_receipts = {
        (str(batch["batch_id"]), str(receipt["vial_id"])): receipt
        for batch in capture_provenance
        for receipt in batch["source_receipts"]
    }
    for accounting in materialization_records.values():
        batch_id = str(accounting["batch_id"])
        vial_id = str(accounting["vial_id"])
        capture = capture_by_batch.get(batch_id)
        source_receipt = accepted_source_receipts.get((batch_id, vial_id))
        if (
            capture is None
            or source_receipt is None
            or accounting["source_capture_manifest_path"]
            != capture["capture_manifest_path"]
            or accounting["source_capture_manifest_sha256"]
            != capture["capture_manifest_sha256"]
            or accounting["source_receipt_path"] != source_receipt["path"]
            or accounting["source_receipt_sha256"] != source_receipt["sha256"]
        ):
            raise _fail("materialization_accepted_source_provenance_mismatch")

    safe_by_image: dict[str, dict[str, object]] = {}
    safe_by_vial: dict[tuple[str, str], dict[str, object]] = {}
    for safe_row in safe_rows:
        vial_key = (str(safe_row["batch_id"]), str(safe_row["vial_id"]))
        if vial_key in safe_by_vial:
            raise _fail("safe_vial_duplicate")
        safe_by_vial[vial_key] = safe_row
        _bound_artifact_path(root, safe_row["video_path"], safe_row["video_sha256"], "fill_clip")
        for stage in safe_row["stages"]:
            image_id = str(stage["image_id"])
            if image_id in safe_by_image:
                raise _fail("safe_image_duplicate")
            safe_by_image[image_id] = {**safe_row, "stage": stage}
    if set(safe_by_image) != set(materialization_records):
        raise _fail("safe_materialization_image_set_mismatch")
    if {
        image_id
        for image_id, joined in safe_by_image.items()
        if joined["split_role"] in {"train", "validation"}
    } != set(registry.image_ids):
        raise _fail("safe_prediction_image_set_mismatch")
    for image_id, accounting in materialization_records.items():
        safe_join = safe_by_image[image_id]
        stage = safe_join["stage"]
        if any(
            accounting.get(field) != safe_join.get(field)
            for field in ("batch_id", "run_id", "vial_id", "split_role")
        ) or any(
            accounting.get(field) != stage.get(field)
            for field in ("image_id", "stage_id", "stage_ordinal")
        ) or accounting.get("source_receipt_sha256") != safe_join.get("source_receipt_sha256"):
            raise _fail("materialization_safe_context_mismatch")
    normalized_locators: dict[str, dict[str, str]] = {}
    for image_id in registry.image_ids:
        locator = registry.entry(image_id).locator()
        _, rgb_relative = _bound_artifact_path(root, locator["rgb_path"], locator["rgb_sha256"], "predicted_rgb")
        _, mask_relative = _bound_artifact_path(root, locator["mask_path"], locator["mask_sha256"], "predicted_mask")
        normalized_locators[image_id] = {
            "rgb_path": rgb_relative, "rgb_sha256": locator["rgb_sha256"],
            "mask_path": mask_relative, "mask_sha256": locator["mask_sha256"],
        }
    if len({row["rgb_path"] for row in normalized_locators.values()}) != 36 or len({row["mask_path"] for row in normalized_locators.values()}) != 36:
        raise _fail("prediction_artifact_path_duplicate")

    measurements_by_evidence, binding_receipts = _validate_measurements(measurement_rows, registry)
    pairwise = _validate_pairs(pair_rows, measurements_by_evidence)
    if {str(row["image_id"]) for row in measurement_rows} != set(registry.image_ids):
        raise _fail("measured_safe_image_set_mismatch")

    ordered = _ordered_measurements(measurement_rows)
    cursor_by_image = {
        str(row["image_id"]): index for index, row in enumerate(accounting_order)
    }
    event_cursors = {
        str(row["evidence_id"]): cursor_by_image[str(row["image_id"])] for row in ordered
    }
    records: list[dict[str, object]] = []
    accounting_rows: list[dict[str, object]] = []
    source_record_hashes: dict[str, str] = {}
    for row in ordered:
        image_id = str(row["image_id"])
        evidence_id = str(row["evidence_id"])
        safe_join = safe_by_image[image_id]
        stage = safe_join["stage"]
        assert isinstance(stage, dict)
        context_fields = ("batch_id", "run_id", "recipe_id", "split_role", "split_version", "vial_id")
        if any(row.get(field) != safe_join.get(field) for field in context_fields) or any(
            row.get(field) != stage.get(field)
            for field in ("stage_id", "stage_ordinal", "tank_id", "image_id")
        ):
            raise _fail("measurement_safe_context_mismatch")
        if row.get("case_key") != f"{row['batch_id']}:{row['vial_id']}" or row.get("view_id") != "vial_fill":
            raise _fail("measurement_source_context_invalid")
        if row.get("source_manifest_sha256") != materialization_records[image_id].get(
            "source_capture_manifest_sha256"
        ):
            raise _fail("measurement_materialization_source_manifest_mismatch")
        _, clip_relative = _bound_artifact_path(
            root, safe_join["video_path"], safe_join["video_sha256"], "fill_clip"
        )
        record = {
            "run_id": row["run_id"], "case_id": safe_join["case_id"],
            "vial_ordinal": safe_join["vial_position"], "block_recipe_id": row["batch_id"],
            "block_position": safe_join["vial_position"], "recipe_id": row["recipe_id"],
            "previous_recipe_id": safe_join["previous_recipe_id"],
            "recipe_ml": safe_join["recipe_ml"], "dispense_order": safe_join["dispense_order"],
            "split_role": row["split_role"], "split_version": row["split_version"],
            "stage_id": row["stage_id"], "tank_id": row["tank_id"],
            "stage_ordinal": row["stage_ordinal"], "fill_clip_path": clip_relative,
            "fill_frame_range": safe_join["fill_frame_range"],
            "transition_frame_ranges": {"entry": None, "exit": safe_join["excluded_post_fill_frame_range"]},
            "clip_sha256": safe_join["video_sha256"], "image_id": image_id,
            "rgb_sha256": row["rgb_sha256"], "mask_sha256": row["mask_sha256"],
            "mask_source": "predicted", "measurement_id": row["measurement_id"],
            "median_lab": row["median_lab"], "mad_lab": row["mad_lab"],
            "iqr_lab": row["iqr_lab"], "pairwise_delta_e76": pairwise[evidence_id],
            "colour_regime_id": None, "evidence_id": evidence_id,
            "decision_id": None, "disposition": None, "limit_codes": [], "history_refs": [],
            "explanation_id": None, "model_revision": None, "prompt_sha256": None,
            "image_ids_sent": [], "guard_result": None,
        }
        records.append(record)
        source_record_hashes[evidence_id] = canonical_source_record_sha256(
            record, normalized_locators[image_id]
        )
    measured_by_image = {str(row["image_id"]): row for row in ordered}
    for cursor, accounting in enumerate(accounting_order):
        image_id = str(accounting["image_id"])
        eligible = accounting["history_eligible"] is True
        measured_row = measured_by_image.get(image_id)
        if eligible != (measured_row is not None):
            raise _fail("accounting_measurement_eligibility_mismatch")
        evidence_id = f"EVID-{image_id}"
        if measured_row is not None and measured_row.get("evidence_id") != evidence_id:
            raise _fail("accounting_evidence_id_mismatch")
        binding = binding_receipts.get(evidence_id)
        accounting_rows.append(
            {
                "event_cursor": cursor,
                "evidence_id": evidence_id,
                "image_id": image_id,
                "split_role": accounting["split_role"],
                "store_disposition": "stored" if eligible else "excluded_evaluation",
                "trusted_measurement_sha256": (
                    binding["measurement_sha256"] if binding is not None else None
                ),
            }
        )
    if any(record["split_role"] not in {"train", "validation"} for record in records):
        raise _fail("production_role_leakage")
    if len(records) != 36 or len(accounting_rows) != 42:
        raise _fail("production_denominator_invalid")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "published"
        staging.mkdir()
        database = staging / "evidence.sqlite3"
        store = EvidenceStore(database, artifact_root=root)
        production_cursors = {
            str(record["evidence_id"]): event_cursors[str(record["evidence_id"])] for record in records
        }
        production_locators = {
            str(record["image_id"]): registry.entry(str(record["image_id"])).locator()
            for record in records
        }
        result = store.ingest(records, production_locators, phase="measured", event_cursors=production_cursors)
        if result.inserted != 36 or result.existing != 0:
            raise _fail("fresh_store_ingest_denominator_invalid")
        with sqlite3.connect(database) as connection:
            roles = {str(row[0]) for row in connection.execute("SELECT DISTINCT split_role FROM stages")}
            if roles != {"train", "validation"}:
                raise _fail("fresh_store_role_leakage")
            if connection.execute("SELECT count(*) FROM history_members").fetchone()[0] != 0:
                raise _fail("fresh_store_history_not_empty")
        database_sha = _hash_file(database)
        receipt = {
            "schema_version": 1,
            "artifact_type": "f2_predicted_evidence_store_population_receipt",
            "status": "populated",
            "artifact_owner": {"owner_thread_id": owner_thread_id},
            "producer": {"producer_id": "fluidmix.predicted_store_bridge", "implementation_source_sha256": _module_sha256()},
            "prediction_producer": producer,
            "trusted_predicted_mask_registry_sha256": registry.registry_sha256,
            "source_artifacts": [
                {"role": "prediction_locators", "path": str(prediction_path), "sha256": prediction_sha},
                {"role": "prediction_materialization_receipt", "path": str(materialization_path), "sha256": materialization_sha},
                {"role": "measured_evidence", "path": str(measured_path), "sha256": measured_sha},
                {"role": "fill_clip_manifest", "path": str(safe_path), "sha256": safe_sha},
                {"role": "downstream_validation_receipt", "path": str(downstream_path), "sha256": downstream_sha},
            ],
            "capture_provenance": capture_provenance,
            "prediction_materializer": deepcopy(materialization["materializer"]),
            "prediction_materialization_predictor": deepcopy(materialization["predictor"]),
            "denominators": {
                "accounted_stage_records": 42, "production_stage_records": 36,
                "excluded_evaluation_stage_records": 6, "train_stage_records": 30,
                "validation_stage_records": 6, "evaluation_stage_records": 6,
                "production_vials": 15, "excluded_evaluation_vials": 3,
                "production_runs": 5, "excluded_evaluation_runs": 1,
            },
            "accounting_rows": accounting_rows,
            "production_source_record_sha256_by_evidence": source_record_hashes,
            "database": {"path": "evidence.sqlite3", "sha256": database_sha, "bytes": database.stat().st_size, "sqlite_user_version": 3},
            "claim_boundary": (
                "All 42 synthetic predicted-mask measurements are integrity-accounted; only "
                "36 train/validation records enter production history. Evaluation records, "
                "labels, semantic/oracle masks, and simulation metadata are excluded."
            ),
        }
        (staging / "population-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        if output.exists():
            raise _fail("output_dir_must_be_fresh")
        staging.rename(output)
    return receipt


__all__ = ["PredictedStoreBridgeError", "populate_predicted_evidence_store"]

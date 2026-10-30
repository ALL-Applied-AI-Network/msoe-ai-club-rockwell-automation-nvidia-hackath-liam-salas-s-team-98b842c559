"""Truthful WP7 development-reference store intake.

The accepted INN-67 export is not yet a normal :class:`EvidenceStore` export.  This
boundary validates its published cross-bindings and fails closed with the exact normal
fields that are absent.  It never creates proxy clips, locator paths, recipe metadata,
or dispersion measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3


DONOR_BATCHES = ("INN67-DEV-RB-B01", "INN67-DEV-RB-B02")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_FIELDS = {
    "evidence_id", "event_cursor", "batch_id", "run_id", "case_id", "recipe_id",
    "split_role", "stage_id", "stage_ordinal", "tank_id", "rgb_sha256",
    "mask_sha256", "mask_source", "median_lab", "source_render_receipt_sha256",
    "source_validation_receipt_sha256", "source_render_plan_sha256",
    "known_good_profile_sha256",
}
MISSING_NORMAL_EVIDENCE_FIELDS = (
    "block_position",
    "block_recipe_id",
    "clip_sha256",
    "dispense_order",
    "fill_clip_path",
    "fill_frame_range",
    "image_id",
    "iqr_lab",
    "mad_lab",
    "mask_path",
    "measurement_id",
    "pairwise_delta_e76",
    "previous_recipe_id",
    "recipe_ml",
    "rgb_path",
    "split_version",
    "transition_frame_ranges",
    "vial_ordinal",
)


class DevelopmentReferenceError(ValueError):
    """Development-reference publication was incomplete or cross-bound incorrectly."""


def _fail(code: str, detail: str | None = None) -> DevelopmentReferenceError:
    return DevelopmentReferenceError(code if detail is None else f"{code}:{detail}")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    try:
        body = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _fail("noncanonical_json") from error
    return _hash_bytes(body)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result


def _load(path_value: str | os.PathLike[str], role: str) -> tuple[dict[str, object], Path, str]:
    path = Path(path_value).expanduser().absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("not an exact regular file")
        body = path.read_bytes()
        value = json.loads(
            body,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _fail(f"{role}_invalid") from error
    if not isinstance(value, dict):
        raise _fail(f"{role}_invalid")
    return value, path, _hash_bytes(body)


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise _fail(f"{role}_sha256_invalid")
    return value


def _bound_file(path_value: object, digest_value: object, role: str) -> Path:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise _fail(f"{role}_path_invalid")
    expected = _sha(digest_value, role)
    path = Path(path_value).absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("not an exact regular file")
        observed = _hash_bytes(path.read_bytes())
    except OSError as error:
        raise _fail(f"{role}_file_invalid") from error
    if observed != expected:
        raise _fail(f"{role}_hash_mismatch")
    return path


def validate_development_reference_inputs(
    *,
    predicted_store_export_path: str | os.PathLike[str],
    development_reference_bank_path: str | os.PathLike[str],
    delta_e_frozen_config_path: str | os.PathLike[str],
    materialization_receipt_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Validate the accepted tuple and report the normal-store publication blocker."""

    export, _, export_sha = _load(predicted_store_export_path, "predicted_store_export")
    bank, _, bank_sha = _load(development_reference_bank_path, "development_reference_bank")
    config, _, config_sha = _load(delta_e_frozen_config_path, "delta_e_frozen_config")
    receipt, _, receipt_sha = _load(materialization_receipt_path, "materialization_receipt")

    if (
        set(export)
        != {"schema_version", "artifact_type", "status", "predictor", "store", "records"}
        or set(bank) != {
            "schema_version", "artifact_type", "status", "split_role",
            "reference_donor_batch_ids", "configuration_target_batch_id", "predictor",
            "denominators", "records", "history_bindings", "claim_boundary", "provenance",
        }
        or set(config) != {
            "schema_version", "artifact_type", "status", "distance_metric", "unit",
            "config_id", "tight_peer_pair_max_delta_e76",
            "target_outlier_margin_delta_e76", "decision_rule", "threshold_formula",
            "configuration_target_gate", "selection_input_batch_ids",
            "reference_geometry_batch_ids", "recipe_stage_configurations",
            "recipe_stage_configuration_count", "wp1_test_accessed", "video5_accessed",
            "claim_boundary", "provenance",
        }
        or set(receipt) != {
            "schema_version", "artifact_type", "status", "cpu_only",
            "gpu_or_renderer_launched", "image_or_video_bytes_opened",
            "wp1_test_accessed", "video5_accessed", "outputs", "provenance",
        }
        or export.get("schema_version") != 1
        or export.get("artifact_type") != "inn67_predicted_store_export"
        or export.get("status") != "accepted"
        or not isinstance(export.get("records"), list)
        or len(export["records"]) != 48
        or not isinstance(export.get("store"), Mapping)
    ):
        raise _fail("predicted_store_export_contract_invalid")
    store = export["store"]
    if set(store) != {
        "database_path", "database_sha256", "population_receipt_path",
        "population_receipt_sha256",
    }:
        raise _fail("predicted_store_tuple_fields_invalid")
    database = _bound_file(store.get("database_path"), store.get("database_sha256"), "predicted_database")
    population_path = _bound_file(
        store.get("population_receipt_path"),
        store.get("population_receipt_sha256"),
        "population_receipt",
    )
    population, _, population_sha = _load(population_path, "population_receipt")
    records = export["records"]
    if (
        population.get("artifact_type") != "inn67_predicted_evidence_store_population"
        or population.get("status") != "pass"
        or population.get("store_sha256") != store.get("database_sha256")
        or population.get("record_count") != 48
        or population.get("records_sha256") != _canonical_hash(records)
        or population.get("evidence_ids")
        != [row.get("evidence_id") if isinstance(row, Mapping) else None for row in records]
        or population_sha != store.get("population_receipt_sha256")
        or population.get("predictor") != export.get("predictor")
    ):
        raise _fail("population_export_binding_mismatch")
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            rows = connection.execute(
                "SELECT event_cursor, evidence_id, record_json, record_sha256 "
                "FROM predicted_records ORDER BY event_cursor"
            ).fetchall()
    except sqlite3.Error as error:
        raise _fail("predicted_database_invalid") from error
    expected_rows = []
    for row in records:
        if not isinstance(row, Mapping) or set(row) != _RECORD_FIELDS:
            raise _fail("predicted_record_invalid")
        lab = row.get("median_lab")
        if (
            row.get("split_role") != "development"
            or row.get("mask_source") != "predicted"
            or not isinstance(lab, list)
            or len(lab) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in lab
            )
        ):
            raise _fail("predicted_record_role_or_measurement_invalid")
        for field in (
            "rgb_sha256", "mask_sha256", "source_render_receipt_sha256",
            "source_validation_receipt_sha256", "source_render_plan_sha256",
            "known_good_profile_sha256",
        ):
            _sha(row.get(field), f"predicted_record_{field}")
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_rows.append(
            (row.get("event_cursor"), row.get("evidence_id"), encoded, _hash_bytes(encoded.encode()))
        )
    if integrity != ("ok",) or rows != expected_rows:
        raise _fail("predicted_database_export_mismatch")

    outputs = receipt.get("outputs")
    provenance = bank.get("provenance")
    if (
        receipt.get("artifact_type") != "inn67_reference_bank_materialization_receipt"
        or receipt.get("status") != "published"
        or receipt.get("wp1_test_accessed") is not False
        or receipt.get("video5_accessed") is not False
        or receipt.get("cpu_only") is not True
        or receipt.get("gpu_or_renderer_launched") is not False
        or receipt.get("image_or_video_bytes_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get("development-reference-bank.json") != bank_sha
        or outputs.get("delta-e-frozen-config.json") != config_sha
        or bank.get("artifact_type") != "inn67_development_reference_bank"
        or bank.get("status") != "accepted_development_only"
        or bank.get("split_role") != "development"
        or bank.get("reference_donor_batch_ids") != list(DONOR_BATCHES)
        or bank.get("predictor") != export.get("predictor")
        or not isinstance(bank.get("records"), list)
        or len(bank["records"]) != 32
        or config.get("artifact_type") != "inn67_delta_e_frozen_development_config"
        or config.get("status") != "frozen_before_test_or_video5_access"
        or config.get("reference_geometry_batch_ids") != list(DONOR_BATCHES)
        or config.get("wp1_test_accessed") is not False
        or config.get("video5_accessed") is not False
        or not isinstance(provenance, Mapping)
        or config.get("provenance") != provenance
        or receipt.get("provenance") != provenance
        or provenance.get("predicted_store_export_sha256") != export_sha
        or provenance.get("accepted_store_tuple") != dict(store)
    ):
        raise _fail("materialization_cross_binding_mismatch")
    donor_records = [
        row for row in records
        if isinstance(row, Mapping) and row.get("batch_id") in DONOR_BATCHES
    ]
    if donor_records != bank["records"] or len(donor_records) != 32:
        raise _fail("donor_record_binding_mismatch")
    if len({row.get("evidence_id") for row in donor_records}) != 32:
        raise _fail("donor_evidence_id_duplicate")
    if len({row.get("event_cursor") for row in donor_records}) != 32:
        raise _fail("donor_event_cursor_duplicate")
    if sorted(row.get("event_cursor") for row in donor_records) != list(range(32)):
        raise _fail("donor_event_order_invalid")
    if any(row.get("split_role") != "development" for row in donor_records):
        raise _fail("donor_split_role_invalid")
    return {
        "status": "blocked_missing_normal_evidence_fields",
        "predicted_store_export_sha256": export_sha,
        "development_reference_bank_sha256": bank_sha,
        "delta_e_frozen_config_sha256": config_sha,
        "materialization_receipt_sha256": receipt_sha,
        "donor_record_count": 32,
        "missing_normal_evidence_fields": list(MISSING_NORMAL_EVIDENCE_FIELDS),
        "claim_boundary": "synthetic reference evidence only; not plant or physical truth",
    }


def materialize_development_reference_store(
    *,
    predicted_store_export_path: str | os.PathLike[str],
    development_reference_bank_path: str | os.PathLike[str],
    delta_e_frozen_config_path: str | os.PathLike[str],
    materialization_receipt_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Fail closed until INN-67 binds every required normal EvidenceStore field."""

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_directory_exists")
    assessment = validate_development_reference_inputs(
        predicted_store_export_path=predicted_store_export_path,
        development_reference_bank_path=development_reference_bank_path,
        delta_e_frozen_config_path=delta_e_frozen_config_path,
        materialization_receipt_path=materialization_receipt_path,
    )
    raise _fail(
        "normal_evidence_store_fields_missing",
        ",".join(assessment["missing_normal_evidence_fields"]),
    )


__all__ = [
    "DevelopmentReferenceError",
    "MISSING_NORMAL_EVIDENCE_FIELDS",
    "materialize_development_reference_store",
    "validate_development_reference_inputs",
]

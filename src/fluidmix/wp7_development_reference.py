"""Truthful WP7 development-reference store intake.

The accepted INN-67 export is not a normal :class:`EvidenceStore` export. This boundary
validates its published cross-bindings and derives only the hash-bound RGB/mask locators
needed for a dedicated minimal donor store. It never creates proxy clips, programmed
recipe metadata, or dispersion measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
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


_CAPABILITY_TOKEN = object()
_STORE_SCHEMA_VERSION = 1
_PROVENANCE_FIELDS = {
    "development_reference_bank_sha256",
    "delta_e_frozen_config_sha256",
    "materialization_receipt_sha256",
    "predicted_store_export_sha256",
    "derived_source_manifest_sha256",
}


@dataclass(frozen=True)
class DevelopmentReferenceMember:
    evidence_id: str
    event_cursor: int
    batch_id: str
    run_id: str
    case_id: str
    recipe_id: str
    stage_id: str
    stage_ordinal: int
    tank_id: str
    split_role: str
    image_id: str
    rgb_path: str
    rgb_sha256: str
    mask_path: str
    mask_sha256: str
    median_lab: tuple[float, float, float]

    def canonical_value(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


class DevelopmentReferenceQuery:
    """Factory-only immutable query with exact evidence/image citations."""

    __slots__ = (
        "query_sha256", "query_kind", "_filters_json", "members",
        "_factory_token", "_sealed",
    )

    def __init__(
        self,
        query_sha256: str,
        query_kind: str,
        filters: Mapping[str, object],
        members: list[DevelopmentReferenceMember],
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CAPABILITY_TOKEN:
            raise TypeError("development-reference queries are factory-only")
        object.__setattr__(self, "query_sha256", query_sha256)
        object.__setattr__(self, "query_kind", query_kind)
        object.__setattr__(
            self,
            "_filters_json",
            json.dumps(dict(filters), sort_keys=True, separators=(",", ":")),
        )
        object.__setattr__(self, "members", tuple(members))
        object.__setattr__(self, "_factory_token", _CAPABILITY_TOKEN)
        object.__setattr__(self, "_sealed", True)

    @property
    def filters(self) -> dict[str, object]:
        return json.loads(self._filters_json)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("development-reference queries are immutable")
        object.__setattr__(self, name, value)


class RuntimeTargetEvidence:
    """Factory-produced target identity available only after runtime measurement."""

    __slots__ = (
        "evidence_id", "event_cursor", "recipe_id", "stage_id", "tank_id",
        "split_role", "_factory_token", "_sealed",
    )

    def __init__(self, value: Mapping[str, object], *, _factory_token: object) -> None:
        if _factory_token is not _CAPABILITY_TOKEN:
            raise TypeError("runtime target evidence is factory-only")
        for field in (
            "evidence_id", "event_cursor", "recipe_id", "stage_id", "tank_id",
            "split_role",
        ):
            object.__setattr__(self, field, value[field])
        object.__setattr__(self, "_factory_token", _CAPABILITY_TOKEN)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("runtime target evidence is immutable")
        object.__setattr__(self, name, value)


def bind_runtime_target_evidence(value: object) -> RuntimeTargetEvidence:
    """Validate the identity fields of a completed predicted runtime measurement."""

    fields = {
        "evidence_id", "event_cursor", "recipe_id", "stage_id", "tank_id",
        "split_role", "mask_source",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DevelopmentReferenceError("runtime_target_fields_invalid")
    for field in ("evidence_id", "recipe_id", "stage_id", "tank_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise DevelopmentReferenceError(f"runtime_target_{field}_invalid")
    if type(value["event_cursor"]) is not int or value["event_cursor"] < 0:
        raise DevelopmentReferenceError("runtime_target_event_cursor_invalid")
    if value["split_role"] not in {"test", "real_check"}:
        raise DevelopmentReferenceError("runtime_target_split_role_invalid")
    if value["mask_source"] != "predicted":
        raise DevelopmentReferenceError("runtime_target_mask_source_invalid")
    return RuntimeTargetEvidence(value, _factory_token=_CAPABILITY_TOKEN)


def assert_development_reference_citation(
    query: object, evidence_id: str, image_id: str
) -> DevelopmentReferenceMember:
    """Return one exact evidence/image citation or reject a forged query/member."""

    if (
        not isinstance(query, DevelopmentReferenceQuery)
        or query._factory_token is not _CAPABILITY_TOKEN
    ):
        raise TypeError("citation requires a trusted development-reference query")
    expected = _canonical_hash(
        {
            "schema_version": "development-reference-query-v1",
            "query_kind": query.query_kind,
            "filters": query.filters,
            "ordered_members": [member.canonical_value() for member in query.members],
        }
    )
    if expected != query.query_sha256:
        raise DevelopmentReferenceError("trusted query binding drifted")
    for member in query.members:
        if member.evidence_id == evidence_id and member.image_id == image_id:
            return member
    raise DevelopmentReferenceError("citation is not a member of the trusted query")


_MINIMAL_SCHEMA = """
CREATE TABLE store_provenance (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    provenance_json TEXT NOT NULL,
    provenance_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE development_reference_records (
    evidence_id TEXT PRIMARY KEY,
    record_sha256 TEXT NOT NULL UNIQUE,
    event_cursor INTEGER NOT NULL UNIQUE CHECK(event_cursor >= 0),
    batch_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    stage_ordinal INTEGER NOT NULL CHECK(stage_ordinal > 0),
    tank_id TEXT NOT NULL,
    split_role TEXT NOT NULL CHECK(split_role = 'development_reference'),
    image_id TEXT NOT NULL UNIQUE,
    rgb_path TEXT NOT NULL UNIQUE,
    rgb_sha256 TEXT NOT NULL,
    mask_path TEXT NOT NULL UNIQUE,
    mask_sha256 TEXT NOT NULL,
    median_lab_json TEXT NOT NULL,
    source_render_receipt_sha256 TEXT NOT NULL,
    source_validation_receipt_sha256 TEXT NOT NULL,
    source_render_plan_sha256 TEXT NOT NULL,
    known_good_profile_sha256 TEXT NOT NULL
) STRICT;
CREATE INDEX development_reference_lookup
ON development_reference_records(recipe_id, stage_id, event_cursor, batch_id);
CREATE TABLE development_reference_configurations (
    configuration_id TEXT PRIMARY KEY,
    recipe_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    stage_ordinal INTEGER NOT NULL,
    tank_id TEXT NOT NULL,
    peer_a_evidence_id TEXT NOT NULL REFERENCES development_reference_records(evidence_id),
    peer_b_evidence_id TEXT NOT NULL REFERENCES development_reference_records(evidence_id),
    threshold_delta_e76 REAL NOT NULL,
    configuration_target_evidence_id TEXT NOT NULL UNIQUE,
    configuration_sha256 TEXT NOT NULL UNIQUE,
    UNIQUE(recipe_id, stage_id)
) STRICT;
CREATE TABLE development_reference_queries (
    query_sha256 TEXT PRIMARY KEY,
    query_kind TEXT NOT NULL,
    filters_json TEXT NOT NULL,
    member_count INTEGER NOT NULL CHECK(member_count > 0)
) STRICT;
CREATE TABLE development_reference_query_members (
    query_sha256 TEXT NOT NULL REFERENCES development_reference_queries(query_sha256),
    member_ordinal INTEGER NOT NULL CHECK(member_ordinal >= 0),
    evidence_id TEXT NOT NULL REFERENCES development_reference_records(evidence_id),
    image_id TEXT NOT NULL,
    member_sha256 TEXT NOT NULL,
    PRIMARY KEY(query_sha256, member_ordinal),
    UNIQUE(query_sha256, evidence_id)
) STRICT;
"""


def _verify_bound_artifact(path_text: str, expected: str, role: str) -> None:
    path = Path(path_text)
    try:
        if not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("invalid artifact")
        observed = _hash_bytes(path.read_bytes())
    except OSError as error:
        raise DevelopmentReferenceError(f"{role} artifact invalid") from error
    if observed != expected:
        raise DevelopmentReferenceError(f"{role} SHA-256 mismatch")


def _create_minimal_store(
    database: Path,
    *,
    records: list[dict[str, object]],
    configurations: list[dict[str, object]],
    provenance: dict[str, object],
) -> None:
    """Create the fresh minimal store after an outer intake validated provenance."""

    if database.exists() or database.is_symlink():
        raise DevelopmentReferenceError("database_exists")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise DevelopmentReferenceError("provenance_fields_invalid")
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_MINIMAL_SCHEMA)
        provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        connection.execute(
            "INSERT INTO store_provenance VALUES (1, ?, ?)",
            (provenance_json, _hash_bytes(provenance_json.encode())),
        )
        for record in records:
            if record.get("split_role") != "development_reference":
                raise DevelopmentReferenceError("record_split_role_invalid")
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            connection.execute(
                "INSERT INTO development_reference_records VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["evidence_id"], _hash_bytes(encoded.encode()),
                    record["event_cursor"], record["batch_id"], record["run_id"],
                    record["case_id"], record["recipe_id"], record["stage_id"],
                    record["stage_ordinal"], record["tank_id"], record["split_role"],
                    record["image_id"], record["rgb_path"], record["rgb_sha256"],
                    record["mask_path"], record["mask_sha256"],
                    json.dumps(record["median_lab"], separators=(",", ":")),
                    record["source_render_receipt_sha256"],
                    record["source_validation_receipt_sha256"],
                    record["source_render_plan_sha256"],
                    record["known_good_profile_sha256"],
                ),
            )
        for config in configurations:
            payload = dict(config)
            config_sha = _canonical_hash(payload)
            configuration_id = "DEV-CONFIG-" + config_sha
            connection.execute(
                "INSERT INTO development_reference_configurations VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    configuration_id, config["recipe_id"], config["stage_id"],
                    config["stage_ordinal"], config["tank_id"],
                    config["peer_evidence_ids"][0], config["peer_evidence_ids"][1],
                    config["threshold_delta_e76"],
                    config["configuration_target_evidence_id"], config_sha,
                ),
            )
        for table in (
            "store_provenance", "development_reference_records",
            "development_reference_configurations", "development_reference_queries",
            "development_reference_query_members",
        ):
            connection.executescript(
                f"""
                CREATE TRIGGER {table}_append_only_update BEFORE UPDATE ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;
                CREATE TRIGGER {table}_append_only_delete BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;
                """
            )
        connection.execute(f"PRAGMA user_version = {_STORE_SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.close()
        database.unlink(missing_ok=True)
        raise
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass


class DevelopmentReferenceStore:
    """Read/query boundary for the dedicated append-only donor store."""

    def __init__(self, database: str | os.PathLike[str]) -> None:
        self.database = Path(database).expanduser().absolute()
        if self.database.is_symlink() or not self.database.is_file():
            raise DevelopmentReferenceError("development_reference_database_invalid")
        with sqlite3.connect(self.database) as connection:
            if connection.execute("PRAGMA user_version").fetchone() != (_STORE_SCHEMA_VERSION,):
                raise DevelopmentReferenceError("development_reference_schema_version_invalid")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise DevelopmentReferenceError("development_reference_integrity_invalid")

    @staticmethod
    def _member(row: tuple[object, ...]) -> DevelopmentReferenceMember:
        return DevelopmentReferenceMember(
            *row[:15],
            tuple(float(value) for value in json.loads(str(row[15]))),
        )

    def _query(
        self,
        config_row: tuple[object, ...],
        *,
        query_kind: str,
        target: RuntimeTargetEvidence | None,
    ) -> DevelopmentReferenceQuery:
        recipe_id, stage_id, stage_ordinal, tank_id, peer_a, peer_b, threshold, config_sha = config_row
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                """
                SELECT evidence_id, event_cursor, batch_id, run_id, case_id,
                       recipe_id, stage_id, stage_ordinal, tank_id, split_role,
                       image_id, rgb_path, rgb_sha256, mask_path, mask_sha256,
                       median_lab_json
                FROM development_reference_records
                WHERE evidence_id IN (?, ?)
                ORDER BY event_cursor, evidence_id
                """,
                (peer_a, peer_b),
            ).fetchall()
        members = [self._member(row) for row in rows]
        if (
            len(members) != 2
            or {member.batch_id for member in members} != set(DONOR_BATCHES)
            or len({member.run_id for member in members}) != 2
            or any(
                member.recipe_id != recipe_id
                or member.stage_id != stage_id
                or member.stage_ordinal != stage_ordinal
                or member.tank_id != tank_id
                or member.split_role != "development_reference"
                for member in members
            )
        ):
            raise DevelopmentReferenceError("configuration_donor_binding_invalid")
        for member in members:
            _verify_bound_artifact(member.rgb_path, member.rgb_sha256, "RGB")
            _verify_bound_artifact(member.mask_path, member.mask_sha256, "mask")
        if target is not None:
            if target._factory_token is not _CAPABILITY_TOKEN:
                raise TypeError("reference_bank requires trusted runtime target evidence")
            if (
                target.recipe_id != recipe_id
                or target.stage_id != stage_id
                or target.tank_id != tank_id
            ):
                raise DevelopmentReferenceError("runtime target selector mismatch")
            if any(member.event_cursor >= target.event_cursor for member in members):
                raise DevelopmentReferenceError("runtime target must be after donor cursors")
        filters: dict[str, object] = {
            "recipe_id": recipe_id,
            "stage_id": stage_id,
            "stage_ordinal": stage_ordinal,
            "tank_id": tank_id,
            "configuration_sha256": config_sha,
            "threshold_delta_e76": threshold,
            "donor_batch_ids": list(DONOR_BATCHES),
        }
        if target is not None:
            filters["target_evidence_id"] = target.evidence_id
            filters["before"] = target.event_cursor
        payload = {
            "schema_version": "development-reference-query-v1",
            "query_kind": query_kind,
            "filters": filters,
            "ordered_members": [member.canonical_value() for member in members],
        }
        query_sha = _canonical_hash(payload)
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                "INSERT OR IGNORE INTO development_reference_queries VALUES (?, ?, ?, ?)",
                (query_sha, query_kind, json.dumps(filters, sort_keys=True, separators=(",", ":")), len(members)),
            )
            for ordinal, member in enumerate(members):
                connection.execute(
                    "INSERT OR IGNORE INTO development_reference_query_members VALUES (?, ?, ?, ?, ?)",
                    (query_sha, ordinal, member.evidence_id, member.image_id, _canonical_hash(member.canonical_value())),
                )
            connection.commit()
        return DevelopmentReferenceQuery(
            query_sha, query_kind, filters, members, _factory_token=_CAPABILITY_TOKEN
        )

    def coverage_queries(self) -> tuple[DevelopmentReferenceQuery, ...]:
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                """
                SELECT recipe_id, stage_id, stage_ordinal, tank_id,
                       peer_a_evidence_id, peer_b_evidence_id,
                       threshold_delta_e76, configuration_sha256
                FROM development_reference_configurations
                ORDER BY recipe_id, stage_ordinal, stage_id
                """
            ).fetchall()
        return tuple(
            self._query(row, query_kind="development_reference_coverage", target=None)
            for row in rows
        )

    def reference_bank(
        self, target: object
    ) -> DevelopmentReferenceQuery:
        if not isinstance(target, RuntimeTargetEvidence):
            raise TypeError("reference_bank requires trusted runtime target evidence")
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                """
                SELECT recipe_id, stage_id, stage_ordinal, tank_id,
                       peer_a_evidence_id, peer_b_evidence_id,
                       threshold_delta_e76, configuration_sha256
                FROM development_reference_configurations
                WHERE recipe_id = ? AND stage_id = ? AND tank_id = ?
                """,
                (target.recipe_id, target.stage_id, target.tank_id),
            ).fetchall()
        if len(rows) != 1:
            raise DevelopmentReferenceError("runtime target configuration unavailable")
        return self._query(
            rows[0], query_kind="development_reference_bank", target=target
        )


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
    """Validate the accepted tuple and prove the minimal donor store is derivable."""

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
    source_bindings = population.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise _fail("derived_source_binding_missing")
    manifest_path = _bound_file(
        source_bindings.get("derived_source_manifest_path"),
        source_bindings.get("derived_source_manifest_sha256"),
        "derived_source_manifest",
    )
    manifest, _, manifest_sha = _load(manifest_path, "derived_source_manifest")
    if (
        manifest.get("artifact_type") != "inn67_receipt_derived_prediction_sources"
        or manifest.get("status") != "derived_from_accepted_receipts"
        or manifest.get("case_count") != 18
        or manifest.get("stage_count") != 48
        or not isinstance(manifest.get("cases"), list)
        or len(manifest["cases"]) != 18
    ):
        raise _fail("derived_source_manifest_contract_invalid")
    source_rows: dict[tuple[str, int], Mapping[str, object]] = {}
    for case in manifest["cases"]:
        if not isinstance(case, Mapping) or not isinstance(case.get("stages"), list):
            raise _fail("derived_source_manifest_case_invalid")
        for stage in case["stages"]:
            if not isinstance(stage, Mapping):
                raise _fail("derived_source_manifest_stage_invalid")
            key = (str(case.get("case_id")), int(stage.get("stage_ordinal", -1)))
            if key in source_rows:
                raise _fail("derived_source_manifest_stage_duplicate")
            source_rows[key] = {
                **stage,
                "render_receipt_sha256": case.get("render_receipt_sha256"),
                "validation_receipt_sha256": case.get("validation_receipt_sha256"),
            }
    if len(source_rows) != 48:
        raise _fail("derived_source_manifest_stage_denominator_invalid")
    minimal_records: list[dict[str, object]] = []
    for row in donor_records:
        evidence_id = str(row["evidence_id"])
        if not evidence_id.startswith("EVIDENCE-"):
            raise _fail("donor_evidence_identity_invalid")
        image_id = evidence_id.removeprefix("EVIDENCE-")
        expected_image = f"INN67-{row['case_id']}-STAGE-{int(row['stage_ordinal']):02d}"
        source = source_rows.get((str(row["case_id"]), int(row["stage_ordinal"])))
        if image_id != expected_image or source is None:
            raise _fail("donor_image_identity_invalid")
        rgb_path = source.get("rgb_path")
        mask_path = database.parent / "predicted-masks" / f"{image_id}.png"
        if (
            not isinstance(rgb_path, str)
            or source.get("rgb_sha256") != row["rgb_sha256"]
            or source.get("render_receipt_sha256")
            != row["source_render_receipt_sha256"]
            or source.get("validation_receipt_sha256")
            != row["source_validation_receipt_sha256"]
        ):
            raise _fail("donor_source_locator_binding_mismatch")
        _verify_bound_artifact(rgb_path, str(row["rgb_sha256"]), "RGB")
        _verify_bound_artifact(str(mask_path), str(row["mask_sha256"]), "mask")
        minimal_records.append(
            {
                **dict(row),
                "split_role": "development_reference",
                "image_id": image_id,
                "rgb_path": rgb_path,
                "mask_path": str(mask_path),
            }
        )
    observations = config.get("recipe_stage_configurations")
    bindings = bank.get("history_bindings")
    if not isinstance(observations, list) or not isinstance(bindings, list):
        raise _fail("configuration_bindings_invalid")
    binding_by_target = {
        str(item.get("target_evidence_id")): item
        for item in bindings if isinstance(item, Mapping)
    }
    configurations = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise _fail("configuration_observation_invalid")
        target_id = str(observation.get("configuration_target_evidence_id"))
        binding = binding_by_target.get(target_id)
        peer_ids = observation.get("peer_evidence_ids")
        if (
            not isinstance(binding, Mapping)
            or not isinstance(peer_ids, list)
            or len(peer_ids) != 2
            or binding.get("earlier_peer_evidence_ids") != peer_ids
            or binding.get("earlier_peer_batch_ids") != list(DONOR_BATCHES)
            or binding.get("recipe_id") != observation.get("recipe_id")
            or binding.get("stage_id") != observation.get("stage_id")
            or binding.get("stage_ordinal") != observation.get("stage_ordinal")
        ):
            raise _fail("configuration_peer_binding_mismatch")
        configurations.append(
            {
                "recipe_id": observation["recipe_id"],
                "stage_id": observation["stage_id"],
                "stage_ordinal": observation["stage_ordinal"],
                "tank_id": observation["tank_id"],
                "peer_evidence_ids": list(peer_ids),
                "threshold_delta_e76": observation["threshold_delta_e76"],
                "configuration_target_evidence_id": target_id,
            }
        )
    if len(configurations) != 16 or len(binding_by_target) != 16:
        raise _fail("configuration_denominator_invalid")
    return {
        "status": "ready_for_minimal_development_reference_store",
        "predicted_store_export_sha256": export_sha,
        "development_reference_bank_sha256": bank_sha,
        "delta_e_frozen_config_sha256": config_sha,
        "materialization_receipt_sha256": receipt_sha,
        "derived_source_manifest_sha256": manifest_sha,
        "donor_record_count": 32,
        "configuration_count": 16,
        "missing_normal_evidence_fields": list(MISSING_NORMAL_EVIDENCE_FIELDS),
        "claim_boundary": "synthetic reference evidence only; not plant or physical truth",
        "_minimal_records": minimal_records,
        "_configurations": configurations,
    }


def materialize_development_reference_store(
    *,
    predicted_store_export_path: str | os.PathLike[str],
    development_reference_bank_path: str | os.PathLike[str],
    delta_e_frozen_config_path: str | os.PathLike[str],
    materialization_receipt_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Publish a fresh minimal 32-donor store without generic record fabrication."""

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_directory_exists")
    try:
        assessment = validate_development_reference_inputs(
            predicted_store_export_path=predicted_store_export_path,
            development_reference_bank_path=development_reference_bank_path,
            delta_e_frozen_config_path=delta_e_frozen_config_path,
            materialization_receipt_path=materialization_receipt_path,
        )
        output.mkdir(parents=True, mode=0o750, exist_ok=False)
        database = output / "development-reference.sqlite3"
        provenance = {
            field: assessment[field]
            for field in _PROVENANCE_FIELDS
        }
        _create_minimal_store(
            database,
            records=assessment["_minimal_records"],
            configurations=assessment["_configurations"],
            provenance=provenance,
        )
        store = DevelopmentReferenceStore(database)
        coverage = store.coverage_queries()
        if len(coverage) != 16 or sum(len(query.members) for query in coverage) != 32:
            raise _fail("coverage_query_denominator_invalid")
        database_sha = _hash_bytes(database.read_bytes())
        receipt = {
            "schema_version": "wp7-development-reference-store-v1",
            "artifact_type": "development_reference_store_provenance",
            "status": "published",
            "store_purpose": "development_reference",
            "store_schema_version": _STORE_SCHEMA_VERSION,
            "store_schema_sha256": _canonical_hash(
                {"schema_version": _STORE_SCHEMA_VERSION, "schema": _MINIMAL_SCHEMA}
            ),
            "store_snapshot_sha256": database_sha,
            "development_reference_bank_sha256": assessment[
                "development_reference_bank_sha256"
            ],
            "delta_e_frozen_config_sha256": assessment[
                "delta_e_frozen_config_sha256"
            ],
            "materialization_receipt_sha256": assessment[
                "materialization_receipt_sha256"
            ],
            "predicted_store_export_sha256": assessment[
                "predicted_store_export_sha256"
            ],
            "derived_source_manifest_sha256": assessment[
                "derived_source_manifest_sha256"
            ],
            "source_classes": ["synthetic_rgb", "predicted_mask"],
            "stage_row_count": 32,
            "configuration_count": 16,
            "donor_batch_ids": list(DONOR_BATCHES),
            "coverage_query_sha256s": [query.query_sha256 for query in coverage],
            "coverage_member_count": 32,
            "authorization": {
                "issue_id": "INN-64",
                "option": 3,
                "approver_name": "Liam",
                "scope": "one_shot_after_accepted_development_reference_bank_and_frozen_config",
            },
            "one_shot_inputs_opened": False,
            "claim_boundary": "synthetic reference evidence only; not plant or physical truth",
        }
        receipt_path = output / "development-reference-store-receipt.json"
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            )
        return {
            "status": "published",
            "output_dir": str(output),
            "database_path": str(database),
            "database_sha256": database_sha,
            "receipt_path": str(receipt_path),
            "receipt_sha256": _hash_bytes(receipt_path.read_bytes()),
            "coverage_queries": coverage,
        }
    except BaseException:
        if output.exists() and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


__all__ = [
    "DevelopmentReferenceMember",
    "DevelopmentReferenceQuery",
    "DevelopmentReferenceStore",
    "DevelopmentReferenceError",
    "MISSING_NORMAL_EVIDENCE_FIELDS",
    "RuntimeTargetEvidence",
    "assert_development_reference_citation",
    "bind_runtime_target_evidence",
    "materialize_development_reference_store",
    "validate_development_reference_inputs",
]

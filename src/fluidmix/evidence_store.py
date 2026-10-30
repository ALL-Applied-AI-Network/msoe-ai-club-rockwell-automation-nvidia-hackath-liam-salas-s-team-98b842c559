"""Append-only SQLite storage and leakage-safe retrieval for Part-B evidence.

The production APIs in this module operate only on camera/program evidence.  The
``sim_metadata`` table is deliberately opaque and has no production read or write API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

from .contracts._common import (
    bounded_control_free_text,
    contained_relative_path,
    relative_path_below_root,
    stable_hash_relative_file,
)
from .contracts.evidence import (
    validate_evidence_sequence_v1,
    verify_evidence_artifacts,
)


SCHEMA_VERSION = 3
CROSS_RUN_REGIME_TOLERANCE_DELTA_E76 = 2.0


_TABLES = (
    "artifacts",
    "runs",
    "vials",
    "fills",
    "clips",
    "stages",
    "images",
    "measurements",
    "decisions",
    "explanations",
    "history_queries",
    "history_members",
    "regime_bindings",
    "regime_references",
    "sim_metadata",
)


_SCHEMA = """
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    UNIQUE(role, path)
) STRICT;
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    split_role TEXT NOT NULL,
    split_version TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE vials (
    vial_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    vial_ordinal INTEGER NOT NULL CHECK (vial_ordinal > 0),
    block_recipe_id TEXT NOT NULL,
    block_position INTEGER NOT NULL CHECK (block_position > 0),
    recipe_id TEXT NOT NULL,
    previous_recipe_id TEXT,
    recipe_json TEXT NOT NULL,
    dispense_order_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    UNIQUE(run_id, vial_ordinal)
) STRICT;
CREATE TABLE fills (
    fill_id TEXT PRIMARY KEY,
    vial_id TEXT NOT NULL UNIQUE REFERENCES vials(vial_id),
    frame_range_json TEXT NOT NULL,
    transition_ranges_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE clips (
    clip_id TEXT PRIMARY KEY,
    fill_id TEXT NOT NULL UNIQUE REFERENCES fills(fill_id),
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE stages (
    evidence_id TEXT PRIMARY KEY,
    record_sha256 TEXT NOT NULL,
    event_cursor INTEGER NOT NULL UNIQUE CHECK (event_cursor >= 0),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    vial_id TEXT NOT NULL REFERENCES vials(vial_id),
    stage_id TEXT NOT NULL,
    stage_ordinal INTEGER NOT NULL CHECK (stage_ordinal > 0),
    tank_id TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    previous_recipe_id TEXT,
    split_role TEXT NOT NULL,
    phase TEXT NOT NULL,
    UNIQUE(vial_id, stage_ordinal)
) STRICT;
CREATE INDEX stages_reference_lookup
    ON stages(recipe_id, stage_id, split_role, event_cursor, evidence_id);
CREATE INDEX stages_vial_lookup
    ON stages(run_id, event_cursor, evidence_id);
CREATE INDEX stages_switch_lookup
    ON stages(previous_recipe_id, recipe_id, split_role, event_cursor, evidence_id);
CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL UNIQUE REFERENCES stages(evidence_id),
    rgb_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    mask_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    mask_source TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE measurements (
    measurement_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL UNIQUE REFERENCES stages(evidence_id),
    median_lab_json TEXT NOT NULL,
    mad_lab_json TEXT NOT NULL,
    iqr_lab_json TEXT NOT NULL,
    pairwise_delta_e76_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE decisions (
    evidence_id TEXT PRIMARY KEY REFERENCES stages(evidence_id),
    decision_id TEXT NOT NULL,
    disposition TEXT NOT NULL,
    limit_codes_json TEXT NOT NULL,
    history_refs_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE explanations (
    evidence_id TEXT PRIMARY KEY REFERENCES stages(evidence_id),
    explanation_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    image_ids_sent_json TEXT NOT NULL,
    guard_result TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE history_queries (
    query_sha256 TEXT PRIMARY KEY,
    query_kind TEXT NOT NULL,
    filters_json TEXT NOT NULL,
    member_count INTEGER NOT NULL CHECK (member_count >= 0)
) STRICT;
CREATE TABLE history_members (
    query_sha256 TEXT NOT NULL REFERENCES history_queries(query_sha256),
    member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
    evidence_id TEXT NOT NULL REFERENCES stages(evidence_id),
    image_id TEXT NOT NULL REFERENCES images(image_id),
    member_sha256 TEXT NOT NULL,
    PRIMARY KEY(query_sha256, member_ordinal),
    UNIQUE(query_sha256, evidence_id)
) STRICT;
CREATE TABLE regime_bindings (
    colour_regime_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL REFERENCES stages(evidence_id),
    measurement_sha256 TEXT NOT NULL,
    binding_sha256 TEXT NOT NULL,
    PRIMARY KEY(colour_regime_id, evidence_id)
) STRICT;
CREATE INDEX regime_lookup ON regime_bindings(colour_regime_id, evidence_id);
CREATE TABLE regime_references (
    colour_regime_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL REFERENCES stages(evidence_id),
    reference_role TEXT NOT NULL,
    split_version TEXT NOT NULL,
    event_cursor INTEGER NOT NULL CHECK (event_cursor >= 0),
    run_id TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    measurement_sha256 TEXT NOT NULL,
    reference_sha256 TEXT NOT NULL,
    PRIMARY KEY(colour_regime_id, evidence_id)
) STRICT;
CREATE TABLE sim_metadata (
    metadata_id TEXT PRIMARY KEY,
    payload BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL
) STRICT;
"""


_ROLES = frozenset(
    {"train", "val", "validation", "evaluation", "test", "real_validation", "calibration"}
)
_CLEAN_REFERENCE_ROLES = frozenset(
    {"train", "val", "validation", "real_validation", "calibration"}
)
_CAPABILITY_TOKEN = object()


def _json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def canonical_source_record_sha256(
    record: Mapping[str, object], normalized_image_locator: Mapping[str, object]
) -> str:
    """Return the store's frozen hash for one record and normalized image locator."""
    required = {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"}
    if set(normalized_image_locator) != required:
        raise ValueError("normalized image locator fields are invalid")
    return _hash(
        {
            "record": record,
            "normalized_image_locator": dict(normalized_image_locator),
        }
    )


@dataclass(frozen=True)
class IngestResult:
    """Counts from one atomic idempotent ingest."""

    inserted: int
    existing: int


@dataclass(frozen=True)
class HistoryMember:
    """One query member with all byte identities required for a citation."""

    evidence_id: str
    event_cursor: int
    run_id: str
    vial_id: str
    recipe_id: str
    previous_recipe_id: str | None
    stage_id: str
    split_role: str
    image_id: str
    rgb_path: str
    rgb_sha256: str
    mask_path: str
    mask_sha256: str
    clip_path: str
    clip_sha256: str
    median_lab: tuple[float, float, float]
    mad_lab: tuple[float, float, float]
    iqr_lab: tuple[float, float, float]
    pairwise_delta_e76: tuple[tuple[str, float], ...]

    def canonical_value(self) -> dict[str, object]:
        value = {field: getattr(self, field) for field in self.__dataclass_fields__}
        value["pairwise_delta_e76"] = [
            {"other_evidence_id": evidence_id, "delta_e76": delta}
            for evidence_id, delta in self.pairwise_delta_e76
        ]
        return value


class MeasuredRegimeBinding:
    """Factory-only proof that store measurements were bound to one regime."""

    __slots__ = (
        "colour_regime_id",
        "binding_sha256",
        "evidence_ids",
        "_reference_provenance_json",
        "_store_token",
        "_factory_token",
        "_sealed",
    )

    def __init__(
        self,
        colour_regime_id: str,
        binding_sha256: str,
        evidence_ids: tuple[str, ...],
        reference_provenance: Sequence[Mapping[str, object]],
        store_token: object,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CAPABILITY_TOKEN:
            raise TypeError("measured-regime capabilities are factory-only")
        object.__setattr__(self, "colour_regime_id", colour_regime_id)
        object.__setattr__(self, "binding_sha256", binding_sha256)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(
            self,
            "_reference_provenance_json",
            _json(list(reference_provenance)),
        )
        object.__setattr__(self, "_store_token", store_token)
        object.__setattr__(self, "_factory_token", _CAPABILITY_TOKEN)
        object.__setattr__(self, "_sealed", True)

    @property
    def reference_provenance(self) -> list[dict[str, object]]:
        return json.loads(self._reference_provenance_json)

    @property
    def reference_roles(self) -> frozenset[str]:
        return frozenset(
            str(row["reference_role"]) for row in self.reference_provenance
        )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("measured-regime capabilities are immutable")
        object.__setattr__(self, name, value)


class HistoryQuery:
    """Factory-only immutable retrieval result used by citation guards."""

    __slots__ = (
        "query_sha256",
        "query_kind",
        "_filters_json",
        "members",
        "_factory_token",
        "_sealed",
    )

    def __init__(
        self,
        query_sha256: str,
        query_kind: str,
        filters: Mapping[str, object],
        members: Sequence[HistoryMember],
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CAPABILITY_TOKEN:
            raise TypeError("history queries are factory-only")
        object.__setattr__(self, "query_sha256", query_sha256)
        object.__setattr__(self, "query_kind", query_kind)
        object.__setattr__(self, "_filters_json", _json(dict(filters)))
        object.__setattr__(self, "members", tuple(members))
        object.__setattr__(self, "_factory_token", _CAPABILITY_TOKEN)
        object.__setattr__(self, "_sealed", True)

    @property
    def filters(self) -> dict[str, object]:
        return json.loads(self._filters_json)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("history queries are immutable")
        object.__setattr__(self, name, value)


def assert_history_citation(
    query: object, evidence_id: str, image_id: str
) -> HistoryMember:
    """Return an exact cited member or fail closed for forged/unreturned IDs."""

    if (
        not isinstance(query, HistoryQuery)
        or query._factory_token is not _CAPABILITY_TOKEN
    ):
        raise TypeError("citation requires a trusted history query")
    expected_sha = _hash(
        {
            "schema_version": "history-query-v1",
            "query_kind": query.query_kind,
            "filters": query.filters,
            "ordered_members": [member.canonical_value() for member in query.members],
        }
    )
    if expected_sha != query.query_sha256:
        raise ValueError("trusted history query binding drifted")
    for member in query.members:
        if member.evidence_id == evidence_id and member.image_id == image_id:
            return member
    raise ValueError("citation is not a member of the trusted history query")


def adapt_fill_clips_v1(
    payload: object,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    """Normalize the small WP4 fill-clips JSON handoff without trusting extras.

    This adapter only reshapes the fixture/runtime envelope.  ``EvidenceStore.ingest``
    remains responsible for record validation and exact artifact verification.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("fill-clips payload must be a mapping")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "part_b_fill_clip_measurements_v1"
    ):
        raise ValueError("fill-clips payload version/type is unsupported")
    records = payload.get("evidence_records")
    raw_locators = payload.get("image_locators")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("fill-clips evidence_records must be a sequence")
    if isinstance(raw_locators, (str, bytes)) or not isinstance(raw_locators, Sequence):
        raise TypeError("fill-clips image_locators must be a sequence")
    copied_records = deepcopy(list(records))
    if any(not isinstance(record, dict) for record in copied_records):
        raise TypeError("fill-clips evidence records must be mappings")
    locators: dict[str, dict[str, str]] = {}
    required = {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"}
    for raw in raw_locators:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("image_id"), str):
            raise TypeError("fill-clips locator rows are invalid")
        image_id = raw["image_id"]
        if image_id in locators or not required <= set(raw):
            raise ValueError("fill-clips locators are duplicate or incomplete")
        locator = {field: raw[field] for field in required}
        if any(not isinstance(value, str) for value in locator.values()):
            raise TypeError("fill-clips locator values must be strings")
        locators[image_id] = locator
    return copied_records, locators


class EvidenceStore:
    """A small append-only evidence database rooted at an artifact directory."""

    def __init__(self, database: str | os.PathLike[str], *, artifact_root: str | os.PathLike[str]) -> None:
        self.database = Path(database).expanduser().absolute()
        self.artifact_root = Path(artifact_root).expanduser().absolute()
        if self.database.is_symlink():
            raise ValueError("evidence database cannot be a symlink")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._capability_token = object()
        self._trace_for_test = None
        self._initialize()

    def _connect(self, *, protect_sim_metadata: bool = True) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        if protect_sim_metadata:
            connection.set_authorizer(self._production_authorizer)
        if self._trace_for_test is not None:
            connection.set_trace_callback(self._trace_for_test)
        return connection

    @staticmethod
    def _production_authorizer(
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        del argument_two, database_name, trigger_name
        if action in {
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
        } and argument_one == "sim_metadata":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _initialize(self) -> None:
        existed = self.database.exists()
        with self._connect(protect_sim_metadata=False) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if existed and version not in {0, SCHEMA_VERSION}:
                raise ValueError(
                    f"unsupported evidence schema version {version}; expected {SCHEMA_VERSION}"
                )
            if version == 0:
                current = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if current:
                    raise ValueError("unversioned evidence database is not empty")
                connection.executescript(_SCHEMA)
                for table in _TABLES:
                    connection.executescript(
                        f"""
                        CREATE TRIGGER {table}_append_only_update
                        BEFORE UPDATE ON {table}
                        BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;
                        CREATE TRIGGER {table}_append_only_delete
                        BEFORE DELETE ON {table}
                        BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;
                        """
                    )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            current = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if current != set(_TABLES):
                raise ValueError("evidence database schema tables do not match its version")

    @staticmethod
    def _append_exact(
        connection: sqlite3.Connection,
        table: str,
        key_column: str,
        key_value: object,
        columns: tuple[str, ...],
        values: tuple[object, ...],
    ) -> bool:
        selected = connection.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE {key_column} = ?",
            (key_value,),
        ).fetchone()
        if selected is not None:
            if tuple(selected) != values:
                raise ValueError(f"logical collision in {table} for {key_value!r}")
            return False
        connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            values,
        )
        return True

    @staticmethod
    def _validate_ingest_inputs(
        records: object,
        image_locators: object,
        event_cursors: object,
    ) -> tuple[list[dict[str, object]], dict[str, Mapping[str, object]], dict[str, int]]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
            raise TypeError("records must be a non-empty sequence")
        if not isinstance(image_locators, Mapping):
            raise TypeError("image_locators must be a normalized mapping")
        if not isinstance(event_cursors, Mapping):
            raise TypeError("event_cursors must be a mapping")
        copied = deepcopy(list(records))
        if any(not isinstance(record, dict) for record in copied):
            raise TypeError("records must contain mappings")
        evidence_ids = [record.get("evidence_id") for record in copied]
        if any(not isinstance(value, str) for value in evidence_ids):
            raise TypeError("every record requires an evidence_id")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("ingest evidence_id values must be unique")
        if set(event_cursors) != set(evidence_ids):
            raise ValueError("event_cursors must exactly cover the ingest evidence IDs")
        cursors: dict[str, int] = {}
        for evidence_id in evidence_ids:
            cursor = event_cursors[evidence_id]
            if type(cursor) is not int or cursor < 0:
                raise ValueError("event cursor must be a non-negative integer")
            cursors[evidence_id] = cursor
        if len(cursors) != len(set(cursors.values())):
            raise ValueError("event cursors must be unique")
        locators: dict[str, Mapping[str, object]] = {}
        for image_id, locator in image_locators.items():
            if not isinstance(image_id, str) or not isinstance(locator, Mapping):
                raise TypeError("image locator registry entries are invalid")
            locators[image_id] = locator
        if set(locators) != {record["image_id"] for record in copied}:
            raise ValueError("image_locators must exactly cover the ingest image IDs")
        return copied, locators, cursors

    def ingest(
        self,
        records: object,
        image_locators: object,
        *,
        phase: str,
        event_cursors: object,
    ) -> IngestResult:
        """Atomically append verified normalized records, or verify an identical retry."""

        copied, locators, cursors = self._validate_ingest_inputs(
            records, image_locators, event_cursors
        )
        grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
        for record in copied:
            run_id = record.get("run_id")
            vial_ordinal = record.get("vial_ordinal")
            if not isinstance(run_id, str) or type(vial_ordinal) is not int:
                raise ValueError("record run/vial identity is invalid")
            grouped.setdefault((run_id, vial_ordinal), []).append(record)
        checked: list[dict[str, object]] = []
        for rows in grouped.values():
            ordered = sorted(rows, key=lambda row: int(row.get("stage_ordinal", -1)))
            sequence = validate_evidence_sequence_v1(
                ordered,
                phase=phase,
                mode="production",
            )
            sequence_cursors = [cursors[str(row["evidence_id"])] for row in sequence]
            if sequence_cursors != sorted(sequence_cursors):
                raise ValueError("event cursors must follow stage order")
            checked.extend(sequence)
        checked.sort(key=lambda row: cursors[str(row["evidence_id"])])
        by_run: dict[str, list[dict[str, object]]] = {}
        for record in checked:
            by_run.setdefault(str(record["run_id"]), []).append(record)
        for run_rows in by_run.values():
            programmed = sorted(
                run_rows,
                key=lambda row: (int(row["vial_ordinal"]), int(row["stage_ordinal"])),
            )
            programmed_cursors = [cursors[str(row["evidence_id"])] for row in programmed]
            if programmed_cursors != sorted(programmed_cursors):
                raise ValueError("event cursors must follow programmed run order")
        for record in checked:
            verify_evidence_artifacts(
                record,
                self.artifact_root,
                predicted_mask_registry=locators,
            )

        inserted = 0
        existing = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_incremental_order(connection, checked, cursors)
            for record in checked:
                was_inserted = self._append_record(
                    connection,
                    record,
                    locator=locators[str(record["image_id"])],
                    phase=phase,
                    event_cursor=cursors[str(record["evidence_id"])],
                )
                inserted += int(was_inserted)
                existing += int(not was_inserted)
        return IngestResult(inserted=inserted, existing=existing)

    @staticmethod
    def _validate_incremental_order(
        connection: sqlite3.Connection,
        records: Sequence[Mapping[str, object]],
        cursors: Mapping[str, int],
    ) -> None:
        by_run: dict[str, list[Mapping[str, object]]] = {}
        for record in records:
            by_run.setdefault(str(record["run_id"]), []).append(record)
        for run_id, run_records in by_run.items():
            stored = connection.execute(
                """
                SELECT s.evidence_id, v.vial_ordinal, s.stage_ordinal, s.event_cursor
                FROM stages AS s
                JOIN vials AS v ON v.vial_id = s.vial_id
                WHERE s.run_id = ?
                """,
                (run_id,),
            ).fetchall()
            if not stored:
                continue
            stored_ids = {str(row[0]) for row in stored}
            new_records = [
                record
                for record in run_records
                if str(record["evidence_id"]) not in stored_ids
            ]
            if not new_records:
                continue
            last_stored_order = max((int(row[1]), int(row[2])) for row in stored)
            last_stored_cursor = max(int(row[3]) for row in stored)
            first_new_order = min(
                (int(record["vial_ordinal"]), int(record["stage_ordinal"]))
                for record in new_records
            )
            first_new_cursor = min(
                cursors[str(record["evidence_id"])] for record in new_records
            )
            if (
                first_new_order <= last_stored_order
                or first_new_cursor <= last_stored_cursor
            ):
                raise ValueError(
                    "incremental ingest must append after stored same-run order"
                )

    def _artifact(
        self,
        connection: sqlite3.Connection,
        *,
        role: str,
        relative_path: str,
        sha256: str,
    ) -> str:
        observed = stable_hash_relative_file(
            self.artifact_root, relative_path, role
        )
        if observed["sha256"] != sha256:
            raise ValueError(f"{role} SHA-256 mismatch")
        artifact_id = "ART-" + _hash(
            {"role": role, "path": relative_path, "sha256": sha256}
        )
        by_path = connection.execute(
            "SELECT artifact_id, sha256, byte_size FROM artifacts WHERE role = ? AND path = ?",
            (role, relative_path),
        ).fetchone()
        expected_by_path = (artifact_id, sha256, int(observed["bytes"]))
        if by_path is not None and by_path != expected_by_path:
            raise ValueError(f"logical collision in artifacts for {(role, relative_path)!r}")
        columns = ("artifact_id", "role", "path", "sha256", "byte_size")
        values = (artifact_id, role, relative_path, sha256, int(observed["bytes"]))
        self._append_exact(
            connection, "artifacts", "artifact_id", artifact_id, columns, values
        )
        return artifact_id

    def _append_record(
        self,
        connection: sqlite3.Connection,
        record: dict[str, object],
        *,
        locator: Mapping[str, object],
        phase: str,
        event_cursor: int,
    ) -> bool:
        evidence_id = str(record["evidence_id"])
        rgb_path = relative_path_below_root(
            self.artifact_root, locator["rgb_path"], "predicted RGB path"
        ).as_posix()
        mask_path = relative_path_below_root(
            self.artifact_root, locator["mask_path"], "predicted mask path"
        ).as_posix()
        record_sha = canonical_source_record_sha256(
            record,
            {
                "rgb_path": rgb_path,
                "rgb_sha256": locator["rgb_sha256"],
                "mask_path": mask_path,
                "mask_sha256": locator["mask_sha256"],
            },
        )
        existing = connection.execute(
            "SELECT record_sha256, event_cursor, phase FROM stages WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        if existing is not None:
            if existing != (record_sha, event_cursor, phase):
                raise ValueError(f"logical collision in stages for {evidence_id!r}")
            return False
        cursor_owner = connection.execute(
            "SELECT evidence_id FROM stages WHERE event_cursor = ?", (event_cursor,)
        ).fetchone()
        if cursor_owner is not None:
            raise ValueError(
                f"logical collision in stages for event cursor {event_cursor!r}"
            )

        run_id = str(record["run_id"])
        run_columns = ("run_id", "split_role", "split_version", "content_sha256")
        run_payload = {
            "run_id": run_id,
            "split_role": record["split_role"],
            "split_version": record["split_version"],
        }
        run_values = (
            run_id,
            str(record["split_role"]),
            str(record["split_version"]),
            _hash(run_payload),
        )
        self._append_exact(connection, "runs", "run_id", run_id, run_columns, run_values)

        vial_id = f"{run_id}:{record['vial_ordinal']}"
        vial_payload = {
            key: record[key]
            for key in (
                "run_id", "case_id", "vial_ordinal", "block_recipe_id",
                "block_position", "recipe_id", "previous_recipe_id", "recipe_ml",
                "dispense_order",
            )
        }
        vial_columns = (
            "vial_id", "run_id", "case_id", "vial_ordinal", "block_recipe_id",
            "block_position", "recipe_id", "previous_recipe_id", "recipe_json",
            "dispense_order_json", "content_sha256",
        )
        vial_values = (
            vial_id, run_id, str(record["case_id"]), int(record["vial_ordinal"]),
            str(record["block_recipe_id"]), int(record["block_position"]),
            str(record["recipe_id"]), record["previous_recipe_id"],
            _json(record["recipe_ml"]), _json(record["dispense_order"]),
            _hash(vial_payload),
        )
        self._append_exact(connection, "vials", "vial_id", vial_id, vial_columns, vial_values)

        fill_id = f"FILL-{vial_id}"
        fill_payload = {
            "fill_frame_range": record["fill_frame_range"],
            "transition_frame_ranges": record["transition_frame_ranges"],
        }
        fill_columns = (
            "fill_id", "vial_id", "frame_range_json", "transition_ranges_json",
            "content_sha256",
        )
        fill_values = (
            fill_id, vial_id, _json(record["fill_frame_range"]),
            _json(record["transition_frame_ranges"]), _hash(fill_payload),
        )
        self._append_exact(connection, "fills", "fill_id", fill_id, fill_columns, fill_values)

        clip_path = contained_relative_path(record["fill_clip_path"], "fill_clip_path").as_posix()
        clip_artifact = self._artifact(
            connection,
            role="fill_clip",
            relative_path=clip_path,
            sha256=str(record["clip_sha256"]),
        )
        clip_id = f"CLIP-{vial_id}"
        clip_columns = ("clip_id", "fill_id", "artifact_id", "content_sha256")
        clip_values = (
            clip_id, fill_id, clip_artifact,
            _hash({"fill_id": fill_id, "artifact_id": clip_artifact}),
        )
        self._append_exact(connection, "clips", "clip_id", clip_id, clip_columns, clip_values)

        stage_columns = (
            "evidence_id", "record_sha256", "event_cursor", "run_id", "vial_id",
            "stage_id", "stage_ordinal", "tank_id", "recipe_id", "previous_recipe_id",
            "split_role", "phase",
        )
        stage_values = (
            evidence_id, record_sha, event_cursor, run_id, vial_id,
            str(record["stage_id"]), int(record["stage_ordinal"]), str(record["tank_id"]),
            str(record["recipe_id"]), record["previous_recipe_id"],
            str(record["split_role"]), phase,
        )
        self._append_exact(connection, "stages", "evidence_id", evidence_id, stage_columns, stage_values)

        rgb_artifact = self._artifact(
            connection, role="rgb", relative_path=rgb_path, sha256=str(record["rgb_sha256"])
        )
        mask_artifact = self._artifact(
            connection, role="mask", relative_path=mask_path, sha256=str(record["mask_sha256"])
        )
        image_id = str(record["image_id"])
        image_columns = (
            "image_id", "evidence_id", "rgb_artifact_id", "mask_artifact_id",
            "mask_source", "content_sha256",
        )
        image_payload = {
            "image_id": image_id, "evidence_id": evidence_id,
            "rgb_artifact_id": rgb_artifact, "mask_artifact_id": mask_artifact,
            "mask_source": record["mask_source"],
        }
        image_values = (
            image_id, evidence_id, rgb_artifact, mask_artifact,
            str(record["mask_source"]), _hash(image_payload),
        )
        self._append_exact(connection, "images", "image_id", image_id, image_columns, image_values)

        measurement_id = str(record["measurement_id"])
        measurement_payload = {
            key: record[key]
            for key in ("measurement_id", "evidence_id", "median_lab", "mad_lab", "iqr_lab", "pairwise_delta_e76")
        }
        measurement_columns = (
            "measurement_id", "evidence_id", "median_lab_json", "mad_lab_json",
            "iqr_lab_json", "pairwise_delta_e76_json", "content_sha256",
        )
        measurement_values = (
            measurement_id, evidence_id, _json(record["median_lab"]),
            _json(record["mad_lab"]), _json(record["iqr_lab"]),
            _json(record["pairwise_delta_e76"]), _hash(measurement_payload),
        )
        self._append_exact(
            connection, "measurements", "measurement_id", measurement_id,
            measurement_columns, measurement_values,
        )

        if record["decision_id"] is not None:
            for history_ref in record["history_refs"]:
                cited = connection.execute(
                    """
                    SELECT 1 FROM history_members
                    WHERE query_sha256 = ? AND evidence_id = ?
                    """,
                    (history_ref["query_sha256"], history_ref["evidence_id"]),
                ).fetchone()
                if cited is None:
                    raise ValueError("history_refs contains a non-member citation")
            decision_payload = {
                key: record[key]
                for key in ("decision_id", "evidence_id", "disposition", "limit_codes", "history_refs")
            }
            columns = (
                "evidence_id", "decision_id", "disposition", "limit_codes_json",
                "history_refs_json", "content_sha256",
            )
            values = (
                evidence_id, str(record["decision_id"]), str(record["disposition"]),
                _json(record["limit_codes"]), _json(record["history_refs"]),
                _hash(decision_payload),
            )
            self._append_exact(connection, "decisions", "evidence_id", evidence_id, columns, values)
        if record["explanation_id"] is not None:
            explanation_payload = {
                key: record[key]
                for key in ("explanation_id", "evidence_id", "model_revision", "prompt_sha256", "image_ids_sent", "guard_result")
            }
            columns = (
                "evidence_id", "explanation_id", "model_revision", "prompt_sha256",
                "image_ids_sent_json", "guard_result", "content_sha256",
            )
            values = (
                evidence_id, str(record["explanation_id"]), str(record["model_revision"]),
                str(record["prompt_sha256"]), _json(record["image_ids_sent"]),
                str(record["guard_result"]), _hash(explanation_payload),
            )
            self._append_exact(connection, "explanations", "evidence_id", evidence_id, columns, values)
        return True

    def bind_measured_regime(
        self,
        target_evidence_id: str,
        *unexpected_membership: object,
        cross_run_clean_reference_evidence_ids: object | None = None,
        allowed_clean_reference_roles: object | None = None,
        allowed_clean_reference_split_versions: object | None = None,
    ) -> MeasuredRegimeBinding:
        """Derive a measured regime from the target run or agreed clean references.

        The default is all measured evidence in the target's production run.  Cross-run
        grouping is available only when one same-recipe/stage clean reference per run
        agrees under the frozen CIE76 tolerance.
        """

        if unexpected_membership:
            raise TypeError(
                "measured regime binding accepts one target evidence ID, not arbitrary membership"
            )
        bounded_control_free_text(target_evidence_id, "target evidence ID")
        with self._connect() as connection:
            target = connection.execute(
                """
                SELECT run_id, event_cursor, recipe_id, stage_id, tank_id
                FROM stages WHERE evidence_id = ?
                """,
                (target_evidence_id,),
            ).fetchone()
        if target is None:
            raise ValueError("target evidence is not stored measured evidence")
        target_run_id = str(target[0])
        target_cursor = int(target[1])
        target_selector = (str(target[2]), str(target[3]), str(target[4]))
        reference_provenance: list[dict[str, object]] = []

        if cross_run_clean_reference_evidence_ids is None:
            if (
                allowed_clean_reference_roles is not None
                or allowed_clean_reference_split_versions is not None
            ):
                raise ValueError(
                    "same-run regimes do not accept cross-run reference policy"
                )
            run_ids = (target_run_id,)
            basis: dict[str, object] = {
                "mode": "same_production_run",
                "run_ids": list(run_ids),
            }
            colour_regime_id = "measured-run-" + _hash(basis)
            member_cursor_limit: int | None = None
        else:
            allowed_reference_roles = self._explicit_text_set(
                allowed_clean_reference_roles,
                "allowed clean-reference roles",
            )
            allowed_split_versions = self._explicit_text_set(
                allowed_clean_reference_split_versions,
                "allowed clean-reference split provenance",
            )
            clean_ids = self._validate_clean_reference_ids(
                cross_run_clean_reference_evidence_ids
            )
            with self._connect() as connection:
                placeholders = ", ".join("?" for _ in clean_ids)
                rows = connection.execute(
                    f"""
                    SELECT s.evidence_id, s.run_id, s.recipe_id, s.stage_id, s.tank_id,
                           m.median_lab_json, m.content_sha256, s.split_role,
                           r.split_version, s.event_cursor
                    FROM stages AS s
                    JOIN measurements AS m ON m.evidence_id = s.evidence_id
                    JOIN runs AS r ON r.run_id = s.run_id
                    WHERE s.evidence_id IN ({placeholders})
                    ORDER BY s.evidence_id
                    """,
                    tuple(clean_ids),
                ).fetchall()
            if len(rows) != len(clean_ids):
                raise ValueError("clean-reference evidence is not stored measured evidence")
            clean_runs = [str(row[1]) for row in rows]
            if len(clean_runs) != len(set(clean_runs)) or target_run_id not in clean_runs:
                raise ValueError(
                    "cross-run clean references require one row per run including the target run"
                )
            comparable = {(row[2], row[3], row[4]) for row in rows}
            if len(comparable) != 1 or next(iter(comparable)) != target_selector:
                raise ValueError(
                    "cross-run clean references must share the target recipe and stage"
                )
            if any(int(row[9]) >= target_cursor for row in rows):
                raise ValueError(
                    "cross-run clean references must be strictly earlier than the target"
                )
            if not allowed_reference_roles <= _CLEAN_REFERENCE_ROLES:
                raise ValueError(
                    "allowed clean-reference roles include a disallowed record role"
                )
            if any(str(row[7]) not in allowed_reference_roles for row in rows):
                raise ValueError(
                    "cross-run clean-reference role is not explicitly allowed"
                )
            if any(str(row[7]) not in _CLEAN_REFERENCE_ROLES for row in rows):
                raise ValueError("cross-run reference uses a disallowed clean-reference role")
            if any(str(row[8]) not in allowed_split_versions for row in rows):
                raise ValueError(
                    "cross-run clean-reference split provenance is not explicitly allowed"
                )
            labs = [tuple(float(value) for value in json.loads(str(row[5]))) for row in rows]
            if any(
                math.dist(left, right) > CROSS_RUN_REGIME_TOLERANCE_DELTA_E76
                for index, left in enumerate(labs)
                for right in labs[index + 1 :]
            ):
                raise ValueError(
                    "cross-run clean references exceed the frozen measured-colour tolerance"
                )
            run_ids = tuple(sorted(clean_runs))
            for row in rows:
                reference = {
                    "evidence_id": str(row[0]),
                    "run_id": str(row[1]),
                    "recipe_id": str(row[2]),
                    "stage_id": str(row[3]),
                    "tank_id": str(row[4]),
                    "measurement_sha256": str(row[6]),
                    "reference_role": str(row[7]),
                    "split_version": str(row[8]),
                    "event_cursor": int(row[9]),
                }
                reference["reference_sha256"] = _hash(reference)
                reference_provenance.append(reference)
            basis = {
                "mode": "cross_run_clean_reference_agreement",
                "tolerance_delta_e76": CROSS_RUN_REGIME_TOLERANCE_DELTA_E76,
                "clean_references": reference_provenance,
            }
            colour_regime_id = "measured-cross-" + _hash(basis)
            member_cursor_limit = target_cursor

        with self._connect() as connection:
            placeholders = ", ".join("?" for _ in run_ids)
            cursor_clause = (
                "" if member_cursor_limit is None else " AND s.event_cursor <= ?"
            )
            parameters: tuple[object, ...] = tuple(run_ids)
            if member_cursor_limit is not None:
                parameters = (*parameters, member_cursor_limit)
            member_rows = connection.execute(
                f"""
                SELECT s.evidence_id, m.content_sha256
                FROM stages AS s
                JOIN measurements AS m ON m.evidence_id = s.evidence_id
                WHERE s.run_id IN ({placeholders}){cursor_clause}
                ORDER BY s.evidence_id
                """,
                parameters,
            ).fetchall()
        ids = [str(row[0]) for row in member_rows]
        if target_evidence_id not in ids:
            raise ValueError("target evidence is unavailable from the measured regime")
        members: list[dict[str, str]] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for evidence_id, raw_measurement_sha in member_rows:
                evidence_id = str(evidence_id)
                measurement_sha = str(raw_measurement_sha)
                binding_payload = {
                    "colour_regime_id": colour_regime_id,
                    "evidence_id": evidence_id,
                    "measurement_sha256": measurement_sha,
                }
                binding_sha = _hash(binding_payload)
                values = (colour_regime_id, evidence_id, measurement_sha, binding_sha)
                existing = connection.execute(
                    """
                    SELECT colour_regime_id, evidence_id, measurement_sha256, binding_sha256
                    FROM regime_bindings
                    WHERE colour_regime_id = ? AND evidence_id = ?
                    """,
                    (colour_regime_id, evidence_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO regime_bindings VALUES (?, ?, ?, ?)", values
                    )
                elif existing != values:
                    raise ValueError("logical collision in regime_bindings")
                members.append({"evidence_id": evidence_id, "binding_sha256": binding_sha})
            for reference in reference_provenance:
                values = (
                    colour_regime_id,
                    reference["evidence_id"],
                    reference["reference_role"],
                    reference["split_version"],
                    reference["event_cursor"],
                    reference["run_id"],
                    reference["recipe_id"],
                    reference["stage_id"],
                    reference["measurement_sha256"],
                    reference["reference_sha256"],
                )
                existing = connection.execute(
                    """
                    SELECT colour_regime_id, evidence_id, reference_role, split_version,
                           event_cursor, run_id, recipe_id, stage_id,
                           measurement_sha256, reference_sha256
                    FROM regime_references
                    WHERE colour_regime_id = ? AND evidence_id = ?
                    """,
                    (colour_regime_id, reference["evidence_id"]),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO regime_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        values,
                    )
                elif existing != values:
                    raise ValueError("logical collision in regime_references")
        capability_sha = _hash(
            {
                "colour_regime_id": colour_regime_id,
                "members": members,
                "reference_provenance": reference_provenance,
            }
        )
        return MeasuredRegimeBinding(
            colour_regime_id,
            capability_sha,
            tuple(sorted(ids)),
            reference_provenance,
            self._capability_token,
            _factory_token=_CAPABILITY_TOKEN,
        )

    @staticmethod
    def _explicit_text_set(value: object, role: str) -> frozenset[str]:
        if isinstance(value, (str, bytes)) or not isinstance(
            value, (set, frozenset, list, tuple)
        ):
            raise ValueError(f"{role} must be explicitly provided")
        values = tuple(value)
        if not values or any(not isinstance(item, str) for item in values):
            raise ValueError(f"{role} must contain nonempty text values")
        if len(values) != len(set(values)):
            raise ValueError(f"{role} must not contain duplicates")
        for item in values:
            bounded_control_free_text(item, role)
        return frozenset(values)

    @staticmethod
    def _validate_clean_reference_ids(value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("cross-run clean-reference IDs must be a sequence")
        ids = tuple(value)
        if len(ids) < 2 or any(not isinstance(item, str) for item in ids):
            raise ValueError("cross-run clean-reference IDs require at least two text IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("cross-run clean-reference IDs must be unique")
        for evidence_id in ids:
            bounded_control_free_text(evidence_id, "clean-reference evidence ID")
        return ids

    def _validate_query_controls(
        self,
        *, before: object, allowed_roles: object, target_evidence_id: object
    ) -> tuple[int, tuple[str, ...], str, dict[str, object]]:
        if type(before) is not int or before < 0:
            raise ValueError("before must be a non-negative integer event cursor")
        if isinstance(allowed_roles, (str, bytes)) or not isinstance(
            allowed_roles, (set, frozenset, list, tuple)
        ):
            raise TypeError("allowed_roles must be an explicit collection")
        roles = tuple(sorted(allowed_roles))
        if not roles or any(type(role) is not str for role in roles) or not set(roles) <= _ROLES:
            raise ValueError("allowed_roles must be a non-empty subset of record roles")
        if len(roles) != len(set(roles)):
            raise ValueError("allowed_roles must not contain duplicates")
        bounded_control_free_text(target_evidence_id, "target_evidence_id")
        with self._connect() as connection:
            target = connection.execute(
                """
                SELECT event_cursor, run_id, recipe_id, previous_recipe_id, stage_id
                FROM stages WHERE evidence_id = ?
                """,
                (target_evidence_id,),
            ).fetchone()
        if target is None:
            raise ValueError("target_evidence_id is not stored evidence")
        if int(target[0]) != before:
            raise ValueError("before must equal the target evidence event cursor")
        context = {
            "run_id": target[1],
            "recipe_id": target[2],
            "previous_recipe_id": target[3],
            "stage_id": target[4],
        }
        return before, roles, target_evidence_id, context

    def _member_query(
        self,
        *,
        query_kind: str,
        filters: dict[str, object],
        where_sql: str,
        parameters: tuple[object, ...],
    ) -> HistoryQuery:
        sql = f"""
            SELECT
                s.evidence_id, s.event_cursor, s.run_id, s.vial_id, s.recipe_id,
                s.previous_recipe_id, s.stage_id, s.split_role, i.image_id,
                rgb.path, rgb.sha256, mask.path, mask.sha256, clip_art.path,
                clip_art.sha256
                , m.median_lab_json, m.mad_lab_json, m.iqr_lab_json,
                m.pairwise_delta_e76_json
            FROM stages AS s
            JOIN images AS i ON i.evidence_id = s.evidence_id
            JOIN measurements AS m ON m.evidence_id = s.evidence_id
            JOIN artifacts AS rgb ON rgb.artifact_id = i.rgb_artifact_id
            JOIN artifacts AS mask ON mask.artifact_id = i.mask_artifact_id
            JOIN fills AS f ON f.vial_id = s.vial_id
            JOIN clips AS c ON c.fill_id = f.fill_id
            JOIN artifacts AS clip_art ON clip_art.artifact_id = c.artifact_id
            {where_sql}
            ORDER BY s.event_cursor ASC, s.evidence_id ASC, i.image_id ASC
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        unfiltered_members = tuple(self._history_member(row) for row in rows)
        returned_evidence_ids = {
            member.evidence_id for member in unfiltered_members
        }
        members = tuple(
            replace(
                member,
                pairwise_delta_e76=tuple(
                    pair
                    for pair in member.pairwise_delta_e76
                    if pair[0] in returned_evidence_ids
                ),
            )
            for member in unfiltered_members
        )
        for member in members:
            for path, expected, role in (
                (member.rgb_path, member.rgb_sha256, "retrieved RGB image"),
                (member.mask_path, member.mask_sha256, "retrieved mask image"),
                (member.clip_path, member.clip_sha256, "retrieved fill clip"),
            ):
                observed = stable_hash_relative_file(self.artifact_root, path, role)
                if observed["sha256"] != expected:
                    raise ValueError(f"{role} SHA-256 mismatch")
        query_payload = {
            "schema_version": "history-query-v1",
            "query_kind": query_kind,
            "filters": filters,
            "ordered_members": [member.canonical_value() for member in members],
        }
        query_sha = _hash(query_payload)
        filters_json = _json(filters)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            columns = ("query_sha256", "query_kind", "filters_json", "member_count")
            values = (query_sha, query_kind, filters_json, len(members))
            self._append_exact(
                connection, "history_queries", "query_sha256", query_sha,
                columns, values,
            )
            for ordinal, member in enumerate(members):
                member_sha = _hash(member.canonical_value())
                existing = connection.execute(
                    """
                    SELECT query_sha256, member_ordinal, evidence_id, image_id, member_sha256
                    FROM history_members
                    WHERE query_sha256 = ? AND member_ordinal = ?
                    """,
                    (query_sha, ordinal),
                ).fetchone()
                expected = (query_sha, ordinal, member.evidence_id, member.image_id, member_sha)
                if existing is None:
                    connection.execute(
                        "INSERT INTO history_members VALUES (?, ?, ?, ?, ?)", expected
                    )
                elif existing != expected:
                    raise ValueError("logical collision in history_members")
        return HistoryQuery(
            query_sha,
            query_kind,
            filters,
            members,
            _factory_token=_CAPABILITY_TOKEN,
        )

    @staticmethod
    def _history_member(row: Sequence[object]) -> HistoryMember:
        median = json.loads(str(row[15]))
        mad = json.loads(str(row[16]))
        iqr = json.loads(str(row[17]))
        pairs = json.loads(str(row[18]))
        return HistoryMember(
            *row[:15],
            tuple(float(value) for value in median),
            tuple(float(value) for value in mad),
            tuple(float(value) for value in iqr),
            tuple(
                (str(pair["other_evidence_id"]), float(pair["delta_e76"]))
                for pair in pairs
            ),
        )

    def reference_bank(
        self,
        recipe_id: str,
        stage_id: str,
        measured_regime: object,
        *,
        before: int,
        allowed_roles: object,
        target_evidence_id: str,
    ) -> HistoryQuery:
        """Retrieve same-recipe/stage/measured-regime evidence strictly before a cursor."""

        bounded_control_free_text(recipe_id, "recipe_id")
        bounded_control_free_text(stage_id, "stage_id")
        if (
            not isinstance(measured_regime, MeasuredRegimeBinding)
            or measured_regime._factory_token is not _CAPABILITY_TOKEN
            or measured_regime._store_token is not self._capability_token
        ):
            raise TypeError("reference_bank requires a measured-regime capability from this store")
        before_value, roles, target, target_context = self._validate_query_controls(
            before=before,
            allowed_roles=allowed_roles,
            target_evidence_id=target_evidence_id,
        )
        if (
            target_context["recipe_id"] != recipe_id
            or target_context["stage_id"] != stage_id
            or target not in measured_regime.evidence_ids
        ):
            raise ValueError("target record does not match reference_bank selectors")
        if not measured_regime.reference_roles <= set(roles):
            raise ValueError(
                "regime clean-reference roles are not allowed for this query"
            )
        placeholders = ", ".join("?" for _ in roles)
        member_placeholders = ", ".join("?" for _ in measured_regime.evidence_ids)
        filters = {
            "recipe_id": recipe_id,
            "stage_id": stage_id,
            "colour_regime_id": measured_regime.colour_regime_id,
            "regime_binding_sha256": measured_regime.binding_sha256,
            "regime_reference_provenance": measured_regime.reference_provenance,
            "before": before_value,
            "allowed_roles": list(roles),
            "target_evidence_id": target,
        }
        return self._member_query(
            query_kind="reference_bank",
            filters=filters,
            where_sql=f"""
                JOIN regime_bindings AS rb ON rb.evidence_id = s.evidence_id
                WHERE s.recipe_id = ? AND s.stage_id = ?
                  AND rb.colour_regime_id = ?
                  AND s.evidence_id IN ({member_placeholders})
                  AND s.event_cursor < ?
                  AND s.split_role IN ({placeholders})
                  AND s.evidence_id <> ?
            """,
            parameters=(
                recipe_id,
                stage_id,
                measured_regime.colour_regime_id,
                *measured_regime.evidence_ids,
                before_value,
                *roles,
                target,
            ),
        )

    def vial_history(
        self,
        run_id: str,
        *,
        before: int,
        allowed_roles: object,
        target_evidence_id: str,
    ) -> HistoryQuery:
        """Retrieve only earlier measured stages from the same production run."""

        bounded_control_free_text(run_id, "run_id")
        before_value, roles, target, target_context = self._validate_query_controls(
            before=before,
            allowed_roles=allowed_roles,
            target_evidence_id=target_evidence_id,
        )
        if target_context["run_id"] != run_id:
            raise ValueError("target record does not match vial_history selector")
        placeholders = ", ".join("?" for _ in roles)
        filters = {
            "run_id": run_id,
            "before": before_value,
            "allowed_roles": list(roles),
            "target_evidence_id": target,
        }
        return self._member_query(
            query_kind="vial_history",
            filters=filters,
            where_sql=f"""
                WHERE s.run_id = ? AND s.event_cursor < ?
                  AND s.split_role IN ({placeholders})
                  AND s.evidence_id <> ?
            """,
            parameters=(run_id, before_value, *roles, target),
        )

    def switch_history(
        self,
        previous_recipe_id: str,
        next_recipe_id: str,
        *,
        before: int,
        allowed_roles: object,
        target_evidence_id: str,
    ) -> HistoryQuery:
        """Retrieve exact previous/next recipe switches strictly before a cursor."""

        bounded_control_free_text(previous_recipe_id, "previous_recipe_id")
        bounded_control_free_text(next_recipe_id, "next_recipe_id")
        before_value, roles, target, target_context = self._validate_query_controls(
            before=before,
            allowed_roles=allowed_roles,
            target_evidence_id=target_evidence_id,
        )
        if (
            target_context["previous_recipe_id"] != previous_recipe_id
            or target_context["recipe_id"] != next_recipe_id
        ):
            raise ValueError("target record does not match switch_history selectors")
        placeholders = ", ".join("?" for _ in roles)
        filters = {
            "previous_recipe_id": previous_recipe_id,
            "next_recipe_id": next_recipe_id,
            "before": before_value,
            "allowed_roles": list(roles),
            "target_evidence_id": target,
        }
        return self._member_query(
            query_kind="switch_history",
            filters=filters,
            where_sql=f"""
                WHERE s.previous_recipe_id = ? AND s.recipe_id = ?
                  AND s.event_cursor < ?
                  AND s.split_role IN ({placeholders})
                  AND s.evidence_id <> ?
            """,
            parameters=(previous_recipe_id, next_recipe_id, before_value, *roles, target),
        )


__all__ = [
    "CROSS_RUN_REGIME_TOLERANCE_DELTA_E76",
    "EvidenceStore",
    "HistoryMember",
    "HistoryQuery",
    "IngestResult",
    "MeasuredRegimeBinding",
    "SCHEMA_VERSION",
    "adapt_fill_clips_v1",
    "assert_history_citation",
]

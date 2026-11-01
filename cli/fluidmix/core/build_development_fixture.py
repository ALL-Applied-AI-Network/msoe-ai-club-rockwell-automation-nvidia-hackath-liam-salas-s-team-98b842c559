#!/usr/bin/env python3
"""Build the committed V3/V4 development UI fixture from approved evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Any, Iterable


SOURCE_DATABASE_SHA256 = "88a910ef5780fb6608f3f0911be1685b84bb2d6ef75f200dd6f5160d016455d7"
SOURCE_RECORDS_SHA256 = "c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9"
DEFAULT_SOURCE = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp7-development-history-preparation-01a18128-attempt01/evidence.sqlite3"
)
APPROVED_RECORDS = DEFAULT_SOURCE.with_name("explanation-records.json")


class FixtureBuildError(ValueError):
    """The approved inputs or their referenced presentation assets are invalid."""


SCHEMA = """
CREATE TABLE artifacts (artifact_id TEXT PRIMARY KEY, role TEXT NOT NULL, path TEXT NOT NULL,
 sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL CHECK(byte_size >= 0), UNIQUE(role,path)) STRICT;
CREATE TABLE runs (run_id TEXT PRIMARY KEY, split_role TEXT NOT NULL, split_version TEXT NOT NULL,
 content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE vials (vial_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
 case_id TEXT NOT NULL, vial_ordinal INTEGER NOT NULL CHECK(vial_ordinal > 0),
 block_recipe_id TEXT NOT NULL, block_position INTEGER NOT NULL CHECK(block_position > 0),
 recipe_id TEXT NOT NULL, previous_recipe_id TEXT, recipe_json TEXT NOT NULL,
 dispense_order_json TEXT NOT NULL, content_sha256 TEXT NOT NULL,
 UNIQUE(run_id,vial_ordinal)) STRICT;
CREATE TABLE fills (fill_id TEXT PRIMARY KEY, vial_id TEXT NOT NULL UNIQUE REFERENCES vials(vial_id),
 frame_range_json TEXT NOT NULL, transition_ranges_json TEXT NOT NULL,
 content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE clips (clip_id TEXT PRIMARY KEY, fill_id TEXT NOT NULL UNIQUE REFERENCES fills(fill_id),
 artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id), content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE stages (evidence_id TEXT PRIMARY KEY, record_sha256 TEXT NOT NULL,
 event_cursor INTEGER NOT NULL UNIQUE CHECK(event_cursor >= 0),
 run_id TEXT NOT NULL REFERENCES runs(run_id), vial_id TEXT NOT NULL REFERENCES vials(vial_id),
 stage_id TEXT NOT NULL, stage_ordinal INTEGER NOT NULL CHECK(stage_ordinal > 0),
 tank_id TEXT NOT NULL, recipe_id TEXT NOT NULL, previous_recipe_id TEXT,
 split_role TEXT NOT NULL, phase TEXT NOT NULL, UNIQUE(vial_id,stage_ordinal)) STRICT;
CREATE INDEX stages_reference_lookup ON stages(recipe_id,stage_id,split_role,event_cursor,evidence_id);
CREATE INDEX stages_vial_lookup ON stages(run_id,event_cursor,evidence_id);
CREATE INDEX stages_switch_lookup ON stages(previous_recipe_id,recipe_id,split_role,event_cursor,evidence_id);
CREATE TABLE images (image_id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL UNIQUE REFERENCES stages(evidence_id),
 rgb_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
 mask_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id), mask_source TEXT NOT NULL,
 content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE measurements (measurement_id TEXT PRIMARY KEY,
 evidence_id TEXT NOT NULL UNIQUE REFERENCES stages(evidence_id), median_lab_json TEXT NOT NULL,
 mad_lab_json TEXT NOT NULL, iqr_lab_json TEXT NOT NULL, pairwise_delta_e76_json TEXT NOT NULL,
 content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE decisions (evidence_id TEXT PRIMARY KEY REFERENCES stages(evidence_id),
 decision_id TEXT NOT NULL, disposition TEXT NOT NULL, limit_codes_json TEXT NOT NULL,
 history_refs_json TEXT NOT NULL, content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE explanations (evidence_id TEXT PRIMARY KEY REFERENCES stages(evidence_id),
 explanation_id TEXT NOT NULL, model_revision TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
 image_ids_sent_json TEXT NOT NULL, guard_result TEXT NOT NULL, content_sha256 TEXT NOT NULL) STRICT;
CREATE TABLE history_queries (query_sha256 TEXT PRIMARY KEY, query_kind TEXT NOT NULL,
 filters_json TEXT NOT NULL, member_count INTEGER NOT NULL CHECK(member_count >= 0)) STRICT;
CREATE TABLE history_members (query_sha256 TEXT NOT NULL REFERENCES history_queries(query_sha256),
 member_ordinal INTEGER NOT NULL CHECK(member_ordinal >= 0),
 evidence_id TEXT NOT NULL REFERENCES stages(evidence_id), image_id TEXT NOT NULL REFERENCES images(image_id),
 member_sha256 TEXT NOT NULL, PRIMARY KEY(query_sha256,member_ordinal),
 UNIQUE(query_sha256,evidence_id)) STRICT;
CREATE TABLE regime_bindings (colour_regime_id TEXT NOT NULL,
 evidence_id TEXT NOT NULL REFERENCES stages(evidence_id), measurement_sha256 TEXT NOT NULL,
 binding_sha256 TEXT NOT NULL, PRIMARY KEY(colour_regime_id,evidence_id)) STRICT;
CREATE INDEX regime_lookup ON regime_bindings(colour_regime_id,evidence_id);
CREATE TABLE regime_references (colour_regime_id TEXT NOT NULL,
 evidence_id TEXT NOT NULL REFERENCES stages(evidence_id), reference_role TEXT NOT NULL,
 split_version TEXT NOT NULL, event_cursor INTEGER NOT NULL CHECK(event_cursor >= 0),
 run_id TEXT NOT NULL, recipe_id TEXT NOT NULL, stage_id TEXT NOT NULL,
 measurement_sha256 TEXT NOT NULL, reference_sha256 TEXT NOT NULL,
 PRIMARY KEY(colour_regime_id,evidence_id)) STRICT;
CREATE TABLE sim_metadata (metadata_id TEXT PRIMARY KEY, payload BLOB NOT NULL,
 payload_sha256 TEXT NOT NULL) STRICT;
"""

COPY_TABLES: dict[str, tuple[str, str]] = {
    "runs": ("run_id,split_role,split_version,content_sha256", "run_id"),
    "vials": (
        "vial_id,run_id,case_id,vial_ordinal,block_recipe_id,block_position,recipe_id,"
        "previous_recipe_id,recipe_json,dispense_order_json,content_sha256",
        "run_id,vial_ordinal",
    ),
    "stages": (
        "evidence_id,record_sha256,event_cursor,run_id,vial_id,stage_id,stage_ordinal,"
        "tank_id,recipe_id,previous_recipe_id,split_role,phase",
        "event_cursor,evidence_id",
    ),
    "images": (
        "image_id,evidence_id,rgb_artifact_id,mask_artifact_id,mask_source,content_sha256",
        "evidence_id",
    ),
    "measurements": (
        "measurement_id,evidence_id,median_lab_json,mad_lab_json,iqr_lab_json,"
        "pairwise_delta_e76_json,content_sha256",
        "evidence_id",
    ),
    "decisions": (
        "evidence_id,decision_id,disposition,limit_codes_json,history_refs_json,content_sha256",
        "evidence_id",
    ),
    "explanations": (
        "evidence_id,explanation_id,model_revision,prompt_sha256,image_ids_sent_json,"
        "guard_result,content_sha256",
        "evidence_id",
    ),
    "history_queries": ("query_sha256,query_kind,filters_json,member_count", "query_sha256"),
    "history_members": (
        "query_sha256,member_ordinal,evidence_id,image_id,member_sha256",
        "query_sha256,member_ordinal",
    ),
    "regime_bindings": (
        "colour_regime_id,evidence_id,measurement_sha256,binding_sha256",
        "colour_regime_id,evidence_id",
    ),
    "regime_references": (
        "colour_regime_id,evidence_id,reference_role,split_version,event_cursor,run_id,"
        "recipe_id,stage_id,measurement_sha256,reference_sha256",
        "colour_regime_id,evidence_id",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_source(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")

    def authorizer(action: int, argument: str | None, *_: object) -> int:
        if action in {
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
        } and argument == "sim_metadata":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorizer)
    if connection.execute("PRAGMA user_version").fetchone()[0] != 3:
        connection.close()
        raise FixtureBuildError("source evidence database must use schema version 3")
    return connection


def rows(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    columns, ordering = COPY_TABLES[table]
    return list(connection.execute(f"SELECT {columns} FROM {table} ORDER BY {ordering}"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def copy_artifacts(
    connection: sqlite3.Connection, destination: Path, artifact_root: Path
) -> list[sqlite3.Row]:
    selected = list(connection.execute(
        """
        SELECT DISTINCT a.artifact_id,a.role,a.path,a.sha256,a.byte_size
        FROM artifacts AS a
        JOIN images AS i ON a.artifact_id IN (i.rgb_artifact_id,i.mask_artifact_id)
        ORDER BY a.artifact_id
        """
    ))
    if len(selected) != 90 or {row["role"] for row in selected} != {"rgb", "mask"}:
        raise FixtureBuildError("expected exactly 45 RGB and 45 mask presentation artifacts")
    root = artifact_root.resolve()
    for row in selected:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise FixtureBuildError(f"unsafe artifact path {row['path']!r}")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise FixtureBuildError(f"artifact escapes root: {row['path']}") from error
        if source.is_symlink() or not source.is_file():
            raise FixtureBuildError(f"artifact is missing or a symlink: {row['path']}")
        if source.stat().st_size != row["byte_size"] or file_sha256(source) != row["sha256"]:
            raise FixtureBuildError(f"artifact hash or size mismatch: {row['path']}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return selected


def rebuild_database(
    connection: sqlite3.Connection, destination: Path, artifact_rows: list[sqlite3.Row]
) -> None:
    database = destination / "evidence.sqlite3"
    target = sqlite3.connect(database)
    try:
        target.execute("PRAGMA page_size=4096")
        target.execute("PRAGMA journal_mode=OFF")
        target.execute("PRAGMA synchronous=OFF")
        target.execute("PRAGMA foreign_keys=ON")
        target.executescript(SCHEMA)
        target.execute("PRAGMA user_version=3")
        artifact_columns = "artifact_id,role,path,sha256,byte_size"
        target.executemany(
            f"INSERT INTO artifacts ({artifact_columns}) VALUES (?,?,?,?,?)",
            [tuple(row) for row in artifact_rows],
        )
        for table in COPY_TABLES:
            source_rows = rows(connection, table)
            columns = COPY_TABLES[table][0]
            placeholders = ",".join("?" for _ in columns.split(","))
            target.executemany(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                [tuple(row) for row in source_rows],
            )
        target.commit()
        if target.execute("SELECT count(*) FROM sim_metadata").fetchone()[0] != 0:
            raise FixtureBuildError("derived sim_metadata table is not empty")
        if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise FixtureBuildError("derived evidence database failed quick_check")
    finally:
        target.close()


def validate_records(
    connection: sqlite3.Connection, source_records: list[dict[str, Any]]
) -> None:
    records_by_key = {(record["run_id"], record["vial_ordinal"]): record for record in source_records}
    if len(source_records) != 18 or len(records_by_key) != 18:
        raise FixtureBuildError("expected 18 uniquely identified development decisions")
    if {record.get("disposition") for record in source_records} != {"insufficient_evidence"}:
        raise FixtureBuildError("development decisions must remain 18/18 insufficient_evidence")
    vial_rows = list(connection.execute(
        "SELECT vial_id,run_id,vial_ordinal,recipe_id FROM vials ORDER BY run_id,vial_ordinal"
    ))
    if len(vial_rows) != 18 or {(r["run_id"], r["vial_ordinal"]) for r in vial_rows} != set(records_by_key):
        raise FixtureBuildError("decision records disagree with the 18 source vials")

def build_reel(
    destination: Path, records: list[dict[str, Any]], database: Path, ffmpeg: Path
) -> None:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        images: list[Path] = []
        for record in records:
            row = connection.execute(
                """SELECT a.path FROM vials v JOIN stages s ON s.vial_id=v.vial_id
                   JOIN images i ON i.evidence_id=s.evidence_id
                   JOIN artifacts a ON a.artifact_id=i.rgb_artifact_id
                   WHERE v.run_id=? AND v.vial_ordinal=? AND s.stage_ordinal=1""",
                (record["run_id"], record["vial_ordinal"]),
            ).fetchone()
            if row is None:
                raise FixtureBuildError("development record has no first-stage presentation frame")
            images.append(destination / row["path"])
    finally:
        connection.close()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        concat = Path(handle.name)
        for image in images:
            escaped = str(image).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\nduration 15\n")
        escaped = str(images[-1]).replace("'", "'\\''")
        handle.write(f"file '{escaped}'\n")
    try:
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(concat), "-vf",
            "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-r", "1", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-flags", "+bitexact", "-fflags", "+bitexact", "-map_metadata", "-1",
            "-t", "270", "-movflags", "+faststart",
            str(destination / "development-reel.mp4"),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise FixtureBuildError(f"ffmpeg failed: {completed.stderr.strip()}")
    finally:
        concat.unlink(missing_ok=True)


def tree_entries(root: Path, *, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded = set(exclude)
    return [{
        "path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
    } for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded]


def build_fixture(database: Path, records_path: Path, destination: Path, ffmpeg: Path) -> Path:
    database = database.resolve()
    records_path = records_path.resolve()
    destination = destination.resolve()
    if database != DEFAULT_SOURCE.resolve() or records_path != APPROVED_RECORDS.resolve():
        raise FixtureBuildError("builder inputs must use the exact approved source path")
    if file_sha256(database) != SOURCE_DATABASE_SHA256:
        raise FixtureBuildError("source evidence.sqlite3 SHA-256 mismatch")
    if file_sha256(records_path) != SOURCE_RECORDS_SHA256:
        raise FixtureBuildError("source explanation-records.json SHA-256 mismatch")
    source_records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(source_records, list):
        raise FixtureBuildError("source explanation records must be a list")
    artifact_root = database.parents[3]
    destination.mkdir(parents=True, exist_ok=False)
    connection = open_source(database)
    try:
        run_rows = rows(connection, "runs")
        if len(run_rows) != 6 or {row["split_version"] for row in run_rows} != {
            "whole-run-varied-v3", "whole-run-varied-v4"
        }:
            raise FixtureBuildError("expected exact six-run V3+V4 provenance")
        artifacts = copy_artifacts(connection, destination, artifact_root)
        rebuild_database(connection, destination, artifacts)
        validate_records(connection, source_records)
    finally:
        connection.close()
    shutil.copyfile(records_path, destination / "development-records.json")
    build_reel(destination, source_records, destination / "evidence.sqlite3", ffmpeg.resolve())
    manifest = {
        "schema_version": "fluidmix-development-fixture-manifest-v1",
        "label": "Development data",
        "source_inputs": [
            {"name": "evidence.sqlite3", "sha256": SOURCE_DATABASE_SHA256,
             "byte_size": database.stat().st_size},
            {"name": "explanation-records.json", "sha256": SOURCE_RECORDS_SHA256,
             "byte_size": records_path.stat().st_size},
        ],
        "source_ids": [row["run_id"] for row in run_rows],
        "split_versions": sorted({row["split_version"] for row in run_rows}),
        "decision_summary": {"insufficient_evidence": 18, "keep": 0, "discard": 0},
        "derivation": (
            "Rebuilt schema-v3 SQLite from explicit non-sim_metadata tables and copied only "
            "DB-addressed RGB/mask files after source hash and size verification. The original "
            "development records are preserved verbatim in development-records.json and are "
            "consumed directly by the development-only UI adapter. No final WP7 decision "
            "artifact is synthesized. "
            "The 270-second H.264 reel repeats each source-backed vial focus frame for 15 seconds."
        ),
        "claim_boundary": (
            "Synthetic Development data only. Not plant evidence, physical validation, a final "
            "WP7 run, or permission to change recipes or operate equipment."
        ),
        "sim_metadata": {"row_count": 0, "source_contents_read": False},
        "derived_files": tree_entries(destination, exclude={"manifest.json"}),
    }
    write_json(destination / "manifest.json", manifest)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    print(build_fixture(DEFAULT_SOURCE, APPROVED_RECORDS, args.output, args.ffmpeg))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize the selected operator UI source into the browser wire contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any


class MaterializationError(ValueError):
    """The selected UI source is missing, ambiguous, or internally inconsistent."""


def resolve_source(config_path: Path, *, repository_root: Path) -> tuple[str, Path | None]:
    """Resolve the one-value source selector without opening source evidence."""

    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read source selector: {error}") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"source"}
        or not isinstance(value.get("source"), str)
        or not value["source"].strip()
    ):
        raise MaterializationError(
            "source.json must contain exactly one non-empty string value named source"
        )
    source = value["source"]
    if source == "mock":
        return ("mock", None)
    if source == "development":
        development = repository_root.resolve() / "ui-data" / "development"
        if not development.is_dir():
            raise MaterializationError(
                f"development source directory does not exist: {development}"
            )
        return ("development", development)
    selected = Path(source).expanduser()
    if not selected.is_absolute():
        selected = repository_root / selected
    selected = selected.resolve()
    if not selected.is_dir():
        raise MaterializationError(f"final source directory does not exist: {selected}")
    return ("final", selected)


_TABLE_COLUMNS = {
    "artifacts": {"artifact_id", "role", "path", "sha256", "byte_size"},
    "runs": {"run_id", "split_role", "split_version", "content_sha256"},
    "vials": {
        "vial_id", "run_id", "case_id", "vial_ordinal", "block_recipe_id",
        "block_position", "recipe_id", "previous_recipe_id", "recipe_json",
        "dispense_order_json", "content_sha256",
    },
    "fills": {"fill_id"},
    "clips": {"clip_id"},
    "stages": {
        "evidence_id", "record_sha256", "event_cursor", "run_id", "vial_id",
        "stage_id", "stage_ordinal", "tank_id", "recipe_id",
        "previous_recipe_id", "split_role", "phase",
    },
    "images": {
        "image_id", "evidence_id", "rgb_artifact_id", "mask_artifact_id",
        "mask_source", "content_sha256",
    },
    "measurements": {
        "measurement_id", "evidence_id", "median_lab_json", "mad_lab_json",
        "iqr_lab_json", "pairwise_delta_e76_json", "content_sha256",
    },
    "decisions": {"evidence_id"},
    "explanations": {"evidence_id"},
    "history_queries": {"query_sha256"},
    "history_members": {"query_sha256", "member_ordinal"},
    "regime_bindings": {"colour_regime_id", "evidence_id"},
    "regime_references": {"colour_regime_id", "evidence_id"},
    # Presence is schema-v3 evidence; its columns and rows are deliberately opaque here.
    "sim_metadata": set(),
}
_SHA256_LENGTH = 64
_STAGE_FIELDS = {
    "evidence_id", "measurement_id", "image_id", "vial_id", "recipe_id",
    "stage_id", "stage_ordinal", "tank_id", "source", "median_lab", "validity",
    "stage_outlier", "target_delta_e76", "peer_delta_e76", "threshold_delta_e76",
    "reference_center_lab", "donors", "query_sha256",
    "donor_store_snapshot_sha256", "configuration_sha256", "limit_codes", "unit",
    "rgb_sha256", "mask_sha256", "entry_sha256", "registry_sha256",
    "measurement_sha256",
}
_TARGET_BINDING_FIELDS = {
    "evidence_id", "measurement_id", "image_id", "vial_id", "recipe_id",
    "stage_id", "stage_ordinal", "tank_id", "source", "median_lab", "validity",
    "limit_codes", "rgb_sha256", "mask_sha256", "entry_sha256",
    "registry_sha256", "measurement_sha256",
}
_DONOR_FIELDS = {
    "evidence_id", "image_id", "batch_id", "run_id", "recipe_id",
    "stage_ordinal", "tank_id", "median_lab",
}
_EXPLANATION_FIELDS = {"template_id", "model", "text", "citations", "claim_boundary"}
_OPERATOR_FIELDS = {
    "schema_version", "decision_id", "disposition", "model", "denominators",
    "source_bindings", "limit_codes", "explanation",
}
_SOURCE_BINDING_FIELDS = {"targets", "development_references"}
_DEVELOPMENT_REFERENCE_FIELDS = {
    "stage_ordinal", "query_sha256", "donor_store_snapshot_sha256",
    "configuration_sha256", "donors",
}
_CLAIM_BOUNDARY = (
    "Camera colour versus synthetic development reference only; cause, physical "
    "defect, carryover, next-vial effect, and plant action are unverified."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read {label}: {error}") from error


def _discover_unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise MaterializationError(
            f"expected exactly one {name}; found {len(matches)}"
        )
    return matches[0]


def _contained_decision_input(
    root: Path, path: Path, label: str, *, directory: bool = False
) -> Path:
    if path.is_symlink():
        raise MaterializationError(f"{label} decision input cannot be a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise MaterializationError(f"{label} decision input escapes the source directory") from error
    expected = resolved.is_dir() if directory else resolved.is_file()
    _require(expected, f"{label} decision input is missing or has the wrong type")
    return resolved


def _decision_records(root: Path) -> list[dict[str, Any]]:
    json_path = root / "wp7-final-decisions.json"
    jsonl_path = root / "wp7-final-decisions.jsonl"
    directory = root / "decisions"
    forms = [
        json_path.is_file() or json_path.is_symlink(),
        jsonl_path.is_file() or jsonl_path.is_symlink(),
        directory.is_dir() or directory.is_symlink(),
    ]
    if sum(forms) > 1:
        raise MaterializationError("ambiguous decision inputs")
    if not any(forms):
        raise MaterializationError("missing WP7 final decision records")
    raw: Any
    if forms[0]:
        json_path = _contained_decision_input(root, json_path, "JSON")
        raw = _load_json(json_path, "WP7 decisions")
        if isinstance(raw, dict) and set(raw) == {"decisions"}:
            raw = raw["decisions"]
        elif isinstance(raw, dict):
            raw = [raw]
    elif forms[1]:
        jsonl_path = _contained_decision_input(root, jsonl_path, "JSONL")
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            raw = [json.loads(line) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            raise MaterializationError(f"cannot read WP7 JSONL decisions: {error}") from error
    else:
        directory = _contained_decision_input(
            root, directory, "decisions directory", directory=True
        )
        paths = [
            _contained_decision_input(root, path, "decisions member")
            for path in sorted(directory.glob("*.json"))
        ]
        _require(bool(paths), "decisions directory contains no JSON records")
        raw = [_load_json(path, f"WP7 decision {path.name}") for path in paths]
    _require(isinstance(raw, list) and bool(raw), "WP7 decision collection must be non-empty")
    _require(all(isinstance(record, dict) for record in raw), "WP7 decisions must be objects")
    records = list(raw)
    ids = [record.get("decision_id") for record in records]
    _require(
        all(isinstance(value, str) and value for value in ids) and len(ids) == len(set(ids)),
        "WP7 decision identifiers must be unique non-empty strings",
    )
    return records


def _read_connection(database: Path) -> sqlite3.Connection:
    uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")

    def authorizer(
        action: int,
        argument_one: str | None,
        _argument_two: str | None,
        _database_name: str | None,
        _trigger_name: str | None,
    ) -> int:
        if action in {
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
        } and argument_one == "sim_metadata":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorizer)
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != 3:
        connection.close()
        raise MaterializationError(
            f"unsupported evidence schema version {version}; expected 3"
        )
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    missing_tables = set(_TABLE_COLUMNS) - tables
    if missing_tables:
        connection.close()
        raise MaterializationError(
            "evidence database is missing required tables: " + ", ".join(sorted(missing_tables))
        )
    for table, expected in _TABLE_COLUMNS.items():
        if table == "sim_metadata":
            continue
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        missing = expected - columns
        if missing:
            connection.close()
            raise MaterializationError(
                f"evidence table {table} is missing required columns: "
                + ", ".join(sorted(missing))
            )
    return connection


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact(source: Path, relative: object, expected_sha: object, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} path is invalid")
    _require(isinstance(expected_sha, str) and len(expected_sha) == _SHA256_LENGTH,
             f"{label} SHA-256 is invalid")
    candidate = source / relative
    _require(not candidate.is_symlink(), f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(source.resolve())
    except ValueError as error:
        raise MaterializationError(f"{label} path escapes the source directory") from error
    _require(resolved.is_file(), f"{label} file is missing")
    _require(_file_sha(resolved) == expected_sha, f"{label} SHA-256 mismatch")
    return resolved


def _lab(value: Any, label: str, *, nullable: bool) -> list[float] | None:
    if value is None and nullable:
        return None
    _require(isinstance(value, list) and len(value) == 3, f"{label} must contain three channels")
    result: list[float] = []
    for channel in value:
        _require(
            not isinstance(channel, bool)
            and isinstance(channel, (int, float))
            and math.isfinite(float(channel)),
            f"{label} channels must be finite numbers",
        )
        result.append(float(channel))
    return result


def _number(value: Any, label: str, *, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{label} must be a finite number",
    )
    return float(value)


def _nonempty_text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def _sha256_text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _delta_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum(
        (left_channel - right_channel) ** 2
        for left_channel, right_channel in zip(left, right, strict=True)
    ))


def _focus_stage(record: dict[str, Any]) -> dict[str, Any]:
    stages = record["stages"]
    if record["disposition"] == "discard":
        for stage in stages:
            if stage["stage_outlier"] is True:
                return stage
    valid = [
        stage for stage in stages
        if stage["validity"] == "valid"
        and stage["target_delta_e76"] is not None
        and stage["threshold_delta_e76"] is not None
    ]
    if valid:
        return max(
            valid,
            key=lambda stage: (
                float("inf") if float(stage["threshold_delta_e76"]) == 0
                else float(stage["target_delta_e76"]) / float(stage["threshold_delta_e76"])
            ),
        )
    return stages[0]


def _validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version", "decision_id", "disposition", "model", "stages",
        "denominators", "limit_codes", "explanation", "operator_record",
    }
    _require(set(record) == required, f"decision {record.get('decision_id')} fields are invalid")
    _require(record["schema_version"] == "wp7-final-decision-v1", "unsupported WP7 decision schema")
    _nonempty_text(record["decision_id"], "WP7 decision_id")
    _require(record["model"] is None, "WP7 final decision model must be null")
    _require(record["disposition"] in {"keep", "discard", "insufficient_evidence"},
             "WP7 disposition is invalid")
    _require(isinstance(record["stages"], list) and len(record["stages"]) in {2, 3},
             "WP7 decision must contain two or three stages")
    ordinals = [stage.get("stage_ordinal") for stage in record["stages"]]
    _require(ordinals == list(range(1, len(ordinals) + 1)), "WP7 stages must be in ordinal order")
    _require(isinstance(record["limit_codes"], list) and all(
        isinstance(code, str) and code for code in record["limit_codes"]
    ), "WP7 limit codes must be strings")
    explanation = record["explanation"]
    _require(isinstance(explanation, dict) and set(explanation) == _EXPLANATION_FIELDS,
             "WP7 explanation fields are invalid")
    _require(
        explanation["template_id"] == "wp7-camera-colour-synthetic-reference-v1"
        and explanation["model"] is None,
        "WP7 explanation template or model is invalid",
    )
    _nonempty_text(explanation["text"], "WP7 explanation text")
    _require(explanation["claim_boundary"] == _CLAIM_BOUNDARY,
             "WP7 explanation claim boundary is invalid")
    citations = explanation["citations"]
    _require(isinstance(citations, list) and bool(citations), "WP7 explanation citations are missing")
    operator = record["operator_record"]
    _require(isinstance(operator, dict) and set(operator) == _OPERATOR_FIELDS,
             f"decision {record['decision_id']} operator record fields are invalid")
    _require(
        operator["schema_version"] == "wp7-operator-record-v1"
        and operator["decision_id"] == record["decision_id"]
        and operator["disposition"] == record["disposition"]
        and operator["model"] is None
        and operator["denominators"] == record["denominators"]
        and operator["limit_codes"] == record["limit_codes"]
        and operator["explanation"] == explanation,
        f"decision {record['decision_id']} operator record mismatch",
    )
    source_bindings = operator["source_bindings"]
    _require(
        isinstance(source_bindings, dict)
        and set(source_bindings) == _SOURCE_BINDING_FIELDS
        and isinstance(source_bindings["targets"], list)
        and isinstance(source_bindings["development_references"], list)
        and len(source_bindings["targets"]) == len(record["stages"])
        and len(source_bindings["development_references"]) == len(record["stages"]),
        "WP7 operator source bindings are invalid",
    )
    expected: list[dict[str, Any]] = []
    for stage, target, development in zip(
        record["stages"],
        source_bindings["targets"],
        source_bindings["development_references"],
        strict=True,
    ):
        _require(set(stage) == _STAGE_FIELDS,
                 f"decision {record['decision_id']} stage fields are invalid")
        _require(isinstance(target, dict) and set(target) == _TARGET_BINDING_FIELDS,
                 "WP7 operator target binding fields are invalid")
        _require(
            isinstance(development, dict)
            and set(development) == _DEVELOPMENT_REFERENCE_FIELDS,
            "WP7 operator development-reference binding fields are invalid",
        )
        for field in _TARGET_BINDING_FIELDS - {"validity", "limit_codes"}:
            _require(target[field] == stage[field],
                     f"WP7 operator target binding {field} mismatch")
        expected_development = {
            "stage_ordinal": stage["stage_ordinal"],
            "query_sha256": stage["query_sha256"],
            "donor_store_snapshot_sha256": stage["donor_store_snapshot_sha256"],
            "configuration_sha256": stage["configuration_sha256"],
            "donors": stage["donors"],
        }
        _require(development == expected_development,
                 "WP7 operator development-reference binding mismatch")
        for field in (
            "evidence_id", "measurement_id", "image_id", "vial_id", "recipe_id",
            "stage_id", "tank_id",
        ):
            _nonempty_text(stage[field], f"WP7 stage {field}")
        _require(type(stage["stage_ordinal"]) is int and stage["stage_ordinal"] > 0,
                 "WP7 stage ordinal is invalid")
        _require(stage["source"] == "runtime_camera", "WP7 stage source is invalid")
        _require(stage["unit"] == "delta_e76", "WP7 stage unit is invalid")
        _require(stage["validity"] in {"valid", "insufficient_evidence"},
                 "WP7 stage validity is invalid")
        _require(isinstance(stage["limit_codes"], list) and all(
            isinstance(code, str) and code for code in stage["limit_codes"]
        ), "WP7 stage limit codes must be strings")
        for hash_field in (
            "rgb_sha256", "mask_sha256", "entry_sha256", "registry_sha256",
            "measurement_sha256",
        ):
            _sha256_text(stage[hash_field], f"WP7 stage {hash_field}")
        target_lab = _lab(target["median_lab"], "WP7 target binding median Lab", nullable=True)
        _require(isinstance(target["validity"], str) and bool(target["validity"]),
                 "WP7 target binding validity is invalid")
        _require(isinstance(target["limit_codes"], list) and all(
            isinstance(code, str) and code for code in target["limit_codes"]
        ), "WP7 target binding limit codes are invalid")
        _lab(stage["median_lab"], "WP7 median Lab", nullable=True)
        reference_center = _lab(
            stage["reference_center_lab"], "WP7 reference center Lab", nullable=True
        )
        for number_field in (
            "target_delta_e76", "peer_delta_e76", "threshold_delta_e76"
        ):
            _number(stage[number_field], f"WP7 {number_field}", nullable=True)
        donors = stage["donors"]
        _require(isinstance(donors, list), "WP7 stage donors are invalid")
        if stage["query_sha256"] is None:
            _require(
                stage["donor_store_snapshot_sha256"] is None
                and stage["configuration_sha256"] is None
                and donors == []
                and stage["validity"] == "insufficient_evidence"
                and stage["stage_outlier"] is None
                and stage["target_delta_e76"] is None
                and stage["peer_delta_e76"] is None
                and stage["threshold_delta_e76"] is None
                and reference_center is None,
                "WP7 unsupported-selector stage binding is invalid",
            )
            expected_codes = ["unsupported_development_selector"]
            if target["validity"] != "valid":
                expected_codes.extend(("target_measurement_invalid", *target["limit_codes"]))
            expected_codes = list(dict.fromkeys(expected_codes))
            _require(stage["limit_codes"] == expected_codes,
                     "WP7 unsupported-selector stage limit codes are invalid")
        else:
            _sha256_text(stage["query_sha256"], "WP7 query hash")
            _sha256_text(
                stage["donor_store_snapshot_sha256"], "WP7 donor snapshot hash"
            )
            _sha256_text(stage["configuration_sha256"], "WP7 configuration hash")
            _require(len(donors) == 2, "WP7 development donor denominator is invalid")
            donor_labs: list[list[float]] = []
            for donor in donors:
                _require(isinstance(donor, dict) and set(donor) == _DONOR_FIELDS,
                         "WP7 donor fields are invalid")
                for field in (
                    "evidence_id", "image_id", "batch_id", "run_id", "recipe_id", "tank_id"
                ):
                    _nonempty_text(donor[field], f"WP7 donor {field}")
                _require(
                    donor["recipe_id"] == stage["recipe_id"]
                    and donor["stage_ordinal"] == stage["stage_ordinal"]
                    and donor["tank_id"] == stage["tank_id"]
                    and donor["evidence_id"] != stage["evidence_id"]
                    and donor["image_id"] != stage["image_id"],
                    "WP7 donor identity binding is invalid",
                )
                donor_lab = _lab(donor["median_lab"], "WP7 donor median Lab", nullable=False)
                assert donor_lab is not None
                donor_labs.append(donor_lab)
            for identity in ("batch_id", "run_id", "evidence_id", "image_id"):
                _require(len({donor[identity] for donor in donors}) == 2,
                         f"WP7 donor {identity} identities are not distinct")
            peer_delta = _delta_distance(donor_labs[0], donor_labs[1])
            center = [
                (donor_labs[0][channel] + donor_labs[1][channel]) / 2.0
                for channel in range(3)
            ]
            threshold = peer_delta / 2.0 + 1.0
            _require(
                stage["peer_delta_e76"] == peer_delta
                and stage["threshold_delta_e76"] == threshold
                and reference_center == center,
                "WP7 donor-derived comparison values are invalid",
            )
            if peer_delta > 2.0:
                _require(
                    stage["validity"] == "insufficient_evidence"
                    and stage["stage_outlier"] is None
                    and stage["target_delta_e76"] is None
                    and stage["limit_codes"] == ["development_peer_delta_above_limit"],
                    "WP7 peer-delta abstention binding is invalid",
                )
            elif target["validity"] != "valid":
                expected_codes = list(dict.fromkeys(
                    ["target_measurement_invalid", *target["limit_codes"]]
                ))
                _require(
                    stage["validity"] == "insufficient_evidence"
                    and stage["stage_outlier"] is None
                    and stage["target_delta_e76"] is None
                    and stage["limit_codes"] == expected_codes,
                    "WP7 invalid-target abstention binding is invalid",
                )
            else:
                _require(target_lab is not None, "valid WP7 target Lab is missing")
                target_delta = _delta_distance(target_lab, center)
                _require(
                    stage["validity"] == "valid"
                    and stage["stage_outlier"] is (target_delta > threshold)
                    and stage["target_delta_e76"] == target_delta
                    and stage["limit_codes"] == [],
                    "WP7 valid stage comparison binding is invalid",
                )
        expected.append({
            "role": "target", "stage_ordinal": stage.get("stage_ordinal"),
            "evidence_id": stage.get("evidence_id"),
            "measurement_id": stage.get("measurement_id"), "image_id": stage.get("image_id"),
        })
        for donor in donors:
            expected.append({
                "role": "synthetic_development_donor",
                "stage_ordinal": stage.get("stage_ordinal"),
                "evidence_id": donor.get("evidence_id"), "image_id": donor.get("image_id"),
                "batch_id": donor.get("batch_id"),
            })
    comparisons: list[str] = []
    for stage in record["stages"]:
        if stage["validity"] == "valid":
            relation = "above" if stage["stage_outlier"] else "at or below"
            comparisons.append(
                f"stage {stage['stage_ordinal']} target {stage['evidence_id']} "
                f"camera-colour distance {stage['target_delta_e76']!r} is {relation} "
                f"threshold {stage['threshold_delta_e76']!r}"
            )
        else:
            comparisons.append(
                f"stage {stage['stage_ordinal']} target {stage['evidence_id']} has "
                "insufficient camera-colour evidence"
            )
    expected_text = (
        f"Disposition {record['disposition']}. Camera-colour versus synthetic-reference comparison: "
        + "; ".join(comparisons)
        + ". Cause, physical defect, carryover, next-vial effect, and plant action are "
        "unverified."
    )
    _require(explanation["text"] == expected_text,
             "WP7 explanation text does not match its stages")
    calculated_disposition = (
        "discard"
        if any(stage["validity"] == "valid" and stage["stage_outlier"] is True
               for stage in record["stages"])
        else "keep"
        if all(stage["validity"] == "valid" and stage["stage_outlier"] is False
               for stage in record["stages"])
        else "insufficient_evidence"
    )
    _require(calculated_disposition == record["disposition"],
             f"decision {record['decision_id']} disposition contradicts its stages")
    aggregate_codes = list(dict.fromkeys(
        code for stage in record["stages"] for code in stage["limit_codes"]
    ))
    _require(aggregate_codes == record["limit_codes"],
             f"decision {record['decision_id']} limit codes contradict its stages")
    calculated_denominators = {
        "programmed_stage_count": len(record["stages"]),
        "valid_stage_count": sum(
            stage["validity"] == "valid" for stage in record["stages"]
        ),
        "outlier_stage_count": sum(
            stage["stage_outlier"] is True for stage in record["stages"]
        ),
        "bound_donor_count": sum(
            len(stage["donors"]) for stage in record["stages"]
        ),
        "required_donors_per_supported_stage": 2,
    }
    _require(record["denominators"] == calculated_denominators,
             f"decision {record['decision_id']} denominators contradict its stages")
    _require(citations == expected, f"decision {record['decision_id']} has missing or mismatched citations")
    for identity in ("evidence_id", "measurement_id", "image_id"):
        _require(
            len({stage[identity] for stage in record["stages"]}) == len(record["stages"]),
            f"WP7 target {identity} identities are not distinct",
        )
    _require(len({stage["tank_id"] for stage in record["stages"]}) == len(record["stages"]),
             "WP7 programmed tank identities are not distinct")
    _require(len({stage["vial_id"] for stage in record["stages"]}) == 1
             and len({stage["recipe_id"] for stage in record["stages"]}) == 1,
             "WP7 decision crosses vial or recipe identity")


def _join_stage(
    connection: sqlite3.Connection,
    source: Path,
    stage: dict[str, Any],
    target_binding: dict[str, Any],
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT s.run_id, s.vial_id, s.stage_id, s.stage_ordinal, s.tank_id,
               s.recipe_id, v.run_id AS vial_run_id, v.vial_ordinal, v.recipe_json, r.split_role,
               r.split_version, i.image_id, m.measurement_id, m.median_lab_json,
               m.mad_lab_json, m.iqr_lab_json,
               rgb.path AS rgb_path, rgb.sha256 AS rgb_sha256,
               rgb.byte_size AS rgb_bytes,
               mask.path AS mask_path, mask.sha256 AS mask_sha256,
               mask.byte_size AS mask_bytes
        FROM stages AS s
        JOIN vials AS v ON v.vial_id = s.vial_id
        JOIN runs AS r ON r.run_id = s.run_id
        JOIN images AS i ON i.evidence_id = s.evidence_id
        JOIN measurements AS m ON m.evidence_id = s.evidence_id
        JOIN artifacts AS rgb ON rgb.artifact_id = i.rgb_artifact_id
        JOIN artifacts AS mask ON mask.artifact_id = i.mask_artifact_id
        WHERE s.evidence_id = ?
        """,
        (stage.get("evidence_id"),),
    ).fetchone()
    _require(row is not None, f"missing SQLite evidence {stage.get('evidence_id')}")
    _require(row["run_id"] == row["vial_run_id"],
             f"SQLite run/vial linkage mismatch for {stage.get('evidence_id')}")
    for field in ("vial_id", "stage_id", "stage_ordinal", "tank_id", "recipe_id", "image_id", "measurement_id"):
        if row[field] != stage.get(field):
            label = "measurement identity" if field == "measurement_id" else f"{field} identity"
            raise MaterializationError(f"{label} mismatch for {stage.get('evidence_id')}")
    median = _lab(json.loads(row["median_lab_json"]), "SQLite median Lab", nullable=True)
    _require(median == _lab(stage.get("median_lab"), "WP7 median Lab", nullable=True),
             f"median Lab mismatch for {stage['evidence_id']}")
    rgb = _safe_artifact(source, row["rgb_path"], row["rgb_sha256"], "RGB artifact")
    mask = _safe_artifact(source, row["mask_path"], row["mask_sha256"], "mask artifact")
    _require(rgb.stat().st_size == row["rgb_bytes"], "RGB artifact byte-size mismatch")
    _require(mask.stat().st_size == row["mask_bytes"], "mask artifact byte-size mismatch")
    _require(stage.get("rgb_sha256") == row["rgb_sha256"],
             f"RGB hash mismatch for {stage['evidence_id']}")
    _require(stage.get("mask_sha256") == row["mask_sha256"],
             f"mask hash mismatch for {stage['evidence_id']}")
    measurement_payload = {
        "image_id": stage.get("image_id"),
        "measurement_id": stage.get("measurement_id"),
        "evidence_id": stage.get("evidence_id"),
        "rgb_sha256": row["rgb_sha256"],
        "mask_sha256": row["mask_sha256"],
        "validity": target_binding.get("validity"),
        "limit_codes": target_binding.get("limit_codes"),
        "median_lab": median,
        "mad_lab": json.loads(row["mad_lab_json"]),
        "iqr_lab": json.loads(row["iqr_lab_json"]),
    }
    _require(
        stage.get("measurement_sha256") == _canonical_sha(measurement_payload),
        f"measurement hash mismatch for {stage['evidence_id']}",
    )
    return {**dict(row), "rgb_source": rgb, "mask_source": mask, "median_lab": median}


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _file_sha(destination) == _file_sha(source):
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _asset_reference(
    source: Path, output_dir: Path, *, kind: str, source_id: str, note: str
) -> dict[str, Any]:
    digest = _file_sha(source)
    suffix = source.suffix.lower() or ".bin"
    relative = Path("assets") / "active" / f"{digest}{suffix}"
    _atomic_copy(source, output_dir / relative)
    return {
        "asset_id": f"{kind}-{digest}",
        "src": "/" + relative.as_posix(),
        "provenance": {"kind": "measured", "source_id": source_id, "note": note},
    }


def _provenance(kind: str, source_id: str, note: str) -> dict[str, str]:
    return {"kind": kind, "source_id": source_id, "note": note}


def _lab_wire(value: list[float] | None, provenance: dict[str, str]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"l": value[0], "a": value[1], "b": value[2], "provenance": provenance}


def _number_wire(value: Any, provenance: dict[str, str]) -> dict[str, Any] | None:
    checked = _number(value, "WP7 comparison value", nullable=True)
    return None if checked is None else {"value": checked, "provenance": provenance}


def _rgb_from_lab(lab: list[float] | None) -> tuple[int, int, int]:
    if lab is None:
        return (128, 128, 128)
    l_value, a_value, b_value = lab
    fy = (l_value + 16.0) / 116.0
    fx = fy + a_value / 500.0
    fz = fy - b_value / 200.0
    delta = 6.0 / 29.0

    def inverse(value: float) -> float:
        return value ** 3 if value > delta else 3 * delta * delta * (value - 4.0 / 29.0)

    x, y, z = 0.95047 * inverse(fx), inverse(fy), 1.08883 * inverse(fz)
    linear = (
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.969266 * x + 1.8760108 * y + 0.041556 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    )

    def encode(value: float) -> int:
        channel = 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, channel)) * 255)

    return tuple(encode(channel) for channel in linear)


def _final_wire(
    source: Path, records: list[dict[str, Any]], connection: sqlite3.Connection,
    output_dir: Path, *, mode: str,
) -> dict[str, Any]:
    mp4s = [path for path in source.rglob("*.mp4") if path.is_file() and not path.is_symlink()]
    _require(len(mp4s) == 1, f"expected exactly one published MP4; found {len(mp4s)}")
    video_source = mp4s[0]
    video = _asset_reference(
        video_source, output_dir, kind="run-video", source_id=_file_sha(video_source),
        note="Hash-verified published presentation media copied without source mutation.",
    )
    vials: list[dict[str, Any]] = []
    recipes: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    cues: list[dict[str, Any]] = []
    selected_run_id: str | None = None
    seen_vial_ids: set[str] = set()
    for slot, record in enumerate(records, start=1):
        _validate_record(record)
        target_bindings = {
            target["evidence_id"]: target
            for target in record["operator_record"]["source_bindings"]["targets"]
        }
        joined = {
            stage["evidence_id"]: _join_stage(
                connection, source, stage, target_bindings[stage["evidence_id"]]
            )
            for stage in record["stages"]
        }
        focus = _focus_stage(record)
        focus_row = joined[focus["evidence_id"]]
        decision_run_ids = {str(row["run_id"]) for row in joined.values()}
        _require(len(decision_run_ids) == 1,
                 f"decision {record['decision_id']} crosses SQLite runs")
        decision_run_id = next(iter(decision_run_ids))
        if selected_run_id is None:
            selected_run_id = decision_run_id
        else:
            _require(
                decision_run_id == selected_run_id,
                f"decision {record['decision_id']} belongs to multiple selected runs",
            )
        vial_id = str(focus["vial_id"])
        _require(all(stage["vial_id"] == vial_id for stage in record["stages"]),
                 f"decision {record['decision_id']} crosses vials")
        _require(vial_id not in seen_vial_ids,
                 f"duplicate vial decision for {vial_id}")
        seen_vial_ids.add(vial_id)
        recipe_id = str(focus["recipe_id"])
        _require(all(stage["recipe_id"] == recipe_id for stage in record["stages"]),
                 f"decision {record['decision_id']} crosses recipes")
        rgb_asset = _asset_reference(
            focus_row["rgb_source"], output_dir, kind="rgb-frame",
            source_id=str(focus["measurement_id"]),
            note=f"WP5 target RGB SHA-256 {focus['rgb_sha256']}.",
        )
        mask_asset = _asset_reference(
            focus_row["mask_source"], output_dir, kind="predicted-mask",
            source_id=str(focus["image_id"]),
            note=f"WP5 predicted mask SHA-256 {focus['mask_sha256']}.",
        )
        target_provenance = _provenance(
            "measured", str(focus["measurement_id"]), "WP5 target median Lab cited by WP7."
        )
        reference_provenance = _provenance(
            "documented", str(focus.get("query_sha256") or focus["evidence_id"]),
            "WP7 synthetic development reference center; unavailable when the comparison abstains.",
        )
        delta_provenance = _provenance(
            "inferred", str(focus["evidence_id"]), "WP7 target Delta E 76 comparison."
        )
        threshold_provenance = _provenance(
            "documented", str(focus.get("configuration_sha256") or focus["evidence_id"]),
            "WP7 comparison threshold; not a plant operating limit.",
        )
        normalized_refs: list[str] = []
        for citation_index, citation in enumerate(record["explanation"]["citations"], start=1):
            normalized_id = f"citation-{record['decision_id']}-{citation_index}"
            normalized_refs.append(normalized_id)
            evidence.append({
                "evidence_id": normalized_id,
                "vial_id": vial_id,
                "target": "stage_citation",
                "label": f"{citation['role']} stage {citation['stage_ordinal']}",
                "provenance": _provenance(
                    "documented", citation["evidence_id"],
                    "Exact evidence identifier preserved from the WP7 explanation citation.",
                ),
            })
        explanation = {
            "source": "deterministic_template" if record["explanation"].get("model") is None else "cosmos_reason2",
            "text": record["explanation"]["text"],
            "evidence_refs": normalized_refs,
        }
        decision = {
            "record_id": record["decision_id"],
            "outcome": record["disposition"],
            "delta_e": _number_wire(focus.get("target_delta_e76"), delta_provenance),
            "threshold": _number_wire(focus.get("threshold_delta_e76"), threshold_provenance),
            "limit_codes": record["limit_codes"],
            "evidence_refs": normalized_refs,
            "explanation": explanation,
        }
        target_lab = _lab(focus.get("median_lab"), "WP7 target Lab", nullable=True)
        reference_lab = _lab(
            focus.get("reference_center_lab"), "WP7 reference center Lab", nullable=True
        )
        vials.append({
            "slot": slot, "vial_id": vial_id, "recipe_id": recipe_id,
            "rgb_frame": rgb_asset, "predicted_mask": mask_asset,
            "target_lab": _lab_wire(target_lab, target_provenance),
            "reference_lab": _lab_wire(reference_lab, reference_provenance),
            "decision": decision, "history_refs": [],
        })
        if recipe_id not in recipes:
            red, green, blue = _rgb_from_lab(target_lab)
            recipes[recipe_id] = {
                "recipe_id": recipe_id, "label": f"Recipe {recipe_id}",
                "rgb": {
                    "r": red, "g": green, "b": blue,
                    "provenance": _provenance(
                        "inferred", str(focus["measurement_id"]),
                        "Display swatch converted from the focus-stage measured Lab value; not a recipe calibration.",
                    ),
                },
            }
        width = min(0.16, 0.8 / len(records))
        x = (slot - 0.5) / len(records)
        geometry = _provenance(
            "inferred", "normalized-ui-layout",
            "Presentation-only normalized layout; not a measured station position.",
        )
        observations.append({
            "vial_id": vial_id, "decision_id": record["decision_id"],
            "position": {"x": x, "y": 0.5, "provenance": geometry},
            "bbox": {
                "x": max(0.0, x - width / 2), "y": 0.4, "width": width,
                "height": 0.2, "provenance": geometry,
            },
        })
        cues.append({
            "vial_id": vial_id, "start_seconds": float(slot - 1),
            "end_seconds": float(slot),
            "provenance": _provenance(
                "inferred", _file_sha(video_source),
                "Deterministic presentation cue; not measured process timing.",
            ),
        })
    source_label = "Development data" if mode == "development" else "Final WP7 data"
    source_origin = "synthetic_development" if mode == "development" else "wp7_final"
    video.update({
        "media_type": "video/mp4",
        "duration_seconds": {
            "value": float(len(records)),
            "provenance": _provenance(
                "inferred", _file_sha(video_source),
                "Deterministic UI presentation duration; not plant timing.",
            ),
        },
    })
    return {
        "schema_version": "2.0.0",
        "dataset_id": f"{mode}-wp7-{_file_sha(_discover_unique(source, 'evidence.sqlite3'))[:12]}",
        "label": source_label,
        "data_origin": source_origin,
        "source": {"mode": mode, "origin": source_origin, "label": source_label},
        "run": {
            "run_id": str(selected_run_id),
            "title": f"{source_label} operator run", "video": video, "cue_times": cues,
        },
        "recipes": list(recipes.values()), "history": [],
        "top_down": {"state": "waiting", "asset": None, "observations": observations},
        "vials": vials, "evidence": evidence,
        "wp6": {
            "disclaimer": "WP6 benchmark unavailable for this source", "rows": [],
            "inn67": {
                "status": "unavailable", "label": "INN-67 unavailable",
                "note": "No WP6 or INN-67 benchmark is attributed to this selected WP7 source.",
            },
        },
    }


def _mock_wire(repository_root: Path) -> dict[str, Any]:
    source = _load_json(repository_root / "ui-data" / "mock-run.json", "Phase 1 mock run")
    _require(isinstance(source, dict), "Phase 1 mock run must be an object")
    data = copy.deepcopy(source)
    data["schema_version"] = "2.0.0"
    data["source"] = {
        "mode": "mock", "origin": "synthetic_twin", "label": "Synthetic twin data"
    }
    return data


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _schema_pointer(root: dict[str, Any], reference: str) -> dict[str, Any]:
    _require(reference.startswith("#/"), f"unsupported schema reference {reference!r}")
    value: Any = root
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        _require(isinstance(value, dict) and token in value,
                 f"unresolved schema reference {reference!r}")
        value = value[token]
    _require(isinstance(value, dict), f"schema reference {reference!r} is not an object")
    return value


def _schema_type_matches(instance: Any, expected: str) -> bool:
    return {
        "object": lambda: isinstance(instance, dict),
        "array": lambda: isinstance(instance, list),
        "string": lambda: isinstance(instance, str),
        "number": lambda: (
            not isinstance(instance, bool)
            and isinstance(instance, (int, float))
            and math.isfinite(float(instance))
        ),
        "integer": lambda: not isinstance(instance, bool) and isinstance(instance, int),
        "boolean": lambda: isinstance(instance, bool),
        "null": lambda: instance is None,
    }.get(expected, lambda: False)()


def _validate_schema_value(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate_schema_value(instance, _schema_pointer(root, schema["$ref"]), root, path)
        return
    if "anyOf" in schema:
        failures: list[str] = []
        for option in schema["anyOf"]:
            try:
                _validate_schema_value(instance, option, root, path)
                break
            except MaterializationError as error:
                failures.append(str(error))
        else:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: no anyOf option matched "
                f"({'; '.join(failures)})"
            )
        return
    if "const" in schema and instance != schema["const"]:
        raise MaterializationError(
            f"normalized schema validation failed at {path}: expected constant {schema['const']!r}"
        )
    if "enum" in schema and instance not in schema["enum"]:
        raise MaterializationError(
            f"normalized schema validation failed at {path}: value is not in enum"
        )
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_schema_type_matches(instance, value) for value in allowed):
            raise MaterializationError(
                f"normalized schema validation failed at {path}: expected type {allowed}"
            )
    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [name for name in required if name not in instance]
        if missing:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: missing required "
                + ", ".join(missing)
            )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = set(instance) - set(properties)
            if extras:
                raise MaterializationError(
                    f"normalized schema validation failed at {path}: unexpected properties "
                    + ", ".join(sorted(extras))
                )
        for name, value in instance.items():
            if name in properties:
                _validate_schema_value(value, properties[name], root, f"{path}/{name}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise MaterializationError(
                f"normalized schema validation failed at {path}: too few items"
            )
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: too many items"
            )
        if schema.get("uniqueItems"):
            encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in instance]
            if len(encoded) != len(set(encoded)):
                raise MaterializationError(
                    f"normalized schema validation failed at {path}: items are not unique"
                )
        if "items" in schema:
            for index, value in enumerate(instance):
                _validate_schema_value(value, schema["items"], root, f"{path}/{index}")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise MaterializationError(
                f"normalized schema validation failed at {path}: string is too short"
            )
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: pattern mismatch"
            )
    if not isinstance(instance, bool) and isinstance(instance, (int, float)):
        if "minimum" in schema and instance < schema["minimum"]:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: below minimum"
            )
        if "maximum" in schema and instance > schema["maximum"]:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: above maximum"
            )
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise MaterializationError(
                f"normalized schema validation failed at {path}: not above exclusive minimum"
            )


def _validate_normalized_schema(wire: dict[str, Any], repository_root: Path) -> None:
    schema_path = repository_root / "ui-data" / "schema.json"
    if not schema_path.is_file():
        schema_path = Path(__file__).parents[1] / "ui-data" / "schema.json"
    schema = _load_json(schema_path, "normalized UI schema")
    _require(isinstance(schema, dict), "normalized UI schema must be an object")
    _validate_schema_value(wire, schema, schema, "/")


def materialize(
    *, repository_root: Path, config_path: Path, output_dir: Path
) -> Path:
    """Materialize the configured source and return the active-run path."""

    repository_root = repository_root.resolve()
    output_dir = output_dir.resolve()
    mode, source = resolve_source(config_path, repository_root=repository_root)
    if mode == "mock":
        wire = _mock_wire(repository_root)
    else:
        assert source is not None
        database = _discover_unique(source, "evidence.sqlite3")
        records = _decision_records(source)
        connection = _read_connection(database)
        try:
            wire = _final_wire(source, records, connection, output_dir, mode=mode)
        finally:
            connection.close()
    destination = output_dir / "active-run.json"
    _validate_normalized_schema(wire, repository_root)
    _atomic_json(wire, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = materialize(
        repository_root=root,
        config_path=args.config or root / "ui-data" / "source.json",
        output_dir=args.output or root / "ui-data",
    )
    print(destination)


if __name__ == "__main__":
    main()

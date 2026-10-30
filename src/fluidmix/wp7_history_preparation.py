"""Preparation-only WP7 population from accepted synthetic development evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from .contracts._common import bounded_control_free_text, exact_lower_sha256
from .evidence_store import EvidenceStore, HistoryQuery, assert_history_citation


ACCEPTED_SOURCE_ROOT = Path("/home/hackathon5/liamreporoot")
ACCEPTED_RECEIPT = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/"
    "wp5-cnn-swap-sixth-dev-blocker-fix-ff0fab6-01a18128-attempt01/"
    "population-receipt.json"
)
ACCEPTED_DATABASE = ACCEPTED_RECEIPT.with_name("evidence.sqlite3")
ACCEPTED_REVIEW = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/wp5-cnn-swap-review-01a18128/"
    "independent-rereview-ff0fab6.md"
)
ACCEPTED_RECEIPT_SHA256 = (
    "836001f775d20e4835ed0cdee4f9197992b7a79def75ae1e06c9863c8e207810"
)
ACCEPTED_DATABASE_SHA256 = (
    "b83deef57e2c17d4b8cc068440c763729f9ebe3a1c078711516f57bc732dc793"
)
ACCEPTED_REVIEW_SHA256 = (
    "0c7f843764179e0533d151320cca461ba9d5cff3b6b96965493754704dbdb63a"
)
ACCEPTED_ORDERED_VIDEO = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/"
    "wp7-ordered-test-video.mp4"
)
ACCEPTED_ORDERED_VIDEO_SHA256 = (
    "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b"
)
ACCEPTED_COMPOSITION_RECEIPT = ACCEPTED_ORDERED_VIDEO.with_suffix(".json")
ACCEPTED_COMPOSITION_RECEIPT_SHA256 = (
    "0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037"
)
ACCEPTED_ORDERED_VIDEO_REVIEW = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/"
    "wp7-ordered-test-video-final-review-01a18128-attempt01/FINAL-REVIEW.json"
)
ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256 = (
    "ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683"
)
_BASE_LIMITS = (
    "frozen_decision_config_unavailable",
    "matched_peer_batches_unavailable",
)
_MODEL_REVISION = "deterministic-template:wp7-development-history-preparation-v1"


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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")


def _strict_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not readable strict JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{role} must be a JSON object")
    return value


def _validate_exact_source(
    *,
    source_receipt: Path,
    expected_receipt_sha256: str,
    source_database: Path,
    expected_database_sha256: str,
    independent_review: Path,
    expected_review_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    expected = (
        (Path(source_receipt), ACCEPTED_RECEIPT, expected_receipt_sha256, ACCEPTED_RECEIPT_SHA256, "receipt"),
        (Path(source_database), ACCEPTED_DATABASE, expected_database_sha256, ACCEPTED_DATABASE_SHA256, "database"),
        (Path(independent_review), ACCEPTED_REVIEW, expected_review_sha256, ACCEPTED_REVIEW_SHA256, "review"),
    )
    for supplied_path, accepted_path, supplied_hash, accepted_hash, role in expected:
        exact_lower_sha256(supplied_hash, f"expected {role} SHA-256")
        if supplied_path.resolve() != accepted_path.resolve():
            if role == "database":
                raise ValueError("source_database is not the exact current source database")
            raise ValueError(f"{role} path is not the exact accepted {role}")
        if supplied_hash != accepted_hash:
            raise ValueError(f"expected {role} SHA-256 is not the accepted {role} SHA-256")
        if _file_hash(supplied_path) != accepted_hash:
            raise ValueError(f"accepted {role} SHA-256 mismatch")

    receipt = _strict_json(Path(source_receipt), "accepted source receipt")
    database = receipt.get("database")
    if not isinstance(database, Mapping) or (
        database.get("path") != "evidence.sqlite3"
        or database.get("sha256") != ACCEPTED_DATABASE_SHA256
        or database.get("sqlite_user_version") != 3
    ):
        raise ValueError("accepted receipt does not bind the exact current source database")
    denominators = receipt.get("denominators")
    expected_denominators = {
        "production_runs": 6,
        "production_vials": 18,
        "production_stage_records": 45,
        "train_stage_records": 39,
        "validation_stage_records": 6,
        "predicted_mask_images": 45,
        "rgb_measurements": 45,
        "fill_clips": 18,
    }
    if not isinstance(denominators, Mapping) or any(
        denominators.get(key) != count for key, count in expected_denominators.items()
    ):
        raise ValueError("accepted receipt denominators drifted")
    locator_binding = receipt.get("combined_prediction_locators")
    if not isinstance(locator_binding, Mapping):
        raise ValueError("accepted receipt lacks prediction-locator binding")
    locator_path = Path(str(locator_binding.get("path")))
    locator_sha = locator_binding.get("sha256")
    if not isinstance(locator_sha, str) or _file_hash(locator_path) != locator_sha:
        raise ValueError("accepted prediction-locator SHA-256 mismatch")
    locator_receipt = _strict_json(locator_path, "accepted prediction locators")
    if (
        locator_receipt.get("data_origin") != "twin"
        or locator_receipt.get("mask_source") != "rgb_only_prediction"
        or not isinstance(locator_receipt.get("images"), dict)
        or len(locator_receipt["images"]) != 45
    ):
        raise ValueError("accepted prediction locators are not exact twin RGB-only predictions")
    return receipt, dict(locator_receipt["images"])


def _source_counts(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("source SQLite integrity check failed")
        values = {
            "run_count": connection.execute("SELECT count(*) FROM runs").fetchone()[0],
            "vial_count": connection.execute("SELECT count(*) FROM vials").fetchone()[0],
            "stage_record_count": connection.execute("SELECT count(*) FROM stages").fetchone()[0],
            "train_stage_record_count": connection.execute(
                "SELECT count(*) FROM stages WHERE split_role = 'train'"
            ).fetchone()[0],
            "validation_stage_record_count": connection.execute(
                "SELECT count(*) FROM stages WHERE split_role = 'validation'"
            ).fetchone()[0],
            "decision_row_count": connection.execute("SELECT count(*) FROM decisions").fetchone()[0],
            "explanation_row_count": connection.execute("SELECT count(*) FROM explanations").fetchone()[0],
            "sim_metadata_row_count": connection.execute("SELECT count(*) FROM sim_metadata").fetchone()[0],
        }
        forbidden_roles = connection.execute(
            "SELECT count(*) FROM stages WHERE split_role IN ('oracle','evaluation','test')"
        ).fetchone()[0]
        non_predicted = connection.execute(
            "SELECT count(*) FROM images WHERE mask_source <> 'predicted'"
        ).fetchone()[0]
    if values != {
        "run_count": 6,
        "vial_count": 18,
        "stage_record_count": 45,
        "train_stage_record_count": 39,
        "validation_stage_record_count": 6,
        "decision_row_count": 0,
        "explanation_row_count": 0,
        "sim_metadata_row_count": 0,
    }:
        raise ValueError("source SQLite denominators drifted")
    if forbidden_roles or non_predicted:
        raise ValueError("source SQLite contains forbidden roles or non-predicted masks")
    return values


def _validate_ordered_video(
    *,
    ordered_video: Path,
    expected_ordered_video_sha256: str,
    composition_receipt: Path,
    expected_composition_receipt_sha256: str,
    ordered_video_review: Path,
    expected_ordered_video_review_sha256: str,
) -> dict[str, object]:
    exact = (
        (ordered_video, ACCEPTED_ORDERED_VIDEO, expected_ordered_video_sha256, ACCEPTED_ORDERED_VIDEO_SHA256, "ordered video"),
        (composition_receipt, ACCEPTED_COMPOSITION_RECEIPT, expected_composition_receipt_sha256, ACCEPTED_COMPOSITION_RECEIPT_SHA256, "composition receipt"),
        (ordered_video_review, ACCEPTED_ORDERED_VIDEO_REVIEW, expected_ordered_video_review_sha256, ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256, "ordered-video review"),
    )
    for supplied, accepted, expected_hash, accepted_hash, role in exact:
        exact_lower_sha256(expected_hash, f"expected {role} SHA-256")
        if Path(supplied).resolve() != accepted.resolve():
            raise ValueError(f"{role} path is not the exact accepted artifact")
        if expected_hash != accepted_hash:
            raise ValueError(f"expected {role} SHA-256 is stale")
        if _file_hash(Path(supplied)) != accepted_hash:
            raise ValueError(f"accepted {role} SHA-256 mismatch")

    composition = _strict_json(Path(composition_receipt), "composition receipt")
    output = composition.get("output")
    runs = composition.get("ordered_runs")
    verification = composition.get("verification")
    if (
        composition.get("artifact_type") != "wp7_ordered_test_video_v1"
        or composition.get("status") != "pass"
        or not isinstance(output, Mapping)
        or output.get("path") != ACCEPTED_ORDERED_VIDEO.name
        or output.get("sha256") != ACCEPTED_ORDERED_VIDEO_SHA256
        or output.get("frame_count") != 7322
        or output.get("duration_seconds") != "732.2"
        or not isinstance(verification, Mapping)
        or verification.get("case_count") != 18
        or not isinstance(runs, list)
        or len(runs) != 2
    ):
        raise ValueError("composition receipt ordered-video contract drifted")
    case_ids = [case for run in runs for case in run.get("case_ids", [])]
    if len(case_ids) != 18 or len(set(case_ids)) != 18:
        raise ValueError("composition receipt does not bind 18 unique cases")

    review = _strict_json(Path(ordered_video_review), "ordered-video final review")
    identity = review.get("identity")
    reviewed_media = review.get("output_media")
    case_contract = review.get("case_contract")
    if (
        review.get("status") != "pass"
        or review.get("verdict") != "pass"
        or not isinstance(identity, Mapping)
        or identity.get("output_mp4_sha256") != ACCEPTED_ORDERED_VIDEO_SHA256
        or identity.get("output_receipt_sha256") != ACCEPTED_COMPOSITION_RECEIPT_SHA256
        or not isinstance(reviewed_media, Mapping)
        or reviewed_media.get("sha256") != ACCEPTED_ORDERED_VIDEO_SHA256
        or reviewed_media.get("frame_count") != 7322
        or reviewed_media.get("duration_seconds") != "732.2"
        or not isinstance(case_contract, Mapping)
        or case_contract.get("count") != 18
        or case_contract.get("unique") is not True
    ):
        raise ValueError("ordered-video final review contract drifted")
    return {
        "video_path": str(ACCEPTED_ORDERED_VIDEO),
        "video_sha256": ACCEPTED_ORDERED_VIDEO_SHA256,
        "composition_receipt_path": str(ACCEPTED_COMPOSITION_RECEIPT),
        "composition_receipt_sha256": ACCEPTED_COMPOSITION_RECEIPT_SHA256,
        "final_review_path": str(ACCEPTED_ORDERED_VIDEO_REVIEW),
        "final_review_sha256": ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256,
        "case_count": 18,
        "frame_count": 7322,
        "duration_seconds": "732.2",
    }


def _group_vials(
    records: list[dict[str, object]], cursors: Mapping[str, int]
) -> list[list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(
            (str(record["run_id"]), int(record["vial_ordinal"])), []
        ).append(record)
    return [
        sorted(rows, key=lambda record: cursors[str(record["evidence_id"])])
        for _, rows in sorted(grouped.items())
    ]


def _derive_vial(
    measured: list[dict[str, object]], query: HistoryQuery
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    first = measured[0]
    limits = list(_BASE_LIMITS)
    if not query.members:
        limits.append("prior_reference_unavailable")
    limits.sort()
    history_refs = [
        {"evidence_id": member.evidence_id, "query_sha256": query.query_sha256}
        for member in sorted(query.members, key=lambda item: item.evidence_id)
    ]
    for member in query.members:
        assert_history_citation(query, member.evidence_id, member.image_id)
    identity = {
        "run_id": first["run_id"],
        "vial_ordinal": first["vial_ordinal"],
        "query_sha256": query.query_sha256,
        "limit_codes": limits,
    }
    token = _canonical_hash(identity)
    decision_id = f"DEC-WP7-{token[:24]}"
    explanation_id = f"EXP-WP7-{token[24:48]}"
    prompt = {
        "schema_version": "wp7-development-history-template-v1",
        "status": "UNKNOWN",
        "action": "abstain_and_collect_evidence",
        "limit_codes": limits,
    }
    guard_result = (
        "UNKNOWN; abstain; collect evidence; limit_codes=" + ",".join(limits)
    )
    explained: list[dict[str, object]] = []
    for predecessor in measured:
        record = deepcopy(predecessor)
        record.update(
            {
                "decision_id": decision_id,
                "disposition": "insufficient_evidence",
                "limit_codes": limits,
                "history_refs": history_refs,
                "explanation_id": explanation_id,
                "model_revision": _MODEL_REVISION,
                "prompt_sha256": _canonical_hash(prompt),
                "image_ids_sent": [],
                "guard_result": "fallback",
            }
        )
        explained.append(record)
    query_receipt = {
        "run_id": first["run_id"],
        "vial_ordinal": first["vial_ordinal"],
        "target_evidence_id": first["evidence_id"],
        "query_sha256": query.query_sha256,
        "query_kind": "vial_history",
        "filters": query.filters,
        "members": [
            {
                "evidence_id": member.evidence_id,
                "image_id": member.image_id,
                "event_cursor": member.event_cursor,
                "split_role": member.split_role,
            }
            for member in query.members
        ],
    }
    explanation = {
        "decision_id": decision_id,
        "explanation_id": explanation_id,
        "run_id": first["run_id"],
        "vial_ordinal": first["vial_ordinal"],
        "disposition": "insufficient_evidence",
        "status": "UNKNOWN",
        "action": "abstain_and_collect_evidence",
        "limit_codes": limits,
        "history_refs": history_refs,
        "model_revision": _MODEL_REVISION,
        "prompt_sha256": _canonical_hash(prompt),
        "image_ids_sent": [],
        "guard_result": guard_result,
    }
    return explained, query_receipt, explanation


def build_development_history_population(
    *,
    source_receipt: Path,
    expected_receipt_sha256: str,
    source_database: Path,
    expected_database_sha256: str,
    independent_review: Path,
    expected_review_sha256: str,
    owner_id: str,
    destination: Path,
) -> dict[str, object]:
    """Build one fresh derived synthetic population without opening one-shot inputs."""

    bounded_control_free_text(owner_id, "owner_id")
    _, accepted_locators = _validate_exact_source(
        source_receipt=source_receipt,
        expected_receipt_sha256=expected_receipt_sha256,
        source_database=source_database,
        expected_database_sha256=expected_database_sha256,
        independent_review=independent_review,
        expected_review_sha256=expected_review_sha256,
    )
    _source_counts(Path(source_database))
    source_hashes_before = {
        Path(source_receipt): _file_hash(Path(source_receipt)),
        Path(source_database): _file_hash(Path(source_database)),
        Path(independent_review): _file_hash(Path(independent_review)),
    }
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    copied_database = root / "evidence.sqlite3"
    shutil.copyfile(source_database, copied_database)
    if _file_hash(copied_database) != ACCEPTED_DATABASE_SHA256:
        raise ValueError("copied source database SHA-256 mismatch")
    if any(_file_hash(path) != digest for path, digest in source_hashes_before.items()):
        raise ValueError("accepted source tuple changed during copy")

    store = EvidenceStore(copied_database, artifact_root=ACCEPTED_SOURCE_ROOT)
    measured, locators, cursors = store.measured_population()
    if locators != accepted_locators:
        raise ValueError("stored normalized locators do not match accepted locator receipt")
    vials = _group_vials(measured, cursors)
    if len(vials) != 18 or len(measured) != 45:
        raise ValueError("reconstructed source population denominators drifted")

    explained_records: list[dict[str, object]] = []
    query_receipts: list[dict[str, object]] = []
    explanations: list[dict[str, object]] = []
    queries: list[HistoryQuery] = []
    for vial in vials:
        first = vial[0]
        query = store.vial_history(
            str(first["run_id"]),
            before=cursors[str(first["evidence_id"])],
            allowed_roles={"train"},
            target_evidence_id=str(first["evidence_id"]),
        )
        queries.append(query)
        explained, query_receipt, explanation = _derive_vial(vial, query)
        explained_records.extend(explained)
        query_receipts.append(query_receipt)
        explanations.append(explanation)
    result = store.append_explained(
        explained_records, locators, history_queries=queries
    )
    if result.inserted != 45 or result.existing:
        raise ValueError("derived decision/explanation append count drifted")

    with sqlite3.connect(copied_database) as connection:
        post_counts = {
            "decision_row_count": connection.execute("SELECT count(*) FROM decisions").fetchone()[0],
            "explanation_row_count": connection.execute("SELECT count(*) FROM explanations").fetchone()[0],
            "store_history_query_row_count": connection.execute("SELECT count(*) FROM history_queries").fetchone()[0],
            "sim_metadata_row_count": connection.execute("SELECT count(*) FROM sim_metadata").fetchone()[0],
            "forbidden_role_count": connection.execute(
                "SELECT count(*) FROM stages WHERE split_role IN ('oracle','evaluation','test')"
            ).fetchone()[0],
            "non_predicted_mask_count": connection.execute(
                "SELECT count(*) FROM images WHERE mask_source <> 'predicted'"
            ).fetchone()[0],
        }
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if (
        integrity != "ok"
        or post_counts["decision_row_count"] != 45
        or post_counts["explanation_row_count"] != 45
        or post_counts["sim_metadata_row_count"]
        or post_counts["forbidden_role_count"]
        or post_counts["non_predicted_mask_count"]
    ):
        raise ValueError("derived store verification failed")

    population_path = root / "population-records.json"
    query_path = root / "query-receipts.json"
    explanation_path = root / "explanation-records.json"
    _publish_json(population_path, explained_records)
    _publish_json(query_path, query_receipts)
    _publish_json(explanation_path, explanations)
    primary_paths = (copied_database, population_path, query_path, explanation_path)
    primary_inventory = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_hash(path)}
        for path in primary_paths
    ]
    limit_counts = Counter(
        code for explanation in explanations for code in explanation["limit_codes"]
    )
    receipt: dict[str, object] = {
        "schema_version": "wp7-development-history-population-v1",
        "artifact_type": "synthetic_development_history_preparation",
        "owner_id": owner_id,
        "source_provenance": "synthetic_twin_development_only",
        "route_approved": False,
        "one_shot_inputs_opened": False,
        "source_tuple": {
            "receipt_path": str(ACCEPTED_RECEIPT),
            "receipt_sha256": ACCEPTED_RECEIPT_SHA256,
            "database_path": str(ACCEPTED_DATABASE),
            "database_sha256": ACCEPTED_DATABASE_SHA256,
            "independent_review_path": str(ACCEPTED_REVIEW),
            "independent_review_sha256": ACCEPTED_REVIEW_SHA256,
        },
        "denominators": {
            "run_count": 6,
            "vial_count": 18,
            "stage_record_count": 45,
            "train_stage_record_count": 39,
            "validation_stage_record_count": 6,
            "decision_row_count": 45,
            "explanation_row_count": 45,
            "unique_vial_decision_count": 18,
            "insufficient_evidence_vial_count": 18,
        },
        "limit_code_vial_counts": dict(sorted(limit_counts.items())),
        "population_query_sha256s": [query.query_sha256 for query in queries],
        "population_query_count": 18,
        "store_history_query_row_count": post_counts["store_history_query_row_count"],
        "output_inventory": primary_inventory,
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    receipt_path = root / "population-receipt.json"
    _publish_json(receipt_path, receipt)
    inventory = {
        "schema_version": "wp7-development-history-sha-inventory-v1",
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_hash(path)}
            for path in (*primary_paths, receipt_path)
        ],
    }
    inventory["inventory_sha256"] = _canonical_hash(inventory)
    _publish_json(root / "sha-inventory.json", inventory)
    if any(_file_hash(path) != digest for path, digest in source_hashes_before.items()):
        raise ValueError("accepted source tuple changed during population build")
    return receipt


def publish_preparation_preflight(
    *,
    source_receipt: Path,
    expected_receipt_sha256: str,
    source_database: Path,
    expected_database_sha256: str,
    independent_review: Path,
    expected_review_sha256: str,
    ordered_video: Path,
    expected_ordered_video_sha256: str,
    composition_receipt: Path,
    expected_composition_receipt_sha256: str,
    ordered_video_review: Path,
    expected_ordered_video_review_sha256: str,
    population_receipt: Path,
    expected_population_receipt_sha256: str,
    owner_id: str,
    destination: Path,
) -> dict[str, object]:
    """Publish a sealed preparation receipt that execution APIs must reject."""

    bounded_control_free_text(owner_id, "owner_id")
    _validate_exact_source(
        source_receipt=source_receipt,
        expected_receipt_sha256=expected_receipt_sha256,
        source_database=source_database,
        expected_database_sha256=expected_database_sha256,
        independent_review=independent_review,
        expected_review_sha256=expected_review_sha256,
    )
    ordered_video_binding = _validate_ordered_video(
        ordered_video=ordered_video,
        expected_ordered_video_sha256=expected_ordered_video_sha256,
        composition_receipt=composition_receipt,
        expected_composition_receipt_sha256=expected_composition_receipt_sha256,
        ordered_video_review=ordered_video_review,
        expected_ordered_video_review_sha256=expected_ordered_video_review_sha256,
    )
    exact_lower_sha256(
        expected_population_receipt_sha256,
        "expected population receipt SHA-256",
    )
    population_path = Path(population_receipt)
    if _file_hash(population_path) != expected_population_receipt_sha256:
        raise ValueError("derived population receipt SHA-256 mismatch")
    population = _strict_json(population_path, "derived population receipt")
    embedded = population.get("receipt_sha256")
    canonical = _canonical_hash(
        {key: value for key, value in population.items() if key != "receipt_sha256"}
    )
    if embedded != canonical:
        raise ValueError("derived population canonical receipt hash mismatch")
    exact_population_fields = {
        "schema_version",
        "artifact_type",
        "owner_id",
        "source_provenance",
        "route_approved",
        "one_shot_inputs_opened",
        "source_tuple",
        "denominators",
        "limit_code_vial_counts",
        "population_query_sha256s",
        "population_query_count",
        "store_history_query_row_count",
        "output_inventory",
        "receipt_sha256",
    }
    expected_denominators = {
        "run_count": 6,
        "vial_count": 18,
        "stage_record_count": 45,
        "train_stage_record_count": 39,
        "validation_stage_record_count": 6,
        "decision_row_count": 45,
        "explanation_row_count": 45,
        "unique_vial_decision_count": 18,
        "insufficient_evidence_vial_count": 18,
    }
    query_hashes = population.get("population_query_sha256s")
    inventory = population.get("output_inventory")
    if (
        set(population) != exact_population_fields
        or population.get("schema_version") != "wp7-development-history-population-v1"
        or population.get("artifact_type") != "synthetic_development_history_preparation"
        or population.get("owner_id") != owner_id
        or population.get("source_provenance") != "synthetic_twin_development_only"
        or population.get("route_approved") is not False
        or population.get("one_shot_inputs_opened") is not False
        or population.get("source_tuple")
        != {
            "receipt_path": str(ACCEPTED_RECEIPT),
            "receipt_sha256": ACCEPTED_RECEIPT_SHA256,
            "database_path": str(ACCEPTED_DATABASE),
            "database_sha256": ACCEPTED_DATABASE_SHA256,
            "independent_review_path": str(ACCEPTED_REVIEW),
            "independent_review_sha256": ACCEPTED_REVIEW_SHA256,
        }
        or population.get("denominators") != expected_denominators
        or population.get("limit_code_vial_counts")
        != {
            "frozen_decision_config_unavailable": 18,
            "matched_peer_batches_unavailable": 18,
            "prior_reference_unavailable": 8,
        }
        or not isinstance(query_hashes, list)
        or len(query_hashes) != 18
        or len(set(query_hashes)) != 18
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or set(value) - set("0123456789abcdef")
            for value in query_hashes
        )
        or population.get("population_query_count") != 18
        or type(population.get("store_history_query_row_count")) is not int
        or population["store_history_query_row_count"] < 18
        or not isinstance(inventory, list)
        or len(inventory) != 4
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or set(row["sha256"]) - set("0123456789abcdef")
            for row in inventory
        )
        or {row.get("path") for row in inventory if isinstance(row, Mapping)}
        != {
            "evidence.sqlite3",
            "population-records.json",
            "query-receipts.json",
            "explanation-records.json",
        }
    ):
        raise ValueError("derived population receipt is not the exact sealed population")
    preflight: dict[str, object] = {
        "schema_version": "wp7-development-history-preparation-preflight-v1",
        "artifact_type": "sealed_synthetic_development_history_preflight",
        "owner_id": owner_id,
        "source_provenance": "synthetic_twin_development_only",
        "source_tuple": population["source_tuple"],
        "ordered_video": ordered_video_binding,
        "population_receipt_sha256": expected_population_receipt_sha256,
        "population_canonical_sha256": embedded,
        "denominators": {
            "vials": 18,
            "insufficient_evidence": {"count": 18, "total": 18},
            "limit_code_vial_counts": population["limit_code_vial_counts"],
        },
        "one_shot_inputs_opened": False,
        "access_status": "sealed_pending_liam_decision",
        "route_decision": None,
        "history_backed_explanations_enabled": False,
        "route_status": "option3_pending_accepted_inn67_reference_bank_and_config",
        "fallback": {
            "trigger": "accepted INN-67 reference bank and config unavailable after approximately 18:00 Saturday America/Chicago",
            "route": "option1_all_insufficient_evidence",
            "option2_approved": False,
        },
    }
    preflight["preflight_sha256"] = _canonical_hash(preflight)
    _publish_json(Path(destination), preflight)
    return preflight


__all__ = [
    "ACCEPTED_DATABASE",
    "ACCEPTED_DATABASE_SHA256",
    "ACCEPTED_RECEIPT",
    "ACCEPTED_RECEIPT_SHA256",
    "ACCEPTED_REVIEW",
    "ACCEPTED_REVIEW_SHA256",
    "ACCEPTED_ORDERED_VIDEO",
    "ACCEPTED_ORDERED_VIDEO_SHA256",
    "ACCEPTED_COMPOSITION_RECEIPT",
    "ACCEPTED_COMPOSITION_RECEIPT_SHA256",
    "ACCEPTED_ORDERED_VIDEO_REVIEW",
    "ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256",
    "build_development_history_population",
    "publish_preparation_preflight",
]

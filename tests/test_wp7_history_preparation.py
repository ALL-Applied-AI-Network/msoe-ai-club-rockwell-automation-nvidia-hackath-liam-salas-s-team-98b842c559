"""Preparation-only WP7 synthetic development-history boundaries."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE_ROOT = Path("/home/hackathon5/liamreporoot")
SOURCE_BUNDLE = SOURCE_ROOT / "outputs/pipeline-20261028/wp5-cnn-swap-sixth-dev-blocker-fix-ff0fab6-01a18128-attempt01"
SOURCE_RECEIPT = SOURCE_BUNDLE / "population-receipt.json"
SOURCE_DATABASE = SOURCE_BUNDLE / "evidence.sqlite3"
SOURCE_REVIEW = SOURCE_ROOT / "outputs/pipeline-20261028/wp5-cnn-swap-review-01a18128/independent-rereview-ff0fab6.md"
RECEIPT_SHA = "836001f775d20e4835ed0cdee4f9197992b7a79def75ae1e06c9863c8e207810"
DATABASE_SHA = "b83deef57e2c17d4b8cc068440c763729f9ebe3a1c078711516f57bc732dc793"
REVIEW_SHA = "0c7f843764179e0533d151320cca461ba9d5cff3b6b96965493754704dbdb63a"
ORDERED_VIDEO = SOURCE_ROOT / "outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/wp7-ordered-test-video.mp4"
ORDERED_VIDEO_SHA = "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b"
COMPOSITION_RECEIPT = ORDERED_VIDEO.with_suffix(".json")
COMPOSITION_RECEIPT_SHA = "0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037"
ORDERED_VIDEO_REVIEW = SOURCE_ROOT / "outputs/pipeline-20261028/wp7-ordered-test-video-final-review-01a18128-attempt01/FINAL-REVIEW.json"
ORDERED_VIDEO_REVIEW_SHA = "ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _build(destination: Path) -> dict[str, object]:
    from fluidmix.wp7_history_preparation import build_development_history_population

    return build_development_history_population(
        source_receipt=SOURCE_RECEIPT,
        expected_receipt_sha256=RECEIPT_SHA,
        source_database=SOURCE_DATABASE,
        expected_database_sha256=DATABASE_SHA,
        independent_review=SOURCE_REVIEW,
        expected_review_sha256=REVIEW_SHA,
        owner_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        destination=destination,
    )


@pytest.mark.skipif(not SOURCE_DATABASE.exists(), reason="accepted synthetic tuple unavailable")
def test_exact_build_is_deterministic_append_only_and_no_clobber(tmp_path: Path) -> None:
    before = (_sha(SOURCE_RECEIPT), _sha(SOURCE_DATABASE), _sha(SOURCE_REVIEW))
    first = _build(tmp_path / "one")
    second = _build(tmp_path / "two")

    assert first == second
    assert first["denominators"] == {
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
    assert len(first["population_query_sha256s"]) == 18
    assert first["store_history_query_row_count"] >= 18
    assert first["source_provenance"] == "synthetic_twin_development_only"
    assert first["route_approved"] is False
    assert first["limit_code_vial_counts"] == {
        "frozen_decision_config_unavailable": 18,
        "matched_peer_batches_unavailable": 18,
        "prior_reference_unavailable": 8,
    }
    assert (_sha(SOURCE_RECEIPT), _sha(SOURCE_DATABASE), _sha(SOURCE_REVIEW)) == before
    for root in (tmp_path / "one", tmp_path / "two"):
        with sqlite3.connect(root / "evidence.sqlite3") as connection:
            assert connection.execute("SELECT count(*) FROM stages").fetchone()[0] == 45
            assert connection.execute("SELECT count(*) FROM decisions").fetchone()[0] == 45
            assert connection.execute("SELECT count(*) FROM explanations").fetchone()[0] == 45
            assert connection.execute("SELECT count(*) FROM sim_metadata").fetchone()[0] == 0
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        records = json.loads((root / "population-records.json").read_text())
        queries = json.loads((root / "query-receipts.json").read_text())
        assert len(records) == 45
        assert {row["disposition"] for row in records} == {"insufficient_evidence"}
        assert {row["mask_source"] for row in records} == {"predicted"}
        assert all(row["image_ids_sent"] == [] for row in records)
        assert len([query for query in queries if not query["members"]]) == 8
        by_query = {query["query_sha256"]: query for query in queries}
        for row in records:
            for reference in row["history_refs"]:
                assert reference["evidence_id"] in {
                    member["evidence_id"]
                    for member in by_query[reference["query_sha256"]]["members"]
                }
    with pytest.raises(FileExistsError):
        _build(tmp_path / "one")


@pytest.mark.skipif(not SOURCE_DATABASE.exists(), reason="accepted synthetic tuple unavailable")
def test_builder_rejects_stale_tuple_and_nested_base_substitution(tmp_path: Path) -> None:
    from fluidmix.wp7_history_preparation import build_development_history_population

    kwargs = dict(
        source_receipt=SOURCE_RECEIPT,
        expected_receipt_sha256=RECEIPT_SHA,
        source_database=SOURCE_DATABASE,
        expected_database_sha256=DATABASE_SHA,
        independent_review=SOURCE_REVIEW,
        expected_review_sha256=REVIEW_SHA,
        owner_id="owner",
        destination=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="accepted receipt SHA-256"):
        build_development_history_population(**{**kwargs, "expected_receipt_sha256": "0" * 64})
    receipt = json.loads(SOURCE_RECEIPT.read_text())
    with pytest.raises(ValueError, match="exact current source database"):
        build_development_history_population(
            **{
                **kwargs,
                "source_database": Path(receipt["base_tuple"]["database_path"]),
                "expected_database_sha256": receipt["base_tuple"]["database_sha256"],
            }
        )


@pytest.mark.skipif(not SOURCE_DATABASE.exists(), reason="accepted synthetic tuple unavailable")
def test_store_rejects_predecessor_drift_and_decision_collision(tmp_path: Path) -> None:
    from fluidmix.evidence_store import EvidenceStore

    _build(tmp_path / "built")
    database = tmp_path / "built/evidence.sqlite3"
    records = json.loads((tmp_path / "built/population-records.json").read_text())
    store = EvidenceStore(database, artifact_root=SOURCE_ROOT)
    measured, locators, cursors = store.measured_population()
    queries = []
    for vial_rows in _group_vials(measured, cursors):
        first = vial_rows[0]
        queries.append(
            store.vial_history(
                first["run_id"],
                before=cursors[first["evidence_id"]],
                allowed_roles={"train"},
                target_evidence_id=first["evidence_id"],
            )
        )
    assert store.append_explained(records, locators, history_queries=queries).existing == 45
    collision = deepcopy(records)
    first_decision = collision[0]["decision_id"]
    for row in collision:
        if row["decision_id"] == first_decision:
            row["guard_result"] = "pass"
    with pytest.raises(ValueError, match="logical collision in explanations"):
        store.append_explained(collision, locators, history_queries=queries)
    decision_collision = deepcopy(records)
    first_decision = decision_collision[0]["decision_id"]
    for row in decision_collision:
        if row["decision_id"] == first_decision:
            row["decision_id"] = "DEC-WP7-COLLISION"
    with pytest.raises(ValueError, match="logical collision in decisions"):
        store.append_explained(decision_collision, locators, history_queries=queries)

    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER stages_append_only_update")
        connection.execute(
            "UPDATE stages SET record_sha256 = ? WHERE evidence_id = ?",
            ("0" * 64, measured[0]["evidence_id"]),
        )
    with pytest.raises(ValueError, match="measured predecessor hash"):
        store.append_explained(records, locators, history_queries=queries)


def _group_vials(records: list[dict[str, object]], cursors: dict[str, int]):
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in records:
        grouped.setdefault((str(row["run_id"]), int(row["vial_ordinal"])), []).append(row)
    return [
        sorted(rows, key=lambda row: cursors[str(row["evidence_id"])])
        for _, rows in sorted(grouped.items())
    ]


@pytest.mark.skipif(not SOURCE_DATABASE.exists(), reason="accepted synthetic tuple unavailable")
def test_preflight_is_sealed_and_rejected_before_one_pass_access(tmp_path: Path, monkeypatch) -> None:
    from fluidmix import carryover_pipeline as pipeline
    from fluidmix.wp7_history_preparation import publish_preparation_preflight

    population = _build(tmp_path / "population")
    population_path = tmp_path / "population/population-receipt.json"
    preflight = publish_preparation_preflight(
        source_receipt=SOURCE_RECEIPT,
        expected_receipt_sha256=RECEIPT_SHA,
        source_database=SOURCE_DATABASE,
        expected_database_sha256=DATABASE_SHA,
        independent_review=SOURCE_REVIEW,
        expected_review_sha256=REVIEW_SHA,
        ordered_video=ORDERED_VIDEO,
        expected_ordered_video_sha256=ORDERED_VIDEO_SHA,
        composition_receipt=COMPOSITION_RECEIPT,
        expected_composition_receipt_sha256=COMPOSITION_RECEIPT_SHA,
        ordered_video_review=ORDERED_VIDEO_REVIEW,
        expected_ordered_video_review_sha256=ORDERED_VIDEO_REVIEW_SHA,
        population_receipt=population_path,
        expected_population_receipt_sha256=_sha(population_path),
        owner_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        destination=tmp_path / "preflight.json",
    )
    assert preflight["one_shot_inputs_opened"] is False
    assert preflight["access_status"] == "sealed_pending_liam_decision"
    assert preflight["route_decision"] is None
    assert preflight["ordered_video"] == {
        "video_path": str(ORDERED_VIDEO),
        "video_sha256": ORDERED_VIDEO_SHA,
        "composition_receipt_path": str(COMPOSITION_RECEIPT),
        "composition_receipt_sha256": COMPOSITION_RECEIPT_SHA,
        "final_review_path": str(ORDERED_VIDEO_REVIEW),
        "final_review_sha256": ORDERED_VIDEO_REVIEW_SHA,
        "case_count": 18,
        "frame_count": 7322,
        "duration_seconds": "732.2",
    }
    assert preflight["fallback"]["route"] == "option1_all_insufficient_evidence"
    assert preflight["fallback"]["option2_approved"] is False
    assert preflight["denominators"]["insufficient_evidence"] == {"count": 18, "total": 18}
    assert preflight["population_receipt_sha256"] == _sha(population_path)
    assert population["receipt_sha256"] == _canonical_hash(
        {key: value for key, value in population.items() if key != "receipt_sha256"}
    )

    touched: list[Path] = []
    monkeypatch.setattr(pipeline, "_read_production_bytes", lambda path, role: touched.append(path))
    with pytest.raises(ValueError, match="selected mode mismatch"):
        pipeline._validate_execution_preflight(
            preflight,
            expected_sha256=preflight["preflight_sha256"],
            selected_mode="history_backed",
        )
    operational_paths: list[Path] = []
    for index, function in enumerate((pipeline.run_test, pipeline.run_real_check)):
        claim = tmp_path / f"claim-{index}.json"
        output = tmp_path / f"output-{index}.json"
        operational_paths.extend((claim, output, Path(f"{output}.journal.jsonl")))
        with pytest.raises(ValueError, match="selected mode mismatch"):
            function(
                video=tmp_path / "never-video",
                recipe_sequence=tmp_path / "never-recipe",
                freeze=tmp_path / "never-freeze",
                expected_freeze_sha256="0" * 64,
                store=tmp_path / "never-store",
                out=output,
                claim=claim,
                owner_id="01a18128-8dc4-7780-ace1-4d816b49d176",
                preflight_receipt=preflight,
                expected_preflight_sha256=preflight["preflight_sha256"],
                selected_mode="history_backed",
            )
    assert touched == []
    assert not any(path.exists() for path in operational_paths)
    with pytest.raises(FileExistsError):
        publish_preparation_preflight(
            source_receipt=SOURCE_RECEIPT,
            expected_receipt_sha256=RECEIPT_SHA,
            source_database=SOURCE_DATABASE,
            expected_database_sha256=DATABASE_SHA,
            independent_review=SOURCE_REVIEW,
            expected_review_sha256=REVIEW_SHA,
            ordered_video=ORDERED_VIDEO,
            expected_ordered_video_sha256=ORDERED_VIDEO_SHA,
            composition_receipt=COMPOSITION_RECEIPT,
            expected_composition_receipt_sha256=COMPOSITION_RECEIPT_SHA,
            ordered_video_review=ORDERED_VIDEO_REVIEW,
            expected_ordered_video_review_sha256=ORDERED_VIDEO_REVIEW_SHA,
            population_receipt=population_path,
            expected_population_receipt_sha256=_sha(population_path),
            owner_id="01a18128-8dc4-7780-ace1-4d816b49d176",
            destination=tmp_path / "preflight.json",
        )


@pytest.mark.parametrize(
    "field",
    (
        "expected_ordered_video_sha256",
        "expected_composition_receipt_sha256",
        "expected_ordered_video_review_sha256",
    ),
)
@pytest.mark.skipif(not SOURCE_DATABASE.exists(), reason="accepted synthetic tuple unavailable")
def test_preflight_rejects_stale_ordered_video_tuple(tmp_path: Path, field: str) -> None:
    from fluidmix.wp7_history_preparation import publish_preparation_preflight

    _build(tmp_path / "population")
    population_path = tmp_path / "population/population-receipt.json"
    kwargs = {
        "source_receipt": SOURCE_RECEIPT,
        "expected_receipt_sha256": RECEIPT_SHA,
        "source_database": SOURCE_DATABASE,
        "expected_database_sha256": DATABASE_SHA,
        "independent_review": SOURCE_REVIEW,
        "expected_review_sha256": REVIEW_SHA,
        "ordered_video": ORDERED_VIDEO,
        "expected_ordered_video_sha256": ORDERED_VIDEO_SHA,
        "composition_receipt": COMPOSITION_RECEIPT,
        "expected_composition_receipt_sha256": COMPOSITION_RECEIPT_SHA,
        "ordered_video_review": ORDERED_VIDEO_REVIEW,
        "expected_ordered_video_review_sha256": ORDERED_VIDEO_REVIEW_SHA,
        "population_receipt": population_path,
        "expected_population_receipt_sha256": _sha(population_path),
        "owner_id": "owner",
        "destination": tmp_path / "preflight.json",
    }
    kwargs[field] = "0" * 64
    with pytest.raises(ValueError, match="stale"):
        publish_preparation_preflight(**kwargs)


@pytest.mark.parametrize(
    ("statement", "parameters"),
    (
        ("UPDATE images SET mask_source = 'oracle' WHERE image_id = ?", ("BATCH_5a4a586e3ec3b085-V1-dispense-1-t200",)),
        ("UPDATE stages SET split_role = 'test' WHERE evidence_id = ?", ("EVID-BATCH_5a4a586e3ec3b085-V1-dispense-1-t200",)),
        ("UPDATE stages SET split_role = 'evaluation' WHERE evidence_id = ?", ("EVID-BATCH_5a4a586e3ec3b085-V1-dispense-1-t200",)),
    ),
)
@pytest.mark.skipif(not SOURCE_DATABASE.exists(), reason="accepted synthetic tuple unavailable")
def test_measured_population_rejects_forbidden_source_drift(
    tmp_path: Path, statement: str, parameters: tuple[str, ...]
) -> None:
    from fluidmix.evidence_store import EvidenceStore

    database = tmp_path / "evidence.sqlite3"
    shutil.copyfile(SOURCE_DATABASE, database)
    table = statement.split()[1]
    with sqlite3.connect(database) as connection:
        connection.execute(f"DROP TRIGGER {table}_append_only_update")
        connection.execute(statement, parameters)
    with pytest.raises((ValueError, sqlite3.DatabaseError), match="predecessor hash|oracle"):
        EvidenceStore(database, artifact_root=SOURCE_ROOT).measured_population()


def test_cli_exposes_preparation_only_arguments() -> None:
    from scripts.prepare_wp7_development_history import build_parser

    help_text = build_parser().format_help().lower()
    assert "run-test" not in help_text
    assert "run-real-check" not in help_text
    assert "video-5" not in help_text
    assert "wp1" not in help_text

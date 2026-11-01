"""Dedicated development-reference store/query contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _store(tmp_path: Path):
    from fluidmix.memory.history.development_reference import (
        DevelopmentReferenceStore,
        _create_minimal_store,
    )

    records = []
    configurations = []
    cursor = 0
    for recipe, stages in (("R-A", ("after-t100", "after-t200")),):
        peers = {}
        for batch, run in (("INN67-DEV-RB-B01", "RUN-A"), ("INN67-DEV-RB-B02", "RUN-B")):
            for ordinal, stage in enumerate(stages, start=1):
                evidence_id = f"E-{batch}-{ordinal}"
                image_id = f"I-{batch}-{ordinal}"
                rgb = tmp_path / "accepted" / "rgb" / f"{image_id}.png"
                mask = tmp_path / "accepted" / "mask" / f"{image_id}.png"
                records.append(
                    {
                        "evidence_id": evidence_id,
                        "event_cursor": cursor,
                        "batch_id": batch,
                        "run_id": run,
                        "case_id": f"CASE-{batch}",
                        "recipe_id": recipe,
                        "stage_id": stage,
                        "stage_ordinal": ordinal,
                        "tank_id": "t100" if ordinal == 1 else "t200",
                        "split_role": "development_reference",
                        "image_id": image_id,
                        "rgb_path": str(rgb),
                        "rgb_sha256": _write(rgb, f"rgb:{evidence_id}".encode()),
                        "mask_path": str(mask),
                        "mask_sha256": _write(mask, f"mask:{evidence_id}".encode()),
                        "median_lab": [50.0, float(ordinal), 0.0],
                        "source_render_receipt_sha256": "1" * 64,
                        "source_validation_receipt_sha256": "2" * 64,
                        "source_render_plan_sha256": "3" * 64,
                        "known_good_profile_sha256": "4" * 64,
                    }
                )
                peers.setdefault(stage, []).append(evidence_id)
                cursor += 1
        for ordinal, stage in enumerate(stages, start=1):
            configurations.append(
                {
                    "recipe_id": recipe,
                    "stage_id": stage,
                    "stage_ordinal": ordinal,
                    "tank_id": "t100" if ordinal == 1 else "t200",
                    "peer_evidence_ids": peers[stage],
                    "threshold_delta_e76": 1.5,
                    "configuration_target_evidence_id": f"CONFIG-{ordinal}",
                }
            )
    database = tmp_path / "development-reference.sqlite3"
    _create_minimal_store(
        database,
        records=records,
        configurations=configurations,
        provenance={
            "development_reference_bank_sha256": "a" * 64,
            "delta_e_frozen_config_sha256": "b" * 64,
            "materialization_receipt_sha256": "c" * 64,
            "predicted_store_export_sha256": "d" * 64,
            "derived_source_manifest_sha256": "e" * 64,
            "predictor": {
                "producer_id": "accepted-test-predictor",
                "model_id": "test-model",
                "model_sha256": "8" * 64,
                "config_sha256": "9" * 64,
                "code_commit": "a" * 40,
            },
            "predictor_sha256": "f" * 64,
            "accepted_store_tuple": {
                "database_path": str(tmp_path / "accepted.sqlite3"),
                "database_sha256": "6" * 64,
                "population_receipt_path": str(tmp_path / "population-receipt.json"),
                "population_receipt_sha256": "7" * 64,
            },
            "inn67_provenance": {"fixed": "test"},
        },
    )
    snapshot = hashlib.sha256(database.read_bytes()).hexdigest()
    return DevelopmentReferenceStore(
        database, expected_snapshot_sha256=snapshot
    ), database, records


def test_coverage_and_target_queries_are_factory_bound_and_image_citable(
    tmp_path: Path,
) -> None:
    from fluidmix.memory.history.development_reference import (
        assert_development_reference_citation,
        bind_runtime_target_evidence,
    )

    store, _, _ = _store(tmp_path)
    coverage = store.coverage_queries()
    assert len(coverage) == 2
    assert all(query.query_kind == "development_reference_coverage" for query in coverage)
    assert all(len(query.members) == 2 for query in coverage)

    target = bind_runtime_target_evidence(
        {
            "evidence_id": "WP1-TARGET",
            "event_cursor": 0,
            "recipe_id": "R-A",
            "stage_id": "STAGE-WP1-CASE-001-1",
            "stage_ordinal": 1,
            "tank_id": "t100",
            "split_role": "test",
            "mask_source": "predicted",
        }
    )
    before = hashlib.sha256(store.database.read_bytes()).hexdigest()
    assert before == store.snapshot_sha256
    query = store.reference_bank(target)
    assert hashlib.sha256(store.database.read_bytes()).hexdigest() == before
    assert query.query_kind == "development_reference_bank"
    assert query.filters["target_stage_id"] == "STAGE-WP1-CASE-001-1"
    assert "before" not in query.filters
    assert store.reference_bank(target).query_sha256 == query.query_sha256
    assert hashlib.sha256(store.database.read_bytes()).hexdigest() == before
    assert {member.batch_id for member in query.members} == {
        "INN67-DEV-RB-B01", "INN67-DEV-RB-B02"
    }
    member = query.members[0]
    assert member.source_render_receipt_sha256 == "1" * 64
    assert member.source_validation_receipt_sha256 == "2" * 64
    assert member.source_render_plan_sha256 == "3" * 64
    assert member.known_good_profile_sha256 == "4" * 64
    assert member.predictor_sha256 == "f" * 64
    assert member.predictor_producer_id == "accepted-test-predictor"
    assert member.accepted_store_database_path == str(tmp_path / "accepted.sqlite3")
    assert member.accepted_store_database_sha256 == "6" * 64
    assert member.accepted_store_population_receipt_path == str(
        tmp_path / "population-receipt.json"
    )
    assert member.accepted_store_population_receipt_sha256 == "7" * 64
    assert member.donor_store_snapshot_sha256 == before
    assert assert_development_reference_citation(
        query, member.evidence_id, member.image_id
    ) is member
    with pytest.raises(ValueError, match="not a member"):
        assert_development_reference_citation(query, member.evidence_id, "wrong")


def test_store_requires_target_stage_ordinal_and_rejects_source_byte_drift(
    tmp_path: Path,
) -> None:
    from fluidmix.memory.history.development_reference import bind_runtime_target_evidence

    store, _, records = _store(tmp_path)
    with pytest.raises(ValueError, match="runtime_target_fields_invalid"):
        bind_runtime_target_evidence(
            {
                "evidence_id": "MISSING-ORDINAL",
                "event_cursor": 0,
                "recipe_id": "R-A",
                "stage_id": "STAGE-WP1-CASE-001-1",
                "tank_id": "t100",
                "split_role": "test",
                "mask_source": "predicted",
            }
        )
    Path(records[0]["rgb_path"]).write_bytes(b"drift")
    with pytest.raises(ValueError, match="RGB.*SHA-256"):
        store.coverage_queries()


def test_minimal_schema_is_append_only_and_has_no_generic_or_sim_tables(
    tmp_path: Path,
) -> None:
    import sqlite3

    _, database, _ = _store(tmp_path)
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "sim_metadata" not in tables
        assert not ({"fills", "clips", "measurements", "runs", "vials"} & tables)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE development_reference_records SET recipe_id='wrong'"
            )


def test_materializer_cleans_only_its_fresh_output_after_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fluidmix.memory.history.development_reference as module

    assessment = {
        **{field: field[0] * 64 for field in module._PROVENANCE_FIELDS},
        "_minimal_records": [],
        "_configurations": [],
    }
    monkeypatch.setattr(
        module, "validate_development_reference_inputs", lambda **_kwargs: assessment
    )
    monkeypatch.setattr(
        module, "_create_minimal_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.DevelopmentReferenceError("injected_publication_failure")
        ),
    )
    output = tmp_path / "fresh-output"
    with pytest.raises(ValueError, match="injected_publication_failure"):
        module.materialize_development_reference_store(
            predicted_store_export_path="unused",
            development_reference_bank_path="unused",
            delta_e_frozen_config_path="unused",
            materialization_receipt_path="unused",
            output_dir=output,
        )
    assert not output.exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "owner.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="output_directory_exists"):
        module.materialize_development_reference_store(
            predicted_store_export_path="unused",
            development_reference_bank_path="unused",
            delta_e_frozen_config_path="unused",
            materialization_receipt_path="unused",
            output_dir=occupied,
        )
    assert marker.read_text() == "preserve"

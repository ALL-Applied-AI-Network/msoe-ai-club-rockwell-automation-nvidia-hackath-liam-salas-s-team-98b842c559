import json
import hashlib
import sqlite3
from pathlib import Path

import pytest

from scripts.materialize_ui_data import MaterializationError, materialize, resolve_source


SHA = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _measurement_sha(stage: dict, *, median_lab: list[float] | None) -> str:
    payload = {
        "image_id": stage["image_id"],
        "measurement_id": stage["measurement_id"],
        "evidence_id": stage["evidence_id"],
        "rgb_sha256": stage["rgb_sha256"],
        "mask_sha256": stage["mask_sha256"],
        "validity": stage["validity"],
        "limit_codes": stage["limit_codes"],
        "median_lab": median_lab,
        "mad_lab": [0.1, 0.1, 0.1],
        "iqr_lab": [0.2, 0.2, 0.2],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_stage(
    source: Path,
    decision_index: int,
    stage_ordinal: int,
    *,
    outcome: str,
) -> tuple[dict, tuple]:
    evidence_id = f"evidence-{decision_index}-{stage_ordinal}"
    image_id = f"image-{decision_index}-{stage_ordinal}"
    measurement_id = f"measurement-{decision_index}-{stage_ordinal}"
    artifact_dir = source / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    rgb = artifact_dir / f"{image_id}.jpg"
    mask = artifact_dir / f"{image_id}.png"
    rgb.write_bytes(b"jpeg-fixture-" + image_id.encode())
    mask.write_bytes(b"png-fixture-" + image_id.encode())
    valid = outcome != "insufficient_evidence"
    outlier = outcome == "discard" and stage_ordinal == 2
    median = [50.0 + decision_index, 2.0, 3.0] if valid else None
    target_delta = (2.0 if stage_ordinal == 1 else 4.0) if valid else None
    threshold = (5.0 if outcome == "keep" else 3.0) if valid else None
    stage = {
        "evidence_id": evidence_id,
        "measurement_id": measurement_id,
        "image_id": image_id,
        "vial_id": f"run-final:{decision_index}",
        "recipe_id": f"recipe-{decision_index}",
        "stage_id": f"dispense-{stage_ordinal}",
        "stage_ordinal": stage_ordinal,
        "tank_id": f"t{stage_ordinal}00",
        "source": "runtime_camera",
        "median_lab": median,
        "validity": "valid" if valid else "insufficient_evidence",
        "stage_outlier": outlier if valid else None,
        "target_delta_e76": target_delta,
        "peer_delta_e76": 1.0 if valid else None,
        "threshold_delta_e76": threshold,
        "reference_center_lab": [49.0, 2.0, 3.0] if valid else None,
        "donors": [
            {
                "evidence_id": f"donor-{decision_index}-{stage_ordinal}-{donor}",
                "image_id": f"donor-image-{decision_index}-{stage_ordinal}-{donor}",
                "batch_id": f"batch-{donor}",
                "run_id": f"donor-run-{donor}",
                "recipe_id": f"recipe-{decision_index}",
                "stage_ordinal": stage_ordinal,
                "tank_id": f"t{stage_ordinal}00",
                "median_lab": [49.0 + donor, 2.0, 3.0],
            }
            for donor in (1, 2)
        ] if valid else [],
        "query_sha256": SHA if valid else None,
        "donor_store_snapshot_sha256": "b" * 64 if valid else None,
        "configuration_sha256": "c" * 64 if valid else None,
        "limit_codes": [] if valid else ["target_measurement_invalid", "fixture_unavailable"],
        "unit": "delta_e76",
        "rgb_sha256": _sha256(rgb),
        "mask_sha256": _sha256(mask),
        "entry_sha256": "d" * 64,
        "registry_sha256": "e" * 64,
    }
    stage["measurement_sha256"] = _measurement_sha(stage, median_lab=median)
    database_row = (
        stage,
        rgb.relative_to(source).as_posix(),
        mask.relative_to(source).as_posix(),
        median,
    )
    return stage, database_row


def _make_decision(source: Path, index: int, outcome: str) -> tuple[dict, list[tuple]]:
    built = [
        _make_stage(source, index, ordinal, outcome=outcome)
        for ordinal in (1, 2)
    ]
    stages = [row[0] for row in built]
    citations = []
    for stage in stages:
        citations.append({
            "role": "target",
            "stage_ordinal": stage["stage_ordinal"],
            "evidence_id": stage["evidence_id"],
            "measurement_id": stage["measurement_id"],
            "image_id": stage["image_id"],
        })
        citations.extend({
            "role": "synthetic_development_donor",
            "stage_ordinal": stage["stage_ordinal"],
            "evidence_id": donor["evidence_id"],
            "image_id": donor["image_id"],
            "batch_id": donor["batch_id"],
        } for donor in stage["donors"])
    explanation = {
        "template_id": "wp7-camera-colour-synthetic-reference-v1",
        "model": None,
        "text": f"Authoritative {outcome} explanation {index}.",
        "citations": citations,
        "claim_boundary": "Camera colour comparison only; plant action is unverified.",
    }
    denominators = {
        "programmed_stage_count": 2,
        "valid_stage_count": 0 if outcome == "insufficient_evidence" else 2,
        "outlier_stage_count": 1 if outcome == "discard" else 0,
        "bound_donor_count": 0 if outcome == "insufficient_evidence" else 4,
        "required_donors_per_supported_stage": 2,
    }
    limit_codes = (
        [] if outcome != "insufficient_evidence"
        else ["target_measurement_invalid", "fixture_unavailable"]
    )
    decision = {
        "schema_version": "wp7-final-decision-v1",
        "decision_id": f"decision-{index}",
        "disposition": outcome,
        "model": None,
        "stages": stages,
        "denominators": denominators,
        "limit_codes": limit_codes,
        "explanation": explanation,
        "operator_record": {
            "schema_version": "wp7-operator-record-v1",
            "decision_id": f"decision-{index}",
            "disposition": outcome,
            "model": None,
            "denominators": denominators,
            "source_bindings": {"targets": [], "development_references": []},
            "limit_codes": limit_codes,
            "explanation": explanation,
        },
    }
    return decision, [row[1] for row in built]


def _write_store(source: Path, rows: list[tuple], *, version: int = 3) -> None:
    database = source / "evidence.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE artifacts (artifact_id TEXT PRIMARY KEY, role TEXT, path TEXT, sha256 TEXT, byte_size INTEGER);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, split_role TEXT, split_version TEXT, content_sha256 TEXT);
        CREATE TABLE vials (vial_id TEXT PRIMARY KEY, run_id TEXT, case_id TEXT, vial_ordinal INTEGER, block_recipe_id TEXT, block_position INTEGER, recipe_id TEXT, previous_recipe_id TEXT, recipe_json TEXT, dispense_order_json TEXT, content_sha256 TEXT);
        CREATE TABLE fills (fill_id TEXT PRIMARY KEY);
        CREATE TABLE clips (clip_id TEXT PRIMARY KEY);
        CREATE TABLE stages (evidence_id TEXT PRIMARY KEY, record_sha256 TEXT, event_cursor INTEGER, run_id TEXT, vial_id TEXT, stage_id TEXT, stage_ordinal INTEGER, tank_id TEXT, recipe_id TEXT, previous_recipe_id TEXT, split_role TEXT, phase TEXT);
        CREATE TABLE images (image_id TEXT PRIMARY KEY, evidence_id TEXT, rgb_artifact_id TEXT, mask_artifact_id TEXT, mask_source TEXT, content_sha256 TEXT);
        CREATE TABLE measurements (measurement_id TEXT PRIMARY KEY, evidence_id TEXT, median_lab_json TEXT, mad_lab_json TEXT, iqr_lab_json TEXT, pairwise_delta_e76_json TEXT, content_sha256 TEXT);
        CREATE TABLE decisions (evidence_id TEXT PRIMARY KEY);
        CREATE TABLE explanations (evidence_id TEXT PRIMARY KEY);
        CREATE TABLE history_queries (query_sha256 TEXT PRIMARY KEY);
        CREATE TABLE history_members (query_sha256 TEXT, member_ordinal INTEGER);
        CREATE TABLE regime_bindings (colour_regime_id TEXT, evidence_id TEXT);
        CREATE TABLE regime_references (colour_regime_id TEXT, evidence_id TEXT);
        CREATE TABLE sim_metadata (metadata_id TEXT PRIMARY KEY, payload BLOB, payload_sha256 TEXT);
    """)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.execute("INSERT INTO runs VALUES (?,?,?,?)", ("run-final", "test", "split-v1", SHA))
    inserted_vials: set[str] = set()
    cursor = 0
    for stage, rgb_path, mask_path, median in rows:
        vial_id = stage["vial_id"]
        if vial_id not in inserted_vials:
            inserted_vials.add(vial_id)
            ordinal = int(vial_id.rsplit(":", 1)[1])
            connection.execute(
                "INSERT INTO vials VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (vial_id, "run-final", f"case-{ordinal}", ordinal, stage["recipe_id"], ordinal,
                 stage["recipe_id"], None, json.dumps({"t100": 1}), json.dumps([stage["tank_id"]]), SHA),
            )
        rgb_id = f"artifact-rgb-{stage['image_id']}"
        mask_id = f"artifact-mask-{stage['image_id']}"
        for artifact_id, role, path, digest in (
            (rgb_id, "rgb", rgb_path, stage["rgb_sha256"]),
            (mask_id, "mask", mask_path, stage["mask_sha256"]),
        ):
            connection.execute(
                "INSERT INTO artifacts VALUES (?,?,?,?,?)",
                (artifact_id, role, path, digest, (source / path).stat().st_size),
            )
        connection.execute(
            "INSERT INTO stages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (stage["evidence_id"], SHA, cursor, "run-final", vial_id, stage["stage_id"],
             stage["stage_ordinal"], stage["tank_id"], stage["recipe_id"], None, "test", "final"),
        )
        cursor += 1
        connection.execute(
            "INSERT INTO images VALUES (?,?,?,?,?,?)",
            (stage["image_id"], stage["evidence_id"], rgb_id, mask_id, "predicted", SHA),
        )
        connection.execute(
            "INSERT INTO measurements VALUES (?,?,?,?,?,?,?)",
            (stage["measurement_id"], stage["evidence_id"], json.dumps(median),
             json.dumps([0.1, 0.1, 0.1]), json.dumps([0.2, 0.2, 0.2]), "[]", SHA),
        )
    connection.execute(
        "INSERT INTO sim_metadata VALUES (?,?,?)", ("sealed", b"must-never-be-read", SHA)
    )
    connection.commit()
    connection.close()


def _published_source(tmp_path: Path, *, decision_format: str = "json") -> tuple[Path, list[dict]]:
    source = tmp_path / f"published-{decision_format}"
    source.mkdir()
    records_and_rows = [
        _make_decision(source, index, outcome)
        for index, outcome in enumerate(("keep", "discard", "insufficient_evidence"), start=1)
    ]
    records = [item[0] for item in records_and_rows]
    rows = [row for item in records_and_rows for row in item[1]]
    _write_store(source, rows)
    (source / "published.mp4").write_bytes(b"fixture-mp4")
    if decision_format == "json":
        (source / "wp7-final-decisions.json").write_text(json.dumps({"decisions": records}))
    elif decision_format == "jsonl":
        (source / "wp7-final-decisions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    else:
        decisions = source / "decisions"
        decisions.mkdir()
        for record in records:
            (decisions / f"{record['decision_id']}.json").write_text(json.dumps(record))
    return source, records


def test_source_selector_accepts_only_one_supported_value(tmp_path: Path) -> None:
    config = tmp_path / "source.json"
    repository = tmp_path / "repository"
    repository.mkdir()

    config.write_text(json.dumps({"source": "mock"}))
    assert resolve_source(config, repository_root=repository) == ("mock", None)

    config.write_text(json.dumps({"source": "development"}))
    with pytest.raises(MaterializationError, match="development source directory does not exist"):
        resolve_source(config, repository_root=repository)

    final = tmp_path / "published-final"
    final.mkdir()
    config.write_text(json.dumps({"source": str(final)}))
    assert resolve_source(config, repository_root=repository) == ("final", final.resolve())

    for value in ({"source": "mock", "extra": True}, {"source": ""}, [], {}):
        config.write_text(json.dumps(value))
        with pytest.raises(MaterializationError, match="exactly one non-empty string"):
            resolve_source(config, repository_root=repository)


@pytest.mark.parametrize("decision_format", ["json", "jsonl", "directory"])
def test_final_materializer_joins_exact_wp7_records_read_only_and_selects_focus(
    tmp_path: Path, decision_format: str
) -> None:
    source, records = _published_source(tmp_path, decision_format=decision_format)
    before = _tree_hashes(source)
    repository = tmp_path / "repository"
    output = repository / "ui-data"
    output.mkdir(parents=True)
    config = output / "source.json"
    config.write_text(json.dumps({"source": str(source)}))

    destination = materialize(repository_root=repository, config_path=config, output_dir=output)

    assert destination == output / "active-run.json"
    normalized = json.loads(destination.read_text())
    assert normalized["schema_version"] == "2.0.0"
    assert normalized["source"] == {
        "mode": "final", "origin": "wp7_final", "label": "Final WP7 data"
    }
    assert [vial["decision"]["outcome"] for vial in normalized["vials"]] == [
        "keep", "discard", "insufficient_evidence"
    ]
    assert normalized["vials"][0]["decision"]["delta_e"]["value"] == 4.0
    assert normalized["vials"][1]["decision"]["delta_e"]["value"] == 4.0
    assert normalized["vials"][2]["decision"]["delta_e"] is None
    assert normalized["vials"][2]["target_lab"] is None
    for vial, source_record in zip(normalized["vials"], records, strict=True):
        cited = {
            citation["evidence_id"]
            for citation in source_record["explanation"]["citations"]
        }
        preserved = {
            evidence["provenance"]["source_id"]
            for evidence in normalized["evidence"]
            if evidence["vial_id"] == vial["vial_id"]
            and evidence["target"] == "stage_citation"
        }
        assert cited == preserved
        assert vial["decision"]["explanation"]["text"] == source_record["explanation"]["text"]
    assert _tree_hashes(source) == before
    assert b"must-never-be-read" not in destination.read_bytes()


def test_final_materializer_fails_closed_on_ambiguity_and_cross_source_drift(tmp_path: Path) -> None:
    source, _records = _published_source(tmp_path)
    repository = tmp_path / "repository"
    output = repository / "ui-data"
    output.mkdir(parents=True)
    config = output / "source.json"
    config.write_text(json.dumps({"source": str(source)}))
    (source / "wp7-final-decisions.jsonl").write_text("{}\n")
    with pytest.raises(MaterializationError, match="ambiguous decision inputs"):
        materialize(repository_root=repository, config_path=config, output_dir=output)
    (source / "wp7-final-decisions.jsonl").unlink()

    records_path = source / "wp7-final-decisions.json"
    document = json.loads(records_path.read_text())
    document["decisions"][0]["stages"][0]["measurement_id"] = "measurement-drift"
    document["decisions"][0]["explanation"]["citations"][0]["measurement_id"] = "measurement-drift"
    document["decisions"][0]["operator_record"]["explanation"]["citations"][0]["measurement_id"] = "measurement-drift"
    records_path.write_text(json.dumps(document))
    with pytest.raises(MaterializationError, match="measurement identity mismatch"):
        materialize(repository_root=repository, config_path=config, output_dir=output)


def test_final_materializer_rejects_missing_citations_hash_drift_and_wrong_store_version(
    tmp_path: Path,
) -> None:
    for case in ("citation", "hash", "version"):
        case_root = tmp_path / case
        case_root.mkdir()
        source, _records = _published_source(case_root)
        repository = case_root / "repository"
        output = repository / "ui-data"
        output.mkdir(parents=True)
        config = output / "source.json"
        config.write_text(json.dumps({"source": str(source)}))
        if case == "citation":
            path = source / "wp7-final-decisions.json"
            document = json.loads(path.read_text())
            document["decisions"][0]["explanation"]["citations"].pop()
            document["decisions"][0]["operator_record"]["explanation"] = document["decisions"][0]["explanation"]
            path.write_text(json.dumps(document))
            pattern = "missing or mismatched citations"
        elif case == "hash":
            (source / "artifacts" / "image-1-1.jpg").write_bytes(b"drifted")
            pattern = "RGB artifact SHA-256 mismatch"
        else:
            database = source / "evidence.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
            connection.close()
            pattern = "unsupported evidence schema version 2"
        with pytest.raises(MaterializationError, match=pattern):
            materialize(repository_root=repository, config_path=config, output_dir=output)


def test_mock_mode_migrates_phase1_without_changing_decisions(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    output = tmp_path / "ui-data"
    output.mkdir()
    config = output / "source.json"
    config.write_text('{"source":"mock"}')
    destination = materialize(repository_root=repository, config_path=config, output_dir=output)
    normalized = json.loads(destination.read_text())
    original = json.loads((repository / "ui-data" / "mock-run.json").read_text())
    assert normalized["schema_version"] == "2.0.0"
    assert normalized["source"]["mode"] == "mock"
    assert [vial["decision"] for vial in normalized["vials"]] == [
        vial["decision"] for vial in original["vials"]
    ]

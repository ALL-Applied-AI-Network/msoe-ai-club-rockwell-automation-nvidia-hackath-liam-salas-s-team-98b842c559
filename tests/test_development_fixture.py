import json
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest

from scripts.build_development_fixture import (
    DEFAULT_SOURCE,
    FixtureBuildError,
    SOURCE_DATABASE_SHA256,
    SOURCE_RECORDS_SHA256,
    build_fixture,
    file_sha256,
)
from scripts.materialize_ui_data import MaterializationError, materialize


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _ffmpeg(repository: Path) -> Path:
    executable = repository / "ui" / "node_modules" / "ffmpeg-static" / "ffmpeg"
    if not executable.is_file():
        pytest.skip("ui/node_modules ffmpeg-static is required for fixture media verification")
    return executable


def test_committed_development_fixture_materializes_as_schema_v2(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    output = tmp_path / "ui-data"
    output.mkdir()
    config = output / "source.json"
    config.write_text('{"source":"development"}', encoding="utf-8")

    destination = materialize(
        repository_root=repository,
        config_path=config,
        output_dir=output,
    )

    normalized = json.loads(destination.read_text(encoding="utf-8"))
    assert normalized["schema_version"] == "2.0.0"
    assert normalized["source"] == {
        "mode": "development",
        "origin": "synthetic_development",
        "label": "Development data",
    }
    assert len(normalized["vials"]) == 18
    assert {vial["decision"]["outcome"] for vial in normalized["vials"]} == {
        "insufficient_evidence"
    }
    assert normalized["run"]["video"]["duration_seconds"]["value"] == 270.0

    source_records = json.loads(
        (repository / "ui-data" / "development" / "development-records.json").read_text()
    )
    database = sqlite3.connect(repository / "ui-data" / "development" / "evidence.sqlite3")
    try:
        associations = {
            (run_id, ordinal): (vial_id, recipe_id)
            for vial_id, run_id, ordinal, recipe_id in database.execute(
                "SELECT vial_id,run_id,vial_ordinal,recipe_id FROM vials"
            )
        }
    finally:
        database.close()
    normalized_by_id = {
        vial["decision"]["record_id"]: vial for vial in normalized["vials"]
    }
    evidence_by_id = {item["evidence_id"]: item for item in normalized["evidence"]}
    assert set(normalized_by_id) == {record["decision_id"] for record in source_records}
    for record in source_records:
        vial = normalized_by_id[record["decision_id"]]
        decision = vial["decision"]
        assert (vial["vial_id"], vial["recipe_id"]) == associations[
            (record["run_id"], record["vial_ordinal"])
        ]
        assert decision["outcome"] == record["disposition"]
        assert decision["limit_codes"] == record["limit_codes"]
        assert decision["delta_e"] is None
        assert decision["threshold"] is None
        assert vial["reference_lab"] is None
        assert decision["explanation"] == {
            "source": "deterministic_template",
            "text": record["guard_result"],
            "evidence_refs": decision["evidence_refs"],
        }
        record_evidence = evidence_by_id[decision["evidence_refs"][0]]
        assert record_evidence["provenance"]["source_id"] == record["model_revision"]
        for index, history_ref in enumerate(record["history_refs"], start=1):
            history_evidence = evidence_by_id[decision["evidence_refs"][index]]
            assert history_evidence["provenance"]["source_id"] == history_ref["evidence_id"]
            assert history_ref["query_sha256"] in history_evidence["provenance"]["note"]
    assert all(
        code != "unsupported_development_selector"
        for vial in normalized["vials"]
        for code in vial["decision"]["limit_codes"]
    )
    assert {vial["decision"]["outcome"] for vial in normalized["vials"]} == {
        "insufficient_evidence"
    }


def test_final_mode_rejects_published_development_record_shape(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    fixture = repository / "ui-data" / "development"
    source = tmp_path / "not-a-final-output"
    source.mkdir()
    shutil.copyfile(fixture / "evidence.sqlite3", source / "evidence.sqlite3")
    shutil.copyfile(fixture / "development-reel.mp4", source / "published.mp4")
    shutil.copyfile(
        fixture / "development-records.json", source / "wp7-final-decisions.json"
    )
    output = tmp_path / "ui-data"
    output.mkdir()
    config = output / "source.json"
    config.write_text(json.dumps({"source": str(source)}))

    with pytest.raises(MaterializationError, match="fields are invalid"):
        materialize(repository_root=repository, config_path=config, output_dir=output)


def test_committed_fixture_has_exact_provenance_assets_and_no_sim_metadata() -> None:
    fixture = Path(__file__).parents[1] / "ui-data" / "development"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "Development data"
    assert manifest["decision_summary"] == {
        "insufficient_evidence": 18,
        "keep": 0,
        "discard": 0,
    }
    assert manifest["sim_metadata"] == {"row_count": 0, "source_contents_read": False}
    assert [item["sha256"] for item in manifest["source_inputs"]] == [
        SOURCE_DATABASE_SHA256,
        SOURCE_RECORDS_SHA256,
    ]
    assert file_sha256(fixture / "development-records.json") == SOURCE_RECORDS_SHA256
    assert len(manifest["source_ids"]) == 6
    assert manifest["split_versions"] == ["whole-run-varied-v3", "whole-run-varied-v4"]

    database = sqlite3.connect(fixture / "evidence.sqlite3")
    try:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 3
        assert database.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert database.execute("SELECT count(*) FROM runs").fetchone()[0] == 6
        assert database.execute("SELECT count(*) FROM vials").fetchone()[0] == 18
        assert database.execute("SELECT count(*) FROM sim_metadata").fetchone()[0] == 0
        assert {
            row[0] for row in database.execute("SELECT DISTINCT split_version FROM runs")
        } == {"whole-run-varied-v3", "whole-run-varied-v4"}
        artifacts = list(database.execute(
            "SELECT path,sha256,byte_size FROM artifacts ORDER BY artifact_id"
        ))
        assert len(artifacts) == 90
        for relative, digest, size in artifacts:
            path = fixture / relative
            assert path.stat().st_size == size
            assert file_sha256(path) == digest
    finally:
        database.close()

    listed = {entry["path"]: entry for entry in manifest["derived_files"]}
    actual = _tree_hashes(fixture)
    del actual["manifest.json"]
    assert set(listed) == set(actual)
    assert all(listed[path]["sha256"] == digest for path, digest in actual.items())


def test_rebuild_is_deterministic_and_does_not_mutate_approved_inputs(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    source = DEFAULT_SOURCE
    records = source.with_name("explanation-records.json")
    if not source.is_file() or not records.is_file():
        pytest.skip("approved local development source is not present")
    ffmpeg = _ffmpeg(repository)
    before = (file_sha256(source), file_sha256(records))
    assert before == (SOURCE_DATABASE_SHA256, SOURCE_RECORDS_SHA256)

    first = build_fixture(source, records, tmp_path / "first", ffmpeg)
    second = build_fixture(source, records, tmp_path / "second", ffmpeg)

    assert _tree_hashes(first) == _tree_hashes(second)
    assert (file_sha256(source), file_sha256(records)) == before


def test_builder_rejects_hash_matching_inputs_at_unapproved_paths(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    ffmpeg = _ffmpeg(repository)
    copied = tmp_path / "copied-inputs"
    copied.mkdir()
    database = copied / "evidence.sqlite3"
    records = copied / "explanation-records.json"
    shutil.copyfile(DEFAULT_SOURCE, database)
    shutil.copyfile(DEFAULT_SOURCE.with_name("explanation-records.json"), records)

    with pytest.raises(FixtureBuildError, match="exact approved source path"):
        build_fixture(database, records, tmp_path / "rejected-output", ffmpeg)


def test_development_reel_decodes_completely_and_is_about_four_and_a_half_minutes() -> None:
    repository = Path(__file__).parents[1]
    ffmpeg = _ffmpeg(repository)
    reel = repository / "ui-data" / "development" / "development-reel.mp4"
    completed = subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(reel), "-f", "null", "-"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    # The 1 fps presentation cadence may quantize the container duration by one second.
    probe = subprocess.run(
        [str(ffmpeg), "-i", str(reel), "-f", "null", "-"],
        text=True,
        capture_output=True,
    )
    duration_line = next(line for line in probe.stderr.splitlines() if "Duration:" in line)
    match = re.search(r"Duration: 00:(\d\d):(\d\d)", duration_line)
    assert match is not None
    duration_seconds = int(match.group(1)) * 60 + int(match.group(2))
    assert 265 <= duration_seconds <= 275

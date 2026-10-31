import json
import re
from pathlib import Path
import sqlite3
import subprocess

import pytest

from scripts.build_development_fixture import (
    DEFAULT_SOURCE,
    SOURCE_DATABASE_SHA256,
    SOURCE_RECORDS_SHA256,
    build_fixture,
    file_sha256,
)
from scripts.materialize_ui_data import materialize


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

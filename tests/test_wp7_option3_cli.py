"""Synthetic, source-only contracts for the WP7 option-3 preparation CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_inn67_predicted_store_producer import _produce, _source_fixture
from test_wp7_route_preflight import (
    _development_reference_queries,
    _development_route_decision,
    _development_store,
    _ordered_video_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_wp7_development_reference_route.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _sha(path)


def _publication_fixture(tmp_path: Path) -> dict[str, object]:
    queries = _development_reference_queries(tmp_path / "synthetic")
    database = tmp_path / "synthetic" / "development-reference.sqlite3"
    database_sha = _sha(database)
    store_receipt_path = tmp_path / "store-receipt.json"
    ordered_path = tmp_path / "ordered-video-provenance.json"
    decision_path = tmp_path / "liam-option-3-route-decision.json"
    store_receipt_sha = _write_json(
        store_receipt_path,
        _development_store(
            [query.query_sha256 for query in queries], snapshot_sha256=database_sha
        ),
    )
    ordered_sha = _write_json(ordered_path, _ordered_video_provenance())
    decision_sha = _write_json(decision_path, _development_route_decision())
    return {
        "database": database,
        "database_sha": database_sha,
        "store_receipt": store_receipt_path,
        "store_receipt_sha": store_receipt_sha,
        "ordered": ordered_path,
        "ordered_sha": ordered_sha,
        "decision": decision_path,
        "decision_sha": decision_sha,
        "preflight_output": tmp_path / "preflight.json",
        "publication_output": tmp_path / "publication-receipt.json",
    }


def _publish_argv(fixture: dict[str, object]) -> list[str]:
    return [
        "publish-preflight",
        "--development-reference-store",
        str(fixture["database"]),
        "--development-reference-store-sha256",
        str(fixture["database_sha"]),
        "--store-provenance-receipt",
        str(fixture["store_receipt"]),
        "--store-provenance-receipt-sha256",
        str(fixture["store_receipt_sha"]),
        "--ordered-video-provenance",
        str(fixture["ordered"]),
        "--ordered-video-provenance-sha256",
        str(fixture["ordered_sha"]),
        "--route-decision",
        str(fixture["decision"]),
        "--route-decision-sha256",
        str(fixture["decision_sha"]),
        "--preflight-output",
        str(fixture["preflight_output"]),
        "--publication-receipt-output",
        str(fixture["publication_output"]),
    ]


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_publish_preflight_writes_bound_execution_valid_wrapper(tmp_path: Path) -> None:
    fixture = _publication_fixture(tmp_path)
    sources = [
        Path(fixture[key])
        for key in ("database", "store_receipt", "ordered", "decision")
    ]
    before = {path: _sha(path) for path in sources}
    completed = _run(_publish_argv(fixture))

    assert completed.returncode == 0, completed.stderr
    preflight_path = Path(fixture["preflight_output"])
    publication_path = Path(fixture["publication_output"])
    preflight = json.loads(preflight_path.read_text())
    publication = json.loads(publication_path.read_text())
    assert json.loads(completed.stdout) == publication
    assert publication["preflight_binding"] == {
        "path": str(preflight_path.absolute()),
        "raw_file_sha256": _sha(preflight_path),
        "canonical_preflight_sha256": preflight["preflight_sha256"],
    }
    assert publication["source_bindings"] == {
        "development_reference_store": {
            "path": str(Path(fixture["database"]).absolute()),
            "raw_file_sha256": fixture["database_sha"],
        },
        "ordered_video_provenance": {
            "path": str(Path(fixture["ordered"]).absolute()),
            "raw_file_sha256": fixture["ordered_sha"],
        },
        "route_decision": {
            "canonical_decision_sha256": preflight["route_decision"][
                "route_decision_sha256"
            ],
            "path": str(Path(fixture["decision"]).absolute()),
            "raw_file_sha256": fixture["decision_sha"],
        },
        "store_provenance_receipt": {
            "path": str(Path(fixture["store_receipt"]).absolute()),
            "raw_file_sha256": fixture["store_receipt_sha"],
        },
    }
    assert preflight["history"]["query_count"] == 16
    assert preflight["history"]["member_count"] == 32
    from fluidmix.carryover_pipeline import (
        _validate_execution_preflight,
        load_preflight_receipt,
    )

    loaded = load_preflight_receipt(preflight_path)
    assert _validate_execution_preflight(
        loaded,
        expected_sha256=preflight["preflight_sha256"],
        selected_mode="development_reference_backed",
    ) == preflight
    with pytest.raises(ValueError, match="selected mode mismatch"):
        _validate_execution_preflight(
            load_preflight_receipt(publication_path),
            expected_sha256=preflight["preflight_sha256"],
            selected_mode="development_reference_backed",
        )
    assert before == {path: _sha(path) for path in sources}


def test_materialize_store_cli_uses_explicit_sources_without_mutating_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fluidmix.inn67_predicted_store_producer as producer
    from fluidmix.inn67_bank_materializer import materialize_reference_bank

    monkeypatch.setattr(producer, "_verify_lineage_file", lambda _path, _expected: True)
    source_fixture = _source_fixture(tmp_path / "source")
    predicted = tmp_path / "predicted"
    _produce(source_fixture, predicted)
    plan_root, amendment, materialization_amendment, inventory, _ = source_fixture
    bank = tmp_path / "bank"
    materialize_reference_bank(
        plan_root=plan_root,
        amendment_path=amendment,
        materialization_amendment_path=materialization_amendment,
        render_inventory_path=inventory,
        predicted_store_export_path=predicted / "predicted-store-export.json",
        output_dir=bank,
    )
    sources = [
        predicted / "predicted-store-export.json",
        bank / "development-reference-bank.json",
        bank / "delta-e-frozen-config.json",
        bank / "materialization-receipt.json",
    ]
    before = {path: _sha(path) for path in sources}
    output = tmp_path / "development-reference-store"

    completed = _run(
        [
            "materialize-store",
            "--predicted-store-export",
            str(sources[0]),
            "--development-reference-bank",
            str(sources[1]),
            "--delta-e-frozen-config",
            str(sources[2]),
            "--materialization-receipt",
            str(sources[3]),
            "--output-dir",
            str(output),
        ]
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["coverage_query_count"] == 16
    assert result["coverage_member_count"] == 32
    assert len(result["coverage_query_sha256s"]) == 16
    assert before == {path: _sha(path) for path in sources}


@pytest.mark.parametrize("existing_role", ["preflight_output", "publication_output"])
def test_publish_rejects_each_existing_output_before_source_read(
    tmp_path: Path, existing_role: str
) -> None:
    destination = tmp_path / f"existing-{existing_role}.json"
    destination.write_text("owner data")
    fixture = {
        "database": tmp_path / "protected-store-must-not-be-read",
        "database_sha": "0" * 64,
        "store_receipt": tmp_path / "protected-receipt-must-not-be-read",
        "store_receipt_sha": "0" * 64,
        "ordered": tmp_path / "protected-ordered-must-not-be-read",
        "ordered_sha": "0" * 64,
        "decision": tmp_path / "protected-decision-must-not-be-read",
        "decision_sha": "0" * 64,
        "preflight_output": tmp_path / "fresh-preflight.json",
        "publication_output": tmp_path / "fresh-publication.json",
    }
    fixture[existing_role] = destination

    completed = _run(_publish_argv(fixture))

    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert destination.read_text() == "owner data"


@pytest.mark.parametrize("conflict", ["other_output", "source"])
def test_publish_requires_outputs_distinct_from_each_other_and_sources(
    tmp_path: Path, conflict: str
) -> None:
    fixture = _publication_fixture(tmp_path)
    if conflict == "other_output":
        fixture["publication_output"] = fixture["preflight_output"]
        expected = "publication receipt output must be distinct from preflight output"
    else:
        fixture["preflight_output"] = fixture["database"]
        expected = "development-reference store must be distinct from preflight output"

    completed = _run(_publish_argv(fixture))

    assert completed.returncode != 0
    assert expected in completed.stderr


def test_exclusive_two_file_write_never_removes_racing_owner_file(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("wp7_option3_cli_write", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = tmp_path / "racing-owner.json"
    owner.write_text("owner data")
    second = tmp_path / "second.json"

    with pytest.raises(FileExistsError):
        module._write_two_exclusive_json(owner, {"first": 1}, second, {"second": 2})

    assert owner.read_text() == "owner data"
    assert not second.exists()


def test_materialize_rejects_existing_output_before_source_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-store"
    output.mkdir()
    owner = output / "owner.txt"
    owner.write_text("owner data")
    missing = tmp_path / "protected-input-must-not-be-read"

    completed = _run(
        [
            "materialize-store",
            "--predicted-store-export",
            str(missing),
            "--development-reference-bank",
            str(missing),
            "--delta-e-frozen-config",
            str(missing),
            "--materialization-receipt",
            str(missing),
            "--output-dir",
            str(output),
        ]
    )

    assert completed.returncode != 0
    assert "output_directory_exists" in completed.stderr
    assert owner.read_text() == "owner data"


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_publish_rejects_nonregular_or_symlink_source(tmp_path: Path, kind: str) -> None:
    fixture = _publication_fixture(tmp_path)
    original = Path(fixture["ordered"])
    invalid = tmp_path / f"invalid-{kind}"
    if kind == "symlink":
        invalid.symlink_to(original)
    else:
        invalid.mkdir()
    fixture["ordered"] = invalid

    completed = _run(_publish_argv(fixture))

    assert completed.returncode != 0
    assert "ordered-video provenance is not an exact regular file" in completed.stderr
    assert not Path(fixture["preflight_output"]).exists()
    assert not Path(fixture["publication_output"]).exists()


@pytest.mark.parametrize(
    "digest_key",
    ["database_sha", "store_receipt_sha", "ordered_sha", "decision_sha"],
)
def test_publish_rejects_each_stale_raw_file_hash(
    tmp_path: Path, digest_key: str
) -> None:
    fixture = _publication_fixture(tmp_path)
    fixture[digest_key] = "0" * 64

    completed = _run(_publish_argv(fixture))

    assert completed.returncode != 0
    assert "raw-file SHA-256 mismatch" in completed.stderr
    assert not Path(fixture["preflight_output"]).exists()
    assert not Path(fixture["publication_output"]).exists()


def test_publish_rejects_store_snapshot_receipt_mismatch(tmp_path: Path) -> None:
    fixture = _publication_fixture(tmp_path)
    receipt_path = Path(fixture["store_receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["store_snapshot_sha256"] = "0" * 64
    fixture["store_receipt_sha"] = _write_json(receipt_path, receipt)

    completed = _run(_publish_argv(fixture))

    assert completed.returncode != 0
    assert "store snapshot/receipt mismatch" in completed.stderr
    assert not Path(fixture["preflight_output"]).exists()
    assert not Path(fixture["publication_output"]).exists()


def test_publish_never_reads_ordered_video_named_inside_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publication_fixture(tmp_path)
    spec = importlib.util.spec_from_file_location("wp7_option3_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = module.build_parser().parse_args(_publish_argv(fixture))
    protected = Path(_ordered_video_provenance()["ordered_video_path"])
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.absolute() == protected.absolute():
            raise AssertionError("ordered video bytes were opened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    publication = module.publish_preflight(arguments)

    assert publication["one_shot_inputs_opened"] is False
    assert Path(fixture["preflight_output"]).is_file()
    assert Path(fixture["publication_output"]).is_file()

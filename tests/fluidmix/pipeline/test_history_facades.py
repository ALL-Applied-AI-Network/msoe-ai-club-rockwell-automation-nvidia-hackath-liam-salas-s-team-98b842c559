"""Pipeline ownership of concrete storage dependencies for memory workflows."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def test_history_population_injects_storage_capabilities(monkeypatch):
    import fluidmix.pipeline.history_population as pipeline

    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(pipeline, "_build_development_history_population", build)

    assert pipeline.build_development_history_population(marker="value") == {
        "status": "ok"
    }
    assert observed == {
        "marker": "value",
        "evidence_store_factory": pipeline.EvidenceStore,
        "assert_history_citation_fn": pipeline.assert_history_citation,
    }


def test_sixth_run_injects_storage_capabilities(monkeypatch):
    import fluidmix.pipeline.sixth_development_run as pipeline

    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(pipeline, "_build_six_run_development_store", build)

    assert pipeline.build_six_run_development_store(marker="value") == {
        "status": "ok"
    }
    assert observed == {
        "marker": "value",
        "evidence_store_factory": pipeline.EvidenceStore,
        "canonical_source_record_sha256_fn": (
            pipeline.canonical_source_record_sha256
        ),
    }


def test_history_population_rejects_missing_storage_capabilities_before_io(tmp_path):
    from fluidmix.memory.history.history import (
        _build_development_history_population,
    )

    destination = tmp_path / "history"
    with pytest.raises(TypeError, match="history storage capabilities"):
        _build_development_history_population(
            source_receipt=tmp_path / "missing-receipt.json",
            expected_receipt_sha256="0" * 64,
            source_database=tmp_path / "missing.sqlite3",
            expected_database_sha256="0" * 64,
            independent_review=tmp_path / "missing-review.json",
            expected_review_sha256="0" * 64,
            owner_id="owner",
            destination=destination,
            evidence_store_factory=None,
            assert_history_citation_fn=None,
        )

    assert not destination.exists()


def test_sixth_run_rejects_missing_storage_capabilities_before_io(tmp_path):
    from fluidmix.memory.history.sixth_development_run import (
        _build_six_run_development_store,
    )

    output = tmp_path / "sixth-run"
    with pytest.raises(TypeError, match="sixth-run storage capabilities"):
        _build_six_run_development_store(
            output_dir=output,
            owner_thread_id="owner",
            code_commit="0" * 40,
            evidence_store_factory=None,
            canonical_source_record_sha256_fn=None,
        )

    assert not output.exists()

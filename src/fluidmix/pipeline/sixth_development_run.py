"""Inject storage capabilities into sixth-development-run preparation."""

from __future__ import annotations

from fluidmix.memory.history.sixth_development_run import (
    _build_six_run_development_store,
)
from fluidmix.storage.sqlite.evidence_store import (
    EvidenceStore,
    canonical_source_record_sha256,
)


def build_six_run_development_store(**kwargs: object) -> dict[str, object]:
    """Build the six-run store while pipeline owns concrete persistence."""

    return _build_six_run_development_store(
        **kwargs,
        evidence_store_factory=EvidenceStore,
        canonical_source_record_sha256_fn=canonical_source_record_sha256,
    )


__all__ = ["build_six_run_development_store"]

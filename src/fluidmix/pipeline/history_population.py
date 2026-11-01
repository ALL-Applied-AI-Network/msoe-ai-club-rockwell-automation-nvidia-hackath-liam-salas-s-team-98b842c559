"""Inject storage capabilities into development-history memory workflows."""

from __future__ import annotations

from fluidmix.memory.history.history import (
    _build_development_history_population,
)
from fluidmix.storage.sqlite.evidence_store import (
    EvidenceStore,
    assert_history_citation,
)


def build_development_history_population(**kwargs: object) -> dict[str, object]:
    """Build history while pipeline owns the concrete persistence dependency."""

    return _build_development_history_population(
        **kwargs,
        evidence_store_factory=EvidenceStore,
        assert_history_citation_fn=assert_history_citation,
    )


__all__ = ["build_development_history_population"]

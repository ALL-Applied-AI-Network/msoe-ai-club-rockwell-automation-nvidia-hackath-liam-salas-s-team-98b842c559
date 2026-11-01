"""Scorer-only construction of the separated RCA-bench hidden oracle artifact.

This module is intentionally not imported by the public input builder or inference
controller.  Callers supply already authorized scoring rows; no Part-A source is opened
here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

from .rca_bench import ORACLE_SCHEMA, validate_oracle


_FROZEN_PROTOCOL_SHA256 = "6a6465048e3839bdcce60c39686f6edf382dfff0b6d23493d0d9b4a4e4d3a4f3"


def build_hidden_oracle(
    plan: Mapping[str, Any],
    oracle_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
    accepted_provenance_by_source_item: Mapping[str, object],
) -> dict[str, Any]:
    """Construct and validate one exact-plan oracle at the scoring boundary."""
    from .rca_bench import canonical_sha256

    if protocol_sha256 != _FROZEN_PROTOCOL_SHA256:
        raise ValueError("oracle protocol hash differs from frozen RCA-bench v1")

    oracle = {
        "schema_version": ORACLE_SCHEMA,
        "plan_sha256": canonical_sha256(plan),
        "protocol_sha256": protocol_sha256,
        "items": copy.deepcopy(list(oracle_rows)),
    }
    return validate_oracle(
        oracle, plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )


__all__ = ["build_hidden_oracle"]

"""Test wrappers around the production reference-bank twin publisher."""

from __future__ import annotations

import hashlib
from pathlib import Path

from twin.cohorts.reference_bank_manifest import (
    publish_reference_bank_twin_artifact_manifest,
)


def published_twin_manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_published_twin_manifest(repo_root: Path, destination: Path) -> Path:
    """Publish once per temporary test root, then reuse the immutable file."""

    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = publish_reference_bank_twin_artifact_manifest(
        repo_root, destination
    )
    assert result["path"] == str(destination.absolute())
    assert result["sha256"] == published_twin_manifest_sha256(destination)
    assert result["artifact_count"] == 18
    return destination

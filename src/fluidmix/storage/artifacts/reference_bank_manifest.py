"""Read hash-bound twin artifacts published for reference-bank planning."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from fluidmix.core.io import load_strict_json_snapshot


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "artifacts",
}
_ARTIFACT_FIELDS = {"relative_path", "payload_sha256", "payload"}


class ReferenceBankManifestError(ValueError):
    """A published reference-bank twin manifest failed closed."""


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReferenceBankManifestError("artifact_payload_invalid") from error
    return hashlib.sha256(encoded).hexdigest()


def _relative_artifact_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReferenceBankManifestError("artifact_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".json"
    ):
        raise ReferenceBankManifestError("artifact_path_invalid")
    return value


def load_reference_bank_twin_artifacts(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, dict[str, object]]:
    """Load payloads from one strict, published, hash-bound twin manifest."""

    try:
        value, snapshot = load_strict_json_snapshot(manifest_path)
    except ValueError as error:
        raise ReferenceBankManifestError("manifest_unreadable") from error
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ReferenceBankManifestError("manifest_hash_invalid")
    if snapshot["sha256"] != expected_sha256:
        raise ReferenceBankManifestError("manifest_hash_mismatch")
    if (
        not isinstance(value, Mapping)
        or set(value) != _MANIFEST_FIELDS
        or value.get("schema_version") != 1
        or value.get("artifact_type")
        != "inn67_reference_bank_twin_artifact_manifest"
        or value.get("status") != "published"
        or not isinstance(value.get("artifacts"), list)
        or not value["artifacts"]
    ):
        raise ReferenceBankManifestError("manifest_schema_invalid")

    artifacts: dict[str, dict[str, object]] = {}
    for row in value["artifacts"]:
        if not isinstance(row, Mapping) or set(row) != _ARTIFACT_FIELDS:
            raise ReferenceBankManifestError("artifact_record_invalid")
        relative = _relative_artifact_path(row.get("relative_path"))
        if relative in artifacts:
            raise ReferenceBankManifestError("duplicate_artifact_path")
        expected = row.get("payload_sha256")
        payload = row.get("payload")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ReferenceBankManifestError("artifact_hash_invalid")
        if not isinstance(payload, Mapping):
            raise ReferenceBankManifestError("artifact_payload_invalid")
        if _canonical_sha256(payload) != expected:
            raise ReferenceBankManifestError("artifact_hash_mismatch")
        artifacts[relative] = deepcopy(dict(payload))
    return artifacts


__all__ = [
    "ReferenceBankManifestError",
    "load_reference_bank_twin_artifacts",
]

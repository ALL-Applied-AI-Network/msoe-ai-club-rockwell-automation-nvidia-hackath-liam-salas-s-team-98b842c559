from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, artifacts: dict[str, object]) -> Path:
    value = {
        "schema_version": 1,
        "artifact_type": "inn67_reference_bank_twin_artifact_manifest",
        "status": "published",
        "artifacts": [
            {
                "relative_path": relative,
                "payload_sha256": _canonical_sha256(payload),
                "payload": payload,
            }
            for relative, payload in artifacts.items()
        ],
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_loads_hash_bound_published_artifacts(tmp_path: Path) -> None:
    from fluidmix.storage.artifacts.reference_bank_manifest import (
        load_reference_bank_twin_artifacts,
    )

    artifacts = {
        "subruns/RUN-A/render-schedule-plan.json": {
            "schema_version": 1,
            "artifact_type": "render-plan",
        },
        "subruns/RUN-A/model-input-plan.json": {"kind": "model-input"},
        "subruns/RUN-A/label-provenance.json": {"kind": "labels"},
    }

    path = _manifest(tmp_path / "published.json", artifacts)
    loaded = load_reference_bank_twin_artifacts(
        path, expected_sha256=_file_sha256(path)
    )

    assert loaded == artifacts
    assert loaded is not artifacts


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (lambda value: value.update(status="draft"), "manifest_schema_invalid"),
        (
            lambda value: value["artifacts"][0].update(payload_sha256="0" * 64),
            "artifact_hash_mismatch",
        ),
        (
            lambda value: value["artifacts"][0].update(relative_path="../escape.json"),
            "artifact_path_invalid",
        ),
        (
            lambda value: value["artifacts"].append(dict(value["artifacts"][0])),
            "duplicate_artifact_path",
        ),
    ],
)
def test_rejects_unpublished_unbound_or_unsafe_artifacts(
    tmp_path: Path, mutation, reason: str
) -> None:
    from fluidmix.storage.artifacts.reference_bank_manifest import (
        ReferenceBankManifestError,
        load_reference_bank_twin_artifacts,
    )

    path = _manifest(
        tmp_path / "published.json",
        {"subruns/RUN-A/render-schedule-plan.json": {"schema_version": 1}},
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ReferenceBankManifestError, match=reason):
        load_reference_bank_twin_artifacts(
            path, expected_sha256=_file_sha256(path)
        )


def test_rejects_a_self_consistent_manifest_without_the_published_hash(
    tmp_path: Path,
) -> None:
    from fluidmix.storage.artifacts.reference_bank_manifest import (
        ReferenceBankManifestError,
        load_reference_bank_twin_artifacts,
    )

    path = _manifest(
        tmp_path / "published.json",
        {"subruns/RUN-A/render-schedule-plan.json": {"schema_version": 1}},
    )
    published_sha = _file_sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value["artifacts"][0]["payload"]
    payload["schema_version"] = 2
    value["artifacts"][0]["payload_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ReferenceBankManifestError, match="manifest_hash_mismatch"):
        load_reference_bank_twin_artifacts(path, expected_sha256=published_sha)

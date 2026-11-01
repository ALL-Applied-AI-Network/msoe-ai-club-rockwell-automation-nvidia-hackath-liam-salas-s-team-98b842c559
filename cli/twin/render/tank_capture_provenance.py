"""Fingerprint all on-disk USD layers and asset dependencies used by a capture."""
import hashlib
import os
from pathlib import Path

from pxr import UsdUtils


def source_dependencies(stage_path):
    """Return sorted relative paths and byte hashes, rejecting missing sources.

    Args:
        stage_path: Root USD file. Paths are relative to its parent directory so
            moving an intact scene package does not invalidate its manifest.
    """
    stage_path = Path(stage_path).resolve()
    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(stage_path))
    if missing:
        raise ValueError("unresolved capture dependencies: " + ", ".join(map(str, missing)))
    paths = {stage_path}
    for layer in layers:
        if not layer.realPath:
            raise ValueError("capture requires file-backed source layers")
        paths.add(Path(layer.realPath).resolve())
    paths.update(Path(asset).resolve() for asset in assets)
    records = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"capture dependency is not a local file: {path}")
        records.append({"path": os.path.relpath(path, stage_path.parent),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sorted(records, key=lambda record: record["path"])


def verify_source_dependencies(stage_path, expected):
    """Reject absent provenance or any changed, added or removed dependency."""
    if not expected or source_dependencies(stage_path) != expected:
        raise ValueError("capture source dependency hashes do not match supplied scene")

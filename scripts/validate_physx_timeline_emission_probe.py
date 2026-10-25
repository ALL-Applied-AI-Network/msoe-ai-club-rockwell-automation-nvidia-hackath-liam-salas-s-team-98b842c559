#!/usr/bin/env python3
"""Outer-validate a timeline emission receipt, artifacts, and full Kit log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.isosurface_geometry import analyze_triangle_mesh_geometry
from fluidmix.timeline_probe import validate_timeline_evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _append_once(failures: list[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def validate_artifacts(
    receipt: dict[str, Any], *, artifact_root: Path
) -> list[str]:
    failures: list[str] = []
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        return ["artifact_index_missing"]
    loaded: dict[str, object] = {}
    for name in ("plan", "callbacks", "particles", "mesh"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            _append_once(failures, f"{name}_artifact_missing")
            continue
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            _append_once(failures, f"{name}_artifact_path_invalid")
            continue
        path = artifact_root / relative
        if not path.is_file():
            _append_once(failures, f"{name}_artifact_unreadable")
            continue
        if _sha256(path) != expected_hash:
            _append_once(failures, f"{name}_artifact_hash_mismatch")
            continue
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            _append_once(failures, f"{name}_artifact_unreadable")

    if loaded.get("plan") != receipt.get("plan"):
        _append_once(failures, "plan_artifact_content_mismatch")
    if loaded.get("callbacks") != receipt.get("callbacks"):
        _append_once(failures, "callback_artifact_content_mismatch")
    particles = loaded.get("particles")
    if not isinstance(particles, dict) or (
        particles.get("initial_positions_m") != receipt.get("initial_positions_m")
        or particles.get("current_positions_m") != receipt.get("current_positions_m")
    ):
        _append_once(failures, "particle_artifact_content_mismatch")

    mesh = loaded.get("mesh")
    if isinstance(mesh, dict):
        points = mesh.get("points_m")
        counts = mesh.get("face_vertex_counts")
        indices = mesh.get("face_vertex_indices")
        if not isinstance(points, list) or not isinstance(counts, list) or not isinstance(indices, list):
            _append_once(failures, "mesh_artifact_geometry_missing")
        else:
            observed = analyze_triangle_mesh_geometry(points, counts, indices)
            if any(mesh.get(key) != value for key, value in observed.items()):
                _append_once(failures, "mesh_artifact_geometry_mismatch")
            if mesh.get("mesh_sha256") != _value_hash([points, counts, indices]):
                _append_once(failures, "mesh_artifact_geometry_hash_mismatch")
            summary = receipt.get("mesh_final")
            if not isinstance(summary, dict) or any(
                summary.get(key) != value
                for key, value in mesh.items()
                if key
                not in {"points_m", "face_vertex_counts", "face_vertex_indices"}
            ):
                _append_once(failures, "mesh_receipt_summary_mismatch")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    failures = validate_timeline_evidence(receipt, log_text=log_text)
    for reason in validate_artifacts(receipt, artifact_root=args.receipt.parent):
        _append_once(failures, reason)
    result = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "receipt_path": str(args.receipt),
        "receipt_sha256": _sha256(args.receipt),
        "log_path": str(args.log),
        "log_sha256": _sha256(args.log),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate one fresh-process PhysX isosurface probe against its outer Kit log."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONDITION_NAMES = (
    "a12_multi_disabled_toggle",
    "nvidia_demo_shared_append_visibility_ablation",
)
_BAD_LOG_MARKERS = (
    "PhysX error: GPU kernel",
    "Cuda context manager error, simulation will be stopped",
    "[Error] [omni.physx.plugin]",
    "Segmentation fault",
    "Traceback (most recent call last)",
)


def _valid_hash(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _append_once(failures: list[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def validate_outer_evidence(
    receipt: dict[str, Any], *, log_text: str, expected_condition: str
) -> list[str]:
    failures: list[str] = []
    if any(marker in log_text for marker in _BAD_LOG_MARKERS):
        _append_once(failures, "physx_gpu_or_cuda_runtime_error")
    if not isinstance(receipt, dict) or receipt.get("status") != "completed":
        _append_once(failures, "run_status_not_completed")
        return failures
    if receipt.get("condition_requested") != expected_condition:
        _append_once(failures, "condition_identity_mismatch")
    if receipt.get("errors") not in ([], None):
        _append_once(failures, "runner_reported_error")
    conditions = receipt.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        _append_once(failures, "condition_denominator_mismatch")
        return failures
    condition = conditions[0]
    if (
        not isinstance(condition, dict)
        or condition.get("name") != expected_condition
        or condition.get("completed") is not True
    ):
        _append_once(failures, "condition_incomplete_or_mismatched")
        return failures
    if condition.get("substantive_isosurface_observed") is not True:
        _append_once(failures, "substantive_isosurface_not_observed")
    first = condition.get("first_substantive_artifact")
    if (
        not isinstance(first, dict)
        or not isinstance(first.get("path"), str)
        or not _valid_hash(first.get("sha256"))
        or not _valid_hash(first.get("mesh_sha256"))
    ):
        _append_once(failures, "first_substantive_artifact_missing")
    snapshots = condition.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        _append_once(failures, "snapshot_evidence_missing")
        return failures
    for snapshot in snapshots:
        particles = snapshot.get("particles", {}) if isinstance(snapshot, dict) else {}
        if (
            particles.get("finite_positions") is not True
            or particles.get("count_matches") is not True
        ):
            _append_once(failures, "particle_output_nonfinite_or_count_mismatch")
    particles = snapshots[-1].get("particles", {})
    movement = particles.get("movement", {})
    maximum = movement.get("maximum_displacement_m")
    if (
        movement.get("motion_observed") is not True
        or movement.get("gravity_moved_down") is not True
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0.0
    ):
        _append_once(failures, "particle_motion_not_observed")
    return failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITION_NAMES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    failures = validate_outer_evidence(
        receipt, log_text=log_text, expected_condition=args.condition
    )
    result = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "condition": args.condition,
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

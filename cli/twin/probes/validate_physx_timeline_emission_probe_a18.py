#!/usr/bin/env python3
"""Outer-validate the layer-corrected A18 receipt, artifacts, and Kit log."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from twin.probes.timeline_probe import validate_corrected_timeline_evidence
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cli.twin.probes.validate_physx_timeline_emission_probe import (
    _append_once,
    _sha256,
    validate_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def validate_log_binding(
    receipt: dict[str, Any], *, log_text: str
) -> list[str]:
    """Bind the raw A18 log markers exactly to the supplied receipt."""

    failures: list[str] = []
    lines = log_text.splitlines()
    plan_marker = "TIMELINE_A18_PLAN " + json.dumps(
        receipt.get("plan"), sort_keys=True
    )
    if lines.count(plan_marker) != 1:
        failures.append("log_plan_marker_mismatch")

    callbacks = receipt.get("callbacks")
    expected_callbacks = (
        [
            "TIMELINE_A18_CALLBACK " + json.dumps(row, sort_keys=True)
            for row in callbacks
        ]
        if isinstance(callbacks, list)
        else []
    )
    observed_callbacks = [
        line for line in lines if line.startswith("TIMELINE_A18_CALLBACK ")
    ]
    if observed_callbacks != expected_callbacks:
        failures.append("log_callback_trace_mismatch")

    mesh = receipt.get("mesh_final")
    observation = {
        "callback_count": len(callbacks) if isinstance(callbacks, list) else None,
        "errors": receipt.get("errors"),
        "movement": receipt.get("movement"),
        "runtime_layer_specs": receipt.get("runtime_layer_specs"),
        "mesh_geometry_valid": (
            mesh.get("mesh_geometry_valid") if isinstance(mesh, dict) else None
        ),
        "mesh_sha256": mesh.get("mesh_sha256") if isinstance(mesh, dict) else None,
    }
    observation_marker = "TIMELINE_A18_OBSERVATION " + json.dumps(
        observation, sort_keys=True
    )
    if lines.count(observation_marker) != 1:
        failures.append("log_observation_marker_mismatch")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    failures = validate_corrected_timeline_evidence(receipt, log_text=log_text)
    for reason in validate_log_binding(receipt, log_text=log_text):
        _append_once(failures, reason)
    for reason in validate_artifacts(receipt, artifact_root=args.receipt.parent):
        _append_once(failures, reason)
    result = {
        "schema_version": 2,
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

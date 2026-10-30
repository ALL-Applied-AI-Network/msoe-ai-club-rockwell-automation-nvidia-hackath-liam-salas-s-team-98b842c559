#!/usr/bin/env python3
"""Publish overlap accounting for the frozen CNN holdout without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cnn_harness_recovery import build_recovery_audit  # noqa: E402


EXPECTED = {
    "training_inventory": "2c68cd25a72aef9975f6cb81faa8c9a574d575accb7f969c20f932578f184d69",
    "frozen_comparison": "f677eab5f9de95958306d4832fb2e23f2fe04e442776ac2c6a5cc23e2ffa94ff",
    "preregister": "3880f50c8b78b7f0fa66e74166241fd966840fb0a2c95d002db0323970e38179",
    "candidate_receipt": "27a4e49553f9fc0d34832e8f8ebc73f11fbf7dfa7e9c39f50353b1fae4b28727",
    "checkpoint": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    "blocked_review": "5be6a7fb484e2514fa5c5b542e3f638c36b4b38e6944c1c40e232a04424da2c2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, role: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    digest = _sha256(resolved)
    if digest != EXPECTED[role]:
        raise ValueError(f"{role} hash mismatch")
    value = json.loads(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be one JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-inventory", type=Path, required=True)
    parser.add_argument("--frozen-comparison", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blocked-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inventory = _load(args.training_inventory, "training_inventory")
    comparison = _load(args.frozen_comparison, "frozen_comparison")
    _load(args.preregister, "preregister")
    candidate = _load(args.candidate_receipt, "candidate_receipt")
    review_path = args.blocked_review.resolve(strict=True)
    if _sha256(review_path) != EXPECTED["blocked_review"]:
        raise ValueError("blocked_review hash mismatch")
    checkpoint = args.checkpoint.resolve(strict=True)
    if _sha256(checkpoint) != EXPECTED["checkpoint"]:
        raise ValueError("checkpoint hash mismatch")
    if (
        candidate.get("checkpoint_sha256") != EXPECTED["checkpoint"]
        or Path(str(candidate.get("checkpoint_path"))).resolve(strict=True) != checkpoint
        or comparison.get("candidate_checkpoint_sha256") != EXPECTED["checkpoint"]
    ):
        raise ValueError("candidate/checkpoint lineage mismatch")

    artifact = build_recovery_audit(inventory, comparison)
    artifact["source_bindings"] = {
        role: {"path": str(getattr(args, role).resolve(strict=True)), "sha256": digest}
        for role, digest in EXPECTED.items()
        if role != "checkpoint"
    }
    artifact["source_bindings"]["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": EXPECTED["checkpoint"],
    }
    artifact["frozen_assets_immutable"] = True
    artifact["liquid_holdout_inference_count"] = 0
    payload = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "scoring-recovery-audit.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "denominators": artifact["denominators"],
                "dice_metrics": artifact["dice_metrics"],
                "status": artifact["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

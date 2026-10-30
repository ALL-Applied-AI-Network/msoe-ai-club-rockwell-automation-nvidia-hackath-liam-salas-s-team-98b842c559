#!/usr/bin/env python3
"""Run one frozen-CNN inference on the authorized same-run empty pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cnn_harness_recovery import (  # noqa: E402
    select_nonidentical_empty_control,
    summarize_replayed_mask,
)
from fluidmix.cnn_predictor import ResNet18UNet, predict_pair  # noqa: E402


EXPECTED = {
    "frozen_comparison": "f677eab5f9de95958306d4832fb2e23f2fe04e442776ac2c6a5cc23e2ffa94ff",
    "preregister": "3880f50c8b78b7f0fa66e74166241fd966840fb0a2c95d002db0323970e38179",
    "candidate_receipt": "27a4e49553f9fc0d34832e8f8ebc73f11fbf7dfa7e9c39f50353b1fae4b28727",
    "checkpoint": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, role: str) -> tuple[Path, dict[str, object]]:
    resolved = path.resolve(strict=True)
    if _sha256(resolved) != EXPECTED[role]:
        raise ValueError(f"{role} hash mismatch")
    value = json.loads(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be one JSON object")
    return resolved, value


def _rgb(path: object, expected: object) -> np.ndarray:
    if not isinstance(path, str) or not isinstance(expected, str):
        raise ValueError("selected empty input binding is invalid")
    resolved = Path(path).resolve(strict=True)
    if _sha256(resolved) != expected:
        raise ValueError("selected empty input hash mismatch")
    with Image.open(resolved) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.shape != (480, 640, 3):
        raise ValueError("selected empty input shape mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-comparison", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparison_path, comparison = _json(args.frozen_comparison, "frozen_comparison")
    prereg_path, prereg = _json(args.preregister, "preregister")
    candidate_path, candidate = _json(args.candidate_receipt, "candidate_receipt")
    checkpoint = args.checkpoint.resolve(strict=True)
    if _sha256(checkpoint) != EXPECTED["checkpoint"]:
        raise ValueError("checkpoint hash mismatch")
    if (
        candidate.get("checkpoint_sha256") != EXPECTED["checkpoint"]
        or Path(str(candidate.get("checkpoint_path"))).resolve(strict=True) != checkpoint
        or comparison.get("candidate_checkpoint_sha256") != EXPECTED["checkpoint"]
    ):
        raise ValueError("candidate/checkpoint lineage mismatch")

    selection = select_nonidentical_empty_control(prereg, comparison)
    current = _rgb(selection["current_path"], selection["current_sha256"])
    reference = _rgb(selection["reference_path"], selection["reference_sha256"])
    if np.array_equal(current, reference):
        raise ValueError("selected empty RGB arrays must be non-identical")

    model = ResNet18UNet(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to("cpu").eval()
    prediction = predict_pair(model, current, reference, device="cpu")
    summary = summarize_replayed_mask(prediction.mask)
    passed = bool(
        prediction.status == "empty_mask_abstain"
        and prediction.surface_row is None
        and summary["mask_pixel_count"] == 0
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact = {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_nonidentical_empty_control",
        "status": "pass" if passed else "fail",
        "negative_gate_passed": passed,
        "execution_device": "cpu",
        "code_commit": head,
        "checkpoint_frozen": True,
        "threshold_frozen": 0.5,
        "empty_only_inference_count": 1,
        "liquid_inference_count": 0,
        "inference_executed": True,
        "input_arrays_identical": False,
        "abstained": prediction.surface_row is None,
        "prediction_status": prediction.status,
        **selection,
        **summary,
        "source_bindings": {
            "frozen_comparison": {
                "path": str(comparison_path),
                "sha256": EXPECTED["frozen_comparison"],
            },
            "preregister": {
                "path": str(prereg_path),
                "sha256": EXPECTED["preregister"],
            },
            "candidate_receipt": {
                "path": str(candidate_path),
                "sha256": EXPECTED["candidate_receipt"],
            },
            "checkpoint": {"path": str(checkpoint), "sha256": EXPECTED["checkpoint"]},
        },
        "claim_boundary": "synthetic V4 empty control only; no physical accuracy claim",
    }
    payload = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "nonidentical-empty-control.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "status": artifact["status"],
                "negative_gate_passed": passed,
                "prediction_status": prediction.status,
                "mask_pixel_count": summary["mask_pixel_count"],
                "selection": selection,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

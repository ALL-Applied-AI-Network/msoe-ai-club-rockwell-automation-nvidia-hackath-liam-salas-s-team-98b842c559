#!/usr/bin/env python3
"""Replay the frozen CNN once on the exact opened pairs and save mask bounds."""

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
    build_recovery_audit,
    score_all_and_cleaned,
    select_nonidentical_empty_control,
    summarize_replayed_mask,
    verify_replay_dice_tolerance,
    visible_height_error_px,
)
from fluidmix.cnn_predictor import ResNet18UNet, predict_pair  # noqa: E402
from fluidmix.empty_reference_rgb_predictor import fillcam_masks  # noqa: E402


EXPECTED = {
    "training_inventory": "2c68cd25a72aef9975f6cb81faa8c9a574d575accb7f969c20f932578f184d69",
    "frozen_comparison": "f677eab5f9de95958306d4832fb2e23f2fe04e442776ac2c6a5cc23e2ffa94ff",
    "preregister": "3880f50c8b78b7f0fa66e74166241fd966840fb0a2c95d002db0323970e38179",
    "candidate_receipt": "27a4e49553f9fc0d34832e8f8ebc73f11fbf7dfa7e9c39f50353b1fae4b28727",
    "checkpoint": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    "empty_control_receipt": "d278c3986e744c498c277e0a44b3d0b6c36b258184c0f8a35b661511bcea1754",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, role: str) -> tuple[Path, dict[str, object]]:
    resolved = path.resolve(strict=True)
    if _sha256(resolved) != EXPECTED[role]:
        raise ValueError(f"{role} hash mismatch")
    value = json.loads(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be one JSON object")
    return resolved, value


def _rgb(path: Path, expected_sha256: str) -> np.ndarray:
    resolved = path.resolve(strict=True)
    if _sha256(resolved) != expected_sha256:
        raise ValueError(f"opened RGB hash mismatch: {resolved}")
    with Image.open(resolved) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.shape != (480, 640, 3):
        raise ValueError(f"opened RGB shape mismatch: {resolved}")
    return value


def _truth(path: Path, expected_sha256: str, roi: np.ndarray, clamp: np.ndarray) -> np.ndarray:
    resolved = path.resolve(strict=True)
    if _sha256(resolved) != expected_sha256:
        raise ValueError(f"opened truth-mask hash mismatch: {resolved}")
    with Image.open(resolved) as image:
        value = np.asarray(image.convert("L")) > 0
    if value.shape != (480, 640):
        raise ValueError(f"opened truth-mask shape mismatch: {resolved}")
    return value & roi & ~clamp


def _dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    denominator = int(predicted.sum()) + int(truth.sum())
    return (
        1.0
        if denominator == 0
        else 2.0 * int(np.count_nonzero(predicted & truth)) / denominator
    )


def _one(opened: list[object], case_id: str, role: str) -> dict[str, str]:
    matches = [
        row
        for row in opened
        if isinstance(row, dict)
        and row.get("case_id") == case_id
        and row.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role} for {case_id}")
    row = matches[0]
    if not all(isinstance(row.get(key), str) and row[key] for key in ("path", "sha256")):
        raise ValueError(f"invalid opened-file row for {case_id}/{role}")
    return {"path": str(row["path"]), "sha256": str(row["sha256"])}


def _many(opened: list[object], case_id: str, role: str) -> list[dict[str, str]]:
    result = [
        {"path": str(row["path"]), "sha256": str(row["sha256"])}
        for row in opened
        if isinstance(row, dict)
        and row.get("case_id") == case_id
        and row.get("role") == role
        and isinstance(row.get("path"), str)
        and isinstance(row.get("sha256"), str)
    ]
    if not result:
        raise ValueError(f"missing {role} rows for {case_id}")
    return result


def replay(args: argparse.Namespace) -> dict[str, object]:
    inventory_path, inventory = _load(args.training_inventory, "training_inventory")
    comparison_path, comparison = _load(args.frozen_comparison, "frozen_comparison")
    prereg_path, prereg = _load(args.preregister, "preregister")
    candidate_path, candidate = _load(args.candidate_receipt, "candidate_receipt")
    checkpoint = args.checkpoint.resolve(strict=True)
    if _sha256(checkpoint) != EXPECTED["checkpoint"]:
        raise ValueError("checkpoint hash mismatch")
    if (
        candidate.get("checkpoint_sha256") != EXPECTED["checkpoint"]
        or Path(str(candidate.get("checkpoint_path"))).resolve(strict=True) != checkpoint
        or comparison.get("candidate_checkpoint_sha256") != EXPECTED["checkpoint"]
    ):
        raise ValueError("candidate/checkpoint lineage mismatch")

    opened = comparison.get("opened_files")
    runs = prereg.get("runs")
    frozen_rows = comparison.get("cnn_rows")
    approach_rows = comparison.get("approach_a_rows")
    if not all(isinstance(value, list) for value in (opened, runs, frozen_rows, approach_rows)):
        raise ValueError("frozen comparison/preregister rows are invalid")

    model = ResNet18UNet(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to("cpu").eval()
    roi, clamp = fillcam_masks()
    replayed: list[dict[str, object]] = []

    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("preregistered run must be an object")
        run_id = str(run.get("run_id"))
        case_ids = run.get("case_ids")
        if not isinstance(case_ids, list):
            raise ValueError("preregistered run is missing case_ids")
        for case_id_value in case_ids:
            case_id = str(case_id_value)
            reference_row = _one(opened, case_id, "reference")
            stage_rows = _many(opened, case_id, "stage_rgb")
            truth_rows = _many(opened, case_id, "simulator_mask")
            if len(stage_rows) != len(truth_rows):
                raise ValueError(f"stage/truth denominator mismatch for {case_id}")
            reference = _rgb(Path(reference_row["path"]), reference_row["sha256"])
            for ordinal, (stage_row, truth_row) in enumerate(
                zip(stage_rows, truth_rows, strict=True), start=1
            ):
                current = _rgb(Path(stage_row["path"]), stage_row["sha256"])
                truth = _truth(Path(truth_row["path"]), truth_row["sha256"], roi, clamp)
                prediction = predict_pair(model, current, reference, device="cpu")
                row_id = f"{case_id}:stage-{ordinal}"
                predicted_summary = summarize_replayed_mask(prediction.mask)
                truth_summary = summarize_replayed_mask(truth)
                replayed.append(
                    {
                        "case_id": row_id,
                        "run_id": run_id,
                        "control_kind": "positive",
                        "low_fill_first_dispense": ordinal == 1,
                        "dice": _dice(prediction.mask, truth),
                        "visible_height_error_px": visible_height_error_px(
                            prediction.mask, truth
                        ),
                        "prediction": predicted_summary,
                        "truth_bounds": {
                            key: truth_summary[key]
                            for key in (
                                "first_occupied_row",
                                "last_occupied_row",
                                "visible_height_px",
                            )
                        },
                        "input_bindings": {
                            "current_rgb_sha256": stage_row["sha256"],
                            "reference_rgb_sha256": reference_row["sha256"],
                            "truth_mask_sha256": truth_row["sha256"],
                        },
                    }
                )

    if len(replayed) != 21:
        raise ValueError(f"authorized replay denominator is 21, got {len(replayed)}")
    determinism = verify_replay_dice_tolerance(frozen_rows, replayed)

    empty_receipt_path, empty_receipt = _load(
        args.empty_control_receipt, "empty_control_receipt"
    )
    empty_selection = select_nonidentical_empty_control(prereg, comparison)
    if any(empty_receipt.get(key) != value for key, value in empty_selection.items()):
        raise ValueError("retained empty-control selection binding mismatch")
    if (
        empty_receipt.get("status") != "pass"
        or empty_receipt.get("negative_gate_passed") is not True
        or empty_receipt.get("prediction_status") != "empty_mask_abstain"
        or empty_receipt.get("mask_pixel_count") != 0
        or empty_receipt.get("abstained") is not True
    ):
        raise ValueError("retained empty-control receipt did not pass")
    empty_control = {
        **empty_receipt,
        "control_kind": "empty_vial_nonidentical",
        "inference_executed": True,
        "retained_receipt_path": str(empty_receipt_path),
        "retained_receipt_sha256": EXPECTED["empty_control_receipt"],
    }

    accounting = build_recovery_audit(inventory, comparison)
    excluded = set(accounting["overlap_exclusions"]["excluded_positive_row_ids"])
    scored = score_all_and_cleaned(
        replayed,
        approach_rows,
        excluded_positive_row_ids=excluded,
        empty_control=empty_control,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_deterministic_bounds_replay",
        "status": (
            "swap_review_ready"
            if scored["cleaned"]["all_gates_pass"] is True
            else "retain_approach_a"
        ),
        "execution_device": "cpu",
        "inference_precision": str(next(model.parameters()).dtype),
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "torch_version": torch.__version__,
        "code_commit": head,
        "checkpoint_frozen": True,
        "threshold_frozen": 0.5,
        "liquid_replay_scope": "exact_same_21_already_opened_pairs",
        "liquid_inference_count": 21,
        "empty_only_inference_count": 0,
        "retained_empty_control_inference_count": 1,
        "new_liquid_pair_count": 0,
        "determinism_gate": determinism,
        "empty_control": empty_control,
        "overlap_exclusions": accounting["overlap_exclusions"],
        "scoring": scored,
        "replayed_rows": replayed,
        "source_bindings": {
            "training_inventory": {
                "path": str(inventory_path),
                "sha256": EXPECTED["training_inventory"],
            },
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
            "empty_control_receipt": {
                "path": str(empty_receipt_path),
                "sha256": EXPECTED["empty_control_receipt"],
            },
        },
        "claim_boundary": "synthetic V4 holdout only; no real-line or physical accuracy claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-inventory", type=Path, required=True)
    parser.add_argument("--frozen-comparison", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--empty-control-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args)
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "deterministic-bounds-replay.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "status": result["status"],
                "determinism_gate": result["determinism_gate"],
                "empty_control": result["empty_control"],
                "scoring": result["scoring"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

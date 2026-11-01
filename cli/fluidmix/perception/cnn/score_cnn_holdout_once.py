#!/usr/bin/env python3
"""One no-clobber CNN versus Approach-A pass on the preregistered V4 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.perception.cnn.cnn_predictor import (  # noqa: E402
    ResNet18UNet,
    predict_pair,
    score_holdout_comparison,
)
from fluidmix.perception.cnn.cnn_harness_recovery import visible_height_error_px  # noqa: E402
from fluidmix.perception.baselines.empty_reference_rgb_predictor import (  # noqa: E402
    fillcam_masks,
    predict_visible_liquid,
)


EXPECTED_PREREG_SHA256 = "3880f50c8b78b7f0fa66e74166241fd966840fb0a2c95d002db0323970e38179"
_EVALUATION_NUISANCE = re.compile(r"(?P<run>RUN-V4-EVALUATION-\d{2})-NUISANCE-V\d+\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    denominator = int(predicted.sum()) + int(truth.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.count_nonzero(predicted & truth)) / denominator


def _metrics(prediction, truth: np.ndarray) -> tuple[float, float]:
    return _dice(prediction.mask, truth), visible_height_error_px(prediction.mask, truth)


def score(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = args.preregister.resolve()
    if _sha256(prereg_path) != EXPECTED_PREREG_SHA256:
        raise ValueError("preregistered holdout hash mismatch")
    prereg = _json(prereg_path)
    plan = _json(args.v4_plan.resolve())
    candidate = _json(args.candidate_receipt.resolve())
    checkpoint = Path(str(candidate["checkpoint_path"]))
    if _sha256(checkpoint) != candidate.get("checkpoint_sha256"):
        raise ValueError("candidate checkpoint hash mismatch")
    if candidate.get("status") != "trained_candidate_not_heldout_scored":
        raise ValueError("candidate was not frozen before scoring")

    plan_runs = {str(run["run_id"]): run for run in plan["runs"] if isinstance(run, dict)}
    if set(prereg["run_ids"]) != set(plan_runs) & set(prereg["run_ids"]):
        raise ValueError("preregistered runs missing from V4 plan")
    model = ResNet18UNet(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(args.device).eval()
    roi, clamp = fillcam_masks()
    cnn_rows: list[dict[str, object]] = []
    approach_rows: list[dict[str, object]] = []
    opened: list[dict[str, str]] = []

    for run in prereg["runs"]:
        run_id = str(run["run_id"])
        if plan_runs[run_id].get("split_role") != "evaluation":
            raise ValueError("holdout run is not evaluation")
        if list(plan_runs[run_id].get("case_ids", [])) != list(run["case_ids"]):
            raise ValueError("whole-run case binding mismatch")
        for case_id in run["case_ids"]:
            matches = list(args.v4_shards.glob(f"*/cases/{case_id}/app-output/receipt.json"))
            if not matches:
                accepted = args.accepted_root / "cases" / str(case_id) / "app-output" / "receipt.json"
                matches = [accepted] if accepted.is_file() else []
            if len(matches) != 1:
                raise ValueError(f"{case_id} must resolve to exactly one receipt")
            receipt_path = matches[0]
            receipt = _json(receipt_path)
            nuisance = receipt.get("nuisance_runtime")
            match = _EVALUATION_NUISANCE.fullmatch(str(nuisance.get("case_id") if isinstance(nuisance, dict) else None))
            if match is None or match.group("run") != run_id:
                raise ValueError("heldout receipt run identity mismatch")
            parity = receipt["pre_pour_parity"]
            reference_path = receipt_path.parent / parity["f2_prefill_path"]
            empty_path = receipt_path.parent / parity["reference_path"]
            if _sha256(reference_path) != parity["f2_prefill_sha256"] or _sha256(empty_path) != parity["reference_sha256"]:
                raise ValueError("heldout empty reference hash mismatch")
            reference = _rgb(reference_path)
            empty = _rgb(empty_path)
            for role, path in (("reference", reference_path), ("empty", empty_path)):
                opened.append({"case_id": str(case_id), "role": role, "path": str(path.resolve()), "sha256": _sha256(path)})
            identical = predict_pair(model, reference, reference.copy(), device=args.device)
            empty_prediction = predict_pair(model, empty, reference, device=args.device)
            if not any(row.get("control_kind") == "identical" for row in cnn_rows):
                cnn_rows.append({"case_id": "identical-control", "dice": 1.0, "visible_height_error_px": None, "control_kind": "identical", "abstained": identical.surface_row is None, "mask_pixel_count": identical.changed_pixel_count})
            if not any(row.get("control_kind") == "empty_vial" for row in cnn_rows):
                cnn_rows.append({"case_id": "empty-control", "dice": 1.0, "visible_height_error_px": None, "control_kind": "empty_vial", "abstained": empty_prediction.surface_row is None, "mask_pixel_count": empty_prediction.changed_pixel_count})

            semantics = receipt["semantics"]
            if semantics.get("semantic_gate_passed") is not True:
                raise ValueError("heldout semantic gate failed")
            for stage_ordinal, pair in enumerate(semantics["pairs"], start=1):
                current_path = receipt_path.parent / pair["rgb_path"]
                mask_path = receipt_path.parent / pair["mask_path"]
                if _sha256(current_path) != pair["rgb_sha256"] or _sha256(mask_path) != pair["mask_sha256"]:
                    raise ValueError("heldout stage binding hash mismatch")
                current = _rgb(current_path)
                truth = _binary_mask(mask_path) & roi & ~clamp
                cnn_prediction = predict_pair(model, current, reference, device=args.device)
                approach_prediction = predict_visible_liquid(current, reference, programmed_ml=None)
                cnn_dice, cnn_height = _metrics(cnn_prediction, truth)
                approach_dice, approach_height = _metrics(approach_prediction, truth)
                row_id = f"{case_id}:stage-{stage_ordinal}"
                cnn_rows.append({"case_id": row_id, "run_id": run_id, "dice": cnn_dice, "visible_height_error_px": cnn_height, "control_kind": "positive", "low_fill_first_dispense": stage_ordinal == 1})
                approach_rows.append({"case_id": row_id, "run_id": run_id, "dice": approach_dice, "visible_height_error_px": approach_height, "control_kind": "positive", "low_fill_first_dispense": stage_ordinal == 1})
                for role, path in (("stage_rgb", current_path), ("simulator_mask", mask_path)):
                    opened.append({"case_id": str(case_id), "role": role, "path": str(path.resolve()), "sha256": _sha256(path)})

    comparison = score_holdout_comparison(cnn_rows, approach_rows)
    return {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_holdout_comparison",
        "status": "scored_once",
        "preregister_sha256": EXPECTED_PREREG_SHA256,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_source_commit": candidate["source_commit"],
        "comparison": comparison,
        "cnn_rows": cnn_rows,
        "approach_a_rows": approach_rows,
        "opened_files": opened,
        "denominators": {"runs": 3, "cases": 9, "positive_stage_rows": len(approach_rows), "controls": 2, "opened_files": len(opened)},
        "claim_boundary": "synthetic V4 holdout only; no real-line or physical accuracy claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--v4-plan", type=Path, required=True)
    parser.add_argument("--v4-shards", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = score(args)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = args.output_dir / "comparison.json"
    output.write_bytes(payload)
    print(json.dumps({"output": str(output.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "decision": result["comparison"]["decision"], "metrics": result["comparison"]["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

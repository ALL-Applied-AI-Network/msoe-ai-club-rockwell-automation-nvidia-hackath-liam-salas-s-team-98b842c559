#!/usr/bin/env python3
"""Build the V3/V4 train-only six-channel CNN inventory from receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.perception.cnn.cnn_training import validate_training_inventory  # noqa: E402


DEFAULT_V3 = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "rgb-mask-predictor-01a18128/validation/"
    "prediction-locators-train-validation.json"
)
DEFAULT_V4 = Path(
    "/home/hackathon5/liamreporoot/f2-varied-v4-package-660bc43/"
    "render-schedule-plan.json"
)
DEFAULT_V4_SHARDS = Path(
    "/home/hackathon5/liamreporoot/f2-varied-v4-shards-660bc43"
)
_NUISANCE = re.compile(r"(?P<run>RUN(?:-V4)?-TRAIN-\d{2})-NUISANCE-V\d+\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _receipt_run_id(receipt: dict[str, object]) -> str:
    nuisance = receipt.get("nuisance_runtime")
    if not isinstance(nuisance, dict):
        raise ValueError("receipt nuisance runtime missing")
    match = _NUISANCE.fullmatch(str(nuisance.get("case_id")))
    if match is None:
        raise ValueError("receipt train run identity invalid")
    return match.group("run")


def _reference(receipt_path: Path, receipt: dict[str, object]) -> tuple[str, str]:
    parity = receipt.get("pre_pour_parity")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise ValueError("receipt pre-pour parity missing or failed")
    relative = parity.get("f2_prefill_path")
    digest = parity.get("f2_prefill_sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError("receipt pre-pour binding invalid")
    return str((receipt_path.parent / relative).resolve()), digest


def _pair_rows(
    *,
    receipt_path: Path,
    receipt: dict[str, object],
    run_id: str,
    case_id: str,
    selected_rgb_sha256: set[str] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    semantics = receipt.get("semantics")
    if not isinstance(semantics, dict) or semantics.get("semantic_gate_passed") is not True:
        raise ValueError(f"{case_id} simulator semantic gate did not pass")
    pairs = semantics.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{case_id} simulator mask pairs missing")
    reference_path, reference_sha = _reference(receipt_path, receipt)
    positives: list[dict[str, object]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"{case_id} simulator pair invalid")
        rgb_sha = pair.get("rgb_sha256")
        if selected_rgb_sha256 is not None and rgb_sha not in selected_rgb_sha256:
            continue
        rgb_relative = pair.get("rgb_path")
        mask_relative = pair.get("mask_path")
        if not all(isinstance(value, str) for value in (rgb_relative, rgb_sha, mask_relative, pair.get("mask_sha256"))):
            raise ValueError(f"{case_id} simulator pair binding invalid")
        positives.append(
            {
                "sample_id": f"{case_id}:{rgb_sha}",
                "run_id": run_id,
                "case_id": case_id,
                "split_role": "train",
                "data_origin": "twin",
                "current_rgb_path": str((receipt_path.parent / rgb_relative).resolve()),
                "current_rgb_sha256": rgb_sha,
                "reference_rgb_path": reference_path,
                "reference_rgb_sha256": reference_sha,
                "label_kind": "simulator_mask",
                "mask_path": str((receipt_path.parent / mask_relative).resolve()),
                "mask_sha256": pair["mask_sha256"],
                "label_source": "simulator_semantic_mask",
            }
        )
    if selected_rgb_sha256 is not None and {row["current_rgb_sha256"] for row in positives} != selected_rgb_sha256:
        raise ValueError(f"{case_id} selected V3 RGB set did not match receipt")
    negative = {
        "sample_id": f"{case_id}:empty-prepour",
        "run_id": run_id,
        "case_id": case_id,
        "split_role": "train",
        "data_origin": "twin",
        "current_rgb_path": reference_path,
        "current_rgb_sha256": reference_sha,
        "reference_rgb_path": reference_path,
        "reference_rgb_sha256": reference_sha,
        "label_kind": "empty_negative",
        "label_source": "same_run_empty_prepour",
    }
    return positives, negative


def build_inventory(v3_locator_path: Path, v4_plan_path: Path, v4_shards: Path) -> dict[str, object]:
    v3 = _json(v3_locator_path)
    selected_by_case: dict[str, set[str]] = {}
    receipt_by_case: dict[str, Path] = {}
    for locator in v3.values():
        if not isinstance(locator, dict) or locator.get("split_role") != "train":
            continue
        rgb_path = Path(str(locator.get("rgb_path"))).resolve()
        case_root = next((parent for parent in rgb_path.parents if parent.name.startswith("CASE_")), None)
        if case_root is None:
            raise ValueError("V3 train RGB is not under a case root")
        selected_by_case.setdefault(case_root.name, set()).add(str(locator.get("rgb_sha256")))
        receipt_by_case[case_root.name] = case_root / "app-output" / "receipt.json"

    rows: list[dict[str, object]] = []
    for case_id in sorted(selected_by_case):
        receipt_path = receipt_by_case[case_id]
        receipt = _json(receipt_path)
        positives, negative = _pair_rows(
            receipt_path=receipt_path,
            receipt=receipt,
            run_id=_receipt_run_id(receipt),
            case_id=case_id,
            selected_rgb_sha256=selected_by_case[case_id],
        )
        rows.extend(positives)
        rows.append(negative)

    v4 = _json(v4_plan_path)
    train_runs = {
        str(run["run_id"]): list(run["case_ids"])
        for run in v4.get("runs", [])
        if isinstance(run, dict) and run.get("split_role") == "train"
    }
    if len(train_runs) != 4:
        raise ValueError("V4 train run denominator invalid")
    for run_id, case_ids in train_runs.items():
        for case_id in case_ids:
            matches = list(v4_shards.glob(f"*/cases/{case_id}/app-output/receipt.json"))
            if len(matches) != 1:
                raise ValueError(f"{case_id} must resolve to exactly one accepted receipt")
            receipt = _json(matches[0])
            if _receipt_run_id(receipt) != run_id:
                raise ValueError(f"{case_id} receipt run mismatch")
            positives, negative = _pair_rows(
                receipt_path=matches[0],
                receipt=receipt,
                run_id=run_id,
                case_id=str(case_id),
                selected_rgb_sha256=None,
            )
            rows.extend(positives)
            rows.append(negative)

    artifact = {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_training_inventory",
        "status": "train_only",
        "sources": {
            "v3_train_locator_path": str(v3_locator_path.resolve()),
            "v3_train_locator_sha256": _sha256(v3_locator_path),
            "v4_source_plan_path": str(v4_plan_path.resolve()),
            "v4_source_plan_sha256": _sha256(v4_plan_path),
        },
        "input_channels": ["current_r", "current_g", "current_b", "same_run_empty_r", "same_run_empty_g", "same_run_empty_b"],
        "label_policy": "simulator semantic masks for positives; same-run empty pre-pour negatives only",
        "samples": rows,
        "denominators": {
            "samples": len(rows),
            "positive_masks": sum(row["label_kind"] == "simulator_mask" for row in rows),
            "empty_negatives": sum(row["label_kind"] == "empty_negative" for row in rows),
            "runs": len({row["run_id"] for row in rows}),
            "cases": len({row["case_id"] for row in rows}),
        },
    }
    validate_training_inventory(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-locators", type=Path, default=DEFAULT_V3)
    parser.add_argument("--v4-plan", type=Path, default=DEFAULT_V4)
    parser.add_argument("--v4-shards", type=Path, default=DEFAULT_V4_SHARDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_inventory(args.v3_locators, args.v4_plan, args.v4_shards)
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "denominators": artifact["denominators"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

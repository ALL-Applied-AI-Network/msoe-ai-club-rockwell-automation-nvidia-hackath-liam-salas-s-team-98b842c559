#!/usr/bin/env python3
"""Evaluate one revised FillCam model on normal and darkened synthetic validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
import evaluate_fill_level_revision as shared  # noqa: E402
from fluidmix import fill_level_revision_eval as revision_eval  # noqa: E402
from fluidmix import fill_level_surface_eval as surface_eval  # noqa: E402
from fluidmix import fill_level_training as training  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--exception-review", type=Path, required=True)
    parser.add_argument("--revised-training-summary", type=Path, required=True)
    parser.add_argument("--revised-training-summary-sha256", required=True)
    parser.add_argument("--revised-checkpoint", type=Path, required=True)
    parser.add_argument("--revised-checkpoint-sha256", required=True)
    parser.add_argument("--revised-rule-summary", type=Path, required=True)
    parser.add_argument("--revised-rule-summary-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def _preflight(args):
    """Bind one revised model and rule to one explicit train/validation audit."""
    audit = training.validate_exception_review(
        args.batch_summary, args.exception_review
    )
    model = revision_eval.validate_training_artifact(
        args.revised_training_summary,
        args.revised_training_summary_sha256,
        args.revised_checkpoint,
        args.revised_checkpoint_sha256,
        audit,
        revision="revised",
    )
    rule = revision_eval.validate_rule_artifact(
        args.revised_rule_summary,
        args.revised_rule_summary_sha256,
        model,
        audit,
        revision="revised",
    )
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    return audit, {"model": model, "rule": rule}


def _build_summary(
    *,
    audit,
    evidence,
    samples,
    preflight,
    metrics,
    darkened_hsv,
    gate_value,
    gate_error,
    device,
    torch_version,
    gpu,
    batch_size,
    workers,
    elapsed_seconds,
):
    """Build the full revised-only evidence document without any baseline binding."""
    return {
        "schema_version": 1,
        "purpose": (
            "Step 8 Revision 2 revised-model normal and darkened synthetic "
            "validation"
        ),
        "status": "complete" if gate_error is None else "gate_failed",
        "evidence_label": "measured_on_synthetic_validation_only",
        "model_revision": "revised",
        "dataset": {
            "role": "val",
            "frame_count": len(samples),
            "seed_range": [int(samples[0].seed), int(samples[-1].seed)],
            "preflight": preflight,
            "variants": ["normal", "darkened"],
            "train_manifests_opened": 0,
            "train_images_loaded": 0,
            "test_manifests_opened": 0,
            "test_images_loaded": 0,
            "real_frames_loaded": 0,
        },
        "darkened_validation": revision_eval.darkened_validation_report(
            darkened_hsv
        ),
        "gate": {
            "metric": "revised normal validation visible-surface p95 rows",
            "maximum_rows": revision_eval.NORMAL_VISIBLE_P95_GATE_ROWS,
            "observed_rows": gate_value,
            "passed": gate_error is None,
            "error": gate_error,
        },
        "model": {
            **evidence["model"],
            "rule_summary_sha256": evidence["rule"]["summary_sha256"],
            "rule_config": evidence["rule"]["rule_config"],
            "metrics": metrics,
        },
        "inputs": {
            "batch_summary_sha256": audit["batch_summary_sha256"],
            "exception_review_sha256": audit["review_sha256"],
            "evaluator_script_sha256": training.file_sha256(__file__),
            "shared_evaluator_script_sha256": training.file_sha256(shared.__file__),
            "evaluator_module_sha256": training.file_sha256(
                revision_eval.__file__
            ),
            "surface_rule_module_sha256": training.file_sha256(
                surface_eval.__file__
            ),
            "current_training_module_sha256": training.file_sha256(
                training.__file__
            ),
        },
        "environment": {
            "device": str(device),
            "torch": torch_version,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "gpu": gpu,
        },
        "execution": {
            "batch_size": int(batch_size),
            "workers": int(workers),
            "shuffle": False,
        },
        "elapsed_seconds": elapsed_seconds,
    }


def run(args):
    # Authority and exact revised artifact/rule bindings precede output and dataset access.
    audit, evidence = _preflight(args)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    shared._append_state(state_path, "RUNNING")
    started = time.time()
    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable")
        device = torch.device(args.device)
        training.seed_everything(revision_eval.DARK_VALIDATION_SEED)
        samples, metadata = surface_eval.discover_validation_frames(
            args.dataset_root, audit
        )
        preflight = training.verify_sample_files(samples)
        if evidence["rule"]["ordered_hash_ledger_sha256"] != preflight[
            "ordered_hash_ledger_sha256"
        ]:
            raise ValueError(
                "revised rule validation ledger does not match the evaluation dataset"
            )

        dataset = shared._make_dataset(torch, samples)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=training.seed_worker,
            persistent_workers=False,
        )
        model = shared._load_model(
            torch,
            args.revised_checkpoint,
            evidence["model"],
            device,
        )
        all_metrics, darkened_hsv = shared._infer(
            torch,
            loader,
            {"revised": model},
            metadata,
            {"revised": evidence["rule"]["rule_config"]},
            device,
        )
        metrics = all_metrics["revised"]
        if metrics["normal"]["surface"] != evidence["rule"]["aggregate"]:
            raise ValueError(
                "revised normal surface metrics do not reproduce the frozen rule summary"
            )
        for variant in ("normal", "darkened"):
            surface_eval.validate_case_counts(
                args.batch_summary,
                metrics[variant]["surface"],
            )

        gate_error = None
        try:
            gate_value = revision_eval.enforce_normal_validation_gate(
                metrics["normal"]
            )
        except ValueError as error:
            gate_value = None
            gate_error = str(error)
        gpu = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        )
        summary = _build_summary(
            audit=audit,
            evidence=evidence,
            samples=samples,
            preflight=preflight,
            metrics=metrics,
            darkened_hsv=darkened_hsv,
            gate_value=gate_value,
            gate_error=gate_error,
            device=device,
            torch_version=torch.__version__,
            gpu=gpu,
            batch_size=args.batch_size,
            workers=args.workers,
            elapsed_seconds=time.time() - started,
        )
        summary_path = args.output_root / "summary.json"
        shared._json_dump(summary_path, summary)
        compact = {
            **summary,
            "model": {
                **{
                    key: value
                    for key, value in summary["model"].items()
                    if key != "metrics"
                },
                "metrics": revision_eval.compact_model_metrics(
                    summary["model"]["metrics"]
                ),
            },
            "full_summary_sha256": training.file_sha256(summary_path),
        }
        compact_path = args.output_root / "compact-summary.json"
        shared._json_dump(compact_path, compact)
        terminal = "COMPLETE" if gate_error is None else "GATE_FAILED"
        shared._append_state(
            state_path,
            terminal,
            compact_summary_sha256=training.file_sha256(compact_path),
        )
        return compact
    except Exception as error:
        shared._json_dump(
            args.output_root / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        shared._append_state(state_path, "FAILED")
        raise


def main():
    result = run(_parser().parse_args())
    print(
        json.dumps(
            {"status": result["status"], "gate": result["gate"]},
            sort_keys=True,
            allow_nan=False,
        )
    )
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

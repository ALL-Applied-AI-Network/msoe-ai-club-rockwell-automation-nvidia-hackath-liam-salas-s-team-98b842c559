#!/usr/bin/env python3
"""Evaluate the frozen FillCam surface-row rule on synthetic validation only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fluidmix import fill_level_surface_eval as surface_eval  # noqa: E402
from fluidmix import fill_level_training as training  # noqa: E402


def _json_dump(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_state(path, status):
    entry = {
        "status": str(status),
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--exception-review", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--column-first", type=int,
        default=surface_eval.DEFAULT_MEASUREMENT_COLUMNS_FULL[0],
    )
    parser.add_argument(
        "--column-last", type=int,
        default=surface_eval.DEFAULT_MEASUREMENT_COLUMNS_FULL[1],
    )
    parser.add_argument(
        "--row-first",
        type=int,
        help=(
            "optional full-frame inclusive first search row; omit with --row-last "
            "to derive the vial-interior interval from validation label outlines"
        ),
    )
    parser.add_argument(
        "--row-last",
        type=int,
        help=(
            "optional full-frame exclusive last search row; omit with --row-first "
            "to derive the vial-interior interval from validation label outlines"
        ),
    )
    parser.add_argument(
        "--min-liquid-pixels", type=int,
        default=surface_eval.DEFAULT_MIN_LIQUID_PIXELS_PER_ROW,
    )
    parser.add_argument(
        "--min-blocked-pixels", type=int,
        default=surface_eval.DEFAULT_MIN_BLOCKED_PIXELS_PER_ROW,
    )
    return parser


def _rule_config(args, measurement_rows_full, row_bounds_evidence):
    config = surface_eval.frozen_rule_config(
        measurement_columns_full=(args.column_first, args.column_last),
        measurement_rows_full=measurement_rows_full,
        min_liquid_pixels=args.min_liquid_pixels,
        min_blocked_pixels=args.min_blocked_pixels,
        row_bounds_evidence=row_bounds_evidence,
    )
    # Validate the rule before any evidence path is created.
    height = training.CROP_XYXY[3] - training.CROP_XYXY[1]
    width = training.CROP_XYXY[2] - training.CROP_XYXY[0]
    surface_eval.surface_prediction(
        np.zeros((height, width), dtype=np.uint8),
        measurement_columns_full=(args.column_first, args.column_last),
        measurement_rows_full=measurement_rows_full,
        min_liquid_pixels=args.min_liquid_pixels,
        min_blocked_pixels=args.min_blocked_pixels,
    )
    return config


def _validation_label_masks(samples):
    x0, y0, x1, y1 = training.CROP_XYXY
    masks = []
    for sample in samples:
        with Image.open(sample.mask_path) as image:
            mask = np.asarray(image)
        if mask.shape != (480, 640):
            raise ValueError(
                f"validation seed {sample.seed} mask must be 640x480; "
                f"got {mask.shape[::-1]}"
            )
        masks.append(np.ascontiguousarray(mask[y0:y1, x0:x1]))
    return masks


def run(args):
    # Authority, model binding and rule validation precede dataset access/output.
    audit = training.validate_exception_review(args.batch_summary, args.exception_review)
    artifacts = surface_eval.validate_training_artifacts(
        args.training_summary, args.checkpoint, audit
    )
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if (args.row_first is None) != (args.row_last is None):
        raise ValueError("--row-first and --row-last must be supplied together")

    samples, metadata = surface_eval.discover_validation_frames(
        args.dataset_root, audit
    )
    preflight = training.verify_sample_files(samples)
    if args.row_first is None:
        row_bounds_evidence = surface_eval.derive_vial_interior_rows(
            _validation_label_masks(samples),
            measurement_columns_full=(args.column_first, args.column_last),
        )
        row_bounds_evidence["role"] = "val"
        row_bounds_evidence["ordered_hash_ledger_sha256"] = preflight[
            "ordered_hash_ledger_sha256"
        ]
        measurement_rows_full = tuple(
            row_bounds_evidence["measurement_rows_full_half_open"]
        )
    else:
        measurement_rows_full = (args.row_first, args.row_last)
        row_bounds_evidence = {
            "measurement_rows_full_half_open": list(measurement_rows_full),
            "selection_source": "explicit_cli_for_comparison",
            "frame_count": len(samples),
            "role": "val",
            "ordered_hash_ledger_sha256": preflight[
                "ordered_hash_ledger_sha256"
            ],
        }
    rule_config = _rule_config(
        args, measurement_rows_full, row_bounds_evidence
    )

    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    _append_state(state_path, "RUNNING")
    started = time.time()
    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        training.seed_everything(artifacts["training_seed"])

        dataset = training.make_torch_dataset(
            samples, augment=False, seed=artifacts["training_seed"]
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=training.seed_worker,
            persistent_workers=False,
        )

        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary")
        if int(checkpoint.get("epoch", -1)) != artifacts["best_epoch"]:
            raise ValueError("checkpoint epoch does not match training summary")
        checkpoint_config = checkpoint.get("config", {})
        if int(checkpoint_config.get("base_channels", -1)) != artifacts["base_channels"]:
            raise ValueError("checkpoint model width does not match training summary")
        model = training.build_unet(base_channels=artifacts["base_channels"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        per_frame = []
        with torch.inference_mode():
            for images, target_masks, seeds in loader:
                predictions = model(images.to(device, non_blocking=True)).argmax(dim=1)
                predictions = predictions.cpu().numpy()
                truths = target_masks.numpy()
                for index, raw_seed in enumerate(seeds.tolist()):
                    seed = int(raw_seed)
                    per_frame.append(
                        surface_eval.frame_result(
                            metadata[seed], truths[index], predictions[index],
                            rule_config=rule_config,
                        )
                    )
        per_frame.sort(key=lambda item: item["seed"])
        if len(per_frame) != audit["eligible_counts"]["val"]:
            raise RuntimeError("inference did not produce every validation frame")
        aggregate = surface_eval.summarize_results(per_frame)
        case_counts = surface_eval.validate_case_counts(args.batch_summary, aggregate)

        summary = {
            "schema_version": 1,
            "purpose": "Step 8 validation-only surface-row rule freeze",
            "status": "complete",
            "evidence_label": "measured_on_synthetic_validation_only",
            "rule_config": rule_config,
            "dataset": {
                "role": "val",
                "frame_count": len(per_frame),
                "seed_range": [per_frame[0]["seed"], per_frame[-1]["seed"]],
                "case_counts": case_counts,
                "preflight": preflight,
                "train_manifests_opened": 0,
                "train_images_loaded": 0,
                "test_manifests_opened": 0,
                "test_images_loaded": 0,
                "real_frames_loaded": 0,
            },
            "aggregate": aggregate,
            "per_frame": per_frame,
            "model": {
                **artifacts,
                "device": str(device),
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "inputs": {
                "batch_summary_sha256": audit["batch_summary_sha256"],
                "exception_review_sha256": audit["review_sha256"],
                "training_summary_sha256": artifacts["training_summary_sha256"],
                "checkpoint_sha256": artifacts["checkpoint_sha256"],
                "evaluator_script_sha256": training.file_sha256(__file__),
                "evaluator_module_sha256": training.file_sha256(surface_eval.__file__),
            },
            "elapsed_seconds": time.time() - started,
        }
        summary_path = args.output_root / "summary.json"
        _json_dump(summary_path, summary)
        compact = {key: value for key, value in summary.items() if key != "per_frame"}
        compact["full_summary_sha256"] = training.file_sha256(summary_path)
        compact_path = args.output_root / "compact-summary.json"
        _json_dump(compact_path, compact)
        _append_state(state_path, "COMPLETE")
        print(json.dumps({
            "status": "complete",
            "validation_frames": len(per_frame),
            "visible_mae_rows": aggregate["by_case"]["all"]["mae_rows"],
            "predicted_hidden_rate": aggregate["by_case"]["all"]["predicted_hidden_rate"],
            "compact_summary_sha256": training.file_sha256(compact_path),
        }, sort_keys=True, allow_nan=False))
        return summary
    except Exception as error:
        _json_dump(args.output_root / "failure.json", {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        _append_state(state_path, "FAILED")
        raise


def main():
    run(_parser().parse_args())


if __name__ == "__main__":
    main()

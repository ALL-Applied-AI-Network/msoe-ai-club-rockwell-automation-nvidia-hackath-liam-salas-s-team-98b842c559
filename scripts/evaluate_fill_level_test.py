#!/usr/bin/env python3
"""Score the frozen FillCam model/rule/ruler on held-out twin test renders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fluidmix import fill_level_test_eval as test_eval  # noqa: E402
from fluidmix import fill_level_training as training  # noqa: E402


def _append_state(path, status, **details):
    value = {
        "status": status,
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **details,
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _json_dump(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_model(torch, checkpoint_path, binding, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported training checkpoint")
    if int(checkpoint.get("epoch", -1)) != binding["best_epoch"]:
        raise ValueError("checkpoint epoch does not match training summary")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("training checkpoint has no config")
    if (
        config.get("crop_xyxy") != binding["crop_xyxy"]
        or config.get("class_ids") != binding["class_ids"]
        or int(config.get("base_channels", -1)) != binding["base_channels"]
    ):
        raise ValueError("checkpoint model contract does not match the frozen binding")
    model = training.build_unet(base_channels=binding["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--batch-summary-sha256", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-summary-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--training-module-sha256", required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--frozen-rule-sha256", required=True)
    parser.add_argument("--frozen-ruler", type=Path, required=True)
    parser.add_argument("--frozen-ruler-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(0,),
        default=0,
        help="Frozen at zero so the one held-out run cannot hit DataLoader shared-memory limits.",
    )
    return parser


def run(args):
    # Every immutable gate runs before the dataset root is listed or output exists.
    authority = test_eval.validate_authority(
        batch_summary_path=args.batch_summary,
        batch_summary_sha256=args.batch_summary_sha256,
        training_summary_path=args.training_summary,
        training_summary_sha256=args.training_summary_sha256,
        checkpoint_path=args.checkpoint,
        weights_sha256=args.weights_sha256,
        training_module_sha256=args.training_module_sha256,
        frozen_rule_path=args.frozen_rule,
        frozen_rule_sha256=args.frozen_rule_sha256,
        frozen_ruler_path=args.frozen_ruler,
        frozen_ruler_sha256=args.frozen_ruler_sha256,
    )
    if args.batch_size <= 0 or args.workers != 0:
        raise ValueError("batch size must be positive and workers must stay frozen at zero")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    _append_state(state_path, "RUNNING")
    started = time.time()
    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        training.seed_everything(authority["model"]["training_seed"])
        model = _load_model(torch, args.checkpoint, authority["model"], device)

        samples, metadata = test_eval.discover_test_frames(
            args.dataset_root, authority["batch"]
        )
        preflight = training.verify_sample_files(samples)
        dataset = training.make_torch_dataset(
            samples,
            augment=False,
            seed=authority["model"]["training_seed"],
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

        confusion = np.zeros((4, 4), dtype=np.int64)
        per_frame = []
        with torch.inference_mode():
            for images, target_masks, seeds in loader:
                predictions = model(images.to(device, non_blocking=True)).argmax(dim=1)
                predictions = predictions.cpu().numpy()
                truths = target_masks.numpy()
                for index, raw_seed in enumerate(seeds.tolist()):
                    seed = int(raw_seed)
                    confusion += training.confusion_matrix(predictions[index], truths[index])
                    per_frame.append(
                        test_eval.frame_result(
                            metadata[seed],
                            truths[index],
                            predictions[index],
                            authority["ruler"]["ruler"],
                            authority["rule"]["rule_config"],
                        )
                    )
        per_frame.sort(key=lambda item: item["seed"])
        if len(per_frame) != test_eval.EXPECTED_TEST_COUNT:
            raise RuntimeError("inference did not produce all 560 held-out test frames")
        aggregate = test_eval.summarize_results(per_frame)
        if {
            name: aggregate["by_case"][name]["frames"]
            for name in test_eval.CASE_NAMES
        } != test_eval.EXPECTED_CASE_COUNTS:
            raise RuntimeError("test case aggregates changed after inference")
        segmentation = test_eval.segmentation_summary(confusion)
        overall = aggregate["by_case"]["all"]
        visible_median_ml = overall["visible_absolute_error_ml_median"]
        visible_p95_ml = overall["visible_absolute_error_ml_p95"]
        hidden_success_rate = overall["hidden_truth_in_range_rate"]
        target_comparison = {
            "visible_typical": {
                "metric": "median absolute error in mL from row error / 3.04",
                "maximum_ml": 1.0,
                "observed_ml": visible_median_ml,
                "passed": visible_median_ml is not None and visible_median_ml <= 1.0,
            },
            "visible_p95": {
                "metric": "p95 absolute error in mL from row error / 3.04",
                "maximum_ml": 2.0,
                "observed_ml": visible_p95_ml,
                "passed": visible_p95_ml is not None and visible_p95_ml <= 2.0,
            },
            "hidden": {
                "metric": "ground-truth-hidden frames called hidden with true level in range",
                "minimum_rate": 0.95,
                "user_practical_minimum_rate": 0.90,
                "observed_rate": hidden_success_rate,
                "passed": hidden_success_rate is not None and hidden_success_rate >= 0.95,
                "meets_user_practical_bar": (
                    hidden_success_rate is not None and hidden_success_rate >= 0.90
                ),
            },
        }

        summary = {
            "schema_version": 1,
            "purpose": "Step 9 held-out synthetic twin-test evaluation",
            "status": "complete",
            "evidence_label": "measured_on_held_out_synthetic_test_only",
            "checkpoint_review_sha256": test_eval.EXPECTED_CHECKPOINT_REVIEW_SHA256,
            "disclosures": {
                "system_calibration": "failed",
                "continued_after_failed_calibration": "explicit_user_decision",
                "ruler": "hand_measured_step3_targets_not_system_fitted",
                "result_use": "measured_once_and_untuned",
            },
            "dataset": {
                "role": "test",
                "frame_count": len(per_frame),
                "seed_ranges_inclusive": [[2100000, 2100499], [2101000, 2101059]],
                "case_counts": dict(test_eval.EXPECTED_CASE_COUNTS),
                "cohort_counts": dict(test_eval.EXPECTED_COHORT_COUNTS),
                "preflight": preflight,
                "test_manifests_opened": len(authority["batch"]["approved_chunks"]),
                "test_images_loaded": len(per_frame),
                "train_manifests_opened": 0,
                "train_images_loaded": 0,
                "val_manifests_opened": 0,
                "val_images_loaded": 0,
                "calibration_manifests_opened": 0,
                "calibration_frames_loaded": 0,
                "real_check_frames_loaded": 0,
            },
            "segmentation": segmentation,
            "target_comparison": target_comparison,
            "aggregate": aggregate,
            "per_frame": per_frame,
            "frozen": {
                "rule_config": authority["rule"]["rule_config"],
                "ruler": authority["ruler"]["ruler"],
                "ruler_source": authority["ruler"]["source"],
            },
            "model": {
                **authority["model"],
                "device": str(device),
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "inputs": {
                "batch_summary_sha256": authority["batch"]["artifact_sha256"],
                "training_summary_sha256": authority["model"][
                    "training_summary_sha256"
                ],
                "weights_sha256": authority["model"]["weights_sha256"],
                "training_module_sha256": authority["model"][
                    "training_module_sha256"
                ],
                "frozen_rule_sha256": authority["rule"]["artifact_sha256"],
                "surface_rule_module_sha256": authority["rule"][
                    "surface_module_sha256"
                ],
                "frozen_ruler_sha256": authority["ruler"]["artifact_sha256"],
                "ruler_module_sha256": authority["ruler"]["ruler_module_sha256"],
                "evaluator_module_sha256": training.file_sha256(test_eval.__file__),
                "evaluator_script_sha256": training.file_sha256(__file__),
            },
            "elapsed_seconds": time.time() - started,
        }
        full_path = args.output_root / "summary.json"
        _json_dump(full_path, summary)
        compact = {key: value for key, value in summary.items() if key != "per_frame"}
        compact["full_summary_sha256"] = training.file_sha256(full_path)
        compact_path = args.output_root / "compact-summary.json"
        _json_dump(compact_path, compact)
        _append_state(
            state_path,
            "COMPLETE",
            full_summary_sha256=training.file_sha256(full_path),
            compact_summary_sha256=training.file_sha256(compact_path),
        )
        print(json.dumps({
            "status": "complete",
            "test_frames": len(per_frame),
            "macro_iou": segmentation["macro_iou"],
            "visible_mae_ml": aggregate["by_case"]["all"][
                "visible_absolute_error_ml_mae"
            ],
            "compact_summary_sha256": training.file_sha256(compact_path),
        }, sort_keys=True, allow_nan=False))
        return summary
    except Exception as error:
        failure_path = args.output_root / "failure.json"
        _json_dump(failure_path, {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        _append_state(state_path, "FAILED", failure_sha256=training.file_sha256(failure_path))
        raise


def main():
    run(_parser().parse_args())


if __name__ == "__main__":
    main()

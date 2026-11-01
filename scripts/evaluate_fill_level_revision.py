#!/usr/bin/env python3
"""Compare original and Revision 1 FillCam models on synthetic validation only."""

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
from fluidmix import fill_level_revision_eval as revision_eval  # noqa: E402
from fluidmix import fill_level_surface_eval as surface_eval  # noqa: E402
from fluidmix import fill_level_training as training  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--exception-review", type=Path, required=True)
    for revision in ("original", "revised"):
        parser.add_argument(
            f"--{revision}-training-summary", type=Path, required=True
        )
        parser.add_argument(
            f"--{revision}-training-summary-sha256", required=True
        )
        parser.add_argument(f"--{revision}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{revision}-checkpoint-sha256", required=True)
        parser.add_argument(f"--{revision}-rule-summary", type=Path, required=True)
        parser.add_argument(f"--{revision}-rule-summary-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def _json_dump(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_state(path, status, **details):
    entry = {
        "status": str(status),
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **details,
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")


def _arg(args, revision, suffix):
    return getattr(args, f"{revision}_{suffix}")


def _preflight(args):
    audit = training.validate_exception_review(
        args.batch_summary, args.exception_review
    )
    evidence = {}
    for revision in ("original", "revised"):
        model = revision_eval.validate_training_artifact(
            _arg(args, revision, "training_summary"),
            _arg(args, revision, "training_summary_sha256"),
            _arg(args, revision, "checkpoint"),
            _arg(args, revision, "checkpoint_sha256"),
            audit,
            revision=revision,
        )
        rule = revision_eval.validate_rule_artifact(
            _arg(args, revision, "rule_summary"),
            _arg(args, revision, "rule_summary_sha256"),
            model,
            audit,
            revision=revision,
        )
        evidence[revision] = {"model": model, "rule": rule}
    if (
        evidence["original"]["model"]["checkpoint_sha256"]
        == evidence["revised"]["model"]["checkpoint_sha256"]
    ):
        raise ValueError("original and revised checkpoints must differ")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    return audit, evidence


def _make_dataset(torch, samples):
    class RevisionValidationDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            sample = samples[index]
            with Image.open(sample.rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(sample.mask_path) as image:
                mask = np.asarray(image)
            rgb, mask = training.crop_rgb_mask(rgb, mask)
            darkened = revision_eval.darkened_validation_rgb(rgb, mask, sample.seed)
            if not np.array_equal(darkened[mask != 1], rgb[mask != 1]):
                raise RuntimeError(
                    f"darkened validation changed non-liquid pixels for seed {sample.seed}"
                )
            saturation_histogram, value_histogram = (
                revision_eval.liquid_hsv8_histograms(darkened, mask)
            )
            normal_tensor = torch.from_numpy(
                np.ascontiguousarray(rgb.transpose(2, 0, 1))
            ).float().div_(255.0)
            darkened_tensor = torch.from_numpy(
                np.ascontiguousarray(darkened.transpose(2, 0, 1))
            ).float().div_(255.0)
            mask_tensor = torch.from_numpy(mask.astype(np.int64, copy=False))
            return (
                normal_tensor,
                darkened_tensor,
                mask_tensor,
                int(sample.seed),
                torch.from_numpy(saturation_histogram),
                torch.from_numpy(value_histogram),
            )

    return RevisionValidationDataset()


def _load_model(torch, path, evidence, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError(f"unsupported checkpoint: {path}")
    if int(checkpoint.get("epoch", -1)) != evidence["best_epoch"]:
        raise ValueError(f"checkpoint epoch does not match summary: {path}")
    config = checkpoint.get("config", {})
    if config.get("crop_xyxy") != list(training.CROP_XYXY):
        raise ValueError(f"checkpoint crop does not match current crop: {path}")
    if int(config.get("base_channels", -1)) != evidence["base_channels"]:
        raise ValueError(f"checkpoint model width does not match summary: {path}")
    if int(config.get("seed", -1)) != evidence["training_seed"]:
        raise ValueError(f"checkpoint training seed does not match summary: {path}")
    if config.get("class_ids") != revision_eval.EXPECTED_CLASS_IDS:
        raise ValueError(f"checkpoint class IDs do not match: {path}")
    if config.get("augmentation") != evidence["augmentation"]:
        raise ValueError(f"checkpoint augmentation does not match summary: {path}")
    if config.get("learning_rate_scheduler") != evidence["learning_rate_scheduler"]:
        raise ValueError(f"checkpoint learning-rate schedule does not match summary: {path}")
    model = training.build_unet(base_channels=evidence["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _infer(torch, loader, models, metadata, rule_configs, device):
    truths = []
    ordered_metadata = []
    dark_saturation_histograms = []
    dark_value_histograms = []
    predictions = {
        revision: {"normal": [], "darkened": []} for revision in models
    }
    with torch.inference_mode():
        for (
            normal,
            darkened,
            target,
            seeds,
            saturation_histograms,
            value_histograms,
        ) in loader:
            views = {
                "normal": normal.to(device, non_blocking=True),
                "darkened": darkened.to(device, non_blocking=True),
            }
            truths.extend(target.numpy())
            ordered_metadata.extend(metadata[int(seed)] for seed in seeds.tolist())
            dark_saturation_histograms.extend(saturation_histograms.numpy())
            dark_value_histograms.extend(value_histograms.numpy())
            for revision, model in models.items():
                for variant, images in views.items():
                    batch = model(images).argmax(dim=1).cpu().numpy()
                    predictions[revision][variant].extend(batch)
    metrics = {}
    for revision, variants in predictions.items():
        metrics[revision] = {}
        for variant, masks in variants.items():
            metrics[revision][variant] = revision_eval.variant_metrics(
                truths,
                masks,
                ordered_metadata,
                rule_configs[revision],
            )
    measured_hsv = revision_eval.summarize_darkened_validation_hsv(
        np.asarray(dark_saturation_histograms, dtype=np.int64),
        np.asarray(dark_value_histograms, dtype=np.int64),
    )
    return metrics, measured_hsv


def _comparison(metrics):
    def optional_delta(original, revised):
        if original is None or revised is None:
            return None
        return float(revised) - float(original)

    result = {}
    for variant in ("normal", "darkened"):
        original = metrics["original"][variant]
        revised = metrics["revised"][variant]
        original_surface = original["surface"]["by_case"]["all"]
        revised_surface = revised["surface"]["by_case"]["all"]
        result[variant] = {
            "macro_iou_delta_revised_minus_original": (
                revised["iou"]["macro"] - original["iou"]["macro"]
            ),
            "visible_p95_rows_original": original_surface[
                "p95_absolute_error_rows"
            ],
            "visible_p95_rows_revised": revised_surface[
                "p95_absolute_error_rows"
            ],
            "visible_p95_rows_delta_revised_minus_original": optional_delta(
                original_surface["p95_absolute_error_rows"],
                revised_surface["p95_absolute_error_rows"],
            ),
        }
    return result


def run(args):
    # All authority, artifact and rule gates precede output or dataset access.
    audit, evidence = _preflight(args)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    _append_state(state_path, "RUNNING")
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
        for revision in ("original", "revised"):
            if evidence[revision]["rule"]["ordered_hash_ledger_sha256"] != (
                preflight["ordered_hash_ledger_sha256"]
            ):
                raise ValueError(
                    f"{revision} rule validation ledger does not match the "
                    "comparison dataset"
                )
        dataset = _make_dataset(torch, samples)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=training.seed_worker,
            persistent_workers=False,
        )
        models = {
            revision: _load_model(
                torch,
                _arg(args, revision, "checkpoint"),
                evidence[revision]["model"],
                device,
            )
            for revision in ("original", "revised")
        }
        rule_configs = {
            revision: evidence[revision]["rule"]["rule_config"]
            for revision in ("original", "revised")
        }
        metrics, darkened_hsv = _infer(
            torch, loader, models, metadata, rule_configs, device
        )
        for revision in ("original", "revised"):
            if (
                metrics[revision]["normal"]["surface"]
                != evidence[revision]["rule"]["aggregate"]
            ):
                raise ValueError(
                    f"{revision} normal surface metrics do not reproduce "
                    "the frozen rule summary"
                )
            for variant in ("normal", "darkened"):
                surface_eval.validate_case_counts(
                    args.batch_summary,
                    metrics[revision][variant]["surface"],
                )
        gate_error = None
        try:
            gate_value = revision_eval.enforce_normal_validation_gate(
                metrics["revised"]["normal"]
            )
        except ValueError as error:
            gate_value = None
            gate_error = str(error)
        summary = {
            "schema_version": 1,
            "purpose": (
                "Step 8 Revision 1 original-versus-revised synthetic "
                "validation comparison"
            ),
            "status": "complete" if gate_error is None else "gate_failed",
            "evidence_label": "measured_on_synthetic_validation_only",
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
            "models": {
                revision: {
                    **evidence[revision]["model"],
                    "rule_summary_sha256": evidence[revision]["rule"][
                        "summary_sha256"
                    ],
                    "rule_config": evidence[revision]["rule"]["rule_config"],
                    "metrics": metrics[revision],
                }
                for revision in ("original", "revised")
            },
            "comparison": _comparison(metrics),
            "inputs": {
                "batch_summary_sha256": audit["batch_summary_sha256"],
                "exception_review_sha256": audit["review_sha256"],
                "comparison_script_sha256": training.file_sha256(__file__),
                "comparison_module_sha256": training.file_sha256(
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
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            },
            "execution": {
                "batch_size": int(args.batch_size),
                "workers": int(args.workers),
                "shuffle": False,
            },
            "elapsed_seconds": time.time() - started,
        }
        summary_path = args.output_root / "summary.json"
        _json_dump(summary_path, summary)
        compact = {
            **summary,
            "models": {
                revision: {
                    **{
                        key: value
                        for key, value in summary["models"][revision].items()
                        if key != "metrics"
                    },
                    "metrics": revision_eval.compact_model_metrics(
                        summary["models"][revision]["metrics"]
                    ),
                }
                for revision in ("original", "revised")
            },
            "full_summary_sha256": training.file_sha256(summary_path),
        }
        compact_path = args.output_root / "compact-summary.json"
        _json_dump(compact_path, compact)
        terminal = "COMPLETE" if gate_error is None else "GATE_FAILED"
        _append_state(
            state_path,
            terminal,
            compact_summary_sha256=training.file_sha256(compact_path),
        )
        return compact
    except Exception as error:
        _json_dump(
            args.output_root / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        _append_state(state_path, "FAILED")
        raise


def main():
    result = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "comparison": result["comparison"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the Step 7 four-class FillCam U-Net on approved synthetic data only."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fluidmix import fill_level_training as training  # noqa: E402


PROFILES = {
    "tiny": {
        "train_limit": 12,
        "val_limit": 12,
        "epochs": 40,
        "patience": 40,
        "batch_size": 4,
        "augment": False,
        "diagnostic_reuses_train_as_validation": True,
    },
    "smoke": {
        "train_limit": 64,
        "val_limit": 32,
        "epochs": 2,
        "patience": 2,
        "batch_size": 8,
        "augment": True,
        "diagnostic_reuses_train_as_validation": False,
    },
    "full": {
        "train_limit": None,
        "val_limit": None,
        "epochs": 30,
        "patience": 6,
        "batch_size": 16,
        "augment": True,
        "diagnostic_reuses_train_as_validation": False,
    },
}


def _json_dump(path, value):
    Path(path).write_text(
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


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _iou_json(confusion):
    per_class, macro = training.iou_from_confusion(confusion)
    return {
        "per_class": [float(value) if math.isfinite(value) else None for value in per_class],
        "macro": float(macro) if math.isfinite(macro) else None,
    }


def _update_confusion(torch, confusion, logits, target, num_classes=4):
    prediction = logits.argmax(dim=1)
    valid = target != training.IGNORE_LABEL
    encoded = target[valid].to(torch.int64) * num_classes + prediction[valid].to(torch.int64)
    if encoded.numel():
        counts = torch.bincount(encoded, minlength=num_classes * num_classes)
        confusion += counts.reshape(num_classes, num_classes).cpu().numpy()


def _autocast(torch, device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _run_epoch(
    torch,
    *,
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
):
    training_mode = optimizer is not None
    model.train(training_mode)
    total_loss = 0.0
    batch_count = 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    context = nullcontext() if training_mode else torch.inference_mode()
    with context:
        for images, masks, _seeds in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training_mode:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if training_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach().item())
            batch_count += 1
            _update_confusion(torch, confusion, logits.detach(), masks)
    if batch_count == 0:
        raise RuntimeError("data loader produced no batches")
    return {
        "loss": total_loss / batch_count,
        "confusion": confusion,
        "iou": _iou_json(confusion),
    }


def _checkpoint(
    torch,
    path,
    *,
    model,
    optimizer,
    scheduler,
    epoch,
    best_macro_iou,
    config,
):
    torch.save(
        {
            "schema_version": 1,
            "epoch": int(epoch),
            "best_val_macro_iou": float(best_macro_iou),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def _profile_config(args):
    profile = dict(PROFILES[args.profile])
    for name in ("epochs", "patience", "batch_size", "train_limit", "val_limit"):
        value = getattr(args, name)
        if value is not None:
            profile[name] = value
    if args.no_augment:
        profile["augment"] = False
    return profile


def _scheduler_config(args, profile):
    initial = float(args.learning_rate)
    minimum = float(args.min_learning_rate)
    if not math.isfinite(initial) or initial <= 0.0:
        raise ValueError("initial learning rate must be finite and positive")
    if not math.isfinite(minimum) or not 0.0 <= minimum < initial:
        raise ValueError(
            "minimum learning rate must be finite, non-negative and below the initial rate"
        )
    epochs = int(profile["epochs"])
    if epochs <= 0:
        raise ValueError("scheduler needs a positive epoch count")
    return {
        "name": "cosine_annealing",
        "initial_learning_rate": initial,
        "minimum_learning_rate": minimum,
        "t_max_epochs": epochs,
        "step_unit": "epoch",
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--exception-review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def run(args):
    # This gate must run before dataset discovery or any manifest/image access.
    audit = training.validate_exception_review(args.batch_summary, args.exception_review)
    population_contract = training.dataset_population_contract(audit)
    profile = _profile_config(args)
    scheduler_contract = _scheduler_config(args, profile)

    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    _append_state(state_path, "RUNNING")
    started = time.time()
    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        training.seed_everything(args.seed)
        discovered = training.discover_train_val_samples(
            args.dataset_root,
            expected_train=audit["eligible_counts"]["train"],
            expected_val=audit["eligible_counts"]["val"],
            excluded_seeds=audit["excluded_seeds"],
            expected_chunks=audit["approved_chunks"],
        )
        train_samples = tuple(discovered["train"])
        val_samples = tuple(discovered["val"])
        train_limit = profile["train_limit"]
        val_limit = profile["val_limit"]
        selected_train = train_samples[:train_limit] if train_limit else train_samples
        if profile["diagnostic_reuses_train_as_validation"]:
            selected_val = selected_train[:val_limit] if val_limit else selected_train
        else:
            selected_val = val_samples[:val_limit] if val_limit else val_samples
        if not selected_train or not selected_val:
            raise ValueError("training and validation selections must be non-empty")

        preflight = training.verify_sample_files((*selected_train, *selected_val))
        train_counts = training.mask_class_counts(
            (sample.mask_path for sample in selected_train)
        )
        class_weights = training.class_weights_from_counts(train_counts)
        train_dataset = training.make_torch_dataset(
            selected_train,
            augment=bool(profile["augment"]),
            seed=args.seed,
        )
        val_dataset = training.make_torch_dataset(
            selected_val,
            augment=False,
            seed=args.seed,
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader_options = {
            "batch_size": int(profile["batch_size"]),
            "num_workers": int(args.workers),
            "pin_memory": device.type == "cuda",
            "worker_init_fn": training.seed_worker,
            "persistent_workers": False,
        }
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_options,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            **loader_options,
        )
        model = training.build_unet(base_channels=args.base_channels).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_contract["t_max_epochs"]),
            eta_min=float(scheduler_contract["minimum_learning_rate"]),
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights_tensor,
            ignore_index=training.IGNORE_LABEL,
        )
        amp_enabled = device.type == "cuda" and not args.no_amp
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        config = {
            "profile": args.profile,
            "seed": args.seed,
            "device": str(device),
            "crop_xyxy": list(training.CROP_XYXY),
            "class_ids": {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3},
            "ignore_index": training.IGNORE_LABEL,
            "base_channels": args.base_channels,
            "learning_rate": args.learning_rate,
            "learning_rate_scheduler": scheduler_contract,
            "weight_decay": args.weight_decay,
            "workers": args.workers,
            "amp": amp_enabled,
            "augmentation": {
                "enabled": bool(profile["augment"]),
                "train_only": True,
                "contract": (
                    training.training_augmentation_config()
                    if profile["augment"]
                    else None
                ),
            },
            "dataset_population_contract": population_contract,
            **profile,
        }
        _json_dump(args.output_root / "config.json", config)
        progress_path = args.output_root / "progress.jsonl"
        best_path = None
        best_macro = -math.inf
        best_epoch = None
        best_val = None
        stale_epochs = 0
        history = []
        for epoch in range(1, int(profile["epochs"]) + 1):
            epoch_started = time.time()
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            train_result = _run_epoch(
                torch,
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            val_result = _run_epoch(
                torch,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
            val_macro = val_result["iou"]["macro"]
            if val_macro is None:
                raise RuntimeError("validation macro IoU is undefined")
            scheduler.step()
            next_learning_rate = float(optimizer.param_groups[0]["lr"])
            improved = float(val_macro) > best_macro
            if improved:
                best_macro = float(val_macro)
                best_epoch = epoch
                best_val = val_result
                stale_epochs = 0
                best_path = args.output_root / f"best-epoch-{epoch:03d}.pt"
                _checkpoint(
                    torch,
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_macro_iou=best_macro,
                    config=config,
                )
            else:
                stale_epochs += 1
            entry = {
                "epoch": epoch,
                "seconds": time.time() - epoch_started,
                "learning_rate": epoch_learning_rate,
                "next_learning_rate": next_learning_rate,
                "train_loss": train_result["loss"],
                "train_iou": train_result["iou"],
                "val_loss": val_result["loss"],
                "val_iou": val_result["iou"],
                "best": improved,
                "stale_epochs": stale_epochs,
            }
            history.append(entry)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)
            if stale_epochs >= int(profile["patience"]):
                break

        if best_epoch is None or best_path is None or not best_path.is_file():
            raise RuntimeError("training did not produce a best checkpoint")
        last_path = args.output_root / f"last-epoch-{len(history):03d}.pt"
        _checkpoint(
            torch,
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=len(history),
            best_macro_iou=best_macro,
            config=config,
        )
        summary = {
            "schema_version": 1,
            "purpose": "FillCam Step 7 synthetic-only U-Net training evidence",
            "status": "complete",
            "profile": args.profile,
            "checkpoint_candidate": args.profile == "full",
            "selection_split": (
                "train_diagnostic"
                if profile["diagnostic_reuses_train_as_validation"]
                else "validation"
            ),
            "dataset": {
                **population_contract,
                "eligible_counts": audit["eligible_counts"],
                "used_counts": {
                    "train": len(selected_train),
                    "val": len(selected_val),
                },
                "excluded_seeds": audit["excluded_seeds"],
                "train_seed_range": [train_samples[0].seed, train_samples[-1].seed],
                "val_seed_range": [val_samples[0].seed, val_samples[-1].seed],
                "test_manifests_opened": 0,
                "test_images_loaded": 0,
                "preflight": preflight,
            },
            "audit": audit,
            "config": config,
            "training_class_pixel_counts": [int(value) for value in train_counts],
            "class_weights": [float(value) for value in class_weights],
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "packages": {
                    "numpy": _package_version("numpy"),
                    "Pillow": _package_version("Pillow"),
                    "torchvision": _package_version("torchvision"),
                    "typing_extensions": _package_version("typing_extensions"),
                },
            },
            "model": {
                "name": "four-level-unet",
                "parameter_count": int(parameter_count),
            },
            "history": history,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_val["loss"],
            "best_val_iou": best_val["iou"],
            "elapsed_seconds": time.time() - started,
            "artifacts": {
                "best_weights": best_path.name,
                "best_weights_sha256": training.file_sha256(best_path),
                "last_weights": last_path.name,
                "last_weights_sha256": training.file_sha256(last_path),
                "progress": "progress.jsonl",
                "progress_sha256": training.file_sha256(progress_path),
            },
            "inputs": {
                "batch_summary_sha256": audit["batch_summary_sha256"],
                "exception_review_sha256": audit["review_sha256"],
                "training_script_sha256": training.file_sha256(__file__),
                "training_module_sha256": training.file_sha256(training.__file__),
            },
        }
        _json_dump(args.output_root / "summary.json", summary)
        _append_state(state_path, "COMPLETE")
        print(
            json.dumps(
                {
                    "status": "complete",
                    "best_epoch": best_epoch,
                    "best_val_macro_iou": best_macro,
                },
                sort_keys=True,
            )
        )
        return summary
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
    args = _parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

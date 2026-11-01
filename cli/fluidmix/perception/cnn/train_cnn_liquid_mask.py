#!/usr/bin/env python3
"""Train or CPU-smoke the sealed six-channel WP5 liquid-mask candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.perception.cnn.cnn_predictor import (  # noqa: E402
    DEFAULT_THRESHOLD,
    IMAGENET_WEIGHTS_URL,
    ResNet18UNet,
)
from fluidmix.perception.cnn.cnn_training import (  # noqa: E402
    PairedMaskDataset,
    dice_bce_loss,
    validate_training_inventory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _smoke(device: torch.device) -> dict[str, object]:
    torch.manual_seed(20261029)
    model = ResNet18UNet(weights=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    current = torch.rand(1, 3, 64, 96, device=device)
    reference = torch.rand(1, 3, 64, 96, device=device)
    target = torch.zeros(1, 1, 64, 96, device=device)
    target[:, :, 24:54, 38:58] = 1.0
    logits = model(current, reference)
    loss = dice_bce_loss(logits, target)
    loss.backward()
    optimizer.step()
    return {
        "status": "pass",
        "device": str(device),
        "input_shape_each_rgb": list(current.shape),
        "combined_input_channels": 6,
        "output_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "weights": None,
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    inventory_path = args.inventory.resolve()
    inventory = validate_training_inventory(json.loads(inventory_path.read_text(encoding="utf-8")))
    dataset = PairedMaskDataset(inventory, augment=True, verify_hashes=not args.skip_hash_verify, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    device = torch.device(args.device)
    model = ResNet18UNet(weights="imagenet").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    started = time.monotonic()
    model.train()
    for _epoch in range(args.epochs):
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["current_rgb"].to(device), batch["reference_rgb"].to(device))
            loss = dice_bce_loss(logits, batch["mask"].to(device))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "resnet18-unet-current-empty-rgb.pt"
    torch.save(model.state_dict(), checkpoint)
    weights_name = Path(urlparse(IMAGENET_WEIGHTS_URL).path).name
    pretrained_cache = Path(torch.hub.get_dir()) / "checkpoints" / weights_name
    receipt = {
        "schema_version": 1,
        "artifact_type": "wp5_cnn_frozen_candidate",
        "status": "trained_candidate_not_heldout_scored",
        "source_commit": _source_commit(),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "architecture": "U-Net decoder with torchvision ResNet-18 encoder",
        "input_channels": ["current_r", "current_g", "current_b", "same_run_empty_r", "same_run_empty_g", "same_run_empty_b"],
        "output": "per-pixel liquid logit; sigmoid is liquid probability",
        "no_absolute_position_channel": True,
        "augmentation": "paired reflection-pad translation plus shared hue/saturation/brightness jitter",
        "clamp_and_temporal_policy": "Approach A clamp raster exclusion and existing temporal hidden-band interpolation remain unchanged",
        "threshold": DEFAULT_THRESHOLD,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "AdamW",
            "loss": "binary_cross_entropy_with_logits_plus_soft_dice",
            "seed": args.seed,
            "device": str(device),
            "epoch_mean_losses": losses,
            "elapsed_seconds": time.monotonic() - started,
        },
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "pretrained_weights": {
            "enum": "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
            "url": IMAGENET_WEIGHTS_URL,
            "cache_path": str(pretrained_cache.resolve()),
            "sha256": _sha256(pretrained_cache),
            "stem_adaptation": "concatenate current/reference RGB stem weights and divide by two",
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
        "licenses": {
            "repository_code": "repository license",
            "torch": "BSD-3-Clause",
            "torchvision_and_resnet18_code": "BSD-3-Clause",
            "imagenet_pretraining_dataset": "ImageNet terms; weights are not committed",
        },
        "claim_boundary": "synthetic training only; no heldout, real-line, or physical accuracy claim",
    }
    receipt_path = args.output_dir / "candidate-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--skip-hash-verify", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:
        print(json.dumps(_smoke(torch.device(args.device)), sort_keys=True))
        return 0
    if args.inventory is None or args.output_dir is None:
        parser.error("--inventory and --output-dir are required outside --smoke")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    receipt = train(args)
    print(json.dumps({"status": receipt["status"], "checkpoint_sha256": receipt["checkpoint_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

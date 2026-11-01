"""Training inventory and dataset for the sealed WP5 CNN candidate."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import re

import numpy as np

from ...core.io import open_image, open_path
import torch
from torch.nn import functional as torch_f
from torch.utils.data import Dataset

from .cnn_predictor import paired_augment


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMON_FIELDS = {
    "sample_id",
    "run_id",
    "case_id",
    "split_role",
    "data_origin",
    "current_rgb_path",
    "current_rgb_sha256",
    "reference_rgb_path",
    "reference_rgb_sha256",
    "label_kind",
    "label_source",
}
_MASK_FIELDS = {"mask_path", "mask_sha256"}
_FORBIDDEN_TOKENS = ("recipe", "volume", "milliliter", "_ml", "vial_position", "ordinal")


def validate_training_inventory(value: object) -> dict[str, object]:
    """Reject non-twin, non-train, oracle-like, or recipe-shaped inputs."""

    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("training inventory schema_version must be 1")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("training inventory samples must be nonempty")
    seen: set[str] = set()
    for row in samples:
        if not isinstance(row, Mapping):
            raise ValueError("training inventory rows must be objects")
        label_kind = row.get("label_kind")
        allowed = _COMMON_FIELDS | (_MASK_FIELDS if label_kind == "simulator_mask" else set())
        extra = set(row) - allowed
        if extra or any(token in str(key).lower() for key in row for token in _FORBIDDEN_TOKENS):
            raise ValueError("forbidden training field")
        if set(row) != allowed:
            raise ValueError("training row fields invalid")
        if row.get("split_role") != "train":
            raise ValueError("training rows must use train split")
        if row.get("data_origin") != "twin":
            raise ValueError("training rows must use twin data")
        if label_kind == "simulator_mask":
            if row.get("label_source") != "simulator_semantic_mask":
                raise ValueError("positive labels must be simulator semantic masks")
        elif label_kind == "empty_negative":
            if row.get("label_source") != "same_run_empty_prepour":
                raise ValueError("empty negatives must be same-run pre-pour frames")
            if row.get("current_rgb_path") != row.get("reference_rgb_path"):
                raise ValueError("empty negative current/reference paths must match")
            if row.get("current_rgb_sha256") != row.get("reference_rgb_sha256"):
                raise ValueError("empty negative current/reference hashes must match")
        else:
            raise ValueError("label_kind must be simulator_mask or empty_negative")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("sample_id must be unique and nonempty")
        seen.add(sample_id)
        for key in (
            "current_rgb_sha256",
            "reference_rgb_sha256",
            *(('mask_sha256',) if label_kind == "simulator_mask" else ()),
        ):
            if not isinstance(row.get(key), str) or not _SHA256.fullmatch(str(row[key])):
                raise ValueError(f"{key} invalid")
    return deepcopy(dict(value))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open_path(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(raw: object, expected: object, role: str, verify: bool) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError(f"{role} path must be absolute")
    path = Path(raw)
    if not path.is_file():
        raise ValueError(f"{role} file missing")
    if verify and _hash(path) != expected:
        raise ValueError(f"{role} hash mismatch")
    return path


class PairedMaskDataset(Dataset):
    """Decode current RGB, same-run empty RGB, and training-only mask."""

    def __init__(
        self,
        inventory: object,
        *,
        augment: bool = True,
        verify_hashes: bool = True,
        seed: int = 20261029,
    ):
        self.inventory = validate_training_inventory(inventory)
        self.samples = self.inventory["samples"]
        self.augment = bool(augment)
        self.verify_hashes = bool(verify_hashes)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.samples[index]
        current_path = _checked_path(
            row["current_rgb_path"], row["current_rgb_sha256"], "current_rgb", self.verify_hashes
        )
        reference_path = _checked_path(
            row["reference_rgb_path"], row["reference_rgb_sha256"], "reference_rgb", self.verify_hashes
        )
        current = torch.from_numpy(np.asarray(open_image(current_path).convert("RGB")).copy()).permute(2, 0, 1).float().div_(255.0)
        reference = torch.from_numpy(np.asarray(open_image(reference_path).convert("RGB")).copy()).permute(2, 0, 1).float().div_(255.0)
        if row["label_kind"] == "simulator_mask":
            mask_path = _checked_path(row["mask_path"], row["mask_sha256"], "mask", self.verify_hashes)
            mask_array = np.asarray(open_image(mask_path).convert("L"))
            mask = torch.from_numpy((mask_array > 0).astype(np.float32, copy=False)).unsqueeze(0)
        else:
            mask = torch.zeros((1, *current.shape[-2:]), dtype=torch.float32)
        if current.shape != reference.shape or mask.shape[-2:] != current.shape[-2:]:
            raise ValueError("training RGB/reference/mask shapes do not align")
        if self.augment:
            current, reference, mask = paired_augment(
                current,
                reference,
                mask,
                generator=torch.Generator().manual_seed(self.seed + index),
            )
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        return {
            "current_rgb": (current - mean) / std,
            "reference_rgb": (reference - mean) / std,
            "mask": mask,
            "sample_id": row["sample_id"],
        }


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable BCE plus soft Dice for a sparse binary liquid mask."""

    if logits.shape != target.shape:
        raise ValueError("logits and target shapes must match")
    bce = torch_f.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=(1, 2, 3)) + 1.0
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1.0 - numerator / denominator).mean()


__all__ = ["PairedMaskDataset", "dice_bce_loss", "validate_training_inventory"]

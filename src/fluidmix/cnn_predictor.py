"""Six-channel ResNet-18 U-Net candidate for synthetic liquid masks.

Only current RGB and the same-run empty pre-pour RGB enter the network.  The
module has no coordinate, recipe, volume, or run-identity input.  The fixed
Approach-A vial ROI and clamp raster remain post-inference exclusions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_f
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as vision_f

from .empty_reference_rgb_predictor import (
    EmptyReferencePrediction,
    IMAGE_SHAPE,
    fillcam_masks,
)


IMAGENET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
IMAGENET_WEIGHTS_URL = IMAGENET_WEIGHTS.url
DEFAULT_THRESHOLD = 0.5


def adapt_resnet18_stem_weight(rgb_weight: torch.Tensor) -> torch.Tensor:
    """Adapt one three-channel ImageNet stem to current+reference RGB."""

    if tuple(rgb_weight.shape) != (64, 3, 7, 7):
        raise ValueError("ResNet-18 RGB stem must have shape (64, 3, 7, 7)")
    return torch.cat((rgb_weight, rgb_weight), dim=1) / 2.0


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.convolution = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = torch_f.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolution(torch.cat((value, skip), dim=1))


class ResNet18UNet(nn.Module):
    """Small U-Net decoder over an ImageNet-pretrained ResNet-18 encoder."""

    def __init__(self, *, weights: str | None = "imagenet"):
        super().__init__()
        if weights not in {None, "imagenet"}:
            raise ValueError("weights must be None or 'imagenet'")
        encoder_weights = IMAGENET_WEIGHTS if weights == "imagenet" else None
        self.encoder = resnet18(weights=encoder_weights)
        original_stem = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if encoder_weights is not None:
            with torch.no_grad():
                self.encoder.conv1.weight.copy_(
                    adapt_resnet18_stem_weight(original_stem.weight)
                )
        self.decode4 = _DecoderBlock(512, 256, 256)
        self.decode3 = _DecoderBlock(256, 128, 128)
        self.decode2 = _DecoderBlock(128, 64, 64)
        self.decode1 = _DecoderBlock(64, 64, 64)
        self.decode0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mask_logits = nn.Conv2d(32, 1, 1)

    def forward(
        self, current_rgb: torch.Tensor, reference_rgb: torch.Tensor
    ) -> torch.Tensor:
        if (
            current_rgb.ndim != 4
            or reference_rgb.ndim != 4
            or current_rgb.shape != reference_rgb.shape
            or current_rgb.shape[1] != 3
        ):
            raise ValueError("current and reference must have matching three RGB channels")
        six_channel = torch.cat((current_rgb, reference_rgb), dim=1)
        x0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(six_channel)))
        x1 = self.encoder.layer1(self.encoder.maxpool(x0))
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        value = self.decode4(x4, x3)
        value = self.decode3(value, x2)
        value = self.decode2(value, x1)
        value = self.decode1(value, x0)
        value = self.decode0(value)
        if value.shape[-2:] != current_rgb.shape[-2:]:
            value = torch_f.interpolate(
                value, size=current_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.mask_logits(value)


def _uniform(
    low: float, high: float, *, generator: torch.Generator
) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def paired_augment(
    current: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator,
    max_translation_px: int = 24,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply aligned translation and RGB hue/saturation/brightness jitter.

    Reflection-pad/crop translation prevents a fixed absolute-position cue.
    The same photometric factors are applied to both RGB images so a negative
    pair stays identical.  The semantic mask receives geometry only.
    """

    if current.shape != reference.shape or current.ndim != 3 or current.shape[0] != 3:
        raise ValueError("current and reference must be matching CHW RGB tensors")
    if mask.shape != (1, *current.shape[-2:]):
        raise ValueError("mask must be one aligned binary channel")
    if max_translation_px < 0:
        raise ValueError("max_translation_px must be nonnegative")
    pad = int(max_translation_px)
    if pad:
        dx = int(torch.randint(-pad, pad + 1, (), generator=generator))
        dy = int(torch.randint(-pad, pad + 1, (), generator=generator))
        height, width = current.shape[-2:]

        def moved(value: torch.Tensor) -> torch.Tensor:
            padded = torch_f.pad(value, (pad, pad, pad, pad), mode="reflect")
            y0 = pad - dy
            x0 = pad - dx
            return padded[:, y0 : y0 + height, x0 : x0 + width]

        current = moved(current)
        reference = moved(reference)
        mask = moved(mask)
    brightness = _uniform(0.75, 1.25, generator=generator)
    saturation = _uniform(0.75, 1.25, generator=generator)
    hue = _uniform(-0.08, 0.08, generator=generator)

    def jitter(image: torch.Tensor) -> torch.Tensor:
        image = vision_f.adjust_brightness(image, brightness)
        image = vision_f.adjust_saturation(image, saturation)
        return vision_f.adjust_hue(image, hue).clamp_(0.0, 1.0)

    return jitter(current), jitter(reference), (mask >= 0.5).to(mask.dtype)


def _empty(status: str) -> EmptyReferencePrediction:
    return EmptyReferencePrediction(
        mask=np.zeros(IMAGE_SHAPE, dtype=bool),
        status=status,
        surface_row=None,
        visible_height_px=None,
        changed_pixel_count=0,
        median_rgb=None,
    )


def probability_to_visible_prediction(
    probability: object, *, threshold: float = DEFAULT_THRESHOLD
) -> EmptyReferencePrediction:
    """Apply the unchanged Approach-A ROI/clamp exclusions to CNN probability."""

    values = np.asarray(probability)
    if values.shape != IMAGE_SHAPE or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("probability must be one floating 640x480 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("probability must be finite")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between zero and one")
    roi, clamp = fillcam_masks()
    mask = (values >= float(threshold)) & roi & ~clamp
    if not np.any(mask):
        return _empty("empty_mask_abstain")
    rows = np.flatnonzero(np.any(mask, axis=1))
    surface = int(rows[0])
    bottom = int(rows[-1])
    return EmptyReferencePrediction(
        mask=mask,
        status="visible_measured",
        surface_row=surface,
        visible_height_px=bottom - surface + 1,
        changed_pixel_count=int(np.count_nonzero(mask)),
        median_rgb=None,
    )


def _rgb_tensor(image: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (value - mean) / std


@torch.no_grad()
def predict_pair(
    model: nn.Module,
    current_rgb: object,
    reference_rgb: object,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | torch.device = "cpu",
) -> EmptyReferencePrediction:
    """Predict one pair, with fail-closed identical and black controls."""

    current = np.asarray(current_rgb)
    reference = np.asarray(reference_rgb)
    if current.shape != (*IMAGE_SHAPE, 3) or current.dtype != np.uint8:
        raise ValueError("current RGB must be one 640x480 uint8 RGB array")
    if reference.shape != current.shape or reference.dtype != np.uint8:
        raise ValueError("reference RGB must match current RGB")
    if not np.any(current):
        return _empty("invalid_black_frame")
    if np.array_equal(current, reference):
        return _empty("identical_control_abstain")
    target = torch.device(device)
    model.eval()
    logits = model(
        _rgb_tensor(current).unsqueeze(0).to(target),
        _rgb_tensor(reference).unsqueeze(0).to(target),
    )
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probability_to_visible_prediction(probability, threshold=threshold)


def _finite_metric(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def score_holdout_comparison(
    cnn_rows: Sequence[Mapping[str, object]],
    approach_a_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply every preregistered gate and a same-case Approach-A comparison."""

    positives = [row for row in cnn_rows if row.get("control_kind") == "positive"]
    if not positives:
        raise ValueError("at least one positive heldout row is required")
    cnn_dice = np.asarray([_finite_metric(row, "dice") for row in positives])
    height_errors = np.asarray(
        [_finite_metric(row, "visible_height_error_px") for row in positives]
    )
    approach_by_case = {str(row.get("case_id")): row for row in approach_a_rows}
    if set(approach_by_case) != {str(row.get("case_id")) for row in positives}:
        raise ValueError("Approach A cases must exactly match CNN positive cases")
    approach_dice = np.asarray(
        [_finite_metric(approach_by_case[str(row.get("case_id"))], "dice") for row in positives]
    )
    controls = [row for row in cnn_rows if row.get("control_kind") != "positive"]
    required_controls = {"empty_vial", "identical"}
    controls_by_kind = {str(row.get("control_kind")): row for row in controls}
    controls_pass = required_controls <= set(controls_by_kind) and all(
        row.get("abstained") is True and row.get("mask_pixel_count") == 0
        for row in controls_by_kind.values()
    )
    metrics = {
        "mean_dice": float(np.mean(cnn_dice)),
        "p05_dice": float(np.quantile(cnn_dice, 0.05)),
        "visible_height_error_px_max": float(np.max(height_errors)),
        "approach_a_mean_dice": float(np.mean(approach_dice)),
    }
    gates = {
        "mean_dice": metrics["mean_dice"] >= 0.85,
        "p05_dice": metrics["p05_dice"] >= 0.70,
        "visible_height_error": metrics["visible_height_error_px_max"] <= 5.0,
        "empty_and_identical_controls": controls_pass,
        "beats_approach_a": metrics["mean_dice"] > metrics["approach_a_mean_dice"],
    }
    low_fill = [row for row in positives if row.get("low_fill_first_dispense") is True]
    return {
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": "replace_approach_a" if all(gates.values()) else "retain_approach_a",
        "low_fill_first_dispense": {
            "count": len(low_fill),
            "mean_dice": (
                float(np.mean([_finite_metric(row, "dice") for row in low_fill]))
                if low_fill
                else None
            ),
        },
    }


__all__ = [
    "DEFAULT_THRESHOLD",
    "IMAGENET_WEIGHTS_URL",
    "ResNet18UNet",
    "adapt_resnet18_stem_weight",
    "paired_augment",
    "predict_pair",
    "probability_to_visible_prediction",
    "score_holdout_comparison",
]

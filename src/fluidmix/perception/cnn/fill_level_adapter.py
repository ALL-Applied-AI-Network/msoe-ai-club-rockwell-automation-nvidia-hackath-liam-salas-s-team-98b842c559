"""Use Joseph LoDuca's fill-level U-Net (branch codex/fill-level-cnn, Revision 2) as the liquid mask.

His model scores four classes (empty 0, liquid 1, foam 2, blocked 3) on a fixed vial crop of
one FillCam RGB frame. This adapter returns the same EmptyReferencePrediction the tracker
already consumes: the liquid-class probability is placed back into the full 640x480 frame
and passed through the unchanged ROI/clamp rule. The empty-vial reference is not used by
his model; it is kept only for the identical-frame and black-frame controls.

Enabled by FLUIDMIX_FILL_CNN_WEIGHTS=<checkpoint path>. Development use; disclosed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fluidmix.perception.cnn.cnn_predictor import (
    IMAGE_SHAPE, EmptyReferencePrediction, _empty, probability_to_visible_prediction,
)

CROP_XYXY = (256, 48, 368, 464)
LIQUID_CLASS = 1
WEIGHTS_ENV = "FLUIDMIX_FILL_CNN_WEIGHTS"


def build_unet(num_classes: int = 4, base_channels: int = 16) -> nn.Module:
    """The four-level U-Net from codex/fill-level-cnn `fill_level_training.build_unet`."""

    class DoubleConv(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = int(base_channels)
            self.enc1 = DoubleConv(3, b)
            self.enc2 = DoubleConv(b, b * 2)
            self.enc3 = DoubleConv(b * 2, b * 4)
            self.enc4 = DoubleConv(b * 4, b * 8)
            self.bottleneck = DoubleConv(b * 8, b * 16)
            self.pool = nn.MaxPool2d(2)
            self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
            self.dec4 = DoubleConv(b * 16, b * 8)
            self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
            self.dec3 = DoubleConv(b * 8, b * 4)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
            self.dec2 = DoubleConv(b * 4, b * 2)
            self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
            self.dec1 = DoubleConv(b * 2, b)
            self.head = nn.Conv2d(b, int(num_classes), 1)

        def forward(self, value):
            e1 = self.enc1(value)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            bottleneck = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat((self.up4(bottleneck), e4), dim=1))
            d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
            d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
            d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
            return self.head(d1)

    return UNet()


def configured_weights() -> Path | None:
    value = os.environ.get(WEIGHTS_ENV)
    return Path(value) if value else None


def load_fill_level_model(path: str | Path, device: str) -> nn.Module:
    """Load a state dict, or a checkpoint dict holding one, into the U-Net."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model_state", "model"):
                if isinstance(checkpoint.get(key), dict):
                    state = checkpoint[key]
                    break
        model = build_unet()
        model.load_state_dict(state)
    return model.to(torch.device(device)).eval()


@torch.no_grad()
def predict_fill_level_pair(
    model: nn.Module, current_rgb: object, reference_rgb: object, *,
    threshold: float = 0.5, device: str = "cpu",
) -> EmptyReferencePrediction:
    """Same contract as cnn_predictor.predict_pair, backed by the fill-level U-Net."""
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
    x0, y0, x1, y1 = CROP_XYXY
    crop = np.ascontiguousarray(current[y0:y1, x0:x1])
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
    logits = model(tensor.to(torch.device(device)))
    liquid = torch.softmax(logits, dim=1)[0, LIQUID_CLASS].float().cpu().numpy()
    probability = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    probability[y0:y1, x0:x1] = liquid
    return probability_to_visible_prediction(probability, threshold=threshold)

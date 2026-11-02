"""Synthetic-only dense cap segmentation with explicit mask-to-instance rules."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    """Two local convolutions without running batch statistics."""

    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1),
                         nn.GroupNorm(2, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(2, outputs), nn.SiLU())


class DenseCapNet(nn.Module):
    """Small U-Net with native-resolution output and no pretrained parameters.

    Run41: base width 12 instead of 8 at every level (GroupNorm keeps 2 groups).
    """

    def __init__(self):
        super().__init__()
        w = 12  # run41: base width (was 8); every level scales with it
        self.enc0 = Block(3, w)
        self.enc1 = Block(w, 2*w)
        self.enc2 = Block(2*w, 4*w)
        self.bottom = Block(4*w, 8*w)
        self.dec2 = Block(12*w, 4*w)
        self.dec1 = Block(6*w, 2*w)
        self.dec0 = Block(3*w, w)
        self.head = nn.Conv2d(w, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, image):
        """Return one foreground logit per original pixel."""
        if tuple(image.shape[-2:]) != (480, 640):
            raise ValueError('Native 480x640 frame required')
        a = self.enc0(image)
        b = self.enc1(F.avg_pool2d(a, 2))
        c = self.enc2(F.avg_pool2d(b, 2))
        z = self.bottom(F.avg_pool2d(c, 2))
        for block, skip in ((self.dec2, c), (self.dec1, b), (self.dec0, a)):
            z = block(torch.cat((F.interpolate(z, size=skip.shape[-2:], mode='nearest'), skip), 1))
        return self.head(z)


def dense_loss(logits, target):
    """Supervise every pixel, including all pixels in an empty frame.

    Fixed positive weight 50 addresses sparse caps. Soft Dice is computed over
    the complete batch and is added with unit weight. Neither term crops boxes.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=logits.new_tensor(50.0))
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (
        probability.sum() + target.sum() + 1)
    return bce + dice


def instances(probability, minimum_pixels=9):
    """Convert a native probability map to components of at least nine pixels.

    Threshold is fixed at 0.5 and minimum predicted component area is nine.
    There is no position, border, confidence, morphology or track-dependent filter.
    Small eligible targets can still be missed; all truth remains in scoring.
    Touching components merge, which remains a scored failure when appropriate.
    """
    if type(minimum_pixels) is not int or minimum_pixels < 1:
        raise ValueError('Positive integer component minimum required')
    if (probability.shape != (480, 640) or not np.isfinite(probability).all()
            or probability.min() < 0 or probability.max() > 1):
        raise ValueError('Finite native probability map in [0,1] required')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= .5).astype(np.uint8), connectivity=8)
    boxes, scores, masks = [], [], []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < minimum_pixels:
            continue
        mask = labels == index
        boxes.append([int(x), int(y), int(x+w), int(y+h)])
        scores.append(float(probability[mask].mean()))
        masks.append(mask)
    return boxes, scores, np.asarray(masks, bool).reshape(-1, 480, 640)


def image_tensor(images, device):
    """Convert full BGR source images to normalized RGB tensors without resizing."""
    if any(im.shape != (480, 640, 3) or im.dtype != np.uint8 for im in images):
        raise ValueError('Native uint8 BGR images required')
    array = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in images])
    return torch.from_numpy(array.copy()).to(device=device, dtype=torch.float32) / 255

"""TRAIN-only instance-area weighting; inference never receives these labels."""
import math

import numpy as np
import torch
from torch.nn import functional as F


def foreground_weights(mask, objects, mean_area):
    """Return native float32 weights after validating each discrete instance.

    Args:
        mask: Native uint16 instance-label plane, with zero background.
        objects: Manifest objects with unique positive mask_value and pixels.
        mean_area: Frozen mean area from the complete TRAIN population.

    Returns:
        A float32 plane; background is one, foreground is mean_area/area.
    """
    if mask.shape != (480, 640) or mask.dtype != np.uint16:
        raise ValueError('Native uint16 instance labels required')
    if not math.isfinite(mean_area) or mean_area <= 0:
        raise ValueError('Positive finite TRAIN mean required')
    labels = []
    for obj in objects:
        value, area = obj['mask_value'], obj['pixels']
        if (type(value) is not int or not 0 < value <= 65535
                or type(area) is not int or area < 1):
            raise ValueError('Positive discrete instance label and area required')
        labels.append(value)
    if len(set(labels)) != len(labels) or set(np.unique(mask)) - {0} != set(labels):
        raise ValueError('Instance inventory mismatch')
    weights = np.ones(mask.shape, np.float32)
    for obj in objects:
        region = mask == obj['mask_value']
        if int(region.sum()) != obj['pixels']:
            raise ValueError('Instance area mismatch')
        weights[region] = mean_area / obj['pixels']
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Invalid computed weights')
    return weights


def instance_balanced_loss(logits, target, weights):
    """Apply area weights to foreground BCE and retain original global Dice.

    The BCE denominator is the number of native batch pixels. Weights are
    constants from labels, never learned or normalized within the batch.
    """
    if logits.shape != target.shape or weights.shape != target.shape:
        raise ValueError('Aligned logit, target and weight tensors required')
    if (weights.requires_grad or target.requires_grad
            or not torch.isfinite(weights).all() or not (weights > 0).all()
            or not ((target == 0) | (target == 1)).all()
            or not (weights[target == 0] == 1).all()):
        raise ValueError('Invalid fixed target/weight tensors')
    # Keep original reduction/backward arithmetic for background and unit
    # weights. The correction is zero there, and only redistributes foreground.
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=logits.new_tensor(50.0))
    bce = bce + (50 * target * (weights - 1) * F.softplus(-logits)).mean()
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (
        probability.sum() + target.sum() + 1)
    return bce + dice

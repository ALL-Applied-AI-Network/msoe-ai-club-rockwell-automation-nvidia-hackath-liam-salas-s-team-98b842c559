"""Score visible instance masks at fixed thresholds without background inflation."""

import numpy as np


def match_iou(matrix, threshold):
    """Return maximum-cardinality prediction/truth pairs above a fixed IoU.

    Args:
        matrix: Finite prediction-by-truth IoUs in [0, 1].
        threshold: Required IoU, in (0, 1].
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or not np.isfinite(matrix).all()
            or np.any((matrix < 0) | (matrix > 1)) or not 0 < threshold <= 1):
        raise ValueError('Invalid IoU matrix or threshold')
    edges = [sorted(np.flatnonzero(row >= threshold), key=lambda j: (-row[j], j))
             for row in matrix]
    owners = {}

    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or visit(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(matrix)):
        visit(i, set())
    return [(int(i), int(j)) for j, i in sorted(owners.items())]


def mask_ious(predictions, truths):
    """Compute exact pixel IoUs, accepting empty predictions but not empty truth."""
    predictions, truths = np.asarray(predictions), np.asarray(truths)
    if (predictions.dtype != bool or truths.dtype != bool
            or predictions.ndim != 3 or truths.ndim != 3
            or predictions.shape[1:] != truths.shape[1:]):
        raise ValueError('Boolean NxHxW mask arrays with matching dimensions required')
    if len(truths) and np.any(truths.sum(axis=(1, 2)) == 0):
        raise ValueError('Truth instances must be nonempty')
    matrix = np.zeros((len(predictions), len(truths)), dtype=float)
    for i, prediction in enumerate(predictions):
        for j, truth in enumerate(truths):
            intersection = np.count_nonzero(prediction & truth)
            union = np.count_nonzero(prediction | truth)
            matrix[i, j] = intersection / union if union else 0.
    return matrix


def score_masks(predictions, truths, thresholds=(.5, .75, .9)):
    """Count misses and duplicates separately, with no true-negative pixels.

    Mean best IoU includes every truth, including missed objects with IoU zero.
    It is descriptive and may reuse a prediction; the detection counts never do.
    """
    matrix = mask_ious(predictions, truths)
    scores = {}
    for threshold in thresholds:
        matched = match_iou(matrix, threshold)
        tp = len(matched)
        scores[str(threshold)] = dict(TP=tp, FP=len(predictions)-tp,
                                     FN=len(truths)-tp)
    best = matrix.max(axis=0) if len(predictions) else np.zeros(len(truths))
    return dict(thresholds=scores, truth_instances=len(truths),
                predicted_instances=len(predictions),
                best_iou_sum=float(best.sum()),
                mean_best_iou=float(best.mean()) if len(truths) else None)

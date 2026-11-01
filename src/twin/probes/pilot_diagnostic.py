"""Frozen oracle-mask Lab visibility diagnostic for the eight-row pilot.

This is an acceptance diagnostic for synthetic material visibility.  It is not
a detector metric and the oracle mask is never exposed as an inference input.
"""

from __future__ import annotations

from itertools import combinations
import json
import math
import platform
from pathlib import Path

import numpy as np
from PIL import __version__ as PIL_VERSION

from fluidmix.core.paths import REPO
from .pilot_dataset import (
    CONFIG_PATH,
    _require_root,
    _write_exclusive,
    load_prepublication_diagnostic_inputs,
)


DIAGNOSTIC_FILE = "pilot-diagnostic.json"
_A_REPEAT_IDS = ("R003", "R004", "R006", "R007")


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.linalg.norm(left - right))
    if not math.isfinite(value):
        raise ValueError("Lab distance is nonfinite")
    return value


def _six(value: float | None) -> str | None:
    return None if value is None else format(value, ".6f")


def _vector_six(value: np.ndarray) -> list[str]:
    return [_six(float(component)) for component in value]  # type: ignore[list-item]


def evaluate_pilot_visibility(
    pilot_root: str | Path,
    *,
    output_path: str | Path | None = None,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> Path:
    """Compute and exclusively publish the preregistered Lab diagnostic."""

    root = _require_root(pilot_root, role="pilot root")
    output = root / DIAGNOSTIC_FILE if output_path is None else Path(output_path)
    if output.parent.resolve(strict=True) != root or output.name != DIAGNOSTIC_FILE:
        raise ValueError("diagnostic output must be pilot-root/pilot-diagnostic.json")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    inputs = load_prepublication_diagnostic_inputs(
        root, config_path=config_path, repo_root=repo_root
    )
    if len(inputs) != 8:
        raise ValueError("diagnostic requires the exact eight-row dataset")
    common_mask = inputs[0]["mask"]
    if not isinstance(common_mask, np.ndarray) or common_mask.dtype != np.dtype(np.bool_):
        raise ValueError("diagnostic common mask is malformed")

    # Imported here so config/graph/checker help paths remain independent of
    # the analysis-only scikit-image dependency.
    import skimage
    from skimage.color import rgb2lab

    descriptors: dict[str, np.ndarray] = {}
    row_reports: list[dict[str, object]] = []
    for record in inputs:
        sample_id = record["sample_id"]
        rgb = record["rgb"]
        mask = record["mask"]
        if not np.array_equal(mask, common_mask):
            raise ValueError("diagnostic input join/mask invariance failed")
        assert isinstance(sample_id, str) and isinstance(rgb, np.ndarray)
        srgb = rgb.astype(np.float64) / 255.0
        lab = rgb2lab(
            srgb,
            illuminant="D65",
            observer="2",
            channel_axis=-1,
        )
        descriptor = np.median(lab[common_mask], axis=0).astype(np.float64, copy=False)
        if descriptor.shape != (3,) or not np.all(np.isfinite(descriptor)):
            raise ValueError("diagnostic Lab descriptor is malformed or nonfinite")
        descriptors[sample_id] = descriptor
        row_reports.append(
            {
                "sample_id": sample_id,
                "rgb_sha256": record["rgb_sha256"],
                "mask_sha256": record["mask_sha256"],
                "median_lab": [float(value) for value in descriptor],
                "median_lab_6dp": _vector_six(descriptor),
            }
        )

    pooled_a = np.median(
        np.stack([descriptors[sample_id] for sample_id in _A_REPEAT_IDS]), axis=0
    ).astype(np.float64, copy=False)
    pooled_b = np.median(
        np.stack([descriptors["R000"], descriptors["R001"]]), axis=0
    ).astype(np.float64, copy=False)
    pair_reports: list[dict[str, object]] = []
    pair_values: list[float] = []
    for left_id, right_id in combinations(_A_REPEAT_IDS, 2):
        value = _distance(descriptors[left_id], descriptors[right_id])
        pair_values.append(value)
        pair_reports.append(
            {
                "left_sample_id": left_id,
                "right_sample_id": right_id,
                "delta_e76": value,
                "delta_e76_6dp": _six(value),
            }
        )
    max_a_spread = max(pair_values)
    r005_to_a = {
        sample_id: _distance(descriptors["R005"], descriptors[sample_id])
        for sample_id in _A_REPEAT_IDS
    }
    beyond_repeats = all(value > max_a_spread for value in r005_to_a.values())

    displacement = pooled_b - pooled_a
    norm = float(np.linalg.norm(displacement))
    direction_valid = math.isfinite(norm) and norm > 0.0
    if direction_valid:
        direction = displacement / norm
        projection_r005 = float(np.dot(descriptors["R005"] - pooled_a, direction))
        projection_r002 = float(np.dot(descriptors["R002"] - pooled_a, direction))
        projections_finite = math.isfinite(projection_r005) and math.isfinite(projection_r002)
        projection_order = bool(
            projections_finite and 0.0 < projection_r005 < projection_r002
        )
    else:
        direction = None
        projection_r005 = None
        projection_r002 = None
        projection_order = False

    a_to_b = _distance(pooled_a, pooled_b)
    r005_to_b = _distance(descriptors["R005"], pooled_b)
    r002_to_b = _distance(descriptors["R002"], pooled_b)
    distance_order = a_to_b > r005_to_b > r002_to_b
    gates = {
        "all_mask_hashes_identical": len({row["mask_sha256"] for row in row_reports}) == 1,
        "descriptors_finite": all(np.all(np.isfinite(value)) for value in descriptors.values()),
        "pooled_direction_nonzero_finite": direction_valid,
        "r005_beyond_every_a_repeat": beyond_repeats,
        "projection_order_a_r005_r002": projection_order,
        "distance_to_b_strictly_decreases": distance_order,
    }
    result = {
        "schema_version": 1,
        "diagnostic": "oracle_mask_lab_visibility_v1",
        "purpose": "acceptance_only_visibility_diagnostic",
        "detector_metric": False,
        "rows": row_reports,
        "pooled": {
            "A": [float(value) for value in pooled_a],
            "A_6dp": _vector_six(pooled_a),
            "B": [float(value) for value in pooled_b],
            "B_6dp": _vector_six(pooled_b),
            "a_to_b_norm": norm,
            "a_to_b_norm_6dp": _six(norm),
            "direction_unit_lab": (
                None if direction is None else [float(value) for value in direction]
            ),
        },
        "a_repeatability": {
            "pair_distances": pair_reports,
            "max_delta_e76": max_a_spread,
            "max_delta_e76_6dp": _six(max_a_spread),
        },
        "r005_distances_to_a_repeats": {
            sample_id: {
                "delta_e76": value,
                "delta_e76_6dp": _six(value),
            }
            for sample_id, value in r005_to_a.items()
        },
        "projections": {"R002": projection_r002, "R005": projection_r005},
        "projections_6dp": {
            "R002": _six(projection_r002),
            "R005": _six(projection_r005),
        },
        "distances_to_pooled_b": {
            "pooled_A": {"delta_e76": a_to_b, "delta_e76_6dp": _six(a_to_b)},
            "R005": {"delta_e76": r005_to_b, "delta_e76_6dp": _six(r005_to_b)},
            "R002": {"delta_e76": r002_to_b, "delta_e76_6dp": _six(r002_to_b)},
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
        "versions": {
            "numpy": np.__version__,
            "Pillow": PIL_VERSION,
            "scikit_image": skimage.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    encoded = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _write_exclusive(output, encoded)

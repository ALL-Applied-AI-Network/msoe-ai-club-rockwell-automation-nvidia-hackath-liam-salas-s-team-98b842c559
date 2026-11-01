"""Deterministic conservative diffusion of particle liquid compositions.

This module is deliberately renderer-independent.  It models only a bounded
composition exchange on a radius graph; it does not infer absorbance, colour,
or physical fluid properties from the result.

The explicit pair update uses the frozen dimensionless factor

``alpha = D * dt / r**2``

where ``D = 1e-5 m2/s``, ``dt = 1/60 s`` and ``r = 0.006 m``.  Thus
``alpha = 0.00462962962962963``.  For a particle of degree ``d``, the sum of
outgoing pair weights is bounded by ``0.25`` via
``weight = min(alpha, 0.25 / max_degree)``.  This deterministic quarter-row
cap leaves at least 75% of every component at its source particle, so one
explicit exchange is non-negative while symmetric pair updates conserve every
component globally.  It is a numerical stability rule, not a calibration.

Rows stay aligned with the supplied stable particle IDs.  IDs are validated
and included in receipts, but are never used to reorder or reassign rows.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


EFFECTIVE_DIFFUSIVITY_M2_S = 1.0e-5
DIFFUSION_NEIGHBOR_RADIUS_M = 0.006
DIFFUSION_DT_S = 1.0 / 60.0
DIFFUSION_STABILITY_CAP_PER_ROW = 0.25
# Descriptive aliases matching the vocabulary used by the existing PhysX plan.
EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S = EFFECTIVE_DIFFUSIVITY_M2_S
DIFFUSION_RADIUS_M = DIFFUSION_NEIGHBOR_RADIUS_M
DIFFUSION_STEP_FACTOR = (
    EFFECTIVE_DIFFUSIVITY_M2_S * DIFFUSION_DT_S / DIFFUSION_NEIGHBOR_RADIUS_M**2
)
COMPOSITION_COMPONENTS = ("t200", "t400", "t100")
INPUT_TOLERANCE = 1.0e-12

__all__ = [
    "COMPOSITION_COMPONENTS",
    "DIFFUSION_DT_S",
    "DIFFUSION_NEIGHBOR_RADIUS_M",
    "DIFFUSION_RADIUS_M",
    "DIFFUSION_STEP_FACTOR",
    "DIFFUSION_STABILITY_CAP_PER_ROW",
    "EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S",
    "EFFECTIVE_DIFFUSIVITY_M2_S",
    "build_update_receipt",
    "diffuse_particle_compositions",
    "diffuse_particle_compositions_with_receipt",
    "update_particle_compositions",
    "update_particle_compositions_with_receipt",
]


def _numeric_matrix(value: object, name: str, row_count: int | None = None) -> np.ndarray:
    """Convert a finite numeric Nx3 matrix without accepting booleans/objects."""

    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers, not booleans or objects")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if row_count is not None and array.shape[0] != row_count:
        raise ValueError(f"{name} row count must match particle IDs")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validate_ids(particle_ids: object) -> np.ndarray:
    """Validate one stable scalar ID per row and return a one-dimensional copy."""

    ids = np.asarray(particle_ids)
    if ids.ndim != 1:
        raise ValueError("particle_ids must have shape (N,)")
    if ids.dtype.kind in "bfc":
        if ids.dtype.kind == "f" and not np.all(np.isfinite(ids)):
            raise ValueError("particle_ids must be finite")
        # Complex and boolean identifiers are needlessly ambiguous.
        raise TypeError("particle_ids must contain scalar integer or string-like IDs")

    seen: set[tuple[str, str]] = set()
    for value in ids.tolist():
        if isinstance(value, (list, dict, set, tuple)):
            raise TypeError("particle_ids must contain scalar values")
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError("particle_ids must contain hashable scalar values") from exc
        key = (type(value).__qualname__, repr(value))
        if key in seen:
            raise ValueError("particle_ids must be unique; duplicate ID found")
        seen.add(key)
    return ids.copy()


def _validated_compositions(value: object, row_count: int | None = None) -> np.ndarray:
    """Validate and canonicalize composition rows on the three-component simplex."""

    rows = _numeric_matrix(value, "compositions", row_count)
    if np.any(rows < -INPUT_TOLERANCE):
        raise ValueError("compositions must be non-negative")
    # Tiny round-off below zero is tolerated at the declared input tolerance;
    # larger negatives abstain above.  Canonicalizing here makes the update's
    # invariants independent of harmless decimal representation noise.
    rows = np.maximum(rows, 0.0)
    sums = rows.sum(axis=1)
    if not np.all(np.abs(sums - 1.0) <= INPUT_TOLERANCE):
        raise ValueError("each compositions row must sum to one")
    if np.any(sums <= 0.0):
        raise ValueError("each compositions row must have positive mass")
    return rows / sums[:, None]


def _validate_inputs(
    particle_ids: object, positions_m: object, compositions: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = _validate_ids(particle_ids)
    positions = _numeric_matrix(positions_m, "positions_m", ids.shape[0])
    rows = _validated_compositions(compositions, ids.shape[0])
    return ids, positions, rows


def _id_digest(ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in ids.tolist():
        encoded = f"{type(value).__module__}.{type(value).__qualname__}:{value!r}".encode(
            "utf-8"
        )
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _component_totals(rows: np.ndarray) -> np.ndarray:
    """Return accurately reduced component totals for conservation checks.

    F2 carries 2,221 rows. A naive float64 reduction can accumulate more than
    the 1e-12 invariant tolerance even when every pair exchange is symmetric.
    ``math.fsum`` isolates that summation noise without loosening the gate.
    """

    return np.asarray(
        [math.fsum(float(value) for value in rows[:, column]) for column in range(3)],
        dtype=np.float64,
    )


def _radius_neighbor_pairs(positions: np.ndarray) -> np.ndarray:
    """Return canonical ``i < j`` pairs whose Euclidean distance is <= radius."""

    if positions.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    pairs = cKDTree(positions).query_pairs(
        DIFFUSION_NEIGHBOR_RADIUS_M,
        eps=0.0,
        output_type="ndarray",
    )
    pairs = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    if pairs.shape[0] > 1:
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return pairs


def _diffuse_validated(
    positions: np.ndarray, rows: np.ndarray, *, dt_s: float = DIFFUSION_DT_S
) -> tuple[np.ndarray, int, int, float, float]:
    pairs = _radius_neighbor_pairs(positions)
    degrees = np.bincount(pairs.reshape(-1), minlength=rows.shape[0])
    max_degree = int(degrees.max()) if degrees.size else 0
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    step_factor = EFFECTIVE_DIFFUSIVITY_M2_S * float(dt_s) / DIFFUSION_NEIGHBOR_RADIUS_M**2
    pair_weight = (
        0.0
        if max_degree == 0
        else min(step_factor, DIFFUSION_STABILITY_CAP_PER_ROW / float(max_degree))
    )

    result = rows.copy()
    if pairs.shape[0]:
        first = np.asarray(pairs[:, 0], dtype=np.int64)
        second = np.asarray(pairs[:, 1], dtype=np.int64)
        exchange = pair_weight * (rows[second] - rows[first])
        np.add.at(result, first, exchange)
        np.add.at(result, second, -exchange)

    if not np.all(np.isfinite(result)) or np.any(result < -INPUT_TOLERANCE):
        raise RuntimeError("diffusion violated finite non-negative composition")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=INPUT_TOLERANCE):
        raise RuntimeError("diffusion violated composition simplex row sums")
    if not np.allclose(
        _component_totals(result),
        _component_totals(rows),
        rtol=0.0,
        atol=INPUT_TOLERANCE,
    ):
        raise RuntimeError("diffusion violated global component conservation")
    if np.any(result < 0.0):
        result = np.maximum(result, 0.0)
    return result, len(pairs), max_degree, step_factor, pair_weight


def diffuse_particle_compositions(
    particle_ids: object, positions_m: object, compositions: object, *, dt_s: float = DIFFUSION_DT_S
) -> np.ndarray:
    """Return one deterministic conservative diffusion update.

    The returned rows retain exactly the input order and therefore remain
    associated with the same stable particle IDs.  No renderer or colour
    conversion is performed.
    """

    _ids, positions, rows = _validate_inputs(particle_ids, positions_m, compositions)
    result, _neighbors, _degree, _step, _weight = _diffuse_validated(positions, rows, dt_s=dt_s)
    return result


def build_update_receipt(
    particle_ids: object,
    positions_m: object,
    before_compositions: object,
    after_compositions: object,
    *,
    neighbor_count: int,
    max_degree: int | None = None,
    step_factor: float | None = None,
    pair_weight: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe receipt for one validated composition update."""

    ids, positions, before = _validate_inputs(
        particle_ids, positions_m, before_compositions
    )
    after = _validated_compositions(after_compositions, ids.shape[0])
    if isinstance(neighbor_count, bool) or not isinstance(neighbor_count, (int, np.integer)):
        raise TypeError("neighbor_count must be a non-negative integer")
    if neighbor_count < 0:
        raise ValueError("neighbor_count must be non-negative")
    if max_degree is not None and max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    component_delta = _component_totals(after) - _component_totals(before)
    row_delta = after.sum(axis=1) - 1.0
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "particle_count": int(ids.shape[0]),
        "component_order": list(COMPOSITION_COMPONENTS),
        "neighbor_count": int(neighbor_count),
        "conservation_error": float(np.max(np.abs(component_delta), initial=0.0)),
        "component_conservation_error": [float(value) for value in component_delta],
        "row_sum_error": float(np.max(np.abs(row_delta), initial=0.0)),
        "component_conservation_passed": bool(
            np.max(np.abs(component_delta), initial=0.0) <= INPUT_TOLERANCE
        ),
        "row_sum_passed": bool(
            np.max(np.abs(row_delta), initial=0.0) <= INPUT_TOLERANCE
        ),
        "nonnegative_passed": bool(np.all(after >= 0.0)),
        "particle_row_order_preserved": True,
        "min_composition": float(np.min(after)) if after.size else 0.0,
        "max_composition": float(np.max(after)) if after.size else 0.0,
        "particle_ids_sha256": _id_digest(ids),
        "positions_sha256": _array_digest(positions),
        "before_sha256": _array_digest(before),
        "after_sha256": _array_digest(after),
    }
    if max_degree is not None:
        receipt["max_degree"] = int(max_degree)
    if step_factor is not None:
        receipt["step_factor"] = float(step_factor)
    if pair_weight is not None:
        receipt["pair_weight"] = float(pair_weight)
    return receipt


def diffuse_particle_compositions_with_receipt(
    particle_ids: object, positions_m: object, compositions: object, *, dt_s: float = DIFFUSION_DT_S
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a diffusion update and its conservation/hash receipt."""

    ids, positions, rows = _validate_inputs(particle_ids, positions_m, compositions)
    result, neighbors, max_degree, step_factor, pair_weight = _diffuse_validated(
        positions, rows, dt_s=dt_s
    )
    receipt = build_update_receipt(
        ids,
        positions,
        rows,
        result,
        neighbor_count=neighbors,
        max_degree=max_degree,
        step_factor=step_factor,
        pair_weight=pair_weight,
    )
    return result, receipt


# Explicitly named aliases keep the public boundary readable to callers that
# think of this operation as an update rather than a diffusion primitive.
update_particle_compositions = diffuse_particle_compositions
update_particle_compositions_with_receipt = diffuse_particle_compositions_with_receipt

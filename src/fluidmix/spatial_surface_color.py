"""Project tracked particle compositions onto a generated surface mesh.

This pure module does not import Isaac Sim or author USD.  It defines the
fail-closed numeric boundary used by a future renderer capability probe.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from fluidmix.particle_palette import (
    COMPOSITION_COMPONENTS,
    DEFAULT_SOURCE_COLORS,
    INPUT_TOLERANCE,
    MIXING_RULE,
    mix_linear_rgb,
)


# The implementation chunks both axes; 512 bounds the largest temporary
# ``(vertices, particles, xyz)`` float64 delta block to roughly 6 MiB.
DEFAULT_CHUNK_SIZE = 512
SURFACE_MAPPING_RULE = "nearest_particle_squared_euclidean_smallest_id_v1"

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "SURFACE_MAPPING_RULE",
    "build_surface_vertex_colors",
]


def _validated_ids(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("particle_ids must have shape (N,)")
    if raw.shape[0] == 0:
        raise ValueError("particle_ids must not be empty")
    if raw.dtype.kind not in "iu":
        raise TypeError("particle_ids must contain integers, not booleans or objects")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("particle_ids must be representable as signed 64-bit integers")
    ids = np.asarray(raw, dtype=np.int64)
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise ValueError("particle_ids must be unique; duplicate ID found")
    return ids.copy()


def _numeric_matrix(value: object, name: str, *, rows: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers, not booleans or objects")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count must match particle_ids")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _validated_compositions(value: object, *, rows: int) -> np.ndarray:
    compositions = _numeric_matrix(value, "particle_compositions", rows=rows)
    if np.any(compositions < -INPUT_TOLERANCE):
        raise ValueError("particle_compositions must be non-negative")
    compositions = np.maximum(compositions, 0.0)
    totals = compositions.sum(axis=1)
    if np.any(totals <= 0.0) or not np.all(
        np.abs(totals - 1.0) <= INPUT_TOLERANCE
    ):
        raise ValueError("each particle_compositions row must sum to one")
    return compositions / totals[:, None]


def _validated_chunk_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("chunk_size must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return result


def _sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.asarray(array, dtype=dtype, order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _component_totals(rows: np.ndarray) -> list[float]:
    return [
        math.fsum(float(value) for value in rows[:, column])
        for column in range(rows.shape[1])
    ]


def build_surface_vertex_colors(
    particle_ids: object,
    particle_positions_m: object,
    particle_compositions: object,
    mesh_vertices_m: object,
    *,
    chunk_size: object = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return surface RGB, nearest stable particle IDs, and hash evidence.

    Vertices and particles are both processed in bounded chunks.  Exact
    squared-distance ties select the smallest stable particle ID, independent
    of input row order and chunk size.  RGB values use the project's fixed,
    uncalibrated linear-light tank appearance palette.

    This is a surface visualization rule, not a volumetric interpolation,
    physical dye model, absorbance fit, or conservation operation.
    """

    ids = _validated_ids(particle_ids)
    positions = _numeric_matrix(
        particle_positions_m, "particle_positions_m", rows=ids.shape[0]
    )
    compositions = _validated_compositions(
        particle_compositions, rows=ids.shape[0]
    )
    vertices = _numeric_matrix(mesh_vertices_m, "mesh_vertices_m")
    bound = _validated_chunk_size(chunk_size)

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_positions = positions[order]
    sorted_compositions = compositions[order]

    vertex_count = vertices.shape[0]
    nearest_ids = np.empty(vertex_count, dtype=np.int64)
    nearest_rows = np.empty(vertex_count, dtype=np.int64)
    nearest_distances_squared = np.empty(vertex_count, dtype=np.float64)

    for vertex_start in range(0, vertex_count, bound):
        vertex_stop = min(vertex_start + bound, vertex_count)
        vertex_chunk = vertices[vertex_start:vertex_stop]
        chunk_rows = vertex_stop - vertex_start
        best_distances = np.full(chunk_rows, np.inf, dtype=np.float64)
        best_ids = np.full(chunk_rows, np.iinfo(np.int64).max, dtype=np.int64)
        best_rows = np.full(chunk_rows, -1, dtype=np.int64)

        for particle_start in range(0, sorted_ids.shape[0], bound):
            particle_stop = min(particle_start + bound, sorted_ids.shape[0])
            delta = (
                vertex_chunk[:, None, :]
                - sorted_positions[None, particle_start:particle_stop, :]
            )
            distances = np.einsum("vpi,vpi->vp", delta, delta)
            local_offsets = np.argmin(distances, axis=1)
            local_distances = distances[np.arange(chunk_rows), local_offsets]
            local_rows = particle_start + local_offsets
            local_ids = sorted_ids[local_rows]
            replace = (local_distances < best_distances) | (
                (local_distances == best_distances)
                & ((best_rows < 0) | (local_ids < best_ids))
            )
            best_distances[replace] = local_distances[replace]
            best_ids[replace] = local_ids[replace]
            best_rows[replace] = local_rows[replace]

        if np.any(best_rows < 0) or not np.all(np.isfinite(best_distances)):
            raise RuntimeError("nearest-particle surface mapping was incomplete")
        nearest_ids[vertex_start:vertex_stop] = best_ids
        nearest_rows[vertex_start:vertex_stop] = best_rows
        nearest_distances_squared[vertex_start:vertex_stop] = best_distances

    vertex_colors = mix_linear_rgb(sorted_compositions[nearest_rows])
    if vertex_colors.shape != vertices.shape or not np.all(np.isfinite(vertex_colors)):
        raise RuntimeError("surface vertex color mapping produced invalid RGB")

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "particle_count": int(ids.shape[0]),
        "mesh_vertex_count": int(vertices.shape[0]),
        "chunk_size": bound,
        "component_order": list(COMPOSITION_COMPONENTS),
        "mapping_rule": SURFACE_MAPPING_RULE,
        "distance_metric": "squared_euclidean_m2",
        "tie_break_rule": "smallest_stable_integer_particle_id",
        "particle_hash_order": "ascending_particle_id",
        "mixing_rule": MIXING_RULE,
        "color_space": "linear_rgb",
        "source_colors": DEFAULT_SOURCE_COLORS.tolist(),
        "calibrated": False,
        "physical_accuracy_claim": False,
        "volumetric_concentration_claim": False,
        "surface_projection_conserves_particle_mass": False,
        "finite_passed": True,
        "simplex_passed": True,
        "all_vertex_mappings_resolved": True,
        "particle_component_totals": _component_totals(sorted_compositions),
        "minimum_nearest_distance_squared_m2": float(
            np.min(nearest_distances_squared)
        ),
        "maximum_nearest_distance_squared_m2": float(
            np.max(nearest_distances_squared)
        ),
        "particle_ids_sha256": _sha256(sorted_ids, "<i8"),
        "particle_positions_sha256": _sha256(sorted_positions, "<f8"),
        "particle_compositions_sha256": _sha256(sorted_compositions, "<f8"),
        "mesh_vertices_sha256": _sha256(vertices, "<f8"),
        "nearest_particle_ids_sha256": _sha256(nearest_ids, "<i8"),
        "nearest_distances_squared_sha256": _sha256(
            nearest_distances_squared, "<f8"
        ),
        "vertex_colors_sha256": _sha256(vertex_colors, "<f8"),
        "source_colors_sha256": _sha256(DEFAULT_SOURCE_COLORS, "<f8"),
    }
    return vertex_colors, nearest_ids, evidence


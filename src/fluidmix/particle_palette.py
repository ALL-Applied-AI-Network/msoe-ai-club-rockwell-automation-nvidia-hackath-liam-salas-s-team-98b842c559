"""Deterministic composition-to-prototype colours for particle rendering.

This module is a renderer-independent bridge from a particle's three-component
simplex composition to a finite palette of transmissions and integer prototype
indices.  The colour rule is Beer-law extinction mixing: component extinctions
are fraction-weighted, then converted to transmission at a reference depth.
It is explicitly uncalibrated and is not a camera or chemistry validation.

The palette is a frozen ternary lattice.  ``resolution`` is the number of
equal intervals on each simplex edge, so the palette has
``(resolution + 1) * (resolution + 2) // 2`` prototypes.  Vertices are first
in component order, followed by the remaining lattice rows in deterministic
lexicographic order.  Nearest-lattice composition with ``numpy.argmin`` gives
stable first-prototype tie breaking; the resulting int64 array is suitable for
an Isaac PointInstancer ``protoIndices`` property.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final

import numpy as np


COMPOSITION_COMPONENTS: Final[tuple[str, str, str]] = ("t200", "t400", "t100")
DEFAULT_PALETTE_RESOLUTION: Final[int] = 4
INPUT_TOLERANCE: Final[float] = 1.0e-12
MIXING_RULE: Final[str] = "beer_law_extinction_weighted_transmission_v1"
DEFAULT_REFERENCE_DEPTH_M: Final[float] = 0.01

# These are appearance parameters for the synthetic twin only.  They are not
# measured tank colours or an absorbance calibration.
DEFAULT_SOURCE_COLORS = np.asarray(
    (
        (0.95, 0.38, 0.08),
        (0.18, 0.55, 0.75),
        (0.92, 0.98, 1.00),
    ),
    dtype=np.float64,
)
DEFAULT_SOURCE_EXTINCTIONS = -np.log(DEFAULT_SOURCE_COLORS) / DEFAULT_REFERENCE_DEPTH_M

__all__ = [
    "COMPOSITION_COMPONENTS",
    "DEFAULT_PALETTE_RESOLUTION",
    "DEFAULT_SOURCE_COLORS",
    "DEFAULT_SOURCE_EXTINCTIONS",
    "DEFAULT_REFERENCE_DEPTH_M",
    "INPUT_TOLERANCE",
    "MIXING_RULE",
    "build_particle_palette",
    "build_particle_palette_with_receipt",
    "build_simplex_palette",
    "map_compositions_to_palette",
    "mix_linear_rgb",
    "mix_extinction_transmission",
    "quantize_compositions_to_palette",
]


def _numeric_matrix(value: object, name: str) -> np.ndarray:
    """Convert a finite numeric ``(N, 3)`` matrix without accepting objects."""

    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers, not booleans or objects")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validated_compositions(value: object, name: str = "compositions") -> np.ndarray:
    """Validate and canonicalize rows on the three-component simplex."""

    rows = _numeric_matrix(value, name)
    if rows.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(rows < -INPUT_TOLERANCE):
        raise ValueError(f"{name} must be non-negative")
    # Match particle_diffusion.py's boundary convention: tiny negative roundoff
    # is harmless, but meaningful negative mass is rejected.
    rows = np.maximum(rows, 0.0)
    sums = rows.sum(axis=1)
    if not np.all(np.abs(sums - 1.0) <= INPUT_TOLERANCE):
        raise ValueError(f"each {name} row must sum to one")
    return rows / sums[:, None]


def _validated_resolution(resolution: object) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise TypeError("resolution must be a positive integer")
    result = int(resolution)
    if result <= 0:
        raise ValueError("resolution must be a positive integer")
    return result


def _validated_source_extinctions(source_extinctions: object | None) -> np.ndarray:
    extinctions = DEFAULT_SOURCE_EXTINCTIONS if source_extinctions is None else _numeric_matrix(
        source_extinctions, "source_extinctions"
    )
    if extinctions.shape != (3, 3):
        raise ValueError("source_extinctions must have shape (3, 3)")
    if np.any(extinctions < 0.0):
        raise ValueError("source_extinctions must be non-negative")
    return np.asarray(extinctions, dtype=np.float64).copy()


def _validated_source_colors(source_colors: object | None) -> np.ndarray:
    """Validate the archived linear-RGB helper's old source parameters."""

    colors = DEFAULT_SOURCE_COLORS if source_colors is None else _numeric_matrix(
        source_colors, "source_colors"
    )
    if colors.shape != (3, 3):
        raise ValueError("source_colors must have shape (3, 3)")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("source_colors must be in the RGB range [0, 1]")
    return np.asarray(colors, dtype=np.float64).copy()


def _validated_reference_depth_m(reference_depth_m: object) -> float:
    if isinstance(reference_depth_m, bool) or not isinstance(reference_depth_m, (int, float, np.integer, np.floating)):
        raise TypeError("reference_depth_m must be a real number")
    depth = float(reference_depth_m)
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("reference_depth_m must be finite and positive")
    return depth


def build_simplex_palette(resolution: object = DEFAULT_PALETTE_RESOLUTION) -> np.ndarray:
    """Return the deterministic ternary composition lattice.

    The first three rows are the one-hot component vertices.  This keeps the
    source prototype indices stable when a resolution is changed only by
    explicitly choosing another palette configuration.
    """

    denominator = _validated_resolution(resolution)
    integer_rows: list[tuple[int, int, int]] = [
        (denominator, 0, 0),
        (0, denominator, 0),
        (0, 0, denominator),
    ]
    seen = set(integer_rows)
    for first in range(denominator, -1, -1):
        for second in range(denominator - first, -1, -1):
            row = (first, second, denominator - first - second)
            if row not in seen:
                integer_rows.append(row)
                seen.add(row)
    return np.asarray(integer_rows, dtype=np.float64) / float(denominator)


def mix_extinction_transmission(
    compositions: object,
    source_extinctions: object | None = None,
    *,
    reference_depth_m: object = DEFAULT_REFERENCE_DEPTH_M,
) -> np.ndarray:
    """Return Beer-law RGB transmission for normalized component fractions."""

    rows = _validated_compositions(compositions)
    extinctions = _validated_source_extinctions(source_extinctions)
    depth = _validated_reference_depth_m(reference_depth_m)
    mixture_extinction = rows @ extinctions
    result = np.exp(-mixture_extinction * depth)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("Beer-law transmission left the finite [0, 1] range")
    return result


def mix_linear_rgb(
    compositions: object,
    source_colors: object | None = None,
) -> np.ndarray:
    """Archived linear-RGB helper for non-palette compatibility callers.

    The particle palette itself uses :func:`mix_extinction_transmission`.
    Retaining this helper prevents historical CPU diagnostics from silently
    changing their frozen appearance math.
    """

    rows = _validated_compositions(compositions)
    colors = _validated_source_colors(source_colors)
    result = rows @ colors
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("linear-light RGB mixture left the finite [0, 1] range")
    return result


def quantize_compositions_to_palette(
    compositions: object,
    *,
    resolution: object = DEFAULT_PALETTE_RESOLUTION,
    palette: object | None = None,
) -> np.ndarray:
    """Return stable nearest-prototype int64 indices for each composition row."""

    rows = _validated_compositions(compositions)
    prototypes = (
        build_simplex_palette(resolution)
        if palette is None
        else _validated_compositions(palette, "palette")
    )
    if prototypes.shape[0] == 0:
        raise ValueError("palette must not be empty")
    distances = np.sum((rows[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64, copy=False)


def map_compositions_to_palette(
    compositions: object,
    palette: object | None = None,
    *,
    resolution: object = DEFAULT_PALETTE_RESOLUTION,
) -> np.ndarray:
    """Compatibility spelling for :func:`quantize_compositions_to_palette`."""

    return quantize_compositions_to_palette(
        compositions, resolution=resolution, palette=palette
    )


def _array_sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.asarray(array, dtype=dtype, order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _build_receipt(
    compositions: np.ndarray,
    palette_compositions: np.ndarray,
    palette_colors: np.ndarray,
    proto_indices: np.ndarray,
    source_extinctions: np.ndarray,
    reference_depth_m: float,
    resolution: int,
) -> dict[str, Any]:
    prototype_count = int(palette_compositions.shape[0])
    particle_count = int(compositions.shape[0])
    finite_passed = bool(
        np.all(np.isfinite(compositions))
        and np.all(np.isfinite(palette_compositions))
        and np.all(np.isfinite(palette_colors))
    )
    nonnegative_passed = bool(
        np.all(compositions >= 0.0)
        and np.all(palette_compositions >= 0.0)
        and np.all(palette_colors >= 0.0)
    )
    simplex_passed = bool(
        np.allclose(compositions.sum(axis=1), 1.0, rtol=0.0, atol=INPUT_TOLERANCE)
        and np.allclose(
            palette_compositions.sum(axis=1), 1.0, rtol=0.0, atol=INPUT_TOLERANCE
        )
    )
    index_bounds_passed = bool(
        np.all(proto_indices >= 0)
        and np.all(proto_indices < prototype_count)
    )
    return {
        "schema_version": 1,
        "component_order": list(COMPOSITION_COMPONENTS),
        "resolution": resolution,
        "prototype_count": prototype_count,
        "particle_count": particle_count,
        "mixing_rule": MIXING_RULE,
        "color_space": "linear_rgb_transmission",
        "calibration_status": "uncalibrated",
        "absorbance_status": "uncalibrated",
        "chemistry_validated": False,
        "reference_path_source": "twin_authored_geometry",
        "calibrated": False,
        "physical_accuracy_claim": False,
        "finite_passed": finite_passed,
        "nonnegative_passed": nonnegative_passed,
        "simplex_passed": simplex_passed,
        "index_bounds_passed": index_bounds_passed,
        "compositions_sha256": _array_sha256(compositions, "<f8"),
        "palette_compositions_sha256": _array_sha256(palette_compositions, "<f8"),
        "palette_colors_sha256": _array_sha256(palette_colors, "<f8"),
        "proto_indices_sha256": _array_sha256(proto_indices, "<i8"),
        "source_extinctions_sha256": _array_sha256(source_extinctions, "<f8"),
        "source_extinctions": source_extinctions.tolist(),
        "reference_depth_m": reference_depth_m,
    }


def build_particle_palette(
    compositions: object,
    *,
    resolution: object = DEFAULT_PALETTE_RESOLUTION,
    source_extinctions: object | None = None,
    reference_depth_m: object = DEFAULT_REFERENCE_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(palette_colors, proto_indices, receipt)`` for particle rows.

    ``palette_colors`` is ordered by the frozen ternary lattice and
    ``proto_indices`` has one int64 entry per input particle.  Receipt hashes
    cover the canonicalized inputs, lattice, colours, source colours and
    indices so a runtime stage can bind the exact palette configuration.
    """

    rows = _validated_compositions(compositions)
    denominator = _validated_resolution(resolution)
    extinctions = _validated_source_extinctions(source_extinctions)
    depth = _validated_reference_depth_m(reference_depth_m)
    palette_compositions = build_simplex_palette(denominator)
    palette_colors = mix_extinction_transmission(
        palette_compositions, extinctions, reference_depth_m=depth
    )
    proto_indices = quantize_compositions_to_palette(
        rows, resolution=denominator, palette=palette_compositions
    )
    receipt = _build_receipt(
        rows,
        palette_compositions,
        palette_colors,
        proto_indices,
        extinctions,
        depth,
        denominator,
    )
    return palette_colors, proto_indices, receipt


build_particle_palette_with_receipt = build_particle_palette

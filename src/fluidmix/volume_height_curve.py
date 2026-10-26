"""Geometry-bound CPU volume/height conversion for the F2 vial collider.

The sampled cross-section curve is deliberately separate from USD authoring:
it hashes the reviewed collider geometry but never changes camera, vial, or
scene transforms.  It is a gate contract, not a measurement of rendered or
physical liquid volume.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

import numpy as np


INTERIOR_WALL_COUNT = 32
DEFAULT_GEOMETRY = {
    "boundary": "existing_invisible_interior_polygon_collider",
    "center_xy_m": [5.2324003281, -1.9811999798],
    "bottom_z_m": 0.9762065645,
    "mouth_z_m": 1.1762601692,
    "circumradius_m": 0.0132029216,
    "wall_count": INTERIOR_WALL_COUNT,
}


def _regular_polygon_area(radius_m: float, sides: int) -> float:
    return 0.5 * sides * radius_m**2 * math.sin(2.0 * math.pi / sides)


@dataclass(frozen=True)
class VolumeHeightCurve:
    """Monotone sampled cross-section curve with a geometry-bound digest."""

    height_offsets_m: tuple[float, ...]
    cross_section_area_m2: tuple[float, ...]
    geometry: dict[str, object]

    @property
    def maximum_height_m(self) -> float:
        return self.height_offsets_m[-1]

    @property
    def capacity_ml(self) -> float:
        return _volume_m3_at_height(self, self.maximum_height_m) * 1.0e6


def _default_curve() -> VolumeHeightCurve:
    height = float(DEFAULT_GEOMETRY["mouth_z_m"]) - float(DEFAULT_GEOMETRY["bottom_z_m"])
    area = _regular_polygon_area(
        float(DEFAULT_GEOMETRY["circumradius_m"]), int(DEFAULT_GEOMETRY["wall_count"])
    )
    return VolumeHeightCurve(
        height_offsets_m=(0.0, height * 0.25, height * 0.5, height * 0.75, height),
        cross_section_area_m2=(area, area, area, area, area),
        geometry=dict(DEFAULT_GEOMETRY),
    )


DEFAULT_VOLUME_HEIGHT_CURVE = _default_curve()


def _validate_curve(curve: object) -> VolumeHeightCurve:
    if not isinstance(curve, VolumeHeightCurve):
        raise TypeError("curve must be a VolumeHeightCurve")
    heights = np.asarray(curve.height_offsets_m, dtype=np.float64)
    areas = np.asarray(curve.cross_section_area_m2, dtype=np.float64)
    if heights.ndim != 1 or areas.ndim != 1 or len(heights) != len(areas) or len(heights) < 2:
        raise ValueError("curve must contain matching height and area samples")
    if not np.all(np.isfinite(heights)) or not np.all(np.isfinite(areas)):
        raise ValueError("curve samples must be finite")
    if heights[0] != 0.0 or np.any(np.diff(heights) <= 0.0):
        raise ValueError("curve heights must start at zero and be strictly increasing")
    if np.any(areas <= 0.0):
        raise ValueError("curve cross-section areas must be positive")
    if not isinstance(curve.geometry, dict) or not curve.geometry:
        raise ValueError("curve geometry must be a non-empty mapping")
    return curve


def volume_height_curve_digest(curve: object = DEFAULT_VOLUME_HEIGHT_CURVE) -> str:
    """Return a stable hash of geometry and the sampled integration curve."""

    checked = _validate_curve(curve)
    payload = {
        "geometry": checked.geometry,
        "height_offsets_m": list(checked.height_offsets_m),
        "cross_section_area_m2": list(checked.cross_section_area_m2),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_volume_height_curve(curve: object = DEFAULT_VOLUME_HEIGHT_CURVE) -> dict[str, object]:
    """Validate the curve and return its capacity and geometry-bound digest."""

    checked = _validate_curve(curve)
    wall_count = checked.geometry.get("wall_count")
    radius = checked.geometry.get("circumradius_m")
    if isinstance(wall_count, bool) or not isinstance(wall_count, int) or wall_count < 3:
        raise ValueError("curve geometry wall_count must be an integer of at least three")
    if isinstance(radius, bool) or not isinstance(radius, Real) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
        raise ValueError("curve geometry circumradius_m must be finite and positive")
    authored_circular_area_m2 = math.pi * float(radius) ** 2
    interior_polygon_area_m2 = _regular_polygon_area(float(radius), wall_count)
    area_relative_error = abs(authored_circular_area_m2 - interior_polygon_area_m2) / authored_circular_area_m2
    if area_relative_error > 0.01:
        raise ValueError("interior polygon area must be within one percent of the authored circle")
    return {
        "geometry_sha256": volume_height_curve_digest(checked),
        "sample_count": len(checked.height_offsets_m),
        "maximum_height_m": checked.maximum_height_m,
        "capacity_ml": checked.capacity_ml,
        "strictly_monotone": True,
        "wall_count": wall_count,
        "authored_circular_area_m2": authored_circular_area_m2,
        "interior_polygon_area_m2": interior_polygon_area_m2,
        "area_relative_error": area_relative_error,
        "area_parity_within_one_percent": area_relative_error <= 0.01,
    }


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _volume_m3_at_height(curve: VolumeHeightCurve, height_m: float) -> float:
    heights = curve.height_offsets_m
    areas = curve.cross_section_area_m2
    if height_m < 0.0 or height_m > heights[-1]:
        raise ValueError("height_m is outside curve capacity")
    volume = 0.0
    for left, right, area_left, area_right in zip(heights, heights[1:], areas, areas[1:], strict=True):
        if height_m <= left:
            break
        segment_height = min(height_m, right) - left
        slope = (area_right - area_left) / (right - left)
        volume += area_left * segment_height + 0.5 * slope * segment_height**2
        if height_m <= right:
            break
    return volume


def height_for_volume_ml(volume_ml: object, curve: object = DEFAULT_VOLUME_HEIGHT_CURVE) -> float:
    """Return curve-derived height for a positive programmed volume in mL."""

    volume_m3 = _positive_finite(volume_ml, "volume_ml") * 1.0e-6
    checked = _validate_curve(curve)
    if volume_m3 > _volume_m3_at_height(checked, checked.maximum_height_m):
        raise ValueError("volume_ml exceeds curve capacity")
    lower, upper = 0.0, checked.maximum_height_m
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if _volume_m3_at_height(checked, middle) < volume_m3:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def volume_for_height_m(height_m: object, curve: object = DEFAULT_VOLUME_HEIGHT_CURVE) -> float:
    """Return positive curve-derived volume in mL for a height in capacity."""

    height = _positive_finite(height_m, "height_m")
    checked = _validate_curve(curve)
    if height > checked.maximum_height_m:
        raise ValueError("height_m exceeds curve capacity")
    return _volume_m3_at_height(checked, height) * 1.0e6


def height_bounds_for_volume_ml(volume_ml: object, curve: object = DEFAULT_VOLUME_HEIGHT_CURVE) -> tuple[float, float]:
    """Return the required [h(0.95V), h(1.05V)] tolerance bounds."""

    volume = _positive_finite(volume_ml, "volume_ml")
    return (
        height_for_volume_ml(0.95 * volume, curve),
        height_for_volume_ml(1.05 * volume, curve),
    )

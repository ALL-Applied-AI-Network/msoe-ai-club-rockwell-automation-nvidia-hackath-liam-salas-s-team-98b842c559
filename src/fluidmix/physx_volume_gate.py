"""Volume-evidence helpers for the F2 particle render gate."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .volume_height_curve import (
    INTERIOR_WALL_COUNT,
    height_bounds_for_volume_ml,
    height_for_volume_ml,
    validate_volume_height_curve,
    volume_for_height_m,
    volume_height_curve_digest,
)


FILL_CAM_WORLD_TO_CAMERA_ROW_MATRIX = np.asarray(
    [
        [0.17111764335899196, 0.56132151188485757, -0.8097139788539951, 0.0],
        [-0.9852414147975451, 0.10102029855909929, -0.13818129023774309, 0.0],
        [0.0042334175321830265, 0.8214090787782503, 0.5703236086658785, 0.0],
        [-2.856628645009652, -3.6330541287385123, 2.9772532159032692, 1.0],
    ],
    dtype=np.float64,
)
FILL_CAM_FOCAL_LENGTH = 0.40993183851242065
FILL_CAM_HORIZONTAL_APERTURE = 0.36000001430511475
FILL_CAM_VERTICAL_APERTURE = 0.27000001072883606
FILL_CAM_WIDTH_PX = 640
FILL_CAM_HEIGHT_PX = 480


def expected_cylindrical_height_m(
    cumulative_volume_ml: object,
    cross_section_area_m2: object,
) -> float:
    """Return the ideal fill height for a programmed volume and fixed area."""

    for name, value in (
        ("cumulative_volume_ml", cumulative_volume_ml),
        ("cross_section_area_m2", cross_section_area_m2),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return float(cumulative_volume_ml) * 1.0e-6 / float(cross_section_area_m2)


def particle_bulk_height_m(
    contained_center_z_m: object,
    render_radius_m: object,
) -> float:
    """Estimate the bulk column height while trimming a narrow falling tail."""

    raw = np.asarray(contained_center_z_m)
    if raw.dtype.kind not in "fiu":
        raise TypeError("contained_center_z_m must contain real numbers")
    values = np.asarray(contained_center_z_m, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("contained_center_z_m must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("contained_center_z_m must be finite")
    lower, upper = np.quantile(values, (0.05, 0.95), method="linear")
    return particle_bulk_height_from_quantiles_m(lower, upper, render_radius_m)


def particle_bulk_height_from_quantiles_m(
    contained_z_q05_m: object,
    contained_z_q95_m: object,
    render_radius_m: object,
) -> float:
    """Return the frozen bulk-height statistic from saved quantiles."""

    values: list[float] = []
    for name, value in (
        ("contained_z_q05_m", contained_z_q05_m),
        ("contained_z_q95_m", contained_z_q95_m),
        ("render_radius_m", render_radius_m),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        values.append(number)
    lower, upper, radius = values
    if upper < lower:
        raise ValueError("contained_z_q95_m must not be below contained_z_q05_m")
    if radius <= 0.0:
        raise ValueError("render_radius_m must be positive")
    return float(2.0 * radius + (upper - lower) / 0.90)


def height_within_tolerance(
    observed_height_m: object,
    expected_height_m: object,
    tolerance_m: object,
) -> bool:
    """Return whether a finite positive observed height is inside the gate."""

    converted: list[float] = []
    for name, value, allow_zero in (
        ("observed_height_m", observed_height_m, False),
        ("expected_height_m", expected_height_m, False),
        ("tolerance_m", tolerance_m, True),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or (not allow_zero and number == 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
        converted.append(number)
    observed, expected, tolerance = converted
    return abs(observed - expected) <= tolerance


def height_within_volume_bounds(
    observed_height_m: object,
    programmed_volume_ml: object,
) -> bool:
    """Return whether an observed height falls in the curve-derived 5% band."""

    if isinstance(observed_height_m, bool) or not isinstance(observed_height_m, Real):
        raise TypeError("observed_height_m must be a real number")
    observed = float(observed_height_m)
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError("observed_height_m must be finite and positive")
    lower, upper = height_bounds_for_volume_ml(programmed_volume_ml)
    return lower <= observed <= upper


def projected_top_dodecagon_min_y_px(
    fill_height_m: object,
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    circumradius_m: float,
    wall_count: int = INTERIOR_WALL_COUNT,
) -> float:
    """Project the highest image row of the collider's top contour."""

    vertices = projected_dodecagon_vertices_px(
        fill_height_m,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        circumradius_m=circumradius_m,
        wall_count=wall_count,
    )
    return float(np.min(vertices[:, 1]))


def projected_dodecagon_vertices_px(
    fill_height_m: object,
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    circumradius_m: float,
    wall_count: int = INTERIOR_WALL_COUNT,
) -> np.ndarray:
    """Project a horizontal collider contour into frozen FillCam pixels."""

    if isinstance(fill_height_m, bool) or not isinstance(fill_height_m, Real):
        raise TypeError("fill_height_m must be a real number")
    height = float(fill_height_m)
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("fill_height_m must be finite and non-negative")
    center = np.asarray(center_xy_m, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy_m must contain two finite values")
    for name, value in (
        ("bottom_z_m", bottom_z_m),
        ("circumradius_m", circumradius_m),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    radius = float(circumradius_m)
    if radius <= 0.0:
        raise ValueError("circumradius_m must be positive")
    if isinstance(wall_count, bool) or not isinstance(wall_count, int):
        raise TypeError("wall_count must be an integer")
    if wall_count < 3:
        raise ValueError("wall_count must be at least three")
    angles = (
        2.0 * math.pi * np.arange(wall_count, dtype=np.float64) / wall_count
        + math.pi / wall_count
    )
    world = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(len(angles), float(bottom_z_m) + height),
            np.ones(len(angles)),
        )
    )
    camera = world @ FILL_CAM_WORLD_TO_CAMERA_ROW_MATRIX
    if np.any(camera[:, 2] >= 0.0):
        raise ValueError("surface ring is not fully in front of FillCam")
    ndc_x = (
        camera[:, 0]
        / -camera[:, 2]
        * (2.0 * FILL_CAM_FOCAL_LENGTH / FILL_CAM_HORIZONTAL_APERTURE)
    )
    ndc_y = (
        camera[:, 1]
        / -camera[:, 2]
        * (2.0 * FILL_CAM_FOCAL_LENGTH / FILL_CAM_VERTICAL_APERTURE)
    )
    return np.column_stack(
        (
            (ndc_x + 1.0) * (FILL_CAM_WIDTH_PX / 2.0),
            (1.0 - ndc_y) * (FILL_CAM_HEIGHT_PX / 2.0),
        )
    )


def visible_height_from_surface_row_m(
    surface_row_px: object,
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    circumradius_m: float,
    maximum_height_m: float,
    wall_count: int = INTERIOR_WALL_COUNT,
) -> float:
    """Invert the frozen FillCam dodecagon-contour projection."""

    if isinstance(surface_row_px, bool) or not isinstance(surface_row_px, Real):
        raise TypeError("surface_row_px must be a real number")
    row = float(surface_row_px)
    if not math.isfinite(row):
        raise ValueError("surface_row_px must be finite")
    if isinstance(maximum_height_m, bool) or not isinstance(maximum_height_m, Real):
        raise TypeError("maximum_height_m must be a real number")
    maximum = float(maximum_height_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_height_m must be finite and positive")
    low_y = projected_top_dodecagon_min_y_px(
        0.0,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        circumradius_m=circumradius_m,
        wall_count=wall_count,
    )
    high_y = projected_top_dodecagon_min_y_px(
        maximum,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        circumradius_m=circumradius_m,
        wall_count=wall_count,
    )
    if row > low_y or row < high_y:
        raise ValueError("surface_row_px is outside the projected vial height")
    lower = 0.0
    upper = maximum
    for _ in range(60):
        middle = (lower + upper) / 2.0
        projected = projected_top_dodecagon_min_y_px(
            middle,
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_z_m,
            circumradius_m=circumradius_m,
            wall_count=wall_count,
        )
        if projected > row:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def find_dense_surface_row(
    liquid_mask: object,
    *,
    geometry_roi_mask: object,
    minimum_width_fraction: float = 1.0 / 3.0,
) -> dict[str, object]:
    """Find two broad liquid rows inside the projected vial geometry."""

    raw = np.asarray(liquid_mask)
    if raw.ndim != 2 or raw.dtype.kind not in "bufiu":
        raise TypeError("liquid_mask must be a two-dimensional boolean/numeric array")
    if raw.dtype.kind == "f" and not np.all(np.isfinite(raw)):
        raise ValueError("liquid_mask must be finite")
    mask = raw.astype(bool, copy=False)
    roi_raw = np.asarray(geometry_roi_mask)
    if roi_raw.shape != mask.shape or roi_raw.dtype.kind not in "bufiu":
        raise TypeError("geometry_roi_mask must be a boolean/numeric mask of the same shape")
    if roi_raw.dtype.kind == "f" and not np.all(np.isfinite(roi_raw)):
        raise ValueError("geometry_roi_mask must be finite")
    roi = roi_raw.astype(bool, copy=False)
    if isinstance(minimum_width_fraction, bool) or not isinstance(
        minimum_width_fraction, Real
    ):
        raise TypeError("minimum_width_fraction must be a real number")
    fraction = float(minimum_width_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("minimum_width_fraction must be finite and in (0, 1]")
    roi_widths = np.count_nonzero(roi, axis=1)
    liquid_counts = np.count_nonzero(mask & roi, axis=1)
    minimum_counts = np.ceil(roi_widths * fraction).astype(np.int64)
    dense = (roi_widths > 0) & (liquid_counts >= minimum_counts)
    surface_row = next(
        (
            row
            for row in range(len(dense) - 1)
            if dense[row] and dense[row + 1]
        ),
        None,
    )
    if surface_row is None:
        raise ValueError("liquid mask has no two-row dense surface")
    ys, xs = np.nonzero(mask & roi)
    return {
        "surface_row_px": int(surface_row),
        "liquid_pixels_inside_geometry_roi": int(len(xs)),
        "liquid_bbox_inside_geometry_roi_xyxy": [
            int(np.min(xs)),
            int(np.min(ys)),
            int(np.max(xs)) + 1,
            int(np.max(ys)) + 1,
        ],
        "surface_row_liquid_pixels": int(liquid_counts[surface_row]),
        "surface_row_roi_width_pixels": int(roi_widths[surface_row]),
        "surface_row_minimum_pixels": int(minimum_counts[surface_row]),
        "minimum_width_fraction": fraction,
    }


def projected_vial_geometry_roi_mask(
    image_shape: tuple[int, int],
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    mouth_z_m: float,
    circumradius_m: float,
    wall_count: int = INTERIOR_WALL_COUNT,
) -> np.ndarray:
    """Rasterize the projected floor-to-mouth collider hull at pixel centers."""

    if (
        len(image_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in image_shape)
        or any(value <= 0 for value in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")
    center = np.asarray(center_xy_m, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy_m must contain two finite values")
    numeric = {
        "bottom_z_m": bottom_z_m,
        "mouth_z_m": mouth_z_m,
        "circumradius_m": circumradius_m,
    }
    for name, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(mouth_z_m) <= float(bottom_z_m):
        raise ValueError("mouth_z_m must exceed bottom_z_m")
    radius = float(circumradius_m)
    if radius <= 0.0:
        raise ValueError("circumradius_m must be positive")
    if isinstance(wall_count, bool) or not isinstance(wall_count, int):
        raise TypeError("wall_count must be an integer")
    if wall_count < 3:
        raise ValueError("wall_count must be at least three")
    angles = (
        2.0 * math.pi * np.arange(wall_count, dtype=np.float64) / wall_count
        + math.pi / wall_count
    )
    rings = []
    for z in (float(bottom_z_m), float(mouth_z_m)):
        world = np.column_stack(
            (
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                np.full(wall_count, z),
                np.ones(wall_count),
            )
        )
        camera = world @ FILL_CAM_WORLD_TO_CAMERA_ROW_MATRIX
        ndc_x = (
            camera[:, 0]
            / -camera[:, 2]
            * (2.0 * FILL_CAM_FOCAL_LENGTH / FILL_CAM_HORIZONTAL_APERTURE)
        )
        ndc_y = (
            camera[:, 1]
            / -camera[:, 2]
            * (2.0 * FILL_CAM_FOCAL_LENGTH / FILL_CAM_VERTICAL_APERTURE)
        )
        rings.extend(
            zip(
                (ndc_x + 1.0) * (FILL_CAM_WIDTH_PX / 2.0),
                (1.0 - ndc_y) * (FILL_CAM_HEIGHT_PX / 2.0),
                strict=True,
            )
        )

    points = sorted(set((float(x), float(y)) for x, y in rings))

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    polygon_area_twice = float(
        np.sum(
            hull[:, 0] * np.roll(hull[:, 1], -1)
            - np.roll(hull[:, 0], -1) * hull[:, 1]
        )
    )
    if polygon_area_twice < 0.0:
        hull = hull[::-1]

    image_height, image_width = image_shape
    x0 = max(0, int(math.floor(float(np.min(hull[:, 0])))))
    x1 = min(image_width, int(math.ceil(float(np.max(hull[:, 0])))) + 1)
    y0 = max(0, int(math.floor(float(np.min(hull[:, 1])))))
    y1 = min(image_height, int(math.ceil(float(np.max(hull[:, 1])))) + 1)
    pixel_x, pixel_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float64) + 0.5,
        np.arange(y0, y1, dtype=np.float64) + 0.5,
    )
    inside = np.ones(pixel_x.shape, dtype=bool)
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        inside &= (
            (end[0] - start[0]) * (pixel_y - start[1])
            - (end[1] - start[1]) * (pixel_x - start[0])
        ) >= -1.0e-9
    result = np.zeros(image_shape, dtype=bool)
    result[y0:y1, x0:x1] = inside
    return result

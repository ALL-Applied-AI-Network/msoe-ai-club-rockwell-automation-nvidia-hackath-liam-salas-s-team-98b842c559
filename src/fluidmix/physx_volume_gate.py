"""Volume-evidence helpers for the F2 particle render gate."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from .transform_attestation import (
    MATRIX_COMPARATOR_VERSION,
    attest_effective_state_matrix,
    verify_effective_state_matrix_attestation,
)
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
FILL_CAM_CARRIER_OCCLUSION_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "fillcam_carrier_occlusion_v1.json"
)
FILL_CAM_CARRIER_OCCLUSION_PROFILE_SHA256 = (
    "52a262b7dee1c3603049a879c6aeedc15b83bf94e8df98d03e63221070e95a0e"
)
FILL_CAM_CARRIER_OCCLUSION_LEGACY_PROFILE_SHA256 = (
    "b056579aec18b2d2ace726d7640289e0401f6239fa4303057646040e880504f7"
)
FILL_CAM_CARRIER_OCCLUSION_PROFILE_ID = (
    "fillcam-active-carrier-upper-clamp-v1"
)
FILL_CAM_CARRIER_OCCLUSION_SOURCE_STAGE_SHA256 = (
    "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9"
)
FILL_CAM_CARRIER_OCCLUSION_GEOMETRY_SHA256 = (
    "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
)
FILL_CAM_CARRIER_OCCLUSION_CAMERA_SHA256 = (
    "772bc8a3d9bb20c8970d9e3ee160d444e68d58418147f498457bfc162e03dfcf"
)
FILL_CAM_CARRIER_OCCLUSION_CAMERA_PATH = (
    "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
)
FILL_CAM_CARRIER_OCCLUSION_MASK_SHA256 = (
    "a225c986cfe23b68e9a70ed4c2e5bffcdd9a2992ce7452be9f9e862aff655ae4"
)
FILL_CAM_CARRIER_OCCLUSION_PRIM_PATH = (
    "/FillStation/FillStation_LocalRoot/V3_carrier_0_clamp_21/Mesh_041"
)
FILL_CAM_CARRIER_OCCLUSION_EFFECTIVE_STATE_SHA256 = (
    "4380371e987a6f234e99f67e704b653b481b668c421dd1e9cf47cb7655481844"
)
FILL_CAM_CARRIER_OCCLUSION_LEGACY_EFFECTIVE_STATE_SHA256 = (
    "803d9be168bba34482f893bf518b62a96380faf6a02973687cf30ca507da995f"
)
FILL_CAM_CARRIER_OCCLUSION_RASTER_EDGE_RULE = (
    "top-left-half-open-v1: y-down pixel centers; a zero oriented-edge value is "
    "included only when dy<0 or (dy==0 and dx>0); boundary fixture (284,253) is "
    "excluded while (284,252) and (285,253) are included"
)


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


def load_fillcam_carrier_occlusion_profile(
    path: Path | None = None,
) -> dict[str, object]:
    """Load and fail closed on the frozen source-mesh/FillCam occlusion profile."""

    profile_path = path or FILL_CAM_CARRIER_OCCLUSION_PROFILE_PATH
    try:
        payload = profile_path.read_bytes()
    except OSError as error:
        raise ValueError("carrier occlusion profile is unreadable") from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FILL_CAM_CARRIER_OCCLUSION_PROFILE_SHA256:
        raise ValueError("carrier occlusion profile SHA-256 mismatch")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("carrier occlusion profile is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("carrier occlusion profile must be an object")
    result: dict[str, object] = dict(value)
    result["profile_file_sha256"] = digest
    _carrier_occlusion_profile_mask(result)
    return result


def expected_carrier_occluder_state(
    profile: Mapping[str, object],
) -> dict[str, object]:
    """Build the frozen raw occluder state, including its derived world matrix."""

    return {
        "prim_path": profile.get("occluder_prim_path"),
        "time_code": profile.get("time_code"),
        "mesh_points_count": profile.get("mesh_points_count"),
        "mesh_face_count": profile.get("mesh_face_count"),
        "mesh_points_f64_sha256": profile.get("mesh_points_f64_sha256"),
        "mesh_face_counts_i64_sha256": profile.get(
            "mesh_face_counts_i64_sha256"
        ),
        "mesh_face_indices_i64_sha256": profile.get(
            "mesh_face_indices_i64_sha256"
        ),
        "mesh_world_transform_matrix": profile.get(
            "mesh_expected_world_transform_matrix"
        ),
        "mesh_world_transform_f64_sha256": profile.get(
            "mesh_world_transform_f64_sha256"
        ),
        "effective_visibility": profile.get("mesh_effective_visibility"),
    }


def _carrier_occlusion_profile_mask(
    profile: Mapping[str, object],
) -> np.ndarray:
    """Validate and expand the compact depth-filtered occlusion raster."""

    if not isinstance(profile, Mapping):
        raise TypeError("carrier occlusion profile must be a mapping")
    required_bindings = {
        "schema_version": 2,
        "artifact_type": "fillcam_carrier_occlusion_profile",
        "profile_id": FILL_CAM_CARRIER_OCCLUSION_PROFILE_ID,
        "supersedes_profile_file_sha256": (
            FILL_CAM_CARRIER_OCCLUSION_LEGACY_PROFILE_SHA256
        ),
        "matrix_comparator_version": MATRIX_COMPARATOR_VERSION,
        "source_stage_sha256": FILL_CAM_CARRIER_OCCLUSION_SOURCE_STAGE_SHA256,
        "source_geometry_layer_sha256": (
            FILL_CAM_CARRIER_OCCLUSION_GEOMETRY_SHA256
        ),
        "camera_prim_path": FILL_CAM_CARRIER_OCCLUSION_CAMERA_PATH,
        "camera_effective_state_sha256": FILL_CAM_CARRIER_OCCLUSION_CAMERA_SHA256,
        "time_code": 1.0,
        "resolution": [FILL_CAM_WIDTH_PX, FILL_CAM_HEIGHT_PX],
        "occluder_prim_path": FILL_CAM_CARRIER_OCCLUSION_PRIM_PATH,
        "mesh_points_count": 912,
        "mesh_face_count": 806,
        "mesh_points_f64_sha256": (
            "935706b42907f1486214e2816021594e7de5a3bd43dc45b84815a888348eba71"
        ),
        "mesh_face_counts_i64_sha256": (
            "15d27e0b6ffa18241bcfd9305ca5d2c83129ae6bab06867f6502b31851ea6f31"
        ),
        "mesh_face_indices_i64_sha256": (
            "68a246900e987d8810fc4a5efa798c2f540da17b48831d4d27944b07e14c41c4"
        ),
        "mesh_world_transform_f64_sha256": (
            "2df92340d8b7d9726f2bac8d79686c3ce1680ae0bc43ed326dca64e0319d22d0"
        ),
        "legacy_occluder_effective_state_sha256": (
            FILL_CAM_CARRIER_OCCLUSION_LEGACY_EFFECTIVE_STATE_SHA256
        ),
        "mesh_effective_visibility": "inherited",
        "occluder_effective_state_sha256": (
            FILL_CAM_CARRIER_OCCLUSION_EFFECTIVE_STATE_SHA256
        ),
        "rasterizer": (
            "projected-face-fan-pixel-center-union-depth-before-vial-prism-v1"
        ),
        "raster_edge_rule": FILL_CAM_CARRIER_OCCLUSION_RASTER_EDGE_RULE,
        "depth_model": (
            "nearest source-mesh triangle must be in front of the closed "
            "source-vial dodecagonal prism"
        ),
        "profile_mask_packbits_little_sha256": (
            FILL_CAM_CARRIER_OCCLUSION_MASK_SHA256
        ),
        "profile_pixel_count": 1358,
    }
    for key, expected in required_bindings.items():
        if profile.get(key) != expected:
            raise ValueError(f"carrier occlusion profile binding mismatch: {key}")
    expected_camera_state = profile.get("camera_expected_effective_state")
    camera_attestation = attest_effective_state_matrix(
        expected_camera_state,
        expected_camera_state,
        matrix_field="world_transform_matrix",
    )
    if (
        camera_attestation.get("passed") is not True
        or camera_attestation.get("expected_state_sha256")
        != FILL_CAM_CARRIER_OCCLUSION_CAMERA_SHA256
        or profile.get("camera_expected_effective_state_provenance")
        != {
            "source": "accepted_gb10_receipt",
            "receipt_sha256": (
                "330b52a872cb3c42257cc84e2f239d9b7a4666ed58c879ad63078a5d08f9d8d6"
            ),
            "case_id": "CASE_d52026e7df33b0cb",
            "receipt_locator": (
                "outputs/pipeline-20261028/wp1-gb10-01a18128/benchmark-dispatch-"
                "01a18128/c2-slot-01-v05/cases/CASE_d52026e7df33b0cb/app-output/"
                "receipt.json"
            ),
            "extraction": "camera.effective_render_state exact JSON value",
        }
    ):
        raise ValueError("carrier occlusion profile camera state provenance mismatch")
    expected_occluder_state = expected_carrier_occluder_state(profile)
    occluder_attestation = attest_effective_state_matrix(
        expected_occluder_state,
        expected_occluder_state,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    )
    if (
        occluder_attestation.get("passed") is not True
        or occluder_attestation["matrix_comparison"].get(
            "expected_matrix_f64_sha256"
        )
        != profile.get("mesh_world_transform_f64_sha256")
        or occluder_attestation.get("expected_state_sha256")
        != FILL_CAM_CARRIER_OCCLUSION_EFFECTIVE_STATE_SHA256
        or profile.get("mesh_expected_world_transform_provenance")
        != {
            "source": "read_only_cpu_openusd_composition",
            "source_stage_sha256": FILL_CAM_CARRIER_OCCLUSION_SOURCE_STAGE_SHA256,
            "source_geometry_layer_sha256": (
                FILL_CAM_CARRIER_OCCLUSION_GEOMETRY_SHA256
            ),
            "prim_path": FILL_CAM_CARRIER_OCCLUSION_PRIM_PATH,
            "time_code": 1.0,
            "derivation": (
                "UsdGeom.Xformable.ComputeLocalToWorldTransform then row-major "
                "little-endian binary64"
            ),
            "preregistration_sha256": (
                "c08057e9873fd3cc171a716059409ec3a93a7a965ea4f9bed7458dbc14a2236f"
            ),
        }
    ):
        raise ValueError("carrier occlusion profile occluder matrix provenance mismatch")
    if profile.get("vial_geometry") != {
        "center_xy_m": [5.2324003281, -1.9811999798],
        "bottom_z_m": 0.9762065645,
        "mouth_z_m": 1.1762601692,
        "circumradius_m": 0.0132029216,
        "wall_count": 12,
    }:
        raise ValueError("carrier occlusion profile vial geometry mismatch")
    if profile.get("surface_detector") != {
        "minimum_width_fraction": 1.0 / 3.0,
        "required_consecutive_dense_rows": 2,
        "contact_connectivity": 8,
    }:
        raise ValueError("carrier occlusion profile detector mismatch")
    body_bindings = profile.get("count_driven_body_geometry_bindings")
    if (
        not isinstance(body_bindings, Mapping)
        or body_bindings.get("derivation")
        != (
            "build_count_driven_liquid_body_geometry_v1_then_vec3f_"
            "readback_and_fixed_usd_world_transform"
        )
        or body_bindings.get("source_mesh_prim_path")
        != "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
        or body_bindings.get("requested_source_time_code") != 686
        or body_bindings.get("resolved_points_time_code") != 681
        or body_bindings.get("source_geometry_sha256")
        != "e1b9711dc50195e559f54f96db2ce5389a34ef719213c443498a5585b6bd712f"
        or body_bindings.get("source_topology_sha256")
        != "edba8a108a846fbb3fd661e95b9a642b727a06e7ba431ca70e013d1255132acd"
        or body_bindings.get("wp1_render_schedule_plan_sha256")
        != "d61964932af76a0d62b03221b01da527a52370104b467fbd45dee3741faf7517"
        or set(body_bindings.get("by_contained_particle_count", {}))
        != {"976", "2484", "2750", "4524", "4790", "5500", "5766"}
    ):
        raise ValueError("carrier occlusion body geometry bindings mismatch")
    measured_cross_check = profile.get("measured_cross_check")
    if (
        not isinstance(measured_cross_check, Mapping)
        or measured_cross_check.get("original_receipt_sha256")
        != "b2f911d58be4644fe20b685f43634282fd2f61657df7fdf0ea40e0cf23980764"
        or measured_cross_check.get("original_stage2_liquid_mask_sha256")
        != "aecb72a8cb2bac1db274fd830cb080250a664fb7f0a04ec0f6072a7b034baa98"
    ):
        raise ValueError("carrier occlusion historical binding mismatch")
    depth_margins = (
        profile.get("minimum_front_depth_margin_m"),
        profile.get("acceptable_band_minimum_front_depth_margin_m"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in depth_margins
    ) or float(depth_margins[1]) < float(depth_margins[0]):
        raise ValueError("carrier occlusion profile depth evidence mismatch")
    file_digest = profile.get("profile_file_sha256")
    if (
        file_digest is not None
        and file_digest != FILL_CAM_CARRIER_OCCLUSION_PROFILE_SHA256
    ):
        raise ValueError("carrier occlusion profile file binding mismatch")
    rows = profile.get("row_runs_xyxy_exclusive")
    if not isinstance(rows, list) or not rows:
        raise ValueError("carrier occlusion profile rows are missing")
    mask = np.zeros((FILL_CAM_HEIGHT_PX, FILL_CAM_WIDTH_PX), dtype=bool)
    previous_end = -1
    for record in rows:
        if (
            not isinstance(record, list)
            or len(record) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in record[:2]
            )
            or not isinstance(record[2], list)
        ):
            raise ValueError("carrier occlusion profile row record is invalid")
        y0, y1, intervals = record
        if not 0 <= y0 < y1 <= FILL_CAM_HEIGHT_PX or y0 < previous_end:
            raise ValueError("carrier occlusion profile row range is invalid")
        previous_end = y1
        previous_x1 = -1
        for interval in intervals:
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in interval
                )
            ):
                raise ValueError("carrier occlusion profile interval is invalid")
            x0, x1 = interval
            if not 0 <= x0 < x1 <= FILL_CAM_WIDTH_PX or x0 < previous_x1:
                raise ValueError("carrier occlusion profile interval range is invalid")
            previous_x1 = x1
            if np.any(mask[y0:y1, x0:x1]):
                raise ValueError("carrier occlusion profile intervals overlap")
            mask[y0:y1, x0:x1] = True
    if int(np.count_nonzero(mask)) != profile.get("profile_pixel_count"):
        raise ValueError("carrier occlusion profile pixel count mismatch")
    mask_digest = hashlib.sha256(
        np.packbits(mask, bitorder="little").tobytes()
    ).hexdigest()
    if mask_digest != profile.get("profile_mask_packbits_little_sha256"):
        raise ValueError("carrier occlusion profile raster SHA-256 mismatch")
    return mask


def evaluate_carrier_occlusion_censored_height(
    liquid_mask: object,
    *,
    geometry_roi_mask: object,
    surface: Mapping[str, object],
    volume_gate: Mapping[str, object],
    render_body_geometry: Mapping[str, object],
    profile: Mapping[str, object],
    source_stage_sha256: object,
    source_geometry_layer_sha256: object,
    camera_effective_state_attestation: object,
    occluder_effective_state_attestation: object,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    circumradius_m: float,
    wall_count: int = INTERIOR_WALL_COUNT,
) -> dict[str, object]:
    """Prove a below-bound mask is lower-censored by the fixed carrier clamp."""

    failures: list[str] = []
    try:
        occluder = _carrier_occlusion_profile_mask(profile)
    except (TypeError, ValueError) as error:
        return {
            "passed": False,
            "failure_reasons": ["occlusion_profile_invalid"],
            "error": str(error),
        }
    profile_wall_count = int(profile["vial_geometry"]["wall_count"])
    exact_bindings = {
        "source_stage_sha256": source_stage_sha256,
        "source_geometry_layer_sha256": source_geometry_layer_sha256,
    }
    for key, observed in exact_bindings.items():
        if observed != profile.get(key):
            failures.append(f"{key}_mismatch")
    expected_camera_state = profile.get("camera_expected_effective_state")
    if not verify_effective_state_matrix_attestation(
        camera_effective_state_attestation,
        expected_camera_state,
        matrix_field="world_transform_matrix",
    ):
        failures.append("camera_effective_state_attestation_invalid")
    expected_occluder_state = expected_carrier_occluder_state(profile)
    if not verify_effective_state_matrix_attestation(
        occluder_effective_state_attestation,
        expected_occluder_state,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    ):
        failures.append("occluder_effective_state_attestation_invalid")

    raw_liquid = np.asarray(liquid_mask)
    raw_roi = np.asarray(geometry_roi_mask)
    if (
        raw_liquid.shape != occluder.shape
        or raw_roi.shape != occluder.shape
        or raw_liquid.dtype.kind not in "bufiu"
        or raw_roi.dtype.kind not in "bufiu"
    ):
        failures.append("occlusion_mask_shape_or_dtype_invalid")
        liquid = np.zeros_like(occluder)
        roi = np.zeros_like(occluder)
    else:
        liquid = raw_liquid.astype(bool, copy=False)
        roi = raw_roi.astype(bool, copy=False)
    if np.any(liquid & occluder):
        failures.append("liquid_overlaps_occluder_profile")

    evidence = volume_gate.get("visible_mask_evidence")
    policy = volume_gate.get("visible_height_policy")
    numeric_values = (
        evidence.get("observed_height_above_cutoff_m")
        if isinstance(evidence, Mapping)
        else None,
        policy.get("lower_bound_height_above_cutoff_m")
        if isinstance(policy, Mapping)
        else None,
        policy.get("upper_bound_height_above_cutoff_m")
        if isinstance(policy, Mapping)
        else None,
        policy.get("opaque_base_cutoff_height_m")
        if isinstance(policy, Mapping)
        else None,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        failures.append("visible_height_values_invalid")
        observed = lower = upper = cutoff = math.nan
    else:
        observed, lower, upper, cutoff = (
            float(value) for value in numeric_values
        )
        if not observed < lower <= upper:
            failures.append("visible_height_is_not_below_lower_bound")

    body_numeric_keys = (
        "particle_volume_m3",
        "target_volume_m3",
        "target_height_m",
        "world_height_m",
        "post_transform_signed_volume_m3",
        "post_transform_volume_residual_m3",
    )
    body_numbers: dict[str, float] = {}
    try:
        render_body_geometry_sha256 = hashlib.sha256(
            json.dumps(
                render_body_geometry,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        render_body_geometry_sha256 = None
        failures.append("render_body_geometry_not_canonical_json")
    contained_particle_count = render_body_geometry.get(
        "contained_particle_count"
    )
    if (
        isinstance(contained_particle_count, bool)
        or not isinstance(contained_particle_count, int)
        or contained_particle_count <= 0
    ):
        failures.append("render_body_contained_particle_count_invalid")
    for key in body_numeric_keys:
        value = render_body_geometry.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            failures.append(f"render_body_{key}_invalid")
        else:
            body_numbers[key] = float(value)
    expected_count = volume_gate.get("expected_cumulative_particle_count")
    programmed_volume = volume_gate.get("programmed_cumulative_volume_ml")
    height_lower = volume_gate.get("height_lower_bound_m")
    height_upper = volume_gate.get("height_upper_bound_m")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or isinstance(programmed_volume, bool)
        or not isinstance(programmed_volume, Real)
        or not math.isfinite(float(programmed_volume))
        or isinstance(height_lower, bool)
        or not isinstance(height_lower, Real)
        or isinstance(height_upper, bool)
        or not isinstance(height_upper, Real)
        or not math.isfinite(float(height_lower))
        or not math.isfinite(float(height_upper))
    ):
        failures.append("render_body_gate_binding_invalid")
    elif len(body_numbers) == len(body_numeric_keys):
        particle_volume = body_numbers["particle_volume_m3"]
        target_volume = body_numbers["target_volume_m3"]
        signed_volume = body_numbers["post_transform_signed_volume_m3"]
        residual = body_numbers["post_transform_volume_residual_m3"]
        if contained_particle_count != expected_count:
            failures.append("render_body_particle_count_mismatch")
        if not math.isclose(
            target_volume,
            expected_count * particle_volume,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            failures.append("render_body_target_volume_mismatch")
        if abs(target_volume - float(programmed_volume) * 1.0e-6) > particle_volume:
            failures.append("render_body_programmed_volume_mismatch")
        if not math.isclose(
            abs(signed_volume - target_volume), residual, rel_tol=0.0, abs_tol=1.0e-15
        ) or residual > particle_volume * 1.0e-4:
            failures.append("render_body_volume_residual_invalid")
        for key in ("target_height_m", "world_height_m"):
            if not float(height_lower) <= body_numbers[key] <= float(height_upper):
                failures.append(f"render_body_{key}_outside_original_bounds")
        if (
            render_body_geometry.get("volume_proof_coordinate_path")
            != "local_points_vec3f_then_inherited_world_transform"
            or render_body_geometry.get("body_visibility") != "inherited"
        ):
            failures.append("render_body_proof_contract_invalid")
        expected_area = (
            0.5
            * profile_wall_count
            * float(circumradius_m) ** 2
            * math.sin(2.0 * math.pi / profile_wall_count)
        )
        cross_section_area = render_body_geometry.get("cross_section_area_m2")
        world_bottom = render_body_geometry.get("world_bottom_z_m")
        try:
            world_extent = np.asarray(
                render_body_geometry.get("world_extent_m"), dtype=np.float64
            )
        except (TypeError, ValueError):
            world_extent = np.empty((0, 0), dtype=np.float64)
        hashes_valid = all(
            isinstance(render_body_geometry.get(key), str)
            and len(render_body_geometry[key]) == 64
            and all(
                character in "0123456789abcdef"
                for character in render_body_geometry[key]
            )
            for key in ("points_sha256", "extent_sha256", "world_points_sha256")
        )
        body_bindings = profile["count_driven_body_geometry_bindings"]
        expected_geometry = body_bindings[
            "by_contained_particle_count"
        ].get(str(expected_count))
        deterministic_geometry_valid = bool(
            isinstance(expected_geometry, Mapping)
            and all(
                render_body_geometry.get(key) == expected_geometry.get(key)
                for key in (
                    "points_sha256",
                    "extent_sha256",
                    "world_points_sha256",
                )
            )
            and math.isclose(
                body_numbers["target_height_m"],
                float(expected_geometry.get("target_height_m", math.nan)),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and np.allclose(
                world_extent,
                np.asarray(
                    expected_geometry.get("world_extent_m"), dtype=np.float64
                ),
                rtol=0.0,
                atol=1.0e-9,
            )
        )
        if (
            render_body_geometry.get("representation")
            != "count_driven_twin_liquid_mesh"
            or isinstance(cross_section_area, bool)
            or not isinstance(cross_section_area, Real)
            or not math.isclose(
                float(cross_section_area),
                expected_area,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or isinstance(world_bottom, bool)
            or not isinstance(world_bottom, Real)
            or not math.isclose(
                float(world_bottom), float(bottom_z_m), rel_tol=0.0, abs_tol=1.0e-9
            )
            or world_extent.shape != (2, 3)
            or not np.all(np.isfinite(world_extent))
            or np.any(world_extent[1] <= world_extent[0])
            or not math.isclose(
                float(world_extent[0, 2]),
                float(bottom_z_m),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not math.isclose(
                float(world_extent[1, 2] - world_extent[0, 2]),
                body_numbers["world_height_m"],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not math.isclose(
                float(np.mean(world_extent[:, 0])),
                float(center_xy_m[0]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not math.isclose(
                float(np.mean(world_extent[:, 1])),
                float(center_xy_m[1]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not np.allclose(
                world_extent[:, :2],
                np.asarray(
                    [
                        [5.219446482856196, -1.9941538249898683],
                        [5.245354173271724, -1.9682461345743407],
                    ],
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=1.0e-9,
            )
            or not math.isclose(
                body_numbers["target_height_m"],
                target_volume / expected_area,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                body_numbers["world_height_m"],
                body_numbers["target_height_m"],
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            or not hashes_valid
        ):
            failures.append("render_body_absolute_geometry_binding_invalid")
        if not deterministic_geometry_valid:
            failures.append(
                "render_body_deterministic_geometry_binding_invalid"
            )

    if not math.isfinite(lower) or not math.isfinite(upper) or not math.isfinite(cutoff):
        band_top_row = band_bottom_row = -1
    else:
        upper_row = projected_top_dodecagon_min_y_px(
            upper + cutoff,
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_z_m,
            circumradius_m=circumradius_m,
            wall_count=profile_wall_count,
        )
        lower_row = projected_top_dodecagon_min_y_px(
            lower + cutoff,
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_z_m,
            circumradius_m=circumradius_m,
            wall_count=profile_wall_count,
        )
        band_top_row = int(math.ceil(upper_row - 0.5))
        band_bottom_row = int(math.floor(lower_row - 0.5))
        if not 0 <= band_top_row <= band_bottom_row < occluder.shape[0]:
            failures.append("projected_acceptable_band_invalid")

    roi_widths = np.count_nonzero(roi, axis=1)
    unoccluded_widths = np.count_nonzero(roi & ~occluder, axis=1)
    minimum_counts = np.ceil(roi_widths / 3.0).astype(np.int64)
    observable_dense = (roi_widths > 0) & (unoccluded_widths >= minimum_counts)
    band_has_window = False
    if band_top_row >= 0:
        band_has_window = any(
            bool(observable_dense[row] and observable_dense[row + 1])
            for row in range(band_top_row, band_bottom_row)
        )
        if band_has_window:
            failures.append("acceptable_height_band_not_fully_censored")
    release_row = None
    if band_bottom_row >= 0:
        release_row = next(
            (
                row
                for row in range(band_bottom_row + 1, len(observable_dense) - 1)
                if observable_dense[row] and observable_dense[row + 1]
            ),
            None,
        )
    if release_row is None:
        failures.append("post_occluder_dense_release_missing")
    surface_row = surface.get("surface_row_px")
    if (
        isinstance(surface_row, bool)
        or not isinstance(surface_row, int)
        or release_row is None
        or surface_row != release_row
    ):
        failures.append("liquid_surface_does_not_contact_release_edge")

    padded = np.pad(occluder, 1)
    adjacent = np.zeros_like(occluder)
    for dy in range(3):
        for dx in range(3):
            adjacent |= padded[
                dy : dy + occluder.shape[0],
                dx : dx + occluder.shape[1],
            ]
    contact_rows: list[dict[str, int]] = []
    if release_row is not None:
        for row in (release_row, release_row + 1):
            liquid_count = int(np.count_nonzero(liquid[row] & roi[row]))
            contact_count = int(np.count_nonzero(liquid[row] & adjacent[row]))
            threshold = int(minimum_counts[row])
            contact_rows.append(
                {
                    "row_px": row,
                    "liquid_pixel_count": liquid_count,
                    "dense_threshold_pixels": threshold,
                    "contact_pixel_count": contact_count,
                }
            )
            if liquid_count < threshold or contact_count <= 0:
                failures.append("liquid_contact_is_not_dense_and_8_connected")

    profile_file_sha256 = profile.get("profile_file_sha256")
    if profile_file_sha256 is None:
        profile_file_sha256 = FILL_CAM_CARRIER_OCCLUSION_PROFILE_SHA256
    camera_attestation = (
        dict(camera_effective_state_attestation)
        if isinstance(camera_effective_state_attestation, Mapping)
        else camera_effective_state_attestation
    )
    occluder_attestation = (
        dict(occluder_effective_state_attestation)
        if isinstance(occluder_effective_state_attestation, Mapping)
        else occluder_effective_state_attestation
    )
    camera_observed_sha256 = (
        camera_attestation.get("observed_state_sha256")
        if isinstance(camera_attestation, Mapping)
        else None
    )
    occluder_observed_sha256 = (
        occluder_attestation.get("observed_state_sha256")
        if isinstance(occluder_attestation, Mapping)
        else None
    )
    return {
        "passed": not failures,
        "failure_reasons": list(dict.fromkeys(failures)),
        "status": "occlusion_censored" if not failures else "occlusion_censor_failed",
        "profile_id": profile.get("profile_id"),
        "profile_file_sha256": profile_file_sha256,
        "profile_mask_packbits_little_sha256": profile.get(
            "profile_mask_packbits_little_sha256"
        ),
        "occluder_effective_state_sha256": occluder_observed_sha256,
        "occluder_expected_effective_state_sha256": profile.get(
            "occluder_effective_state_sha256"
        ),
        "occluder_effective_state_attestation": occluder_attestation,
        "source_stage_sha256": source_stage_sha256,
        "source_geometry_layer_sha256": source_geometry_layer_sha256,
        "camera_effective_state_sha256": camera_observed_sha256,
        "camera_expected_effective_state_sha256": profile.get(
            "camera_effective_state_sha256"
        ),
        "camera_effective_state_attestation": camera_attestation,
        "occluder_prim_path": profile.get("occluder_prim_path"),
        "numeric_height_passed": False,
        "observed_height_above_cutoff_m": observed,
        "lower_bound_height_above_cutoff_m": lower,
        "upper_bound_height_above_cutoff_m": upper,
        "projected_acceptable_band_rows_inclusive": [
            band_top_row,
            band_bottom_row,
        ],
        "acceptable_band_has_observable_dense_window": band_has_window,
        "first_post_occluder_dense_release_row_px": release_row,
        "observed_surface_row_px": surface_row,
        "contact_connectivity": 8,
        "contact_rows": contact_rows,
        "render_body_geometry_verified": not any(
            reason.startswith("render_body_") for reason in failures
        ),
        "render_body_geometry_sha256": render_body_geometry_sha256,
    }

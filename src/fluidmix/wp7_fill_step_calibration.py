"""Development-only builder for the WP7 camera height/volume mapping.

This is the only Task-1A module allowed to import Part-A projection and
geometry helpers.  Its output is a closed numeric mapping consumed by the
production tracker without exposing simulator labels, residue, or palette.
"""

from __future__ import annotations

from dataclasses import replace

from .physx_volume_gate import visible_height_from_surface_row_m
from .volume_height_curve import (
    DEFAULT_GEOMETRY,
    volume_for_height_m,
    volume_height_curve_digest,
)
from .wp7_fill_step_tracker import (
    CALIBRATION_SCHEMA,
    FROZEN_FILLCAM_CAMERA_SHA256,
    FROZEN_GEOMETRY_SHA256,
    FROZEN_VOLUME_CURVE_SHA256,
    FillStepTrackerError,
    PixelToMetreCalibration,
    calibration_payload_sha256,
)


def build_development_calibration(
    *,
    calibration_id: str,
    fill_height_surface_row_samples: tuple[tuple[float, float], ...],
) -> PixelToMetreCalibration:
    """Freeze observed CNN pixel heights through the reviewed V3/V4 geometry."""

    if not calibration_id or len(fill_height_surface_row_samples) < 2:
        raise FillStepTrackerError("development calibration needs an ID and two samples")
    ordered = tuple(sorted(fill_height_surface_row_samples))
    if len({height for height, _ in ordered}) != len(ordered):
        raise FillStepTrackerError("development fill-height samples must be unique")
    center_xy_m = tuple(float(item) for item in DEFAULT_GEOMETRY["center_xy_m"])
    bottom_z_m = float(DEFAULT_GEOMETRY["bottom_z_m"])
    circumradius_m = float(DEFAULT_GEOMETRY["circumradius_m"])
    maximum_height_m = (
        float(DEFAULT_GEOMETRY["mouth_z_m"])
        - float(DEFAULT_GEOMETRY["bottom_z_m"])
    )
    wall_count = int(DEFAULT_GEOMETRY["wall_count"])
    heights_m: list[float] = []
    volumes_ml: list[float] = []
    for fill_height_px, surface_row_px in ordered:
        height_m = visible_height_from_surface_row_m(
            surface_row_px,
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_z_m,
            circumradius_m=circumradius_m,
            maximum_height_m=maximum_height_m,
            wall_count=wall_count,
        )
        heights_m.append(height_m)
        volumes_ml.append(volume_for_height_m(height_m))
    if volume_height_curve_digest() != FROZEN_VOLUME_CURVE_SHA256:
        raise FillStepTrackerError("development volume-height curve drifted")
    calibration = PixelToMetreCalibration(
        schema_version=CALIBRATION_SCHEMA,
        calibration_id=calibration_id,
        camera_effective_state_sha256=FROZEN_FILLCAM_CAMERA_SHA256,
        geometry_sha256=FROZEN_GEOMETRY_SHA256,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
        fill_height_px_samples=tuple(item[0] for item in ordered),
        visible_height_m_samples=tuple(heights_m),
        twin_geometry_equivalent_volume_ml_samples=tuple(volumes_ml),
        artifact_sha256="0" * 64,
    )
    return replace(
        calibration,
        artifact_sha256=calibration_payload_sha256(calibration),
    )

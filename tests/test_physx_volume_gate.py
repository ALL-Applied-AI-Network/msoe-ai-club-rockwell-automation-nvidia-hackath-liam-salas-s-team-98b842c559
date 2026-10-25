from __future__ import annotations

import pytest
import numpy as np

from fluidmix.physx_volume_gate import (
    expected_cylindrical_height_m,
    find_dense_surface_row,
    height_within_tolerance,
    particle_bulk_height_m,
    projected_dodecagon_vertices_px,
    projected_top_dodecagon_min_y_px,
    projected_vial_geometry_roi_mask,
    visible_height_from_surface_row_m,
)


@pytest.mark.parametrize(
    ("volume_ml", "expected_height_m"),
    [
        (10.0, 0.019122235),
        (20.0, 0.038244470),
        (60.0, 0.114733411),
    ],
)
def test_expected_height_uses_fitted_collider_cross_section(
    volume_ml: float,
    expected_height_m: float,
) -> None:
    # Independently measured from the authored 12-sided collider: 522.951416 mm^2.
    assert expected_cylindrical_height_m(
        volume_ml,
        0.000522951416,
    ) == pytest.approx(expected_height_m, rel=0.0, abs=5.0e-10)


def test_particle_bulk_height_debiases_trimmed_uniform_column() -> None:
    bottom_m = 0.9762065645
    radius_m = 0.0015
    expected_height_m = 0.019122235
    centers = np.linspace(
        bottom_m + radius_m,
        bottom_m + expected_height_m - radius_m,
        1001,
    )

    assert particle_bulk_height_m(centers, radius_m) == pytest.approx(
        expected_height_m,
        rel=0.0,
        abs=1.0e-12,
    )


def test_six_millimetre_gate_rejects_a_thin_particle_smear() -> None:
    bottom_m = 0.9762065645
    radius_m = 0.0015
    smear_height_m = 0.008
    centers = np.linspace(
        bottom_m + radius_m,
        bottom_m + smear_height_m - radius_m,
        1001,
    )
    observed = particle_bulk_height_m(centers, radius_m)

    assert not height_within_tolerance(observed, 0.019122235, 0.006)


@pytest.mark.parametrize(
    ("height_m", "expected_min_y_px"),
    [
        (0.019122235, 360.752),
        (0.038244470, 336.516),
        (0.114733411, 225.447),
    ],
)
def test_fillcam_projection_matches_independent_usd_measurement(
    height_m: float,
    expected_min_y_px: float,
) -> None:
    assert projected_top_dodecagon_min_y_px(
        height_m,
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=0.9762065645,
        circumradius_m=0.0132029216,
    ) == pytest.approx(expected_min_y_px, rel=0.0, abs=0.002)


def test_projected_contour_vertices_preserve_the_frozen_minimum_row() -> None:
    geometry = {
        "center_xy_m": (5.2324003281, -1.9811999798),
        "bottom_z_m": 0.9762065645,
        "circumradius_m": 0.0132029216,
    }
    vertices = projected_dodecagon_vertices_px(0.019122235, **geometry)

    assert vertices.shape == (12, 2)
    assert np.all(np.isfinite(vertices))
    assert float(np.min(vertices[:, 1])) == pytest.approx(
        projected_top_dodecagon_min_y_px(0.019122235, **geometry),
        rel=0.0,
        abs=1.0e-12,
    )


def test_visible_height_inverts_fillcam_projection() -> None:
    geometry = {
        "center_xy_m": (5.2324003281, -1.9811999798),
        "bottom_z_m": 0.9762065645,
        "circumradius_m": 0.0132029216,
    }
    surface_row = projected_top_dodecagon_min_y_px(0.019122235, **geometry)

    assert visible_height_from_surface_row_m(
        surface_row,
        **geometry,
        maximum_height_m=0.2000536047,
    ) == pytest.approx(0.019122235, rel=0.0, abs=1.0e-8)


def test_mask_surface_ignores_a_narrow_connected_falling_stream() -> None:
    mask = np.zeros((480, 640), dtype=bool)
    roi = np.zeros_like(mask)
    roi[75:418, 290:335] = True
    mask[360:418, 294:329] = True
    mask[80:360, 309:312] = True

    evidence = find_dense_surface_row(
        mask,
        geometry_roi_mask=roi,
    )

    assert evidence["surface_row_px"] == 360
    assert evidence["liquid_pixels_inside_geometry_roi"] == int(np.count_nonzero(mask))


def test_mask_surface_can_cross_the_holder_occlusion() -> None:
    mask = np.zeros((480, 640), dtype=bool)
    roi = np.zeros_like(mask)
    roi[75:418, 290:335] = True
    mask[212:244, 294:329] = True
    mask[265:418, 294:329] = True

    evidence = find_dense_surface_row(mask, geometry_roi_mask=roi)

    assert evidence["surface_row_px"] == 212


def test_mask_surface_requires_two_broad_rows() -> None:
    mask = np.zeros((480, 640), dtype=bool)
    roi = np.zeros_like(mask)
    roi[75:418, 290:335] = True
    mask[100:200, 309:312] = True

    with pytest.raises(ValueError, match="two-row dense"):
        find_dense_surface_row(
            mask,
            geometry_roi_mask=roi,
        )


def test_geometry_roi_is_rasterized_from_floor_and_mouth_hulls() -> None:
    roi = projected_vial_geometry_roi_mask(
        (480, 640),
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=0.9762065645,
        mouth_z_m=1.1762601692,
        circumradius_m=0.0132029216,
    )
    ys, xs = np.nonzero(roi)

    assert [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] == [279, 66, 339, 416]
    assert 44 <= int(np.count_nonzero(roi[400])) <= 46
    assert 59 <= int(np.count_nonzero(roi[78])) <= 62

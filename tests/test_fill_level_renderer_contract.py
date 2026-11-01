"""CPU-only checks for renderer summary integrity rules."""

import colorsys
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.render_fill_level_dataset import (  # noqa: E402
    BASELINE_STICKER_INK_DEPTH_M,
    REAL_LOOK_LABEL_SEGMENTS,
    REAL_LOOK_STICKER_INK_DEPTH_M,
    cylindrical_label_patch_mesh,
    foam_manifest_record,
    frame_integrity_ok,
    liquid_surface_material_params,
    real_look_glass_material_params,
    real_look_liquid_material_params,
    real_look_print_area_fraction,
    real_look_print_primitives,
    real_look_small_bubble_count,
    real_look_surface_and_foam_rgb,
    srgb_to_linear_rgb,
)


def test_positive_foamy_frame_may_have_no_visible_liquid_pixels():
    record = {
        "level_ml": 0.065,
        "foam": {"enabled": True, "visible_in_camera": True},
        "label_stats": {
            "class_pixels": {
                "empty": 16850,
                "liquid": 0,
                "foam": 1391,
                "blocked": 3411,
                "ignore": 285548,
            }
        },
    }
    assert frame_integrity_ok(record)


def test_zero_level_without_visible_empty_pixels_fails_integrity():
    record = {
        "level_ml": 0.0,
        "foam": {"enabled": False, "visible_in_camera": False},
        "label_stats": {
            "class_pixels": {
                "empty": 0,
                "liquid": 10,
                "foam": 0,
                "blocked": 4,
                "ignore": 100,
            }
        },
    }
    assert not frame_integrity_ok(record)


def test_real_look_surface_is_grey_reflective_target_and_foam_is_liquid_tinted():
    liquid = np.array([0.02, 0.15, 0.04])
    surface, foam = real_look_surface_and_foam_rgb(liquid, 0.15)
    _hue, surface_saturation, surface_value = colorsys.rgb_to_hsv(*surface)
    assert 0.16 <= surface_value <= 0.28
    assert 0.08 <= surface_saturation <= 0.250001
    assert np.ptp(surface) < np.ptp(liquid)
    assert np.ptp(foam) < np.ptp(liquid)
    assert surface[1] > surface[0]
    assert foam[1] > foam[0]
    assert np.max(surface) < np.min(foam)


def test_real_look_surface_is_opaque_without_changing_baseline_material():
    assert liquid_surface_material_params(False) == (0.90, 0.06)
    assert liquid_surface_material_params(True) == (1.0, 0.30)


def test_real_look_glass_is_clear_and_baseline_keeps_source_material():
    assert real_look_glass_material_params(False) is None
    params = real_look_glass_material_params(True)
    assert params["diffuse_rgb"] == (0.92, 0.96, 0.98)
    assert 0.10 <= params["opacity"] <= 0.18
    assert 0.05 <= params["roughness"] <= 0.10
    assert params["ior"] == pytest.approx(1.46)
    expected_f0 = ((params["ior"] - 1.0) / (params["ior"] + 1.0)) ** 2
    assert params["specular_rgb"] == pytest.approx(
        (expected_f0, expected_f0, expected_f0), abs=1e-12
    )
    assert max(params["specular_rgb"]) < 0.06
    assert params["use_specular_workflow"] == 1
    assert params["metallic"] == 0.0


def test_clear_glass_authors_metallic_instead_of_getting_a_missing_input():
    source = (ROOT / "scripts" / "render_fill_level_dataset.py").read_text()
    assert 'clear_glass_shader.GetInput("metallic")' not in source
    assert "clear_glass_shader.CreateInput(" in source
    assert '"metallic", Sdf.ValueTypeNames.Float' in source


def test_real_look_print_is_outward_facing_and_includes_barcode_bank():
    primitives = real_look_print_primitives()
    assert BASELINE_STICKER_INK_DEPTH_M == -0.0008
    assert REAL_LOOK_STICKER_INK_DEPTH_M == 0.00012
    assert len(primitives) == 11
    assert all(
        primitive[2] == REAL_LOOK_STICKER_INK_DEPTH_M
        for primitive in primitives
    )
    assert [primitive[0] for primitive in primitives[:4]] == [
        "TextBar00",
        "TextBar01",
        "TextBar02",
        "TextBar03",
    ]
    barcode = primitives[4:]
    assert [primitive[0] for primitive in barcode] == [
        f"Barcode{index:02d}" for index in range(7)
    ]
    assert all(primitive[3] == -0.0030 for primitive in barcode)
    assert all(primitive[5] == 0.0100 for primitive in barcode)
    assert [primitive[1] for primitive in barcode] == pytest.approx(
        [-0.0048, -0.0032, -0.0016, 0.0, 0.0016, 0.0032, 0.0048]
    )
    assert real_look_print_area_fraction() == pytest.approx(
        0.14973076923076922
    )


def test_real_look_label_mesh_wraps_around_vial_at_measured_size():
    geometry = cylindrical_label_patch_mesh(
        0.032,
        0.040,
        0.0137,
        37.0,
        center_xy=(0.12, -0.08),
        center_z=0.21,
    )
    points = geometry["points"]
    assert points.shape == (2 * (REAL_LOOK_LABEL_SEGMENTS + 1), 3)
    radii = np.linalg.norm(points[:, :2] - [0.12, -0.08], axis=1)
    np.testing.assert_allclose(radii, 0.0137, rtol=0.0, atol=1e-12)
    assert points[:, 2].min() == pytest.approx(0.19)
    assert points[:, 2].max() == pytest.approx(0.23)
    assert geometry["face_vertex_counts"] == (4,) * REAL_LOOK_LABEL_SEGMENTS
    assert len(geometry["face_vertex_indices"]) == 4 * REAL_LOOK_LABEL_SEGMENTS
    # A flat card would have one surface normal; the radial direction changes
    # materially across this 32 mm arc.
    lower_edge = points[: REAL_LOOK_LABEL_SEGMENTS + 1, :2] - [0.12, -0.08]
    first = lower_edge[0] / np.linalg.norm(lower_edge[0])
    last = lower_edge[-1] / np.linalg.norm(lower_edge[-1])
    assert float(np.dot(first, last)) < 0.0


def test_wrapped_print_is_outside_paper_and_rolls_with_label():
    paper = cylindrical_label_patch_mesh(
        0.032, 0.040, 0.0137, -150.0, 7.0
    )["points"]
    ink = cylindrical_label_patch_mesh(
        0.006,
        0.002,
        0.0137 + REAL_LOOK_STICKER_INK_DEPTH_M,
        -150.0,
        7.0,
        center_u_m=0.003,
        center_v_m=0.008,
        segments=REAL_LOOK_LABEL_SEGMENTS,
    )["points"]
    assert np.linalg.norm(ink[:, :2], axis=1).min() > np.linalg.norm(
        paper[:, :2], axis=1
    ).max()
    assert np.ptp(ink[:, 2]) > 0.002


def test_every_wrapped_print_face_clears_paper_at_label_extremes():
    def sampled_face_points(geometry):
        points = geometry["points"]
        samples = []
        for offset in range(0, len(geometry["face_vertex_indices"]), 4):
            indices = geometry["face_vertex_indices"][offset : offset + 4]
            corners = points[list(indices)]
            for u in np.linspace(0.0, 1.0, 9):
                for v in np.linspace(0.0, 1.0, 9):
                    samples.append(
                        (1.0 - u) * (1.0 - v) * corners[0]
                        + u * (1.0 - v) * corners[1]
                        + u * v * corners[2]
                        + (1.0 - u) * v * corners[3]
                    )
        return np.asarray(samples)

    for label_width_mm in (28.0, 36.0):
        for label_height_mm in (34.0, 44.0):
            width_scale = label_width_mm / 20.0
            height_scale = label_height_mm / 26.0
            for paper_radius in (0.0132, 0.0144):
                for roll_deg in (-8.0, 0.0, 8.0):
                    for (
                        _name,
                        bar_x,
                        bar_depth,
                        bar_z,
                        bar_width,
                        bar_height,
                    ) in real_look_print_primitives():
                        geometry = cylindrical_label_patch_mesh(
                            bar_width * width_scale,
                            bar_height * height_scale,
                            paper_radius + bar_depth,
                            137.0,
                            roll_deg,
                            center_u_m=bar_x * width_scale,
                            center_v_m=bar_z * height_scale,
                            segments=REAL_LOOK_LABEL_SEGMENTS,
                        )
                        sampled = sampled_face_points(geometry)
                        clearance = (
                            np.linalg.norm(sampled[:, :2], axis=1)
                            - paper_radius
                        )
                        # The ideal paper cylinder is a stricter boundary than
                        # its inward-sagging 16-facet mesh. Retaining at least
                        # 40 microns here prevents every ink face from crossing
                        # behind or becoming occluded by the paper.
                        assert clearance.min() >= 0.00004


def test_real_look_liquid_material_retains_hue_without_v2_brightness_regression():
    target = colorsys.hsv_to_rgb(0.40, 0.78, 0.155)
    params = real_look_liquid_material_params(target, "very_dark")
    diffuse_hsv = colorsys.rgb_to_hsv(*params["diffuse_rgb"])
    specular_hsv = colorsys.rgb_to_hsv(*params["specular_rgb"])
    assert diffuse_hsv[0] == pytest.approx(0.40, abs=1e-12)
    assert diffuse_hsv[1] >= 0.78
    assert diffuse_hsv[2] == pytest.approx(
        srgb_to_linear_rgb([0.155] * 3)[0], abs=1e-15
    )
    assert specular_hsv[0] == pytest.approx(0.40, abs=1e-12)
    assert specular_hsv[1] >= 0.78
    assert specular_hsv[2] == pytest.approx(0.0106875, abs=1e-15)
    assert params["use_specular_workflow"] == 1
    assert params["metallic"] == 0.0

    for value, darkness_band, expected_specular_value in (
        (0.08, "very_dark", 0.007875),
        (0.16, "very_dark", 0.010875),
        (0.20, "dark", 0.012375),
        (0.35, "dark", 0.018),
    ):
        mapped = real_look_liquid_material_params(
            colorsys.hsv_to_rgb(0.40, 0.78, value), darkness_band
        )
        mapped_diffuse_hsv = colorsys.rgb_to_hsv(*mapped["diffuse_rgb"])
        mapped_specular_hsv = colorsys.rgb_to_hsv(*mapped["specular_rgb"])
        assert mapped_diffuse_hsv[0] == pytest.approx(0.40, abs=1e-12)
        assert mapped_diffuse_hsv[2] == pytest.approx(
            srgb_to_linear_rgb([value] * 3)[0], abs=1e-15
        )
        assert mapped_specular_hsv[0] == pytest.approx(0.40, abs=1e-12)
        assert mapped_specular_hsv[1] >= 0.78
        assert mapped_specular_hsv[2] == pytest.approx(
            expected_specular_value, abs=1e-15
        )

    darkest = real_look_liquid_material_params(
        colorsys.hsv_to_rgb(0.21, 0.40, 0.03), "extreme_dark"
    )
    assert max(darkest["diffuse_rgb"]) == pytest.approx(
        srgb_to_linear_rgb([0.03] * 3)[0], abs=1e-15
    )
    assert max(darkest["specular_rgb"]) == 0.0


def test_real_look_surface_color_contract_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="three finite"):
        real_look_surface_and_foam_rgb([0.1, 0.2], 0.15)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        real_look_surface_and_foam_rgb([0.1, 0.2, 0.3], 1.1)


def test_srgb_targets_are_decoded_for_linear_preview_surface_materials():
    actual = srgb_to_linear_rgb([0.0, 0.04045, 0.5])
    np.testing.assert_allclose(
        actual,
        [0.0, 0.04045 / 12.92, 0.21404114048223255],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        srgb_to_linear_rgb([0.05, 0.16, 0.35]),
        [0.003935939504088967, 0.02198094707988721, 0.10048150993772331],
        rtol=0.0,
        atol=1e-15,
    )


def test_srgb_decode_rejects_malformed_or_out_of_range_colors():
    with pytest.raises(ValueError, match="three finite"):
        srgb_to_linear_rgb([0.1, 0.2])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        srgb_to_linear_rgb([0.1, 0.2, 1.1])
    with pytest.raises(ValueError, match="three finite"):
        real_look_liquid_material_params([0.1, 0.2])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        real_look_liquid_material_params([0.1, 0.2, 1.1])
    with pytest.raises(ValueError, match="darkness_band"):
        real_look_liquid_material_params([0.1, 0.2, 0.3], "unknown")


def test_wrapped_label_geometry_rejects_flat_or_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        cylindrical_label_patch_mesh(0.0, 0.04, 0.0137, 0.0)
    with pytest.raises(ValueError, match="circumference"):
        cylindrical_label_patch_mesh(0.10, 0.04, 0.0137, 0.0)
    with pytest.raises(ValueError, match="segments"):
        cylindrical_label_patch_mesh(0.03, 0.04, 0.0137, 0.0, segments=0)


def test_real_look_foam_has_at_least_three_small_bubbles_when_enabled():
    assert real_look_small_bubble_count(0.0, True) == 3
    assert real_look_small_bubble_count(0.25, True) == 3
    assert real_look_small_bubble_count(1.0, True) == 10
    assert real_look_small_bubble_count(1.0, False) == 0
    with pytest.raises(ValueError, match="bumpiness"):
        real_look_small_bubble_count(-0.1, True)


def test_baseline_foam_manifest_shape_is_unchanged_and_v2_logs_authored_bubbles():
    spec_foam = {"enabled": True, "bumpiness": 0.4}
    assert foam_manifest_record(
        spec_foam, True, real_look=False, bubble_count=4
    ) == {
        "enabled": True,
        "bumpiness": 0.4,
        "visible_in_camera": True,
    }
    assert foam_manifest_record(
        spec_foam, True, real_look=True, bubble_count=4
    )["authored_small_bubble_count"] == 4

    injected = {**spec_foam, "authored_small_bubble_count": 99}
    assert foam_manifest_record(
        injected, True, real_look=False, bubble_count=0
    ) == {
        "enabled": True,
        "bumpiness": 0.4,
        "visible_in_camera": True,
    }
    assert foam_manifest_record(
        injected, True, real_look=True, bubble_count=4
    )["authored_small_bubble_count"] == 4

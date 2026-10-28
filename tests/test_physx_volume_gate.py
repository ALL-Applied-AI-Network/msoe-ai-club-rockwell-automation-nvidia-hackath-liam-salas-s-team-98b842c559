from __future__ import annotations

import copy
import math
import struct

import pytest
import numpy as np

from fluidmix.physx_volume_gate import (
    _carrier_occlusion_profile_mask,
    evaluate_carrier_occlusion_censored_height,
    expected_carrier_occluder_state,
    expected_cylindrical_height_m,
    find_dense_surface_row,
    height_within_tolerance,
    load_fillcam_carrier_occlusion_profile,
    particle_bulk_height_m,
    projected_dodecagon_vertices_px,
    projected_top_dodecagon_min_y_px,
    projected_vial_geometry_roi_mask,
    visible_height_from_surface_row_m,
)
from fluidmix.transform_attestation import (
    attest_effective_state_matrix,
    attest_finite_f64_matrix4x4,
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
        (0.019122235, 360.742660),
        (0.038244470, 336.506989),
        (0.114733411, 225.443580),
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

    assert vertices.shape == (32, 2)
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


def _occluded_stage2_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    profile = load_fillcam_carrier_occlusion_profile()
    occluder = _carrier_occlusion_profile_mask(profile)
    roi = projected_vial_geometry_roi_mask(
        (480, 640),
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=0.9762065645,
        mouth_z_m=1.1762601692,
        circumradius_m=0.0132029216,
    )
    liquid = np.zeros_like(roi)
    liquid[251:] = roi[251:] & ~occluder[251:]
    surface = find_dense_surface_row(liquid, geometry_roi_mask=roi)
    gate = {
        "required": True,
        "state_passed": True,
        "visible_passed": False,
        "passed": False,
        "failure_reasons": ["visible_height_outside_geometry_volume_bounds"],
        "programmed_cumulative_volume_ml": 51.0,
        "expected_cumulative_particle_count": 4524,
        "height_lower_bound_m": 0.08904263108834418,
        "height_upper_bound_m": 0.09841553962395935,
        "visible_height_policy": {
            "required": True,
            "opaque_base_cutoff_height_m": 0.036756504061235984,
            "lower_bound_height_above_cutoff_m": 0.05228612702710819,
            "upper_bound_height_above_cutoff_m": 0.06165903556272337,
        },
        "visible_mask_evidence": {
            "observed_height_above_cutoff_m": 0.051302453734033415,
            "lower_bound_height_above_cutoff_m": 0.05228612702710819,
            "upper_bound_height_above_cutoff_m": 0.06165903556272337,
        },
    }
    particle_volume = 1.1272866937444934e-8
    target_volume = 4524 * particle_volume
    body = {
        "representation": "count_driven_twin_liquid_mesh",
        "contained_particle_count": 4524,
        "particle_volume_m3": particle_volume,
        "cross_section_area_m2": 0.0005229514163272396,
        "target_volume_m3": target_volume,
        "target_height_m": 0.09752043580485942,
        "world_height_m": 0.09752043720889136,
        "world_bottom_z_m": 0.9762065644917131,
        "world_extent_m": [
            [5.219446482856196, -1.9941538249898683, 0.9762065644917131],
            [5.245354173271724, -1.9682461345743407, 1.0737270017006044],
        ],
        "points_sha256": (
            "a475e4c516f059cb43f14c2e0e8de6e90e4d81eba08a65fa9584d533f520d511"
        ),
        "extent_sha256": (
            "13981f99fe55d18c264128eea2de44e45ce9aecf64de408e71c85cceccf35f2b"
        ),
        "world_points_sha256": (
            "8f0c7e19e55af05dd0bf43448bc46f39ac5aaa65860a18903e918b007d7fd6e7"
        ),
        "post_transform_signed_volume_m3": 0.00005099845096163262,
        "post_transform_volume_residual_m3": 9.366317400491542e-13,
        "volume_proof_coordinate_path": (
            "local_points_vec3f_then_inherited_world_transform"
        ),
        "body_visibility": "inherited",
    }
    return liquid, roi, surface, gate, body


def _evaluate_occlusion_fixture(
    liquid: np.ndarray,
    roi: np.ndarray,
    surface: dict[str, object],
    gate: dict[str, object],
    body: dict[str, object],
    **overrides: object,
) -> dict[str, object]:
    profile = load_fillcam_carrier_occlusion_profile()
    arguments = {
        "profile": profile,
        "source_stage_sha256": profile["source_stage_sha256"],
        "source_geometry_layer_sha256": profile[
            "source_geometry_layer_sha256"
        ],
        "camera_effective_state_attestation": attest_effective_state_matrix(
            profile["camera_expected_effective_state"],
            profile["camera_expected_effective_state"],
            matrix_field="world_transform_matrix",
        ),
        "occluder_effective_state_attestation": attest_effective_state_matrix(
            expected_carrier_occluder_state(profile),
            expected_carrier_occluder_state(profile),
            matrix_field="mesh_world_transform_matrix",
            matrix_raw_hash_field="mesh_world_transform_f64_sha256",
        ),
    }
    arguments.update(overrides)
    return evaluate_carrier_occlusion_censored_height(
        liquid,
        geometry_roi_mask=roi,
        surface=surface,
        volume_gate=gate,
        render_body_geometry=body,
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=0.9762065645,
        circumradius_m=0.0132029216,
        **arguments,
    )


def test_source_bound_carrier_profile_accepts_only_dense_edge_contact() -> None:
    liquid, roi, surface, gate, body = _occluded_stage2_fixture()

    proof = _evaluate_occlusion_fixture(liquid, roi, surface, gate, body)

    assert proof["passed"] is True
    assert proof["status"] == "occlusion_censored"
    assert proof["projected_acceptable_band_rows_inclusive"] == [251, 265]
    assert proof["first_post_occluder_dense_release_row_px"] == 267
    assert proof["observed_surface_row_px"] == 267
    assert proof["acceptable_band_has_observable_dense_window"] is False
    assert [row["contact_pixel_count"] for row in proof["contact_rows"]] == [
        11,
        10,
    ]


def test_genuinely_short_fill_below_carrier_fails_contact_proof() -> None:
    _liquid, roi, _surface, gate, body = _occluded_stage2_fixture()
    occluder = _carrier_occlusion_profile_mask(
        load_fillcam_carrier_occlusion_profile()
    )
    short = np.zeros_like(roi)
    short[272:] = roi[272:] & ~occluder[272:]
    surface = find_dense_surface_row(short, geometry_roi_mask=roi)
    gate["visible_mask_evidence"]["observed_height_above_cutoff_m"] = 0.047

    proof = _evaluate_occlusion_fixture(short, roi, surface, gate, body)

    assert proof["passed"] is False
    assert "liquid_surface_does_not_contact_release_edge" in proof["failure_reasons"]


def test_thin_contact_and_above_upper_result_cannot_use_occlusion_alternate() -> None:
    _liquid, roi, surface, gate, body = _occluded_stage2_fixture()
    thin = np.zeros_like(roi)
    thin[267:269, 304:306] = True

    thin_proof = _evaluate_occlusion_fixture(thin, roi, surface, gate, body)
    assert thin_proof["passed"] is False
    assert "liquid_contact_is_not_dense_and_8_connected" in thin_proof[
        "failure_reasons"
    ]

    gate["visible_mask_evidence"]["observed_height_above_cutoff_m"] = 0.07
    overfill = _evaluate_occlusion_fixture(thin, roi, surface, gate, body)
    assert overfill["passed"] is False
    assert "visible_height_is_not_below_lower_bound" in overfill["failure_reasons"]


def test_occlusion_alternate_requires_exact_integer_body_count() -> None:
    liquid, roi, surface, gate, body = _occluded_stage2_fixture()
    body["contained_particle_count"] = 4524.9

    proof = _evaluate_occlusion_fixture(liquid, roi, surface, gate, body)

    assert proof["passed"] is False
    assert "render_body_contained_particle_count_invalid" in proof[
        "failure_reasons"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("world_bottom_z_m", 0.5),
        (
            "world_extent_m",
            [[5.0, -2.0, 0.5], [5.1, -1.9, 0.5975204372088914]],
        ),
        (
            "world_extent_m",
            [
                [
                    5.2324003281 - 1.0,
                    -1.9811999798 - 0.5,
                    0.9762065644917131,
                ],
                [
                    5.2324003281 + 1.0,
                    -1.9811999798 + 0.5,
                    1.0737270017006044,
                ],
            ],
        ),
        ("points_sha256", "f" * 64),
    ],
)
def test_occlusion_alternate_rejects_body_geometry_or_hash_drift(
    field: str, value: object
) -> None:
    liquid, roi, surface, gate, body = _occluded_stage2_fixture()
    body[field] = value

    proof = _evaluate_occlusion_fixture(liquid, roi, surface, gate, body)

    assert proof["passed"] is False
    if field == "points_sha256":
        assert "render_body_deterministic_geometry_binding_invalid" in proof[
            "failure_reasons"
        ]
    else:
        assert "render_body_absolute_geometry_binding_invalid" in proof[
            "failure_reasons"
        ]


def test_occlusion_profile_pins_top_left_boundary_convention() -> None:
    profile = load_fillcam_carrier_occlusion_profile()
    mask = _carrier_occlusion_profile_mask(profile)

    assert mask[253, 284] == 0
    assert mask[252, 284] == 1
    assert mask[253, 285] == 1
    expected_occluder = expected_carrier_occluder_state(profile)
    attestation = attest_effective_state_matrix(
        expected_occluder,
        expected_occluder,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    )
    assert attestation["matrix_comparison"]["expected_matrix_f64_sha256"] == (
        "2df92340d8b7d9726f2bac8d79686c3ce1680ae0bc43ed326dca64e0319d22d0"
    )
    assert profile["mesh_expected_world_transform_provenance"]["source"] == (
        "read_only_cpu_openusd_composition"
    )


@pytest.mark.parametrize(
    ("binding", "reason"),
    [
        ("source_stage_sha256", "source_stage_sha256_mismatch"),
        ("source_geometry_layer_sha256", "source_geometry_layer_sha256_mismatch"),
        (
            "camera_effective_state_attestation",
            "camera_effective_state_attestation_invalid",
        ),
    ],
)
def test_occlusion_alternate_fails_closed_on_source_or_camera_drift(
    binding: str, reason: str
) -> None:
    liquid, roi, surface, gate, body = _occluded_stage2_fixture()

    proof = _evaluate_occlusion_fixture(
        liquid, roi, surface, gate, body, **{binding: "f" * 64}
    )

    assert proof["passed"] is False
    assert reason in proof["failure_reasons"]


def test_l40s_one_ulp_camera_readback_is_not_rejected_by_single_gb10_hash() -> None:
    gb10_value = -0.8097141285498902
    l40s_value = -0.8097141285498903

    def ordered_bits(value: float) -> int:
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)

    assert abs(ordered_bits(gb10_value) - ordered_bits(l40s_value)) == 1

    liquid, roi, surface, gate, body = _occluded_stage2_fixture()
    profile = load_fillcam_carrier_occlusion_profile()
    l40s_state = copy.deepcopy(profile["camera_expected_effective_state"])
    l40s_state["world_transform_matrix"][2][0] = l40s_value
    l40s_attestation = attest_effective_state_matrix(
        profile["camera_expected_effective_state"],
        l40s_state,
        matrix_field="world_transform_matrix",
    )
    assert l40s_attestation["observed_state_sha256"] == (
        "b5d6e350c532428638d66be35bbcf94b2704ee477105f95ee31bfcc6a4192264"
    )
    proof = _evaluate_occlusion_fixture(
        liquid,
        roi,
        surface,
        gate,
        body,
        camera_effective_state_attestation=l40s_attestation,
    )

    assert proof["passed"] is True


def test_one_ulp_occluder_world_matrix_keeps_raw_hash_evidence() -> None:
    liquid, roi, surface, gate, body = _occluded_stage2_fixture()
    profile = load_fillcam_carrier_occlusion_profile()
    expected_state = expected_carrier_occluder_state(profile)
    observed_state = copy.deepcopy(expected_state)
    observed_state["mesh_world_transform_matrix"][0][0] = math.nextafter(
        observed_state["mesh_world_transform_matrix"][0][0],
        -math.inf,
    )
    matrix_proof = attest_finite_f64_matrix4x4(
        expected_state["mesh_world_transform_matrix"],
        observed_state["mesh_world_transform_matrix"],
    )
    observed_state["mesh_world_transform_f64_sha256"] = matrix_proof[
        "observed_matrix_f64_sha256"
    ]
    occluder_attestation = attest_effective_state_matrix(
        expected_state,
        observed_state,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    )

    proof = _evaluate_occlusion_fixture(
        liquid,
        roi,
        surface,
        gate,
        body,
        occluder_effective_state_attestation=occluder_attestation,
    )

    assert proof["passed"] is True
    assert proof["occluder_effective_state_sha256"] == occluder_attestation[
        "observed_state_sha256"
    ]
    assert proof["occluder_expected_effective_state_sha256"] != proof[
        "occluder_effective_state_sha256"
    ]

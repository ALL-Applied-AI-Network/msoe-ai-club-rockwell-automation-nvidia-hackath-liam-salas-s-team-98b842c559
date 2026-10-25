"""CPU contracts for the bounded F2 one-system three-pour experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fluidmix.physx_three_pour import (
    CAMERA_PATH,
    COMPOSITION_COMPONENTS,
    COMPOSITION_PALETTE_DENOMINATOR,
    DIFFUSION_DT_S,
    DIFFUSION_NEIGHBOR_RADIUS_M,
    EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S,
    FLUIDMIX_NAMESPACE,
    NOMINAL_TANK_TRANSMISSION_COLORS,
    O1B_MATERIAL_PARAMETERS,
    PARTICLE_CONTACT_OFFSET_M,
    SOURCE_LIQUID_PATH,
    SOURCE_STREAM_PATH,
    TWIN_REFERENCE_TIME_CODE,
    build_composition_palette,
    build_palette_transmission_colors,
    build_source_compositions,
    build_twin_collider_specs,
    build_three_pour_plan,
    diffuse_particle_compositions,
    distribute_particle_count,
    map_compositions_to_palette,
    validate_three_pour_receipt,
)
import fluidmix.physx_three_pour as physx_three_pour


SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence", ROOT / "scripts" / "physx_three_pour_sequence.py"
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
sequence = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(sequence)


def test_frozen_plan_has_exact_recipe_order_counts_and_nominal_volume_limit():
    plan = build_three_pour_plan()

    assert plan["schema_version"] == 4
    assert plan["experiment"] == "physx_three_pour_f2"
    assert plan["seed"] == 20261025
    assert plan["fluid_rest_offset_m"] == 0.0015
    assert plan["nominal_particle_volume_ml"] == pytest.approx(0.027)
    assert [row["tank_id"] for row in plan["pours"]] == ["t200", "t400", "t100"]
    assert [row["programmed_volume_ml"] for row in plan["pours"]] == [10, 10, 40]
    assert [row["particle_count"] for row in plan["pours"]] == [370, 370, 1481]
    assert [row["duration_steps"] for row in plan["pours"]] == [706, 605, 673]
    assert [row["end_step"] for row in plan["pours"]] == [706, 1311, 1984]
    assert [row["stage_capture_step"] for row in plan["pours"]] == [706, 1311, 2002]
    assert [row["emission_chunk_count"] for row in plan["pours"]] == [118, 101, 113]
    assert all(
        row["emission_schedule"][-1]["physics_step"] == row["end_step"]
        for row in plan["pours"]
    )
    assert plan["totals"] == {
        "programmed_volume_ml": 60,
        "particle_count": 2221,
        "nominal_mass_convention_equivalent_ml": pytest.approx(59.967),
        "nominal_error_ml": pytest.approx(-0.033),
    }
    assert plan["timing"] == {
        "physics_hz": 60,
        "capture_hz": 10,
        "no_inter_pour_settle": True,
        "fill_end_step": 1984,
        "primary_final_capture_step": 2002,
        "primary_final_offset_s": 0.3,
        "source": "video5_recipe_median_active_tank_linear_fit_v1",
        "source_fit_r2": pytest.approx(0.92595),
        "source_fit_rmse_s": pytest.approx(2.371),
        "source_fit_loocv_rmse_s": pytest.approx(4.765),
        "interpretation": "real_like_approximation_not_measured_tank_boundaries",
    }
    assert plan["mixing_model"] == "pbd_motion_global_vial_blend_render_stream_fallback_v1"
    assert plan["nominal_tank_transmission_colors"] == {
        "t200": [0.95, 0.38, 0.08],
        "t400": [0.18, 0.55, 0.75],
        "t100": [0.92, 0.98, 1.0],
    }
    assert plan["transmission_colors_uncalibrated"] is True
    assert plan["fallback_trigger"] == {
        "run_id": "20261025-brev-physx-palette-smoke-a1",
        "receipt_sha256": "c34da1875d098cc791ce2237d40cf4b32fcac14acab98f247dc32c98b9993e17",
        "visual_review_sha256": "8d8e1f0077340d9e64ce954bd32d365d6b6c4ca3252ac72b6977feb63190fe49",
        "decision": "fail",
        "failure_reasons": [
            "maximum_particle_movement_m_zero",
            "proto_indices_unchanged",
            "before_after_rgb_byte_identical",
        ],
    }
    fallback = plan["fallback_render_model"]
    assert fallback["physical_motion"] == "one_physx_pbd_vial_particle_system"
    assert fallback["vial_representation"] == "visible_isosurface_one_global_material"
    assert fallback["stream_representation"] == "render_only_mesh_current_tank_material"
    assert fallback["global_blend_rule"] == "cumulative_particle_weighted_linear_transmission_parameter"
    assert fallback["blend_update_timing"] == "every_emission_event"
    assert fallback["spatial_mixing_color_rendered"] is False
    assert fallback["spatial_layering_color_rendered"] is False
    assert fallback["particle_composition_tracked"] is False
    assert fallback["diffusion_simulated"] is False
    assert plan["semantic_gate"] == {
        "required": True,
        "semantic_label": "liquid",
        "semantic_type": "class",
        "required_targets": ["visible_isosurface", "render_only_stream_mesh"],
        "annotator": "semantic_segmentation",
        "resolution": [640, 480],
        "aligned_mask_per_kept_rgb": True,
        "nonempty_liquid_pixels_per_mask": True,
        "unsupported_isosurface_semantics_outcome": "fail",
        "qualification_training_eligible": False,
    }
    states = fallback["stage_states"]
    assert [row["cumulative_particle_counts"] for row in states] == [
        [370, 0, 0],
        [370, 370, 0],
        [370, 370, 1481],
    ]
    np.testing.assert_allclose(states[0]["component_fractions"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(states[1]["component_fractions"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(
        states[2]["component_fractions"],
        [370 / 2221, 370 / 2221, 1481 / 2221],
    )
    np.testing.assert_allclose(states[0]["transmission_color"], [0.95, 0.38, 0.08])
    np.testing.assert_allclose(states[1]["transmission_color"], [0.565, 0.465, 0.415])
    np.testing.assert_allclose(
        states[2]["transmission_color"],
        np.asarray([370, 370, 1481])
        @ np.asarray([[0.95, 0.38, 0.08], [0.18, 0.55, 0.75], [0.92, 0.98, 1.0]])
        / 2221,
    )
    assert plan["claims"]["physically_displaced_volume"] is False
    assert plan["claims"]["absorbance_fitted"] is False
    assert plan["claims"]["particle_composition_tracked"] is False
    assert plan["claims"]["dye_concentration_simulated"] is False
    assert plan["claims"]["diffusion_simulated"] is False
    assert plan["claims"]["spatial_mixing_color_rendered"] is False
    assert plan["claims"]["spatial_layering_color_rendered"] is False
    assert plan["claims"]["mixing_physically_validated"] is False
    assert plan["claims"]["navier_stokes_validated"] is False


def test_twin_scene_contract_is_fixed_to_existing_fill_station():
    assert CAMERA_PATH == "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
    assert SOURCE_LIQUID_PATH == "/FillStation/FillStation_LocalRoot/Fill_liquid"
    assert SOURCE_STREAM_PATH == "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream"
    assert FLUIDMIX_NAMESPACE == "/FillStation/FluidmixF2"
    assert TWIN_REFERENCE_TIME_CODE == 1.0
    assert O1B_MATERIAL_PARAMETERS == {
        "specular_transmission_color": (0.18, 0.55, 0.75),
        "specular_transmission_scattering_depth_m": 0.01,
        "thin_walled": False,
        "specular_transmission_weight": 1.0,
        "diffuse_reflection_weight": 0.0,
        "specular_reflection_ior": 1.333,
        "specular_reflection_roughness": 0.08,
        "enable_opacity": False,
    }


def test_twin_colliders_are_invisible_cube_only_containment_geometry():
    specs = build_twin_collider_specs()

    assert len(specs) == 13
    assert specs[0]["name"] == "Floor"
    assert [spec["name"] for spec in specs[1:]] == [
        f"Wall{index:02d}" for index in range(12)
    ]
    assert all(spec["type"] == "Cube" for spec in specs)
    assert all(spec["visible"] is False for spec in specs)
    assert all(str(spec["path"]).startswith(f"{FLUIDMIX_NAMESPACE}/Colliders/") for spec in specs)
    assert specs[0]["center_m"][2] < 0.9762065645
    assert all(spec["center_m"][2] > 0.9762065645 for spec in specs[1:])


@pytest.mark.parametrize("total,chunks", [(370, 20), (1481, 48), (1, 1)])
def test_emission_chunk_distribution_is_positive_ordered_and_exact(total, chunks):
    counts = distribute_particle_count(total, chunks)

    assert len(counts) == chunks
    assert sum(counts) == total
    assert all(type(value) is int and value > 0 for value in counts)
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("total,chunks", [(0, 1), (1, 0), (3, 4), (True, 1), (1, False)])
def test_emission_chunk_distribution_rejects_invalid_or_empty_chunks(total, chunks):
    with pytest.raises((TypeError, ValueError)):
        distribute_particle_count(total, chunks)


def test_emission_schedule_spans_every_pour_at_ten_hz_and_ends_exactly():
    plan = build_three_pour_plan()
    previous_end = 0

    for pour in plan["pours"]:
        schedule = pour["emission_schedule"]
        steps = [event["physics_step"] for event in schedule]
        counts = [event["particle_count"] for event in schedule]

        assert pour["start_step"] == previous_end + 1
        assert len(schedule) == pour["emission_chunk_count"]
        assert sum(counts) == pour["particle_count"]
        assert all(count > 0 for count in counts)
        assert steps == sorted(set(steps))
        assert steps[-1] == pour["end_step"]
        assert max(np.diff([previous_end, *steps])) <= 6
        previous_end = pour["end_step"]


def test_source_compositions_are_one_hot_in_frozen_component_order():
    for index, tank_id in enumerate(COMPOSITION_COMPONENTS):
        rows = build_source_compositions(tank_id, 3)
        expected = np.zeros((3, 3), dtype=np.float64)
        expected[:, index] = 1.0
        np.testing.assert_array_equal(rows, expected)


def test_nominal_transmission_colors_and_mixed_palette_color_are_uncalibrated_parameters():
    assert NOMINAL_TANK_TRANSMISSION_COLORS == {
        "t200": (0.95, 0.38, 0.08),
        "t400": (0.18, 0.55, 0.75),
        "t100": (0.92, 0.98, 1.0),
    }
    one_hot_colors = build_palette_transmission_colors(np.eye(3))
    np.testing.assert_allclose(
        one_hot_colors,
        np.asarray([NOMINAL_TANK_TRANSMISSION_COLORS[tank] for tank in COMPOSITION_COMPONENTS]),
    )
    mixed = build_palette_transmission_colors(np.asarray([[0.5, 0.5, 0.0]]))
    np.testing.assert_allclose(mixed[0], [0.565, 0.465, 0.415])


def test_neighbor_diffusion_is_conservative_nonnegative_and_simplex_preserving():
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.004, 0.0, 0.0]],
        dtype=np.float64,
    )
    compositions = np.eye(3, dtype=np.float64)

    updated = diffuse_particle_compositions(compositions, positions)

    assert np.all(np.isfinite(updated))
    assert np.all(updated >= 0.0)
    np.testing.assert_allclose(updated.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(updated.sum(axis=0), compositions.sum(axis=0), atol=1e-14)
    assert not np.array_equal(updated, compositions)
    assert updated[0, 1] > 0.0
    assert updated[1, 0] > 0.0


def test_neighbor_diffusion_is_identity_without_neighbors_and_does_not_mutate_input():
    positions = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    compositions = np.asarray([[0.75, 0.25, 0.0], [0.0, 0.5, 0.5]])
    original = compositions.copy()

    updated = diffuse_particle_compositions(compositions, positions)

    np.testing.assert_array_equal(updated, original)
    np.testing.assert_array_equal(compositions, original)
    assert updated is not compositions


def test_neighbor_diffusion_caps_dense_exchange_for_stability():
    positions = np.zeros((6, 3), dtype=np.float64)
    compositions = np.vstack([np.eye(3), np.eye(3)])

    updated = diffuse_particle_compositions(
        compositions,
        positions,
        diffusion_coefficient_m2_s=1e6,
    )

    assert np.all(updated >= 0.0)
    np.testing.assert_allclose(updated.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(updated.sum(axis=0), compositions.sum(axis=0), atol=1e-14)


def test_neighbor_diffusion_defaults_are_tied_to_particle_contact_offset():
    assert PARTICLE_CONTACT_OFFSET_M == 0.003
    assert DIFFUSION_NEIGHBOR_RADIUS_M == 2 * PARTICLE_CONTACT_OFFSET_M
    assert EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S == 1e-5
    assert DIFFUSION_DT_S == pytest.approx(1 / 60)


@pytest.mark.parametrize(
    ("compositions", "positions", "kwargs"),
    [
        ([[1.0, 0.0]], [[0.0, 0.0, 0.0]], {}),
        ([[1.0, 0.0, 0.0]], [[0.0, 0.0]], {}),
        ([[0.8, 0.1, 0.0]], [[0.0, 0.0, 0.0]], {}),
        ([[1.1, -0.1, 0.0]], [[0.0, 0.0, 0.0]], {}),
        ([[np.nan, 0.0, 1.0]], [[0.0, 0.0, 0.0]], {}),
        ([[1.0, 0.0, 0.0]], [[np.inf, 0.0, 0.0]], {}),
        ([[True, False, False]], [[0.0, 0.0, 0.0]], {}),
        ([[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], {"diffusion_coefficient_m2_s": -1.0}),
        ([[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], {"neighbor_radius_m": 0.0}),
        ([[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], {"dt_s": False}),
    ],
)
def test_neighbor_diffusion_rejects_invalid_inputs(compositions, positions, kwargs):
    with pytest.raises((TypeError, ValueError)):
        diffuse_particle_compositions(compositions, positions, **kwargs)


def test_composition_palette_is_finite_simplex_and_maps_one_hot_and_mixed_rows():
    palette = build_composition_palette()

    assert COMPOSITION_PALETTE_DENOMINATOR == 4
    assert palette.shape == (15, 3)
    np.testing.assert_array_equal(palette[:3], np.eye(3))
    assert np.all(palette >= 0.0)
    np.testing.assert_array_equal(palette.sum(axis=1), np.ones(15))
    one_hot_indices = map_compositions_to_palette(np.eye(3), palette)
    np.testing.assert_array_equal(one_hot_indices, [0, 1, 2])
    mixed = np.asarray([[0.51, 0.49, 0.0], [0.24, 0.27, 0.49]])
    mixed_indices = map_compositions_to_palette(mixed, palette)
    np.testing.assert_allclose(palette[mixed_indices], [[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    np.testing.assert_array_equal(
        map_compositions_to_palette(mixed, palette),
        mixed_indices,
    )
    np.testing.assert_array_equal(
        map_compositions_to_palette([[0.625, 0.375, 0.0]], palette),
        [3],
    )


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (build_composition_palette, (0,)),
        (build_composition_palette, (True,)),
        (map_compositions_to_palette, ([[0.5, 0.5, 0.1]],)),
        (map_compositions_to_palette, ([[np.nan, 0.0, 1.0]],)),
        (build_source_compositions, ("t999", 1)),
        (build_source_compositions, ("t200", 0)),
    ],
)
def test_composition_helpers_reject_invalid_inputs(call, args):
    with pytest.raises((TypeError, ValueError)):
        call(*args)


def _passing_receipt() -> dict[str, object]:
    plan = build_three_pour_plan()
    stages = []
    cumulative = 0
    for ordinal, pour in enumerate(plan["pours"], start=1):
        cumulative += pour["particle_count"]
        stages.append(
            {
                "ordinal": ordinal,
                "tank_id": pour["tank_id"],
                "programmed_volume_ml": pour["programmed_volume_ml"],
                "emitted_particle_count": pour["particle_count"],
                "cumulative_particle_count": cumulative,
                "start_step": pour["start_step"],
                "duration_steps": pour["duration_steps"],
                "end_step": pour["end_step"],
                "stage_capture_step": pour["stage_capture_step"],
                "emission_chunk_count": pour["emission_chunk_count"],
                "emission_events": pour["emission_schedule"],
                "finite_positions": True,
                "moved_under_physics": True,
                "maximum_particle_movement_m": 0.01,
                "escaped_particle_count": 0,
                "isosurface_nonempty": True,
                "used_gpu_memory_mib_before": 1000,
                "used_gpu_memory_mib_after": 1100,
            }
        )
    stage_steps = [pour["stage_capture_step"] for pour in plan["pours"]]
    frame_steps = sorted(
        {
            *range(
                physx_three_pour.CAPTURE_CADENCE_STEPS,
                plan["timing"]["primary_final_capture_step"] + 1,
                physx_three_pour.CAPTURE_CADENCE_STEPS,
            ),
            *stage_steps,
        }
    )
    frame_hashes = [f"{100 + index:064x}" for index in range(len(frame_steps))]
    stage_frame_indices = [frame_steps.index(step) for step in stage_steps]
    stage_hashes = [frame_hashes[index] for index in stage_frame_indices]
    mask_hashes = [f"{200 + index:064x}" for index in range(len(frame_steps))]
    return {
        "schema_version": 4,
        "experiment": "physx_three_pour_f2",
        "status": "pass",
        "versions": {
            "isaac": "6.0.1.0",
            "kit": "110.1.2",
            "physx": "110.1.13",
        },
        "gpu_simulation": {
            "cuda_device_available": True,
            "scene_enable_gpu_dynamics": True,
            "broadphase": "GPU",
        },
        "particle_system": {
            "count": 1,
            "exactly_one": True,
            "mass_api_count": 0,
            "all_particle_sets_target_one_system": True,
            "collider_types": ["Cube"] * 13,
            "collider_paths": [f"{FLUIDMIX_NAMESPACE}/Colliders/{index}" for index in range(13)],
            "all_colliders_invisible": True,
            "decorative_vial_collision_api": False,
            "pbd_material_binding": True,
            "render_material_binding": True,
            "finite_motion_observed": True,
            "maximum_particle_movement_m": 0.012,
        },
        "scene_contract": {
            "default_prim": "/FillStation",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "frozen_time_code": 1.0,
            "physics_namespace": FLUIDMIX_NAMESPACE,
            "source_layer_records": [{"sha256": "a" * 64}],
            "source_layers_unchanged": True,
            "hidden_source_visibility": {
                path: {str(code): "invisible" for code in physx_three_pour.VISIBILITY_CHECK_TIME_CODES}
                for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH)
            },
            "forbidden_authored_paths": [],
            "custom_camera_count": 0,
            "custom_light_count": 0,
            "custom_backdrop_count": 0,
        },
        "camera": {
            "path": CAMERA_PATH,
            "source_has_authored_clipping_range": True,
            "clipping_range_authored_by_runner": False,
            "clipping_range_effective": list(physx_three_pour.CAMERA_CLIPPING_RANGE),
            "resolution": [640, 480],
        },
        "material": {
            "pbd_path": f"{FLUIDMIX_NAMESPACE}/PBDMaterial",
            "vial_representation": "visible_isosurface_one_global_material",
            "stream_representation": "render_only_mesh_current_tank_material",
            "omnisurface_parameter_family": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in O1B_MATERIAL_PARAMETERS.items()
                if key != "specular_transmission_color"
            },
            "tank_transmission_colors": {
                tank: list(color)
                for tank, color in NOMINAL_TANK_TRANSMISSION_COLORS.items()
            },
            "transmission_colors_uncalibrated": True,
            "global_blend_rule": "cumulative_particle_weighted_linear_transmission_parameter",
            "isosurface_global_material_binding": True,
        },
        "emitter": {
            "source_nozzle_path": physx_three_pour.SOURCE_NOZZLE_PATH,
            "source_nozzle_valid": True,
            "origin_m": list(physx_three_pour.NOZZLE_EXIT_CENTER_M),
            "velocity_m_s": [0.0, 0.0, -0.1],
        },
        "pre_pour_parity": {
            "passed": True,
            "thresholds": physx_three_pour.PARITY_THRESHOLDS,
            "mean_abs_rgb": 0.5,
            "p95_abs_rgb": 2.0,
            "fraction_abs_rgb_gt_16": 0.0,
            "reference_sha256": "b" * 64,
            "f2_prefill_sha256": "c" * 64,
            "absdiff_sha256": "d" * 64,
        },
        "pours": stages,
        "fallback_evidence": plan["fallback_trigger"],
        "stream_render": {
            "representation": "render_only_mesh_current_tank_material",
            "records": [
                {
                    "ordinal": ordinal,
                    "tank_id": pour["tank_id"],
                    "start_step": pour["start_step"],
                    "end_step": pour["end_step"],
                    "transmission_color": list(
                        NOMINAL_TANK_TRANSMISSION_COLORS[pour["tank_id"]]
                    ),
                    "active_only_during_own_pour": True,
                    "state_artifact_path": f"stream/tank-{ordinal}.json",
                    "state_artifact_sha256": str(6 + ordinal) * 64,
                }
                for ordinal, pour in enumerate(plan["pours"], start=1)
            ],
        },
        "global_blend": {
            "model": "pbd_motion_global_vial_blend_render_stream_fallback_v1",
            "component_order": ["t200", "t400", "t100"],
            "global_blend_rule": "cumulative_particle_weighted_linear_transmission_parameter",
            "blend_update_timing": "every_emission_event",
            "blend_update_count": sum(
                pour["emission_chunk_count"] for pour in plan["pours"]
            ),
            "updated_on_every_emission_event": True,
            "all_updates_usd_readback_verified": True,
            "emission_trace_event_count": sum(
                pour["emission_chunk_count"] for pour in plan["pours"]
            ),
            "emission_trace_path": "blend/emission-updates.json",
            "emission_trace_sha256": "a" * 64,
            "particle_composition_tracked": False,
            "diffusion_simulated": False,
            "spatial_mixing_color_rendered": False,
            "spatial_layering_color_rendered": False,
            "stage_states": [
                {
                    **state,
                    "usd_readback_transmission_color": state[
                        "transmission_color"
                    ],
                    "state_artifact_path": f"blend/stage-{state['ordinal']}.json",
                    "state_artifact_sha256": str(3 + state["ordinal"]) * 64,
                }
                for state in plan["fallback_render_model"]["stage_states"]
            ],
        },
        "totals": {
            "programmed_volume_ml": 60,
            "emitted_particle_count": 2221,
            "final_particle_count": 2221,
            "nominal_mass_convention_equivalent_ml": 59.967,
            "nominal_error_ml": -0.033,
        },
        "isosurface": {
            "enabled": True,
            "visible_for_global_blend": True,
            "mesh_point_count": 900,
            "mesh_face_count": 1700,
            "mesh_finite_points": True,
            "mesh_extent_nonempty": True,
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 64,
            "warmup_updates": 60,
            "frame_count": len(frame_steps),
            "frame_paths": [
                f"frames/frame-{index:04d}.png"
                for index in range(len(frame_steps))
            ],
            "frame_sha256": frame_hashes,
            "frame_physics_steps": frame_steps,
            "stage_frame_indices": stage_frame_indices,
            "stage_frame_sha256": stage_hashes,
            "stage_frame_physics_steps": [706, 1311, 2002],
            "primary_final_frame_sha256": stage_hashes[-1],
            "video_sha256": "f" * 64,
            "video_decodable": True,
            "video_decoded_frame_count": len(frame_steps),
            "stage_frames_are_video_subsets": True,
        },
        "semantics": {
            "semantic_label": "liquid",
            "semantic_type": "class",
            "isosurface_semantic_bound": True,
            "stream_semantic_bound": True,
            "annotator": "semantic_segmentation",
            "resolution": [640, 480],
            "pairs": [
                {
                    "physics_step": step,
                    "rgb_path": f"frames/frame-{index:04d}.png",
                    "rgb_sha256": frame_hashes[index],
                    "mask_path": f"masks/mask-{index:04d}.png",
                    "mask_sha256": mask_hashes[index],
                    "mask_readable": True,
                    "mask_resolution": [640, 480],
                    "liquid_pixel_count": 120 + index,
                }
                for index, step in enumerate(frame_steps)
            ],
            "stage_mask_sha256": [mask_hashes[index] for index in stage_frame_indices],
            "semantic_gate_passed": True,
            "training_eligible": False,
            "ineligibility_reason": "qualification_run_not_case_dataset",
        },
        "memory": {
            "hard_limit_mib": 40000,
            "peak_used_mib": 1400,
            "hard_limit_exceeded": False,
            "sample_source": "nvidia_smi_gpu_memory_used",
            "sample_scope": "gpu_framebuffer",
            "stage_growth_stop_mib": 4096,
            "stage_growth_events": [],
            "isosurface_enabled_seconds": 10.0,
            "isosurface_time_limit_seconds": 2700,
        },
        "timing": {
            "wall_seconds": 100.0,
            "wall_limit_seconds": 5400,
            "physics_hz": 60,
            "capture_hz": 10,
            "pour_duration_steps": [706, 605, 673],
            "pour_end_steps": [706, 1311, 1984],
            "no_inter_pour_settle": True,
            "fill_end_step": 1984,
            "primary_final_capture_step": 2002,
            "primary_final_offset_s": 0.3,
            "executed_physics_steps": 2002,
            "source": "video5_recipe_median_active_tank_linear_fit_v1",
            "source_fit_r2": 0.92595,
            "source_fit_rmse_s": 2.371,
            "source_fit_loocv_rmse_s": 4.765,
            "interpretation": "real_like_approximation_not_measured_tank_boundaries",
        },
        "mixing_assumption": plan["mixing_assumption"],
        "claims": plan["claims"],
        "errors": [],
    }


def test_substantive_receipt_accepts_observed_three_pour_run():
    assert validate_three_pour_receipt(_passing_receipt()) == ()


def test_receipt_does_not_require_settling_or_stability():
    receipt = _passing_receipt()
    receipt["pours"][0]["stable"] = False
    receipt["pours"][1]["settling_steps"] = 999

    assert validate_three_pour_receipt(receipt) == ()


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda r: r["particle_system"].update(count=2, exactly_one=False), "particle_system_count"),
        (lambda r: r["particle_system"].update(mass_api_count=1), "mass_api_present"),
        (lambda r: r["particle_system"].update(collider_types=["Cylinder"]), "twin_collider_contract"),
        (lambda r: r["pours"][1].update(tank_id="t100"), "pour_order_or_volume"),
        (lambda r: r["pours"][0].update(escaped_particle_count=1), "particles_escaped"),
        (lambda r: r["pours"][2].update(end_step=1985), "real_timing_contract"),
        (
            lambda r: r["pours"][0]["emission_events"][0].update(physics_step=1),
            "emission_schedule_mismatch",
        ),
        (
            lambda r: r["timing"].update(no_inter_pour_settle=False),
            "real_timing_contract",
        ),
        (
            lambda r: r["render"].update(primary_final_frame_sha256="0" * 64),
            "primary_capture_contract",
        ),
        (
            lambda r: r["global_blend"]["stage_states"][2].update(
                cumulative_particle_counts=[369, 370, 1482]
            ),
            "global_blend_state_mismatch",
        ),
        (
            lambda r: r["global_blend"].update(spatial_mixing_color_rendered=True),
            "global_blend_contract",
        ),
        (
            lambda r: r["global_blend"].update(spatial_layering_color_rendered=True),
            "global_blend_contract",
        ),
        (
            lambda r: r["global_blend"].update(blend_update_count=0),
            "global_blend_not_observed",
        ),
        (
            lambda r: r["global_blend"].update(
                all_updates_usd_readback_verified=False
            ),
            "global_blend_contract",
        ),
        (
            lambda r: r["global_blend"].update(emission_trace_sha256=None),
            "global_blend_trace_missing",
        ),
        (
            lambda r: r["particle_system"].update(finite_motion_observed=False),
            "particle_motion_not_observed",
        ),
        (
            lambda r: r["global_blend"]["stage_states"][1].update(
                state_artifact_sha256="4" * 64
            ),
            "global_blend_state_artifact_missing",
        ),
        (
            lambda r: r["stream_render"]["records"][1].update(tank_id="t100"),
            "stream_render_contract",
        ),
        (
            lambda r: r["fallback_evidence"].update(receipt_sha256="0" * 64),
            "fallback_trigger_mismatch",
        ),
        (
            lambda r: r["semantics"].update(isosurface_semantic_bound=False),
            "liquid_semantic_binding_missing",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(liquid_pixel_count=0),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(mask_readable=False),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(
                rgb_sha256=r["semantics"]["pairs"][0]["rgb_sha256"]
            ),
            "liquid_semantic_mask_invalid",
        ),
        (lambda r: r["isosurface"].update(mesh_point_count=0), "isosurface_empty"),
        (lambda r: r["render"].update(video_decodable=False), "video_not_decodable"),
        (lambda r: r["render"].update(stage_frames_are_video_subsets=False), "stage_frame_not_subset"),
        (
            lambda r: r["render"].update(
                stage_frame_sha256=[r["render"]["stage_frame_sha256"][0]] * 3
            ),
            "primary_capture_contract",
        ),
        (
            lambda r: r["render"].update(stage_frame_indices=[99, 100, 101]),
            "primary_capture_contract",
        ),
        (
            lambda r: r["render"]["frame_physics_steps"].pop(10),
            "frame_sequence_contract",
        ),
        (
            lambda r: r["render"]["frame_sha256"].__setitem__(
                10, r["render"]["frame_sha256"][9]
            ),
            "frame_sequence_contract",
        ),
        (lambda r: r["memory"].update(hard_limit_exceeded=True), "memory_limit_exceeded"),
        (
            lambda r: r["pours"][1].update(
                used_gpu_memory_mib_before=1000,
                used_gpu_memory_mib_after=5096,
            ),
            "stage_memory_growth_stop",
        ),
        (lambda r: r["timing"].update(wall_seconds=5400.1), "wall_time_exceeded"),
        (
            lambda r: r["camera"].update(
                source_has_authored_clipping_range=False,
                clipping_range_effective=[1.0, 1000000.0],
            ),
            "source_fillcam_contract",
        ),
        (lambda r: r["pre_pour_parity"].update(passed=False), "pre_pour_twin_parity"),
        (
            lambda r: r["claims"].update(sim_to_real_validated=True),
            "claim_limit_mismatch",
        ),
    ],
)
def test_substantive_receipt_rejects_real_failures(mutator, expected):
    receipt = deepcopy(_passing_receipt())
    mutator(receipt)

    assert expected in validate_three_pour_receipt(receipt)


def test_version_metadata_is_not_a_substantive_gate_failure():
    receipt = _passing_receipt()
    receipt["versions"]["physx"] = None

    assert validate_three_pour_receipt(receipt) == ()


def test_runner_plan_only_path_needs_no_isaac_and_refuses_overwrite(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    output = tmp_path / "plan"
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    command = [sys.executable, str(script), "--out", str(output), "--stage", str(stage), "--plan-only"]

    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "plan.json").read_text(encoding="utf-8")) == build_three_pour_plan()
    repeated = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "output path already exists" in repeated.stderr


def test_runner_cli_freezes_capture_retry_default_and_reports_invalid_value(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--max-capture-tries MAX_CAPTURE_TRIES" in help_result.stdout
    assert "(default: 20)" in help_result.stdout
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")

    invalid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(tmp_path / "bad"),
            "--stage",
            str(stage),
            "--plan-only",
            "--max-capture-tries",
            "0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert invalid.stderr.rstrip().endswith(
        "physx_three_pour_sequence.py: error: max_capture_tries must be a positive integer"
    )


def test_kit_start_disables_async_throttling_for_stopped_timeline(monkeypatch):
    started = []
    running_app = object()

    isaacsim = types.ModuleType("isaacsim")

    def simulation_app(config):
        started.append(config)
        return object()

    isaacsim.SimulationApp = simulation_app
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: None if not started else running_app
    omni.kit = omni_kit
    omni_kit.app = omni_kit_app
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.app", omni_kit_app)

    app, owned = sequence._start_or_reuse_kit()

    assert app is running_app
    assert owned is not None
    assert started == [
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "PathTracing",
            "extra_args": [
                "--/exts/isaacsim.core.throttling/enable_async=false"
            ],
        }
    ]


def test_replicator_capture_contract_waits_for_render_without_advancing_physics():
    calls = []

    class Orchestrator:
        def set_capture_on_play(self, value):
            calls.append(("set_capture_on_play", value))

        def step(self, **kwargs):
            calls.append(("step", kwargs))

    rep = types.SimpleNamespace(orchestrator=Orchestrator())

    sequence._configure_replicator_capture(rep)
    sequence._replicator_same_state_step(rep, rt_subframes=8)

    assert calls == [
        ("set_capture_on_play", False),
        (
            "step",
            {
                "delta_time": 0.0,
                "pause_timeline": True,
                "rt_subframes": 8,
                "wait_for_render": True,
            },
        ),
    ]


def test_pre_pour_parity_uses_frozen_full_frame_limits():
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    within = np.ones((2, 2, 3), dtype=np.uint8)
    outside = np.full((2, 2, 3), 17, dtype=np.uint8)

    accepted = sequence._parity_metrics(np, reference, within)
    rejected = sequence._parity_metrics(np, reference, outside)

    assert accepted["mean_abs_rgb"] == 1.0
    assert accepted["p95_abs_rgb"] == 1.0
    assert accepted["fraction_abs_rgb_gt_16"] == 0.0
    assert accepted["passed"] is True
    assert rejected["passed"] is False


def test_receipt_preserves_the_renderer_compatibility_contract(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", "unused", "--stage", str(stage)]
    )

    receipt = sequence._empty_receipt(args)

    assert receipt["render"]["capture_on_play"] is False
    assert receipt["render"]["wait_for_render"] is True
    assert receipt["render"]["same_state_delta_time"] == 0.0
    assert receipt["render"]["async_throttling_enabled"] is None
    assert receipt["render"]["rejected_rgb_diagnostics"] == []
    assert receipt["render"]["rt_subframes"] == 64
    assert receipt["render"]["warmup_updates"] == 60
    assert receipt["camera"] == {
        "path": CAMERA_PATH,
        "source_has_authored_clipping_range": False,
        "clipping_range_authored_by_runner": False,
        "clipping_range_effective": None,
        "resolution": [640, 480],
    }


def test_rejected_full_shape_rgb_is_saved_only_as_diagnostic_evidence(tmp_path):
    rgb = np.full((2, 3, 3), 17, dtype=np.uint8)

    class FakeCv2:
        COLOR_RGB2BGR = 7

        @staticmethod
        def cvtColor(value, code):
            assert code == FakeCv2.COLOR_RGB2BGR
            return value[..., ::-1]

        @staticmethod
        def imwrite(path, value):
            assert value.shape == (2, 3, 3)
            Path(path).write_bytes(b"diagnostic png fixture")
            return True

    record = sequence._write_rejected_rgb_diagnostic(
        FakeCv2,
        np,
        tmp_path,
        rgb,
        step=0,
        phase="prefill",
    )

    artifact = tmp_path / record["path"]
    assert artifact.read_bytes() == b"diagnostic png fixture"
    assert record == {
        "path": "diagnostics/rejected-rgb-step-00000-prefill.png",
        "sha256": hashlib.sha256(b"diagnostic png fixture").hexdigest(),
        "shape": [2, 3, 3],
        "dtype": "uint8",
        "channel_mean": [17.0, 17.0, 17.0],
        "channel_std": [0.0, 0.0, 0.0],
        "spatial_rgb_std": 0.0,
        "minimum": 17,
        "maximum": 17,
        "spatially_constant": True,
        "included_in_frame_count": False,
        "included_in_video": False,
        "included_as_stage": False,
    }

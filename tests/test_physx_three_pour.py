"""CPU contracts for the bounded F2 one-system three-pour experiment."""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
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
    build_paint_ball_three_pour_plan,
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


def test_failure_evidence_preserves_authoring_phase_and_traceback():
    try:
        raise RuntimeError("invalid USD palette path")
    except RuntimeError as exc:
        evidence = sequence._failure_evidence(
            exc, phase="author_palette_prototypes"
        )

    assert evidence["phase"] == "author_palette_prototypes"
    assert evidence["exception_type"] == "RuntimeError"
    assert evidence["message"] == "invalid USD palette path"
    assert "Traceback (most recent call last)" in evidence["traceback"]
    assert "RuntimeError: invalid USD palette path" in evidence["traceback"]


def test_runner_configures_zero_origin_before_current_time_and_records_readback():
    class FakeTimeline:
        def __init__(self):
            self.calls = []
            self.start_time = 1.0 / 30.0
            self.current_time = self.start_time
            self.time_codes_per_second = 30.0

        def stop(self):
            self.calls.append(("stop",))

        def set_start_time(self, value):
            self.calls.append(("set_start_time", value))
            self.start_time = value

        def set_current_time(self, value):
            self.calls.append(("set_current_time", value))
            self.current_time = max(value, self.start_time)

        def set_time_codes_per_second(self, value):
            self.calls.append(("set_time_codes_per_second", value))
            self.time_codes_per_second = value

        def set_ticks_per_frame(self, value):
            self.calls.append(("set_ticks_per_frame", value))

        def set_target_framerate(self, value):
            self.calls.append(("set_target_framerate", value))

        def set_play_every_frame(self, value):
            self.calls.append(("set_play_every_frame", value))

        def commit(self):
            self.calls.append(("commit",))

        def get_start_time(self):
            return self.start_time

        def get_current_time(self):
            return self.current_time

        def get_time_codes_per_second(self):
            return self.time_codes_per_second

    class FakeStage:
        def GetStartTimeCode(self):
            return 0.0

        def GetTimeCodesPerSecond(self):
            return 60.0

    class FakeApp:
        def __init__(self):
            self.update_count = 0

        def update(self):
            self.update_count += 1

    class FakeUsd:
        @staticmethod
        def EditContext(_stage, _session_layer):
            return nullcontext()

    timeline = FakeTimeline()
    app = FakeApp()

    observed = sequence._configure_zero_origin_timeline(
        FakeStage(), object(), app, timeline, FakeUsd, physics_hz=60
    )

    assert timeline.calls.index(("set_start_time", 0.0)) < timeline.calls.index(
        ("set_current_time", 0.0)
    )
    assert app.update_count == 1
    assert observed == {
        "stage_start_time_code": 0.0,
        "stage_time_codes_per_second": 60.0,
        "timeline_start_time_s": 0.0,
        "timeline_current_time_s": 0.0,
        "timeline_time_codes_per_second": 60.0,
    }


def test_timeline_callback_failure_evidence_names_only_the_failed_predicates():
    evidence = sequence._timeline_callback_contract_evidence(
        {
            "callback_index": 1,
            "dt_s": 1.0 / 60.0,
            "timeline_time_s": 2.0 / 60.0,
            "app_update_index": 1,
        },
        global_step=1,
        app_update_index=1,
        physics_hz=60,
    )

    assert evidence == {
        "actual_callback": {
            "callback_index": 1,
            "dt_s": 1.0 / 60.0,
            "timeline_time_s": 2.0 / 60.0,
            "app_update_index": 1,
        },
        "expected": {
            "callback_index": 1,
            "dt_s": 1.0 / 60.0,
            "timeline_time_s": 1.0 / 60.0,
            "app_update_index": 1,
        },
        "failed_predicates": ["timeline_time_s"],
        "passed": False,
    }


def test_frozen_plan_has_exact_recipe_order_counts_and_nominal_volume_limit():
    plan = build_three_pour_plan()

    assert plan["schema_version"] == 15
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
    assert plan["mixing_model"] == "timeline_owned_runtime_shared_points_stable_usd_ids_pbd_motion_composition_diffusion_stream_prefix_proxy_global_blend_fallback_v5"
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
    assert fallback["vial_representation"] == "exact_standard_mesh_proxy_one_global_material"
    assert fallback["stream_representation"] == "render_only_mesh_current_tank_material"
    assert fallback["global_blend_rule"] == "mean_composition_linear_light_rgb_uncalibrated"
    assert fallback["blend_update_timing"] == "every_emission_event"
    assert fallback["spatial_mixing_color_rendered"] is False
    assert fallback["spatial_layering_color_rendered"] is False
    assert fallback["particle_composition_tracked"] is True
    assert fallback["diffusion_simulated"] is True
    assert fallback["composition_state"] == "internal_unrendered"
    assert plan["semantic_gate"] == {
        "required": True,
        "semantic_label": "liquid",
        "semantic_type": "class",
        "required_targets": ["exact_standard_mesh_proxy", "render_only_stream_mesh"],
        "annotator": "semantic_segmentation",
        "annotator_init_params": {
            "colorize": False,
            "semanticTypes": ["class"],
        },
        "mask_source": "exact_renderer_semantic_class_ids",
        "generated_isosurface_path": physx_three_pour.GENERATED_ISOSURFACE_PATH,
        "required_isosurface_path": physx_three_pour.LIQUID_PROXY_PATH,
        "allowed_stream_paths": [
            f"{FLUIDMIX_NAMESPACE}/RenderStreams/T200",
            f"{FLUIDMIX_NAMESPACE}/RenderStreams/T400",
            f"{FLUIDMIX_NAMESPACE}/RenderStreams/T100",
        ],
        "resolution": [640, 480],
        "aligned_mask_per_kept_rgb": True,
        "nonempty_liquid_pixels_per_mask": True,
        "whole_frame_mask_outcome": "fail",
        "stage_requires_all_render_streams_hidden": True,
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
    assert plan["claims"]["particle_composition_tracked"] is True
    assert plan["claims"]["dye_concentration_simulated"] is False
    assert plan["claims"]["diffusion_simulated"] is True
    assert plan["claims"]["particle_composition_rendered"] is False
    assert plan["claims"]["spatial_mixing_color_rendered"] is False
    assert plan["claims"]["spatial_layering_color_rendered"] is False
    assert plan["claims"]["mixing_physically_validated"] is False
    assert plan["claims"]["navier_stokes_validated"] is False


def test_frozen_plan_tracks_internal_diffusion_but_not_rendered_spatial_colour():
    plan = build_three_pour_plan()

    assert plan["schema_version"] == 15
    assert plan["mixing_model"].endswith("_v5")
    assert plan["fallback_render_model"]["particle_composition_tracked"] is True
    assert plan["fallback_render_model"]["diffusion_simulated"] is True
    assert plan["fallback_render_model"]["composition_state"] == "internal_unrendered"
    assert plan["fallback_render_model"]["spatial_mixing_color_rendered"] is False
    assert plan["fallback_render_model"]["spatial_layering_color_rendered"] is False
    assert plan["claims"]["particle_composition_tracked"] is True
    assert plan["claims"]["diffusion_simulated"] is True
    assert plan["claims"]["particle_composition_rendered"] is False


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
    semantic_hashes = [f"{600 + index:064x}" for index in range(len(frame_steps))]
    stage_phase_by_step = {
        int(pour["stage_capture_step"]): f"{pour['tank_id']}-stage"
        for pour in plan["pours"]
    }

    def phase_and_visible_streams(step: int) -> tuple[str, list[str]]:
        if step in stage_phase_by_step:
            return stage_phase_by_step[step], []
        for pour in plan["pours"]:
            if int(pour["start_step"]) <= step <= int(pour["end_step"]):
                return f"{pour['tank_id']}-pour", [str(pour["tank_id"])]
        return "post-fill", []

    def proxy_geometry(status: str) -> dict[str, object]:
        return {
            "status": status,
            "source_path": physx_three_pour.GENERATED_ISOSURFACE_PATH,
            "proxy_path": physx_three_pour.LIQUID_PROXY_PATH,
            "same_parent_transform": True,
            "point_count": 900,
            "face_count": 1700,
            "corner_count": 5100,
            "points_sha256": "1" * 64,
            "proxy_points_sha256": "1" * 64,
            "face_vertex_counts_sha256": "2" * 64,
            "proxy_face_vertex_counts_sha256": "2" * 64,
            "face_vertex_indices_sha256": "3" * 64,
            "proxy_face_vertex_indices_sha256": "3" * 64,
            "topology_sha256": "4" * 64,
            "proxy_topology_sha256": "4" * 64,
            "local_transform_sha256": "5" * 64,
            "proxy_local_transform_sha256": "5" * 64,
            "mesh_sha256": "6" * 64,
            "proxy_mesh_sha256": "6" * 64,
            "exact_geometry_match": True,
            "material_binding": f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial",
            "semantic_binding": {"type": "class", "labels": ["liquid"]},
        }

    stream_only_prefix_count = 2
    proxy_frame_steps = frame_steps[stream_only_prefix_count:]
    refresh_records = [
        {
            "physics_step": step,
            "phase": phase_and_visible_streams(step)[0],
            "evidence": proxy_geometry("proxy_refreshed_exact"),
            "source_visibility": "invisible",
            "proxy_visibility": "inherited",
            "shared_rgb_semantic_capture": True,
            "artifact_path": f"mesh-proxy/refresh/frame-{index:04d}.json",
            "artifact_sha256": f"{1000 + index:064x}",
        }
        for index, step in enumerate(proxy_frame_steps)
    ]
    refresh_by_step = {row["physics_step"]: row for row in refresh_records}

    return {
        "schema_version": 15,
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
            "velocity_writeback_initial_readback": False,
            "velocity_writeback_enabled_readback": True,
            "velocity_writeback_restored_readback": False,
        },
        "particle_system": {
            "count": 1,
            "exactly_one": True,
            "particle_set_count": 1,
            "ccd_enabled": True,
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
            "runtime_points_path": f"{FLUIDMIX_NAMESPACE}/FluidParticles/Shared",
            "runtime_particle_row_count": 2221,
            "runtime_particle_ids_exact": True,
            "runtime_particle_ids_authored": True,
            "runtime_particle_ids_exact_each_callback": True,
            "runtime_particle_identity_check_count": 2002,
            "runtime_particle_id_mismatch_count": 0,
            "runtime_particle_ids_final_sha256": "9" * 64,
            "runtime_emission_event_count": 332,
            "runtime_property_specs": {
                "points_session_layer_spec": True,
                "points_root_layer_spec": False,
                "velocities_session_layer_spec": True,
                "velocities_root_layer_spec": False,
                "widths_session_layer_spec": True,
                "widths_root_layer_spec": False,
                "ids_session_layer_spec": True,
                "ids_root_layer_spec": False,
            },
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
            "vial_representation": "exact_standard_mesh_proxy_one_global_material",
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
            "global_blend_rule": "mean_composition_linear_light_rgb_uncalibrated",
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
            "model": "timeline_owned_runtime_shared_points_stable_usd_ids_pbd_motion_composition_diffusion_stream_prefix_proxy_global_blend_fallback_v5",
            "component_order": ["t200", "t400", "t100"],
            "global_blend_rule": "mean_composition_linear_light_rgb_uncalibrated",
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
            "particle_composition_tracked": True,
            "diffusion_simulated": True,
            "composition_state": "internal_unrendered",
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
        "composition_evidence": {
            "schema_version": 1,
            "component_order": ["t200", "t400", "t100"],
            "internal_unrendered": True,
            "particle_id_scheme": "zero_based_emission_order_stable_row_identity",
            "diffusion": {
                "coefficient_m2_s": physx_three_pour.EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S,
                "neighbor_radius_m": physx_three_pour.DIFFUSION_NEIGHBOR_RADIUS_M,
                "dt_s": physx_three_pour.DIFFUSION_DT_S,
            },
            "diffusion_update_count": 2002,
            "stages": [
                {
                    "ordinal": state["ordinal"],
                    "tank_id": state["tank_id"],
                    "physics_step": state["physics_step"],
                    "particle_count": state["cumulative_particle_count"],
                    "cumulative_particle_count": state["cumulative_particle_count"],
                    "particle_ids_sha256": "1" * 64,
                    "positions_sha256": "2" * 64,
                    "compositions_sha256": "3" * 64,
                    "component_totals": [float(value) for value in state["cumulative_particle_counts"]],
                    "source_component_totals": [float(value) for value in state["cumulative_particle_counts"]],
                    "component_fractions": state["component_fractions"],
                    "row_sum_error": 0.0,
                    "conservation_error": 0.0,
                    "nonnegative_passed": True,
                    "conservation_passed": True,
                    "particle_row_order_preserved": True,
                    "diffusion_update_count": state["physics_step"],
                    "diffusion_neighbor_count_total": 1,
                    "diffusion_max_degree": 1,
                    "diffusion_pair_weight_max": physx_three_pour.DIFFUSION_STEP_FACTOR if hasattr(physx_three_pour, "DIFFUSION_STEP_FACTOR") else 0.004,
                    "vertical_composition_summary": [{"particle_count": state["cumulative_particle_count"], "z_min_m": 0.0, "z_max_m": 1.0, "composition_mean": state["component_fractions"]}],
                    "palette_resolution": 4,
                    "palette_prototype_count": 15,
                    "palette_colors_sha256": "4" * 64,
                    "palette_indices_sha256": "5" * 64,
                    "artifact_path": f"composition-evidence/stage-{state['ordinal']}.json",
                    "artifact_sha256": ("8" if state["ordinal"] == 1 else "9" if state["ordinal"] == 2 else "a") * 64,
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
        "mesh_proxy": {
            "source_path": physx_three_pour.GENERATED_ISOSURFACE_PATH,
            "proxy_path": physx_three_pour.LIQUID_PROXY_PATH,
            "material_path": f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial",
            "created": True,
            "creation_step": proxy_frame_steps[0] - 1,
            "creation_evidence": proxy_geometry(
                "proxy_authored_pending_rgb_parity"
            ),
            "creation_artifact_path": "mesh-proxy/creation-step-00001.json",
            "creation_artifact_sha256": "7" * 64,
            "stage_surface_sync_records": [],
            "stage_isosurface_lifecycle_refresh_records": [],
            "rgb_parity": {
                "status": "rgb_parity_pass_pending_shared_capture",
                "hide_source_allowed": True,
                "source_visibility_change_executed": False,
                "thresholds": dict(
                    physx_three_pour.PROXY_RGB_PARITY_THRESHOLDS
                ),
                "same_physics_step": True,
                "same_renderer_settings": True,
                "parity_capture_isolated": True,
                "source_restored_visible": True,
                "physics_step": proxy_frame_steps[0],
                "phase": phase_and_visible_streams(proxy_frame_steps[0])[0],
                "parity_geometry_mesh_sha256": "6" * 64,
                "parity_geometry_points_sha256": "1" * 64,
                "original_capture_visibility": ["inherited", "invisible"],
                "proxy_capture_visibility": ["invisible", "inherited"],
                "restored_visibility": ["inherited", "invisible"],
                "original_rgb_sha256": "8" * 64,
                "proxy_rgb_sha256": "9" * 64,
                "absdiff_sha256": "a" * 64,
                "metrics": {
                    "mean_abs_rgb": 0.5,
                    "p95_abs_rgb": 2.0,
                    "fraction_abs_rgb_gt_16": 0.0,
                },
            },
            "refresh_records": refresh_records,
            "source_hidden_at_all_vial_proxy_captures": True,
            "proxy_visible_at_all_vial_proxy_captures": True,
            "one_shared_rgb_semantic_capture_per_vial_proxy_frame": True,
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
            "video_codec": "mp4v",
            "video_container": "mp4",
            "video_writer_fallback_used": False,
            "video_decodable": True,
            "video_decoded_frame_count": len(frame_steps),
            "stage_frames_are_video_subsets": True,
            "stream_only_frame_count": stream_only_prefix_count,
            "vial_proxy_frame_count": len(proxy_frame_steps),
            "vial_proxy_transition_step": proxy_frame_steps[0],
            "liquid_representation_monotonic": True,
        },
        "semantics": {
            "semantic_label": "liquid",
            "semantic_type": "class",
            "mask_source": "exact_renderer_semantic_class_ids",
            "required_isosurface_path": physx_three_pour.LIQUID_PROXY_PATH,
            "allowed_stream_paths": [
                f"{FLUIDMIX_NAMESPACE}/RenderStreams/T200",
                f"{FLUIDMIX_NAMESPACE}/RenderStreams/T400",
                f"{FLUIDMIX_NAMESPACE}/RenderStreams/T100",
            ],
            "material_binding_authored": True,
            "material_binding": {
                "path": f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial",
                "schema": "SemanticsLabelsAPI:class",
                "labels": ["liquid"],
                "direct_binding_verified": True,
            },
            "isosurface_semantic_bound": True,
            "isosurface_direct_binding_authored": True,
            "isosurface_direct_binding_step": proxy_frame_steps[0] - 1,
            "isosurface_direct_binding": {
                "path": physx_three_pour.LIQUID_PROXY_PATH,
                "schema": "SemanticsLabelsAPI:class",
                "labels": ["liquid"],
                "direct_binding_verified": True,
            },
            "stream_semantic_bound": True,
            "annotator_attached_after_stream_labels": True,
            "annotator": "semantic_segmentation",
            "annotator_init_params": {
                "colorize": False,
                "semanticTypes": ["class"],
            },
            "resolution": [640, 480],
            "pairs": [
                {
                    "physics_step": step,
                    "phase": phase_and_visible_streams(step)[0],
                    "visible_stream_tanks": phase_and_visible_streams(step)[1],
                    "liquid_representation": (
                        "stream_only"
                        if index < stream_only_prefix_count
                        else "vial_proxy"
                    ),
                    "rgb_path": f"frames/frame-{index:04d}.png",
                    "rgb_sha256": frame_hashes[index],
                    "mask_path": f"masks/mask-{index:04d}.png",
                    "mask_sha256": mask_hashes[index],
                    "semantic_path": f"semantic/frame-{index:04d}.npy",
                    "semantic_sha256": semantic_hashes[index],
                    "semantic_unique_ids": [0, 5, 7],
                    "liquid_semantic_ids": [5],
                    "semantic_id_to_class_labels": {
                        "0": ["BACKGROUND"],
                        "5": ["liquid"],
                        "7": ["vial"],
                    },
                    "liquid_pixel_counts_by_id": {"5": 100 + index},
                    "mask_readable": True,
                    "mask_resolution": [640, 480],
                    "liquid_pixel_count": 100 + index,
                    "mesh_proxy_refresh_artifact_path": (
                        refresh_by_step[step]["artifact_path"]
                        if step in refresh_by_step
                        else None
                    ),
                    "mesh_proxy_refresh_artifact_sha256": (
                        refresh_by_step[step]["artifact_sha256"]
                        if step in refresh_by_step
                        else None
                    ),
                    "source_isosurface_visibility": "invisible",
                    "proxy_visibility": (
                        "absent"
                        if index < stream_only_prefix_count
                        else "inherited"
                    ),
                    "shared_rgb_semantic_capture": True,
                }
                for index, step in enumerate(frame_steps)
            ],
            "stage_mask_sha256": [mask_hashes[index] for index in stage_frame_indices],
            "frame_semantic_paths": [
                f"semantic/frame-{index:04d}.npy"
                for index in range(len(frame_steps))
            ],
            "frame_semantic_sha256": semantic_hashes,
            "semantic_gate_passed": True,
            "training_eligible": False,
            "ineligibility_reason": "qualification_run_not_case_dataset",
            "capture_volumes_setting_path": "/syntheticdata/sensors/captureVolumes",
            "capture_volumes_initial_readback": False,
            "capture_volumes_enabled_readback": True,
            "capture_volumes_restored_readback": False,
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
            "simulation_owner": "timeline",
            "callbacks_expected": 2002,
            "callbacks_observed": 2002,
            "callback_indices": list(range(1, 2003)),
            "callback_dts_s": [1 / 60 for _ in range(2002)],
            "callback_times_s": [index / 60 for index in range(1, 2003)],
            "callback_app_update_indices": list(range(1, 2003)),
            "one_callback_per_app_update": True,
            "emissions_authored_on_main_thread": True,
            "emission_boundary_verified": True,
            "emission_boundaries": [
                {
                    "physics_step": event["physics_step"],
                    "authored_after_callback_count": event["physics_step"] - 1,
                    "scheduled_before_callback_index": event["physics_step"],
                    "authored_on_main_thread": True,
                }
                for pour in plan["pours"]
                for event in pour["emission_schedule"]
            ],
            "points_session_layer_only": True,
            "velocities_session_layer_only": True,
            "widths_session_layer_only": True,
            "ids_session_layer_only": True,
            "capture_callbacks_advanced": 0,
            "capture_timeline_time_advanced": False,
            "capture_playback_preserved": True,
            "capture_guard_records": [
                {
                    "callbacks_before": 1,
                    "callbacks_after": 1,
                    "timeline_time_before_s": 0.0,
                    "timeline_time_after_s": 0.0,
                    "timeline_playing_before": False,
                    "timeline_playing_after": False,
                    "physics_step": 0,
                    "phase": "twin-reference-time-0001",
                    "attempt": 1,
                }
            ],
            "direct_physx_api_used": False,
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


def test_validator_requires_ccd_and_bounded_composition_artifacts():
    receipt = _passing_receipt()
    receipt["particle_system"]["ccd_enabled"] = False
    assert "particle_ccd_not_observed" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["composition_evidence"]["stages"][1]["component_totals"] = [0.0, 0.0, 0.0]
    assert "composition_evidence_mismatch" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["semantics"]["isosurface_direct_binding_authored"] = False
    assert "liquid_semantic_binding_missing" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["semantics"]["material_binding_authored"] = False
    assert "liquid_semantic_binding_missing" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    stage_index = receipt["render"]["stage_frame_indices"][0]
    receipt["semantics"]["pairs"][stage_index]["visible_stream_tanks"] = ["t200"]
    assert "liquid_semantic_mask_invalid" in validate_three_pour_receipt(receipt)


def test_validator_requires_a18_timeline_ownership_and_callback_receipt():
    receipt = _passing_receipt()
    receipt["timing"]["callbacks_observed"] = 2001

    assert "timeline_callback_contract" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["timing"]["capture_callbacks_advanced"] = 1

    assert "timeline_capture_contract" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["timing"]["callback_dts_s"][99] = 0.1

    assert "timeline_callback_contract" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["timing"]["capture_guard_records"][0]["callbacks_after"] = 2

    assert "timeline_capture_contract" in validate_three_pour_receipt(receipt)


def test_validator_requires_observed_usd_particle_identity_on_every_callback():
    receipt = _passing_receipt()
    receipt["particle_system"]["runtime_particle_ids_exact_each_callback"] = False

    assert "particle_identity_contract" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["particle_system"]["runtime_particle_ids_final_sha256"] = "z" * 64

    assert "particle_identity_contract" in validate_three_pour_receipt(receipt)


def test_validator_requires_velocity_writeback_and_restoration():
    receipt = _passing_receipt()
    receipt["gpu_simulation"]["velocity_writeback_enabled_readback"] = False

    assert "velocity_writeback_contract" in validate_three_pour_receipt(receipt)

    receipt = _passing_receipt()
    receipt["gpu_simulation"]["velocity_writeback_restored_readback"] = True

    assert "velocity_writeback_contract" in validate_three_pour_receipt(receipt)


def test_validator_fails_closed_on_noninteger_stage_indices():
    receipt = _passing_receipt()
    receipt["render"]["stage_frame_indices"][0] = "not-an-index"

    failures = validate_three_pour_receipt(receipt)

    assert "primary_capture_contract" in failures
    assert "liquid_representation_contract" in failures


def test_validator_rejects_schema_ten_direct_route_receipt():
    receipt = _passing_receipt()
    receipt["schema_version"] = 10

    assert "receipt_identity" in validate_three_pour_receipt(receipt)


def test_validator_accepts_bounded_composition_roundoff():
    receipt = _passing_receipt()
    receipt["composition_evidence"]["stages"][2]["row_sum_error"] = 2.22e-16

    assert validate_three_pour_receipt(receipt) == ()


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
            lambda r: r["mesh_proxy"]["creation_evidence"].update(
                proxy_points_sha256="0" * 64
            ),
            "exact_mesh_proxy_contract",
        ),
        (
            lambda r: r["mesh_proxy"]["rgb_parity"].update(
                hide_source_allowed=False
            ),
            "exact_mesh_proxy_contract",
        ),
        (
            lambda r: r["mesh_proxy"]["refresh_records"][0].update(
                source_visibility="inherited"
            ),
            "exact_mesh_proxy_contract",
        ),
        (
            lambda r: r["mesh_proxy"].pop("stage_surface_sync_records"),
            "exact_mesh_proxy_contract",
        ),
        (
            lambda r: r["mesh_proxy"].pop(
                "stage_isosurface_lifecycle_refresh_records"
            ),
            "exact_mesh_proxy_contract",
        ),
        (
            lambda r: r["semantics"]["pairs"][3].update(
                liquid_representation="stream_only",
                mesh_proxy_refresh_artifact_path=None,
                mesh_proxy_refresh_artifact_sha256=None,
                proxy_visibility="absent",
            ),
            "liquid_representation_contract",
        ),
        (
            lambda r: r["semantics"]["pairs"][0].update(
                visible_stream_tanks=["t400"]
            ),
            "liquid_representation_contract",
        ),
        (
            lambda r: r["semantics"]["pairs"][
                r["render"]["stage_frame_indices"][0]
            ].update(
                liquid_representation="stream_only",
                mesh_proxy_refresh_artifact_path=None,
                mesh_proxy_refresh_artifact_sha256=None,
                proxy_visibility="absent",
            ),
            "liquid_representation_contract",
        ),
        (
            lambda r: r["semantics"].update(isosurface_semantic_bound=False),
            "liquid_semantic_binding_missing",
        ),
        (
            lambda r: r["semantics"].update(capture_volumes_enabled_readback=False),
            "liquid_volume_capture_setting_missing",
        ),
        (
            lambda r: r["semantics"].update(capture_volumes_restored_readback=True),
            "liquid_volume_capture_setting_missing",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(liquid_pixel_count=0),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(
                liquid_pixel_count=640 * 480,
                liquid_pixel_counts_by_id={"5": 640 * 480},
            ),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(mask_readable=False),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(semantic_sha256=None),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["frame_semantic_sha256"].__setitem__(
                1, "0" * 64
            ),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(
                semantic_id_to_class_labels={
                    "0": ["BACKGROUND"],
                    "5": ["vial"],
                    "7": ["vial"],
                }
            ),
            "liquid_semantic_mask_invalid",
        ),
        (
            lambda r: r["semantics"]["pairs"][1].update(
                semantic_unique_ids=[0, 5, 5, 7]
            ),
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
    assert json.loads((output / "plan.json").read_text(encoding="utf-8")) == build_paint_ball_three_pour_plan()
    repeated = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "output path already exists" in repeated.stderr


def test_runner_plan_only_records_water_pbd_qualification_treatment(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    output = tmp_path / "plan"
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(output),
            "--stage",
            str(stage),
            "--qualification-mode",
            "--qualification-water-pbd-preset",
            "--plan-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    treatment = plan.pop("pbd_material_treatment")
    assert plan == build_paint_ball_three_pour_plan()
    assert treatment["name"] == "nvidia_add_pbd_material_water"
    assert treatment["source_sha256"] == (
        "85da0b0a39931eff3e54cea5ca8987c07d1c903557d87cb609b205ab85f9b3df"
    )


def test_runner_wires_point_instancer_and_paired_same_state_captures():
    source = (ROOT / "scripts" / "physx_three_pour_sequence.py").read_text(
        encoding="utf-8"
    )

    assert "add_physx_particleset_pointinstancer" in source
    assert "GetSimulationPointsAttr" in source
    assert "GetOrientationsAttr" in source
    assert "GetScalesAttr" in source
    assert "copy_exact_mesh_proxy(" not in source[source.index("def run_sequence"):]
    assert "refresh_exact_mesh_proxy(" not in source[source.index("def run_sequence"):]
    assert "acquire_paired_frame" in source
    assert '"shared_rgb_semantic_capture": False' in source
    assert '"paired_same_physics_state_rgb_semantic_capture": True' in source
    assert "_paired_capture_state_evidence" in source
    assert "hand polygon" not in source.lower()


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
                "pause_timeline": False,
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


def test_empty_legacy_receipt_retains_the_schema15_step_metadata(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", "unused", "--stage", str(stage)]
    )

    receipt = sequence._empty_receipt(args)

    assert receipt["schema_version"] == 15
    assert receipt["timing"]["pour_duration_steps"] == [706, 605, 673]
    assert receipt["timing"]["pour_end_steps"] == [706, 1311, 1984]
    assert receipt["timing"]["fill_end_step"] == 1984
    assert receipt["timing"]["primary_final_capture_step"] == 2002
    assert receipt["timing"]["callbacks_expected"] == 2002


@pytest.mark.parametrize(
    ("qualification_mode", "expected_callbacks"),
    ((False, 4004), (True, 1412)),
)
def test_empty_paint_ball_receipt_uses_120hz_runtime_plan_timing(
    tmp_path, qualification_mode, expected_callbacks
):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    argv = ["--out", "unused", "--stage", str(stage)]
    if qualification_mode:
        argv.append("--qualification-mode")
    args = sequence.build_parser().parse_args(argv)

    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["timing"]["physics_hz"] == 120
    assert receipt["timing"]["pour_duration_steps"] == [1412, 1210, 1346]
    assert receipt["timing"]["pour_end_steps"] == [1412, 2622, 3968]
    assert receipt["timing"]["fill_end_step"] == 3968
    assert receipt["timing"]["primary_final_capture_step"] == 4004
    assert receipt["timing"]["callbacks_expected"] == expected_callbacks
    assert receipt["qualification"]["target_step"] == (
        1412 if qualification_mode else None
    )


def test_water_pbd_preset_qualification_receipt_records_frozen_treatment(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        [
            "--out",
            "unused",
            "--stage",
            str(stage),
            "--qualification-mode",
            "--qualification-water-pbd-preset",
        ]
    )

    sequence._validate_args(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["plan"]["pbd_material_treatment"] == {
        "name": "nvidia_add_pbd_material_water",
        "qualification_only": False,
        "supported_modes": ["full_run", "first_stage_qualification"],
        "api": "omni.physx.scripts.particleUtils.AddPBDMaterialWater",
        "source_path": "omni/physx/scripts/particleUtils.py:191-197",
        "source_sha256": (
            "85da0b0a39931eff3e54cea5ca8987c07d1c903557d87cb609b205ab85f9b3df"
        ),
        "expected_meters_per_unit": 1.0,
        "expected_authored_values": {
            "cohesion": 0.01,
            "damping": 0.0,
            "friction": 0.1,
            "surface_tension": 0.0074,
            "viscosity": 1.7e-6,
            "vorticity_confinement": 0.0,
        },
        "density_kg_m3_authored_after_preset": 1000.0,
    }
    assert receipt["particle_system"]["pbd_material_treatment"] == {
        "requested": True,
        "applied": False,
        "requested_via": ["qualification_water_pbd_preset"],
        "plan": receipt["plan"]["pbd_material_treatment"],
    }


def test_water_pbd_preset_is_applied_before_density_without_default_drift():
    events = []
    material_prim = object()

    class Attribute:
        def __init__(self, name):
            self.name = name

        def Set(self, value):
            events.append((self.name, value))

    class Pbd:
        def CreateDensityAttr(self):
            return Attribute("density")

        def CreateFrictionAttr(self):
            return Attribute("friction")

    class ParticleUtils:
        @staticmethod
        def AddPBDMaterialWater(value):
            events.append(("water_preset", value))

    sequence._author_pbd_material_treatment(
        pbd=Pbd(),
        material_prim=material_prim,
        particle_utils=ParticleUtils,
        use_water_preset=True,
    )
    assert events == [("water_preset", material_prim), ("density", 1000.0)]

    events.clear()
    sequence._author_pbd_material_treatment(
        pbd=Pbd(),
        material_prim=material_prim,
        particle_utils=ParticleUtils,
        use_water_preset=False,
    )
    assert events == [("density", 1000.0), ("friction", 0.1)]


def test_water_pbd_preset_readback_must_match_all_authored_values():
    readback = {
        "density_kg_m3": 1000.0,
        "friction": 0.1,
        "cohesion": 0.01,
        "damping": 0.0,
        "surface_tension": 0.0074,
        "viscosity": 1.7e-6,
        "vorticity_confinement": 0.0,
        "density_authored": True,
        "friction_authored": True,
        "cohesion_authored": True,
        "damping_authored": True,
        "surface_tension_authored": True,
        "viscosity_authored": True,
        "vorticity_confinement_authored": True,
    }

    assert sequence._water_pbd_preset_readback_matches_plan(
        readback,
        meters_per_unit=1.0,
    )
    readback["surface_tension"] = 0.0
    assert not sequence._water_pbd_preset_readback_matches_plan(
        readback,
        meters_per_unit=1.0,
    )


def test_water_pbd_preset_supports_full_run_and_keeps_qualification_alias(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    full_args = sequence.build_parser().parse_args(
        [
            "--out",
            "unused",
            "--stage",
            str(stage),
            "--water-pbd-preset",
        ]
    )
    alias_args = sequence.build_parser().parse_args(
        [
            "--out",
            "unused",
            "--stage",
            str(stage),
            "--qualification-mode",
            "--qualification-water-pbd-preset",
        ]
    )

    sequence._validate_args(full_args)
    sequence._validate_args(alias_args)
    full_receipt = sequence._empty_paint_ball_receipt(full_args)
    alias_receipt = sequence._empty_paint_ball_receipt(alias_args)

    assert full_receipt["plan"]["pbd_material_treatment"]["qualification_only"] is False
    assert full_receipt["particle_system"]["pbd_material_treatment"]["requested_via"] == [
        "water_pbd_preset"
    ]
    assert alias_receipt["particle_system"]["pbd_material_treatment"]["requested_via"] == [
        "qualification_water_pbd_preset"
    ]


def test_nondefault_wall_limit_is_recorded_in_arguments_plan_and_receipt(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        [
            "--out",
            "unused",
            "--stage",
            str(stage),
            "--wall-limit-seconds",
            "10800",
        ]
    )

    sequence._validate_args(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["arguments"]["wall_limit_seconds"] == 10800
    assert receipt["plan"]["execution_limits"] == {"wall_limit_seconds": 10800}
    assert receipt["timing"]["wall_limit_seconds"] == 10800


def test_default_paint_ball_receipt_has_no_a29_treatment_metadata(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", "unused", "--stage", str(stage), "--qualification-mode"]
    )

    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["plan"] == build_paint_ball_three_pour_plan()
    assert "pbd_material_treatment" not in receipt["plan"]
    assert "pbd_material_treatment" not in receipt["particle_system"]


def test_empty_paint_ball_receipt_freezes_two_pass_transparent_vial_and_bottom_roi(
    tmp_path,
):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", "unused", "--stage", str(stage), "--qualification-mode"]
    )

    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["semantics"]["transparent_annotation_override"] == {
        "prim_path": "/FillStation/FillStation_LocalRoot/Fill_vial_active/Mesh_998",
        "parent_path": "/FillStation/FillStation_LocalRoot/Fill_vial_active",
        "attribute_name": "primvars:doNotCastShadows",
        "semantic_pass_value": True,
        "rgb_and_restore_value": False,
        "capture_mode": "paired_same_physics_state_rgb_then_semantic",
        "documentation_url": "https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotations_with_transparency.html",
        "source": None,
        "initial_rgb_state": None,
        "transitions": [],
        "property_stack_strongest_is_session_layer": False,
        "authored_after_untouched_reference_capture": False,
        "source_layer_unchanged_expected": True,
    }
    assert receipt["semantics"]["paired_capture_contract"] == {
        "shared_single_render_pass": False,
        "paired_same_physics_state": True,
        "kept_rgb_uses_source_equivalent_glass": True,
        "semantic_pass_uses_transparent_annotation_flag": True,
        "flag_restored_false_after_semantic_pass": True,
        "state_fields_required_unchanged": [
            "callback_count",
            "timeline_time_s",
            "camera_path",
            "positions",
            "simulation_points",
            "velocities",
            "ids",
            "proto_indices",
        ],
        "state_records": [],
    }
    assert receipt["qualification"]["first_vial_bottom_roi"] == {
        "xyxy_exclusive": [280, 300, 340, 430],
        "resolution": [640, 480],
        "minimum_liquid_pixels": 16,
        "definition": "frozen_fillcam_first_vial_interior_bottom",
    }


def test_paired_capture_state_evidence_accepts_exact_state_and_names_drift():
    before = {
        "callback_count": 1412,
        "timeline_time_s": 1412 / 120,
        "camera_path": CAMERA_PATH,
        "positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "simulation_points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "velocities": [[0.0, 0.0, -0.1], [0.0, 0.0, -0.2]],
        "ids": [20, 21],
        "proto_indices": [0, 2],
    }

    passed = sequence._paired_capture_state_evidence(before, deepcopy(before))

    assert passed["passed"] is True
    assert passed["failed_predicates"] == []
    assert passed["row_count"] == 2
    assert passed["callback_count_before"] == passed["callback_count_after"] == 1412
    assert passed["camera_path_before"] == passed["camera_path_after"] == CAMERA_PATH
    assert passed["positions_sha256_before"] == passed["positions_sha256_after"]
    assert passed["simulation_points_sha256_before"] == passed["simulation_points_sha256_after"]
    assert passed["velocities_sha256_before"] == passed["velocities_sha256_after"]
    assert passed["ids_sha256_before"] == passed["ids_sha256_after"]
    assert passed["proto_indices_sha256_before"] == passed["proto_indices_sha256_after"]

    drifted = deepcopy(before)
    drifted["positions"][1][2] = 6.25
    failed = sequence._paired_capture_state_evidence(before, drifted)

    assert failed["passed"] is False
    assert failed["failed_predicates"] == ["positions"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("callback_count", 1413),
        ("timeline_time_s", 12.0),
        ("camera_path", "/World/OtherCamera"),
        ("simulation_points", [[1.0, 2.0, 3.0]]),
        ("velocities", [[0.0, 0.0, -0.1]]),
        ("ids", [20, 22]),
        ("proto_indices", [0, 1]),
    ],
)
def test_paired_capture_state_evidence_fails_each_required_drift(field, value):
    before = {
        "callback_count": 1412,
        "timeline_time_s": 1412 / 120,
        "camera_path": CAMERA_PATH,
        "positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "simulation_points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "velocities": [[0.0, 0.0, -0.1], [0.0, 0.0, -0.2]],
        "ids": [20, 21],
        "proto_indices": [0, 2],
    }
    after = deepcopy(before)
    after[field] = value

    evidence = sequence._paired_capture_state_evidence(before, after)

    assert field in evidence["failed_predicates"]
    assert evidence["passed"] is False


@pytest.mark.parametrize(
    ("mask_box_xyxy", "expected_roi_pixels", "expected_gate"),
    [
        ((309, 81, 315, 90), 0, False),
        ((300, 380, 303, 385), 15, False),
        ((300, 380, 304, 384), 16, True),
    ],
)
def test_qualification_semantic_gate_requires_liquid_in_first_vial_bottom_roi(
    mask_box_xyxy, expected_roi_pixels, expected_gate
):
    mask = np.zeros((480, 640), dtype=np.uint8)
    x0, y0, x1, y1 = mask_box_xyxy
    mask[y0:y1, x0:x1] = 255
    roi = sequence._first_vial_bottom_roi_evidence(mask)
    assert roi == {
        "roi_xyxy_exclusive": [280, 300, 340, 430],
        "liquid_pixel_count_in_roi": expected_roi_pixels,
        "minimum_liquid_pixels": 16,
        "passed": expected_gate,
    }
    records = [
        {
            "physics_step": 706,
            "liquid_pixel_count": 10,
            "liquid_pixel_count_in_first_vial_bottom_roi": 0,
            "visible_stream_tanks": ["t200"],
            "paired_same_physics_state_rgb_semantic_capture": True,
            "paired_capture_state_passed": True,
        },
        {
            "physics_step": 1412,
            "liquid_pixel_count": max(expected_roi_pixels, 1),
            "liquid_pixel_count_in_first_vial_bottom_roi": expected_roi_pixels,
            "visible_stream_tanks": [],
            "paired_same_physics_state_rgb_semantic_capture": True,
            "paired_capture_state_passed": True,
        },
    ]

    assert sequence._qualification_semantic_gate(
        records=records,
        target_step=1412,
        capture_steps=(706, 1412),
        frame_count=2,
        frame_pixel_count=640 * 480,
        point_instancer_semantic_bound=True,
        stream_semantic_bound=True,
    ) is expected_gate

    records[0]["paired_capture_state_passed"] = False
    assert sequence._qualification_semantic_gate(
        records=records,
        target_step=1412,
        capture_steps=(706, 1412),
        frame_count=2,
        frame_pixel_count=640 * 480,
        point_instancer_semantic_bound=True,
        stream_semantic_bound=True,
    ) is False


def test_first_vial_bottom_roi_rejects_non_fillcam_mask_shape():
    with pytest.raises(ValueError, match="liquid mask must have shape"):
        sequence._first_vial_bottom_roi_evidence(
            np.zeros((479, 640), dtype=np.uint8)
        )


@pytest.mark.parametrize("qualification_mode", (False, True))
def test_empty_paint_ball_receipt_derives_timing_from_runtime_plan(
    tmp_path, monkeypatch, qualification_mode
):
    runtime_plan = deepcopy(build_paint_ball_three_pour_plan())
    duration_steps = [1402, 1203, 1304]
    end_steps = [1402, 2605, 3909]
    for pour, duration_step, end_step in zip(
        runtime_plan["pours"], duration_steps, end_steps
    ):
        pour["duration_steps"] = duration_step
        pour["end_step"] = end_step
    runtime_plan["timing"]["fill_end_step"] = end_steps[-1]
    runtime_plan["timing"]["primary_final_capture_step"] = 3945
    runtime_plan["paint_ball_emitter"]["qualification_first_stage_step"] = 1402
    monkeypatch.setattr(
        sequence,
        "build_paint_ball_three_pour_plan",
        lambda: runtime_plan,
    )
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    argv = ["--out", "unused", "--stage", str(stage)]
    if qualification_mode:
        argv.append("--qualification-mode")

    receipt = sequence._empty_paint_ball_receipt(
        sequence.build_parser().parse_args(argv)
    )

    assert receipt["timing"]["pour_duration_steps"] == duration_steps
    assert receipt["timing"]["pour_end_steps"] == end_steps
    assert receipt["timing"]["fill_end_step"] == end_steps[-1]
    assert receipt["timing"]["primary_final_capture_step"] == 3945
    assert receipt["timing"]["callbacks_expected"] == (
        1402 if qualification_mode else 3945
    )
    assert receipt["qualification"]["target_step"] == (
        1402 if qualification_mode else None
    )


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

"""CPU contracts for the Paint Ball Emitter PointInstancer route."""

from __future__ import annotations

from copy import deepcopy
import numpy as np
import pytest
from pathlib import Path

from fluidmix.physx_three_pour import build_three_pour_plan

from fluidmix.paint_ball_emitter import (
    MAX_VELOCITY_M_S,
    PAINT_BALL_RECEIPT_SCHEMA_VERSION,
    append_point_instancer_state,
    attach_visible_volume_evidence,
    build_capture_record,
    build_prototype_specs,
    containment_evidence,
    validate_paint_ball_receipt,
)
import fluidmix.paint_ball_emitter as paint_ball_emitter


def test_emitter_append_extends_all_point_instancer_arrays_and_stable_ids():
    previous = {
        "positions": [[1.0, 2.0, 3.0]],
        "velocities": [[0.0, 0.0, -0.1]],
        "orientations": [[1.0, 0.0, 0.0, 0.0]],
        "scales": [[1.0, 1.0, 1.0]],
        "ids": [4],
        "proto_indices": [2],
    }
    updated = append_point_instancer_state(
        previous,
        positions=[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
        velocities=[[0.0, 0.0, -0.1], [0.0, 0.0, -0.1]],
        prototype_index=1,
        particle_ids=[5, 6],
    )
    assert updated["ids"] == [4, 5, 6]
    assert updated["proto_indices"] == [2, 1, 1]
    assert len(updated["positions"]) == len(updated["velocities"]) == 3
    assert len(updated["orientations"]) == len(updated["scales"]) == 3


def test_emitter_append_rejects_id_reordering_or_length_drift():
    with pytest.raises(ValueError, match="stable|length"):
        append_point_instancer_state(
            {"positions": [[0, 0, 0]], "velocities": [[0, 0, 0]], "orientations": [[1, 0, 0, 0]], "scales": [[1, 1, 1]], "ids": [1], "proto_indices": [0]},
            positions=[[1, 1, 1]], velocities=[[0, 0, 0]], prototype_index=0, particle_ids=[1],
        )


def test_prototypes_are_one_sphere_per_fixed_palette_colour():
    specs = build_prototype_specs({"t200": (0.95, 0.38, 0.08), "t400": (0.18, 0.55, 0.75)})
    assert specs == (
        {"prototype_index": 0, "tank_id": "t200", "shape": "Sphere", "semantic_label": "liquid", "color": [0.95, 0.38, 0.08]},
        {"prototype_index": 1, "tank_id": "t400", "shape": "Sphere", "semantic_label": "liquid", "color": [0.18, 0.55, 0.75]},
    )


def test_palette_material_path_uses_a_valid_named_usd_prim_leaf():
    path = paint_ball_emitter.build_palette_material_path(
        "/FillStation/FluidmixF2", 0
    )

    assert path == "/FillStation/FluidmixF2/PaletteMaterials/Palette_00"
    assert path.rsplit("/", 1)[-1][0].isalpha()


def test_containment_record_counts_outside_particles_and_max_bounds():
    positions = np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    evidence = containment_evidence(
        positions,
        lower_bounds=(-1.0, -1.0, 0.0),
        upper_bounds=(1.0, 1.0, 2.0),
    )
    assert evidence["active_particle_count"] == 3
    assert evidence["inside_vial_interior_count"] == 1
    assert evidence["outside_vial_interior_count"] == 2
    assert evidence["max_abs_bounds_m"] == [2.0, 0.0, 3.0]
    assert evidence["leakage_detected"] is True


def test_containment_uses_cylindrical_vial_interior_not_aabb_corners():
    evidence = containment_evidence(
        [[0.9, 0.9, 1.0]],
        lower_bounds=(-1.0, -1.0, 0.0),
        upper_bounds=(1.0, 1.0, 2.0),
        cylindrical_center_xy=(0.0, 0.0),
        inner_radius_m=1.0,
    )
    assert evidence["inside_vial_interior_count"] == 0
    assert evidence["leaked_particle_count"] == 1


def test_containment_uses_exact_dodecagon_and_reports_state_height_inputs():
    angle = np.pi / 12.0
    corner_inside_dodecagon = [
        1.02 * np.cos(angle),
        1.02 * np.sin(angle),
        1.0,
    ]
    evidence = containment_evidence(
        [corner_inside_dodecagon, [0.0, 0.0, 1.2]],
        lower_bounds=(-2.0, -2.0, 0.0),
        upper_bounds=(2.0, 2.0, 2.0),
        cylindrical_center_xy=(0.0, 0.0),
        inner_radius_m=1.0,
        polygon_center_xy=(0.0, 0.0),
        polygon_apothem_m=1.0,
        polygon_wall_count=12,
    )

    assert evidence["inside_vial_interior_count"] == 2
    assert evidence["inside_conservative_circle_count"] == 1
    assert evidence["exact_polygon_containment"] is True
    assert evidence["finite_state"] is True
    assert evidence["contained_z_min_m"] == pytest.approx(1.0)
    assert evidence["contained_z_q05_m"] == pytest.approx(1.01)
    assert evidence["contained_z_q95_m"] == pytest.approx(1.19)
    assert evidence["contained_z_max_m"] == pytest.approx(1.2)


def test_capture_and_receipt_explicitly_record_limits_and_max_velocity():
    assert PAINT_BALL_RECEIPT_SCHEMA_VERSION == 18
    containment = containment_evidence(
        [[0.0, 0.0, 1.0]],
        lower_bounds=(-1.0, -1.0, 0.0),
        upper_bounds=(1.0, 1.0, 2.0),
        cylindrical_center_xy=(0.0, 0.0),
        inner_radius_m=1.0,
        polygon_center_xy=(0.0, 0.0),
        polygon_apothem_m=1.0,
        polygon_wall_count=12,
    )
    record = build_capture_record(
        physics_step=120,
        phase="t200-stage",
        containment=containment,
        prototype_count=3,
        instancer_path="/FillStation/FluidmixF2/Particles",
        emitted_particle_count=1,
        programmed_cumulative_volume_ml=3000.0,
        expected_cumulative_particle_count=1,
        cross_section_area_m2=1.0,
        height_tolerance_m=0.006,
        render_radius_m=0.0015,
        point_instancer_scales=[[1.0, 1.0, 1.0]],
        smoothed_render_containment=containment,
    )
    center_xy_m = (5.2324003281, -1.9811999798)
    bottom_z_m = 0.9762065645
    mouth_z_m = 1.1762601692
    circumradius_m = 0.0132029216
    roi = paint_ball_emitter.projected_vial_geometry_roi_mask(
        (480, 640),
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        mouth_z_m=mouth_z_m,
        circumradius_m=circumradius_m,
    )
    surface_row = int(
        np.ceil(
            paint_ball_emitter.projected_top_dodecagon_min_y_px(
                0.003,
                center_xy_m=center_xy_m,
                bottom_z_m=bottom_z_m,
                circumradius_m=circumradius_m,
            )
        )
    )
    liquid_mask = np.zeros_like(roi)
    liquid_mask[surface_row:] = roi[surface_row:]
    record = attach_visible_volume_evidence(
        record,
        liquid_mask,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        mouth_z_m=mouth_z_m,
        circumradius_m=circumradius_m,
    )
    assert record["physics_step"] == 120
    assert record["prototype_count"] == 3
    assert record["max_velocity_m_s"] == MAX_VELOCITY_M_S
    assert record["render_limitations"]["continuous_surface"] is False
    receipt = {
        "schema_version": PAINT_BALL_RECEIPT_SCHEMA_VERSION,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "pass",
        "render_path": "point_instancer_sphere_emitter",
        "particle_system": {
            "max_velocity_m_s": MAX_VELOCITY_M_S,
            "smoothing": True,
            "smoothing_enabled": True,
            "smoothing_strength": 1.0,
            "anisotropy": True,
            "anisotropy_enabled": True,
            "anisotropy_scale": 1.0,
            "ccd": True,
            "particle_contact_offset_m": 0.003,
            "fluid_rest_offset_m": 0.0015,
            "contact_offset_m": 0.003,
            "rest_offset_m": 0.0029,
            "solid_rest_offset_m": 0.0029,
            "solver_position_iteration_count": 16,
            "max_neighborhood": 96,
            "pbd_material_readback": {
                "density_kg_m3": 1000.0,
                "friction": 0.1,
                "cohesion": 0.0,
                "damping": 0.0,
                "surface_tension": 0.0,
                "viscosity": 0.0,
                "vorticity_confinement": 0.0,
            },
        },
        "point_instancer": {
            "path": "/FillStation/FluidmixF2/FluidParticles/Shared",
            "max_particles": 2221,
            "ids_stable": True,
            "simulation_points_authored": True,
            "mass_density_readback": {
                "mass_api_present": True,
                "total_mass_attr_kg": 0.0,
                "density_attr_kg_m3": 1000.0,
                "precedence": "nonzero_mass_overrides_density_else_density_computes_mass",
                "source": "NVIDIA particleUtils.py",
            },
        },
        "captures": [record],
        "render_limitations": {"continuous_surface": False, "palette_uncalibrated": True, "pbd_not_navier_stokes": True},
        "semantics": {
            "transparent_annotation_override": {
                "capture_mode": "paired_same_physics_state_rgb_then_semantic",
                "documentation_url": "https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotations_with_transparency.html",
                "semantic_pass_value": True,
                "rgb_and_restore_value": False,
                "source": {"value": None},
                "initial_rgb_state": {
                    "composed_value": False,
                    "property_stack_strongest_is_session_layer": True,
                    "source_equivalent": True,
                },
                "transitions": [
                    {
                        "capture_pass": "kept_rgb_source_equivalent",
                        "requested_value": False,
                        "composed_type_name": "bool",
                        "composed_value": False,
                        "property_stack_strongest_is_session_layer": True,
                        "physics_step": 120,
                        "phase": "t200-stage",
                        "attempt": 1,
                    },
                    {
                        "capture_pass": "semantic_transparent_annotation",
                        "requested_value": True,
                        "composed_type_name": "bool",
                        "composed_value": True,
                        "property_stack_strongest_is_session_layer": True,
                        "physics_step": 120,
                        "phase": "t200-stage",
                        "attempt": 1,
                    },
                    {
                        "capture_pass": "restore_after_semantic",
                        "requested_value": False,
                        "composed_type_name": "bool",
                        "composed_value": False,
                        "property_stack_strongest_is_session_layer": True,
                        "physics_step": 120,
                        "phase": "t200-stage",
                        "attempt": 1,
                    },
                ],
                "property_stack_strongest_is_session_layer": True,
                "authored_after_untouched_reference_capture": True,
                "source_layer_unchanged_expected": True,
            },
            "paired_capture_contract": {
                "shared_single_render_pass": False,
                "paired_same_physics_state": True,
                "kept_rgb_uses_source_equivalent_glass": True,
                "semantic_pass_uses_transparent_annotation_flag": True,
                "flag_restored_false_after_semantic_pass": True,
                "state_fields_required_unchanged": [
                    "callback_count", "timeline_time_s", "camera_path", "positions",
                    "simulation_points", "velocities", "ids", "proto_indices",
                ],
                "state_records": [
                    {
                        "passed": True,
                        "failed_predicates": [],
                        "row_count": 1,
                        "callback_count_before": 120,
                        "callback_count_after": 120,
                        "timeline_time_s_before": 1.0,
                        "timeline_time_s_after": 1.0,
                        "camera_path_before": "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
                        "camera_path_after": "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
                        **{
                            f"{field}_sha256_{side}": "a" * 64
                            for field in (
                                "positions",
                                "simulation_points",
                                "velocities",
                                "ids",
                                "proto_indices",
                            )
                            for side in ("before", "after")
                        },
                        "physics_step": 120,
                        "phase": "t200-stage",
                        "attempt": 1,
                        "rgb_guard_record_index": 0,
                        "semantic_guard_record_index": 1,
                    }
                ],
            },
            "pairs": [
                {
                    "paired_same_physics_state_rgb_semantic_capture": True,
                    "paired_capture_state_passed": True,
                }
            ],
        },
        "timing": {
            "capture_guard_records": [
                {
                    "capture_pass": "kept_rgb_source_equivalent",
                    "physics_step": 120,
                    "phase": "t200-stage",
                    "attempt": 1,
                    "callbacks_before": 120,
                    "callbacks_after": 120,
                    "timeline_time_before_s": 1.0,
                    "timeline_time_after_s": 1.0,
                    "timeline_playing_before": True,
                    "timeline_playing_after": True,
                },
                {
                    "capture_pass": "semantic_transparent_annotation",
                    "physics_step": 120,
                    "phase": "t200-stage",
                    "attempt": 1,
                    "callbacks_before": 120,
                    "callbacks_after": 120,
                    "timeline_time_before_s": 1.0,
                    "timeline_time_after_s": 1.0,
                    "timeline_playing_before": True,
                    "timeline_playing_after": True,
                },
            ]
        },
    }
    assert validate_paint_ball_receipt(receipt) == ()

    mutations = (
        (
            lambda value: value["semantics"]["paired_capture_contract"].update(
                shared_single_render_pass=True
            ),
            "paired_capture_contract",
        ),
        (
            lambda value: value["semantics"]["paired_capture_contract"][
                "state_records"
            ][0].update(passed=False, failed_predicates=["positions"]),
            "paired_capture_state_drift",
        ),
        (
            lambda value: value["semantics"]["transparent_annotation_override"][
                "transitions"
            ][-1].update(composed_value=True),
            "transparent_annotation_restore",
        ),
        (
            lambda value: value["timing"].update(capture_guard_records=[]),
            "paired_capture_guards",
        ),
        (
            lambda value: value["captures"][0]["volume_gate"].update(
                passed=False,
                state_passed=False,
                failure_reasons=["particle_height_outside_tolerance"],
            ),
            "volume_state_gate",
        ),
    )
    for mutate, expected in mutations:
        changed = deepcopy(receipt)
        mutate(changed)
        assert expected in validate_paint_ball_receipt(changed)

    visible_failure = deepcopy(receipt)
    visible_failure["captures"][0]["volume_gate"].update(
        passed=False,
        visible_passed=False,
        failure_reasons=["visible_height_outside_tolerance"],
    )
    visible_failures = validate_paint_ball_receipt(visible_failure)
    assert "volume_visible_gate" in visible_failures
    assert "volume_state_gate" not in visible_failures


def test_stage_capture_records_complete_programmed_volume_state_gate():
    bottom_m = 0.9762065645
    target_height_m = 0.019122235236979125
    radius_m = 0.0015
    positions = np.column_stack(
        (
            np.zeros(370),
            np.zeros(370),
            np.linspace(
                bottom_m + radius_m,
                bottom_m + target_height_m - radius_m,
                370,
            ),
        )
    )
    containment = containment_evidence(
        positions,
        lower_bounds=(-1.0, -1.0, bottom_m),
        upper_bounds=(1.0, 1.0, 1.1762601692),
        cylindrical_center_xy=(0.0, 0.0),
        inner_radius_m=0.012753042956,
        polygon_center_xy=(0.0, 0.0),
        polygon_apothem_m=0.012753042956,
        polygon_wall_count=12,
    )

    record = build_capture_record(
        physics_step=1412,
        phase="t200-stage",
        containment=containment,
        prototype_count=15,
        instancer_path="/FillStation/FluidmixF2/FluidParticles/Shared",
        emitted_particle_count=370,
        programmed_cumulative_volume_ml=10.0,
        expected_cumulative_particle_count=370,
        cross_section_area_m2=0.0005229514163,
        height_tolerance_m=0.006,
        render_radius_m=radius_m,
        point_instancer_scales=np.ones((370, 3)),
        smoothed_render_containment=containment,
    )

    assert record["emitted_particle_count"] == 370
    assert record["in_flight_above_mouth_count"] == 0
    assert record["outside_or_escaped_particle_count"] == 0
    assert record["contained_z_min_m"] == pytest.approx(bottom_m + radius_m)
    assert record["contained_z_max_m"] == pytest.approx(
        bottom_m + target_height_m - radius_m
    )
    assert record["point_instancer_scale_component_min"] == [1.0, 1.0, 1.0]
    assert record["point_instancer_scale_component_max"] == [1.0, 1.0, 1.0]
    assert record["volume_gate"]["particle_bulk_height_m"] == pytest.approx(
        target_height_m
    )
    assert record["volume_gate"]["state_passed"] is True
    assert record["volume_gate"]["passed"] is False

    roi = paint_ball_emitter.projected_vial_geometry_roi_mask(
        (480, 640),
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=bottom_m,
        mouth_z_m=1.1762601692,
        circumradius_m=0.0132029216,
    )
    surface_row = int(np.ceil(paint_ball_emitter.projected_top_dodecagon_min_y_px(
        target_height_m,
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=bottom_m,
        circumradius_m=0.0132029216,
    )))
    liquid_mask = np.zeros_like(roi)
    liquid_mask[surface_row:] = roi[surface_row:]
    completed = attach_visible_volume_evidence(
        record,
        liquid_mask,
        center_xy_m=(5.2324003281, -1.9811999798),
        bottom_z_m=bottom_m,
        mouth_z_m=1.1762601692,
        circumradius_m=0.0132029216,
    )

    assert completed["volume_gate"]["visible_passed"] is True
    assert completed["volume_gate"]["passed"] is True


def test_120hz_plan_has_frozen_first_stage_and_final_capture_steps():
    plan = build_three_pour_plan(physics_hz=120)
    assert plan["timing"]["physics_hz"] == 120
    assert [pour["stage_capture_step"] for pour in plan["pours"]] == [1412, 2622, 4004]
    assert plan["timing"]["fill_end_step"] == 3968
    assert plan["timing"]["primary_final_capture_step"] == 4004
    assert plan["timing"]["primary_final_offset_s"] == pytest.approx(0.3)


def test_emitter_route_has_no_live_mesh_or_isosurface_lifecycle_call():
    source = (Path(__file__).parents[1] / "scripts" / "physx_three_pour_sequence.py").read_text()
    run_source = source[source.index("def run_sequence") : source.index("\ndef main")]
    assert "build_liquid_proxy_intent(" not in run_source
    assert "copy_exact_mesh_proxy(" not in run_source
    assert "refresh_exact_mesh_proxy(" not in run_source
    assert "_refresh_stage_isosurface_lifecycle(" not in run_source


def test_qualification_capture_contract_is_bounded_and_requires_hidden_target():
    source = (
        Path(__file__).parents[1] / "scripts" / "physx_three_pour_sequence.py"
    ).read_text()
    assert "{706, 1404, 1412}" in source
    assert "sorted(observed_steps) == sorted(int(step) for step in capture_steps)" in source
    assert "len(target_records) == 1" in source
    assert 'target_records[0].get("visible_stream_tanks") == []' in source
    assert "FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS" in source


def test_transparent_vial_control_follows_reference_and_pairs_only_semantic_pass():
    source = (
        Path(__file__).parents[1] / "scripts" / "physx_three_pour_sequence.py"
    ).read_text()
    run_source = source[source.index("def run_sequence") : source.index("\ndef main")]
    assert (
        run_source.index('failure_phase = "capture_twin_reference"')
        < run_source.index(
            'failure_phase = "prepare_transparent_vial_annotation_control"'
        )
        < run_source.index('failure_phase = "author_particle_system"')
    )
    assert '"property_stack_strongest_is_session_layer": True' in source
    assert 'set_annotation_state(False, "kept_rgb_source_equivalent"' in source
    assert 'True, "semantic_transparent_annotation"' in source
    assert 'set_annotation_state(False, "restore_after_semantic"' in source
    assert run_source.index("prefill_rgb = acquire_rgb") < run_source.index(
        "timeline.play()"
    )

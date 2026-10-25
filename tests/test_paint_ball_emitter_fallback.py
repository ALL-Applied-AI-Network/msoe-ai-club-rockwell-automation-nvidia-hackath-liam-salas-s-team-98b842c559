"""CPU contracts for the Paint Ball Emitter PointInstancer route."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import numpy as np
import pytest
from pathlib import Path

from fluidmix.physx_three_pour import build_three_pour_plan

from fluidmix.paint_ball_emitter import (
    COUNT_DRIVEN_DODECAGON_AREA_M2,
    COUNT_DRIVEN_PARTICLE_VOLUME_M3,
    COUNT_DRIVEN_RENDER_PATH,
    MAX_VELOCITY_M_S,
    PAINT_BALL_RECEIPT_SCHEMA_VERSION,
    append_point_instancer_state,
    attach_visible_volume_evidence,
    build_count_driven_liquid_body_geometry,
    build_capture_record,
    build_prototype_specs,
    containment_evidence,
    signed_mesh_volume_m3,
    validate_paint_ball_receipt,
)
import fluidmix.paint_ball_emitter as paint_ball_emitter


def _artifact_hash(index: int, kind: str) -> str:
    return hashlib.sha256(f"{kind}-{index}".encode()).hexdigest()


def test_count_driven_body_preserves_bottom_and_proves_nominal_particle_volume():
    source_points = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.122894103],
            [0.5, -0.5, 0.122894103],
            [0.5, 0.5, 0.122894103],
            [-0.5, 0.5, 0.122894103],
        ],
        dtype=np.float64,
    )

    geometry = build_count_driven_liquid_body_geometry(
        source_points,
        contained_particle_count=370,
        source_world_height_m=0.122894103,
        source_effective_area_m2=543.255377e-6,
    )

    assert geometry["representation"] == "count_driven_twin_liquid_mesh"
    assert geometry["contained_particle_count"] == 370
    assert geometry["visible"] is True
    assert geometry["particle_volume_m3"] == COUNT_DRIVEN_PARTICLE_VOLUME_M3
    assert geometry["cross_section_area_m2"] == pytest.approx(
        COUNT_DRIVEN_DODECAGON_AREA_M2
    )
    assert geometry["target_height_m"] == pytest.approx(
        370 * 27e-9 / COUNT_DRIVEN_DODECAGON_AREA_M2
    )
    assert geometry["xy_area_correction_scale"] == pytest.approx(
        np.sqrt(COUNT_DRIVEN_DODECAGON_AREA_M2 / 543.255377e-6)
    )
    points = np.asarray(geometry["points"], dtype=np.float64)
    assert points[:, 2].min() == pytest.approx(0.0)
    assert points[:, 2].max() == pytest.approx(geometry["target_height_m"])
    assert geometry["expected_post_transform_volume_m3"] == pytest.approx(
        370 * 27e-9
    )
    assert geometry["expected_volume_residual_m3"] == pytest.approx(0.0, abs=1e-18)
    assert len(geometry["points_sha256"]) == 64

    empty = build_count_driven_liquid_body_geometry(
        source_points,
        contained_particle_count=0,
        source_world_height_m=0.122894103,
        source_effective_area_m2=543.255377e-6,
    )
    assert empty["visible"] is False
    assert np.ptp(np.asarray(empty["points"])[:, 2]) == pytest.approx(0.0)


def test_signed_mesh_volume_triangulates_polygon_faces():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    face_counts = [4, 4, 4, 4, 4, 4]
    face_indices = [
        0, 3, 2, 1,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
    ]

    assert signed_mesh_volume_m3(points, face_counts, face_indices) == pytest.approx(1.0)


def _full_run_contract_fixture():
    frame_steps = sorted({*range(12, 4005, 12), 1412, 2622, 4004})
    frame_paths = [f"frames/frame-{index:04d}.png" for index in range(336)]
    frame_hashes = [_artifact_hash(index, "rgb") for index in range(336)]
    mask_paths = [f"liquid/mask-{index:04d}.png" for index in range(336)]
    mask_hashes = [_artifact_hash(index, "mask") for index in range(336)]
    semantic_paths = [
        f"semantic/frame-{index:04d}.npy" for index in range(336)
    ]
    semantic_hashes = [
        _artifact_hash(index, "semantic") for index in range(336)
    ]
    stage_indices = [117, 219, 335]
    required_rows = []
    for step, volume_ml, count in ((1412, 10, 370), (2622, 20, 740), (4004, 60, 2221)):
        required_rows.append(
            {
                "physics_step": step,
                "emitted_particle_count": count,
                "active_particle_count": count,
                "volume_gate": {
                    "required": True,
                    "programmed_cumulative_volume_ml": volume_ml,
                    "expected_cumulative_particle_count": count,
                    "state_passed": True,
                    "visible_passed": True,
                    "passed": True,
                    "failure_reasons": [],
                },
            }
        )
    pairs = [
        {
            "physics_step": step,
            "rgb_path": frame_paths[index],
            "rgb_sha256": frame_hashes[index],
            "mask_path": mask_paths[index],
            "mask_sha256": mask_hashes[index],
            "semantic_path": semantic_paths[index],
            "semantic_sha256": semantic_hashes[index],
            "liquid_representation": "point_instancer",
            "shared_rgb_semantic_capture": False,
            "liquid_pixel_count": 100,
            "mask_readable": True,
            "mask_resolution": [640, 480],
            "paired_same_physics_state_rgb_semantic_capture": True,
            "paired_capture_state_passed": True,
            "visible_stream_tanks": (
                []
                if index in stage_indices or step > 3968
                else ["t200" if step <= 1412 else "t400" if step <= 2622 else "t100"]
            ),
        }
        for index, step in enumerate(frame_steps)
    ]
    return {
        "status": "pass",
        "qualification": {"first_stage_only": False},
        "arguments": {
            "resolution": "640x480",
            "capture_stride": 12,
            "video_fps": 10.0,
        },
        "captures": required_rows,
        "scene_contract": {
            "default_prim": "/FillStation",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "source_layer_records": [
                {
                    "resolved_path": "/tmp/fill_station.usda",
                    "sha256": _artifact_hash(0, "layer"),
                }
            ],
            "source_layers_unchanged": True,
            "hidden_source_visibility": {
                "/FillStation/FillStation_LocalRoot/Fill_liquid": {"1": "invisible"},
                "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream": {
                    "1": "invisible"
                },
            },
            "forbidden_authored_paths": [],
            "custom_camera_count": 0,
            "custom_light_count": 0,
            "custom_backdrop_count": 0,
        },
        "camera": {
            "path": "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
            "source_has_authored_clipping_range": True,
            "clipping_range_authored_by_runner": False,
            "clipping_range_effective": [0.0007619999814778566, 25.399999618530273],
            "resolution": [640, 480],
        },
        "pre_pour_parity": {
            "reference_path": "pre-pour-parity/reference.png",
            "reference_sha256": _artifact_hash(0, "parity-reference"),
            "f2_prefill_path": "pre-pour-parity/prefill.png",
            "f2_prefill_sha256": _artifact_hash(0, "parity-prefill"),
            "absdiff_path": "pre-pour-parity/absdiff.png",
            "absdiff_sha256": _artifact_hash(0, "parity-difference"),
            "passed": True,
        },
        "stream_render": {
            "representation": "render_only_mesh_current_tank_material",
            "records": [
                {
                    "ordinal": ordinal,
                    "tank_id": tank_id,
                    "start_step": start,
                    "end_step": end,
                    "active_only_during_own_pour": True,
                    "state_artifact_path": f"stream-states/{ordinal}-{tank_id}.json",
                    "state_artifact_sha256": _artifact_hash(ordinal, "stream"),
                }
                for ordinal, tank_id, start, end in (
                    (1, "t200", 1, 1412),
                    (2, "t400", 1413, 2622),
                    (3, "t100", 2623, 3968),
                )
            ],
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 64,
            "warmup_updates": 60,
            "frame_count": 336,
            "frame_paths": frame_paths,
            "frame_sha256": frame_hashes,
            "frame_physics_steps": frame_steps,
            "stage_frame_paths": [frame_paths[index] for index in stage_indices],
            "stage_frame_sha256": [frame_hashes[index] for index in stage_indices],
            "stage_frame_indices": stage_indices,
            "stage_frame_physics_steps": [1412, 2622, 4004],
            "primary_final_frame_sha256": frame_hashes[-1],
            "video_path": "physx-three-pour.mp4",
            "video_sha256": _artifact_hash(0, "video"),
            "video_codec": "mp4v",
            "video_container": "mp4",
            "video_writer_fallback_used": False,
            "video_decodable": True,
            "video_decoded_frame_count": 336,
            "stage_frames_are_video_subsets": True,
            "stream_only_frame_count": 0,
            "vial_proxy_frame_count": 336,
            "vial_proxy_transition_step": 12,
            "liquid_representation_monotonic": True,
        },
        "timing": {
            "physics_hz": 120,
            "capture_hz": 10,
            "pour_duration_steps": [1412, 1210, 1346],
            "pour_end_steps": [1412, 2622, 3968],
            "no_inter_pour_settle": True,
            "fill_end_step": 3968,
            "primary_final_capture_step": 4004,
            "primary_final_offset_s": 0.3,
            "executed_physics_steps": 4004,
            "callbacks_expected": 4004,
            "callbacks_observed": 4004,
            "one_callback_per_app_update": True,
            "capture_callbacks_advanced": 0,
            "capture_timeline_time_advanced": False,
            "capture_playback_preserved": True,
            "capture_guard_records": [{}] * 672,
        },
        "semantics": {
            "semantic_label": "liquid",
            "semantic_type": "class",
            "annotator": "semantic_segmentation",
            "mask_source": "exact_renderer_semantic_class_ids",
            "resolution": [640, 480],
            "point_instancer_semantic_bound": True,
            "stream_semantic_bound": True,
            "capture_volumes_enabled_readback": True,
            "capture_volumes_restored_readback": False,
            "semantic_gate_passed": True,
            "training_eligible": False,
            "pairs": pairs,
            "stage_mask_sha256": [mask_hashes[index] for index in stage_indices],
            "frame_semantic_paths": semantic_paths,
            "frame_semantic_sha256": semantic_hashes,
            "paired_capture_contract": {"state_records": [{}] * 336},
            "transparent_annotation_override": {"transitions": [{}] * 1008},
        },
    }


def _count_driven_body_contract_fixture():
    receipt = _full_run_contract_fixture()
    receipt["render_path"] = COUNT_DRIVEN_RENDER_PATH
    receipt["claims"] = {
        "particle_composition_rendered": False,
        "spatial_mixing_color_rendered": False,
        "spatial_layering_color_rendered": False,
        "global_vial_blend_rendered": True,
    }
    receipt["containment"] = {
        "capture_records": [
            {
                "physics_step": step,
                "emitted_particle_count": count,
                "active_particle_count": count,
                "inside_vial_interior_count": count,
                "outside_vial_interior_count": 0,
                "expected_in_flight_stream_count": 0,
                "leaked_particle_count": 0,
                "finite_state": True,
            }
            for step, _volume, count in paint_ball_emitter.FULL_RUN_VOLUME_ROWS
        ]
    }
    receipt["render"].update(
        {
            "count_driven_body_frame_count": 336,
            "count_driven_body_transition_step": 12,
        }
    )
    receipt["render"].pop("vial_proxy_frame_count")
    receipt["render"].pop("vial_proxy_transition_step")
    for pair in receipt["semantics"]["pairs"]:
        pair["liquid_representation"] = COUNT_DRIVEN_RENDER_PATH
        pair["body_points_sha256"] = "a" * 64
        pair["body_extent_sha256"] = "b" * 64
        pair["body_visibility"] = "inherited"
        pair["body_material_color_sha256"] = "c" * 64
        pair["stream_visibility_sha256"] = "d" * 64
        pair["compositions_sha256"] = "e" * 64
    receipt["semantics"].update(
        {
            "count_driven_body_semantic_bound": True,
            "point_instancer_rgb_visible": False,
            "point_instancer_semantic_visible": False,
            "hidden_particle_control": {
                "physics_step": 1412,
                "emitted_particle_count": 370,
                "inside_vial_interior_count": 370,
                "body_and_stream_hidden_liquid_pixel_count": 0,
                "renderer_accumulation_reset_after_hide": {
                    "supported": True,
                    "called": True,
                },
                "control_semantic_path": "semantic-controls/emitted-spheres-hidden.npy",
                "control_semantic_sha256": "f" * 64,
                "body_only_target_path": (
                    "/FillStation/FillStation_LocalRoot/FluidmixF2CountDrivenLiquid/Fill_liquid"
                ),
                "body_only_target_ids": [7],
                "body_only_capture_step": 1412,
                "body_only_liquid_pixel_count": 1234,
                "passed": True,
            },
        }
    )
    receipt["count_driven_liquid_body"] = {
        "representation": COUNT_DRIVEN_RENDER_PATH,
        "source_xform_path": "/FillStation/FillStation_LocalRoot/Fill_liquid",
        "source_mesh_path": (
            "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
        ),
        "body_xform_path": (
            "/FillStation/FillStation_LocalRoot/FluidmixF2CountDrivenLiquid"
        ),
        "body_mesh_path": (
            "/FillStation/FillStation_LocalRoot/FluidmixF2CountDrivenLiquid/Fill_liquid"
        ),
        "requested_source_time_code": 686,
        "resolved_points_time_code": 681,
        "resolved_normals_time_code": 681,
        "resolved_extent_time_code": 668,
        "source_point_count": 512,
        "source_face_count": 448,
        "source_corner_count": 1664,
        "source_fan_triangle_count": 768,
        "source_geometry_sha256": (
            "e1b9711dc50195e559f54f96db2ce5389a34ef719213c443498a5585b6bd712f"
        ),
        "source_topology_sha256": (
            "edba8a108a846fbb3fd661e95b9a642b727a06e7ba431ca70e013d1255132acd"
        ),
        "copied_primvar_names": [
            "primvars:displayColor",
            "primvars:displayOpacity",
        ],
        "source_world_bottom_z_m": 0.9762065645,
        "source_world_height_m": 0.122894103,
        "source_signed_volume_m3": 66.762882e-6,
        "source_effective_area_m2": 543.255377e-6,
        "cross_section_area_m2": COUNT_DRIVEN_DODECAGON_AREA_M2,
        "xy_area_correction_scale": np.sqrt(
            COUNT_DRIVEN_DODECAGON_AREA_M2 / (543.255377e-6)
        ),
        "particle_volume_m3": COUNT_DRIVEN_PARTICLE_VOLUME_M3,
        "body_starts_hidden_at_zero_count": True,
        "sphere_instances_hidden": True,
        "physics_point_instancer_preserved": True,
        "pbd_state_independent": True,
        "global_uncalibrated_blend": True,
        "spatial_mixing_color_rendered": False,
        "spatial_layering_color_rendered": False,
        "normals_policy": "omit_authored_normals_after_anisotropic_scale",
        "extent_recomputed_each_update": True,
        "geometry_authoring_numeric_path": (
            "local_points_vec3f_then_inherited_world_transform"
        ),
        "post_transform_volume_residual_abs_tolerance_m3": 2.0e-10,
        "material_path": "/FillStation/FluidmixF2/GlobalLiquidMaterial",
        "semantic_label": "liquid",
        "stage_geometry": [
            {
                "physics_step": step,
                "contained_particle_count": count,
                "target_volume_m3": count * COUNT_DRIVEN_PARTICLE_VOLUME_M3,
                "target_height_m": (
                    count
                    * COUNT_DRIVEN_PARTICLE_VOLUME_M3
                    / COUNT_DRIVEN_DODECAGON_AREA_M2
                ),
                "world_bottom_z_m": 0.9762065645,
                "world_extent_m": [
                    [5.2194, -1.9942, 0.9762065645],
                    [
                        5.2454,
                        -1.9682,
                        0.9762065645
                        + count
                        * COUNT_DRIVEN_PARTICLE_VOLUME_M3
                        / COUNT_DRIVEN_DODECAGON_AREA_M2,
                    ],
                ],
                "post_transform_signed_volume_m3": (
                    count * COUNT_DRIVEN_PARTICLE_VOLUME_M3
                ),
                "post_transform_volume_residual_m3": 0.0,
                "volume_proof_coordinate_path": (
                    "local_points_vec3f_then_inherited_world_transform"
                ),
                "points_sha256": "1" * 64,
                "extent_sha256": "2" * 64,
            }
            for step, _volume, count in paint_ball_emitter.FULL_RUN_VOLUME_ROWS
        ],
        "training_eligible": False,
    }
    receipt["global_blend"] = {
        "global_blend_rule": "contained_particle_mean_linear_transmission_parameter",
        "spatial_mixing_color_rendered": False,
        "spatial_layering_color_rendered": False,
    }
    return receipt


def test_full_run_contract_accepts_only_complete_count_driven_body_evidence():
    receipt = _count_driven_body_contract_fixture()

    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()

    changed = deepcopy(receipt)
    changed["count_driven_liquid_body"]["stage_geometry"][2][
        "post_transform_volume_residual_m3"
    ] = 1.0e-6
    assert "count_driven_body_contract" in paint_ball_emitter._validate_full_run_contract(
        changed
    )


def test_full_run_contract_accepts_hashed_stage_copies_of_video_frames():
    receipt = _count_driven_body_contract_fixture()
    receipt["render"]["stage_frame_paths"] = [
        f"stages/stage-{index + 1}.png" for index in range(3)
    ]

    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()


def test_full_run_contract_requires_exact_frozen_artifact_and_stage_denominators():
    receipt = _full_run_contract_fixture()

    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()

    mutations = (
        (lambda value: value["captures"][1]["volume_gate"].update(
            expected_cumulative_particle_count=739
        ), "full_run_volume_contract"),
        (lambda value: value["render"]["frame_paths"].pop(), "full_run_rgb_contract"),
        (lambda value: value["semantics"]["pairs"].pop(), "full_run_semantic_contract"),
        (lambda value: value["render"].update(video_decoded_frame_count=335), "full_run_video_contract"),
        (lambda value: value["timing"].update(primary_final_offset_s=2.0), "full_run_timing_contract"),
        (lambda value: value["scene_contract"].update(custom_light_count=1), "full_run_scene_contract"),
        (lambda value: value["stream_render"]["records"][2].update(tank_id="t200"), "full_run_stream_contract"),
        (lambda value: value["semantics"].update(training_eligible=True), "training_eligibility_contract"),
        (lambda value: value.update(status="insufficient_evidence"), "full_run_status_contract"),
    )
    for mutate, expected in mutations:
        changed = deepcopy(receipt)
        mutate(changed)
        assert expected in paint_ball_emitter._validate_full_run_contract(changed)


def test_public_receipt_validator_applies_full_run_contract():
    receipt = _full_run_contract_fixture()
    receipt.update(
        schema_version=PAINT_BALL_RECEIPT_SCHEMA_VERSION,
        experiment="physx_three_pour_f2_paint_ball_emitter",
        render_path="point_instancer_sphere_emitter",
        status="pass",
    )
    receipt["render"]["frame_count"] = 335

    assert "full_run_rgb_contract" in validate_paint_ball_receipt(receipt)


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

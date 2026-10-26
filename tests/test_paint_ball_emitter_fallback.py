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
    RENDER_PRODUCT_CLEANUP_DOCUMENTATION_URL,
    RENDER_PRODUCT_GATING_DOCUMENTATION_URL,
    RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES,
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


def _attach_valid_render_product_gating(
    receipt: dict[str, object], *, physics_step_count: int, frame_count: int
) -> None:
    """Attach a compact, internally consistent accepted-run gating fixture."""

    rgb_pass_count = frame_count + 2
    semantic_pass_count = frame_count
    explicit_pass_count = 1 + rgb_pass_count + semantic_pass_count
    rotation_count = (
        (frame_count - 1) // RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES
    )
    render = receipt.setdefault("render", {})
    assert isinstance(render, dict)
    render["frame_count"] = frame_count
    render["render_product_update_gating"] = {
        "documented_source": RENDER_PRODUCT_GATING_DOCUMENTATION_URL,
        "cleanup_documented_source": (
            RENDER_PRODUCT_CLEANUP_DOCUMENTATION_URL
        ),
        "initial_disabled_readback": True,
        "final_disabled_readback": True,
        "current_updates_enabled": False,
        "enabled_only_for_explicit_render_passes": True,
        "rotation_interval_logical_frames": (
            RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES
        ),
        "rotation_count": rotation_count,
        "rotations": [
            {
                "completed_frame_count": (
                    index * RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES
                ),
                "generation_before": index - 1,
                "generation_after": index,
                "wait_until_complete": True,
                "annotator_count": 3,
                "detached_annotator_count": 3,
                "destroy_called": True,
                "flush_invariants": {
                    "callbacks_unchanged": True,
                    "timeline_time_unchanged": True,
                    "playback_restored": True,
                    "passed": True,
                },
                "memory_before_destroy_mib": 18000,
                "memory_after_destroy_mib": 6000,
                "memory_released_mib": 12000,
                "measurable_memory_release": True,
                "new_product_disabled_readback": True,
                "reattached_annotator_count": 3,
                "passed": True,
            }
            for index in range(1, rotation_count + 1)
        ],
        "physics_updates_while_disabled": physics_step_count,
        "enable_request_count": explicit_pass_count,
        "disable_request_count": explicit_pass_count + 2 + rotation_count,
        "explicit_render_pass_count": explicit_pass_count,
        "transitions": (
            [
                {
                    "phase": "initial_after_create",
                    "requested_updates_enabled": False,
                    "updates_enabled_before": False,
                    "updates_enabled_after": False,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
            ]
            + [
                {
                    "phase": f"rotation_generation_{index:02d}_after_create",
                    "requested_updates_enabled": False,
                    "updates_enabled_before": True,
                    "updates_enabled_after": False,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
                for index in range(1, rotation_count + 1)
            ]
            + [
                {
                    "phase": "fixture_explicit_pass",
                    "requested_updates_enabled": enabled,
                    "updates_enabled_before": not enabled,
                    "updates_enabled_after": enabled,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
                for _ in range(explicit_pass_count)
                for enabled in (True, False)
            ]
            + [
                {
                    "phase": "final_before_acceptance",
                    "requested_updates_enabled": False,
                    "updates_enabled_before": False,
                    "updates_enabled_after": False,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
            ]
        ),
        "explicit_render_passes": (
            [
                {
                    "phase": "warmup",
                    "timing_phase": "warmup_app_update",
                    "elapsed_seconds": 0.01,
                    "passed": True,
                    "disabled_after": True,
                }
            ]
            + [
                {
                    "phase": "rgb",
                    "timing_phase": "capture_rgb_orchestrator",
                    "elapsed_seconds": 0.01,
                    "passed": True,
                    "disabled_after": True,
                }
            ]
            * rgb_pass_count
            + [
                {
                    "phase": "semantic",
                    "timing_phase": "capture_semantic_orchestrator",
                    "elapsed_seconds": 0.01,
                    "passed": True,
                    "disabled_after": True,
                }
            ]
            * semantic_pass_count
        ),
    }
    render["phase_timings_s"] = {
        "warmup_app_update": {"count": 1, "total": 1.0, "maximum": 1.0},
        "physics_app_update": {
            "count": physics_step_count,
            "total": 2.0,
            "maximum": 0.01,
        },
        "diffusion": {
            "count": physics_step_count,
            "total": 3.0,
            "maximum": 0.01,
        },
        "palette_mapping": {
            "count": physics_step_count,
            "total": 0.2,
            "maximum": 0.001,
        },
        "prototype_authoring": {
            "count": physics_step_count,
            "total": 0.3,
            "maximum": 0.001,
        },
        "capture_rgb_orchestrator": {
            "count": rgb_pass_count,
            "total": 4.0,
            "maximum": 0.1,
        },
        "capture_semantic_orchestrator": {
            "count": semantic_pass_count,
            "total": 4.0,
            "maximum": 0.1,
        },
        "capture_artifact_io": {
            "count": frame_count,
            "total": 2.0,
            "maximum": 0.1,
        },
    }
    timing = receipt.setdefault("timing", {})
    assert isinstance(timing, dict)
    timing["callbacks_observed"] = physics_step_count


def _persistent_direct_render_product_fixture(
    *, frame_count: int = 96, physics_step_count: int = 1152
) -> dict[str, object]:
    frame_steps = [12 * index for index in range(1, frame_count + 1)]
    rgb_pass_count = frame_count + 2
    semantic_pass_count = frame_count
    explicit_pass_count = 1 + rgb_pass_count + semantic_pass_count
    passes = [
        {
            "phase": "warmup",
            "timing_phase": "warmup_app_update",
            "elapsed_seconds": 0.01,
            "renderer_accumulation_reset_before": False,
            "enabled_before": True,
            "enabled_after": True,
            "passed": True,
        }
    ]
    passes.extend(
        {
            "phase": "rgb",
            "timing_phase": "capture_rgb_orchestrator",
            "elapsed_seconds": 0.01,
            "renderer_accumulation_reset_before": True,
            "enabled_before": True,
            "enabled_after": True,
            "passed": True,
        }
        for _ in range(rgb_pass_count)
    )
    passes.extend(
        {
            "phase": "semantic",
            "timing_phase": "capture_semantic_orchestrator",
            "elapsed_seconds": 0.01,
            "renderer_accumulation_reset_before": True,
            "enabled_before": True,
            "enabled_after": True,
            "passed": True,
        }
        for _ in range(semantic_pass_count)
    )
    paused_time = 0.0
    probe_steps = []
    for step in range(1, 13):
        particle_count = 1 if step == 12 else 0
        probe_steps.append(
            {
                "direct_step_index": step,
                "logical_time_s": step / 120.0,
                "callback_count_before": step - 1,
                "callback_count_after": step,
                "callback_count_delta": 1,
                "physics_step_count_before": step - 1,
                "physics_step_count_after": step,
                "physics_step_count_delta": 1,
                "simulation_time_before_s": (step - 1) / 120.0,
                "simulation_time_after_s": step / 120.0,
                "simulation_time_delta_s": 1.0 / 120.0,
                "timeline_time_before_s": paused_time,
                "timeline_time_after_s": paused_time,
                "timeline_playing_before": False,
                "timeline_playing_after": False,
                "callback_contract": {"passed": True},
                "failed_predicates": [],
                "passed": True,
                "particle_state": {
                    "particle_count": particle_count,
                    "positions_shape": [particle_count, 3],
                    "velocities_shape": [particle_count, 3],
                    "finite_positions": True,
                    "finite_velocities": True,
                    "ids_exact": True,
                    "passed": True,
                },
            }
        )
    capture_guards = [
        {
            "callbacks_before": 12,
            "callbacks_after": 12,
            "timeline_time_before_s": paused_time,
            "timeline_time_after_s": paused_time,
            "timeline_playing_before": False,
            "timeline_playing_after": False,
            "physics_steps_before": 12,
            "physics_steps_after": 12,
            "simulation_time_before_s": 0.1,
            "simulation_time_after_s": 0.1,
        }
        for _ in range(explicit_pass_count - 1)
    ]
    periodic_samples = [
        {
            "completed_frame_count": count,
            "physics_step": frame_steps[count - 1],
            "used_gpu_memory_mib": 18_000 + count,
        }
        for count in range(48, frame_count + 1, 48)
    ]
    return {
        "render": {
            "frame_count": frame_count,
            "frame_physics_steps": frame_steps,
            "render_product_update_gating": {
                "mode": "persistent_enabled_direct_physics",
                "documented_source": RENDER_PRODUCT_GATING_DOCUMENTATION_URL,
                "physics_step_documented_source": (
                    paint_ball_emitter.SIMULATION_MANAGER_STEP_DOCUMENTATION_URL
                ),
                "initial_enabled_readback": True,
                "final_enabled_readback": True,
                "current_updates_enabled": True,
                "persistent_enabled": True,
                "rotation_enabled": False,
                "rotation_disproof": {
                    "run_id": "C6",
                    "status": "failed",
                    "failure": (
                        "render-product destruction did not release GPU memory"
                    ),
                    "receipt_sha256": (
                        "bba03fe424408bf90e4585b92bc128e376aa66f8443780a4a76bdd0166793a4c"
                    ),
                },
                "render_product_create_count": 1,
                "physics_steps_while_enabled": physics_step_count,
                "physics_loop_app_update_count": 0,
                "update_toggle_request_count": 0,
                "explicit_render_pass_count": explicit_pass_count,
                "renderer_accumulation_reset_count": explicit_pass_count - 1,
                "explicit_render_passes": passes,
            },
            "phase_timings_s": {
                "warmup_app_update": {
                    "count": 1,
                    "total": 1.0,
                    "maximum": 1.0,
                },
                "physics_direct_step": {
                    "count": physics_step_count,
                    "total": 2.0,
                    "maximum": 0.01,
                },
                "diffusion": {
                    "count": physics_step_count,
                    "total": 3.0,
                    "maximum": 0.01,
                },
                "palette_mapping": {
                    "count": physics_step_count,
                    "total": 0.2,
                    "maximum": 0.001,
                },
                "prototype_authoring": {
                    "count": physics_step_count,
                    "total": 0.3,
                    "maximum": 0.001,
                },
                "capture_rgb_orchestrator": {
                    "count": rgb_pass_count,
                    "total": 4.0,
                    "maximum": 0.1,
                },
                "capture_semantic_orchestrator": {
                    "count": semantic_pass_count,
                    "total": 4.0,
                    "maximum": 0.1,
                },
                "capture_artifact_io": {
                    "count": frame_count,
                    "total": 2.0,
                    "maximum": 0.1,
                },
            },
        },
        "memory": {
            "hard_limit_mib": 40_000,
            "hard_limit_exceeded": False,
            "samples_mib": [
                row["used_gpu_memory_mib"] for row in periodic_samples
            ],
            "periodic_capture_interval_frames": 48,
            "periodic_capture_samples": periodic_samples,
        },
        "timing": {
            "physics_hz": 120,
            "simulation_owner": "SimulationManager.step",
            "physics_step_method": (
                "SimulationManager.step(steps=1, update_fabric=False)"
            ),
            "physics_step_documented_source": (
                paint_ball_emitter.SIMULATION_MANAGER_STEP_DOCUMENTATION_URL
            ),
            "executed_physics_steps": physics_step_count,
            "callbacks_expected": physics_step_count,
            "callbacks_observed": physics_step_count,
            "callback_indices": list(range(1, physics_step_count + 1)),
            "callback_dts_s": [1.0 / 120.0] * physics_step_count,
            "callback_times_s": [paused_time] * physics_step_count,
            "callback_direct_step_indices": list(
                range(1, physics_step_count + 1)
            ),
            "callback_logical_times_s": [
                index / 120.0 for index in range(1, physics_step_count + 1)
            ],
            "callback_contract_failure": None,
            "one_callback_per_direct_step": True,
            "physics_loop_app_update_count": 0,
            "direct_physx_api_used": True,
            "emissions_authored_on_main_thread": True,
            "emission_boundary_verified": True,
            "emission_boundaries": [
                {
                    "physics_step": 12,
                    "authored_after_callback_count": 11,
                    "scheduled_before_callback_index": 12,
                    "authored_on_main_thread": True,
                }
            ],
            "multitick_support": {"enabled_readback": True},
            "direct_step_initialization": {
                "timeline_playing": False,
                "paused_timeline_time_s": paused_time,
                "baseline_num_physics_steps": 0,
                "baseline_simulation_time_s": 0.0,
                "update_fabric": False,
            },
            "direct_step_probe": {
                "step_limit": 12,
                "first_capture_step": 12,
                "completed_steps": 12,
                "emission_events_observed": 1,
                "physics_contract_passed": True,
                "capture_contract_passed": True,
                "status": "pass",
                "step_evidence": probe_steps,
            },
            "capture_callbacks_advanced": 0,
            "capture_direct_physics_steps_advanced": 0,
            "capture_simulation_time_advanced": False,
            "capture_timeline_time_advanced": False,
            "capture_playback_preserved": True,
            "capture_guard_records": capture_guards,
        },
    }


def test_persistent_direct_render_product_validator_fails_closed():
    receipt = _persistent_direct_render_product_fixture()

    assert paint_ball_emitter._render_product_update_gating_valid(receipt)

    mutations = (
        lambda value: value["render"]["render_product_update_gating"].update(
            render_product_create_count=2
        ),
        lambda value: value["render"]["phase_timings_s"][
            "physics_direct_step"
        ].update(count=1151),
        lambda value: value["render"]["render_product_update_gating"].update(
            update_toggle_request_count=1
        ),
        lambda value: value["timing"]["capture_guard_records"][0].update(
            physics_steps_after=13
        ),
        lambda value: value["timing"]["direct_step_probe"]["step_evidence"][
            0
        ].update(timeline_time_before_s=None),
        lambda value: value["memory"]["periodic_capture_samples"].pop(),
    )
    for mutate in mutations:
        changed = deepcopy(receipt)
        mutate(changed)
        assert not paint_ball_emitter._render_product_update_gating_valid(
            changed
        )


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


def test_count_driven_body_accepts_the_measured_packing_particle_volume():
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
    particle_volume_m3 = 0.011272866937444935e-6

    geometry = build_count_driven_liquid_body_geometry(
        source_points,
        contained_particle_count=1774,
        particle_volume_m3=particle_volume_m3,
        source_world_height_m=0.122894103,
        source_effective_area_m2=543.255377e-6,
    )

    assert geometry["particle_volume_m3"] == pytest.approx(particle_volume_m3)
    assert geometry["target_volume_m3"] == pytest.approx(
        1774 * particle_volume_m3
    )
    assert geometry["target_height_m"] == pytest.approx(
        1774 * particle_volume_m3 / COUNT_DRIVEN_DODECAGON_AREA_M2
    )


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
            "pathtracing_total_spp_requested": 64,
            "pathtracing_total_spp_initial": 64,
            "pathtracing_total_spp_readback": 64,
            "pathtracing_spp_requested": 64,
            "pathtracing_spp_initial": 64,
            "pathtracing_spp_readback": 64,
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


def _recipe8_full_run_contract_fixture(
    *,
    affected: bool = False,
    particle_count_per_pour: int = 1280,
    capture_stride_steps: int = paint_ball_emitter.F2_SAFE_CAPTURE_STRIDE_STEPS,
):
    receipt = _full_run_contract_fixture()
    pours = []
    for ordinal, (tank, start, duration, end) in enumerate(
        (
            ("t200", 1, 1842, 1842),
            ("t400", 1843, 1440, 3282),
            ("t100", 3283, 1164, 4446),
        ),
        start=1,
    ):
        pour = {
                "ordinal": ordinal,
                "tank_id": tank,
                "programmed_volume_ml": 20.0,
                "particle_count": particle_count_per_pour,
                "cumulative_particle_count": ordinal * particle_count_per_pour,
                "start_step": start,
                "duration_steps": duration,
                "end_step": end,
                "stage_capture_step": 4482 if ordinal == 3 else end,
            }
        if affected:
            residual_end = (92, 1914, 3340)[ordinal - 1]
            predecessor = ("t100", "t200", "t400")[ordinal - 1]
            pour["emission_segments"] = [
                {
                    "ordinal": 1,
                    "segment_role": "leading_residual",
                    "source_tank_id": predecessor,
                    "particle_count": 89,
                    "start_step": start,
                    "end_step": residual_end,
                },
                {
                    "ordinal": 2,
                    "segment_role": "current_tank",
                    "source_tank_id": tank,
                    "particle_count": particle_count_per_pour - 89,
                    "start_step": residual_end + 1,
                    "end_step": end,
                },
            ]
        pours.append(pour)
    plan = {
        "timing": {
            "physics_hz": 120,
            "capture_hz": 10,
            "fill_end_step": 4446,
            "primary_final_capture_step": 4482,
            "primary_final_offset_s": 0.3,
            "no_inter_pour_settle": True,
        },
        "pours": pours,
        "totals": {"particle_count": 3 * particle_count_per_pour},
    }
    schedule = paint_ball_emitter.build_runtime_capture_schedule(
        plan, capture_stride_steps=capture_stride_steps
    )
    rows = schedule["captures"]
    frame_steps = [int(row["physics_step"]) for row in rows]
    frame_count = len(frame_steps)
    frame_paths = [f"frames/frame-{index:04d}.png" for index in range(frame_count)]
    frame_hashes = [_artifact_hash(index, "recipe8-rgb") for index in range(frame_count)]
    mask_paths = [f"liquid/mask-{index:04d}.png" for index in range(frame_count)]
    mask_hashes = [_artifact_hash(index, "recipe8-mask") for index in range(frame_count)]
    semantic_paths = [f"semantic/frame-{index:04d}.npy" for index in range(frame_count)]
    semantic_hashes = [
        _artifact_hash(index, "recipe8-semantic") for index in range(frame_count)
    ]
    stage_steps = [1842, 3282, 4482]
    stage_indices = [frame_steps.index(step) for step in stage_steps]

    required_rows = []
    for capture in rows:
        if capture["visible_stream_tanks"]:
            continue
        step = int(capture["physics_step"])
        completed = [pour for pour in pours if step >= int(pour["end_step"])]
        if not completed:
            continue
        count = sum(int(pour["particle_count"]) for pour in completed)
        volume_ml = sum(float(pour["programmed_volume_ml"]) for pour in completed)
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

    receipt.update(
        {
            "plan": plan,
            "runtime_capture_schedule": schedule,
            "particle_system": {"runtime_configuration": {}},
            "captures": required_rows,
        }
    )
    receipt["arguments"].update(
        {
            "capture_stride": capture_stride_steps,
            "video_fps": paint_ball_emitter.F2_VIDEO_PLAYBACK_FPS,
        }
    )
    video_repeat_plan = (
        paint_ball_emitter.build_time_preserving_video_repeat_plan(
            frame_steps,
            physics_hz=120,
            playback_fps=paint_ball_emitter.F2_VIDEO_PLAYBACK_FPS,
        )
    )
    stream_records = []
    for pour in pours:
        segments = pour.get("emission_segments")
        if not isinstance(segments, list):
            stream_records.append(
                {
                    "ordinal": pour["ordinal"],
                    "tank_id": pour["tank_id"],
                    "start_step": pour["start_step"],
                    "end_step": pour["end_step"],
                    "active_only_during_own_pour": True,
                    "state_artifact_path": f"stream-states/{pour['ordinal']}-{pour['tank_id']}.json",
                    "state_artifact_sha256": _artifact_hash(
                        pour["ordinal"], "recipe8-stream"
                    ),
                }
            )
            continue
        for segment in segments:
            stream_records.append(
                {
                    "ordinal": pour["ordinal"],
                    "segment_ordinal": segment["ordinal"],
                    "segment_role": segment["segment_role"],
                    "programmed_tank": pour["tank_id"],
                    "tank_id": segment["source_tank_id"],
                    "start_step": segment["start_step"],
                    "end_step": segment["end_step"],
                    "active_only_during_own_pour": True,
                    "active_only_during_own_segment": True,
                    "stream_and_birth_composition_equal": True,
                    "state_artifact_path": (
                        f"stream-states/{pour['ordinal']}-{segment['ordinal']}-"
                        f"{segment['source_tank_id']}.json"
                    ),
                    "state_artifact_sha256": _artifact_hash(
                        10 * pour["ordinal"] + segment["ordinal"],
                        "recipe8-segment-stream",
                    ),
                }
            )
    receipt["stream_render"]["records"] = stream_records
    receipt["render"].update(
        {
            "frame_count": frame_count,
            "frame_paths": frame_paths,
            "frame_sha256": frame_hashes,
            "frame_physics_steps": frame_steps,
            "stage_frame_paths": [frame_paths[index] for index in stage_indices],
            "stage_frame_sha256": [frame_hashes[index] for index in stage_indices],
            "stage_frame_indices": stage_indices,
            "stage_frame_physics_steps": stage_steps,
            "primary_final_frame_sha256": frame_hashes[stage_indices[-1]],
            "video_decoded_frame_count": video_repeat_plan[
                "video_frame_count"
            ],
            "video_playback_fps": video_repeat_plan["playback_fps"],
            "video_timing_algorithm": video_repeat_plan["algorithm"],
            "video_encoded_frame_time_rule": video_repeat_plan[
                "encoded_frame_time_rule"
            ],
            "video_timing_source_frame_steps": video_repeat_plan[
                "source_frame_steps"
            ],
            "video_source_capture_onset_frame_indices": video_repeat_plan[
                "source_capture_onset_frame_indices"
            ],
            "video_source_capture_onset_times_s": video_repeat_plan[
                "source_capture_onset_times_s"
            ],
            "video_source_capture_onset_errors_s": video_repeat_plan[
                "source_capture_onset_errors_s"
            ],
            "video_maximum_absolute_onset_error_s": video_repeat_plan[
                "maximum_absolute_onset_error_s"
            ],
            "video_frame_repeat_counts": video_repeat_plan["repeat_counts"],
            "video_frame_source_capture_indices": video_repeat_plan[
                "source_capture_indices"
            ],
            "video_source_duration_s": video_repeat_plan["source_duration_s"],
            "video_playback_duration_s": video_repeat_plan[
                "playback_duration_s"
            ],
            "video_duration_error_s": video_repeat_plan["duration_error_s"],
            "vial_proxy_frame_count": frame_count,
            "vial_proxy_transition_step": frame_steps[0],
        }
    )
    receipt["timing"].update(
        {
            "capture_hz": 120.0 / capture_stride_steps,
            "pour_duration_steps": [1842, 1440, 1164],
            "pour_end_steps": [1842, 3282, 4446],
            "fill_end_step": 4446,
            "primary_final_capture_step": 4482,
            "executed_physics_steps": 4806,
            "callbacks_expected": 4806,
            "callbacks_observed": 4806,
            "capture_guard_records": [{}] * (2 * frame_count),
        }
    )
    receipt["semantics"].update(
        {
            "pairs": [
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
                    "visible_stream_tanks": list(rows[index]["visible_stream_tanks"]),
                }
                for index, step in enumerate(frame_steps)
            ],
            "stage_mask_sha256": [mask_hashes[index] for index in stage_indices],
            "frame_semantic_paths": semantic_paths,
            "frame_semantic_sha256": semantic_hashes,
            "paired_capture_contract": {"state_records": [{}] * frame_count},
            "transparent_annotation_override": {
                "transitions": [{}] * (3 * frame_count)
            },
        }
    )
    return receipt


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
        "copied_primvar_names": [],
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

    changed = deepcopy(receipt)
    del changed["count_driven_liquid_body"]["stage_geometry"][0][
        "volume_proof_coordinate_path"
    ]
    assert "count_driven_body_contract" in paint_ball_emitter._validate_full_run_contract(
        changed
    )

    changed = deepcopy(receipt)
    changed["count_driven_liquid_body"]["copied_primvar_names"] = [
        "primvars:displayColor",
        "primvars:displayOpacity",
    ]
    assert "count_driven_body_contract" in paint_ball_emitter._validate_full_run_contract(
        changed
    )


def test_count_driven_contract_derives_recipe8_stages_and_particle_volume_from_plan():
    receipt = _count_driven_body_contract_fixture()
    particle_volume_m3 = 0.011272866937444935e-6
    expected_rows = (
        (1842, 20.0, 1774),
        (3282, 40.0, 3548),
        (4482, 60.0, 5322),
    )
    receipt["plan"] = {
        "pours": [
            {
                "stage_capture_step": step,
                "programmed_volume_ml": 20.0,
                "cumulative_particle_count": count,
            }
            for step, _volume, count in expected_rows
        ],
        "render_treatment": {
            "particle_volume_m3": particle_volume_m3,
        },
    }
    body = receipt["count_driven_liquid_body"]
    body["particle_volume_m3"] = particle_volume_m3
    body["stage_geometry"] = [
        {
            "physics_step": step,
            "contained_particle_count": count,
            "particle_volume_m3": particle_volume_m3,
            "target_volume_m3": count * particle_volume_m3,
            "target_height_m": (
                count
                * particle_volume_m3
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
                    * particle_volume_m3
                    / COUNT_DRIVEN_DODECAGON_AREA_M2,
                ],
            ],
            "post_transform_signed_volume_m3": count * particle_volume_m3,
            "post_transform_volume_residual_m3": 0.0,
            "volume_proof_coordinate_path": (
                "local_points_vec3f_then_inherited_world_transform"
            ),
            "points_sha256": "1" * 64,
            "extent_sha256": "2" * 64,
        }
        for step, _volume, count in expected_rows
    ]
    receipt["containment"]["capture_records"] = [
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
        for step, _volume, count in expected_rows
    ]
    control = receipt["semantics"]["hidden_particle_control"]
    control.update(
        physics_step=1842,
        emitted_particle_count=1774,
        inside_vial_interior_count=1774,
        body_only_capture_step=1842,
    )

    assert paint_ball_emitter._validate_count_driven_body_contract(receipt) == ()


def test_full_run_contract_accepts_hashed_stage_copies_of_video_frames():
    receipt = _count_driven_body_contract_fixture()
    receipt["render"]["stage_frame_paths"] = [
        f"stages/stage-{index + 1}.png" for index in range(3)
    ]

    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()


def test_full_run_contract_derives_recipe8_denominators_from_bound_plan():
    receipt = _recipe8_full_run_contract_fixture()

    assert len(receipt["render"]["frame_physics_steps"]) == 18
    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()


def test_affected_recipe8_receipt_binds_six_stream_segments_and_sources():
    receipt = _recipe8_full_run_contract_fixture(
        affected=True, particle_count_per_pour=1774
    )

    assert len(receipt["stream_render"]["records"]) == 6
    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()

    wrong_source = deepcopy(receipt)
    residual_frame = next(
        pair
        for pair in wrong_source["semantics"]["pairs"]
        if pair["physics_step"] == 12
    )
    residual_frame["visible_stream_tanks"] = ["t200"]
    assert "full_run_semantic_contract" in paint_ball_emitter._validate_full_run_contract(
        wrong_source
    )

    missing_segment = deepcopy(receipt)
    missing_segment["stream_render"]["records"].pop(0)
    assert "full_run_stream_contract" in paint_ball_emitter._validate_full_run_contract(
        missing_segment
    )


def test_recipe8_receipt_accepts_bound_solver16_and_capacity5322():
    receipt = _recipe8_full_run_contract_fixture(
        affected=True, particle_count_per_pour=1774
    )
    runtime_configuration = paint_ball_emitter.build_runtime_particle_configuration(
        solver_position_iterations=16,
        max_particles=5322,
    )
    receipt["arguments"].update(solver_position_iterations=16)
    receipt["particle_system"].update(
        {
            "runtime_configuration": runtime_configuration,
            "max_velocity_m_s": MAX_VELOCITY_M_S,
            "smoothing": True,
            "smoothing_enabled": True,
            "smoothing_strength": 1.0,
            "anisotropy": True,
            "anisotropy_enabled": True,
            "anisotropy_scale": 1.0,
            "ccd": True,
            "particle_contact_offset_m": runtime_configuration[
                "particle_contact_offset_m"
            ],
            "fluid_rest_offset_m": runtime_configuration["fluid_rest_offset_m"],
            "contact_offset_m": runtime_configuration["contact_offset_m"],
            "rest_offset_m": runtime_configuration["rest_offset_m"],
            "solid_rest_offset_m": runtime_configuration["solid_rest_offset_m"],
            "solver_position_iteration_count": 16,
            "solver_position_iteration_count_requested": 16,
            "max_neighborhood": runtime_configuration["max_neighborhood"],
            "fabric_enabled": False,
            "pbd_material_readback": {
                "density_kg_m3": 1000.0,
                "friction": 0.1,
                "cohesion": 0.0,
                "damping": 0.0,
                "surface_tension": 0.0,
                "viscosity": 0.0,
                "vorticity_confinement": 0.0,
            },
        }
    )
    receipt.setdefault("point_instancer", {}).update(
        {
            "path": "/FillStation/FluidmixF2/FluidParticles/Shared",
            "max_particles": 5322,
            "ids_stable": True,
            "simulation_points_authored": True,
            "mass_density_readback": {
                "mass_api_present": False,
                "particle_mass_argument_kg": None,
                "total_mass_attr_kg": None,
                "total_mass_attr_authored": False,
                "density_attr_kg_m3": 1000.0,
                "density_attr_authored": True,
                "nominal_per_particle_mass_kg": (
                    runtime_configuration["density_kg_m3"]
                    * (2.0 * runtime_configuration["fluid_rest_offset_m"]) ** 3
                ),
            },
        }
    )
    receipt["gpu_simulation"] = {
        "position_writeback_initial_readback": False,
        "velocity_writeback_initial_readback": False,
        "fabric_initial_readback": True,
        "position_writeback_enabled_readback": True,
        "velocity_writeback_enabled_readback": True,
        "fabric_disabled_readback": True,
        "position_writeback_restored_readback": False,
        "velocity_writeback_restored_readback": False,
        "fabric_restored_readback": True,
    }

    failures = validate_paint_ball_receipt(receipt)
    assert "runtime_particle_configuration" not in failures
    assert "point_instancer_contract" not in failures
    assert "floor_tunneling_mitigation" in failures

    receipt["particle_system"].update(
        {
            "collider_collision_readbacks": [
                {
                    "path": "/FillStation/FluidmixF2/Colliders/Floor",
                    "contact_offset_authored": True,
                    "rest_offset_authored": True,
                    "contact_offset_m": 0.025,
                    "rest_offset_m": 0.0,
                }
            ],
            "floor_tunneling_mitigation": {
                "documented_source": (
                    "https://docs.omniverse.nvidia.com/kit/docs/"
                    "omni_physics/107.0/dev_guide/rigid_bodies_"
                    "articulations/collision.html"
                ),
                "observed_failure_receipt_sha256": (
                    "5ff739bf2fe49615c8fe174cb3400031a88cdc8bc2fbc851c92ab504b62547e3"
                ),
                "floor_contact_offset_m": 0.025,
                "floor_rest_offset_m": 0.0,
                "maximum_capped_displacement_per_step_m": 2.5 / 120.0,
                "contact_offset_exceeds_capped_step": True,
                "floor_top_unchanged_m": 0.9762065645,
                "readback_passed": True,
            },
        }
    )
    assert "floor_tunneling_mitigation" not in validate_paint_ball_receipt(
        receipt
    )
    assert "wall_tunneling_mitigation" in validate_paint_ball_receipt(receipt)

    receipt["particle_system"]["collider_collision_readbacks"].extend(
        {
            "path": f"/FillStation/FluidmixF2/Colliders/Wall{index:02d}",
            "contact_offset_authored": True,
            "rest_offset_authored": True,
            "contact_offset_m": 0.025,
            "rest_offset_m": 0.0,
        }
        for index in range(32)
    )
    receipt["particle_system"]["wall_tunneling_mitigation"] = {
        "documented_source": (
            "https://docs.omniverse.nvidia.com/kit/docs/"
            "omni_physics/107.0/dev_guide/rigid_bodies_"
            "articulations/collision.html"
        ),
        "observed_failure_receipt_sha256": (
            "3ea78b1d91eb64b2dab30db7d0db9182d416d35abb851c2f6489d1ebf26e34da"
        ),
        "wall_count": 32,
        "wall_contact_offset_m": 0.025,
        "wall_rest_offset_m": 0.0,
        "maximum_capped_displacement_per_step_m": 2.5 / 120.0,
        "contact_offset_exceeds_capped_step": True,
        "wall_thickness_unchanged_m": 0.001,
        "readback_passed": True,
    }
    assert "wall_tunneling_mitigation" not in validate_paint_ball_receipt(
        receipt
    )
    assert "render_product_update_gating" in validate_paint_ball_receipt(
        receipt
    )

    frame_count = receipt["render"]["frame_count"]
    physics_step_count = receipt["timing"]["callbacks_observed"]
    rgb_pass_count = frame_count + 2
    semantic_pass_count = frame_count
    explicit_pass_count = 1 + rgb_pass_count + semantic_pass_count
    receipt["render"]["render_product_update_gating"] = {
        "documented_source": (
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/"
            "replicator_tutorials/tutorial_replicator_sdg_workflows.html"
        ),
        "initial_disabled_readback": True,
        "final_disabled_readback": True,
        "current_updates_enabled": False,
        "enabled_only_for_explicit_render_passes": True,
        "physics_updates_while_disabled": physics_step_count,
        "enable_request_count": explicit_pass_count,
        "disable_request_count": explicit_pass_count + 2,
        "explicit_render_pass_count": explicit_pass_count,
        "transitions": (
            [
                {
                    "phase": "initial_after_create",
                    "requested_updates_enabled": False,
                    "updates_enabled_before": False,
                    "updates_enabled_after": False,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
            ]
            + [
                {
                    "phase": "fixture_explicit_pass",
                    "requested_updates_enabled": enabled,
                    "updates_enabled_before": not enabled,
                    "updates_enabled_after": enabled,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
                for _ in range(explicit_pass_count)
                for enabled in (True, False)
            ]
            + [
                {
                    "phase": "final_before_acceptance",
                    "requested_updates_enabled": False,
                    "updates_enabled_before": False,
                    "updates_enabled_after": False,
                    "readback_matches_request": True,
                    "set_and_readback_seconds": 0.001,
                }
            ]
        ),
        "explicit_render_passes": (
            [
                {
                    "phase": "warmup",
                    "timing_phase": "warmup_app_update",
                    "elapsed_seconds": 0.01,
                    "passed": True,
                    "disabled_after": True,
                }
            ]
            + [
                {
                    "phase": "rgb",
                    "timing_phase": "capture_rgb_orchestrator",
                    "elapsed_seconds": 0.01,
                    "passed": True,
                    "disabled_after": True,
                }
            ]
            * rgb_pass_count
            + [
                {
                    "phase": "semantic",
                    "timing_phase": "capture_semantic_orchestrator",
                    "elapsed_seconds": 0.01,
                    "passed": True,
                    "disabled_after": True,
                }
            ]
            * semantic_pass_count
        ),
    }
    receipt["render"]["phase_timings_s"] = {
        "warmup_app_update": {"count": 1, "total": 1.0, "maximum": 1.0},
        "physics_app_update": {
            "count": physics_step_count,
            "total": 2.0,
            "maximum": 0.01,
        },
        "diffusion": {
            "count": physics_step_count,
            "total": 3.0,
            "maximum": 0.01,
        },
        "palette_mapping": {
            "count": physics_step_count,
            "total": 0.2,
            "maximum": 0.001,
        },
        "prototype_authoring": {
            "count": physics_step_count,
            "total": 0.3,
            "maximum": 0.001,
        },
        "capture_rgb_orchestrator": {
            "count": rgb_pass_count,
            "total": 4.0,
            "maximum": 0.1,
        },
        "capture_semantic_orchestrator": {
            "count": semantic_pass_count,
            "total": 4.0,
            "maximum": 0.1,
        },
        "capture_artifact_io": {
            "count": frame_count,
            "total": 2.0,
            "maximum": 0.1,
        },
    }
    _attach_valid_render_product_gating(
        receipt,
        physics_step_count=physics_step_count,
        frame_count=frame_count,
    )
    assert "render_product_update_gating" not in validate_paint_ball_receipt(
        receipt
    )

    stale_final_readback = deepcopy(receipt)
    stale_final_readback["render"]["render_product_update_gating"][
        "final_disabled_readback"
    ] = False
    assert "render_product_update_gating" in validate_paint_ball_receipt(
        stale_final_readback
    )

    missing_physics_timing = deepcopy(receipt)
    missing_physics_timing["render"]["phase_timings_s"]["physics_app_update"][
        "count"
    ] -= 1
    assert "render_product_update_gating" in validate_paint_ball_receipt(
        missing_physics_timing
    )

    enabled_outside_explicit_transition = deepcopy(receipt)
    transition_gating = enabled_outside_explicit_transition["render"][
        "render_product_update_gating"
    ]
    first_explicit_transition_index = transition_gating["rotation_count"] + 1
    transition_gating["transitions"][first_explicit_transition_index][
        "updates_enabled_before"
    ] = True
    assert "render_product_update_gating" in validate_paint_ball_receipt(
        enabled_outside_explicit_transition
    )

    # The frozen sparse schedule has fewer than the 48 captures required for
    # a render-product rotation.  Rotation release is exercised by the dense
    # generic fixture; this Recipe-8 fixture must bind the real zero-rotation
    # denominator used by C15.
    assert receipt["render"]["render_product_update_gating"]["rotation_count"] == 0
    assert receipt["render"]["render_product_update_gating"]["rotations"] == []

    wrong_solver = deepcopy(receipt)
    wrong_solver["particle_system"]["solver_position_iteration_count"] = 4
    assert "runtime_particle_configuration" in validate_paint_ball_receipt(
        wrong_solver
    )

    wrong_capacity = deepcopy(receipt)
    wrong_capacity["point_instancer"]["max_particles"] = 4096
    assert "point_instancer_contract" in validate_paint_ball_receipt(wrong_capacity)


def test_recipe8_full_run_contract_accepts_reviewed_pathtracing16():
    receipt = _recipe8_full_run_contract_fixture()
    receipt["render"]["rt_subframes"] = 16
    receipt["render"].update(
        pathtracing_total_spp_requested=16,
        pathtracing_total_spp_readback=16,
        pathtracing_spp_requested=16,
        pathtracing_spp_readback=16,
    )

    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()


def test_offline_validator_binds_safe_runtime_schedule_not_plan_cadence():
    receipt = _recipe8_full_run_contract_fixture(capture_stride_steps=312)
    expected = paint_ball_emitter.build_runtime_capture_schedule(
        receipt["plan"], capture_stride_steps=312
    )

    assert receipt["render"]["frame_count"] == len(expected["captures"])
    assert receipt["render"]["frame_physics_steps"] == [
        row["physics_step"] for row in expected["captures"]
    ]
    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()

    drifted = deepcopy(receipt)
    drifted["runtime_capture_schedule"]["capture_stride_steps"] = 12
    assert "full_run_capture_schedule_contract" in (
        paint_ball_emitter._validate_full_run_contract(drifted)
    )


def test_offline_validator_rejects_c14_low_fps_sparse_receipt():
    receipt = _recipe8_full_run_contract_fixture(capture_stride_steps=312)
    receipt["arguments"]["video_fps"] = 120 / 312

    assert receipt["arguments"]["video_fps"] == 120 / 312
    assert "full_run_timing_contract" in (
        paint_ball_emitter._validate_full_run_contract(receipt)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda render: render["video_frame_repeat_counts"].__setitem__(0, 1),
        lambda render: render["video_frame_source_capture_indices"].__setitem__(0, 1),
        lambda render: render["video_source_capture_onset_frame_indices"].__setitem__(
            1, 1
        ),
        lambda render: render.pop("video_encoded_frame_time_rule"),
        lambda render: render.update(video_playback_duration_s=40.3),
        lambda render: render.update(video_duration_error_s=0.25),
        lambda render: render.update(video_decoded_frame_count=19),
    ),
)
def test_offline_validator_rejects_sparse_video_timeline_drift(mutation):
    receipt = _recipe8_full_run_contract_fixture(capture_stride_steps=312)
    mutation(receipt["render"])

    assert "full_run_video_contract" in (
        paint_ball_emitter._validate_full_run_contract(receipt)
    )


def test_recipe8_full_run_contract_rejects_pathtracing_sampling_readback_drift():
    receipt = _recipe8_full_run_contract_fixture()
    receipt["render"]["pathtracing_total_spp_readback"] = 32

    assert "full_run_rgb_contract" in paint_ball_emitter._validate_full_run_contract(
        receipt
    )


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


def test_full_run_contract_accepts_direct_physics_timing_owner():
    receipt = _full_run_contract_fixture()
    receipt["render"]["render_product_update_gating"] = {
        "mode": "persistent_enabled_direct_physics"
    }
    receipt["timing"].pop("one_callback_per_app_update")
    receipt["timing"].update(
        {
            "simulation_owner": "SimulationManager.step",
            "physics_step_method": (
                "SimulationManager.step(steps=1, update_fabric=False)"
            ),
            "one_callback_per_direct_step": True,
            "physics_loop_app_update_count": 0,
            "capture_direct_physics_steps_advanced": 0,
            "capture_simulation_time_advanced": False,
        }
    )

    assert paint_ball_emitter._validate_full_run_contract(receipt) == ()

    advanced_during_capture = deepcopy(receipt)
    advanced_during_capture["timing"][
        "capture_direct_physics_steps_advanced"
    ] = 1
    assert "full_run_timing_contract" in paint_ball_emitter._validate_full_run_contract(
        advanced_during_capture
    )


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
    assert PAINT_BALL_RECEIPT_SCHEMA_VERSION == 20
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
    _attach_valid_render_product_gating(
        receipt, physics_step_count=120, frame_count=1
    )
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


def test_10ml_stage_uses_state_height_and_skips_hidden_visible_height_gate():
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

    assert completed["volume_gate"]["visible_height_policy"]["required"] is False
    assert completed["volume_gate"]["visible_passed"] is None
    assert completed["volume_gate"]["visible_mask_evidence"] == {
        **completed["volume_gate"]["visible_mask_evidence"],
        "required": False,
        "mask_evaluated": False,
        "status": "not_required_at_or_below_opaque_base_cutoff",
    }
    assert completed["volume_gate"]["passed"] is True


def test_recipe5_visible_height_starts_only_above_frozen_opaque_base_cutoff():
    center_xy_m = (5.2324003281, -1.9811999798)
    bottom_m = 0.9762065645
    mouth_m = 1.1762601692
    circumradius_m = 0.0132029216
    inner_apothem_m = 0.012753042956
    render_radius_m = 0.00125
    recipe5_stages = (
        (1412, 10.0, 640, 0.018378252030617992),
        (2622, 20.0, 1280, 0.036756504061235984),
        (4004, 30.0, 1920, 0.055134756091853976),
    )
    completed = []
    for physics_step, volume_ml, count, expected_height_m in recipe5_stages:
        positions = np.column_stack(
            (
                np.full(count, center_xy_m[0]),
                np.full(count, center_xy_m[1]),
                np.linspace(
                    bottom_m + render_radius_m,
                    bottom_m + expected_height_m - render_radius_m,
                    count,
                ),
            )
        )
        containment = containment_evidence(
            positions,
            lower_bounds=(center_xy_m[0] - 1.0, center_xy_m[1] - 1.0, bottom_m),
            upper_bounds=(center_xy_m[0] + 1.0, center_xy_m[1] + 1.0, mouth_m),
            cylindrical_center_xy=center_xy_m,
            inner_radius_m=inner_apothem_m,
            polygon_center_xy=center_xy_m,
            polygon_apothem_m=inner_apothem_m,
            polygon_wall_count=12,
        )
        record = build_capture_record(
            physics_step=physics_step,
            phase=f"recipe5-{int(volume_ml)}ml-stage",
            containment=containment,
            prototype_count=15,
            instancer_path="/FillStation/FluidmixF2/FluidParticles/Shared",
            emitted_particle_count=count,
            programmed_cumulative_volume_ml=volume_ml,
            expected_cumulative_particle_count=count,
            cross_section_area_m2=0.0005229514163,
            height_tolerance_m=0.006,
            render_radius_m=render_radius_m,
            point_instancer_scales=np.ones((count, 3)),
            smoothed_render_containment=containment,
        )
        roi = paint_ball_emitter.projected_vial_geometry_roi_mask(
            (480, 640),
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_m,
            mouth_z_m=mouth_m,
            circumradius_m=circumradius_m,
        )
        mask = np.zeros_like(roi)
        if volume_ml > 20.0:
            surface_row = int(
                np.ceil(
                    paint_ball_emitter.projected_top_dodecagon_min_y_px(
                        expected_height_m,
                        center_xy_m=center_xy_m,
                        bottom_z_m=bottom_m,
                        circumradius_m=circumradius_m,
                    )
                )
            )
            cutoff_row = int(
                np.floor(
                    paint_ball_emitter.projected_top_dodecagon_min_y_px(
                        0.036756504061235984,
                        center_xy_m=center_xy_m,
                        bottom_z_m=bottom_m,
                        circumradius_m=circumradius_m,
                    )
                )
            )
            mask[surface_row : cutoff_row + 1] = roi[surface_row : cutoff_row + 1]
        completed.append(
            attach_visible_volume_evidence(
                record,
                mask,
                center_xy_m=center_xy_m,
                bottom_z_m=bottom_m,
                mouth_z_m=mouth_m,
                circumradius_m=circumradius_m,
            )
        )

    for row in completed[:2]:
        gate = row["volume_gate"]
        assert gate["state_passed"] is True
        assert gate["visible_height_policy"]["required"] is False
        assert gate["visible_passed"] is None
        assert gate["visible_mask_evidence"]["status"] == (
            "not_required_at_or_below_opaque_base_cutoff"
        )
        assert gate["passed"] is True
        assert paint_ball_emitter._combined_volume_gate_valid(gate) is True

    final_gate = completed[2]["volume_gate"]
    assert final_gate["state_passed"] is True
    assert final_gate["visible_height_policy"] == {
        **final_gate["visible_height_policy"],
        "required": True,
        "opaque_base_cutoff_volume_ml": 20.0,
        "opaque_base_cutoff_height_m": pytest.approx(0.036756504061235984),
        "expected_height_above_cutoff_m": pytest.approx(0.018378252030617992),
    }
    assert final_gate["visible_passed"] is True
    assert final_gate["visible_mask_evidence"]["observed_height_above_cutoff_m"] == pytest.approx(
        0.018378252030617992,
        abs=1.0e-3,
    )
    assert final_gate["passed"] is True
    assert paint_ball_emitter._combined_volume_gate_valid(final_gate) is True

    drifted = deepcopy(final_gate)
    drifted["visible_height_policy"]["opaque_base_cutoff_volume_ml"] = 15.0
    assert paint_ball_emitter._combined_volume_gate_valid(drifted) is False


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
        "SimulationManager.initialize_physics()"
    )
    assert 'receipt["timing"]["timeline_playback_start"] = None' in run_source

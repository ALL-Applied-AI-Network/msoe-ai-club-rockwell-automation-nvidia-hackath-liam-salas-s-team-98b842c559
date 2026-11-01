"""CPU contracts for the Isaac 6.2 F2 runtime integration."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import twin.physx.paint_ball_emitter as runtime_contract
from twin.physx.dispense_residual import (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    build_recipe8_runtime_leading_residual_schedule,
)
from twin.physx.particle_contract import (
    DENSITY_KG_M3,
    FLUID_REST_OFFSET_M,
    MINIMUM_PARTICLE_CAPACITY,
    PARTICLE_CONTACT_OFFSET_M,
    REST_OFFSET_M,
    SOLID_REST_OFFSET_M,
)
from twin.physx.three_pour import build_paint_ball_three_pour_plan
from twin.physx.volume_height_curve import (
    height_bounds_for_volume_ml,
    height_for_volume_ml,
    volume_height_curve_digest,
)


def test_source_state_is_the_single_immutable_stream_and_birth_owner() -> None:
    factory = getattr(runtime_contract, "build_source_state", None)
    assert callable(factory), "production SourceState boundary is missing"

    source = factory(
        scheduled_tank="t400",
        composition=(0.25, 0.75, 0.0),
        palette_prototype_index=7,
        material_path="/FillStation/FluidmixF2/PaletteMaterials/Palette_07",
        pour_event_id="pour-2",
        birth_event_id="birth-17",
        birth_step=811,
    )

    assert source.scheduled_tank == "t400"
    assert source.composition == (0.25, 0.75, 0.0)
    assert source.stream_composition == source.birth_composition
    assert source.palette_prototype_index == 7
    assert source.material_path.endswith("/Palette_07")
    assert source.pour_event_id == "pour-2"
    assert source.birth_event_id == "birth-17"
    assert source.birth_step == 811


def test_runtime_particle_configuration_consumes_task1_and_forbids_mass_override() -> None:
    factory = getattr(runtime_contract, "build_runtime_particle_configuration", None)
    assert callable(factory), "runtime particle configuration boundary is missing"

    config = factory()

    assert config == {
        "spawn_spacing_m": 0.0025,
        "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
        "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
        "contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
        "rest_offset_m": REST_OFFSET_M,
        "solid_rest_offset_m": SOLID_REST_OFFSET_M,
        "density_kg_m3": DENSITY_KG_M3,
        "particle_set_mass_override": None,
        "max_particles": MINIMUM_PARTICLE_CAPACITY,
        "physics_hz": 120,
        "solver_position_iterations": 4,
        "max_neighborhood": 96,
        "max_velocity_m_s": 2.5,
        "ccd": True,
        "smoothing": True,
        "anisotropy": True,
        "fabric_enabled": False,
        "positions_writeback_required": True,
        "velocities_writeback_required": True,
    }


def test_atomic_append_extends_every_usd_and_composition_row_with_stable_ids() -> None:
    append = getattr(runtime_contract, "append_runtime_particle_state", None)
    assert callable(append), "atomic runtime append boundary is missing"
    source = runtime_contract.build_source_state(
        scheduled_tank="t200",
        composition=(1.0, 0.0, 0.0),
        palette_prototype_index=0,
        material_path="/FillStation/FluidmixF2/PaletteMaterials/Palette_00",
        pour_event_id="pour-1",
        birth_event_id="birth-1",
        birth_step=1,
    )

    state = append(
        {},
        source_state=source,
        positions=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        velocities=((0.0, 0.0, -0.1), (0.0, 0.0, -0.1)),
    )

    assert state["ids"] == [0, 1]
    assert state["simulation_points"] == state["positions"]
    assert state["proto_indices"] == [0, 0]
    assert state["scheduled_sources"] == ["t200", "t200"]
    assert state["birth_compositions"] == [[1.0, 0.0, 0.0]] * 2
    assert state["current_compositions"] == state["birth_compositions"]
    assert state["birth_steps"] == [1, 1]
    assert state["birth_event_ids"] == ["birth-1", "birth-1"]
    assert len(state["orientations"]) == len(state["scales"]) == 2

    snapshot = {key: list(value) for key, value in state.items()}
    with pytest.raises(ValueError, match="length mismatch"):
        append(
            state,
            source_state=source,
            positions=((7.0, 8.0, 9.0),),
            velocities=(),
        )
    assert all(np.array_equal(state[key], snapshot[key]) for key in state)


def test_failed_snapshot_saves_all_runtime_arrays_before_identity_assertion(tmp_path) -> None:
    persist = getattr(runtime_contract, "persist_runtime_particle_snapshot", None)
    assert callable(persist), "failure-first array persistence boundary is missing"
    source = runtime_contract.build_source_state(
        scheduled_tank="t100",
        composition=(0.0, 0.0, 1.0),
        palette_prototype_index=2,
        material_path="/FillStation/FluidmixF2/PaletteMaterials/Palette_02",
        pour_event_id="pour-3",
        birth_event_id="birth-3",
        birth_step=1312,
    )
    state = runtime_contract.append_runtime_particle_state(
        {},
        source_state=source,
        positions=((1.0, 2.0, 3.0),),
        velocities=((0.0, 0.0, -0.1),),
    )

    with pytest.raises(RuntimeError, match="stable particle IDs"):
        persist(tmp_path, "stage-t100", state, expected_ids=(99,))

    snapshot_dir = tmp_path / "stage-t100"
    for name in (
        "simulation_points",
        "positions",
        "velocities",
        "orientations",
        "scales",
        "ids",
        "proto_indices",
        "birth_compositions",
        "current_compositions",
    ):
        assert (snapshot_dir / f"{name}.npy").is_file(), name
    assert (snapshot_dir / "snapshot.json").is_file()
    assert (snapshot_dir / "failure.json").is_file()


def test_runtime_capture_schedule_has_every_in_pour_interval_and_postfill_anchors() -> None:
    build = getattr(runtime_contract, "build_runtime_capture_schedule", None)
    assert callable(build), "runtime capture schedule boundary is missing"
    plan = build_paint_ball_three_pour_plan()

    schedule = build(
        plan,
        cap_delay_anchor={
            "validated": True,
            "delay_s": 2.5,
            "source_sha256": "a" * 64,
        },
    )

    assert schedule["physics_hz"] == 120
    assert schedule["capture_hz"] == 10
    assert schedule["capture_stride_steps"] == 12
    assert schedule["final_plus_0_3_step"] == plan["timing"]["fill_end_step"] + 36
    assert schedule["final_plus_3_0_step"] == plan["timing"]["fill_end_step"] + 360
    assert schedule["validated_cap_delay_step"] == plan["timing"]["fill_end_step"] + 300
    rows = schedule["captures"]
    for pour in plan["pours"]:
        interior = [
            row["physics_step"]
            for row in rows
            if row["kind"] == "in_pour" and row["pour_ordinal"] == pour["ordinal"]
        ]
        assert len(interior) >= 2
        assert all(
            pour["start_step"] < step < pour["end_step"] for step in interior
        )
        assert max(b - a for a, b in zip(interior, interior[1:])) <= 12
        positions = [
            row["evidence_position"]
            for row in rows
            if row["kind"] == "in_pour" and row["pour_ordinal"] == pour["ordinal"]
        ]
        assert positions.count("first") == 1
        assert positions.count("middle") == 1
        assert positions.count("last") == 1
    assert all(
        row["visible_stream_tanks"] == [row["scheduled_tank"]]
        for row in rows
        if row["kind"] == "in_pour"
    )
    assert all(
        row["visible_stream_tanks"] == []
        for row in rows
        if row["kind"] != "in_pour"
        and row["physics_step"] != plan["timing"]["fill_end_step"]
    )
    final_boundary = next(
        row
        for row in rows
        if row["physics_step"] == plan["timing"]["fill_end_step"]
    )
    assert final_boundary["visible_stream_tanks"] == [plan["pours"][-1]["tank_id"]]


def test_recipe8_runtime_capture_schedule_has_exact_401_unique_frames() -> None:
    plan = {
        "timing": {
            "physics_hz": 120,
            "capture_hz": 10,
            "fill_end_step": 4446,
        },
        "pours": [
            {
                "ordinal": 1,
                "tank_id": "t200",
                "start_step": 1,
                "end_step": 1842,
            },
            {
                "ordinal": 2,
                "tank_id": "t400",
                "start_step": 1843,
                "end_step": 3282,
            },
            {
                "ordinal": 3,
                "tank_id": "t100",
                "start_step": 3283,
                "end_step": 4446,
            },
        ],
    }

    schedule = runtime_contract.build_runtime_capture_schedule(plan)
    rows = schedule["captures"]
    steps = [row["physics_step"] for row in rows]

    assert len(rows) == len(set(steps)) == 401
    assert steps == sorted(steps)
    assert steps[:3] == [12, 24, 36]
    assert 1842 in steps
    assert [step for step in steps if 1842 < step <= 3282][:3] == [1854, 1866, 1878]
    assert [step for step in steps if 3282 < step <= 4446][:3] == [3294, 3306, 3318]
    assert len([step for step in steps if step <= 4446]) == 371
    assert [step for step in steps if step > 4446] == list(range(4458, 4807, 12))
    assert next(row for row in rows if row["physics_step"] == 4482)["kind"] == "final_plus_0_3"
    assert rows[-1]["physics_step"] == 4806
    assert rows[-1]["kind"] == "final_plus_3_0"


@pytest.mark.parametrize(
    "residual_volume_ml,expected_segment_count",
    ((0.0, 3), (1.0, 6)),
)
def test_recipe8_capture_streams_follow_each_nonempty_source_segment(
    residual_volume_ml: float, expected_segment_count: int
) -> None:
    plan = build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=residual_volume_ml,
        effective_particle_volume_ml=MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    )
    plan["timing"].update(
        {
            "capture_hz": 10,
            "primary_final_capture_step": 4482,
            "primary_final_offset_s": 0.3,
        }
    )
    for pour in plan["pours"]:
        pour["particle_count"] = pour["programmed_particle_count"]

    rows = runtime_contract.build_runtime_capture_schedule(plan)["captures"]
    segment_rows = [
        segment
        for pour in plan["pours"]
        for segment in pour["emission_segments"]
        if segment["particle_count"] > 0
    ]

    assert len(segment_rows) == expected_segment_count
    for pour in plan["pours"]:
        in_pour = [
            row
            for row in rows
            if row["kind"] == "in_pour"
            and row["pour_ordinal"] == pour["ordinal"]
        ]
        assert len(in_pour) > 2
        assert all(len(row["visible_stream_tanks"]) == 1 for row in in_pour)
    for segment in segment_rows:
        captured = [
            row
            for row in rows
            if row["kind"] == "in_pour"
            and row["pour_ordinal"]
            == next(
                pour["ordinal"]
                for pour in plan["pours"]
                if segment in pour["emission_segments"]
            )
            and segment["start_step"] <= row["physics_step"] <= segment["end_step"]
        ]
        assert len(captured) >= 2
        assert all(
            row["visible_stream_tanks"] == [segment["source_tank_id"]]
            for row in captured
        )
    assert all(
        row["visible_stream_tanks"] == []
        for row in rows
        if row["kind"] != "in_pour"
        and row["physics_step"] != plan["timing"]["fill_end_step"]
    )
    final_segments = [
        segment
        for segment in plan["pours"][-1]["emission_segments"]
        if segment["particle_count"] > 0
    ]
    final_source = (
        final_segments[-1]["source_tank_id"]
        if final_segments
        else plan["pours"][-1]["tank_id"]
    )
    final_boundary = next(
        row
        for row in rows
        if row["physics_step"] == plan["timing"]["fill_end_step"]
    )
    assert final_boundary["visible_stream_tanks"] == [final_source]


def test_safe_sparse_schedule_preserves_exact_semantic_anchors() -> None:
    plan = build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=1.0,
        effective_particle_volume_ml=MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    )
    plan["timing"].update(
        {
            "capture_hz": 10,
            "primary_final_capture_step": 4482,
            "primary_final_offset_s": 0.3,
        }
    )
    for pour in plan["pours"]:
        pour["particle_count"] = pour["programmed_particle_count"]

    schedule = runtime_contract.build_runtime_capture_schedule(
        plan, capture_stride_steps=312
    )
    rows = schedule["captures"]
    by_step = {row["physics_step"]: row for row in rows}
    transition_steps = {
        step
        for pour in plan["pours"]
        for left, right in zip(
            pour["emission_segments"], pour["emission_segments"][1:]
        )
        for step in (left["end_step"], right["start_step"])
    }

    assert schedule["physics_hz"] == 120
    assert schedule["capture_stride_steps"] == 312
    assert schedule["capture_hz"] == 120 / 312
    assert transition_steps == {92, 93, 1914, 1915, 3340, 3341}
    assert transition_steps.issubset(by_step)
    assert all(
        by_step[step]["semantic_anchor"] == "source_segment_boundary"
        for step in transition_steps
    )
    assert {pour["end_step"] for pour in plan["pours"]}.issubset(by_step)
    assert schedule["final_plus_0_3_step"] in by_step
    assert schedule["final_plus_3_0_step"] in by_step
    assert all(
        sum(
            row["kind"] == "in_pour"
            and row["pour_ordinal"] == pour["ordinal"]
            for row in rows
        )
        >= 3
        for pour in plan["pours"]
    )
    assert all(
        sum(
            row["kind"] == "in_pour"
            and row["pour_ordinal"] == pour["ordinal"]
            and row.get("source_segment_ordinal") == segment["ordinal"]
            for row in rows
        )
        >= 2
        for pour in plan["pours"]
        for segment in pour["emission_segments"]
        if segment["particle_count"] > 0
    )


def test_sparse_capture_video_plan_preserves_real_timeline_at_10_fps() -> None:
    build = getattr(runtime_contract, "build_time_preserving_video_repeat_plan", None)
    assert callable(build), "time-preserving video repeat-plan boundary is missing"
    plan = build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=0.0,
        effective_particle_volume_ml=MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    )
    plan["timing"].update(
        {
            "capture_hz": 10,
            "primary_final_capture_step": 4482,
            "primary_final_offset_s": 0.3,
        }
    )
    for pour in plan["pours"]:
        pour["particle_count"] = pour["programmed_particle_count"]
    schedule = runtime_contract.build_runtime_capture_schedule(
        plan, capture_stride_steps=312
    )
    frame_steps = [row["physics_step"] for row in schedule["captures"]]

    repeat_plan = build(
        frame_steps,
        physics_hz=120,
        playback_fps=10.0,
    )

    assert len(frame_steps) == len(repeat_plan["repeat_counts"]) == 19
    assert repeat_plan["source_frame_steps"] == frame_steps
    assert repeat_plan["video_frame_count"] == 401
    assert sum(repeat_plan["repeat_counts"]) == 401
    assert min(repeat_plan["repeat_counts"]) >= 1
    assert repeat_plan["source_capture_indices"] == [
        capture_index
        for capture_index, repeat_count in enumerate(repeat_plan["repeat_counts"])
        for _ in range(repeat_count)
    ]
    assert set(repeat_plan["source_capture_indices"]) == set(range(19))
    assert repeat_plan["source_duration_s"] == pytest.approx(4806 / 120)
    assert repeat_plan["playback_duration_s"] == pytest.approx(401 / 10)
    assert abs(repeat_plan["duration_error_s"]) <= 0.1
    assert repeat_plan["maximum_absolute_onset_error_s"] <= 0.1 + 1.0e-12
    assert repeat_plan["source_capture_onset_frame_indices"][0] == 0
    assert repeat_plan["source_capture_onset_frame_indices"][-1] == 400
    assert repeat_plan["source_capture_onset_frame_indices"][
        frame_steps.index(312)
    ] == 26
    assert repeat_plan["source_capture_indices"][26] == frame_steps.index(312)
    assert repeat_plan["source_capture_onset_times_s"][-1] == pytest.approx(
        4806 / 120
    )


def test_sparse_capture_video_plan_keeps_adjacent_source_states() -> None:
    build = getattr(runtime_contract, "build_time_preserving_video_repeat_plan", None)
    assert callable(build), "time-preserving video repeat-plan boundary is missing"

    repeat_plan = build((12, 92, 93, 312), physics_hz=120, playback_fps=10.0)

    assert repeat_plan["video_frame_count"] == 26
    assert len(repeat_plan["repeat_counts"]) == 4
    assert min(repeat_plan["repeat_counts"]) >= 1
    assert set(repeat_plan["source_capture_indices"]) == {0, 1, 2, 3}
    assert repeat_plan["source_capture_onset_frame_indices"] == [0, 7, 8, 25]
    assert repeat_plan["source_capture_indices"].index(1) == 7
    assert repeat_plan["source_capture_indices"].index(2) == 8
    assert repeat_plan["source_capture_indices"].index(3) == 25
    assert repeat_plan["source_capture_onset_times_s"] == pytest.approx(
        [0.05, 0.75, 0.85, 2.55]
    )
    assert repeat_plan["source_capture_onset_errors_s"] == pytest.approx(
        [-0.05, -1 / 60, 0.075, -0.05]
    )
    assert repeat_plan["maximum_absolute_onset_error_s"] == pytest.approx(0.075)
    assert repeat_plan["playback_duration_s"] == pytest.approx(2.6)
    assert repeat_plan["source_duration_s"] == pytest.approx(2.6)


@pytest.mark.parametrize(
    "frame_steps,physics_hz,playback_fps",
    (
        ((), 120, 10.0),
        ((12, 12), 120, 10.0),
        ((12, 11), 120, 10.0),
        ((True,), 120, 10.0),
        ((12,), 0, 10.0),
        ((12,), 120, float("nan")),
    ),
)
def test_sparse_capture_video_plan_rejects_invalid_inputs(
    frame_steps: tuple[object, ...], physics_hz: int, playback_fps: float
) -> None:
    build = getattr(runtime_contract, "build_time_preserving_video_repeat_plan", None)
    assert callable(build), "time-preserving video repeat-plan boundary is missing"

    with pytest.raises((TypeError, ValueError)):
        build(frame_steps, physics_hz=physics_hz, playback_fps=playback_fps)


def test_capture_schedule_anchors_a_short_segment_in_two_frames() -> None:
    plan = {
        "timing": {"physics_hz": 120, "capture_hz": 10, "fill_end_step": 48},
        "pours": [
            {
                "ordinal": 1,
                "tank_id": "t200",
                "start_step": 1,
                "end_step": 48,
                "emission_segments": [
                    {
                        "ordinal": 1,
                        "segment_role": "leading_residual",
                        "source_tank_id": "t100",
                        "particle_count": 1,
                        "start_step": 1,
                        "end_step": 24,
                    },
                    {
                        "ordinal": 2,
                        "segment_role": "current_tank",
                        "source_tank_id": "t200",
                        "particle_count": 3,
                        "start_step": 25,
                        "end_step": 48,
                    },
                ],
            }
        ],
    }

    rows = runtime_contract.build_runtime_capture_schedule(plan)["captures"]
    residue = [
        row for row in rows if row.get("source_segment_ordinal") == 1
    ]

    assert [row["physics_step"] for row in residue] == [12, 24]
    assert [row["semantic_anchor"] for row in residue] == [
        "initial_stream_after_motion_falsifier",
        "source_segment_boundary",
    ]


def test_stream_hidden_volume_gate_uses_geometry_curve_not_cylinder_tolerance() -> None:
    expected_height = height_for_volume_ml(10.0)
    lower, upper = height_bounds_for_volume_ml(10.0)
    radius = FLUID_REST_OFFSET_M
    q05 = 1.0
    q95 = q05 + (expected_height - 2.0 * radius) * 0.90
    containment = {
        "active_particle_count": 640,
        "inside_vial_interior_count": 640,
        "outside_vial_interior_count": 0,
        "expected_in_flight_stream_count": 0,
        "leaked_particle_count": 0,
        "max_abs_bounds_m": [1.0, 1.0, 1.1],
        "leakage_detected": False,
        "inside_conservative_circle_count": 640,
        "exact_polygon_containment": True,
        "finite_state": True,
        "contained_z_min_m": q05,
        "contained_z_q05_m": q05,
        "contained_z_q95_m": q95,
        "contained_z_max_m": q95,
    }

    capture = runtime_contract.build_capture_record(
        physics_step=1412,
        phase="t200-stage",
        containment=containment,
        prototype_count=15,
        instancer_path="/FillStation/FluidmixF2/FluidParticles/Shared",
        emitted_particle_count=640,
        programmed_cumulative_volume_ml=10.0,
        expected_cumulative_particle_count=640,
        cross_section_area_m2=123.0,
        height_tolerance_m=0.5,
        render_radius_m=radius,
        point_instancer_scales=np.ones((640, 3)),
        smoothed_render_containment=containment,
    )

    gate = capture["volume_gate"]
    assert gate["height_gate_model"] == "geometry_curve_h_95_105_volume"
    assert gate["volume_height_curve_sha256"] == volume_height_curve_digest()
    assert gate["expected_height_m"] == pytest.approx(expected_height)
    assert gate["height_lower_bound_m"] == pytest.approx(lower)
    assert gate["height_upper_bound_m"] == pytest.approx(upper)
    assert gate["state_passed"] is True


def test_render_stream_surface_uses_robust_contained_particle_quantile() -> None:
    estimate = getattr(runtime_contract, "robust_contained_surface_z_m", None)
    assert callable(estimate), "contained-surface estimator is missing"
    positions = np.column_stack(
        (
            np.zeros(101),
            np.zeros(101),
            np.concatenate((np.linspace(1.0, 1.1, 100), [9.0])),
        )
    )

    surface = estimate(
        positions,
        bottom_z_m=0.99,
        mouth_z_m=1.2,
        render_radius_m=0.00125,
    )

    assert 1.09 < surface < 1.12
    assert surface != pytest.approx(9.00125)


def test_external_nuisance_profile_is_hash_bound_without_becoming_physics(tmp_path) -> None:
    bind = getattr(runtime_contract, "bind_external_nuisance_receipt", None)
    assert callable(bind), "external nuisance receipt boundary is missing"
    path = tmp_path / "photometric-receipt.json"
    payload = {
        "schema_version": 1,
        "receipt_type": "real_photometric_nuisance_range",
        "status": "measured",
        "photometric_status": "measured",
        "shadow_status": "measured",
        "pooled": {
            "candidate_count": 888,
            "usable_count": 884,
            "shadow_observation_count": 884,
            "percentile_bounds": {"luma": {"p05": 1.0, "p50": 2.0, "p95": 3.0}},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    bound = bind(path, digest)

    assert bound["receipt_sha256"] == digest
    assert bound["status"] == "measured"
    assert bound["usable_count"] == 884
    assert bound["shadow_observation_count"] == 884
    assert bound["physics_parameters_modified"] is False
    assert bound["authoring_status"] == "pending_task5_runtime_readback"

    with pytest.raises(ValueError, match="SHA-256"):
        bind(path, "0" * 64)

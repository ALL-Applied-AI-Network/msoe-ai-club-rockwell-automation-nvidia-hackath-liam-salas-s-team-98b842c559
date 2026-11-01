"""CPU contracts for the real-recipe-8 leading-residue Isaac runner path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli/twin/physx"))
SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence_recipe8",
    ROOT / "cli/twin/physx" / "physx_three_pour_sequence.py",
)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)


# Argument validation needs an existing absolute path; these CPU tests never open it.
STAGE = ROOT / "README.md"


def _args(*extra: str):
    return sequence.build_parser().parse_args(
        ["--out", "/tmp/recipe8", "--stage", str(STAGE), *extra]
    )


@pytest.mark.parametrize("residual_ml,event_count", [(0.0, 371), (1.0, 372)])
def test_recipe8_cli_builds_source_pure_runtime_plan(
    residual_ml: float, event_count: int
) -> None:
    args = _args("--recipe8-residual-volume-ml", str(residual_ml))
    sequence._validate_args(args)

    plan = sequence._runtime_paint_ball_plan(args)

    assert [row["tank_id"] for row in plan["pours"]] == ["t200", "t400", "t100"]
    assert [row["programmed_volume_ml"] for row in plan["pours"]] == [20.0] * 3
    assert [row["duration_steps"] for row in plan["pours"]] == [1842, 1440, 1164]
    assert [row["end_step"] for row in plan["pours"]] == [1842, 3282, 4446]
    assert [row["stage_capture_step"] for row in plan["pours"]] == [
        1842,
        3282,
        4482,
    ]
    assert [row["particle_count"] for row in plan["pours"]] == [1774] * 3
    assert sum(row["particle_count"] for row in plan["pours"]) == 5322
    assert sum(row["emission_chunk_count"] for row in plan["pours"]) == event_count
    assert plan["timing"]["primary_final_capture_step"] == 4482
    assert plan["recipe8_leading_residual"]["label_provenance"][
        "leading_residual_volume_ml"
    ] == residual_ml
    assert plan["recipe8_leading_residual"]["packing_calibration"][
        "effective_particle_volume_ml"
    ] == pytest.approx(0.01127286694)
    assert plan["recipe8_leading_residual"]["packing_calibration"][
        "source_receipt_sha256"
    ] == "11173728015f6c8371e7639987fc5992517d1b60271477252b9b0de0975eb934"
    assert plan["totals"]["nominal_mass_convention_equivalent_ml"] == pytest.approx(
        83.15625
    )
    assert plan["totals"]["packing_equivalent_volume_ml"] == pytest.approx(60.0, abs=0.01)
    assert "residual" not in " ".join(
        plan["recipe8_leading_residual"]["model_input_record"]
    ).lower()
    assert all(
        np.isclose(sum(event["particle_composition"]), 1.0)
        and np.count_nonzero(event["particle_composition"]) == 1
        for pour in plan["pours"]
        for event in pour["emission_schedule"]
    )
    expected_residual_count = 0 if residual_ml == 0.0 else 89
    assert all(
        pour["leading_residual_particle_count"] == expected_residual_count
        for pour in plan["pours"]
    )


def test_recipe8_count_driven_fallback_uses_measured_packing_volume() -> None:
    args = _args(
        "--recipe8-residual-volume-ml",
        "0",
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--count-driven-twin-liquid-mesh",
        "--plan-only",
    )
    sequence._validate_args(args)

    plan = sequence._runtime_paint_ball_plan(args)
    receipt = sequence._empty_paint_ball_receipt(args)
    expected_particle_volume_m3 = 0.011272866937444935e-6

    assert plan["render_treatment"]["particle_volume_m3"] == pytest.approx(
        expected_particle_volume_m3
    )
    assert receipt["render_path"] == "count_driven_twin_liquid_mesh"
    assert receipt["count_driven_liquid_body"][
        "particle_volume_m3"
    ] == pytest.approx(expected_particle_volume_m3)
    assert receipt["claims"]["spatial_mixing_color_rendered"] is False
    assert receipt["claims"]["global_vial_blend_rendered"] is True


def test_recipe8_stream_source_switches_at_each_residue_segment_boundary() -> None:
    plan = sequence._runtime_paint_ball_plan(
        _args("--recipe8-residual-volume-ml", "1")
    )

    assert sequence._active_tank_for_step(plan, 1) == "t200"
    assert sequence._active_stream_source_for_step(plan, 1) == "t100"
    assert sequence._active_stream_source_for_step(plan, 92) == "t100"
    assert sequence._active_stream_source_for_step(plan, 93) == "t200"
    assert sequence._active_stream_source_for_step(plan, 1843) == "t200"
    assert sequence._active_stream_source_for_step(plan, 1914) == "t200"
    assert sequence._active_stream_source_for_step(plan, 1915) == "t400"
    assert sequence._active_stream_source_for_step(plan, 3283) == "t400"
    assert sequence._active_stream_source_for_step(plan, 3340) == "t400"
    assert sequence._active_stream_source_for_step(plan, 3341) == "t100"
    assert sequence._active_stream_source_for_step(plan, 4447) is None


def test_runtime_schedule_keeps_only_final_fill_boundary_stream_visible() -> None:
    plan = sequence._runtime_paint_ball_plan(
        _args("--recipe8-residual-volume-ml", "1")
    )

    schedule = sequence.build_runtime_capture_schedule(plan)
    by_step = {row["physics_step"]: row for row in schedule["captures"]}

    assert by_step[1842]["visible_stream_tanks"] == []
    assert by_step[3282]["visible_stream_tanks"] == []
    final_boundary = by_step[plan["timing"]["fill_end_step"]]
    assert final_boundary["visible_stream_tanks"] == [
        sequence._active_stream_source_for_step(
            plan, plan["timing"]["fill_end_step"]
        )
    ]
    assert final_boundary["visible_stream_tanks"] == ["t100"]
    assert by_step[4482]["visible_stream_tanks"] == []


def test_source_state_is_built_from_each_event_not_from_whole_pour() -> None:
    plan = sequence._runtime_paint_ball_plan(
        _args("--recipe8-residual-volume-ml", "1")
    )
    residue, current = plan["pours"][0]["emission_schedule"][:2]

    residue_state = sequence._source_state_for_event(residue)
    current_state = sequence._source_state_for_event(current)

    assert residue_state.scheduled_tank == "t100"
    assert residue_state.birth_composition == (0.0, 0.0, 1.0)
    # The second event is still in the residue segment; select the first current event.
    current_event = next(
        event
        for event in plan["pours"][0]["emission_schedule"]
        if event["segment_role"] == "current_tank"
    )
    current_state = sequence._source_state_for_event(current_event)
    assert current_state.scheduled_tank == "t200"
    assert current_state.birth_composition == (1.0, 0.0, 0.0)
    assert residue_state.birth_event_id != current_state.birth_event_id


def test_full_run_solver_iterations_are_validated_without_changing_default() -> None:
    default = _args()
    tuned = _args("--solver-position-iterations", "16")
    sequence._validate_args(default)
    sequence._validate_args(tuned)

    assert default.solver_position_iterations == 4
    assert sequence._requested_solver_position_iterations(default) == 4
    assert sequence._requested_solver_position_iterations(tuned) == 16

    invalid = _args("--solver-position-iterations", "0")
    with pytest.raises(ValueError, match="solver_position_iterations"):
        sequence._validate_args(invalid)


def test_recipe8_rejects_unapproved_residual_severity() -> None:
    args = _args("--recipe8-residual-volume-ml", "0.5")
    with pytest.raises(ValueError, match="0 or 1"):
        sequence._validate_args(args)


def test_q2_count_override_rebuilds_only_first_pour_and_receipt_targets() -> None:
    args = _args(
        "--q2-no-render",
        "--q2-particle-count",
        "887",
        "--q2-solver-position-iterations",
        "16",
        "--q2-post-emission-settle-steps",
        "360",
    )
    sequence._validate_args(args)

    plan = sequence._q2_runtime_plan(args)
    receipt = sequence._empty_q2_no_render_receipt(args)

    assert plan["pours"][0]["particle_count"] == 887
    assert sum(
        row["particle_count"] for row in plan["pours"][0]["emission_schedule"]
    ) == 887
    assert plan["pours"][1]["particle_count"] == 640
    assert receipt["target"]["programmed_volume_ml"] == 10.0
    assert receipt["target"]["expected_particle_count"] == 887
    assert receipt["target"]["expected_stable_ids"] == [0, 886]
    assert receipt["evidence"]["mass"]["expected_total_mass_kg"] == pytest.approx(
        887 * 0.000015625
    )
    assert receipt["target"]["height_bounds_m"] == pytest.approx(
        [0.01745933942908709, 0.019297164632148894]
    )


def test_recipe8_receipt_binds_dynamic_solver_capacity_and_capture_denominator() -> None:
    args = _args(
        "--recipe8-residual-volume-ml",
        "1",
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--plan-only",
    )
    sequence._validate_args(args)

    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["point_instancer"]["max_particles"] == 5322
    assert receipt["particle_system"]["runtime_configuration"][
        "max_particles"
    ] == 5322
    assert receipt["particle_system"]["runtime_configuration"][
        "solver_position_iterations"
    ] == 16
    captures = receipt["runtime_capture_schedule"]["captures"]
    capture_steps = [row["physics_step"] for row in captures]
    assert len(captures) == len(set(capture_steps))
    assert receipt["runtime_capture_schedule"]["capture_stride_steps"] == 312
    assert receipt["runtime_capture_schedule"]["capture_hz"] == 120 / 312
    assert receipt["timing"]["capture_hz"] == 120 / 312
    in_pour = [row for row in captures if row["kind"] == "in_pour"]
    assert len(in_pour) >= 3 * len(receipt["plan"]["pours"])
    segment_counts = {
        (row["pour_ordinal"], row["source_segment_ordinal"]): sum(
            candidate["pour_ordinal"] == row["pour_ordinal"]
            and candidate.get("source_segment_ordinal")
            == row["source_segment_ordinal"]
            for candidate in in_pour
        )
        for row in in_pour
    }
    assert len(segment_counts) == 6
    assert min(segment_counts.values()) >= 2


def test_c15_time_preserving_video_is_accepted_at_the_runner_boundary() -> None:
    """C15 keeps sparse captures but encodes their real timeline at 10 fps."""

    args = _args(
        "--recipe8-residual-volume-ml",
        "0",
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--count-driven-twin-liquid-mesh",
        "--capture-stride",
        "312",
        "--video-fps",
        "10",
        "--wall-limit-seconds",
        "10800",
        "--plan-only",
    )

    sequence._validate_args(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert receipt["timing"]["physics_hz"] == 120
    assert receipt["arguments"]["capture_stride"] == 312
    assert receipt["arguments"]["video_fps"] == 10.0
    assert receipt["runtime_capture_schedule"]["capture_stride_steps"] == 312
    assert receipt["runtime_capture_schedule"]["capture_hz"] == 120 / 312
    rows = receipt["runtime_capture_schedule"]["captures"]
    assert len(rows) == 19
    assert [
        sum(
            row["kind"] == "in_pour" and row["pour_ordinal"] == ordinal
            for row in rows
        )
        for ordinal in (1, 2, 3)
    ] == [6, 4, 3]
    assert sum(row["kind"] == "stage_handoff" for row in rows) == 3
    assert sum(row["kind"] == "final_plus_0_3" for row in rows) == 1
    assert sum(row["kind"] == "post_fill" for row in rows) == 1
    assert sum(row["kind"] == "final_plus_3_0" for row in rows) == 1
    assert {
        row["physics_step"]
        for row in rows
        if row.get("semantic_anchor")
        == "initial_stream_after_motion_falsifier"
    } == {12}
    assert min(row["physics_step"] for row in rows) == 12
    assert {
        row["physics_step"]
        for row in rows
        if row["kind"] in {
            "stage_handoff",
            "final_plus_0_3",
            "final_plus_3_0",
        }
    } == {1842, 3282, 4446, 4482, 4806}


def test_c15_runner_rejects_c14_low_video_fps() -> None:
    args = _args(
        "--recipe8-residual-volume-ml",
        "0",
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--count-driven-twin-liquid-mesh",
        "--capture-stride",
        "312",
        "--video-fps",
        "0.38461538461538464",
        "--wall-limit-seconds",
        "10800",
        "--plan-only",
    )

    with pytest.raises(ValueError, match="playback"):
        sequence._validate_args(args)

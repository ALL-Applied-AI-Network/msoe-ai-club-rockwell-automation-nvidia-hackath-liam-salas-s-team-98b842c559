"""Frozen CPU contracts for the C20 two-PhysX-substep ablation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli/twin/physx"))
import twin.physx.paint_ball_emitter as paint_ball_emitter


SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence_c20",
    ROOT / "cli/twin/physx" / "physx_three_pour_sequence.py",
)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)


STAGE = ROOT / "README.md"


def _args(*extra: str):
    return sequence.build_parser().parse_args(
        ["--out", "/tmp/c20-substeps", "--stage", str(STAGE), *extra]
    )


def _c20_args(*extra: str):
    return _args(
        "--recipe8-residual-volume-ml",
        "0",
        "--solver-position-iterations",
        "16",
        "--physics-substeps",
        "2",
        "--water-pbd-preset",
        "--count-driven-twin-liquid-mesh",
        "--plan-only",
        *extra,
    )


def test_c20_cli_is_explicit_and_default_remains_one_substep() -> None:
    control = _args()
    probe = _c20_args()

    sequence._validate_args(control)
    sequence._validate_args(probe)

    assert control.physics_substeps == 1
    assert probe.physics_substeps == 2


@pytest.mark.parametrize("value", [True, False, 0, -1, 3, 4])
def test_c20_rejects_boolean_or_unsupported_substep_counts(value: object) -> None:
    args = _args()
    args.physics_substeps = value

    with pytest.raises(ValueError, match="physics_substeps"):
        sequence._validate_args(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda args: setattr(args, "recipe8_residual_volume_ml", None), "Recipe-8"),
        (lambda args: setattr(args, "recipe8_residual_volume_ml", 1.0), "zero-residual"),
        (lambda args: setattr(args, "solver_position_iterations", 8), "16 solver"),
        (lambda args: setattr(args, "water_pbd_preset", False), "water PBD"),
        (
            lambda args: setattr(args, "count_driven_twin_liquid_mesh", False),
            "count-driven",
        ),
        (lambda args: setattr(args, "qualification_mode", True), "full Recipe-8"),
    ],
)
def test_c20_is_confined_to_the_frozen_qualification_configuration(
    mutation, message: str
) -> None:
    args = _c20_args()
    mutation(args)

    with pytest.raises(ValueError, match=message):
        sequence._validate_args(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 20261026),
        ("rt_subframes", 32),
        ("pathtracing_total_spp", 32),
    ],
)
def test_c20_pins_the_frozen_render_invocation(field: str, value: object) -> None:
    args = _c20_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match="frozen C20"):
        sequence._validate_args(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nuisance_receipt", STAGE),
        ("nuisance_receipt_sha256", "a" * 64),
        ("nuisance_case_id", "nuisance-case"),
    ],
)
def test_c20_forbids_every_nuisance_binding_field(
    field: str, value: object
) -> None:
    args = _c20_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match="frozen C20"):
        sequence._validate_args(args)


def test_c20_cannot_be_used_for_a_train_or_cohort_schedule() -> None:
    args = _c20_args()
    args.train_render_plan = ROOT / "README.md"

    with pytest.raises(ValueError, match="training or cohort"):
        sequence._validate_args(args)


def test_c20_clock_derivation_preserves_outer_duration() -> None:
    clock = sequence._build_physics_clock_contract(
        physics_substeps=2,
        report_interval_count=4806,
    )

    assert clock == {
        "report_hz": 120,
        "report_dt_s": pytest.approx(1.0 / 120.0),
        "report_interval_count": 4806,
        "physics_steps_per_second": 240,
        "physics_substeps_per_report_interval": 2,
        "physics_dt_s": pytest.approx(1.0 / 240.0),
        "physics_callback_count": 9612,
        "physical_duration_s": pytest.approx(40.05),
        "min_frame_rate": 120,
        "timeline_ticks_per_frame": 1,
        "timeline_end_s": pytest.approx(4807 / 120.0),
        "diffusion_updates_per_report_interval": 1,
        "diffusion_update_count": 4806,
        "diffusion_dt_s": pytest.approx(1.0 / 120.0),
        "diffusion_integrated_time_s": pytest.approx(40.05),
    }


def test_c20_collider_displacement_diagnostic_uses_the_physics_step() -> None:
    wall_rows = [
        {
            "path": str(spec["path"]),
            "contact_offset_authored": True,
            "rest_offset_authored": True,
            "contact_offset_m": sequence.WALL_CONTACT_OFFSET_M,
            "rest_offset_m": sequence.WALL_REST_OFFSET_M,
        }
        for spec in sequence.build_twin_collider_specs()[1:]
    ]

    evidence = sequence._build_wall_tunneling_mitigation(
        [{"path": "floor"}, *wall_rows],
        physics_steps_per_second=240,
    )

    assert evidence["readback_passed"] is True
    assert evidence["maximum_capped_displacement_per_step_m"] == pytest.approx(
        sequence.PAINT_BALL_MAX_VELOCITY_M_S / 240.0
    )


def _callback_group() -> dict[str, object]:
    return sequence._timeline_callback_group_contract_evidence(
        [
            {
                "callback_index": 3,
                "dt_s": 1.0 / 240.0,
                "timeline_time_s": 2.0 / 120.0,
                "app_update_index": 2,
            },
            {
                "callback_index": 4,
                "dt_s": 1.0 / 240.0,
                "timeline_time_s": 2.0 / 120.0,
                "app_update_index": 2,
            },
        ],
        report_index=2,
        app_update_index=2,
        physics_substeps=2,
        callback_count_before=2,
        physics_steps_before=101,
        physics_steps_after=103,
        simulation_time_before_s=1.0 / 120.0,
        simulation_time_after_s=2.0 / 120.0,
        timeline_time_before_s=1.0 / 120.0,
        timeline_time_after_s=2.0 / 120.0,
    )


def test_c20_callback_group_maps_two_callbacks_to_one_report() -> None:
    evidence = _callback_group()

    assert evidence["passed"] is True
    assert evidence["failed_predicates"] == []
    assert evidence["expected"]["callback_indices"] == [3, 4]
    assert evidence["expected"]["app_update_indices"] == [2, 2]
    assert evidence["expected"]["logical_physics_times_s"] == pytest.approx(
        [3.0 / 240.0, 4.0 / 240.0]
    )
    assert evidence["actual"]["physics_step_count_delta"] == 2
    assert evidence["actual"]["simulation_time_delta_s"] == pytest.approx(
        1.0 / 120.0
    )


@pytest.mark.parametrize(
    ("mutation", "predicate"),
    [
        (lambda kwargs: kwargs["callbacks"].pop(), "callback_count_delta"),
        (
            lambda kwargs: kwargs["callbacks"].append(
                {
                    "callback_index": 5,
                    "dt_s": 1.0 / 240.0,
                    "timeline_time_s": 2.0 / 120.0,
                    "app_update_index": 2,
                }
            ),
            "callback_count_delta",
        ),
        (
            lambda kwargs: kwargs["callbacks"][1].update(callback_index=3),
            "callback_indices",
        ),
        (
            lambda kwargs: kwargs["callbacks"][1].update(callback_index=5),
            "callback_indices",
        ),
        (
            lambda kwargs: kwargs.update(callback_count_before=1),
            "callback_count_before",
        ),
        (
            lambda kwargs: kwargs["callbacks"][1].update(app_update_index=3),
            "app_update_indices",
        ),
        (
            lambda kwargs: kwargs["callbacks"][0].update(dt_s=1.0 / 120.0),
            "callback_dts_s",
        ),
        (
            lambda kwargs: kwargs.update(physics_steps_after=102),
            "physics_step_count_delta",
        ),
        (
            lambda kwargs: kwargs.update(simulation_time_after_s=1.5 / 120.0),
            "simulation_time_delta_s",
        ),
        (
            lambda kwargs: kwargs.update(timeline_time_after_s=3.0 / 120.0),
            "timeline_time_after_s",
        ),
    ],
)
def test_c20_callback_group_fails_closed(mutation, predicate: str) -> None:
    kwargs = {
        "callbacks": [
            {
                "callback_index": 3,
                "dt_s": 1.0 / 240.0,
                "timeline_time_s": 2.0 / 120.0,
                "app_update_index": 2,
            },
            {
                "callback_index": 4,
                "dt_s": 1.0 / 240.0,
                "timeline_time_s": 2.0 / 120.0,
                "app_update_index": 2,
            },
        ],
        "report_index": 2,
        "app_update_index": 2,
        "physics_substeps": 2,
        "callback_count_before": 2,
        "physics_steps_before": 101,
        "physics_steps_after": 103,
        "simulation_time_before_s": 1.0 / 120.0,
        "simulation_time_after_s": 2.0 / 120.0,
        "timeline_time_before_s": 1.0 / 120.0,
        "timeline_time_after_s": 2.0 / 120.0,
    }
    mutation(kwargs)

    evidence = sequence._timeline_callback_group_contract_evidence(**kwargs)

    assert evidence["passed"] is False
    assert predicate in evidence["failed_predicates"]


def test_c20_emission_boundary_is_authored_between_report_groups() -> None:
    assert sequence._emission_callback_boundary(
        report_step=1, physics_substeps=2
    ) == {
        "report_step": 1,
        "authored_after_callback_count": 0,
        "scheduled_before_callback_index": 1,
        "scheduled_before_callback_indices": [1, 2],
    }
    assert sequence._emission_callback_boundary(
        report_step=4806, physics_substeps=2
    ) == {
        "report_step": 4806,
        "authored_after_callback_count": 9610,
        "scheduled_before_callback_index": 9611,
        "scheduled_before_callback_indices": [9611, 9612],
    }


def test_c20_preserves_recipe_capture_diffusion_and_video_schedule() -> None:
    control_args = _args(
        "--recipe8-residual-volume-ml",
        "0",
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--count-driven-twin-liquid-mesh",
        "--plan-only",
    )
    probe_args = _c20_args()
    sequence._validate_args(control_args)
    sequence._validate_args(probe_args)
    control = sequence._runtime_paint_ball_plan(control_args)
    probe = sequence._runtime_paint_ball_plan(probe_args)
    control_schedule = sequence.build_runtime_capture_schedule(
        control, capture_stride_steps=312
    )
    probe_schedule = sequence.build_runtime_capture_schedule(
        probe, capture_stride_steps=312
    )

    assert probe["pours"] == control["pours"]
    assert probe["timing"] == control["timing"]
    assert probe["totals"] == control["totals"]
    assert probe["paint_ball_emitter"] == control["paint_ball_emitter"]
    assert probe_schedule == control_schedule
    assert [row["physics_step"] for row in probe_schedule["captures"]] == [
        12,
        312,
        624,
        936,
        1248,
        1560,
        1842,
        2154,
        2466,
        2778,
        3090,
        3282,
        3594,
        3906,
        4218,
        4446,
        4482,
        4758,
        4806,
    ]
    video = sequence.build_time_preserving_video_repeat_plan(
        [row["physics_step"] for row in probe_schedule["captures"]],
        physics_hz=120,
        playback_fps=10.0,
    )
    assert len(video["source_frame_steps"]) == 19
    assert video["video_frame_count"] == 401
    assert video["source_duration_s"] == pytest.approx(40.05)

    wrong_video_clock = sequence.build_time_preserving_video_repeat_plan(
        video["source_frame_steps"], physics_hz=240, playback_fps=10.0
    )
    assert wrong_video_clock["video_frame_count"] != 401
    assert wrong_video_clock["source_duration_s"] != pytest.approx(40.05)


def test_c20_receipt_provenance_separates_report_and_physics_clocks() -> None:
    control_args = _args(
        "--recipe8-residual-volume-ml",
        "0",
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--count-driven-twin-liquid-mesh",
        "--plan-only",
    )
    receipt = sequence._empty_paint_ball_receipt(_c20_args())
    control = sequence._empty_paint_ball_receipt(control_args)
    timing = receipt["timing"]

    assert receipt["arguments"]["physics_substeps"] == 2
    assert receipt["ablation"] == {
        "id": "c20_two_physx_substeps_per_report_interval",
        "qualification_only": True,
        "training_or_cohort_use_authorized": False,
        "changed_variable": "physics_substeps_per_report_interval",
    }
    assert timing["report_hz"] == 120
    assert timing["physics_hz"] == 240
    assert timing["physics_steps_per_second"] == 240
    assert timing["physics_substeps_per_report_interval"] == 2
    assert timing["physics_callback_count"] == 9612
    assert timing["timeline_end_s"] == pytest.approx(4807 / 120.0)
    assert timing["diffusion_updates_per_report_interval"] == 1
    assert timing["diffusion_dt_s"] == pytest.approx(1.0 / 120.0)
    assert timing["report_interval_count"] == 4806
    assert timing["callbacks_expected"] == 9612
    assert receipt["semantics"]["training_eligible"] is False
    assert receipt["plan"] == control["plan"]
    assert (
        receipt["particle_system"]["runtime_configuration"]
        == control["particle_system"]["runtime_configuration"]
    )
    assert receipt["arguments"]["solver_position_iterations"] == 16
    assert control["arguments"]["solver_position_iterations"] == 16


def _c20_timing_fixture(*, report_count: int = 3) -> dict[str, object]:
    callback_count = report_count * 2
    return {
        "report_hz": 120,
        "report_dt_s": 1.0 / 120.0,
        "report_interval_count": report_count,
        "executed_report_intervals": report_count,
        "physics_hz": 240,
        "physics_steps_per_second": 240,
        "physics_substeps_per_report_interval": 2,
        "physics_dt_s": 1.0 / 240.0,
        "physics_callback_count": callback_count,
        "min_frame_rate": 120,
        "timeline_ticks_per_frame": 1,
        "timeline_end_s": (report_count + 1) / 120.0,
        "physical_duration_s": report_count / 120.0,
        "executed_physics_steps": callback_count,
        "callbacks_expected": callback_count,
        "callbacks_observed": callback_count,
        "callback_indices": list(range(1, callback_count + 1)),
        "callback_app_update_indices": [
            report
            for report in range(1, report_count + 1)
            for _ in range(2)
        ],
        "callback_dts_s": [1.0 / 240.0] * callback_count,
        "callback_times_s": [
            report / 120.0
            for report in range(1, report_count + 1)
            for _ in range(2)
        ],
        "callback_logical_times_s": [
            callback / 240.0 for callback in range(1, callback_count + 1)
        ],
        "callback_dt_sum_s": report_count / 120.0,
        "one_callback_per_app_update": False,
        "callbacks_per_app_update_contract_passed": True,
        "simulation_owner": "timeline_app_update",
        "physics_step_method": "app.update()",
        "direct_physx_api_used": False,
        "physics_loop_app_update_count": report_count,
        "callback_group_trace_path": "timeline-callback-groups.json",
        "callback_group_trace_sha256": "a" * 64,
        "callback_group_trace_count": report_count,
        "final_edit_target_is_session": True,
        "positions_session_layer_only": True,
        "velocities_session_layer_only": True,
        "orientations_session_layer_only": True,
        "scales_session_layer_only": True,
        "proto_indices_session_layer_only": True,
        "ids_session_layer_only": True,
        "simulation_points_root_layer_absent": True,
        "simulation_points_runtime_readback_passed": True,
        "physics_manager_readback": {
            "baseline_num_physics_steps": 10,
            "final_num_physics_steps": 10 + callback_count,
            "delta_num_physics_steps": callback_count,
            "baseline_simulation_time_s": 7.0,
            "final_simulation_time_s": 7.0 + report_count / 120.0,
            "delta_simulation_time_s": report_count / 120.0,
            "passed": True,
        },
        "physics_settings_readback": {
            "physics_scene_path": "/FillStation/FluidmixF2/PhysicsScene",
            "requested_time_steps_per_second": 240,
            "readback_time_steps_per_second": 240,
            "requested_min_frame_rate": 120,
            "readback_min_frame_rate": 120,
            "requested_fixed_time_stepping": True,
            "readback_fixed_time_stepping": True,
            "requested_timeline_rate": 120,
            "readback_timeline_rate": 120.0,
            "requested_timeline_ticks_per_frame": 1,
            "readback_timeline_ticks_per_frame": 1,
            "passed": True,
        },
        "diffusion_updates_per_report_interval": 1,
        "diffusion_update_count": report_count,
        "diffusion_dt_s": 1.0 / 120.0,
        "diffusion_integrated_time_s": report_count / 120.0,
        "emissions_authored_on_main_thread": True,
        "emission_boundary_verified": True,
        "emission_boundaries": [
            {
                "physics_step": 2,
                "report_step": 2,
                "authored_after_callback_count": 2,
                "scheduled_before_callback_index": 3,
                "scheduled_before_callback_indices": [3, 4],
                "authored_on_main_thread": True,
            }
        ],
    }


def test_c20_receipt_timing_validator_accepts_exact_two_substep_evidence() -> None:
    timing = _c20_timing_fixture()
    timing["simulation_points_session_layer_only"] = False

    assert paint_ball_emitter._timeline_substep_timing_contract_valid(
        timing, expected_report_interval_count=3
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda timing: timing.update(physics_steps_per_second=120),
        lambda timing: timing["callback_app_update_indices"].__setitem__(1, 2),
        lambda timing: timing["callback_dts_s"].__setitem__(0, 1.0 / 120.0),
        lambda timing: timing.update(callback_dt_sum_s=1.0 / 120.0),
        lambda timing: timing.update(diffusion_update_count=6),
        lambda timing: timing.update(diffusion_dt_s=1.0 / 240.0),
        lambda timing: timing["physics_manager_readback"].update(
            delta_num_physics_steps=5
        ),
        lambda timing: timing["physics_settings_readback"].update(
            readback_min_frame_rate=240
        ),
        lambda timing: timing["emission_boundaries"][0].update(
            authored_after_callback_count=1
        ),
        lambda timing: timing.update(callback_group_trace_sha256=None),
        lambda timing: timing.update(positions_session_layer_only=False),
        lambda timing: timing.update(simulation_points_root_layer_absent=False),
        lambda timing: timing.update(
            simulation_points_runtime_readback_passed=False
        ),
    ],
)
def test_c20_receipt_timing_validator_fails_closed(mutation) -> None:
    timing = _c20_timing_fixture()
    mutation(timing)

    assert not paint_ball_emitter._timeline_substep_timing_contract_valid(
        timing, expected_report_interval_count=3
    )

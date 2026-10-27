"""Fail-closed CPU boundary for rendering the source-bound train manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence_train_schedule",
    ROOT / "scripts" / "physx_three_pour_sequence.py",
)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)

from fluidmix.physx_train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_nuisance_receipt(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "receipt_type": "real_photometric_nuisance_range",
        "status": "measured",
        "photometric_status": "measured",
        "shadow_status": "measured",
        "pooled": {
            "candidate_count": 1,
            "usable_count": 1,
            "shadow_observation_count": 1,
            "percentile_bounds": {"luma": {"p05": 1.0, "p50": 2.0, "p95": 3.0}},
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = build_train_residual_artifacts(repo_root=ROOT)
    plan = artifacts["render-schedule-plan.json"]
    case = next(
        row
        for row in plan["execution_schedules"]
        if row["case_binding"]["vial_ordinal"] == 1
        and row["case_binding"]["residual_volume_ml"] == 9.0
    )
    case_id = case["case_binding"]["case_id"]
    batch = next(
        row for row in plan["batches"] if row["batch_id"] == case["case_binding"]["batch_id"]
    )
    render_seed = batch["render_seed_by_vial"][0]
    plan_path = (tmp_path / "render-plan.json").resolve()
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stage = (tmp_path / "fill_station.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    nuisance_path = _write_nuisance_receipt((tmp_path / "nuisance.json").resolve())
    qualification_args = sequence.build_parser().parse_args(
        [
            "--out",
            str((tmp_path / "qualification-output").resolve()),
            "--stage",
            str(stage),
            "--recipe8-residual-volume-ml",
            "0",
            "--solver-position-iterations",
            "16",
            "--physics-substeps",
            "1",
            "--water-pbd-preset",
            "--count-driven-twin-liquid-mesh",
            "--particle-resolution-profile",
            "c20_2p5mm",
            "--recipe8-effective-particle-volume-ml",
            "0.011272866937444935",
            "--plan-only",
        ]
    )
    qualification_plan = sequence._runtime_paint_ball_plan(qualification_args)
    qualification_schedule = sequence.build_runtime_capture_schedule(
        qualification_plan, capture_stride_steps=312
    )
    qualification = {
        "status": "pass",
        "render_path": "count_driven_twin_liquid_mesh",
        "qualification": {"first_stage_only": False},
        "arguments": {
            "recipe8_residual_volume_ml": 0.0,
            "seed": 20261025,
            "solver_position_iterations": 16,
            "physics_substeps": 1,
            "particle_resolution_profile": "c20_2p5mm",
            "count_driven_twin_liquid_mesh": True,
            "resolution": "640x480",
            "rt_subframes": 64,
            "pathtracing_total_spp": 64,
            "capture_stride": 312,
            "video_fps": 10.0,
            "wall_limit_seconds": 10800,
            "nuisance_receipt": None,
            "nuisance_receipt_sha256": None,
            "nuisance_case_id": None,
        },
        "versions": {"isaac": "6.2.0.0", "physx": "110.4.4"},
        "provenance": {
            "requested_stage_sha256": hashlib.sha256(stage.read_bytes()).hexdigest(),
            "script_sha256": "a" * 64,
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "rt_subframes": 64,
            "pathtracing_total_spp_requested": 64,
            "pathtracing_total_spp_readback": 64,
            "pathtracing_spp_requested": 64,
            "pathtracing_spp_readback": 64,
            "warmup_updates": 60,
            "frame_count": len(qualification_schedule["captures"]),
        },
        "plan": qualification_plan,
        "runtime_capture_schedule": qualification_schedule,
        "particle_resolution_profile": qualification_plan[
            "particle_resolution_profile"
        ],
        "source_emission_geometry": {
            "profile_id": "c20_2p5mm",
            "events": [],
            "expected_event_count": sum(
                len(pour["emission_schedule"])
                for pour in qualification_plan["pours"]
            ),
            "observed_event_count": 0,
            "all_support_gates_passed": None,
            "all_centroid_gates_passed": None,
            "all_separation_gates_passed": None,
            "all_flux_gates_passed": None,
            "gate_passed": True,
        },
        "timing": {
            "report_hz": 120,
            "physics_steps_per_second": 120,
            "physics_substeps_per_report_interval": 1,
            "report_interval_count": qualification_schedule["final_plus_3_0_step"],
            "executed_report_intervals": qualification_schedule["final_plus_3_0_step"],
            "physics_callback_count": qualification_schedule["final_plus_3_0_step"],
            "executed_physics_steps": qualification_schedule["final_plus_3_0_step"],
            "callbacks_expected": qualification_schedule["final_plus_3_0_step"],
            "callbacks_observed": qualification_schedule["final_plus_3_0_step"],
            "diffusion_update_count": qualification_schedule["final_plus_3_0_step"],
            "diffusion_updates_per_report_interval": 1,
            "final_edit_target_is_session": True,
            "positions_session_layer_only": True,
            "velocities_session_layer_only": True,
            "orientations_session_layer_only": True,
            "scales_session_layer_only": True,
            "proto_indices_session_layer_only": True,
            "ids_session_layer_only": True,
            "simulation_points_root_layer_absent": True,
            "simulation_points_runtime_readback_passed": True,
        },
    }
    qualification_path = (tmp_path / "qualification.json").resolve()
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": _canonical_sha256(plan),
        "case": case,
        "case_id": case_id,
        "render_seed": render_seed,
        "stage": stage,
        "qualification_path": qualification_path,
        "qualification_sha256": hashlib.sha256(
            qualification_path.read_bytes()
        ).hexdigest(),
        "nuisance_path": nuisance_path,
        "nuisance_sha256": hashlib.sha256(nuisance_path.read_bytes()).hexdigest(),
    }


def _args(inputs: dict[str, object], *extra: str):
    return sequence.build_parser().parse_args(
        [
            "--out",
            "/tmp/train-schedule",
            "--stage",
            str(inputs["stage"]),
            "--seed",
            str(inputs["render_seed"]),
            "--water-pbd-preset",
            "--solver-position-iterations",
            "16",
            "--physics-substeps",
            "1",
            "--count-driven-twin-liquid-mesh",
            "--particle-resolution-profile",
            "c20_2p5mm",
            "--recipe8-effective-particle-volume-ml",
            "0.011272866937444935",
            "--train-render-plan",
            str(inputs["plan_path"]),
            "--train-render-plan-sha256",
            str(inputs["plan_sha256"]),
            "--train-case-id",
            str(inputs["case_id"]),
            "--accepted-qualification-receipt",
            str(inputs["qualification_path"]),
            "--accepted-qualification-receipt-sha256",
            str(inputs["qualification_sha256"]),
            "--nuisance-receipt",
            str(inputs["nuisance_path"]),
            "--nuisance-receipt-sha256",
            str(inputs["nuisance_sha256"]),
            "--nuisance-case-id",
            next(
                batch["nuisance_group_ids"][batch["case_ids"].index(inputs["case_id"])]
                for batch in inputs["plan"]["batches"]
                if inputs["case_id"] in batch["case_ids"]
            ),
            *extra,
        ]
    )


def _enable_legacy_two_substep_qualification(inputs: dict[str, object]) -> None:
    qualification_path = Path(inputs["qualification_path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["render_path"] = "count_driven_twin_liquid_mesh"
    qualification["arguments"].update(
        {
            "seed": 20261025,
            "physics_substeps": 2,
            "count_driven_twin_liquid_mesh": True,
            "rt_subframes": 64,
            "pathtracing_total_spp": 64,
            "nuisance_receipt": None,
            "nuisance_receipt_sha256": None,
            "nuisance_case_id": None,
        }
    )
    final_step = qualification["runtime_capture_schedule"]["final_plus_3_0_step"]
    qualification["timing"] = {
        "report_hz": 120,
        "physics_steps_per_second": 240,
        "physics_substeps_per_report_interval": 2,
        "report_interval_count": final_step,
        "executed_report_intervals": final_step,
        "physics_callback_count": 2 * final_step,
        "executed_physics_steps": 2 * final_step,
        "callbacks_expected": 2 * final_step,
        "callbacks_observed": 2 * final_step,
        "diffusion_update_count": final_step,
        "diffusion_updates_per_report_interval": 1,
        "final_edit_target_is_session": True,
        "positions_session_layer_only": True,
        "velocities_session_layer_only": True,
        "orientations_session_layer_only": True,
        "scales_session_layer_only": True,
        "proto_indices_session_layer_only": True,
        "ids_session_layer_only": True,
        "simulation_points_root_layer_absent": True,
        "simulation_points_runtime_readback_passed": True,
    }
    qualification["ablation"] = {
        "id": "c20_two_physx_substeps_per_report_interval",
        "qualification_only": True,
        "training_or_cohort_use_authorized": False,
        "changed_variable": "physics_substeps_per_report_interval",
    }
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["qualification_sha256"] = hashlib.sha256(
        qualification_path.read_bytes()
    ).hexdigest()


def test_train_case_route_binds_source_plan_and_qualification_before_authorizing_render(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())
    args = _args(inputs)

    sequence._validate_args(args)
    plan = sequence._runtime_paint_ball_plan(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert plan["totals"]["particle_count"] == 4435
    assert [row["stage_capture_step"] for row in plan["pours"]] == [
        1842,
        3282,
        4390,
    ]
    assert plan["timing"]["primary_final_capture_step"] == 4390
    assert plan["execution_schedule"]["adapter_schema_version"] == 2
    assert plan["execution_schedule"]["contract_kind"] == "train_residual_v2"
    assert plan["execution_schedule"]["case_id"] == inputs["case_id"]
    assert plan["execution_schedule"]["render_authorized"] is True
    assert plan["execution_schedule"]["training_eligible"] is False
    assert plan["execution_schedule"]["qualification_receipt_sha256"] == inputs[
        "qualification_sha256"
    ]
    assert plan["fallback_render_model"]["vial_representation"] == (
        "count_driven_twin_liquid_mesh"
    )
    assert all(
        event["particle_composition"].count(1.0) == 1
        and sum(event["particle_composition"]) == 1.0
        for pour in plan["pours"]
        for event in pour["emission_schedule"]
    )
    model_input = plan["execution_schedule"]["model_input_record"]
    assert not any(
        fragment in key.lower()
        for key in model_input
        for fragment in ("residual", "label", "oracle", "disposition", "profile")
    )
    assert receipt["point_instancer"]["max_particles"] == 4435
    assert receipt["particle_system"]["runtime_configuration"][
        "max_particles"
    ] == 4435
    assert receipt["particle_system"]["runtime_configuration"][
        "solver_position_iterations"
    ] == 16
    assert receipt["execution_schedule"]["training_eligible"] is False
    assert receipt["execution_schedule"]["render_authorized"] is True
    assert receipt["arguments"]["train_case_id"] == inputs["case_id"]
    assert receipt["arguments"]["accepted_qualification_receipt_sha256"] == (
        inputs["qualification_sha256"]
    )
    capture_rows = receipt["runtime_capture_schedule"]["captures"]
    expected_stage_steps = [1842, 3282, 4390]
    assert all(
        any(
            row["physics_step"] == step
            and row["visible_stream_tanks"] == []
            for row in capture_rows
        )
        for step in expected_stage_steps
    )
    assert receipt["runtime_capture_schedule"]["final_plus_3_0_step"] == 4714


def test_legacy_train_case_records_one_substep_qualification_without_ablation(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())
    args = _args(inputs)

    sequence._validate_args(args)
    contract = sequence._execution_schedule_contract(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert contract["qualification"]["physics_substeps"] == 1
    assert contract["qualification"]["report_hz"] == 120
    assert contract["qualification"]["physics_steps_per_second"] == 120
    assert receipt["arguments"]["seed"] == inputs["render_seed"]
    assert receipt["arguments"]["physics_substeps"] == 1
    assert receipt["execution_schedule"]["render_seed"] == inputs["render_seed"]
    assert "ablation" not in receipt
    assert receipt["particle_resolution_profile"]["profile_id"] == "c20_2p5mm"
    assert receipt["particle_resolution_profile"]["source_sampler"] == (
        "c20_3x3_lattice"
    )
    assert receipt["particle_resolution_profile"][
        "effective_particle_volume_ml"
    ] == pytest.approx(0.011272866937444935)


def test_legacy_one_substep_train_case_rejects_two_substep_qualification(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    _enable_legacy_two_substep_qualification(inputs)
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())

    with pytest.raises(ValueError, match="qualification.*substeps"):
        sequence._validate_args(_args(inputs))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda timing: timing.update(callbacks_observed=9611),
        lambda timing: timing.update(positions_session_layer_only=False),
        lambda timing: timing.update(simulation_points_root_layer_absent=False),
    ],
)
def test_legacy_train_case_rejects_drifted_qualification_evidence(
    tmp_path, monkeypatch, mutation
):
    inputs = _write_inputs(tmp_path)
    _enable_legacy_two_substep_qualification(inputs)
    qualification_path = Path(inputs["qualification_path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    mutation(qualification["timing"])
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["qualification_sha256"] = hashlib.sha256(
        qualification_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())

    with pytest.raises(ValueError, match="accepted qualification receipt"):
        sequence._validate_args(_args(inputs))


def test_train_case_route_requires_reviewed_subframes_to_match_qualification(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    qualification_path = Path(inputs["qualification_path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["render"]["rt_subframes"] = 16
    qualification["render"].update(
        pathtracing_total_spp_requested=16,
        pathtracing_total_spp_readback=16,
        pathtracing_spp_requested=16,
        pathtracing_spp_readback=16,
    )
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["qualification_sha256"] = hashlib.sha256(
        qualification_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())

    mismatched = _args(
        inputs,
        "--rt-subframes",
        "16",
        "--pathtracing-total-spp",
        "16",
    )
    with pytest.raises(ValueError, match="rt_subframes=64"):
        sequence._validate_args(mismatched)


def test_train_case_route_rejects_previous_c13_qualification_schedule(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    qualification_path = Path(inputs["qualification_path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    previous_schedule = sequence.build_runtime_capture_schedule(
        qualification["plan"], capture_stride_steps=48
    )
    qualification["arguments"].update(capture_stride=48, video_fps=2.5)
    qualification["runtime_capture_schedule"] = previous_schedule
    qualification["render"]["frame_count"] = len(
        previous_schedule["captures"]
    )
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["qualification_sha256"] = hashlib.sha256(
        qualification_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())

    with pytest.raises(ValueError, match="full-run contract"):
        sequence._validate_args(_args(inputs))


def test_train_case_count_driven_fallback_must_match_accepted_qualification(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    qualification_path = Path(inputs["qualification_path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["arguments"]["count_driven_twin_liquid_mesh"] = True
    qualification["render_path"] = "count_driven_twin_liquid_mesh"
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["qualification_sha256"] = hashlib.sha256(
        qualification_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())

    args = _args(inputs, "--count-driven-twin-liquid-mesh")
    sequence._validate_args(args)
    plan = sequence._runtime_paint_ball_plan(args)

    assert plan["render_treatment"]["particle_volume_m3"] == pytest.approx(
        0.011272866937444935e-6
    )
    assert plan["render_treatment"]["spatial_mixing_color_rendered"] is False
    assert plan["render_treatment"]["global_uncalibrated_blend"] is True


def test_train_case_route_rejects_failed_qualification_and_source_plan_tamper(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        sequence,
        "validate_paint_ball_receipt",
        lambda receipt: ("full_run_volume_contract",),
    )
    with pytest.raises(ValueError, match="qualification receipt failed"):
        sequence._validate_args(_args(inputs))

    inputs = _write_inputs(tmp_path / "tampered")
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())
    plan = json.loads(Path(inputs["plan_path"]).read_text(encoding="utf-8"))
    plan["execution_schedules"][0]["case_binding"]["residual_volume_ml"] = 2.0
    Path(inputs["plan_path"]).write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["plan_sha256"] = _canonical_sha256(plan)
    with pytest.raises(ValueError, match="source-bound train render plan"):
        sequence._validate_args(_args(inputs))


def test_train_case_route_requires_every_binding_and_matching_render_representation(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())

    missing = _args(inputs)
    missing.accepted_qualification_receipt_sha256 = None
    with pytest.raises(ValueError, match="train render bindings are required together"):
        sequence._validate_args(missing)

    mismatched = _args(inputs)
    mismatched.count_driven_twin_liquid_mesh = False
    with pytest.raises(ValueError, match="count-driven"):
        sequence._validate_args(mismatched)


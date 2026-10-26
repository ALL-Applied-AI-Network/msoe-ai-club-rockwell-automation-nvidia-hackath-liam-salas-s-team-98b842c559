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


def _write_inputs(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = build_train_residual_artifacts(repo_root=ROOT)
    plan = artifacts["render-schedule-plan.json"]
    case = plan["execution_schedules"][3]
    case_id = case["case_binding"]["case_id"]
    plan_path = (tmp_path / "render-plan.json").resolve()
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stage = (tmp_path / "fill_station.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    qualification = {
        "status": "pass",
        "render_path": "point_instancer",
        "qualification": {"first_stage_only": False},
        "arguments": {
            "recipe8_residual_volume_ml": 0.0,
            "solver_position_iterations": 16,
            "count_driven_twin_liquid_mesh": False,
            "resolution": "640x480",
            "capture_stride": 12,
            "video_fps": 10.0,
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
            "frame_count": 401,
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
        "stage": stage,
        "qualification_path": qualification_path,
        "qualification_sha256": hashlib.sha256(
            qualification_path.read_bytes()
        ).hexdigest(),
    }


def _args(inputs: dict[str, object], *extra: str):
    return sequence.build_parser().parse_args(
        [
            "--out",
            "/tmp/train-schedule",
            "--stage",
            str(inputs["stage"]),
            "--seed",
            "340101",
            "--water-pbd-preset",
            "--solver-position-iterations",
            "16",
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
            *extra,
        ]
    )


def test_train_case_route_binds_source_plan_and_qualification_before_authorizing_render(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(sequence, "validate_paint_ball_receipt", lambda receipt: ())
    args = _args(inputs)

    sequence._validate_args(args)
    plan = sequence._runtime_paint_ball_plan(args)
    receipt = sequence._empty_paint_ball_receipt(args)

    assert plan["totals"]["particle_count"] == 6653
    assert [row["stage_capture_step"] for row in plan["pours"]] == [
        2704,
        4260,
        5368,
    ]
    assert plan["timing"]["primary_final_capture_step"] == 5368
    assert plan["execution_schedule"]["adapter_schema_version"] == 2
    assert plan["execution_schedule"]["contract_kind"] == "train_residual_v2"
    assert plan["execution_schedule"]["case_id"] == inputs["case_id"]
    assert plan["execution_schedule"]["render_authorized"] is True
    assert plan["execution_schedule"]["training_eligible"] is False
    assert plan["execution_schedule"]["qualification_receipt_sha256"] == inputs[
        "qualification_sha256"
    ]
    assert plan["fallback_render_model"]["vial_representation"] != (
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
    assert receipt["point_instancer"]["max_particles"] == 6653
    assert receipt["particle_system"]["runtime_configuration"][
        "max_particles"
    ] == 6653
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
    expected_stage_steps = [2704, 4260, 5368]
    assert all(
        any(
            row["physics_step"] == step
            and row["visible_stream_tanks"] == []
            for row in capture_rows
        )
        for step in expected_stage_steps
    )
    assert receipt["runtime_capture_schedule"]["final_plus_3_0_step"] == 5692


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

    matched = _args(
        inputs,
        "--rt-subframes",
        "16",
        "--pathtracing-total-spp",
        "16",
    )
    sequence._validate_args(matched)
    contract = sequence._execution_schedule_contract(matched)
    assert contract["qualification"]["rt_subframes"] == 16
    assert contract["qualification"]["pathtracing_total_spp"] == 16

    mismatched = _args(inputs)
    with pytest.raises(ValueError, match="renderer settings do not match"):
        sequence._validate_args(mismatched)


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

    with pytest.raises(ValueError, match="full-run contract"):
        sequence._validate_args(_args(inputs, "--count-driven-twin-liquid-mesh"))


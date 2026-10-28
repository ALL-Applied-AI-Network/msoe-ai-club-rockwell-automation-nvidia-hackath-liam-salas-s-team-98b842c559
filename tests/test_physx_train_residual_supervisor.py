from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PARENT_QUALIFICATION_RENDERER_SHA256 = (
    "2e357b0b09a1fce901889b030291e59ed1540dc7b322fb3d203432d3fd9f6b3b"
)
CURRENT_COHORT_RENDERER_SHA256 = (
    "cfd446262c91ddb4cbfc8a34659711565206faac14c657867a75b311ac982038"
)
EXPECTED_PARENT_RENDERER_COMPATIBILITY = {
    "mode": "approved_parent_train_adapter_delta",
    "qualification_renderer_sha256": PARENT_QUALIFICATION_RENDERER_SHA256,
    "cohort_renderer_sha256": CURRENT_COHORT_RENDERER_SHA256,
    "qualification_renderer_commit": "3903789c78d1b4accc241a413db964657e143648",
    "cohort_renderer_commit": "bound_by_cohort_renderer_sha256",
    "declared_delta_scope": [
        "additive_v4_source_plan_adapter",
        "additive_tank_colour_resolution_adapter",
    ],
    "qualification_render_path_changed": False,
}
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_physx_train_residual_cohort",
    ROOT / "scripts" / "run_physx_train_residual_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)

from fluidmix.physx_train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)
from fluidmix.nuisance_authoring import LIGHT_SPECS, EXPECTED_LIGHT_PATHS  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _nuisance_rows() -> tuple[dict[str, object], dict[str, object]]:
    lights = {}
    transforms = {}
    for path in EXPECTED_LIGHT_PATHS:
        spec = LIGHT_SPECS[path]
        transform_path = spec["transform_path"]
        lights[path] = {
            "type_name": spec["type_name"],
            "intensity": 1.0,
            "exposure": 0.0,
            "transform_path": transform_path,
            "translate": [0.0, 0.0, 0.0] if transform_path else None,
            "rotate_xyz_degrees": [0.0, 0.0, 0.0] if transform_path else None,
        }
        transforms[path] = {
            "transform_path": transform_path,
            "translate": [0.0, 0.0, 0.0] if transform_path else None,
            "rotate_xyz_degrees": [0.0, 0.0, 0.0] if transform_path else None,
        }
    return lights, transforms


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_inputs(tmp_path: Path):
    plan = build_train_residual_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    plan_path = (tmp_path / "render-plan.json").resolve()
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stage = (tmp_path / "fill_station.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    renderer = (ROOT / "scripts" / "physx_three_pour_sequence.py").resolve()
    qualification_args = supervisor.sequence.build_parser().parse_args(
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
    qualification_plan = supervisor.sequence._runtime_paint_ball_plan(
        qualification_args
    )
    qualification_schedule = supervisor.sequence.build_runtime_capture_schedule(
        qualification_plan, capture_stride_steps=312
    )
    qualification = {
        "status": "pass",
        "render_path": "count_driven_twin_liquid_mesh",
        "qualification": {"first_stage_only": False},
        "arguments": {
            "seed": 20261025,
            "recipe8_residual_volume_ml": 0.0,
            "solver_position_iterations": 16,
            "physics_substeps": 1,
            "particle_resolution_profile": "c20_2p5mm",
            "recipe8_effective_particle_volume_ml": 0.011272866937444935,
            "count_driven_twin_liquid_mesh": True,
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
            "requested_stage_sha256": _sha256(stage),
            "script_sha256": PARENT_QUALIFICATION_RENDERER_SHA256,
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
        json.dumps(qualification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_root = (tmp_path / "cohort").resolve()
    nuisance_path = _write_nuisance_receipt((tmp_path / "nuisance.json").resolve())
    return {
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": _canonical_sha256(plan),
        "qualification_path": qualification_path,
        "qualification_sha256": _sha256(qualification_path),
        "stage": stage,
        "renderer": renderer,
        "output_root": output_root,
        "nuisance_path": nuisance_path,
        "nuisance_sha256": _sha256(nuisance_path),
    }


def _args(inputs: dict[str, object], *extra: str):
    return supervisor.build_parser().parse_args(
        [
            "--plan",
            str(inputs["plan_path"]),
            "--plan-sha256",
            str(inputs["plan_sha256"]),
            "--accepted-qualification-receipt",
            str(inputs["qualification_path"]),
            "--accepted-qualification-receipt-sha256",
            str(inputs["qualification_sha256"]),
            "--nuisance-receipt",
            str(inputs["nuisance_path"]),
            "--nuisance-receipt-sha256",
            str(inputs["nuisance_sha256"]),
            "--output-root",
            str(inputs["output_root"]),
            "--stage",
            str(inputs["stage"]),
            "--renderer-script",
            str(inputs["renderer"]),
            "--python-executable",
            sys.executable,
            "--host-lock",
            str(Path(inputs["output_root"]).parent / "host.lock"),
            *extra,
        ]
    )


def test_supervisor_requires_nuisance_receipt_and_hash():
    with pytest.raises(SystemExit):
        supervisor.build_parser().parse_args([])


def test_nuisance_readback_accepts_usd_quantization_but_rejects_material_drift():
    authored, transforms = _nuisance_rows()
    readback = deepcopy(authored)
    readback[next(iter(readback))]["intensity"] += 5.0e-6
    readback[next(iter(readback))]["translate"][0] += 5.0e-13
    transforms_readback = deepcopy(transforms)
    transform_path = next(
        path for path, row in transforms_readback.items() if row["transform_path"]
    )
    transforms_readback[transform_path]["rotate_xyz_degrees"][0] += 5.0e-6
    assert supervisor.light_rows_match(authored, readback)
    assert supervisor.transform_rows_match(transforms, transforms_readback)
    readback[next(iter(readback))]["intensity"] += 1.0e-3
    assert not supervisor.light_rows_match(authored, readback)


def _completed_receipt(command: list[str], inputs: dict[str, object]) -> dict[str, object]:
    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    case_id = value("--train-case-id")
    schedule = next(
        row
        for row in inputs["plan"]["execution_schedules"]
        if row["case_binding"]["case_id"] == case_id
    )
    runtime_schedule = deepcopy(schedule)
    runtime_schedule["timing"]["capture_hz"] = 10
    capture_schedule = supervisor.sequence.build_runtime_capture_schedule(
        runtime_schedule, capture_stride_steps=312
    )
    capture_steps = [
        int(row["physics_step"]) for row in capture_schedule["captures"]
    ]
    video = supervisor.sequence.build_time_preserving_video_repeat_plan(
        capture_steps, physics_hz=120, playback_fps=10.0
    )
    final_step = capture_schedule["final_plus_3_0_step"]
    light_rows, transform_rows = _nuisance_rows()
    return {
        "status": "pass",
        "render_path": "count_driven_twin_liquid_mesh",
        "arguments": {
            "seed": int(value("--seed")),
            "physics_substeps": 1,
            "particle_resolution_profile": "c20_2p5mm",
            "recipe8_effective_particle_volume_ml": 0.011272866937444935,
            "resolution": "640x480",
            "rt_subframes": 64,
            "pathtracing_total_spp": 64,
            "capture_stride": 312,
            "video_fps": 10.0,
            "wall_limit_seconds": 10800,
            "train_render_plan_sha256": value("--train-render-plan-sha256"),
            "train_case_id": value("--train-case-id"),
            "accepted_qualification_receipt_sha256": value(
                "--accepted-qualification-receipt-sha256"
            ),
            "count_driven_twin_liquid_mesh": True,
            "nuisance_receipt": str(inputs["nuisance_path"]),
            "nuisance_receipt_sha256": inputs["nuisance_sha256"],
            "nuisance_case_id": next(
                batch["nuisance_group_ids"][batch["case_ids"].index(case_id)]
                for batch in inputs["plan"]["batches"]
                if case_id in batch["case_ids"]
            ),
        },
        "provenance": {
            "requested_stage_sha256": _sha256(Path(inputs["stage"])),
            "script_sha256": _sha256(Path(inputs["renderer"])),
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
            "frame_count": len(capture_steps),
            "video_decoded_frame_count": video["video_frame_count"],
        },
        "nuisance_profile_binding": supervisor.sequence.bind_external_nuisance_receipt(
            inputs["nuisance_path"], inputs["nuisance_sha256"]
        ),
        "nuisance_runtime": {
            "status": "applied",
            "session_layer_only": True,
            "source_receipt_sha256": inputs["nuisance_sha256"],
            "case_id": next(
                batch["nuisance_group_ids"][batch["case_ids"].index(case_id)]
                for batch in inputs["plan"]["batches"]
                if case_id in batch["case_ids"]
            ),
            "seed": int(value("--seed")),
            "authored_light_properties": light_rows,
            "readback_light_properties": deepcopy(light_rows),
            "authored_transforms": transform_rows,
            "readback_transforms": deepcopy(transform_rows),
            "camera_pose_unchanged": True,
            "non_light_scene_unchanged": True,
        },
        "timing": {
            "report_hz": 120,
            "physics_steps_per_second": 120,
            "physics_substeps_per_report_interval": 1,
            "report_interval_count": final_step,
            "physics_callback_count": final_step,
            "diffusion_update_count": final_step,
            "diffusion_updates_per_report_interval": 1,
            "positions_session_layer_only": True,
            "velocities_session_layer_only": True,
            "orientations_session_layer_only": True,
            "scales_session_layer_only": True,
            "proto_indices_session_layer_only": True,
            "ids_session_layer_only": True,
            "simulation_points_root_layer_absent": True,
            "simulation_points_runtime_readback_passed": True,
        },
        "execution_schedule": {
            "case_id": value("--train-case-id"),
            "cohort_plan_sha256": value("--train-render-plan-sha256"),
            "schedule_file_sha256": _sha256(Path(inputs["plan_path"])),
            "qualification_receipt_sha256": value(
                "--accepted-qualification-receipt-sha256"
            ),
        },
    }


def _success_runner(inputs: dict[str, object], commands: list[list[str]]):
    active = 0

    def run(command, **kwargs):
        nonlocal active
        active += 1
        assert active == 1
        commands.append(list(command))
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "receipt.json").write_text(
            json.dumps(_completed_receipt(command, inputs), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        active -= 1
        return subprocess.CompletedProcess(command, 0)

    return run


def test_supervisor_runs_all_source_bound_cases_serially_with_qualified_mesh_contract(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    commands: list[list[str]] = []
    summary = supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    assert summary["status"] == "pass"
    assert summary["planned_case_count"] == 18
    assert summary["attempts_started"] == 18
    assert summary["processes_completed"] == 18
    assert summary["validated_completed_case_count"] == 18
    assert summary["observed_source_capture_count"] == 381
    assert summary["observed_decoded_video_frame_count"] == 7074
    manifest = json.loads(
        (Path(inputs["output_root"]) / "cohort-launch.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["expected_output_contract"]["case_count"] == 18
    assert manifest["expected_output_contract"]["keep_count"] == 12
    assert manifest["expected_output_contract"]["discard_count"] == 6
    assert manifest["expected_output_contract"]["source_capture_count"] == 381
    assert manifest["expected_output_contract"]["decoded_video_frame_count"] == 7074
    assert manifest["renderer_qualification_compatibility"] == (
        EXPECTED_PARENT_RENDERER_COMPATIBILITY
    )
    assert len(commands) == 18
    expected_cases = [
        row["case_binding"]["case_id"] for row in inputs["plan"]["execution_schedules"]
    ]
    assert [command[command.index("--train-case-id") + 1] for command in commands] == expected_cases
    assert [int(command[command.index("--seed") + 1]) for command in commands] == [
        330001,
        330002,
        330003,
        330004,
        330005,
        330006,
        330101,
        330102,
        330103,
        330104,
        330105,
        330106,
        330107,
        330108,
        330109,
        330110,
        330111,
        330112,
    ]
    for command in commands:
        assert command[command.index("--rt-subframes") + 1] == "64"
        assert command[command.index("--pathtracing-total-spp") + 1] == "64"
        assert command.count("--count-driven-twin-liquid-mesh") == 1
        assert command[command.index("--warmup") + 1] == "60"
        assert command[command.index("--resolution") + 1] == "640x480"
        assert command[command.index("--capture-stride") + 1] == "312"
        assert command[command.index("--video-fps") + 1] == "10.0"
        assert command[command.index("--physics-substeps") + 1] == "1"
        assert command[command.index("--particle-resolution-profile") + 1] == (
            "c20_2p5mm"
        )
        assert command[
            command.index("--recipe8-effective-particle-volume-ml") + 1
        ] == "0.011272866937444935"
        assert command[command.index("--nuisance-receipt") + 1] == str(
            inputs["nuisance_path"]
        )
        assert command[command.index("--nuisance-receipt-sha256") + 1] == inputs[
            "nuisance_sha256"
        ]
        case_id = command[command.index("--train-case-id") + 1]
        launch_path = (
            Path(inputs["output_root"])
            / "cases"
            / case_id
            / "launch-receipt.json"
        )
        assert launch_path.is_file()
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        assert launch["renderer_qualification_compatibility"] == (
            EXPECTED_PARENT_RENDERER_COMPATIBILITY
        )


def test_supervisor_runs_one_stable_disjoint_shard_with_subset_denominators(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    commands: list[list[str]] = []
    summary = supervisor.run_supervisor(
        _args(inputs, "--shard-count", "3", "--shard-index", "1"),
        run_process=_success_runner(inputs, commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    all_case_ids = [
        row["case_binding"]["case_id"]
        for row in inputs["plan"]["execution_schedules"]
    ]
    selected_case_ids = all_case_ids[1::3]
    assert summary["status"] == "pass"
    assert summary["shard"] == {
        "index": 1,
        "count": 3,
        "selection_rule": "global_ordinal_modulo_shard_count",
        "selected_case_ids": selected_case_ids,
    }
    assert summary["planned_case_count"] == 6
    assert summary["validated_completed_case_count"] == 6
    assert summary["expected_output_contract"]["case_count"] == 6
    assert summary["expected_output_contract"]["keep_count"] == 4
    assert summary["expected_output_contract"]["discard_count"] == 2
    assert summary["expected_output_contract"]["source_capture_count"] == 126
    assert summary["expected_output_contract"]["decoded_video_frame_count"] == 2358
    assert summary["observed_source_capture_count"] == 126
    assert summary["observed_decoded_video_frame_count"] == 2358
    assert [
        command[command.index("--train-case-id") + 1] for command in commands
    ] == selected_case_ids

    manifest = json.loads(
        (Path(inputs["output_root"]) / "cohort-launch.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["case_ids"] == all_case_ids
    assert manifest["expected_output_contract"]["case_count"] == 18
    assert manifest["expected_output_contract"]["source_capture_count"] == 381


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--shard-count", "0", "--shard-index", "0"), "shard_count"),
        (("--shard-count", "3", "--shard-index", "3"), "shard_index"),
        (("--shard-count", "2", "--shard-index", "-1"), "shard_index"),
    ],
)
def test_supervisor_rejects_invalid_shard_selection_before_creating_output(
    tmp_path, monkeypatch, extra, message
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    with pytest.raises(ValueError, match=message):
        supervisor.run_supervisor(
            _args(inputs, *extra),
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
    assert not Path(inputs["output_root"]).exists()


def _build_three_completed_shard_copies(
    inputs: dict[str, object], monkeypatch, tmp_path: Path, *, shard_count: int = 3
) -> list[Path]:
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    output_root = Path(inputs["output_root"])
    shard_roots: list[Path] = []
    for shard_index in range(shard_count):
        commands: list[list[str]] = []
        summary = supervisor.run_supervisor(
            _args(
                inputs,
                "--shard-count",
                str(shard_count),
                "--shard-index",
                str(shard_index),
            ),
            run_process=_success_runner(inputs, commands),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
        assert summary["status"] == "pass"
        staged = (tmp_path / f"completed-shard-{shard_index}").resolve()
        shutil.copytree(output_root, staged)
        shard_roots.append(staged)
        shutil.rmtree(output_root)
    shutil.copytree(shard_roots[0], output_root)
    return shard_roots


def test_merge_completed_shards_preserves_cases_and_emits_full_resume_attempt(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    shard_roots = _build_three_completed_shard_copies(
        inputs, monkeypatch, tmp_path
    )
    launches = 0

    def forbidden_launch(*args, **kwargs):
        nonlocal launches
        launches += 1
        return subprocess.CompletedProcess([], 99)

    merge = supervisor.merge_completed_shard_roots(
        _args(inputs),
        shard_roots=shard_roots,
        expected_shard_count=3,
        run_process=forbidden_launch,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    output_root = Path(inputs["output_root"])
    assert merge["status"] == "pass"
    assert merge["shard_count"] == 3
    assert merge["merged_case_count"] == 18
    assert merge["full_resume_summary"]["status"] == "pass"
    assert merge["full_resume_summary"]["planned_case_count"] == 18
    assert merge["full_resume_summary"][
        "previous_completed_cases_recognized"
    ] == 18
    assert merge["full_resume_summary"]["attempts_started"] == 0
    assert launches == 0
    assert len(list((output_root / "cases").glob("CASE_*"))) == 18
    assert len(list((output_root / "shard-attempts").glob("shard-*"))) == 3
    assert (output_root / "shard-merge.json").is_file()
    attempt_paths = sorted(
        (output_root / "supervisor-attempts").glob("attempt-*.json")
    )
    assert len(attempt_paths) == 2
    latest = json.loads(attempt_paths[-1].read_text(encoding="utf-8"))
    assert latest["status"] == "pass"
    assert latest["validated_completed_case_count"] == 18
    assert [row["outcome"] for row in latest["case_results"]] == [
        "recognized_completed"
    ] * 18


def test_merge_accepts_immutable_height_only_failure_after_fresh_offline_revalidation(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    shard_roots = _build_three_completed_shard_copies(
        inputs, monkeypatch, tmp_path, shard_count=18
    )
    shard_root = shard_roots[0]
    attempt_path = next((shard_root / "supervisor-attempts").glob("attempt-*.json"))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    failed_row = attempt["case_results"][0]
    warning = {
        "warning_type": "particle_height_outside_geometry_volume_bounds",
        "hard_gate": False,
        "claim": "diagnostic_warning_not_a_physics_pass",
        "physics_step": 120,
        "tank_id": "t100",
        "programmed_cumulative_volume_ml": 60.0,
        "measured_particle_height_m": 0.08,
        "lower_bound_m": 0.10,
        "upper_bound_m": 0.11,
        "miss_m": 0.02,
        "direction": "below",
        "original_gate_passed": False,
        "original_state_passed": False,
        "original_failure_reasons": [
            "particle_height_outside_geometry_volume_bounds"
        ],
    }
    validation_result = {
        "status": "pass",
        "failures": [],
        "verified_artifact_count": 4,
        "warnings": [warning],
    }

    case_root = shard_root / "cases" / failed_row["case_id"]
    result_path = case_root / "case-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["outcome"] = "failed"
    result["evidence"] = {
        "artifact_validation": {
            "status": "insufficient_evidence",
            "failures": ["receipt:realtime_volume_height_gate"],
        },
        "exit_code": 0,
        "launch_failures": [],
        "receipt_failures": [],
        "failures": ["receipt:realtime_volume_height_gate"],
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    attempt["status"] = "failed"
    attempt["validated_completed_case_count"] = 0
    attempt["observed_source_capture_count"] = 0
    attempt["observed_decoded_video_frame_count"] = 0
    attempt["case_results"] = [deepcopy(result)]
    attempt["failure"] = {
        "reason": "completed_case_failed_validation",
        "case_id": result["case_id"],
        "evidence": deepcopy(result["evidence"]),
    }
    attempt_path.write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(Path(inputs["output_root"]))

    merged = supervisor.merge_completed_shard_roots(
        _args(inputs),
        shard_roots=shard_roots,
        expected_shard_count=18,
        run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
        validate_artifacts=lambda output: deepcopy(validation_result),
        active_processes=lambda: [],
    )

    assert merged["status"] == "pass"
    source = merged["source_shards"][0]
    assert source["offline_height_warning_revalidated_case_ids"] == [
        result["case_id"]
    ]
    copied_result = (
        Path(inputs["output_root"])
        / "cases"
        / result["case_id"]
        / "case-result.json"
    )
    assert json.loads(copied_result.read_text(encoding="utf-8"))["outcome"] == "failed"


def _relocate_completed_root_to_remote_host(root: Path) -> None:
    manifest_path = root / "cohort-launch.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    remote = {
        "plan_path": "/remote/payloads/render-plan.json",
        "qualification_receipt_path": "/remote/payloads/qualification.json",
        "stage_path": "/remote/twin/fill-station.usda",
        "renderer_script": "/remote/code/physx-three-pour.py",
        "python_executable": "/remote/isaac/python",
        "nuisance_receipt_path": "/remote/payloads/nuisance.json",
    }
    manifest.update(remote)
    manifest["nuisance_plan"]["receipt_path"] = remote["nuisance_receipt_path"]
    manifest["supervisor_script_sha256"] = "a" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    replacements = {
        "--stage": remote["stage_path"],
        "--train-render-plan": remote["plan_path"],
        "--accepted-qualification-receipt": remote[
            "qualification_receipt_path"
        ],
        "--nuisance-receipt": remote["nuisance_receipt_path"],
    }
    for case_root in sorted((root / "cases").glob("CASE_*")):
        launch_path = case_root / "launch-receipt.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        launch.update(remote)
        argv = launch["argv"]
        argv[0] = remote["python_executable"]
        argv[1] = remote["renderer_script"]
        argv[argv.index("--out") + 1] = (
            f"/remote/runs/{root.name}/cases/{case_root.name}/app-output"
        )
        for flag, value in replacements.items():
            argv[argv.index(flag) + 1] = value
        launch_path.write_text(
            json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        receipt_path = case_root / "app-output" / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["arguments"]["nuisance_receipt"] = remote[
            "nuisance_receipt_path"
        ]
        receipt["nuisance_profile_binding"]["receipt_path"] = remote[
            "nuisance_receipt_path"
        ]
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def test_merge_accepts_hash_bound_copied_host_path_relocation(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    shard_roots = _build_three_completed_shard_copies(
        inputs, monkeypatch, tmp_path
    )
    for root in shard_roots:
        _relocate_completed_root_to_remote_host(root)
    shutil.rmtree(Path(inputs["output_root"]))

    merged = supervisor.merge_completed_shard_roots(
        _args(inputs),
        shard_roots=shard_roots,
        expected_shard_count=3,
        run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    assert merged["status"] == "pass"
    assert Path(inputs["output_root"], "cohort-launch.json").is_file()
    assert {row["source_supervisor_script_sha256"] for row in merged["source_shards"]} == {
        "a" * 64
    }
    assert all(
        row["host_path_relocation"]["mode"] == "hash_bound"
        for row in merged["source_shards"]
    )


def test_hash_bound_host_relocation_rejects_immutable_hash_drift(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    inputs = _write_inputs(tmp_path)
    args = _args(inputs)
    expected = supervisor._cohort_manifest(args, supervisor._bind_inputs(args))
    observed = deepcopy(expected)
    for field in supervisor._HOST_PATH_MANIFEST_FIELDS:
        observed[field] = f"/remote/{field}"
    observed["nuisance_plan"]["receipt_path"] = "/remote/nuisance.json"
    observed["supervisor_script_sha256"] = "a" * 64

    assert supervisor._host_relocated_manifest(observed, expected) is not None
    observed["stage_sha256"] = "b" * 64
    assert supervisor._host_relocated_manifest(observed, expected) is None


def test_merge_rejects_source_supervisor_hash_drift_before_scaffold(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    shard_roots = _build_three_completed_shard_copies(
        inputs, monkeypatch, tmp_path
    )
    for root in shard_roots:
        _relocate_completed_root_to_remote_host(root)
    drifted_path = shard_roots[1] / "cohort-launch.json"
    drifted = json.loads(drifted_path.read_text(encoding="utf-8"))
    drifted["supervisor_script_sha256"] = "b" * 64
    drifted_path.write_text(
        json.dumps(drifted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(Path(inputs["output_root"]))

    with pytest.raises(ValueError, match="source supervisor SHA-256 values differ"):
        supervisor.merge_completed_shard_roots(
            _args(inputs),
            shard_roots=shard_roots,
            expected_shard_count=3,
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
    assert not Path(inputs["output_root"]).exists()


def test_merge_rejects_duplicate_shard_index_before_copying(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    shard_roots = _build_three_completed_shard_copies(
        inputs, monkeypatch, tmp_path
    )
    output_root = Path(inputs["output_root"])
    assert len(list((output_root / "cases").glob("CASE_*"))) == 6

    with pytest.raises(ValueError, match="duplicate shard_index"):
        supervisor.merge_completed_shard_roots(
            _args(inputs),
            shard_roots=[shard_roots[0], shard_roots[0], shard_roots[2]],
            expected_shard_count=3,
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
    assert len(list((output_root / "cases").glob("CASE_*"))) == 6
    assert not (output_root / "shard-merge.json").exists()


def test_supervisor_rejects_any_other_qualification_renderer_mismatch(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    qualification_path = Path(inputs["qualification_path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["provenance"]["script_sha256"] = "b" * 64
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    inputs["qualification_sha256"] = _sha256(qualification_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )

    with pytest.raises(
        ValueError, match="renderer script hash differs from the accepted qualification"
    ):
        supervisor._bind_inputs(_args(inputs))


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (
            lambda receipt: receipt["timing"].update(
                physics_callback_count=(
                    receipt["timing"]["physics_callback_count"] - 1
                )
            ),
            "physics_callback_count",
        ),
        (
            lambda receipt: receipt["timing"].update(
                positions_session_layer_only=False
            ),
            "authored_rows_session_only",
        ),
        (
            lambda receipt: receipt["timing"].update(
                simulation_points_root_layer_absent=False
            ),
            "simulation_points_root_absent",
        ),
        (
            lambda receipt: receipt["arguments"].update(
                nuisance_case_id="nuisance-case"
            ),
            "nuisance_case_id_argument",
        ),
        (
            lambda receipt: receipt["render"].update(
                frame_count=receipt["render"]["frame_count"] - 1
            ),
            "source_capture_count",
        ),
    ],
)
def test_case_receipt_validator_rejects_legacy_contract_drift(
    tmp_path, monkeypatch, mutation, failure
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    args = _args(inputs)
    binding = supervisor._bind_inputs(args)
    case_id = binding["plan"]["execution_schedules"][0]["case_binding"][
        "case_id"
    ]
    seed = binding["case_seeds"][case_id]
    nuisance_case_id = binding["nuisance_case_ids"][case_id]
    command = supervisor._case_command(
        args,
        case_id=case_id,
        seed=seed,
        nuisance_case_id=nuisance_case_id,
        app_output=(tmp_path / "case-output").resolve(),
    )
    receipt = _completed_receipt(command, inputs)
    mutation(receipt)
    receipt_path = (tmp_path / "receipt.json").resolve()
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert failure in supervisor._case_receipt_failures(
        receipt_path,
        args=args,
        binding=binding,
        case_id=case_id,
        seed=seed,
        nuisance_case_id=nuisance_case_id,
    )


def test_case_receipt_validator_rejects_unapplied_nuisance_runtime(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    args = _args(inputs)
    binding = supervisor._bind_inputs(args)
    case_id = binding["plan"]["execution_schedules"][0]["case_binding"]["case_id"]
    seed = binding["case_seeds"][case_id]
    nuisance_case_id = binding["nuisance_case_ids"][case_id]
    command = supervisor._case_command(
        args,
        case_id=case_id,
        seed=seed,
        nuisance_case_id=nuisance_case_id,
        app_output=(tmp_path / "case-output").resolve(),
    )
    receipt = _completed_receipt(command, inputs)
    receipt["nuisance_runtime"]["status"] = "pending"
    receipt_path = (tmp_path / "receipt.json").resolve()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    assert "nuisance_runtime_applied" in supervisor._case_receipt_failures(
        receipt_path,
        args=args,
        binding=binding,
        case_id=case_id,
        seed=seed,
        nuisance_case_id=nuisance_case_id,
    )


def test_supervisor_stops_on_first_failed_process_and_preserves_denominator(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    commands: list[list[str]] = []
    success = _success_runner(inputs, commands)

    def fail_second(command, **kwargs):
        if len(commands) == 1:
            commands.append(list(command))
            return subprocess.CompletedProcess(command, 7)
        return success(command, **kwargs)

    summary = supervisor.run_supervisor(
        _args(inputs),
        run_process=fail_second,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )

    case_ids = [
        row["case_binding"]["case_id"] for row in inputs["plan"]["execution_schedules"]
    ]
    assert summary["status"] == "failed"
    assert summary["attempts_started"] == 2
    assert summary["processes_completed"] == 2
    assert summary["cases_passed_this_invocation"] == 1
    assert summary["cases_failed_this_invocation"] == 1
    assert len(commands) == 2
    assert (Path(inputs["output_root"]) / "cases" / case_ids[1] / "exit-code.txt").read_text() == "7\n"
    assert not (Path(inputs["output_root"]) / "cases" / case_ids[2]).exists()


def test_resume_is_explicit_and_only_recognizes_valid_completed_cases(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    first_commands: list[list[str]] = []
    probes = 0

    def stop_before_third():
        nonlocal probes
        probes += 1
        return [] if probes <= 2 else [{"pid": 999, "command": "isaac-sim"}]

    first = supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, first_commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=stop_before_third,
    )
    assert first["status"] == "failed"
    assert first["attempts_started"] == 2
    assert first["validated_completed_case_count"] == 2

    with pytest.raises(ValueError, match="--resume-completed is required"):
        supervisor.run_supervisor(
            _args(inputs),
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )

    resumed_commands: list[list[str]] = []
    resumed = supervisor.run_supervisor(
        _args(inputs, "--resume-completed"),
        run_process=_success_runner(inputs, resumed_commands),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )
    assert resumed["status"] == "pass"
    assert resumed["previous_completed_cases_recognized"] == 2
    assert resumed["attempts_started"] == 16
    assert resumed["processes_completed"] == 16
    assert resumed["validated_completed_case_count"] == 18
    assert len(resumed_commands) == 16
    assert len(list((Path(inputs["output_root"]) / "supervisor-attempts").glob("attempt-*.json"))) == 2


def test_resume_refuses_an_existing_case_with_drift_instead_of_overwriting(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    probes = 0

    def stop_after_first():
        nonlocal probes
        probes += 1
        return [] if probes == 1 else [{"pid": 999, "command": "isaac-sim"}]

    supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, []),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=stop_after_first,
    )
    first_case = inputs["plan"]["execution_schedules"][0]["case_binding"]["case_id"]
    launch_path = Path(inputs["output_root"]) / "cases" / first_case / "launch-receipt.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["seed"] += 1
    launch_path.write_text(json.dumps(launch, sort_keys=True) + "\n", encoding="utf-8")
    launches = 0

    def forbidden_launch(*args, **kwargs):
        nonlocal launches
        launches += 1
        return subprocess.CompletedProcess([], 0)

    resumed = supervisor.run_supervisor(
        _args(inputs, "--resume-completed"),
        run_process=forbidden_launch,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )
    assert resumed["status"] == "failed"
    assert resumed["failure"]["reason"] == "existing_case_is_not_a_valid_explicit_resume"
    assert resumed["attempts_started"] == 0
    assert launches == 0


def test_resume_rechecks_the_fixed_renderer_contract_in_the_completed_receipt(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    probes = 0

    def stop_after_first():
        nonlocal probes
        probes += 1
        return [] if probes == 1 else [{"pid": 999, "command": "isaac-sim"}]

    supervisor.run_supervisor(
        _args(inputs),
        run_process=_success_runner(inputs, []),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=stop_after_first,
    )
    first_case = inputs["plan"]["execution_schedules"][0]["case_binding"]["case_id"]
    receipt_path = Path(inputs["output_root"]) / "cases" / first_case / "app-output" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arguments"]["capture_stride"] = 99
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    launches = 0

    def forbidden_launch(*args, **kwargs):
        nonlocal launches
        launches += 1
        return subprocess.CompletedProcess([], 0)

    resumed = supervisor.run_supervisor(
        _args(inputs, "--resume-completed"),
        run_process=forbidden_launch,
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [],
    )
    assert resumed["status"] == "failed"
    assert "capture_stride" in resumed["failure"]["evidence"]["receipt_failures"]
    assert resumed["attempts_started"] == 0
    assert launches == 0


def test_supervisor_refuses_an_existing_isaac_process_before_creating_a_case(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    summary = supervisor.run_supervisor(
        _args(inputs),
        run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
        validate_artifacts=lambda output: {"status": "pass", "failures": []},
        active_processes=lambda: [{"pid": 123, "command": "isaac-sim"}],
    )
    assert summary["status"] == "failed"
    assert summary["failure"]["reason"] == "other_isaac_or_kit_process_is_running"
    assert summary["attempts_started"] == 0
    assert list((Path(inputs["output_root"]) / "cases").iterdir()) == []


def test_supervisor_rejects_nonfrozen_plan_hash_before_creating_output(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        supervisor.sequence, "validate_paint_ball_receipt", lambda receipt: ()
    )
    args = _args(inputs)
    args.plan_sha256 = "0" * 64
    with pytest.raises(ValueError, match="frozen source-bound plan"):
        supervisor.run_supervisor(
            args,
            run_process=lambda *args, **kwargs: pytest.fail("must not launch"),
            validate_artifacts=lambda output: {"status": "pass", "failures": []},
            active_processes=lambda: [],
        )
    assert not Path(inputs["output_root"]).exists()

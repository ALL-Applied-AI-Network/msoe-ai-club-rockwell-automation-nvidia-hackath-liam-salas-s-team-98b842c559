"""CPU contracts for the one-attempt timeline-owned PhysX emission probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli/twin/probes"))

from twin.probes.timeline_probe import (
    build_timeline_probe_plan,
    movement_evidence,
    validate_timeline_evidence,
)
import twin.probes.timeline_probe as timeline_probe
from twin.physx.isosurface_geometry import analyze_triangle_mesh_geometry

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_physx_timeline_emission_probe",
    ROOT / "cli/twin/probes" / "validate_physx_timeline_emission_probe.py",
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)
A18_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_physx_timeline_emission_probe_a18",
    ROOT / "cli/twin/probes" / "validate_physx_timeline_emission_probe_a18.py",
)
assert A18_VALIDATOR_SPEC is not None and A18_VALIDATOR_SPEC.loader is not None
a18_validator = importlib.util.module_from_spec(A18_VALIDATOR_SPEC)
A18_VALIDATOR_SPEC.loader.exec_module(a18_validator)


def _valid_mesh() -> dict[str, object]:
    return {
        "mesh_geometry_valid": True,
        "mesh_point_count": 4,
        "mesh_face_count": 4,
        "mesh_sha256": "a" * 64,
    }


def _receipt() -> dict[str, object]:
    fixed_dt = 1.0 / 60.0
    callbacks = [
        {
            "callback_index": index,
            "dt_s": fixed_dt,
            "app_update_index": 10 + index,
            "timeline_time_s": index * fixed_dt,
        }
        for index in range(1, 13)
    ]
    initial = [
        [-0.0015, -0.0015, 0.15],
        [0.0015, -0.0015, 0.15],
        [-0.0015, 0.0015, 0.15],
        [0.0015, 0.0015, 0.15],
    ]
    current = [[x, y, z - 0.01] for x, y, z in initial]
    return {
        "schema_version": 1,
        "status": "completed",
        "errors": [],
        "plan": build_timeline_probe_plan(seed=20261025),
        "settings_observed": {
            "stage_time_codes_per_second": 60.0,
            "timeline_time_codes_per_second": 60.0,
            "timeline_target_framerate": 60.0,
            "timeline_ticks_per_frame": 1,
            "timeline_play_every_frame": True,
            "use_fixed_time_stepping": True,
            "physics_time_steps_per_second": 60,
            "physics_min_frame_rate": 60,
        },
        "callbacks": callbacks,
        "emission": {
            "scheduled_before_callback_index": 6,
            "authored_after_callback_count": 5,
            "authored_at_app_update_index": 15,
            "particle_count": 4,
        },
        "initial_positions_m": initial,
        "current_positions_m": current,
        "movement": timeline_probe.movement_evidence(
            initial=initial, current=current
        ),
        "mesh_final": _valid_mesh(),
    }


def test_frozen_plan_is_one_tiny_timeline_owned_attempt():
    plan = build_timeline_probe_plan(seed=20261025)

    assert plan["schema_version"] == 1
    assert plan["seed"] == 20261025
    assert plan["attempt_limit"] == 1
    assert plan["fixed_hz"] == 60
    assert plan["fixed_dt_s"] == 1.0 / 60.0
    assert plan["target_callback_count"] == 12
    assert plan["emission_before_callback_index"] == 6
    assert plan["particles_per_emission"] == 4
    assert plan["simulation_owner"] == "timeline"
    assert plan["renderer_exercised"] is False
    assert len(plan["falsifiers"]) >= 5


def test_frozen_plan_rejects_another_seed():
    with pytest.raises(ValueError, match="20261025"):
        build_timeline_probe_plan(seed=7)


def test_movement_evidence_requires_stable_finite_rows_and_downward_motion():
    evidence = movement_evidence(
        initial=[[0.0, 0.0, 0.15], [0.01, 0.0, 0.15]],
        current=[[0.0, 0.0, 0.14], [0.01, 0.0, 0.13]],
    )

    assert evidence["row_count_matches"] is True
    assert evidence["finite_positions"] is True
    assert evidence["maximum_displacement_m"] == pytest.approx(0.02)
    assert evidence["mean_displacement_m"] == pytest.approx(0.015)
    assert evidence["mean_initial_z_m"] == pytest.approx(0.15)
    assert evidence["mean_current_z_m"] == pytest.approx(0.135)
    assert evidence["motion_observed"] is True
    assert evidence["gravity_moved_down"] is True


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda row: row["callbacks"].pop(), "callback_count_mismatch"),
        (
            lambda row: row["callbacks"][3].update(dt_s=1.0 / 30.0),
            "callback_dt_mismatch",
        ),
        (
            lambda row: row["callbacks"][5].update(timeline_time_s=0.5),
            "timeline_step_skipped_or_duplicated",
        ),
        (
            lambda row: row["callbacks"][5].update(app_update_index=15),
            "callback_app_update_not_lockstep",
        ),
        (
            lambda row: row["emission"].update(authored_after_callback_count=4),
            "emission_timing_mismatch",
        ),
        (
            lambda row: row.update(current_positions_m=row["initial_positions_m"]),
            "particle_motion_not_observed",
        ),
        (
            lambda row: row["mesh_final"].update(mesh_geometry_valid=False),
            "substantive_mesh_not_observed",
        ),
    ],
)
def test_outer_contract_fails_closed_on_substantive_mismatch(mutate, expected):
    receipt = _receipt()
    mutate(receipt)

    assert expected in validate_timeline_evidence(receipt, log_text="clean shutdown")


def test_clean_receipt_passes_and_nonfinite_positions_fail():
    assert validate_timeline_evidence(_receipt(), log_text="clean shutdown") == []

    receipt = _receipt()
    receipt["current_positions_m"][0][0] = math.nan
    assert "particle_positions_nonfinite" in validate_timeline_evidence(
        receipt, log_text="clean shutdown"
    )


def test_physx_cuda_errors_fail_even_when_receipt_claims_completion():
    failures = validate_timeline_evidence(
        _receipt(),
        log_text=(
            "[Error] [omni.physx.plugin] PhysX error: GPU kernel failed\n"
            "Cuda context manager error, simulation will be stopped\n"
        ),
    )

    assert failures == ["physx_gpu_or_cuda_runtime_error"]


def test_runtime_source_has_no_direct_simulation_or_renderer_path():
    source = (ROOT / "cli/twin/probes" / "physx_timeline_emission_probe.py").read_text(
        encoding="utf-8"
    )

    assert "attach_stage" not in source
    assert ".simulate(" not in source
    assert "fetch_results" not in source
    assert "omni.replicator" not in source
    assert "render_product" not in source
    callback = source.split("def on_physics_step", maxsplit=1)[1].split(
        "physics_step_subscription", maxsplit=1
    )[0]
    assert ".Set(" not in callback
    assert ".Create" not in callback


def test_outer_validator_recomputes_saved_mesh_and_all_artifact_hashes(tmp_path):
    receipt = _receipt()
    receipt["runtime_layer_specs"] = {
        "points_session_layer_spec": True,
        "points_root_layer_spec": False,
    }
    points = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    counts = [3, 3, 3, 3]
    indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    mesh = {
        "prim_path": "/World/ParticleSystem/Isosurface",
        "prim_present": True,
        "mesh_is_mesh": True,
        "points_m": points,
        "face_vertex_counts": counts,
        "face_vertex_indices": indices,
        **analyze_triangle_mesh_geometry(points, counts, indices),
        "mesh_sha256": validator._value_hash([points, counts, indices]),
    }
    receipt["mesh_final"] = {
        key: value
        for key, value in mesh.items()
        if key not in {"points_m", "face_vertex_counts", "face_vertex_indices"}
    }
    values = {
        "plan": receipt["plan"],
        "callbacks": receipt["callbacks"],
        "particles": {
            "initial_positions_m": receipt["initial_positions_m"],
            "current_positions_m": receipt["current_positions_m"],
            "runtime_layer_specs": receipt["runtime_layer_specs"],
        },
        "mesh": mesh,
    }
    receipt["artifacts"] = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        receipt["artifacts"][name] = {
            "path": path.name,
            "sha256": validator._sha256(path),
        }

    assert validator.validate_artifacts(receipt, artifact_root=tmp_path) == []

    particle_path = tmp_path / "particles.json"
    particle_value = json.loads(particle_path.read_text(encoding="utf-8"))
    particle_value["runtime_layer_specs"]["points_root_layer_spec"] = True
    particle_path.write_text(
        json.dumps(particle_value, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["artifacts"]["particles"]["sha256"] = validator._sha256(
        particle_path
    )
    assert validator.validate_artifacts(receipt, artifact_root=tmp_path) == [
        "particle_artifact_layer_spec_mismatch"
    ]
    particle_value["runtime_layer_specs"] = receipt["runtime_layer_specs"]
    particle_path.write_text(
        json.dumps(particle_value, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["artifacts"]["particles"]["sha256"] = validator._sha256(
        particle_path
    )

    (tmp_path / "mesh.json").write_text("{}\n", encoding="utf-8")
    assert validator.validate_artifacts(receipt, artifact_root=tmp_path) == [
        "mesh_artifact_hash_mismatch"
    ]


def _corrected_receipt() -> dict[str, object]:
    fixed_dt = 1.0 / 60.0
    initial = [
        [float(index % 4) * 0.003, float(index // 4) * 0.003, 0.15]
        for index in range(12)
    ]
    current = [[x, y, z - 0.01] for x, y, z in initial]
    return {
        "schema_version": 2,
        "status": "completed",
        "errors": [],
        "plan": timeline_probe.build_corrected_timeline_probe_plan(seed=20261025),
        "settings_observed": deepcopy(_receipt()["settings_observed"]),
        "timeline_initialization": {
            "play_committed_before_subscription": True,
            "timeline_time_before_subscription_s": 0.0,
            "callbacks_before_subscription": 0,
        },
        "callbacks": [
            {
                "callback_index": index,
                "dt_s": fixed_dt,
                "app_update_index": 20 + index,
                "timeline_time_s": index * fixed_dt,
            }
            for index in range(1, 25)
        ],
        "emission": {
            "scheduled_before_callback_index": 6,
            "authored_after_callback_count": 5,
            "authored_at_app_update_index": 25,
            "particle_count": 12,
            "position_source": (
                "a15_emission_ordinals_1_through_3_concatenated_unchanged"
            ),
        },
        "runtime_layer_specs": {
            "points_session_layer_spec": True,
            "points_root_layer_spec": False,
            "velocities_session_layer_spec": True,
            "velocities_root_layer_spec": False,
            "widths_session_layer_spec": True,
            "widths_root_layer_spec": False,
            "final_edit_target_is_session": True,
        },
        "initial_positions_m": initial,
        "current_positions_m": current,
        "movement": timeline_probe.movement_evidence(
            initial=initial, current=current
        ),
        "mesh_final": _valid_mesh(),
    }


def _corrected_log(receipt: dict[str, object]) -> str:
    observation = {
        "callback_count": len(receipt["callbacks"]),
        "errors": receipt["errors"],
        "movement": timeline_probe.movement_evidence(
            initial=receipt["initial_positions_m"],
            current=receipt["current_positions_m"],
        ),
        "runtime_layer_specs": receipt["runtime_layer_specs"],
        "mesh_geometry_valid": receipt["mesh_final"]["mesh_geometry_valid"],
        "mesh_sha256": receipt["mesh_final"]["mesh_sha256"],
    }
    lines = [
        "TIMELINE_A18_PLAN " + json.dumps(receipt["plan"], sort_keys=True),
        *(
            "TIMELINE_A18_CALLBACK " + json.dumps(row, sort_keys=True)
            for row in receipt["callbacks"]
        ),
        "TIMELINE_A18_OBSERVATION " + json.dumps(observation, sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def test_corrected_plan_preserves_a17_and_only_expands_frozen_evidence_window():
    original = build_timeline_probe_plan(seed=20261025)
    corrected = timeline_probe.build_corrected_timeline_probe_plan(seed=20261025)

    assert original["target_callback_count"] == 12
    assert original["particles_per_emission"] == 4
    assert corrected["schema_version"] == 2
    assert corrected["attempt_limit"] == 1
    assert corrected["target_callback_count"] == 24
    assert corrected["emission_before_callback_index"] == 6
    assert corrected["particles_per_emission"] == 12
    assert corrected["position_source"] == (
        "a15_emission_ordinals_1_through_3_concatenated_unchanged"
    )
    assert corrected["runtime_writeback_edit_target"] == "session_layer"
    assert corrected["subscription_phase"] == "after_play_commit_at_time_zero"
    assert corrected["writeback_evidence"] == (
        "changed composed points with session-only point spec, no root point spec, "
        "and no manual final points Set"
    )


def test_corrected_validator_requires_session_specs_and_post_preroll_subscription():
    assert (
        timeline_probe.validate_corrected_timeline_evidence(
            _corrected_receipt(), log_text="clean shutdown"
        )
        == []
    )

    wrong_layer = _corrected_receipt()
    wrong_layer["runtime_layer_specs"]["points_root_layer_spec"] = True
    assert "runtime_layer_spec_mismatch" in (
        timeline_probe.validate_corrected_timeline_evidence(
            wrong_layer, log_text="clean shutdown"
        )
    )

    wrong_init = _corrected_receipt()
    wrong_init["timeline_initialization"]["timeline_time_before_subscription_s"] = 1.0 / 60.0
    assert "timeline_subscription_initialization_mismatch" in (
        timeline_probe.validate_corrected_timeline_evidence(
            wrong_init, log_text="clean shutdown"
        )
    )

    wrong_absolute_time = _corrected_receipt()
    wrong_absolute_time["callbacks"][0]["timeline_time_s"] = 0.0
    assert "timeline_callback_absolute_time_mismatch" in (
        timeline_probe.validate_corrected_timeline_evidence(
            wrong_absolute_time, log_text="clean shutdown"
        )
    )


def test_corrected_runtime_keeps_updates_in_session_and_subscribes_after_play_commit():
    source = (
        ROOT / "cli/twin/probes" / "physx_timeline_emission_probe_a18.py"
    ).read_text(encoding="utf-8")

    assert "attach_stage" not in source
    assert ".simulate(" not in source
    assert "fetch_results" not in source
    assert "omni.replicator" not in source
    assert source.count("app.update()") == 2
    assert source.count("_app_update_in_session(") >= 4
    assert "corrected_callback_completion_failures(len(callbacks))" in source
    subscribe_at = source.index("subscribe_physics_step_events")
    assert source.index("timeline.play()") < subscribe_at
    assert source.index("timeline.commit()", source.index("timeline.play()")) < subscribe_at
    callback = source.split("def on_physics_step", maxsplit=1)[1].split(
        "physics_step_subscription", maxsplit=1
    )[0]
    assert ".Set(" not in callback
    assert ".Create" not in callback


def test_incomplete_corrected_callback_pump_is_a_runtime_error():
    assert timeline_probe.corrected_callback_completion_failures(24) == []
    assert timeline_probe.corrected_callback_completion_failures(23) == [
        "callback_count_incomplete"
    ]
    assert timeline_probe.corrected_callback_completion_failures(25) == [
        "callback_count_incomplete"
    ]


def test_a18_outer_validator_binds_exact_plan_callbacks_and_observation_to_log():
    receipt = _corrected_receipt()
    clean_log = _corrected_log(receipt)

    assert a18_validator.validate_log_binding(receipt, log_text=clean_log) == []

    wrong_callback = clean_log.replace(
        '"callback_index": 12', '"callback_index": 99', 1
    )
    assert "log_callback_trace_mismatch" in a18_validator.validate_log_binding(
        receipt, log_text=wrong_callback
    )
    wrong_observation = clean_log.replace(
        '"mesh_geometry_valid": true', '"mesh_geometry_valid": false', 1
    )
    assert "log_observation_marker_mismatch" in (
        a18_validator.validate_log_binding(
            receipt, log_text=wrong_observation
        )
    )
    failures = a18_validator.validate_log_binding(
        receipt, log_text="clean shutdown\n"
    )
    assert failures == [
        "log_plan_marker_mismatch",
        "log_callback_trace_mismatch",
        "log_observation_marker_mismatch",
    ]

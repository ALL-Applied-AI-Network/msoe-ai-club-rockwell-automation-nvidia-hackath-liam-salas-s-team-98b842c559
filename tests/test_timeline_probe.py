"""CPU contracts for the one-attempt timeline-owned PhysX emission probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.timeline_probe import (
    build_timeline_probe_plan,
    movement_evidence,
    validate_timeline_evidence,
)
from fluidmix.isosurface_geometry import analyze_triangle_mesh_geometry

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_physx_timeline_emission_probe",
    ROOT / "scripts" / "validate_physx_timeline_emission_probe.py",
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)


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
    source = (ROOT / "scripts" / "physx_timeline_emission_probe.py").read_text(
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

    (tmp_path / "mesh.json").write_text("{}\n", encoding="utf-8")
    assert validator.validate_artifacts(receipt, artifact_root=tmp_path) == [
        "mesh_artifact_hash_mismatch"
    ]

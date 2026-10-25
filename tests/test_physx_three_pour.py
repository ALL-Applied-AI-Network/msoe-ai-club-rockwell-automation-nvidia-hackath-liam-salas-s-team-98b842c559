"""CPU contracts for the bounded F2 one-system three-pour experiment."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_three_pour import (
    build_three_pour_plan,
    distribute_particle_count,
    validate_three_pour_receipt,
)


def test_frozen_plan_has_exact_recipe_order_counts_and_nominal_volume_limit():
    plan = build_three_pour_plan()

    assert plan["schema_version"] == 1
    assert plan["experiment"] == "physx_three_pour_f2"
    assert plan["seed"] == 20261025
    assert plan["fluid_rest_offset_m"] == 0.0015
    assert plan["nominal_particle_volume_ml"] == pytest.approx(0.027)
    assert [row["tank_id"] for row in plan["pours"]] == ["t200", "t400", "t100"]
    assert [row["programmed_volume_ml"] for row in plan["pours"]] == [10, 10, 40]
    assert [row["particle_count"] for row in plan["pours"]] == [370, 370, 1481]
    assert [row["emission_chunk_count"] for row in plan["pours"]] == [20, 20, 48]
    assert plan["totals"] == {
        "programmed_volume_ml": 60,
        "particle_count": 2221,
        "nominal_mass_convention_equivalent_ml": pytest.approx(59.967),
        "nominal_error_ml": pytest.approx(-0.033),
    }
    assert plan["mixing_assumption"] == (
        "fully_mixed_after_each_settled_pour_uncalibrated_render_parameter"
    )
    assert plan["claims"]["physically_displaced_volume"] is False
    assert plan["claims"]["absorbance_fitted"] is False


@pytest.mark.parametrize("total,chunks", [(370, 20), (1481, 48), (1, 1)])
def test_emission_chunk_distribution_is_positive_ordered_and_exact(total, chunks):
    counts = distribute_particle_count(total, chunks)

    assert len(counts) == chunks
    assert sum(counts) == total
    assert all(type(value) is int and value > 0 for value in counts)
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("total,chunks", [(0, 1), (1, 0), (3, 4), (True, 1), (1, False)])
def test_emission_chunk_distribution_rejects_invalid_or_empty_chunks(total, chunks):
    with pytest.raises((TypeError, ValueError)):
        distribute_particle_count(total, chunks)


def _passing_receipt() -> dict[str, object]:
    plan = build_three_pour_plan()
    stages = []
    cumulative = 0
    frame_hashes = []
    for ordinal, pour in enumerate(plan["pours"], start=1):
        cumulative += pour["particle_count"]
        digest = f"{ordinal:064x}"
        frame_hashes.append(digest)
        stages.append(
            {
                "ordinal": ordinal,
                "tank_id": pour["tank_id"],
                "programmed_volume_ml": pour["programmed_volume_ml"],
                "emitted_particle_count": pour["particle_count"],
                "cumulative_particle_count": cumulative,
                "finite_positions": True,
                "escaped_particle_count": 0,
                "stable": True,
                "stable_frame_index": ordinal * 10,
                "stable_frame_sha256": digest,
                "isosurface_nonempty": True,
                "used_gpu_memory_mib_before": 1000,
                "used_gpu_memory_mib_after": 1100,
                "memory_recovered_during_settle": True,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "physx_three_pour_f2",
        "status": "pass",
        "versions": {
            "isaac": "6.0.1.0",
            "kit": "110.1.2",
            "physx": "110.1.13",
        },
        "gpu_simulation": {
            "cuda_device_available": True,
            "scene_enable_gpu_dynamics": True,
            "broadphase": "GPU",
        },
        "particle_system": {
            "count": 1,
            "exactly_one": True,
            "mass_api_count": 0,
            "all_particle_sets_target_one_system": True,
            "collider_types": ["Cube"],
            "pbd_material_binding": True,
            "render_material_binding": True,
        },
        "pours": stages,
        "totals": {
            "programmed_volume_ml": 60,
            "emitted_particle_count": 2221,
            "final_particle_count": 2221,
            "nominal_mass_convention_equivalent_ml": 59.967,
            "nominal_error_ml": -0.033,
        },
        "isosurface": {
            "enabled": True,
            "mesh_point_count": 900,
            "mesh_face_count": 1700,
            "mesh_finite_points": True,
            "mesh_extent_nonempty": True,
        },
        "render": {
            "frame_count": 31,
            "frame_sha256": [f"{index:064x}" for index in range(1, 32)],
            "stage_frame_sha256": frame_hashes,
            "video_sha256": "f" * 64,
            "video_decodable": True,
            "video_decoded_frame_count": 31,
            "stage_frames_are_video_subsets": True,
        },
        "memory": {
            "hard_limit_mib": 40000,
            "peak_used_mib": 1400,
            "hard_limit_exceeded": False,
            "isosurface_enabled_seconds": 10.0,
            "isosurface_time_limit_seconds": 2700,
        },
        "timing": {"wall_seconds": 100.0, "wall_limit_seconds": 5400},
        "errors": [],
    }


def test_substantive_receipt_accepts_observed_three_pour_run():
    assert validate_three_pour_receipt(_passing_receipt()) == ()


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda r: r["particle_system"].update(count=2, exactly_one=False), "particle_system_count"),
        (lambda r: r["particle_system"].update(mass_api_count=1), "mass_api_present"),
        (lambda r: r["particle_system"].update(collider_types=["Cylinder"]), "collider_type_unsupported"),
        (lambda r: r["pours"][1].update(tank_id="t100"), "pour_order_or_volume"),
        (lambda r: r["pours"][0].update(escaped_particle_count=1), "particles_escaped"),
        (lambda r: r["pours"][2].update(stable=False), "stage_not_stable"),
        (lambda r: r["isosurface"].update(mesh_point_count=0), "isosurface_empty"),
        (lambda r: r["render"].update(video_decodable=False), "video_not_decodable"),
        (lambda r: r["render"].update(stage_frames_are_video_subsets=False), "stage_frame_not_subset"),
        (lambda r: r["memory"].update(hard_limit_exceeded=True), "memory_limit_exceeded"),
        (lambda r: r["timing"].update(wall_seconds=5400.1), "wall_time_exceeded"),
    ],
)
def test_substantive_receipt_rejects_real_failures(mutator, expected):
    receipt = deepcopy(_passing_receipt())
    mutator(receipt)

    assert expected in validate_three_pour_receipt(receipt)


def test_version_metadata_is_not_a_substantive_gate_failure():
    receipt = _passing_receipt()
    receipt["versions"]["physx"] = None

    assert validate_three_pour_receipt(receipt) == ()


def test_runner_plan_only_path_needs_no_isaac_and_refuses_overwrite(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    output = tmp_path / "plan"
    command = [sys.executable, str(script), "--out", str(output), "--plan-only"]

    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "plan.json").read_text(encoding="utf-8")) == build_three_pour_plan()
    repeated = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "output path already exists" in repeated.stderr


def test_runner_cli_freezes_capture_retry_default_and_reports_invalid_value(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--max-capture-tries MAX_CAPTURE_TRIES" in help_result.stdout
    assert "(default: 20)" in help_result.stdout

    invalid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(tmp_path / "bad"),
            "--plan-only",
            "--max-capture-tries",
            "0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert invalid.stderr.rstrip().endswith(
        "physx_three_pour_sequence.py: error: max_capture_tries must be a positive integer"
    )

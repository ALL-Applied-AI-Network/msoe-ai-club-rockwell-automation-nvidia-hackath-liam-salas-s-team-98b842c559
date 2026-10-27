from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.physx_train_residual_manifest_v2 import (
    build_train_residual_artifacts,
)
from fluidmix.paint_ball_emitter import build_runtime_capture_schedule


ROOT = Path(__file__).resolve().parents[1]


def _keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def test_train_manifest_uses_two_reference_and_four_target_three_vial_batches():
    artifacts = build_train_residual_artifacts(repo_root=ROOT)
    render = artifacts["render-schedule-plan.json"]

    assert render["partition"] == "train"
    assert render["family"]["previous_sim_recipe_id"] == "S0437"
    assert render["family"]["current_sim_recipe_id"] == "S0117"
    assert render["family"]["previous_role"] == "train"
    assert render["family"]["current_role"] == "train"
    assert render["family"]["transition_role"] == "train"
    assert len(render["batches"]) == 6
    assert {row["vial_count"] for row in render["batches"]} == {3}
    assert [row["batch_role"] for row in render["batches"]] == [
        "reference_only",
        "reference_only",
        "scored_target",
        "scored_target",
        "scored_target",
        "scored_target",
    ]
    reference_batches = [
        row for row in render["batches"] if row["batch_role"] == "reference_only"
    ]
    target_batches = [
        row for row in render["batches"] if row["batch_role"] == "scored_target"
    ]
    assert [row["render_seed_by_vial"] for row in reference_batches] == [
        [330001, 330002, 330003],
        [330004, 330005, 330006],
    ]
    assert {tuple(row["render_seed_by_vial"]) for row in target_batches} == {
        (330101, 330102, 330103)
    }
    assert render["denominators"] == {
        "families": 1,
        "batches": 6,
        "reference_only_batches": 2,
        "scored_target_batches": 4,
        "vials": 18,
        "reference_only_vials": 6,
        "scored_target_vials": 12,
        "keep_vials": 12,
        "discard_vials": 6,
        "scored_target_keep_vials": 6,
        "scored_target_discard_vials": 6,
        "execution_schedules": 18,
        "stage_images_planned": 72,
        "accepted_render_artifacts": 0,
    }
    assert render["render_gate"]["render_authorized"] is False
    assert render["render_gate"]["training_eligible"] is False


def test_train_manifest_uses_corrected_leading_residue_and_packing_calibration():
    render = build_train_residual_artifacts(repo_root=ROOT)[
        "render-schedule-plan.json"
    ]
    schedules = render["execution_schedules"]
    affected = next(
        row
        for row in schedules
        if row["case_binding"]["vial_ordinal"] == 1
        and row["case_binding"]["residual_volume_ml"] == 1.0
    )

    assert affected["physical_sequence"] == {
        "premixed_before_vial": False,
        "mixing_location": "vial",
        "dispense_order": ["t200", "t400", "t100"],
        "previous_final_tank_id": "t100",
    }
    assert affected["particle_count_calibration"]["effective_particle_volume_ml"] == pytest.approx(
        0.005771707871971808
    )
    assert affected["particle_count_calibration"]["mode"] == (
        "resolution_scaled_effective_volume_hypothesis"
    )
    assert affected["particle_count_calibration"]["validated_at_2p0mm_resolution"] is False
    assert affected["particle_count_calibration"]["directly_measured"] is False
    assert affected["label_provenance"]["nominal_particle_volume_ml"] == 0.008
    assert affected["particle_resolution_profile"]["profile_id"] == "c21_2p0mm"
    assert all(
        event["source_reference_particle_count"] > 0
        for pour in affected["pours"]
        for event in pour["emission_schedule"]
    )
    assert affected["timing"]["physics_hz"] == 120
    assert affected["timing"]["pour_end_steps"] == [2704, 4260, 5332]
    assert affected["timing"]["fill_plus_0p3_step"] == 5368
    assert affected["timing"]["fill_plus_3p0_step"] == 5692
    assert affected["totals"]["particle_count_conserved"] is True
    assert affected["totals"]["programmed_particle_count"] == 12994
    assert [
        (segment["segment_role"], segment["source_tank_id"])
        for pour in affected["pours"]
        for segment in pour["emission_segments"]
    ] == [
        ("leading_residual", "t100"),
        ("current_tank", "t200"),
        ("leading_residual", "t200"),
        ("current_tank", "t400"),
        ("leading_residual", "t400"),
        ("current_tank", "t100"),
    ]
    assert all(
        len({tuple(event["particle_composition"]) for event in segment["emission_schedule"]}) <= 1
        for pour in affected["pours"]
        for segment in pour["emission_segments"]
    )


def test_every_train_case_exposes_each_source_pure_stream_for_two_capture_frames():
    render = build_train_residual_artifacts(repo_root=ROOT)[
        "render-schedule-plan.json"
    ]
    capture_schedule_by_case = {}
    for schedule in render["execution_schedules"]:
        capture_schedule_by_case[schedule["case_binding"]["case_id"]] = (
            build_runtime_capture_schedule(
                {
                    "timing": {
                        "physics_hz": schedule["timing"]["physics_hz"],
                        "capture_hz": 10,
                        "fill_end_step": schedule["timing"]["fill_end_step"],
                    },
                    "pours": schedule["pours"],
                },
                capture_stride_steps=312,
            )
        )

    assert len(capture_schedule_by_case) == 18
    assert sorted(
        len(value["captures"]) for value in capture_schedule_by_case.values()
    ) == [22] * 12 + [30] * 6
    assert {
        value["capture_stride_steps"]
        for value in capture_schedule_by_case.values()
    } == {312}
    assert {value["capture_hz"] for value in capture_schedule_by_case.values()} == {
        120 / 312
    }
    assert [
        max(
            sum(
                row.get("kind") == "in_pour"
                and row.get("pour_ordinal") == pour_ordinal
                for row in value["captures"]
            )
            for value in capture_schedule_by_case.values()
        )
        for pour_ordinal in (1, 2, 3)
    ] == [11, 7, 6]

    smallest = next(
        schedule
        for schedule in render["execution_schedules"]
        if schedule["case_binding"]["profile_id"] == "target_all_three"
        and schedule["case_binding"]["vial_ordinal"] == 3
    )
    assert smallest["case_binding"]["residual_volume_ml"] == 0.4
    assert [
        pour["emission_segments"][0]["particle_count"]
        for pour in smallest["pours"]
    ] == [69, 69, 69]
    capture_rows = capture_schedule_by_case[smallest["case_binding"]["case_id"]][
        "captures"
    ]
    assert [
        sum(
            row.get("kind") == "in_pour"
            and row.get("pour_ordinal") == pour_ordinal
            and row.get("source_segment_role") == "leading_residual"
            for row in capture_rows
        )
        for pour_ordinal in (1, 2, 3)
    ] == [2, 2, 2]


def test_model_inputs_recursively_exclude_oracle_fields_and_labels_are_separate():
    artifacts = build_train_residual_artifacts(repo_root=ROOT)
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance.json"]

    forbidden = ("residual", "disposition", "label", "oracle", "severity", "profile")
    assert not any(
        fragment in key.lower()
        for key in _keys(model)
        for fragment in forbidden
    )
    assert len(model["vial_records"]) == 18
    assert len(labels["vial_labels"]) == 18
    assert sum(row["expected_disposition"] == "keep" for row in labels["vial_labels"]) == 12
    assert sum(row["expected_disposition"] == "discard" for row in labels["vial_labels"]) == 6
    scored = [
        row for row in labels["vial_labels"] if row["batch_role"] == "scored_target"
    ]
    assert len(scored) == 12
    assert sum(row["expected_disposition"] == "keep" for row in scored) == 6
    assert sum(row["expected_disposition"] == "discard" for row in scored) == 6
    assert all(row["excluded_from_model_inputs"] is True for row in labels["vial_labels"])
    assert all(row["excluded_from_deploy_inputs"] is True for row in labels["vial_labels"])


def test_builder_script_writes_three_separated_artifacts_outside_repo(tmp_path):
    output = tmp_path / "plans"
    script = ROOT / "scripts/build_physx_train_residual_manifest_v2.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "label-provenance.json",
        "model-input-plan.json",
        "render-schedule-plan.json",
    ]
    for path in output.iterdir():
        json.loads(path.read_text())

    repeated = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "refusing existing output directory" in repeated.stderr

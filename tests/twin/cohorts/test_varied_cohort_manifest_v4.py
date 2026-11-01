from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.cohorts.varied_cohort_manifest_v4 import (  # noqa: E402
    RUN_SPECS,
    SOURCE_RELATIVE_PATH,
    build_varied_cohort_artifacts,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_v4_source_rebuild_has_exact_runs_seeds_profiles_and_recipe_bindings():
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    render = artifacts["render-schedule-plan.json"]
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance.json"]

    assert render["schema_version"] == 4
    assert render["artifact_type"] == "physx_f2_varied_run_residual_render_plan_v4"
    assert model["schema_version"] == labels["schema_version"] == 4
    assert model["split_version"] == "whole-run-varied-v4"
    assert len(render["runs"]) == len(render["batches"]) == 10
    assert len(render["execution_schedules"]) == len(model["vial_records"]) == len(labels["vial_labels"]) == 30
    assert {
        name: _canonical_sha256(value) for name, value in artifacts.items()
    } == {
        "render-schedule-plan.json": "e5e63594073300c158c9c97fab6aa48ace3644e8dcd6c46fd1acdf076d3970f9",
        "model-input-plan.json": "ffa82189482de986a969886ffda1fd322067e9ce8864163d86b0f6a4d6af2a47",
        "label-provenance.json": "400fcda7d10432cf01c0d1eea643cb06c4d403f4efb83b956a47e1185992439e",
    }

    expected = [
        ("RUN-V4-TRAIN-01", 52001, "train", "14", "5", (520101, 520102, 520103), (0.0, 0.0, 0.0)),
        ("RUN-V4-TRAIN-02", 52002, "train", "5", "6", (520201, 520202, 520203), (9.0, 0.0, 0.0)),
        ("RUN-V4-TRAIN-03", 52003, "train", "6", "7", (520301, 520302, 520303), (9.0, 4.5, 0.0)),
        ("RUN-V4-TRAIN-04", 52004, "train", "7", "8", (520401, 520402, 520403), (9.0, 4.5, 2.25)),
        ("RUN-V4-VALIDATION-01", 52005, "validation", "8", "9", (520501, 520502, 520503), (0.0, 0.0, 0.0)),
        ("RUN-V4-VALIDATION-02", 52006, "validation", "9", "10", (520601, 520602, 520603), (6.0, 0.0, 0.0)),
        ("RUN-V4-VALIDATION-03", 52007, "validation", "10", "11", (520701, 520702, 520703), (6.0, 3.0, 1.5)),
        ("RUN-V4-EVALUATION-01", 52008, "evaluation", "11", "12", (520801, 520802, 520803), (0.0, 0.0, 0.0)),
        ("RUN-V4-EVALUATION-02", 52009, "evaluation", "12", "13", (520901, 520902, 520903), (6.0, 3.0, 0.0)),
        ("RUN-V4-EVALUATION-03", 52010, "evaluation", "13", "14", (521001, 521002, 521003), (6.0, 3.0, 1.5)),
    ]
    assert list(RUN_SPECS) == expected
    assert [
        (row["run_id"], row["run_seed"], row["split_role"], row["initial_previous_recipe_id"], row["current_recipe_id"])
        for row in render["runs"]
    ] == [row[:5] for row in expected]
    assert [tuple(row["render_seed_by_vial"]) for row in render["batches"]] == [row[5] for row in expected]

    schedules_by_run: dict[str, list[dict[str, object]]] = {}
    for schedule in render["execution_schedules"]:
        schedules_by_run.setdefault(schedule["case_binding"]["run_id"], []).append(schedule)
    for expected_row in expected:
        run_id, _, _, previous_id, current_id, _, profile = expected_row
        schedules = schedules_by_run[run_id]
        assert [row["case_binding"]["residual_volume_ml"] for row in schedules] == list(profile)
        assert all(
            row["case_binding"]["current_recipe_id"] == current_id
            and row["case_binding"]["previous_recipe_id"] == (previous_id if row["case_binding"]["vial_ordinal"] == 1 else current_id)
            for row in schedules
        )


def test_v4_counts_labels_allowlist_conservation_and_monotone_profiles():
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    render = artifacts["render-schedule-plan.json"]
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance.json"]

    assert render["denominators"] == {
        "runs": 10,
        "batches": 10,
        "vials": 30,
        "train_vials": 12,
        "validation_vials": 9,
        "evaluation_vials": 9,
        "keep_vials": 15,
        "discard_vials": 15,
        "accepted_render_artifacts": 0,
    }
    assert labels["denominators"] == {"keep_vials": 15, "discard_vials": 15}
    forbidden = ("residual", "label", "oracle", "disposition", "severity", "palette")
    assert all(not any(fragment in key.lower() for fragment in forbidden) for row in model["vial_records"] for key in row)
    assert all(row["excluded_from_model_inputs"] and row["excluded_from_deploy_inputs"] for row in labels["vial_labels"])
    assert [row["expected_disposition"] for row in labels["vial_labels"]].count("keep") == 15
    assert [row["expected_disposition"] for row in labels["vial_labels"]].count("discard") == 15

    for split in ("train", "validation", "evaluation"):
        for ordinal in (1, 2, 3):
            dispositions = {
                row["expected_disposition"]
                for row in labels["vial_labels"]
                if row["split_role"] == split and row["vial_ordinal"] == ordinal
            }
            assert dispositions == {"keep", "discard"}
    for run_id, schedules in {
        run_id: [row for row in render["execution_schedules"] if row["case_binding"]["run_id"] == run_id]
        for run_id in {row["case_binding"]["run_id"] for row in render["execution_schedules"]}
    }.items():
        profile = [row["case_binding"]["residual_volume_ml"] for row in schedules]
        assert profile == sorted(profile, reverse=True)
    assert all(
        schedule["totals"]["particle_count_conserved"] is True
        and schedule["totals"]["programmed_particle_count"] == schedule["totals"]["emitted_particle_count"] > 0
        for schedule in render["execution_schedules"]
    )


def test_v4_ids_palettes_and_source_hashes_are_unique_and_reproducible():
    first = build_varied_cohort_artifacts(repo_root=ROOT)
    second = build_varied_cohort_artifacts(repo_root=ROOT)
    assert first == second
    render = first["render-schedule-plan.json"]
    assert render["source_sha256"][SOURCE_RELATIVE_PATH] == hashlib.sha256(
        (ROOT / SOURCE_RELATIVE_PATH).read_bytes()
    ).hexdigest()
    assert len({row["run_id"] for row in render["runs"]}) == 10
    assert len({row["run_seed"] for row in render["runs"]}) == 10
    assert len({seed for row in render["batches"] for seed in row["render_seed_by_vial"]}) == 30
    assert len({row["case_binding"]["case_id"] for row in render["execution_schedules"]}) == 30
    assert len({row["case_binding"]["batch_id"] for row in render["execution_schedules"]}) == 10
    assert all(
        render["execution_schedule_sha256_by_case"][schedule["case_binding"]["case_id"]]
        == _canonical_sha256(schedule)
        for schedule in render["execution_schedules"]
    )


def test_v4_cli_writes_three_source_rebuilt_artifacts(tmp_path):
    import subprocess

    output = tmp_path / "manifest"
    subprocess.run(
        [sys.executable, str(ROOT / "cli/twin/cohorts" / "build_physx_varied_cohort_manifest_v4.py"), "--out", str(output)],
        check=True,
        cwd=ROOT,
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "label-provenance.json",
        "model-input-plan.json",
        "render-schedule-plan.json",
    ]
    assert json.loads((output / "render-schedule-plan.json").read_text())["schema_version"] == 4


def test_v4_runtime_loader_branches_with_contract_and_palette_schema(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("physx_sequence_v4_test", ROOT / "cli/twin/physx" / "physx_three_pour_sequence.py")
    assert spec is not None and spec.loader is not None
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    plan = artifacts["render-schedule-plan.json"]
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sequence, "_bind_accepted_qualification_receipt", lambda *args, **kwargs: {"receipt_sha256": "b" * 64})
    batch = next(row for row in plan["batches"] if case_id in row["case_ids"])
    args = sequence.build_parser().parse_args([
        "--out", str(tmp_path / "out"), "--stage", str(stage), "--seed", str(batch["render_seed_by_vial"][0]),
        "--count-driven-twin-liquid-mesh", "--height-banded-body", "--realtime-video",
        "--run-palette-id", batch["run_palette_id"], "--run-palette-seed", str(batch["run_palette_seed"]),
        "--train-render-plan", str(path), "--train-render-plan-sha256", _canonical_sha256(plan),
        "--train-case-id", case_id, "--accepted-qualification-receipt", str(qualification),
        "--accepted-qualification-receipt-sha256", "b" * 64,
    ])
    contract = sequence._execution_schedule_contract(args)
    runtime = sequence._runtime_paint_ball_plan(args)
    assert contract["contract_kind"] == "varied_run_residual_v4"
    assert contract["run_palette_binding"]["run_palette_id"] == batch["run_palette_id"]
    assert runtime["execution_schedule"]["adapter_schema_version"] == 4

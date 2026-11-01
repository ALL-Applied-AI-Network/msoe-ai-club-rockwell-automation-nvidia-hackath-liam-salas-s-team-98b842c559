from __future__ import annotations

from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.cohorts.full_transition_manifest import (  # noqa: E402
    build_default_full_transition_source_spec,
    build_full_transition_artifacts,
)
from twin.cohorts.one_run_manifest_v1 import (  # noqa: E402
    build_one_run_source_artifacts,
    rebuild_bound_one_run_artifacts,
)
from twin.physx.run_variation import (  # noqa: E402
    bind_tank_colour_resolution_to_artifacts,
)
from twin.postrender.wp1_dp3_fallback import (  # noqa: E402
    TARGETS,
    build_dp3_fallback_plan_artifacts,
    canonical_sha256,
    load_dp3_fallback_plan_bundle,
)


def _frozen_run1_plan() -> dict[str, object]:
    artifacts = build_full_transition_artifacts(
        build_default_full_transition_source_spec(repo_root=ROOT), repo_root=ROOT
    )
    binding = artifacts["full-transition-manifest.json"]["run_bindings"][0]
    return artifacts[binding["render_plan_path"]]


def _self_consistent_wrong_bundle(
    frozen: dict[str, object], mutation: str
) -> dict[str, dict[str, object]]:
    expected = build_dp3_fallback_plan_artifacts(frozen, target_ordinal=5)
    source = deepcopy(expected["source-run-spec.json"])
    palette = deepcopy(expected["render-schedule-plan.json"]["tank_colour_resolution"])
    if mutation == "programmed_ml":
        source["recipes"][0]["programmed_ml"]["t200"] += 1.0
    elif mutation == "run_id":
        source["run_id"] = "WP1-DP3-FALLBACK-WRONG-RUN"
    else:
        all_artifacts = build_full_transition_artifacts(
            build_default_full_transition_source_spec(repo_root=ROOT), repo_root=ROOT
        )
        run2 = all_artifacts["full-transition-manifest.json"]["run_bindings"][1]
        palette = deepcopy(
            all_artifacts[run2["render_plan_path"]]["tank_colour_resolution"]
        )
    derived = bind_tank_colour_resolution_to_artifacts(
        build_one_run_source_artifacts(source), palette
    )
    return {"source-run-spec.json": source, **derived}


@pytest.mark.parametrize("target_ordinal", [5, 9])
def test_fallback_plan_is_real_one_run_contract_with_fresh_deterministic_identity(
    target_ordinal: int,
) -> None:
    original = _frozen_run1_plan()
    artifacts = build_dp3_fallback_plan_artifacts(
        original, target_ordinal=target_ordinal
    )
    plan = artifacts["render-schedule-plan.json"]
    rebuilt = rebuild_bound_one_run_artifacts(plan)
    target = TARGETS[target_ordinal]
    schedule = plan["execution_schedules"][0]
    binding = schedule["case_binding"]

    assert rebuilt == {
        key: value for key, value in artifacts.items() if key != "source-run-spec.json"
    }
    assert binding["case_id"] != target["original_case_id"]
    assert binding["vial_ordinal"] == 1
    assert binding["current_recipe_id"] == target["recipe_id"]
    assert binding["previous_recipe_id"] == target["recipe_id"]
    assert binding["residual_volume_ml"] == 3.0
    assert plan["batches"][0]["render_seed_by_vial"] == [target["render_seed"]]
    assert plan["batches"][0]["nuisance_group_ids"] == [
        target["nuisance_group_id"]
    ]
    assert plan["execution_schedule_sha256_by_case"][binding["case_id"]] == (
        canonical_sha256(schedule)
    )
    original_schedule = next(
        row
        for row in original["execution_schedules"]
        if row["case_binding"]["vial_ordinal"] == target_ordinal
    )
    assert schedule["model_input_record"]["programmed_ml"] == (
        original_schedule["model_input_record"]["programmed_ml"]
    )
    assert plan["tank_colour_resolution"] == original["tank_colour_resolution"]


def test_fallback_plan_targets_are_distinct_and_reproducible() -> None:
    original = _frozen_run1_plan()
    first = build_dp3_fallback_plan_artifacts(original, target_ordinal=5)
    repeated = build_dp3_fallback_plan_artifacts(original, target_ordinal=5)
    other = build_dp3_fallback_plan_artifacts(original, target_ordinal=9)

    assert first == repeated
    assert first["render-schedule-plan.json"]["execution_schedules"][0][
        "case_binding"
    ]["case_id"] != other["render-schedule-plan.json"]["execution_schedules"][0][
        "case_binding"
    ]["case_id"]


def test_plan_bundle_loader_rejects_source_or_full_schedule_drift(tmp_path: Path) -> None:
    artifacts = build_dp3_fallback_plan_artifacts(
        _frozen_run1_plan(), target_ordinal=5
    )
    for name, value in artifacts.items():
        (tmp_path / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    frozen = _frozen_run1_plan()
    assert load_dp3_fallback_plan_bundle(
        tmp_path, target_ordinal=5, frozen_run1_plan=frozen
    )["case_id"]

    plan_path = tmp_path / "render-schedule-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["execution_schedules"][0]["case_binding"]["residual_volume_ml"] = 1.5
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes|rebuild|differs"):
        load_dp3_fallback_plan_bundle(
            tmp_path, target_ordinal=5, frozen_run1_plan=frozen
        )


@pytest.mark.parametrize("mutation", ["programmed_ml", "run_id", "run2_palette"])
def test_plan_bundle_loader_rejects_self_consistent_nonfrozen_derivation(
    tmp_path: Path, mutation: str
) -> None:
    frozen = _frozen_run1_plan()
    for name, value in _self_consistent_wrong_bundle(frozen, mutation).items():
        (tmp_path / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="exact frozen run-1 derivation"):
        load_dp3_fallback_plan_bundle(
            tmp_path, target_ordinal=5, frozen_run1_plan=frozen
        )


def test_plan_cli_writes_two_supervisor_compatible_source_bound_plans(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "manifest"
    manifest_root.mkdir()
    artifacts = build_full_transition_artifacts(
        build_default_full_transition_source_spec(repo_root=ROOT), repo_root=ROOT
    )
    for relative, value in artifacts.items():
        path = manifest_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "fallback-plans"
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/twin/postrender/wp1_dp3_fallback.py"),
            "plan",
            "--manifest-root",
            str(manifest_root),
            "--output-root",
            str(output),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    index = json.loads((output / "fallback-plan-index.json").read_text())
    assert index["render_authorized"] is False
    assert index["cross_case_visibility_aggregation"] == "approval_required"
    assert [row["target_vial_ordinal"] for row in index["cases"]] == [5, 9]
    for ordinal in (5, 9):
        bundle = load_dp3_fallback_plan_bundle(
            output / f"ordinal-{ordinal:02d}",
            target_ordinal=ordinal,
            frozen_run1_plan=artifacts[
                artifacts["full-transition-manifest.json"]["run_bindings"][0][
                    "render_plan_path"
                ]
            ],
        )
        assert bundle["values"]["render-schedule-plan.json"]["artifact_type"] == (
            "physx_f2_source_bound_one_run_render_plan_v1"
        )


def test_generated_plan_loads_through_existing_cohort_supervisor(tmp_path: Path) -> None:
    artifacts = build_dp3_fallback_plan_artifacts(
        _frozen_run1_plan(), target_ordinal=5
    )
    path = tmp_path / "render-schedule-plan.json"
    path.write_text(
        json.dumps(artifacts["render-schedule-plan.json"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "wp1_dp3_supervisor_probe",
        ROOT / "cli/twin/cohorts/run_physx_train_residual_cohort.py",
    )
    assert spec is not None and spec.loader is not None
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    loaded = supervisor._load_bound_plan(
        path, canonical_sha256(artifacts["render-schedule-plan.json"])
    )
    assert loaded == artifacts["render-schedule-plan.json"]

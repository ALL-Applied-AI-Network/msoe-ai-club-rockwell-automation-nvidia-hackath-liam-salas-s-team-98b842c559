"""Contracts for the synthetic F2 keep/discard case manifest."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_case_plan import (  # noqa: E402
    build_physx_case_plan,
    validate_physx_case_plan,
)


EXPECTED_PROFILE_LABELS = {
    "clean": ("keep", "keep", "keep"),
    "first_only": ("discard", "keep", "keep"),
    "first_two": ("discard", "discard", "keep"),
    "all_three": ("discard", "discard", "discard"),
}


def _expected_recipe_timing(programmed_ml):
    intercept = 8.16571903
    coefficients = {"t100": 0.07633141, "t200": 0.35935439, "t400": 0.19177068}
    durations = {
        tank: int(
            math.floor(
                (intercept + coefficients[tank] * programmed_ml[tank]) * 60 + 0.5
            )
        )
        for tank in ("t200", "t400", "t100")
    }
    stage1 = durations["t200"]
    stage2 = stage1 + durations["t400"]
    fill_end = stage2 + durations["t100"]
    return durations, [stage1, stage2, fill_end + 18], fill_end


def test_plan_has_realistic_three_vial_batches_and_balanced_labels_by_split():
    plan = build_physx_case_plan()

    assert plan["schema_version"] == 1
    assert plan["artifact_type"] == "physx_f2_keep_discard_case_plan"
    assert plan["batch_size_vials"] == 3
    assert plan["denominators"] == {
        "families": 4,
        "train_families": 1,
        "val_families": 1,
        "test_families": 2,
        "batches": 16,
        "vials": 48,
        "stage_captures": 144,
        "keep_vials": 24,
        "discard_vials": 24,
        "by_partition": {
            "train": {"batches": 4, "vials": 12, "keep": 6, "discard": 6},
            "val": {"batches": 4, "vials": 12, "keep": 6, "discard": 6},
            "test": {"batches": 8, "vials": 24, "keep": 12, "discard": 12},
        },
    }
    assert len(plan["batches"]) == 16
    assert len(plan["vials"]) == 48
    assert len(plan["stage_captures"]) == 144


def test_each_family_has_four_render_parameter_profiles_and_exact_vial_labels():
    plan = build_physx_case_plan()
    families = {row["family_id"]: row for row in plan["families"]}
    batches = plan["batches"]
    vials_by_batch = {
        batch["batch_id"]: [
            vial for vial in plan["vials"] if vial["batch_id"] == batch["batch_id"]
        ]
        for batch in batches
    }

    assert {row["family_id"] for row in batches} == set(families)
    for family_id, family in families.items():
        family_batches = [row for row in batches if row["family_id"] == family_id]
        assert {row["profile_id"] for row in family_batches} == set(EXPECTED_PROFILE_LABELS)
        assert all(row["partition"] == family["partition"] for row in family_batches)
        for batch in family_batches:
            vials = sorted(vials_by_batch[batch["batch_id"]], key=lambda row: row["vial_ordinal"])
            assert [row["expected_disposition"] for row in vials] == list(
                EXPECTED_PROFILE_LABELS[batch["profile_id"]]
            )
            assert [row["synthetic_previous_recipe_blend"] for row in vials] == batch[
                "synthetic_previous_recipe_blend_by_vial"
            ]
            assert all(row["label_source"] == "render_parameters" for row in vials)
            assert all(row["ground_truth_not_model_input"] is True for row in vials)


def test_every_vial_binds_exact_recipes_runtime_placeholder_and_three_f2_captures():
    plan = build_physx_case_plan()
    batch_by_id = {row["batch_id"]: row for row in plan["batches"]}
    captures_by_vial = {
        vial["vial_id"]: [
            capture
            for capture in plan["stage_captures"]
            if capture["vial_id"] == vial["vial_id"]
        ]
        for vial in plan["vials"]
    }

    assert plan["f2_runtime_binding"] == {
        "binding_status": "pending_accepted_f2_receipt",
        "planned_environment_profile": "brev-x86_64-isaac601",
        "host_id": None,
        "isaac_sim_version": None,
        "source_commit": None,
        "f2_receipt_sha256": None,
        "palette_smoke_receipt_sha256": "c34da1875d098cc791ce2237d40cf4b32fcac14acab98f247dc32c98b9993e17",
        "palette_smoke_visual_review_sha256": "8d8e1f0077340d9e64ce954bd32d365d6b6c4ca3252ac72b6977feb63190fe49",
        "palette_smoke_decision": "fail",
        "same_host_version_commit_for_every_case": True,
    }
    for vial in plan["vials"]:
        batch = batch_by_id[vial["batch_id"]]
        assert vial["previous_sim_recipe_id"] == batch["previous_sim_recipe_id"]
        assert vial["current_sim_recipe_id"] == batch["current_sim_recipe_id"]
        assert vial["previous_programmed_ml"] == batch["previous_programmed_ml"]
        assert vial["current_programmed_ml"] == batch["current_programmed_ml"]
        assert vial["stage_capture_ids"] == [
            f"{vial['vial_id']}-AFTER_T200",
            f"{vial['vial_id']}-AFTER_T400",
            f"{vial['vial_id']}-FINAL_PLUS_0P3S",
        ]
        captures = sorted(captures_by_vial[vial["vial_id"]], key=lambda row: row["stage_ordinal"])
        assert [row["stage_id"] for row in captures] == [
            "AFTER_T200",
            "AFTER_T400",
            "FINAL_PLUS_0P3S",
        ]
        durations, expected_steps, fill_end = _expected_recipe_timing(
            batch["current_programmed_ml"]
        )
        assert [row["physics_step"] for row in captures] == expected_steps
        assert all(
            row["current_programmed_ml"] == batch["current_programmed_ml"]
            and row["timing_model_source"]
            == "video5_recipe_median_active_tank_linear_fit_v1"
            and row["timing_interpretation"]
            == "real_like_approximation_not_measured_tank_boundaries"
            for row in captures
        )
        assert batch["timing"]["duration_steps_by_tank"] == durations
        assert batch["timing"]["stage_capture_steps"] == expected_steps
        assert batch["timing"]["fill_end_step"] == fill_end
        assert vial["timing"] == batch["timing"]
        assert all(row["render_status"] == "unrendered" for row in captures)
        assert all(row["liquid_semantic_class"] is None for row in captures)


def test_selected_current_recipes_produce_multiple_formula_derived_timelines():
    plan = build_physx_case_plan()
    timelines = {
        tuple(batch["timing"]["stage_capture_steps"]) for batch in plan["batches"]
    }

    assert len(timelines) >= 2
    assert (706, 1311, 2002) in timelines
    assert (1352, 2130, 2684) in timelines
    assert all("physics_step" not in stage for stage in plan["stage_contract"])
    assert plan["real_like_timing_model"] == {
        "physics_hz": 60,
        "active_tank_intercept_s": 8.16571903,
        "tank_coefficient_s_per_ml": {
            "t100": 0.07633141,
            "t200": 0.35935439,
            "t400": 0.19177068,
        },
        "duration_step_rule": "nearest integer of (intercept_s + coefficient_s_per_ml * current_tank_ml) * physics_hz",
        "pour_order": ["t200", "t400", "t100"],
        "primary_final_offset_steps": 18,
        "primary_final_offset_s": 0.3,
        "source": "video5_recipe_median_active_tank_linear_fit_v1",
        "source_fit_r2": 0.92595,
        "source_fit_rmse_s": 2.371,
        "source_fit_loocv_rmse_s": 4.765,
        "interpretation": "real_like_approximation_not_measured_tank_boundaries",
    }


def test_family_batch_recipe_cell_seed_and_nuisance_never_cross_partition():
    plan = build_physx_case_plan()
    family_partition = {row["family_id"]: row["partition"] for row in plan["families"]}
    recipe_partitions: dict[str, set[str]] = {}
    cell_partitions: dict[str, set[str]] = {}
    seed_partitions: dict[int, set[str]] = {}
    for family in plan["families"]:
        for binding in family["recipe_source_binding"].values():
            recipe_partitions.setdefault(binding["sim_recipe_id"], set()).add(family["partition"])
            cell_partitions.setdefault(binding["cell"], set()).add(family["partition"])
    for batch in plan["batches"]:
        assert batch["partition"] == family_partition[batch["family_id"]]
        for seed in batch["render_seed_by_vial"]:
            seed_partitions.setdefault(seed, set()).add(batch["partition"])
    assert all(len(values) == 1 for values in recipe_partitions.values())
    assert all(len(values) == 1 for values in cell_partitions.values())
    assert all(len(values) == 1 for values in seed_partitions.values())
    assert plan["partition_policy"]["test_data_used_for_training_or_tuning"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda plan: plan["families"][0].update(partition="test"),
        lambda plan: plan["batches"][0].update(partition="val"),
        lambda plan: plan["batches"][0].update(
            synthetic_previous_recipe_blend_by_vial=[0.1, 0.0, 0.0]
        ),
        lambda plan: plan["vials"][0].update(expected_disposition="discard"),
        lambda plan: plan["vials"][0].update(label_source="camera_score"),
        lambda plan: plan["stage_captures"][0].update(physics_step=705),
        lambda plan: plan["batches"][0]["timing"].update(fill_end_step=1),
        lambda plan: plan["f2_runtime_binding"].update(binding_status="accepted"),
    ],
)
def test_validator_fails_closed_on_partition_label_capture_or_runtime_drift(mutator):
    plan = deepcopy(build_physx_case_plan())
    mutator(plan)
    with pytest.raises(ValueError, match="drift|leakage"):
        validate_physx_case_plan(plan)


def test_claim_boundary_keeps_synthetic_parameters_out_of_physical_and_real_claims():
    plan = build_physx_case_plan()
    assert plan["data_origin"] == "twin"
    assert plan["real_data_used"] is False
    assert plan["training_eligible"] is False
    assert plan["render_status"] == "unrendered"
    assert plan["liquid_semantic_class_required_before_training"] is True
    assert "not a measured residue volume or fraction" in plan["profile_parameter_source"]
    assert plan["synthetic_label_rule"] == (
        "keep iff the injected previous-recipe blend weight is exactly zero; otherwise discard; "
        "this is a synthetic benchmark class and not a plant threshold"
    )
    assert "no sim-to-real" in plan["claim_boundary"]
    assert "no physical carryover threshold" in plan["claim_boundary"]


def test_cli_writes_one_deterministic_manifest_and_refuses_overwrite(tmp_path):
    script = ROOT / "scripts" / "build_physx_case_plan.py"
    output = tmp_path / "plan.json"
    command = [sys.executable, str(script), "--output", str(output)]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == build_physx_case_plan()
    before = output.read_bytes()
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert repeated.returncode != 0
    assert output.read_bytes() == before

"""Contracts for the train-only synthetic residual-visibility pilot."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_residual_pilot import (  # noqa: E402
    build_physx_residual_pilot_plan,
    validate_physx_residual_pilot_plan,
)


def test_candidate_ladder_is_frozen_train_only_and_covers_three_vial_extents():
    plan = build_physx_residual_pilot_plan()

    assert plan["schema_version"] == 1
    assert plan["artifact_type"] == "physx_f2_train_residual_visibility_pilot"
    assert plan["partition"] == "train"
    assert plan["batch_size_vials"] == 3
    assert {
        row["profile_id"]: tuple(row["residual_particle_count_by_vial"])
        for row in plan["candidate_profiles"]
    } == {
        "clean": (0, 0, 0),
        "first_only_low": (56, 0, 0),
        "first_only_high": (277, 0, 0),
        "first_two_low": (56, 14, 0),
        "first_two_high": (277, 70, 0),
        "all_three_low": (56, 28, 14),
        "all_three_high": (277, 139, 70),
    }
    assert all(
        counts[0] >= counts[1] >= counts[2] >= 0
        for counts in (
            row["residual_particle_count_by_vial"]
            for row in plan["candidate_profiles"]
        )
    )


def test_only_train_transition_recipe_timing_groups_and_unique_seeds_are_bound():
    plan = build_physx_residual_pilot_plan()

    assert plan["family"] == {
        "family_id": "TR_RESIDUAL_PILOT",
        "family_group_id": "TR_RESIDUAL_PILOT",
        "partition": "train",
        "previous_recipe": {
            "sim_recipe_id": "S0437",
            "role": "train",
            "cell": "0.8-0.1-0.1",
            "programmed_ml": {"t100": 60, "t200": 10, "t400": 10},
            "total_ml": 80,
            "row_sha256": "f51e3e40134eb0cf5d79c95e96e3543550df55b77325e6ec812ede538085bda7",
        },
        "current_recipe": {
            "sim_recipe_id": "S0117",
            "role": "train",
            "cell": "0.1-0.5-0.3",
            "programmed_ml": {"t100": 10, "t200": 40, "t400": 25},
            "total_ml": 75,
            "row_sha256": "ebb4419f41397f0f16d7cd69417b7b3d1b1fb0837a1d517c37acaf2081f68f67",
        },
        "transition_row_sha256": "3b3db95c2ed734d397bf1b14bf2eedbab3fb73cb4d51232a04b54e46152eb31d",
    }
    assert plan["programmed_dispense_order"] == ["t200", "t400", "t100"]
    assert plan["timing"] == {
        "physics_hz": 60,
        "duration_steps_by_tank": {"t200": 1352, "t400": 778, "t100": 536},
        "pour_end_steps": [1352, 2130, 2666],
        "stage_capture_steps": [1352, 2130, 2684],
        "fill_end_step": 2666,
        "primary_final_offset_steps": 18,
        "primary_final_offset_s": 0.3,
        "timing_model_source": "video5_recipe_median_active_tank_linear_fit_v1",
        "timing_interpretation": "real_like_approximation_not_measured_tank_boundaries",
    }

    profiles = plan["candidate_profiles"]
    seeds = [seed for profile in profiles for seed in profile["render_seed_by_vial"]]
    assert seeds == list(range(320001, 320022))
    assert len(seeds) == len(set(seeds))
    assert all(0 <= seed <= 999999 for seed in seeds)
    assert all(profile["partition"] == "train" for profile in profiles)
    assert all(
        profile["family_group_id"] == "TR_RESIDUAL_PILOT"
        and profile["batch_group_id"] == profile["batch_id"]
        and profile["nuisance_group_id"] == profile["batch_id"]
        for profile in profiles
    )
    assert plan["held_out_family_policy"] == {
        "val_families_included": False,
        "test_families_included": False,
        "release_condition": "freeze_one_profile_ladder_from_train_pilot_before_building_held_out_plans",
    }


def test_each_vial_uses_conserved_residual_plan_exact_source_allocation_and_oracle_label():
    plan = build_physx_residual_pilot_plan()
    expected_source_counts = {
        0: {"t200": 0, "t400": 0, "t100": 0},
        14: {"t200": 2, "t400": 2, "t100": 10},
        28: {"t200": 4, "t400": 3, "t100": 21},
        56: {"t200": 7, "t400": 7, "t100": 42},
        70: {"t200": 9, "t400": 9, "t100": 52},
        139: {"t200": 18, "t400": 17, "t100": 104},
        277: {"t200": 35, "t400": 34, "t100": 208},
    }

    assert len(plan["vials"]) == 21
    assert len({vial["vial_id"] for vial in plan["vials"]}) == 21
    for vial in plan["vials"]:
        residual_count = vial["residual_particle_count"]
        residual_plan = vial["residual_particle_plan"]
        assert residual_count <= residual_plan["clean_nominal_particle_count"] * 0.10
        assert residual_plan["residual_particle_count"] == residual_count
        assert residual_plan["residual_source_component_counts"] == expected_source_counts[
            residual_count
        ]
        assert residual_plan["clean_nominal_particle_count"] == 2777
        assert residual_plan["final_vial_particle_count"] == 2777
        assert residual_plan["particle_count_conserved"] is True
        assert residual_plan["programmed_current_dispense_unchanged"] is True
        assert vial["previous_programmed_ml"] == {"t100": 60, "t200": 10, "t400": 10}
        assert vial["current_programmed_ml"] == {"t100": 10, "t200": 40, "t400": 25}
        assert vial["programmed_dispense_order"] == ["t200", "t400", "t100"]
        assert vial["timing"] == plan["timing"]
        assert vial["expected_disposition"] == (
            "keep" if residual_count == 0 else "discard"
        )
        assert vial["label_source"] == "prescribed_synthetic_latent_oracle"
        assert vial["residual_oracle_excluded_from_model_inputs"] is True
        assert vial["residual_oracle_excluded_from_deploy_inputs"] is True


def test_selection_is_train_only_unfrozen_and_rendering_is_blocked_without_f2_receipt():
    plan = build_physx_residual_pilot_plan()

    assert plan["data_origin"] == "twin"
    assert plan["real_data_used"] is False
    assert plan["residual_counts_calibrated"] is False
    assert plan["physical_carryover_claimed"] is False
    assert plan["sim_to_real_claimed"] is False
    assert plan["residual_count_policy"] == {
        "clean_nominal_particle_count": 2777,
        "maximum_fraction_of_nominal": 0.10,
        "maximum_allowed_residual_particle_count": 277,
        "applies_independently_to_each_vial": True,
    }
    assert plan["synthetic_label_rule"] == (
        "keep iff prescribed residual_particle_count is exactly zero; otherwise discard; "
        "the count and label are synthetic scoring oracles, never model or deploy inputs"
    )
    assert plan["selection_policy"] == {
        "selection_partition": "train",
        "selection_allowed_only_from_this_pilot": True,
        "val_or_test_selection_allowed": False,
        "winning_profile_ladder_selected": False,
        "winning_profile_ladder_id": None,
        "selection_requires_accepted_render_artifacts": True,
    }
    assert plan["model_input_policy"] == {
        "residual_particle_count_is_latent_oracle": True,
        "residual_particle_count_allowed_as_model_input": False,
        "residual_particle_count_allowed_as_deploy_input": False,
        "expected_disposition_allowed_as_model_input": False,
        "labels_are_scoring_only": True,
    }
    assert plan["render_gate"] == {
        "render_status": "blocked",
        "block_reason": "accepted_f2_receipt_required",
        "accepted_f2_receipt_sha256": None,
        "accepted_f2_source_commit": None,
        "training_eligible": False,
    }
    assert plan["denominators"] == {
        "families": 1,
        "candidate_profiles": 7,
        "batches": 7,
        "vials": 21,
        "keep_vials": 9,
        "discard_vials": 12,
    }
    assert "uncalibrated" in plan["profile_parameter_source"]
    assert "no physical carryover" in plan["claim_boundary"]
    assert "no sim-to-real" in plan["claim_boundary"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda plan: plan.update(partition="val"),
        lambda plan: plan["family"]["current_recipe"]["programmed_ml"].update(t100=11),
        lambda plan: plan["candidate_profiles"][0]["render_seed_by_vial"].__setitem__(
            1, 320001
        ),
        lambda plan: plan["candidate_profiles"][1][
            "residual_particle_count_by_vial"
        ].__setitem__(1, 57),
        lambda plan: plan["vials"][1]["residual_particle_plan"][
            "residual_source_component_counts"
        ].update(t100=1),
        lambda plan: plan["vials"][0].update(expected_disposition="discard"),
        lambda plan: plan["selection_policy"].update(
            winning_profile_ladder_selected=True,
            winning_profile_ladder_id="all_three_high",
        ),
        lambda plan: plan["render_gate"].update(
            render_status="accepted", training_eligible=True
        ),
        lambda plan: plan.update(real_data_used=True),
        lambda plan: plan["timing"].update(fill_end_step=1),
        lambda plan: plan["candidate_profiles"][0].update(
            nuisance_group_id="CROSS_SPLIT"
        ),
    ],
)
def test_validator_fails_closed_on_source_split_seed_label_gate_or_claim_drift(mutator):
    plan = build_physx_residual_pilot_plan()
    validate_physx_residual_pilot_plan(plan)

    mutated = deepcopy(plan)
    mutator(mutated)
    with pytest.raises(ValueError, match="drift|leakage"):
        validate_physx_residual_pilot_plan(mutated)


def test_validator_rejects_non_object_plan():
    with pytest.raises(TypeError, match="object"):
        validate_physx_residual_pilot_plan([])


def test_plan_binds_only_synthetic_split_sources_and_fails_on_source_drift(tmp_path):
    plan = build_physx_residual_pilot_plan()

    assert plan["split_version"] == "inn-7-splits-v2"
    assert plan["source_sha256"] == {
        "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
        "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
        "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
    }
    assert plan["partition_policy"] == {
        "whole_transition_family_one_partition": True,
        "whole_three_vial_batch_one_partition": True,
        "whole_nuisance_group_one_partition": True,
        "training_uses_train_family_only": True,
        "val_and_test_families_present": False,
        "test_data_used_for_training_or_tuning": False,
    }

    copied_data = tmp_path / "data"
    copied_data.mkdir()
    for name in ("sim_recipes.csv", "sim_transitions.csv", "sim_split_spec.json"):
        shutil.copy2(ROOT / "data" / name, copied_data / name)
    with (copied_data / "sim_recipes.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="source hash drift"):
        build_physx_residual_pilot_plan(repo_root=tmp_path)

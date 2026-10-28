"""Train-only synthetic residual-visibility pilot planning for F2.

The prescribed residual particle counts in this module are uncalibrated render
parameters and latent synthetic ground truth. They are not deploy inputs and do
not estimate real carryover.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from .paths import REPO
from .physx_residual_case import build_residual_particle_plan


PROFILE_ROWS = (
    ("clean", "clean", "control", (0, 0, 0)),
    ("first_only_low", "first_only", "low", (56, 0, 0)),
    ("first_only_high", "first_only", "high", (277, 0, 0)),
    ("first_two_low", "first_two", "low", (56, 14, 0)),
    ("first_two_high", "first_two", "high", (277, 70, 0)),
    ("all_three_low", "all_three", "low", (56, 28, 14)),
    ("all_three_high", "all_three", "high", (277, 139, 70)),
)
SOURCE_HASHES = {
    "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
    "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
    "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
}
FAMILY_ID = "TR_RESIDUAL_PILOT"
PREVIOUS_RECIPE = {
    "sim_recipe_id": "S0437",
    "role": "train",
    "cell": "0.8-0.1-0.1",
    "programmed_ml": {"t100": 60, "t200": 10, "t400": 10},
    "total_ml": 80,
    "row_sha256": "f51e3e40134eb0cf5d79c95e96e3543550df55b77325e6ec812ede538085bda7",
}
CURRENT_RECIPE = {
    "sim_recipe_id": "S0117",
    "role": "train",
    "cell": "0.1-0.5-0.3",
    "programmed_ml": {"t100": 10, "t200": 40, "t400": 25},
    "total_ml": 75,
    "row_sha256": "ebb4419f41397f0f16d7cd69417b7b3d1b1fb0837a1d517c37acaf2081f68f67",
}
TRANSITION_ROW_SHA256 = "3b3db95c2ed734d397bf1b14bf2eedbab3fb73cb4d51232a04b54e46152eb31d"
PROGRAMMED_DISPENSE_ORDER = ("t200", "t400", "t100")
TIMING = {
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


def _timing_copy() -> dict[str, object]:
    return {
        **TIMING,
        "duration_steps_by_tank": dict(TIMING["duration_steps_by_tank"]),
        "pour_end_steps": list(TIMING["pour_end_steps"]),
        "stage_capture_steps": list(TIMING["stage_capture_steps"]),
    }


def _validate_source_files(repo_root: Path) -> None:
    for relative, expected in SOURCE_HASHES.items():
        path = repo_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash drift: {relative}")
    split = json.loads(
        (repo_root / "data/sim_split_spec.json").read_text(encoding="utf-8")
    )
    if split.get("version") != "v2" or split.get("render_seed_ranges", {}).get(
        "train"
    ) != [0, 999999]:
        raise ValueError("source split drift or leakage")


def _build_expected(repo_root: Path) -> dict[str, object]:
    _validate_source_files(repo_root)
    candidate_profiles = []
    vials = []
    next_seed = 320001
    for profile_index, (profile_id, coverage_class, severity, counts) in enumerate(
        PROFILE_ROWS, start=1
    ):
        batch_id = f"F2RP_TR_B{profile_index:02d}"
        seeds = list(range(next_seed, next_seed + 3))
        next_seed += 3
        vial_ids = []
        for ordinal, (residual_count, render_seed) in enumerate(
            zip(counts, seeds, strict=True), start=1
        ):
            vial_id = f"{batch_id}_V{ordinal}"
            vial_ids.append(vial_id)
            residual_plan = build_residual_particle_plan(
                current_programmed_ml=CURRENT_RECIPE["programmed_ml"],
                previous_programmed_ml=PREVIOUS_RECIPE["programmed_ml"],
                residual_particle_count=residual_count,
            )
            vials.append(
                {
                    "vial_id": vial_id,
                    "vial_ordinal": ordinal,
                    "profile_id": profile_id,
                    "partition": "train",
                    "family_group_id": FAMILY_ID,
                    "batch_id": batch_id,
                    "batch_group_id": batch_id,
                    "nuisance_group_id": batch_id,
                    "render_seed": render_seed,
                    "previous_sim_recipe_id": PREVIOUS_RECIPE["sim_recipe_id"],
                    "current_sim_recipe_id": CURRENT_RECIPE["sim_recipe_id"],
                    "previous_programmed_ml": dict(PREVIOUS_RECIPE["programmed_ml"]),
                    "current_programmed_ml": dict(CURRENT_RECIPE["programmed_ml"]),
                    "programmed_dispense_order": list(PROGRAMMED_DISPENSE_ORDER),
                    "timing": _timing_copy(),
                    "residual_particle_count": residual_count,
                    "residual_particle_plan": residual_plan,
                    "expected_disposition": "keep" if residual_count == 0 else "discard",
                    "label_source": "prescribed_synthetic_latent_oracle",
                    "residual_oracle_excluded_from_model_inputs": True,
                    "residual_oracle_excluded_from_deploy_inputs": True,
                }
            )
        candidate_profiles.append(
            {
                "profile_id": profile_id,
                "coverage_class": coverage_class,
                "severity": severity,
                "residual_particle_count_by_vial": list(counts),
                "partition": "train",
                "family_group_id": FAMILY_ID,
                "batch_id": batch_id,
                "batch_group_id": batch_id,
                "nuisance_group_id": batch_id,
                "render_seed_by_vial": seeds,
                "vial_ids": vial_ids,
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_train_residual_visibility_pilot",
        "data_origin": "twin",
        "split_version": "inn-7-splits-v2",
        "partition": "train",
        "real_data_used": False,
        "residual_counts_calibrated": False,
        "physical_carryover_claimed": False,
        "sim_to_real_claimed": False,
        "residual_count_policy": {
            "clean_nominal_particle_count": 2777,
            "maximum_fraction_of_nominal": 0.10,
            "maximum_allowed_residual_particle_count": 277,
            "applies_independently_to_each_vial": True,
        },
        "synthetic_label_rule": (
            "keep iff prescribed residual_particle_count is exactly zero; otherwise discard; "
            "the count and label are synthetic scoring oracles, never model or deploy inputs"
        ),
        "batch_size_vials": 3,
        "source_sha256": dict(SOURCE_HASHES),
        "family": {
            "family_id": FAMILY_ID,
            "family_group_id": FAMILY_ID,
            "partition": "train",
            "previous_recipe": {
                **PREVIOUS_RECIPE,
                "programmed_ml": dict(PREVIOUS_RECIPE["programmed_ml"]),
            },
            "current_recipe": {
                **CURRENT_RECIPE,
                "programmed_ml": dict(CURRENT_RECIPE["programmed_ml"]),
            },
            "transition_row_sha256": TRANSITION_ROW_SHA256,
        },
        "programmed_dispense_order": list(PROGRAMMED_DISPENSE_ORDER),
        "timing": _timing_copy(),
        "candidate_profiles": candidate_profiles,
        "vials": vials,
        "selection_policy": {
            "selection_partition": "train",
            "selection_allowed_only_from_this_pilot": True,
            "val_or_test_selection_allowed": False,
            "winning_profile_ladder_selected": False,
            "winning_profile_ladder_id": None,
            "selection_requires_accepted_render_artifacts": True,
        },
        "model_input_policy": {
            "residual_particle_count_is_latent_oracle": True,
            "residual_particle_count_allowed_as_model_input": False,
            "residual_particle_count_allowed_as_deploy_input": False,
            "expected_disposition_allowed_as_model_input": False,
            "labels_are_scoring_only": True,
        },
        "render_gate": {
            "render_status": "blocked",
            "block_reason": "accepted_f2_receipt_required",
            "accepted_f2_receipt_sha256": None,
            "accepted_f2_source_commit": None,
            "training_eligible": False,
        },
        "denominators": {
            "families": 1,
            "candidate_profiles": len(candidate_profiles),
            "batches": len(candidate_profiles),
            "vials": len(vials),
            "keep_vials": sum(
                vial["expected_disposition"] == "keep" for vial in vials
            ),
            "discard_vials": sum(
                vial["expected_disposition"] == "discard" for vial in vials
            ),
        },
        "profile_parameter_source": (
            "synthetic uncalibrated candidate counts chosen only for a bounded "
            "train visibility pilot; not a measured residue volume or fraction"
        ),
        "claim_boundary": (
            "synthetic render-planning evidence only; no physical carryover or "
            "calibration claim and no sim-to-real claim"
        ),
        "held_out_family_policy": {
            "val_families_included": False,
            "test_families_included": False,
            "release_condition": "freeze_one_profile_ladder_from_train_pilot_before_building_held_out_plans",
        },
        "partition_policy": {
            "whole_transition_family_one_partition": True,
            "whole_three_vial_batch_one_partition": True,
            "whole_nuisance_group_one_partition": True,
            "training_uses_train_family_only": True,
            "val_and_test_families_present": False,
            "test_data_used_for_training_or_tuning": False,
        },
    }


def build_physx_residual_pilot_plan(
    *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Build and validate the frozen train-only three-vial candidate pilot."""

    root = Path(repo_root)
    plan = _build_expected(root)
    validate_physx_residual_pilot_plan(plan, repo_root=root)
    return plan


def validate_physx_residual_pilot_plan(
    plan: object, *, repo_root: str | Path = REPO
) -> None:
    """Fail closed when a residual-pilot plan departs from its frozen contract."""

    if not isinstance(plan, Mapping):
        raise TypeError("residual pilot plan must be an object")
    if plan != _build_expected(Path(repo_root)):
        raise ValueError("residual pilot plan drift or leakage")

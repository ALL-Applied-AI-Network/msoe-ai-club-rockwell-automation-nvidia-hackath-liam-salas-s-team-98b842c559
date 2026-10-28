"""Build a synthetic-only F2 keep/discard case manifest."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

from .paths import REPO


SOURCE_HASHES = {
    "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
    "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
    "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
    "data/twin_vision_dataset_v1.json": "1d7d8fee69c7714c0278b563261f4504ac804220d2d63aaa8da38268c1c923a2",
}
FAMILY_ROWS = (
    ("TR", "train", "S0437", "S0117", 310000),
    ("VA", "val", "S0160", "S0380", 1310000),
    ("TE_HIGH", "test", "S0177", "S0431", 2310000),
    ("TE_LOW", "test", "S0147", "S0074", 2320000),
)
PROFILE_ROWS = (
    ("clean", (0.0, 0.0, 0.0)),
    ("first_only", (0.30, 0.0, 0.0)),
    ("first_two", (0.30, 0.12, 0.0)),
    ("all_three", (0.30, 0.18, 0.08)),
)
STAGE_ROWS = (
    ("AFTER_T200", 1, "end_of_t200_pour", None, "cumulative_end_after_t200"),
    ("AFTER_T400", 2, "end_of_t400_pour", None, "cumulative_end_after_t400"),
    ("FINAL_PLUS_0P3S", 3, "after_fill_end", 0.3, "fill_end_plus_18_steps"),
)
PROGRAMMED_DISPENSE_ORDER = ("t200", "t400", "t100")
PHYSICS_HZ = 60
ACTIVE_TANK_INTERCEPT_S = 8.16571903
TANK_COEFFICIENT_S_PER_ML = {
    "t100": 0.07633141,
    "t200": 0.35935439,
    "t400": 0.19177068,
}
PRIMARY_FINAL_OFFSET_STEPS = 18
REAL_LIKE_TIMING_MODEL = {
    "physics_hz": PHYSICS_HZ,
    "active_tank_intercept_s": ACTIVE_TANK_INTERCEPT_S,
    "tank_coefficient_s_per_ml": dict(TANK_COEFFICIENT_S_PER_ML),
    "duration_step_rule": "nearest integer of (intercept_s + coefficient_s_per_ml * current_tank_ml) * physics_hz",
    "pour_order": list(PROGRAMMED_DISPENSE_ORDER),
    "primary_final_offset_steps": PRIMARY_FINAL_OFFSET_STEPS,
    "primary_final_offset_s": PRIMARY_FINAL_OFFSET_STEPS / PHYSICS_HZ,
    "source": "video5_recipe_median_active_tank_linear_fit_v1",
    "source_fit_r2": 0.92595,
    "source_fit_rmse_s": 2.371,
    "source_fit_loocv_rmse_s": 4.765,
    "interpretation": "real_like_approximation_not_measured_tank_boundaries",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_sources(root: Path) -> tuple[dict[str, dict[str, object]], dict[tuple[str, str, str], str], dict[str, object]]:
    for relative, expected in SOURCE_HASHES.items():
        if _sha256_path(root / relative) != expected:
            raise ValueError(f"source hash drift: {relative}")
    split = json.loads((root / "data/sim_split_spec.json").read_text(encoding="utf-8"))
    if split.get("version") != "v2":
        raise ValueError("split spec version drift")

    recipe_lines = (root / "data/sim_recipes.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    recipes: dict[str, dict[str, object]] = {}
    for line, row in zip(recipe_lines[1:], csv.DictReader(recipe_lines), strict=True):
        recipe_id = row["sim_recipe_id"]
        recipes[recipe_id] = {
            "sim_recipe_id": recipe_id,
            "role": row["role"],
            "cell": row["cell"],
            "programmed_ml": {
                tank: int(row[f"{tank}_ml"]) for tank in ("t100", "t200", "t400")
            },
            "total_ml": int(row["total_ml"]),
            "row_sha256": _sha256_bytes(line.encode("utf-8")),
        }

    transition_lines = (root / "data/sim_transitions.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    transitions = {
        (row["role"], row["prev_sim_recipe_id"], row["sim_recipe_id"]): _sha256_bytes(
            line.encode("utf-8")
        )
        for line, row in zip(
            transition_lines[1:], csv.DictReader(transition_lines), strict=True
        )
    }
    return recipes, transitions, split


def _validate_selected_sources(
    recipes: Mapping[str, Mapping[str, object]],
    transitions: Mapping[tuple[str, str, str], str],
    split: Mapping[str, object],
) -> None:
    recipe_roles: dict[str, str] = {}
    cell_roles: dict[str, str] = {}
    seed_roles: dict[int, str] = {}
    reserved = split["real_recipes_reserved"]
    near_rule = split["near_rule"]
    seed_ranges = split["render_seed_ranges"]
    if not isinstance(reserved, Mapping) or not isinstance(near_rule, Mapping) or not isinstance(seed_ranges, Mapping):
        raise ValueError("split source structure drift")
    for family_id, role, previous_id, current_id, seed_base in FAMILY_ROWS:
        transition_key = (role, previous_id, current_id)
        if transition_key not in transitions:
            raise ValueError(f"missing transition row for {family_id}")
        for recipe_id in (previous_id, current_id):
            recipe = recipes.get(recipe_id)
            if recipe is None or recipe.get("role") != role:
                raise ValueError(f"recipe role leakage for {recipe_id}")
            programmed = recipe.get("programmed_ml")
            if not isinstance(programmed, Mapping) or set(programmed) != {"t100", "t200", "t400"}:
                raise ValueError(f"recipe source malformed for {recipe_id}")
            total = int(recipe["total_ml"])
            for real_recipe in reserved.values():
                if not isinstance(real_recipe, Mapping):
                    raise ValueError("reserved recipe source malformed")
                real_total = sum(int(real_recipe[tank]) for tank in PROGRAMMED_DISPENSE_ORDER)
                near_ml = all(
                    abs(int(programmed[tank]) - int(real_recipe[tank]))
                    <= int(near_rule["ml_per_tank"])
                    for tank in PROGRAMMED_DISPENSE_ORDER
                )
                near_fraction = all(
                    abs(int(programmed[tank]) / total - int(real_recipe[tank]) / real_total)
                    <= float(near_rule["fraction_per_tank"])
                    for tank in PROGRAMMED_DISPENSE_ORDER
                )
                if near_ml or near_fraction:
                    raise ValueError(f"selected recipe is reserved-real-near: {recipe_id}")
            prior_role = recipe_roles.setdefault(recipe_id, role)
            if prior_role != role:
                raise ValueError(f"recipe cross-partition leakage: {recipe_id}")
            cell = str(recipe["cell"])
            prior_cell_role = cell_roles.setdefault(cell, role)
            if prior_cell_role != role:
                raise ValueError(f"recipe cell cross-partition leakage: {cell}")
        role_range = seed_ranges.get(role)
        if not isinstance(role_range, list) or len(role_range) != 2:
            raise ValueError(f"seed range missing for {role}")
        for offset in (1, 2, 3):
            seed = seed_base + offset
            if not int(role_range[0]) <= seed <= int(role_range[1]):
                raise ValueError(f"seed leaves partition range: {seed}")
            prior_seed_role = seed_roles.setdefault(seed, role)
            if prior_seed_role != role:
                raise ValueError(f"nuisance seed cross-partition leakage: {seed}")


def _batch_id(family_id: str, profile_id: str) -> str:
    value = f"physx-f2-keep-discard-v1:{family_id}:{profile_id}".encode("utf-8")
    return f"F2B_{hashlib.sha256(value).hexdigest()[:12].upper()}"


def _recipe_timing(programmed_ml: Mapping[str, object]) -> dict[str, object]:
    if set(programmed_ml) != set(PROGRAMMED_DISPENSE_ORDER):
        raise ValueError("recipe timing requires exactly t100, t200, and t400")
    duration_steps = {}
    for tank in PROGRAMMED_DISPENSE_ORDER:
        amount = programmed_ml[tank]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"{tank} programmed mL must be numeric")
        if not math.isfinite(float(amount)) or float(amount) <= 0:
            raise ValueError(f"{tank} programmed mL must be finite and positive")
        raw_steps = (
            ACTIVE_TANK_INTERCEPT_S
            + TANK_COEFFICIENT_S_PER_ML[tank] * float(amount)
        ) * PHYSICS_HZ
        duration_steps[tank] = int(math.floor(raw_steps + 0.5))
    stage1 = duration_steps["t200"]
    stage2 = stage1 + duration_steps["t400"]
    fill_end = stage2 + duration_steps["t100"]
    return {
        "duration_steps_by_tank": duration_steps,
        "pour_end_steps": [stage1, stage2, fill_end],
        "fill_end_step": fill_end,
        "stage_capture_steps": [
            stage1,
            stage2,
            fill_end + PRIMARY_FINAL_OFFSET_STEPS,
        ],
        "primary_final_offset_steps": PRIMARY_FINAL_OFFSET_STEPS,
        "primary_final_offset_s": PRIMARY_FINAL_OFFSET_STEPS / PHYSICS_HZ,
        "timing_model_source": REAL_LIKE_TIMING_MODEL["source"],
        "timing_interpretation": REAL_LIKE_TIMING_MODEL["interpretation"],
    }


def _build_expected(root: Path) -> dict[str, object]:
    recipes, transitions, split = _read_sources(root)
    _validate_selected_sources(recipes, transitions, split)
    families = []
    for family_id, role, previous_id, current_id, seed_base in FAMILY_ROWS:
        families.append(
            {
                "family_id": family_id,
                "partition": role,
                "previous_sim_recipe_id": previous_id,
                "current_sim_recipe_id": current_id,
                "seed_base": seed_base,
                "recipe_source_binding": {
                    "previous": dict(recipes[previous_id]),
                    "current": dict(recipes[current_id]),
                },
                "transition_source_binding": {
                    "role": role,
                    "previous_sim_recipe_id": previous_id,
                    "current_sim_recipe_id": current_id,
                    "row_sha256": transitions[(role, previous_id, current_id)],
                },
            }
        )

    batches = []
    vials = []
    stage_captures = []
    family_lookup = {row["family_id"]: row for row in families}
    for family_id, role, previous_id, current_id, seed_base in FAMILY_ROWS:
        family = family_lookup[family_id]
        previous = family["recipe_source_binding"]["previous"]
        current = family["recipe_source_binding"]["current"]
        timing = _recipe_timing(current["programmed_ml"])
        seeds = [seed_base + ordinal for ordinal in (1, 2, 3)]
        for profile_id, blend_weights in PROFILE_ROWS:
            batch_id = _batch_id(family_id, profile_id)
            batches.append(
                {
                    "batch_id": batch_id,
                    "family_id": family_id,
                    "partition": role,
                    "profile_id": profile_id,
                    "previous_sim_recipe_id": previous_id,
                    "current_sim_recipe_id": current_id,
                    "previous_programmed_ml": dict(previous["programmed_ml"]),
                    "current_programmed_ml": dict(current["programmed_ml"]),
                    "programmed_dispense_order": list(PROGRAMMED_DISPENSE_ORDER),
                    "synthetic_previous_recipe_blend_by_vial": list(blend_weights),
                    "timing": deepcopy(timing),
                    "render_seed_by_vial": list(seeds),
                    "nuisance_group_ids": [
                        f"{family_id}-MATCHED-V{ordinal}" for ordinal in (1, 2, 3)
                    ],
                    "nuisance_pairing": "matched_across_profiles_inside_one_family_and_partition",
                    "vial_count": 3,
                    "render_status": "unrendered",
                    "training_eligible": False,
                }
            )
            for vial_ordinal, (blend_weight, seed) in enumerate(
                zip(blend_weights, seeds, strict=True), start=1
            ):
                vial_id = f"{batch_id}-V{vial_ordinal}"
                stage_ids = [f"{vial_id}-{row[0]}" for row in STAGE_ROWS]
                disposition = "keep" if blend_weight == 0.0 else "discard"
                vials.append(
                    {
                        "vial_id": vial_id,
                        "batch_id": batch_id,
                        "family_id": family_id,
                        "partition": role,
                        "vial_ordinal": vial_ordinal,
                        "previous_sim_recipe_id": previous_id,
                        "current_sim_recipe_id": current_id,
                        "previous_programmed_ml": dict(previous["programmed_ml"]),
                        "current_programmed_ml": dict(current["programmed_ml"]),
                        "programmed_dispense_order": list(PROGRAMMED_DISPENSE_ORDER),
                        "render_seed": seed,
                        "nuisance_group_id": f"{family_id}-MATCHED-V{vial_ordinal}",
                        "synthetic_previous_recipe_blend": blend_weight,
                        "synthetic_blend_semantics": (
                            "uncalibrated render-only composition/appearance blend weight toward "
                            "the previous recipe; not a measured residue volume or fraction"
                        ),
                        "timing": deepcopy(timing),
                        "expected_disposition": disposition,
                        "fault_type": (
                            "none" if disposition == "keep" else "synthetic_carryover_like"
                        ),
                        "label_source": "render_parameters",
                        "ground_truth_not_model_input": True,
                        "stage_capture_ids": stage_ids,
                        "render_status": "unrendered",
                        "training_eligible": False,
                    }
                )
                for (
                    stage_id,
                    ordinal,
                    relative_event,
                    offset_s,
                    _step_rule,
                ), physics_step in zip(
                    STAGE_ROWS, timing["stage_capture_steps"], strict=True
                ):
                    stage_captures.append(
                        {
                            "capture_id": f"{vial_id}-{stage_id}",
                            "vial_id": vial_id,
                            "batch_id": batch_id,
                            "family_id": family_id,
                            "partition": role,
                            "current_sim_recipe_id": current_id,
                            "current_programmed_ml": dict(current["programmed_ml"]),
                            "stage_id": stage_id,
                            "stage_ordinal": ordinal,
                            "physics_step": physics_step,
                            "relative_event": relative_event,
                            "offset_from_fill_end_s": offset_s,
                            "timing_model_source": timing["timing_model_source"],
                            "timing_interpretation": timing["timing_interpretation"],
                            "render_status": "unrendered",
                            "rgb_sha256": None,
                            "liquid_mask_sha256": None,
                            "liquid_semantic_class": None,
                        }
                    )

    by_partition = {}
    for role in ("train", "val", "test"):
        role_batches = [row for row in batches if row["partition"] == role]
        role_vials = [row for row in vials if row["partition"] == role]
        by_partition[role] = {
            "batches": len(role_batches),
            "vials": len(role_vials),
            "keep": sum(row["expected_disposition"] == "keep" for row in role_vials),
            "discard": sum(row["expected_disposition"] == "discard" for row in role_vials),
        }
    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_keep_discard_case_plan",
        "config_id": "physx-f2-keep-discard-three-vial-v1",
        "data_origin": "twin",
        "split_version": "inn-7-splits-v2",
        "render_status": "unrendered",
        "training_eligible": False,
        "real_data_used": False,
        "batch_size_vials": 3,
        "source_sha256": dict(SOURCE_HASHES),
        "claim_boundary": (
            "Synthetic F2 render parameters only: no sim-to-real, no real defect or cause, "
            "no physical carryover threshold, no calibrated residue amount, and no plant action."
        ),
        "synthetic_label_rule": (
            "keep iff the injected previous-recipe blend weight is exactly zero; otherwise discard; "
            "this is a synthetic benchmark class and not a plant threshold"
        ),
        "profile_parameters_uncalibrated": True,
        "profile_parameter_source": (
            "Reuses the frozen 0/0.30/0.12/0.18/0.08 synthetic severity grid from "
            "data/twin_vision_dataset_v1.json; it is reinterpreted only as an F2 render "
            "composition/appearance blend parameter and is not a measured residue volume or fraction."
        ),
        "real_like_timing_model": deepcopy(REAL_LIKE_TIMING_MODEL),
        "liquid_semantic_class_required_before_training": True,
        "f2_runtime_binding": {
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
        },
        "partition_policy": {
            "whole_transition_family_one_partition": True,
            "whole_three_vial_batch_one_partition": True,
            "whole_nuisance_group_one_partition": True,
            "training_uses_train_families_only": True,
            "val_and_test_families_held_out_from_training": True,
            "test_data_used_for_training_or_tuning": False,
            "thresholds_freeze_on_val_before_test": True,
        },
        "profiles": [
            {
                "profile_id": profile_id,
                "synthetic_previous_recipe_blend_by_vial": list(blend_weights),
            }
            for profile_id, blend_weights in PROFILE_ROWS
        ],
        "stage_contract": [
            {
                "stage_id": stage_id,
                "stage_ordinal": ordinal,
                "relative_event": relative_event,
                "offset_from_fill_end_s": offset_s,
                "step_rule": step_rule,
            }
            for stage_id, ordinal, relative_event, offset_s, step_rule in STAGE_ROWS
        ],
        "denominators": {
            "families": len(families),
            "train_families": sum(row["partition"] == "train" for row in families),
            "val_families": sum(row["partition"] == "val" for row in families),
            "test_families": sum(row["partition"] == "test" for row in families),
            "batches": len(batches),
            "vials": len(vials),
            "stage_captures": len(stage_captures),
            "keep_vials": sum(row["expected_disposition"] == "keep" for row in vials),
            "discard_vials": sum(
                row["expected_disposition"] == "discard" for row in vials
            ),
            "by_partition": by_partition,
        },
        "families": families,
        "batches": batches,
        "vials": vials,
        "stage_captures": stage_captures,
    }


def build_physx_case_plan(*, repo_root: str | Path = REPO) -> dict[str, object]:
    """Return a source-bound, not-yet-rendered F2 case plan."""

    plan = _build_expected(Path(repo_root))
    validate_physx_case_plan(plan, repo_root=repo_root)
    return plan


def validate_physx_case_plan(
    plan: Mapping[str, object], *, repo_root: str | Path = REPO
) -> None:
    """Reject drift from the frozen F2 case plan."""

    if not isinstance(plan, Mapping):
        raise TypeError("F2 case plan must be an object")
    if plan != _build_expected(Path(repo_root)):
        raise ValueError("F2 case plan drift or leakage")

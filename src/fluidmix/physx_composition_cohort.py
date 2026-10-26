"""Frozen train-only composition-substitution cohort for F2.

This module is a CPU contract.  It does not launch Isaac Sim or claim that the
synthetic composition parameter represents a physical residue fraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .paths import REPO
from .physx_three_pour import distribute_particle_count


REFERENCE_BATCH_COUNT = 2
SOURCE_HASHES = {
    "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
    "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
    "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
}
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
COMPONENT_ORDER = ("t200", "t400", "t100")
TIMING = {
    "physics_hz": 120,
    "capture_hz": 10,
    "capture_cadence_steps": 12,
    "duration_steps_by_tank": {"t200": 2704, "t400": 1556, "t100": 1072},
    "pour_end_steps": [2704, 4260, 5332],
    "stage_capture_steps": [2704, 4260, 5368],
    "fill_end_step": 5332,
    "primary_final_offset_steps": 36,
    "primary_final_offset_s": 0.3,
    "no_inter_pour_settle": True,
    "timing_model_source": "video5_recipe_median_active_tank_linear_fit_v1",
    "timing_interpretation": "real_like_approximation_not_measured_tank_boundaries",
}
CROSS_SECTION_AREA_M2 = 0.0005229514163272396
PARTICLE_VOLUME_M3 = 27.0e-9
HEIGHT_TOLERANCE_M = 0.006
STAGE_ROWS = (
    ("AFTER_T200", "t200", 2704, 40.0, 1481),
    ("AFTER_T400", "t400", 4260, 65.0, 2407),
    ("FINAL_PLUS_0P3S", "t100", 5368, 75.0, 2777),
)
TARGET_PROFILE_ROWS = (
    ("clean", (0.0, 0.0, 0.0)),
    ("first_only", (0.10, 0.0, 0.0)),
    ("first_two", (0.10, 0.05, 0.0)),
    ("all_three", (0.10, 0.05, 0.025)),
)


def _validate_sources(repo_root: Path) -> None:
    for relative, expected in SOURCE_HASHES.items():
        path = repo_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash drift: {relative}")
    split = json.loads(
        (repo_root / "data/sim_split_spec.json").read_text(encoding="utf-8")
    )
    if (
        split.get("version") != "v2"
        or split.get("render_seed_ranges", {}).get("train") != [0, 999999]
    ):
        raise ValueError("source split drift or leakage")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_batches() -> list[dict[str, object]]:
    batches: list[dict[str, object]] = []
    seed = 330001
    for ordinal in range(1, REFERENCE_BATCH_COUNT + 1):
        batches.append(
            {
                "batch_id": f"F2CS_TR_R{ordinal:02d}",
                "cohort_role": "reference_only",
                "profile_id": "reference_control",
                "composition_substitution_alpha_by_vial": [0.0, 0.0, 0.0],
                "render_seed_by_vial": list(range(seed, seed + 3)),
            }
        )
        seed += 3
    for ordinal, (profile_id, alpha_rows) in enumerate(TARGET_PROFILE_ROWS, start=1):
        batches.append(
            {
                "batch_id": f"F2CS_TR_T{ordinal:02d}",
                "cohort_role": "scored_target",
                "profile_id": profile_id,
                "composition_substitution_alpha_by_vial": list(alpha_rows),
                "render_seed_by_vial": list(range(seed, seed + 3)),
            }
        )
        seed += 3
    return batches


def _build_vials(batches: list[dict[str, object]]) -> list[dict[str, object]]:
    vials: list[dict[str, object]] = []
    for batch in batches:
        for ordinal, (alpha, seed) in enumerate(
            zip(
                batch["composition_substitution_alpha_by_vial"],
                batch["render_seed_by_vial"],
                strict=True,
            ),
            start=1,
        ):
            vial_id = f"{batch['batch_id']}_V{ordinal}"
            scored = batch["cohort_role"] == "scored_target"
            vials.append(
                {
                    "vial_id": vial_id,
                    "batch_id": batch["batch_id"],
                    "cohort_role": batch["cohort_role"],
                    "profile_id": batch["profile_id"],
                    "vial_ordinal": ordinal,
                    "partition": "train",
                    "family_id": "TR_COMPOSITION_SUBSTITUTION",
                    "render_seed": seed,
                    "previous_sim_recipe_id": PREVIOUS_RECIPE["sim_recipe_id"],
                    "current_sim_recipe_id": CURRENT_RECIPE["sim_recipe_id"],
                    "previous_programmed_ml": dict(PREVIOUS_RECIPE["programmed_ml"]),
                    "current_programmed_ml": dict(CURRENT_RECIPE["programmed_ml"]),
                    "programmed_dispense_order": list(COMPONENT_ORDER),
                    "composition_substitution_alpha": alpha,
                    "composition_parameter_semantics": (
                        "uncalibrated synthetic composition substitution toward the "
                        "previous recipe; not residue volume or a physical fraction"
                    ),
                    "scored_target": scored,
                    "expected_disposition": (
                        None if not scored else ("keep" if alpha == 0.0 else "discard")
                    ),
                    "label_source": (
                        "reference_only_not_scored"
                        if not scored
                        else "prescribed_synthetic_composition_oracle"
                    ),
                    "oracle_excluded_from_model_inputs": True,
                    "stage_capture_ids": [
                        f"{vial_id}-{stage_id}" for stage_id, *_rest in STAGE_ROWS
                    ],
                }
            )
    return vials


def build_composition_cohort_plan(
    *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Return the frozen cohort plan."""

    _validate_sources(Path(repo_root))
    batches = _build_batches()
    vials = _build_vials(batches)
    target_alphas = [
        alpha
        for batch in batches
        if batch["cohort_role"] == "scored_target"
        for alpha in batch["composition_substitution_alpha_by_vial"]
    ]
    scored_keep = sum(alpha == 0.0 for alpha in target_alphas)
    scored_discard = len(target_alphas) - scored_keep
    stage_contract = []
    for ordinal, (stage_id, tank_id, physics_step, volume_ml, count) in enumerate(
        STAGE_ROWS, start=1
    ):
        stage_contract.append(
            {
                "stage_id": stage_id,
                "stage_ordinal": ordinal,
                "tank_id": tank_id,
                "physics_step": physics_step,
                "programmed_cumulative_volume_ml": volume_ml,
                "expected_cumulative_particle_count": count,
                "count_derived_volume_ml": count * PARTICLE_VOLUME_M3 * 1_000_000.0,
                "programmed_height_m": volume_ml * 1.0e-6 / CROSS_SECTION_AREA_M2,
                "count_derived_body_height_m": count
                * PARTICLE_VOLUME_M3
                / CROSS_SECTION_AREA_M2,
                "height_tolerance_m": HEIGHT_TOLERANCE_M,
                "required_counts": {
                    "emitted": count,
                    "active": count,
                    "inside": count,
                    "in_flight": 0,
                    "outside": 0,
                    "escaped": 0,
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_train_composition_substitution_cohort",
        "data_origin": "twin",
        "split_version": "inn-7-splits-v2",
        "partition": "train",
        "batch_size_vials": 3,
        "source_sha256": dict(SOURCE_HASHES),
        "family": {
            "family_id": "TR_COMPOSITION_SUBSTITUTION",
            "partition": "train",
            "previous_recipe": deepcopy(PREVIOUS_RECIPE),
            "current_recipe": deepcopy(CURRENT_RECIPE),
            "transition_row_sha256": TRANSITION_ROW_SHA256,
        },
        "programmed_dispense_order": list(COMPONENT_ORDER),
        "timing": deepcopy(TIMING),
        "stage_contract": stage_contract,
        "volume_geometry": {
            "cross_section_area_m2": CROSS_SECTION_AREA_M2,
            "circumradius_m": 0.0132029216,
            "apothem_m": 0.0132029216 * math.cos(math.pi / 12.0),
            "floor_top_z_m": 0.9762065645,
            "copied_twin_liquid_bottom_z_m": 0.9762065644917131,
            "floor_alignment_tolerance_m": 1.0e-6,
            "floor_alignment_passed": True,
            "opaque_base_visibility": "requires_top_level_rgb_stage_review",
        },
        "per_stage_evidence_required": [
            "emitted_particle_count",
            "active_particle_count",
            "inside_vial_interior_count",
            "outside_vial_interior_count",
            "expected_in_flight_stream_count",
            "leaked_particle_count",
            "contained_z_min_m",
            "contained_z_q05_m",
            "contained_z_q95_m",
            "contained_z_max_m",
            "particle_bulk_height_m",
            "visible_liquid_column_height_m",
            "collider_floor_top_z_m",
            "count_driven_body_world_bottom_z_m",
            "point_instancer_scale_component_min",
            "point_instancer_scale_component_max",
            "body_material_visibility",
            "liquid_semantic_mask_sha256",
            "top_level_rgb_crop_judgment_sha256",
        ],
        "particle_render_diagnostics": {
            "sphere_prototype_radius_m": 0.0015,
            "fluid_rest_offset_m": 0.0015,
            "radius_to_fluid_rest_offset_ratio": 1.0,
            "sphere_prototypes_rgb_and_semantic_visible": False,
            "point_instancer_scale_range_required_per_stage": True,
            "count_driven_body_visible_for_positive_count": True,
            "count_driven_body_material_binding_required": True,
        },
        "batches": batches,
        "vials": vials,
        "reference_policy": {
            "reference_batch_ids": ["F2CS_TR_R01", "F2CS_TR_R02"],
            "exact_same_recipe_required": True,
            "minimum_independent_batch_ids": 2,
            "reference_only_batches_never_scored_targets": True,
            "reference_validity_requires_rendered_measurements": True,
            "planned_eligibility_is_not_observed_validity": True,
        },
        "partition_policy": {
            "whole_transition_family_one_partition": True,
            "whole_three_vial_batch_one_partition": True,
            "training_uses_train_family_only": True,
            "val_and_test_families_present": False,
            "test_data_used_for_training_or_tuning": False,
            "thresholds_freeze_on_validation_before_test": True,
        },
        "render_gate": {
            "status": "blocked",
            "block_reasons": [
                "accepted_a34_full_run_receipt_required",
                "runner_schedule_ingest_not_implemented",
            ],
            "accepted_a34_full_run_receipt_sha256": None,
            "render_authorized": False,
            "training_eligible": False,
        },
        "runtime_handoff": {
            "target_runner": "scripts/physx_three_pour_sequence.py",
            "required_new_cli": [
                "--execution-schedule <json>",
                "--execution-schedule-sha256 <lowercase_sha256>",
            ],
            "schedule_ingest_implemented": False,
            "required_fail_closed_checks": [
                "schedule_hash_matches",
                "source_plan_hashes_match",
                "120hz_timing_matches",
                "particle_ids_and_counts_match_every_event",
                "event_compositions_match",
                "all_three_programmed_volume_gates_pass",
                "receipt_binds_schedule_hash",
            ],
        },
        "composition_substitution_contract": {
            "previous_recipe_component_fraction_t200_t400_t100": [
                0.125,
                0.125,
                0.75,
            ],
            "rule": (
                "(1-alpha)*current_tank_one_hot + "
                "alpha*previous_recipe_final_fraction"
            ),
            "fixed_particle_count": True,
            "initial_extra_particle_count": 0,
            "programmed_volume_displaced": False,
            "stream_material_remains_current_tank_source_colour": True,
            "uncalibrated": True,
            "physical_residue_fraction": False,
        },
        "real_data_used": False,
        "physical_carryover_claimed": False,
        "sim_to_real_claimed": False,
        "claim_boundary": (
            "synthetic composition/render contract only; no physical carryover, "
            "real defect, calibrated residue, real cause, or sim-to-real claim"
        ),
        "denominators": {
            "families": 1,
            "batches": len(batches),
            "reference_only_batches": 2,
            "scored_target_batches": 4,
            "vials": len(batches) * 3,
            "reference_only_vials": 6,
            "scored_target_vials": len(target_alphas),
            "scored_keep_vials": scored_keep,
            "scored_discard_vials": scored_discard,
            "stage_captures": len(batches) * 3 * 3,
            "scored_stage_targets": len(target_alphas) * 3,
            "planned_reference_eligible_stage_targets": len(target_alphas) * 3,
            "observed_reference_valid_stage_targets": 0,
        },
    }


def build_composition_execution_schedule(
    *, plan: object, vial_record: object
) -> dict[str, object]:
    """Return one CPU execution schedule for the future Isaac runner."""

    if not isinstance(plan, dict):
        raise TypeError("composition cohort plan must be an object")
    expected_plan = build_composition_cohort_plan()
    if plan != expected_plan:
        raise ValueError("composition cohort plan drift or leakage")
    if not isinstance(vial_record, dict):
        raise TypeError("vial_record must be an object")
    vial_id = vial_record.get("vial_id")
    expected_vial = next(
        (row for row in plan["vials"] if row["vial_id"] == vial_id), None
    )
    if expected_vial is None or vial_record != expected_vial:
        raise ValueError("vial_record drift or leakage")

    alpha = float(vial_record["composition_substitution_alpha"])
    previous_fraction = (0.125, 0.125, 0.75)
    particle_counts = {"t200": 1481, "t400": 926, "t100": 370}
    particle_id = 0
    previous_end_step = 0
    pours: list[dict[str, object]] = []
    for ordinal, tank_id in enumerate(COMPONENT_ORDER, start=1):
        duration_steps = int(TIMING["duration_steps_by_tank"][tank_id])
        end_step = previous_end_step + duration_steps
        count = particle_counts[tank_id]
        chunk_count = math.ceil(duration_steps / int(TIMING["capture_cadence_steps"]))
        chunk_counts = distribute_particle_count(count, chunk_count)
        one_hot = [0.0, 0.0, 0.0]
        one_hot[ordinal - 1] = 1.0
        particle_composition = [
            round((1.0 - alpha) * one_hot[index] + alpha * previous_fraction[index], 12)
            for index in range(3)
        ]
        pour_start_id = particle_id
        events: list[dict[str, object]] = []
        for index, event_count in enumerate(chunk_counts):
            event_end_id = particle_id + event_count
            events.append(
                {
                    "ordinal": index + 1,
                    "tank_id": tank_id,
                    "physics_step": previous_end_step
                    + math.ceil((index + 1) * duration_steps / chunk_count),
                    "particle_count": event_count,
                    "particle_id_range": [particle_id, event_end_id],
                    "particle_composition": list(particle_composition),
                }
            )
            particle_id = event_end_id
        pours.append(
            {
                "ordinal": ordinal,
                "tank_id": tank_id,
                "programmed_volume_ml": vial_record["current_programmed_ml"][tank_id],
                "particle_count": count,
                "particle_id_range": [pour_start_id, particle_id],
                "start_step": previous_end_step + 1,
                "duration_steps": duration_steps,
                "end_step": end_step,
                "stage_capture_step": TIMING["stage_capture_steps"][ordinal - 1],
                "emission_chunk_count": chunk_count,
                "stream_one_hot_composition": one_hot,
                "emitted_particle_composition": particle_composition,
                "emission_schedule": events,
            }
        )
        previous_end_step = end_step

    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_composition_substitution_schedule",
        "component_order": list(COMPONENT_ORDER),
        "vial_binding": {
            "vial_id": vial_record["vial_id"],
            "batch_id": vial_record["batch_id"],
            "render_seed": vial_record["render_seed"],
            "partition": vial_record["partition"],
        },
        "provenance_binding": {
            "cohort_plan_sha256": _canonical_sha256(plan),
            "vial_record_sha256": _canonical_sha256(vial_record),
            "source_manifest_sha256": _canonical_sha256(plan["source_sha256"]),
        },
        "initial_state": {
            "physics_step": 0,
            "particle_count": 0,
            "particle_id_range": [0, 0],
            "extra_residual_slug_present": False,
        },
        "timing": deepcopy(TIMING),
        "pours": pours,
        "totals": {
            "initial_extra_particle_count": 0,
            "emitted_particle_count": particle_id,
            "final_particle_count": particle_id,
            "maximum_particle_count": particle_id,
            "programmed_total_volume_ml": 75.0,
            "count_derived_total_volume_ml": particle_id
            * PARTICLE_VOLUME_M3
            * 1_000_000.0,
            "fixed_particle_count_vs_clean": True,
        },
        "stage_volume_gates": deepcopy(plan["stage_contract"]),
        "render_gate": {
            "status": "blocked",
            "reason": "runner_schedule_ingest_not_implemented",
            "runner_compatible": False,
        },
        "synthetic_oracle_record": {
            "composition_substitution_alpha": alpha,
            "expected_disposition": vial_record["expected_disposition"],
            "excluded_from_model_inputs": True,
            "excluded_from_deploy_inputs": True,
        },
        "future_model_input_record": {
            "vial_id": vial_record["vial_id"],
            "batch_id": vial_record["batch_id"],
            "vial_position": vial_record["vial_ordinal"],
            "data_origin": "twin",
            "split_role": vial_record["partition"],
            "split_version": plan["split_version"],
            "previous_sim_recipe_id": vial_record["previous_sim_recipe_id"],
            "current_sim_recipe_id": vial_record["current_sim_recipe_id"],
            "previous_programmed_ml": dict(vial_record["previous_programmed_ml"]),
            "current_programmed_ml": dict(vial_record["current_programmed_ml"]),
            "programmed_dispense_order": list(
                vial_record["programmed_dispense_order"]
            ),
            "stage_image_evidence_ids": [],
            "quantitative_evidence_ids": [],
        },
    }


def validate_composition_cohort_plan(
    plan: object, *, repo_root: str | Path = REPO
) -> None:
    """Validate one composition cohort plan."""

    if not isinstance(plan, Mapping):
        raise TypeError("composition cohort plan must be an object")
    if plan != build_composition_cohort_plan(repo_root=repo_root):
        raise ValueError("composition cohort plan drift or leakage")


def validate_composition_execution_schedule(
    schedule: object, *, plan: object, vial_record: object
) -> None:
    """Validate one future-runner execution schedule."""

    if not isinstance(schedule, Mapping):
        raise TypeError("composition execution schedule must be an object")
    expected = build_composition_execution_schedule(
        plan=plan, vial_record=vial_record
    )
    if schedule != expected:
        raise ValueError("composition execution schedule drift or leakage")


def build_composition_cohort_bundle(
    *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Return the plan and every deterministic per-vial schedule."""

    plan = build_composition_cohort_plan(repo_root=repo_root)
    schedules = [
        build_composition_execution_schedule(plan=plan, vial_record=vial)
        for vial in plan["vials"]
    ]
    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_composition_substitution_bundle",
        "cohort_plan": plan,
        "cohort_plan_sha256": _canonical_sha256(plan),
        "execution_schedules": schedules,
        "execution_schedule_sha256_by_vial": {
            schedule["vial_binding"]["vial_id"]: _canonical_sha256(schedule)
            for schedule in schedules
        },
        "runner_integration_status": {
            "implemented": False,
            "render_authorized": False,
            "block_reason": "runner_schedule_ingest_not_implemented",
        },
    }

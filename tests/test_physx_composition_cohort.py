"""Contracts for the 120 Hz composition-substitution cohort."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_composition_cohort import build_composition_cohort_plan  # noqa: E402
from fluidmix.physx_composition_cohort import (  # noqa: E402
    build_composition_cohort_bundle,
    build_composition_execution_schedule,
    validate_composition_cohort_plan,
    validate_composition_execution_schedule,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def test_plan_has_two_reference_batches_and_balanced_scored_three_vial_profiles():
    plan = build_composition_cohort_plan()

    assert plan["schema_version"] == 1
    assert plan["artifact_type"] == "physx_f2_train_composition_substitution_cohort"
    assert plan["partition"] == "train"
    assert plan["batch_size_vials"] == 3
    assert plan["denominators"] == {
        "families": 1,
        "batches": 6,
        "reference_only_batches": 2,
        "scored_target_batches": 4,
        "vials": 18,
        "reference_only_vials": 6,
        "scored_target_vials": 12,
        "scored_keep_vials": 6,
        "scored_discard_vials": 6,
        "stage_captures": 54,
        "scored_stage_targets": 36,
        "planned_reference_eligible_stage_targets": 36,
        "observed_reference_valid_stage_targets": 0,
    }
    references = [
        batch for batch in plan["batches"] if batch["cohort_role"] == "reference_only"
    ]
    targets = [
        batch for batch in plan["batches"] if batch["cohort_role"] == "scored_target"
    ]
    assert len(references) == 2
    assert len({batch["batch_id"] for batch in references}) == 2
    assert [batch["profile_id"] for batch in targets] == [
        "clean",
        "first_only",
        "first_two",
        "all_three",
    ]
    assert [batch["composition_substitution_alpha_by_vial"] for batch in targets] == [
        [0.0, 0.0, 0.0],
        [0.10, 0.0, 0.0],
        [0.10, 0.05, 0.0],
        [0.10, 0.05, 0.025],
    ]


def test_plan_binds_train_recipe_120hz_timing_and_programmed_volume_gates():
    plan = build_composition_cohort_plan()

    assert plan["data_origin"] == "twin"
    assert plan["split_version"] == "inn-7-splits-v2"
    assert plan["family"] == {
        "family_id": "TR_COMPOSITION_SUBSTITUTION",
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
    expected = [
        ("AFTER_T200", 2704, 40.0, 1481, 0.07648894094393233, 0.07646408203812555),
        ("AFTER_T400", 4260, 65.0, 2407, 0.12429452903389002, 0.12427349457513044),
        ("FINAL_PLUS_0P3S", 5368, 75.0, 2777, 0.1434167642698731, 0.14337660757587756),
    ]
    assert len(plan["stage_contract"]) == 3
    for stage, values in zip(plan["stage_contract"], expected, strict=True):
        stage_id, step, volume_ml, count, programmed_height, count_height = values
        assert stage["stage_id"] == stage_id
        assert stage["physics_step"] == step
        assert stage["programmed_cumulative_volume_ml"] == volume_ml
        assert stage["expected_cumulative_particle_count"] == count
        assert stage["programmed_height_m"] == pytest.approx(programmed_height)
        assert stage["count_derived_body_height_m"] == pytest.approx(count_height)
        assert stage["height_tolerance_m"] == 0.006
        assert stage["required_counts"] == {
            "emitted": count,
            "active": count,
            "inside": count,
            "in_flight": 0,
            "outside": 0,
            "escaped": 0,
        }
    assert plan["volume_geometry"] == {
        "cross_section_area_m2": 0.0005229514163272396,
        "circumradius_m": 0.0132029216,
        "apothem_m": 0.012753042955909788,
        "floor_top_z_m": 0.9762065645,
        "copied_twin_liquid_bottom_z_m": 0.9762065644917131,
        "floor_alignment_tolerance_m": 1e-6,
        "floor_alignment_passed": True,
        "opaque_base_visibility": "requires_top_level_rgb_stage_review",
    }
    assert plan["per_stage_evidence_required"] == [
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
    ]
    assert plan["particle_render_diagnostics"] == {
        "sphere_prototype_radius_m": 0.0015,
        "fluid_rest_offset_m": 0.0015,
        "radius_to_fluid_rest_offset_ratio": 1.0,
        "sphere_prototypes_rgb_and_semantic_visible": False,
        "point_instancer_scale_range_required_per_stage": True,
        "count_driven_body_visible_for_positive_count": True,
        "count_driven_body_material_binding_required": True,
    }
    seeds = [seed for batch in plan["batches"] for seed in batch["render_seed_by_vial"]]
    assert seeds == list(range(330001, 330019))
    assert len(seeds) == len(set(seeds))
    assert all(0 <= seed <= 999999 for seed in seeds)


def test_vial_oracles_references_model_inputs_and_runtime_blocker_are_explicit():
    plan = build_composition_cohort_plan()

    assert len(plan["vials"]) == 18
    assert len({vial["vial_id"] for vial in plan["vials"]}) == 18
    by_batch = {
        batch["batch_id"]: sorted(
            [vial for vial in plan["vials"] if vial["batch_id"] == batch["batch_id"]],
            key=lambda vial: vial["vial_ordinal"],
        )
        for batch in plan["batches"]
    }
    scored_labels = []
    for batch in plan["batches"]:
        vials = by_batch[batch["batch_id"]]
        assert [vial["composition_substitution_alpha"] for vial in vials] == batch[
            "composition_substitution_alpha_by_vial"
        ]
        assert [vial["render_seed"] for vial in vials] == batch["render_seed_by_vial"]
        assert all(vial["partition"] == "train" for vial in vials)
        assert all(vial["current_sim_recipe_id"] == "S0117" for vial in vials)
        assert all(vial["previous_sim_recipe_id"] == "S0437" for vial in vials)
        if batch["cohort_role"] == "reference_only":
            assert all(vial["scored_target"] is False for vial in vials)
            assert all(vial["expected_disposition"] is None for vial in vials)
        else:
            assert all(vial["scored_target"] is True for vial in vials)
            scored_labels.extend(vial["expected_disposition"] for vial in vials)
    assert scored_labels.count("keep") == 6
    assert scored_labels.count("discard") == 6
    assert plan["reference_policy"] == {
        "reference_batch_ids": ["F2CS_TR_R01", "F2CS_TR_R02"],
        "exact_same_recipe_required": True,
        "minimum_independent_batch_ids": 2,
        "reference_only_batches_never_scored_targets": True,
        "reference_validity_requires_rendered_measurements": True,
        "planned_eligibility_is_not_observed_validity": True,
    }
    assert plan["partition_policy"] == {
        "whole_transition_family_one_partition": True,
        "whole_three_vial_batch_one_partition": True,
        "training_uses_train_family_only": True,
        "val_and_test_families_present": False,
        "test_data_used_for_training_or_tuning": False,
        "thresholds_freeze_on_validation_before_test": True,
    }
    assert plan["render_gate"] == {
        "status": "blocked",
        "block_reasons": [
            "accepted_a34_full_run_receipt_required",
            "runner_schedule_ingest_not_implemented",
        ],
        "accepted_a34_full_run_receipt_sha256": None,
        "render_authorized": False,
        "training_eligible": False,
    }
    assert plan["runtime_handoff"] == {
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
    }
    assert plan["composition_substitution_contract"] == {
        "previous_recipe_component_fraction_t200_t400_t100": [0.125, 0.125, 0.75],
        "rule": "(1-alpha)*current_tank_one_hot + alpha*previous_recipe_final_fraction",
        "fixed_particle_count": True,
        "initial_extra_particle_count": 0,
        "programmed_volume_displaced": False,
        "stream_material_remains_current_tank_source_colour": True,
        "uncalibrated": True,
        "physical_residue_fraction": False,
    }
    assert plan["real_data_used"] is False
    assert plan["physical_carryover_claimed"] is False
    assert plan["sim_to_real_claimed"] is False
    assert "no physical carryover" in plan["claim_boundary"]


def test_execution_schedule_preserves_volume_ids_and_substitutes_composition_only():
    plan = build_composition_cohort_plan()
    vial = next(
        row
        for row in plan["vials"]
        if row["profile_id"] == "first_two" and row["vial_ordinal"] == 2
    )
    schedule = build_composition_execution_schedule(plan=plan, vial_record=vial)

    assert schedule["schema_version"] == 1
    assert schedule["artifact_type"] == "physx_f2_composition_substitution_schedule"
    assert schedule["vial_binding"] == {
        "vial_id": vial["vial_id"],
        "batch_id": "F2CS_TR_T03",
        "render_seed": 330014,
        "partition": "train",
    }
    assert schedule["initial_state"] == {
        "physics_step": 0,
        "particle_count": 0,
        "particle_id_range": [0, 0],
        "extra_residual_slug_present": False,
    }
    assert [
        {
            key: pour[key]
            for key in (
                "ordinal",
                "tank_id",
                "programmed_volume_ml",
                "particle_count",
                "particle_id_range",
                "start_step",
                "duration_steps",
                "end_step",
                "stage_capture_step",
                "emission_chunk_count",
                "stream_one_hot_composition",
                "emitted_particle_composition",
            )
        }
        for pour in schedule["pours"]
    ] == [
        {
            "ordinal": 1,
            "tank_id": "t200",
            "programmed_volume_ml": 40,
            "particle_count": 1481,
            "particle_id_range": [0, 1481],
            "start_step": 1,
            "duration_steps": 2704,
            "end_step": 2704,
            "stage_capture_step": 2704,
            "emission_chunk_count": 226,
            "stream_one_hot_composition": [1.0, 0.0, 0.0],
            "emitted_particle_composition": [0.95625, 0.00625, 0.0375],
        },
        {
            "ordinal": 2,
            "tank_id": "t400",
            "programmed_volume_ml": 25,
            "particle_count": 926,
            "particle_id_range": [1481, 2407],
            "start_step": 2705,
            "duration_steps": 1556,
            "end_step": 4260,
            "stage_capture_step": 4260,
            "emission_chunk_count": 130,
            "stream_one_hot_composition": [0.0, 1.0, 0.0],
            "emitted_particle_composition": [0.00625, 0.95625, 0.0375],
        },
        {
            "ordinal": 3,
            "tank_id": "t100",
            "programmed_volume_ml": 10,
            "particle_count": 370,
            "particle_id_range": [2407, 2777],
            "start_step": 4261,
            "duration_steps": 1072,
            "end_step": 5332,
            "stage_capture_step": 5368,
            "emission_chunk_count": 90,
            "stream_one_hot_composition": [0.0, 0.0, 1.0],
            "emitted_particle_composition": [0.00625, 0.00625, 0.9875],
        },
    ]
    assert [pour["emission_schedule"][0]["physics_step"] for pour in schedule["pours"]] == [
        12,
        2716,
        4272,
    ]
    assert [pour["emission_schedule"][-1]["physics_step"] for pour in schedule["pours"]] == [
        2704,
        4260,
        5332,
    ]
    assert all(
        sum(event["particle_count"] for event in pour["emission_schedule"])
        == pour["particle_count"]
        for pour in schedule["pours"]
    )
    ranges = [
        event["particle_id_range"]
        for pour in schedule["pours"]
        for event in pour["emission_schedule"]
    ]
    assert ranges[0] == [0, ranges[0][1]]
    assert ranges[-1][1] == 2777
    assert all(first[1] == second[0] for first, second in zip(ranges, ranges[1:]))
    assert schedule["totals"] == {
        "initial_extra_particle_count": 0,
        "emitted_particle_count": 2777,
        "final_particle_count": 2777,
        "maximum_particle_count": 2777,
        "programmed_total_volume_ml": 75.0,
        "count_derived_total_volume_ml": pytest.approx(74.979),
        "fixed_particle_count_vs_clean": True,
    }
    assert schedule["stage_volume_gates"] == plan["stage_contract"]
    assert schedule["render_gate"] == {
        "status": "blocked",
        "reason": "runner_schedule_ingest_not_implemented",
        "runner_compatible": False,
    }
    assert schedule["synthetic_oracle_record"] == {
        "composition_substitution_alpha": 0.05,
        "expected_disposition": "discard",
        "excluded_from_model_inputs": True,
        "excluded_from_deploy_inputs": True,
    }
    assert set(schedule["future_model_input_record"]) == {
        "vial_id",
        "batch_id",
        "vial_position",
        "data_origin",
        "split_role",
        "split_version",
        "previous_sim_recipe_id",
        "current_sim_recipe_id",
        "previous_programmed_ml",
        "current_programmed_ml",
        "programmed_dispense_order",
        "stage_image_evidence_ids",
        "quantitative_evidence_ids",
    }


def test_all_schedules_are_finite_volume_preserving_and_validator_fails_closed():
    plan = build_composition_cohort_plan()
    validate_composition_cohort_plan(plan)
    for vial in plan["vials"]:
        schedule = build_composition_execution_schedule(plan=plan, vial_record=vial)
        validate_composition_execution_schedule(
            schedule, plan=plan, vial_record=vial
        )
        assert schedule["totals"]["emitted_particle_count"] == 2777
        assert schedule["totals"]["initial_extra_particle_count"] == 0
        assert [gate["expected_cumulative_particle_count"] for gate in schedule["stage_volume_gates"]] == [
            1481,
            2407,
            2777,
        ]
        for pour in schedule["pours"]:
            composition = pour["emitted_particle_composition"]
            assert len(composition) == 3
            assert all(0.0 <= value <= 1.0 for value in composition)
            assert sum(composition) == pytest.approx(1.0, abs=1e-12)
            assert all(
                event["particle_composition"] == composition
                for event in pour["emission_schedule"]
            )

    vial = next(
        row
        for row in plan["vials"]
        if row["scored_target"] and row["composition_substitution_alpha"] > 0.0
    )
    schedule = build_composition_execution_schedule(plan=plan, vial_record=vial)
    for mutator in (
        lambda value: value["pours"][0].update(particle_count=1480),
        lambda value: value["pours"][0]["emission_schedule"][0].update(
            physics_step=13
        ),
        lambda value: value["pours"][1].update(
            emitted_particle_composition=[0.0, 1.0, 0.0]
        ),
        lambda value: value["stage_volume_gates"][0].update(
            expected_cumulative_particle_count=1482
        ),
        lambda value: value["future_model_input_record"].update(
            composition_substitution_alpha=0.10
        ),
        lambda value: value["render_gate"].update(
            status="ready", runner_compatible=True
        ),
    ):
        mutated = deepcopy(schedule)
        mutator(mutated)
        with pytest.raises(ValueError, match="drift|leakage"):
            validate_composition_execution_schedule(
                mutated, plan=plan, vial_record=vial
            )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda plan: plan.update(partition="val"),
        lambda plan: plan["batches"][0]["render_seed_by_vial"].__setitem__(
            1, 330001
        ),
        lambda plan: plan["vials"][6].update(composition_substitution_alpha=0.2),
        lambda plan: plan["stage_contract"][0].update(
            programmed_cumulative_volume_ml=41.0
        ),
        lambda plan: plan["reference_policy"].update(
            minimum_independent_batch_ids=1
        ),
        lambda plan: plan["render_gate"].update(
            status="ready", render_authorized=True
        ),
        lambda plan: plan.update(real_data_used=True),
    ],
)
def test_plan_validator_rejects_split_seed_oracle_volume_reference_and_gate_drift(
    mutator,
):
    plan = build_composition_cohort_plan()
    mutated = deepcopy(plan)
    mutator(mutated)
    with pytest.raises(ValueError, match="drift|leakage"):
        validate_composition_cohort_plan(mutated)


def test_builders_and_validators_reject_non_objects_and_tampered_vial():
    plan = build_composition_cohort_plan()
    vial = plan["vials"][0]
    with pytest.raises(TypeError, match="object"):
        validate_composition_cohort_plan([])
    with pytest.raises(TypeError, match="object"):
        build_composition_execution_schedule(plan=[], vial_record=vial)
    with pytest.raises(TypeError, match="object"):
        build_composition_execution_schedule(plan=plan, vial_record=[])
    tampered = deepcopy(vial)
    tampered["render_seed"] = 1
    with pytest.raises(ValueError, match="vial_record drift"):
        build_composition_execution_schedule(plan=plan, vial_record=tampered)


def test_bundle_and_cli_write_all_hash_bound_schedules_and_refuse_overwrite(tmp_path):
    bundle = build_composition_cohort_bundle()
    assert bundle["schema_version"] == 1
    assert bundle["artifact_type"] == "physx_f2_composition_substitution_bundle"
    assert bundle["cohort_plan_sha256"] == _canonical_sha256(bundle["cohort_plan"])
    assert len(bundle["execution_schedules"]) == 18
    assert len(bundle["execution_schedule_sha256_by_vial"]) == 18
    assert {
        schedule["vial_binding"]["vial_id"] for schedule in bundle["execution_schedules"]
    } == set(bundle["execution_schedule_sha256_by_vial"])
    for schedule in bundle["execution_schedules"]:
        vial_id = schedule["vial_binding"]["vial_id"]
        assert bundle["execution_schedule_sha256_by_vial"][vial_id] == _canonical_sha256(
            schedule
        )
    assert bundle["runner_integration_status"] == {
        "implemented": False,
        "render_authorized": False,
        "block_reason": "runner_schedule_ingest_not_implemented",
    }

    output = tmp_path / "cohort-bundle.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_physx_composition_cohort.py"),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == bundle
    before = output.read_bytes()
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert repeated.returncode != 0
    assert output.read_bytes() == before

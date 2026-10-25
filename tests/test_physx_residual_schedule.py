"""Contracts for the pure residual-case execution adapter."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_residual_pilot import build_physx_residual_pilot_plan  # noqa: E402
from fluidmix.physx_residual_schedule import (  # noqa: E402
    build_residual_execution_schedule,
    validate_residual_execution_schedule,
)
from fluidmix.physx_three_pour import build_three_pour_plan  # noqa: E402


def _vial(profile_id: str, ordinal: int):
    pilot = build_physx_residual_pilot_plan()
    return next(
        vial
        for vial in pilot["vials"]
        if vial["profile_id"] == profile_id and vial["vial_ordinal"] == ordinal
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_residual_slug_is_present_at_step_zero_with_stable_ids_and_compositions():
    vial = _vial("first_two_high", 2)
    schedule = build_residual_execution_schedule(
        vial_record=vial,
        three_pour_plan=build_three_pour_plan(),
    )

    assert schedule["schema_version"] == 1
    assert schedule["artifact_type"] == "physx_f2_residual_particle_schedule"
    assert schedule["vial_binding"]["vial_id"] == vial["vial_id"]
    assert schedule["vial_binding"]["profile_id"] == "first_two_high"
    assert schedule["vial_binding"]["batch_id"] == vial["batch_id"]
    assert schedule["vial_binding"]["render_seed"] == 320014
    assert schedule["initial_residual_slug"] == {
        "physics_step": 0,
        "particle_count": 70,
        "particle_id_range": [0, 70],
        "stable_particle_ids": True,
        "component_order": ["t200", "t400", "t100"],
        "composition_runs": [
            {
                "tank_id": "t200",
                "particle_count": 9,
                "particle_id_range": [0, 9],
                "one_hot_composition": [1.0, 0.0, 0.0],
            },
            {
                "tank_id": "t400",
                "particle_count": 9,
                "particle_id_range": [9, 18],
                "one_hot_composition": [0.0, 1.0, 0.0],
            },
            {
                "tank_id": "t100",
                "particle_count": 52,
                "particle_id_range": [18, 70],
                "one_hot_composition": [0.0, 0.0, 1.0],
            },
        ],
    }


def test_pours_keep_recipe_windows_and_reduce_only_displaced_t100_particles():
    vial = _vial("first_two_high", 2)
    schedule = build_residual_execution_schedule(
        vial_record=vial,
        three_pour_plan=build_three_pour_plan(),
    )

    assert [
        {
            key: pour[key]
            for key in (
                "ordinal",
                "tank_id",
                "programmed_volume_ml",
                "clean_particle_count",
                "delivered_particle_count",
                "displaced_particle_count",
                "start_step",
                "duration_steps",
                "end_step",
                "stage_capture_step",
                "emission_chunk_count",
                "particle_id_range",
                "one_hot_composition",
            )
        }
        for pour in schedule["pours"]
    ] == [
        {
            "ordinal": 1,
            "tank_id": "t200",
            "programmed_volume_ml": 40,
            "clean_particle_count": 1481,
            "delivered_particle_count": 1481,
            "displaced_particle_count": 0,
            "start_step": 1,
            "duration_steps": 1352,
            "end_step": 1352,
            "stage_capture_step": 1352,
            "emission_chunk_count": 226,
            "particle_id_range": [70, 1551],
            "one_hot_composition": [1.0, 0.0, 0.0],
        },
        {
            "ordinal": 2,
            "tank_id": "t400",
            "programmed_volume_ml": 25,
            "clean_particle_count": 926,
            "delivered_particle_count": 926,
            "displaced_particle_count": 0,
            "start_step": 1353,
            "duration_steps": 778,
            "end_step": 2130,
            "stage_capture_step": 2130,
            "emission_chunk_count": 130,
            "particle_id_range": [1551, 2477],
            "one_hot_composition": [0.0, 1.0, 0.0],
        },
        {
            "ordinal": 3,
            "tank_id": "t100",
            "programmed_volume_ml": 10,
            "clean_particle_count": 370,
            "delivered_particle_count": 300,
            "displaced_particle_count": 70,
            "start_step": 2131,
            "duration_steps": 536,
            "end_step": 2666,
            "stage_capture_step": 2684,
            "emission_chunk_count": 90,
            "particle_id_range": [2477, 2777],
            "one_hot_composition": [0.0, 0.0, 1.0],
        },
    ]
    assert schedule["timing"] == {
        "physics_hz": 60,
        "capture_cadence_steps": 6,
        "pour_order": ["t200", "t400", "t100"],
        "pour_end_steps": [1352, 2130, 2666],
        "stage_capture_steps": [1352, 2130, 2684],
        "fill_end_step": 2666,
        "primary_final_capture_step": 2684,
        "primary_final_offset_s": 0.3,
        "no_inter_pour_settle": True,
        "no_settle_then_recolour_step": True,
    }
    assert schedule["totals"] == {
        "residual_particle_count": 70,
        "delivered_current_particle_count": 2707,
        "final_vial_particle_count": 2777,
        "clean_nominal_particle_count": 2777,
        "particle_count_conserved": True,
    }
    assert [pour["emission_schedule"][0]["physics_step"] for pour in schedule["pours"]] == [
        6,
        1358,
        2136,
    ]
    assert [pour["emission_schedule"][-1]["physics_step"] for pour in schedule["pours"]] == [
        1352,
        2130,
        2666,
    ]
    assert all(
        sum(event["particle_count"] for event in pour["emission_schedule"])
        == pour["delivered_particle_count"]
        for pour in schedule["pours"]
    )


def test_provenance_oracle_model_input_and_receipt_gate_are_separate():
    pilot = build_physx_residual_pilot_plan()
    vial = _vial("all_three_low", 3)
    profile = next(
        row for row in pilot["candidate_profiles"] if row["profile_id"] == "all_three_low"
    )
    base_plan = build_three_pour_plan()
    blocked = build_residual_execution_schedule(
        vial_record=vial,
        three_pour_plan=base_plan,
    )

    assert blocked["provenance_binding"] == {
        "vial_record_sha256": _canonical_sha256(vial),
        "profile_record_sha256": _canonical_sha256(profile),
        "batch_id_sha256": _canonical_sha256(vial["batch_id"]),
        "render_seed_sha256": _canonical_sha256(vial["render_seed"]),
        "pilot_plan_sha256": _canonical_sha256(pilot),
        "three_pour_plan_sha256": _canonical_sha256(base_plan),
        "source_manifest_sha256": _canonical_sha256(pilot["source_sha256"]),
        "source_sha256": pilot["source_sha256"],
        "previous_recipe_row_sha256": pilot["family"]["previous_recipe"][
            "row_sha256"
        ],
        "current_recipe_row_sha256": pilot["family"]["current_recipe"][
            "row_sha256"
        ],
        "transition_row_sha256": pilot["family"]["transition_row_sha256"],
    }
    assert blocked["synthetic_oracle_record"] == {
        "residual_particle_count": 14,
        "residual_source_component_counts": {"t200": 2, "t400": 2, "t100": 10},
        "expected_disposition": "discard",
        "label_source": "prescribed_synthetic_latent_oracle",
        "excluded_from_model_inputs": True,
        "excluded_from_deploy_inputs": True,
    }
    model_input = blocked["future_model_input_record"]
    assert model_input == {
        "vial_id": vial["vial_id"],
        "batch_id": vial["batch_id"],
        "vial_position": vial["vial_ordinal"],
        "data_origin": "twin",
        "split_role": "train",
        "split_version": "inn-7-splits-v2",
        "previous_sim_recipe_id": "S0437",
        "current_sim_recipe_id": "S0117",
        "previous_programmed_ml": {"t100": 60, "t200": 10, "t400": 10},
        "current_programmed_ml": {"t100": 10, "t200": 40, "t400": 25},
        "programmed_dispense_order": ["t200", "t400", "t100"],
        "timing": vial["timing"],
        "stage_image_evidence_ids": [],
        "quantitative_evidence_ids": [],
    }
    assert not {
        "residual_particle_count",
        "residual_source_component_counts",
        "expected_disposition",
        "label_source",
        "profile_id",
        "render_seed",
    }.intersection(model_input)
    assert blocked["render_gate"] == {
        "status": "blocked",
        "block_reason": "accepted_f2_receipt_hash_required",
        "accepted_f2_receipt_sha256": None,
        "render_authorized": False,
    }
    assert blocked["selection_performed"] is False

    accepted_hash = "a" * 64
    ready = build_residual_execution_schedule(
        vial_record=vial,
        three_pour_plan=base_plan,
        accepted_f2_receipt_sha256=accepted_hash,
    )
    assert ready["render_gate"] == {
        "status": "ready",
        "block_reason": None,
        "accepted_f2_receipt_sha256": accepted_hash,
        "render_authorized": True,
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["initial_residual_slug"].update(physics_step=1),
        lambda value: value["pours"][2].update(delivered_particle_count=299),
        lambda value: value["pours"][0]["emission_schedule"][0].update(
            physics_step=7
        ),
        lambda value: value["totals"].update(final_vial_particle_count=2776),
        lambda value: value["provenance_binding"].update(
            vial_record_sha256="0" * 64
        ),
        lambda value: value["future_model_input_record"].update(
            residual_particle_count=14
        ),
        lambda value: value["render_gate"].update(
            status="ready", render_authorized=True
        ),
        lambda value: value.update(selection_performed=True),
    ],
)
def test_schedule_validator_fails_closed_on_execution_provenance_or_leakage_drift(
    mutator,
):
    vial = _vial("all_three_low", 3)
    base_plan = build_three_pour_plan()
    schedule = build_residual_execution_schedule(
        vial_record=vial,
        three_pour_plan=base_plan,
    )
    validate_residual_execution_schedule(
        schedule,
        vial_record=vial,
        three_pour_plan=base_plan,
    )

    mutated = deepcopy(schedule)
    mutator(mutated)
    with pytest.raises(ValueError, match="drift|leakage"):
        validate_residual_execution_schedule(
            mutated,
            vial_record=vial,
            three_pour_plan=base_plan,
        )


def test_adapter_rejects_tampered_vial_base_plan_and_receipt_hash():
    vial = _vial("first_only_low", 1)
    base_plan = build_three_pour_plan()

    tampered_vial = deepcopy(vial)
    tampered_vial["residual_particle_count"] = 55
    with pytest.raises(ValueError, match="vial_record drift"):
        build_residual_execution_schedule(
            vial_record=tampered_vial,
            three_pour_plan=base_plan,
        )

    tampered_plan = deepcopy(base_plan)
    tampered_plan["timing"]["no_inter_pour_settle"] = False
    with pytest.raises(ValueError, match="three_pour_plan drift"):
        build_residual_execution_schedule(
            vial_record=vial,
            three_pour_plan=tampered_plan,
        )

    with pytest.raises(TypeError, match="SHA-256"):
        build_residual_execution_schedule(
            vial_record=vial,
            three_pour_plan=base_plan,
            accepted_f2_receipt_sha256=True,
        )
    with pytest.raises(ValueError, match="64 lowercase hex"):
        build_residual_execution_schedule(
            vial_record=vial,
            three_pour_plan=base_plan,
            accepted_f2_receipt_sha256="A" * 64,
        )


def test_schedule_validator_rejects_non_object():
    vial = _vial("clean", 1)
    with pytest.raises(TypeError, match="object"):
        validate_residual_execution_schedule(
            [],
            vial_record=vial,
            three_pour_plan=build_three_pour_plan(),
        )


def test_clean_control_emits_no_zero_length_slug_runs_or_displacement():
    vial = _vial("clean", 1)
    schedule = build_residual_execution_schedule(
        vial_record=vial,
        three_pour_plan=build_three_pour_plan(),
    )

    assert schedule["initial_residual_slug"]["particle_count"] == 0
    assert schedule["initial_residual_slug"]["particle_id_range"] == [0, 0]
    assert schedule["initial_residual_slug"]["composition_runs"] == []
    assert [pour["displaced_particle_count"] for pour in schedule["pours"]] == [0, 0, 0]
    assert [pour["particle_id_range"] for pour in schedule["pours"]] == [
        [0, 1481],
        [1481, 2407],
        [2407, 2777],
    ]
    assert schedule["totals"]["final_vial_particle_count"] == 2777


def test_all_frozen_vials_have_contiguous_ids_events_and_t100_only_displacement():
    pilot = build_physx_residual_pilot_plan()
    base_plan = build_three_pour_plan()

    for vial in pilot["vials"]:
        schedule = build_residual_execution_schedule(
            vial_record=vial,
            three_pour_plan=base_plan,
        )
        ranges = [schedule["initial_residual_slug"]["particle_id_range"]] + [
            pour["particle_id_range"] for pour in schedule["pours"]
        ]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 2777
        assert all(first[1] == second[0] for first, second in zip(ranges, ranges[1:]))
        assert [pour["displaced_particle_count"] for pour in schedule["pours"]] == [
            0,
            0,
            vial["residual_particle_count"],
        ]
        for pour in schedule["pours"]:
            event_ranges = [
                event["particle_id_range"] for event in pour["emission_schedule"]
            ]
            assert event_ranges[0][0] == pour["particle_id_range"][0]
            assert event_ranges[-1][1] == pour["particle_id_range"][1]
            assert all(
                first[1] == second[0]
                for first, second in zip(event_ranges, event_ranges[1:])
            )

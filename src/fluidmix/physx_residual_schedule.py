"""Pure execution scheduling for one frozen synthetic residual-pilot vial.

This adapter produces deterministic particle IDs, compositions, and emission
events. It does not launch Isaac Sim, select a profile, or establish physical
carryover or sim-to-real validity.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

from .physx_residual_pilot import build_physx_residual_pilot_plan
from .physx_three_pour import (
    CAPTURE_CADENCE_STEPS,
    build_three_pour_plan,
    distribute_particle_count,
)


COMPONENT_ORDER = ("t200", "t400", "t100")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_receipt_hash(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("accepted F2 receipt SHA-256 must be a string or None")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("accepted F2 receipt SHA-256 must be 64 lowercase hex characters")
    return value


def _frozen_vial(vial_record: object) -> tuple[Mapping[str, object], dict[str, object]]:
    if not isinstance(vial_record, Mapping):
        raise TypeError("vial_record must be an object")
    vial_id = vial_record.get("vial_id")
    if not isinstance(vial_id, str):
        raise ValueError("vial_record is missing a valid vial_id")
    pilot = build_physx_residual_pilot_plan()
    expected = next(
        (row for row in pilot["vials"] if row["vial_id"] == vial_id), None
    )
    if expected is None or vial_record != expected:
        raise ValueError("vial_record drift or leakage")
    return vial_record, pilot


def _validate_three_pour_plan(three_pour_plan: object) -> Mapping[str, object]:
    if not isinstance(three_pour_plan, Mapping):
        raise TypeError("three_pour_plan must be an object")
    if three_pour_plan != build_three_pour_plan():
        raise ValueError("three_pour_plan drift or leakage")
    return three_pour_plan


def _initial_residual_slug(vial: Mapping[str, object]) -> dict[str, object]:
    residual_plan = vial["residual_particle_plan"]
    if not isinstance(residual_plan, Mapping):
        raise ValueError("vial residual particle plan is invalid")
    counts = residual_plan["residual_source_component_counts"]
    if not isinstance(counts, Mapping):
        raise ValueError("vial residual component counts are invalid")
    particle_id = 0
    runs = []
    for component_index, tank_id in enumerate(COMPONENT_ORDER):
        count = counts[tank_id]
        if count == 0:
            continue
        end = particle_id + count
        composition = [0.0, 0.0, 0.0]
        composition[component_index] = 1.0
        runs.append(
            {
                "tank_id": tank_id,
                "particle_count": count,
                "particle_id_range": [particle_id, end],
                "one_hot_composition": composition,
            }
        )
        particle_id = end
    return {
        "physics_step": 0,
        "particle_count": residual_plan["residual_particle_count"],
        "particle_id_range": [0, residual_plan["residual_particle_count"]],
        "stable_particle_ids": True,
        "component_order": list(COMPONENT_ORDER),
        "composition_runs": runs,
    }


def _pour_schedule(vial: Mapping[str, object]) -> list[dict[str, object]]:
    residual_plan = vial["residual_particle_plan"]
    timing = vial["timing"]
    clean_counts = residual_plan["clean_current_particle_counts"]
    delivered_counts = residual_plan["delivered_current_particle_counts"]
    programmed_ml = vial["current_programmed_ml"]
    durations = timing["duration_steps_by_tank"]
    stage_steps = timing["stage_capture_steps"]
    particle_id = residual_plan["residual_particle_count"]
    previous_end_step = 0
    pours = []
    for component_index, tank_id in enumerate(COMPONENT_ORDER):
        duration_steps = durations[tank_id]
        end_step = previous_end_step + duration_steps
        delivered_count = delivered_counts[tank_id]
        chunk_count = math.ceil(duration_steps / CAPTURE_CADENCE_STEPS)
        chunk_counts = distribute_particle_count(delivered_count, chunk_count)
        composition = [0.0, 0.0, 0.0]
        composition[component_index] = 1.0
        pour_start_id = particle_id
        emission_schedule = []
        for index, count in enumerate(chunk_counts):
            event_end_id = particle_id + count
            emission_schedule.append(
                {
                    "ordinal": index + 1,
                    "tank_id": tank_id,
                    "physics_step": previous_end_step
                    + math.ceil((index + 1) * duration_steps / chunk_count),
                    "particle_count": count,
                    "particle_id_range": [particle_id, event_end_id],
                    "one_hot_composition": list(composition),
                }
            )
            particle_id = event_end_id
        clean_count = clean_counts[tank_id]
        pours.append(
            {
                "ordinal": component_index + 1,
                "tank_id": tank_id,
                "programmed_volume_ml": programmed_ml[tank_id],
                "clean_particle_count": clean_count,
                "delivered_particle_count": delivered_count,
                "displaced_particle_count": clean_count - delivered_count,
                "start_step": previous_end_step + 1,
                "duration_steps": duration_steps,
                "end_step": end_step,
                "stage_capture_step": stage_steps[component_index],
                "emission_chunk_count": chunk_count,
                "emission_schedule": emission_schedule,
                "particle_id_range": [pour_start_id, particle_id],
                "one_hot_composition": composition,
            }
        )
        previous_end_step = end_step
    return pours


def _build_expected(
    *,
    vial_record: object,
    three_pour_plan: object,
    accepted_f2_receipt_sha256: str | None = None,
) -> dict[str, object]:

    vial, pilot = _frozen_vial(vial_record)
    base_plan = _validate_three_pour_plan(three_pour_plan)
    receipt_hash = _validated_receipt_hash(accepted_f2_receipt_sha256)
    profile = next(
        row
        for row in pilot["candidate_profiles"]
        if row["profile_id"] == vial["profile_id"]
    )
    pours = _pour_schedule(vial)
    residual_plan = vial["residual_particle_plan"]
    timing = vial["timing"]
    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_residual_particle_schedule",
        "vial_binding": {
            "vial_id": vial["vial_id"],
            "profile_id": vial["profile_id"],
            "batch_id": vial["batch_id"],
            "render_seed": vial["render_seed"],
        },
        "provenance_binding": {
            "vial_record_sha256": _canonical_sha256(vial),
            "profile_record_sha256": _canonical_sha256(profile),
            "batch_id_sha256": _canonical_sha256(vial["batch_id"]),
            "render_seed_sha256": _canonical_sha256(vial["render_seed"]),
            "pilot_plan_sha256": _canonical_sha256(pilot),
            "three_pour_plan_sha256": _canonical_sha256(base_plan),
            "source_manifest_sha256": _canonical_sha256(pilot["source_sha256"]),
            "source_sha256": dict(pilot["source_sha256"]),
            "previous_recipe_row_sha256": pilot["family"]["previous_recipe"][
                "row_sha256"
            ],
            "current_recipe_row_sha256": pilot["family"]["current_recipe"][
                "row_sha256"
            ],
            "transition_row_sha256": pilot["family"]["transition_row_sha256"],
        },
        "initial_residual_slug": _initial_residual_slug(vial),
        "pours": pours,
        "timing": {
            "physics_hz": timing["physics_hz"],
            "capture_cadence_steps": CAPTURE_CADENCE_STEPS,
            "pour_order": list(COMPONENT_ORDER),
            "pour_end_steps": list(timing["pour_end_steps"]),
            "stage_capture_steps": list(timing["stage_capture_steps"]),
            "fill_end_step": timing["fill_end_step"],
            "primary_final_capture_step": timing["stage_capture_steps"][-1],
            "primary_final_offset_s": timing["primary_final_offset_s"],
            "no_inter_pour_settle": True,
            "no_settle_then_recolour_step": True,
        },
        "totals": {
            "residual_particle_count": residual_plan["residual_particle_count"],
            "delivered_current_particle_count": residual_plan[
                "delivered_current_particle_count"
            ],
            "final_vial_particle_count": residual_plan["final_vial_particle_count"],
            "clean_nominal_particle_count": residual_plan[
                "clean_nominal_particle_count"
            ],
            "particle_count_conserved": residual_plan["particle_count_conserved"],
        },
        "synthetic_oracle_record": {
            "residual_particle_count": residual_plan["residual_particle_count"],
            "residual_source_component_counts": dict(
                residual_plan["residual_source_component_counts"]
            ),
            "expected_disposition": vial["expected_disposition"],
            "label_source": vial["label_source"],
            "excluded_from_model_inputs": True,
            "excluded_from_deploy_inputs": True,
        },
        "future_model_input_record": {
            "vial_id": vial["vial_id"],
            "batch_id": vial["batch_id"],
            "vial_position": vial["vial_ordinal"],
            "data_origin": "twin",
            "split_role": vial["partition"],
            "split_version": pilot["split_version"],
            "previous_sim_recipe_id": vial["previous_sim_recipe_id"],
            "current_sim_recipe_id": vial["current_sim_recipe_id"],
            "previous_programmed_ml": dict(vial["previous_programmed_ml"]),
            "current_programmed_ml": dict(vial["current_programmed_ml"]),
            "programmed_dispense_order": list(vial["programmed_dispense_order"]),
            "timing": deepcopy(vial["timing"]),
            "stage_image_evidence_ids": [],
            "quantitative_evidence_ids": [],
        },
        "render_gate": {
            "status": "blocked" if receipt_hash is None else "ready",
            "block_reason": (
                "accepted_f2_receipt_hash_required" if receipt_hash is None else None
            ),
            "accepted_f2_receipt_sha256": receipt_hash,
            "render_authorized": receipt_hash is not None,
        },
        "selection_performed": False,
    }


def build_residual_execution_schedule(
    *,
    vial_record: object,
    three_pour_plan: object,
    accepted_f2_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Return one validated residual-aware F2 execution schedule."""

    schedule = _build_expected(
        vial_record=vial_record,
        three_pour_plan=three_pour_plan,
        accepted_f2_receipt_sha256=accepted_f2_receipt_sha256,
    )
    validate_residual_execution_schedule(
        schedule,
        vial_record=vial_record,
        three_pour_plan=three_pour_plan,
        accepted_f2_receipt_sha256=accepted_f2_receipt_sha256,
    )
    return schedule


def validate_residual_execution_schedule(
    schedule: object,
    *,
    vial_record: object,
    three_pour_plan: object,
    accepted_f2_receipt_sha256: str | None = None,
) -> None:
    """Fail closed when a schedule or any bound input is not frozen."""

    if not isinstance(schedule, Mapping):
        raise TypeError("residual execution schedule must be an object")
    expected = _build_expected(
        vial_record=vial_record,
        three_pour_plan=three_pour_plan,
        accepted_f2_receipt_sha256=accepted_f2_receipt_sha256,
    )
    if schedule != expected:
        raise ValueError("residual execution schedule drift or leakage")

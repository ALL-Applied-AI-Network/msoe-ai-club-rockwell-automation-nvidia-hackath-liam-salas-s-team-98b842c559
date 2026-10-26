"""Pure scheduling contract for leading residue at each sequential dispense."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real

from .physx_particle_contract import (
    NOMINAL_PARTICLE_VOLUME_ML,
    particle_count_for_volume_ml,
)


COMPONENT_ORDER = ("t200", "t400", "t100")


def _recipe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _tank_id(value: object, name: str) -> str:
    if not isinstance(value, str) or value not in COMPONENT_ORDER:
        raise ValueError(f"{name} must be one of {COMPONENT_ORDER}")
    return value


def _residual_particle_count(value: object) -> tuple[float, int]:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("residual_volume_ml must be a real number")
    volume = float(value)
    if not math.isfinite(volume) or volume < 0.0:
        raise ValueError("residual_volume_ml must be finite and non-negative")
    count = 0 if volume == 0.0 else particle_count_for_volume_ml(volume)
    return volume, count


def _one_hot(tank_id: str) -> list[float]:
    return [1.0 if component == tank_id else 0.0 for component in COMPONENT_ORDER]


def build_leading_residual_schedule(
    *,
    current_recipe_id: str,
    previous_recipe_id: str,
    programmed_ml: Mapping[str, object],
    dispense_order: Sequence[str],
    previous_final_tank_id: str,
    residual_volume_ml: object,
) -> dict[str, object]:
    """Return a conserved per-dispense residual schedule without importing Isaac.

    The uncalibrated leading residual replaces the same nominal volume of the
    current tank inside each programmed dispense. This conserves the programmed
    vial volume; it is a synthetic label assumption, not a measured line value.
    """

    current_id = _recipe_id(current_recipe_id, "current_recipe_id")
    previous_id = _recipe_id(previous_recipe_id, "previous_recipe_id")
    if not isinstance(programmed_ml, Mapping):
        raise TypeError("programmed_ml must be an object")
    if isinstance(dispense_order, (str, bytes)) or not isinstance(
        dispense_order, Sequence
    ):
        raise TypeError("dispense_order must be a sequence of tank IDs")
    order = [_tank_id(tank, "dispense_order tank") for tank in dispense_order]
    if not order:
        raise ValueError("dispense_order must not be empty")
    if len(set(order)) != len(order):
        raise ValueError("dispense_order tank IDs must be unique")
    predecessor = _tank_id(previous_final_tank_id, "previous_final_tank_id")
    residual_volume, residual_count = _residual_particle_count(residual_volume_ml)

    volumes: dict[str, float] = {}
    particle_counts: dict[str, int] = {}
    for tank_id in order:
        if tank_id not in programmed_ml:
            raise ValueError(f"programmed_ml is missing {tank_id}")
        value = programmed_ml[tank_id]
        count = particle_count_for_volume_ml(value)
        volume = float(value)
        if residual_volume > volume:
            raise ValueError(
                "residual_volume_ml must not exceed any programmed dispense volume"
            )
        volumes[tank_id] = volume
        particle_counts[tank_id] = count

    particle_id = 0
    pours: list[dict[str, object]] = []
    for ordinal, tank_id in enumerate(order, start=1):
        total_count = particle_counts[tank_id]
        current_count = total_count - residual_count
        pour_start = particle_id
        residual_end = particle_id + residual_count
        residual_segment = {
            "ordinal": 1,
            "segment_role": "leading_residual",
            "source_tank_id": predecessor,
            "particle_count": residual_count,
            "particle_id_range": [particle_id, residual_end],
            "one_hot_composition": _one_hot(predecessor),
        }
        current_end = residual_end + current_count
        current_segment = {
            "ordinal": 2,
            "segment_role": "current_tank",
            "source_tank_id": tank_id,
            "particle_count": current_count,
            "particle_id_range": [residual_end, current_end],
            "one_hot_composition": _one_hot(tank_id),
        }
        particle_id = current_end
        pours.append(
            {
                "ordinal": ordinal,
                "tank_id": tank_id,
                "programmed_volume_ml": volumes[tank_id],
                "programmed_particle_count": total_count,
                "residual_source_tank_id": predecessor,
                "leading_residual_volume_ml": residual_volume,
                "leading_residual_particle_count": residual_count,
                "current_tank_particle_count": current_count,
                "particle_id_range": [pour_start, current_end],
                "emission_segments": [residual_segment, current_segment],
            }
        )
        predecessor = tank_id

    programmed_total = sum(volumes[tank_id] for tank_id in order)
    programmed_particle_count = sum(particle_counts[tank_id] for tank_id in order)
    model_input_record = {
        "current_recipe_id": current_id,
        "previous_recipe_id": previous_id,
        "programmed_ml": {tank_id: volumes[tank_id] for tank_id in order},
        "dispense_order": list(order),
        "previous_final_tank_id": previous_final_tank_id,
    }
    return {
        "schema_version": 1,
        "artifact_type": "physx_f2_leading_residual_schedule",
        "physical_sequence": {
            "premixed_before_vial": False,
            "mixing_location": "vial",
            "dispense_order": list(order),
            "previous_final_tank_id": previous_final_tank_id,
        },
        "pours": pours,
        "totals": {
            "programmed_volume_ml": programmed_total,
            "programmed_particle_count": programmed_particle_count,
            "emitted_particle_count": particle_id,
            "particle_count_conserved": particle_id == programmed_particle_count,
        },
        "label_provenance": {
            "mechanism_source": "liam_operator_correction_2026-10-26",
            "leading_residual_volume_ml": residual_volume,
            "leading_residual_particle_count_per_dispense": residual_count,
            "calibrated_to_real_line": False,
            "parameter_role": "synthetic_carryover_severity_label_only",
            "excluded_from_model_inputs": True,
            "displacement_assumption": (
                "leading_residual_displaces_equal_current_tank_volume_within_"
                "each_programmed_dispense"
            ),
            "nominal_particle_volume_ml": NOMINAL_PARTICLE_VOLUME_ML,
        },
        "model_input_record": model_input_record,
    }

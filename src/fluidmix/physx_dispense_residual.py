"""Pure scheduling contract for leading residue at each sequential dispense."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real

from .physx_particle_contract import (
    NOMINAL_PARTICLE_VOLUME_ML,
    particle_count_for_volume_ml,
)


COMPONENT_ORDER = ("t200", "t400", "t100")
RECIPE8_PROGRAMMED_ML = {"t100": 20, "t200": 20, "t400": 20}
RECIPE8_DISPENSE_ORDER = ("t200", "t400", "t100")
RECIPE8_DURATION_STEPS_120HZ = {"t200": 1842, "t400": 1440, "t100": 1164}
RECIPE8_PREVIOUS_FINAL_TANK_ID = "t100"
RECIPE8_PHYSICS_HZ = 120
RECIPE8_CAPTURE_CADENCE_STEPS = 12
MEASURED_PACKING_RECEIPT_SHA256 = (
    "11173728015f6c8371e7639987fc5992517d1b60271477252b9b0de0975eb934"
)
MEASURED_PACKING_SOURCE_PARTICLE_COUNT = 640
MEASURED_PACKING_SOURCE_PROGRAMMED_VOLUME_ML = 10.0
MEASURED_PACKING_SOURCE_REPRESENTED_VOLUME_ML = 7.214634839964758
MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML = 0.011272866937444935
MEASURED_PACKING_PARTICLES_PER_ML = 88.70857835448346
MEASURED_PACKING_SOLVER_POSITION_ITERATIONS = 16
MEASURED_PACKING_POST_EMISSION_SETTLE_S = 3.0


def _recipe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _tank_id(value: object, name: str) -> str:
    if not isinstance(value, str) or value not in COMPONENT_ORDER:
        raise ValueError(f"{name} must be one of {COMPONENT_ORDER}")
    return value


def _effective_particle_volume(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("effective_particle_volume_ml must be a real number or None")
    volume = float(value)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("effective_particle_volume_ml must be finite and positive")
    return volume


def _count_for_volume_ml(
    volume_ml: object,
    *,
    effective_particle_volume_ml: float | None,
    allow_zero: bool,
) -> tuple[float, int]:
    if effective_particle_volume_ml is None:
        if (
            allow_zero
            and not isinstance(volume_ml, bool)
            and isinstance(volume_ml, Real)
            and float(volume_ml) == 0.0
        ):
            return 0.0, 0
        count = particle_count_for_volume_ml(volume_ml)
        return float(volume_ml), count
    if isinstance(volume_ml, bool) or not isinstance(volume_ml, Real):
        raise TypeError("volume must be a real number")
    volume = float(volume_ml)
    invalid_sign = volume < 0.0 or (not allow_zero and volume == 0.0)
    if not math.isfinite(volume) or invalid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"volume must be finite and {qualifier}")
    count = int(math.floor(volume / effective_particle_volume_ml + 0.5))
    if volume > 0.0 and count == 0:
        raise ValueError(
            "positive volume is below half one effective particle and rounds to zero"
        )
    return volume, count


def _residual_particle_count(
    value: object, *, effective_particle_volume_ml: float | None = None
) -> tuple[float, int]:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("residual_volume_ml must be a real number")
    volume = float(value)
    if not math.isfinite(volume) or volume < 0.0:
        raise ValueError("residual_volume_ml must be finite and non-negative")
    return _count_for_volume_ml(
        volume,
        effective_particle_volume_ml=effective_particle_volume_ml,
        allow_zero=True,
    )


def _particle_count_calibration(effective_particle_volume_ml: float) -> dict[str, object]:
    if not math.isclose(
        effective_particle_volume_ml,
        MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    ):
        raise ValueError(
            "recipe8 effective_particle_volume_ml must equal the measured-packing value"
        )
    return {
        "mode": "measured_packing_effective_particle_volume",
        "source_receipt_sha256": MEASURED_PACKING_RECEIPT_SHA256,
        "source_particle_count": MEASURED_PACKING_SOURCE_PARTICLE_COUNT,
        "source_programmed_volume_ml": MEASURED_PACKING_SOURCE_PROGRAMMED_VOLUME_ML,
        "source_represented_volume_ml": MEASURED_PACKING_SOURCE_REPRESENTED_VOLUME_ML,
        "source_solver_position_iterations": (
            MEASURED_PACKING_SOLVER_POSITION_ITERATIONS
        ),
        "source_post_emission_settle_s": MEASURED_PACKING_POST_EMISSION_SETTLE_S,
        "effective_particle_volume_ml": MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        "particles_per_ml": MEASURED_PACKING_PARTICLES_PER_ML,
        "count_rounding_rule": "nearest_integer_half_up",
        "scaling_beyond_source_10ml_validated": False,
        "interpretation": (
            "renderer_packing_scaling_from_one_10ml_diagnostic_not_real_fluid_"
            "density_or_sim_to_real_calibration"
        ),
    }


def _one_hot(tank_id: str) -> list[float]:
    return [1.0 if component == tank_id else 0.0 for component in COMPONENT_ORDER]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _distributed_counts(total: int, chunks: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, chunks)
    return tuple(
        quotient + (1 if ordinal < remainder else 0)
        for ordinal in range(chunks)
    )


def _segment_duration_steps(
    *, total_duration_steps: int, segment_counts: Sequence[int]
) -> tuple[int, ...]:
    """Allocate contiguous time windows at the pour's constant nominal rate."""

    if len(segment_counts) != 2:
        raise ValueError("exactly two residue/current segments are required")
    residual_count, current_count = segment_counts
    total_count = residual_count + current_count
    if total_duration_steps < sum(count > 0 for count in segment_counts):
        raise ValueError(
            "pour duration must provide at least one physics step per nonempty segment"
        )
    if residual_count == 0:
        return (0, total_duration_steps)
    if current_count == 0:
        return (total_duration_steps, 0)
    residual_duration = int(
        math.floor(total_duration_steps * residual_count / total_count + 0.5)
    )
    residual_duration = min(max(residual_duration, 1), total_duration_steps - 1)
    return (residual_duration, total_duration_steps - residual_duration)


def _runtime_segment(
    segment: Mapping[str, object],
    *,
    pour_ordinal: int,
    programmed_tank_id: str,
    segment_ordinal: int,
    previous_end_step: int,
    duration_steps: int,
    capture_cadence_steps: int,
) -> dict[str, object]:
    count = int(segment["particle_count"])
    source_tank_id = str(segment["source_tank_id"])
    particle_start, particle_end = segment["particle_id_range"]
    if count == 0:
        return {
            **deepcopy(dict(segment)),
            "start_step": None,
            "duration_steps": 0,
            "end_step": previous_end_step,
            "emission_event_count": 0,
            "emission_schedule": [],
        }

    start_step = previous_end_step + 1
    end_step = previous_end_step + duration_steps
    chunk_count = min(count, math.ceil(duration_steps / capture_cadence_steps))
    counts = _distributed_counts(count, chunk_count)
    events: list[dict[str, object]] = []
    next_particle_id = int(particle_start)
    segment_role = str(segment["segment_role"])
    for event_ordinal, event_count in enumerate(counts, start=1):
        event_end_id = next_particle_id + event_count
        physics_step = previous_end_step + math.ceil(
            event_ordinal * duration_steps / chunk_count
        )
        event_id = (
            f"pour-{pour_ordinal:02d}-{programmed_tank_id}-"
            f"segment-{segment_ordinal:02d}-{segment_role}-{source_tank_id}-"
            f"event-{event_ordinal:04d}"
        )
        events.append(
            {
                "event_id": event_id,
                "birth_event_id": event_id,
                "pour_event_id": f"pour-{pour_ordinal:02d}-{programmed_tank_id}",
                "ordinal": event_ordinal,
                "pour_ordinal": pour_ordinal,
                "segment_ordinal": segment_ordinal,
                "segment_role": segment_role,
                "programmed_tank_id": programmed_tank_id,
                # The runner historically consumes tank_id as the emitted source.
                "tank_id": source_tank_id,
                "source_tank_id": source_tank_id,
                "physics_step": physics_step,
                "particle_count": event_count,
                "particle_id_range": [next_particle_id, event_end_id],
                "particle_composition": list(segment["one_hot_composition"]),
            }
        )
        next_particle_id = event_end_id
    if next_particle_id != int(particle_end):
        raise RuntimeError("runtime emission schedule did not conserve particle IDs")
    return {
        **deepcopy(dict(segment)),
        "start_step": start_step,
        "duration_steps": duration_steps,
        "end_step": end_step,
        "emission_event_count": len(events),
        "emission_schedule": events,
    }


def build_leading_residual_schedule(
    *,
    current_recipe_id: str,
    previous_recipe_id: str,
    programmed_ml: Mapping[str, object],
    dispense_order: Sequence[str],
    previous_final_tank_id: str,
    residual_volume_ml: object,
    effective_particle_volume_ml: object | None = None,
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
    effective_volume = _effective_particle_volume(effective_particle_volume_ml)
    residual_volume, residual_count = _residual_particle_count(
        residual_volume_ml,
        effective_particle_volume_ml=effective_volume,
    )

    volumes: dict[str, float] = {}
    particle_counts: dict[str, int] = {}
    for tank_id in order:
        if tank_id not in programmed_ml:
            raise ValueError(f"programmed_ml is missing {tank_id}")
        value = programmed_ml[tank_id]
        volume, count = _count_for_volume_ml(
            value,
            effective_particle_volume_ml=effective_volume,
            allow_zero=False,
        )
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
    schedule = {
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
    if effective_volume is not None:
        schedule["particle_count_calibration"] = _particle_count_calibration(
            effective_volume
        )
        schedule["label_provenance"][
            "effective_particle_volume_ml"
        ] = effective_volume
        schedule["label_provenance"][
            "particle_count_basis"
        ] = "measured_renderer_packing"
    return schedule


def build_runtime_leading_residual_schedule(
    *,
    current_recipe_id: str,
    previous_recipe_id: str,
    programmed_ml: Mapping[str, object],
    dispense_order: Sequence[str],
    previous_final_tank_id: str,
    residual_volume_ml: object,
    duration_steps_by_tank: Mapping[str, object],
    physics_hz: object,
    capture_cadence_steps: object,
    final_capture_offsets_s: Sequence[object] = (0.3, 3.0),
    effective_particle_volume_ml: object | None = None,
) -> dict[str, object]:
    """Add source-pure runtime events to the conserved residue schedule.

    Segment windows are proportional to their particle counts, representing a
    constant nominal emission rate. This is a scheduling assumption only; the
    PhysX runtime remains responsible for particle motion and containment.
    """

    base = build_leading_residual_schedule(
        current_recipe_id=current_recipe_id,
        previous_recipe_id=previous_recipe_id,
        programmed_ml=programmed_ml,
        dispense_order=dispense_order,
        previous_final_tank_id=previous_final_tank_id,
        residual_volume_ml=residual_volume_ml,
        effective_particle_volume_ml=effective_particle_volume_ml,
    )
    hz = _positive_integer(physics_hz, "physics_hz")
    cadence = _positive_integer(
        capture_cadence_steps, "capture_cadence_steps"
    )
    if not isinstance(duration_steps_by_tank, Mapping):
        raise TypeError("duration_steps_by_tank must be an object")
    offsets: list[float] = []
    for value in final_capture_offsets_s:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("final_capture_offsets_s must contain real numbers")
        offset = float(value)
        if not math.isfinite(offset) or offset < 0.0:
            raise ValueError(
                "final_capture_offsets_s must contain finite non-negative values"
            )
        if not math.isclose(offset * hz, round(offset * hz), abs_tol=1.0e-12):
            raise ValueError("final capture offsets must land on a physics step")
        offsets.append(offset)

    previous_end_step = 0
    runtime_pours: list[dict[str, object]] = []
    event_ids: set[str] = set()
    for pour in base["pours"]:
        tank_id = str(pour["tank_id"])
        if tank_id not in duration_steps_by_tank:
            raise ValueError(f"duration_steps_by_tank is missing {tank_id}")
        duration_steps = _positive_integer(
            duration_steps_by_tank[tank_id], f"duration_steps_by_tank[{tank_id}]"
        )
        raw_segments = pour["emission_segments"]
        segment_durations = _segment_duration_steps(
            total_duration_steps=duration_steps,
            segment_counts=[int(row["particle_count"]) for row in raw_segments],
        )
        runtime_segments: list[dict[str, object]] = []
        segment_previous_end = previous_end_step
        for segment_ordinal, (segment, segment_duration) in enumerate(
            zip(raw_segments, segment_durations, strict=True), start=1
        ):
            runtime_segment = _runtime_segment(
                segment,
                pour_ordinal=int(pour["ordinal"]),
                programmed_tank_id=tank_id,
                segment_ordinal=segment_ordinal,
                previous_end_step=segment_previous_end,
                duration_steps=segment_duration,
                capture_cadence_steps=cadence,
            )
            runtime_segments.append(runtime_segment)
            if segment_duration:
                segment_previous_end = int(runtime_segment["end_step"])
        end_step = previous_end_step + duration_steps
        if segment_previous_end != end_step:
            raise RuntimeError("runtime segment windows do not cover the pour")
        events = [
            event
            for segment in runtime_segments
            for event in segment["emission_schedule"]
        ]
        for event in events:
            event_id = str(event["event_id"])
            if event_id in event_ids:
                raise RuntimeError("runtime event IDs are not unique")
            event_ids.add(event_id)
        runtime_pours.append(
            {
                **deepcopy(dict(pour)),
                "start_step": previous_end_step + 1,
                "duration_steps": duration_steps,
                "duration_s": duration_steps / hz,
                "end_step": end_step,
                "stage_capture_step": end_step,
                "emission_segments": runtime_segments,
                "emission_chunk_count": len(events),
                "emission_schedule": events,
            }
        )
        previous_end_step = end_step

    offset_steps = [int(round(value * hz)) for value in offsets]
    if offset_steps:
        # Intermediate stages are observed when their pour ends.  The final
        # review stage is the first frozen post-fill observation, not the
        # instant the final pour stops.
        runtime_pours[-1]["stage_capture_step"] = (
            previous_end_step + offset_steps[0]
        )
    timing: dict[str, object] = {
        "physics_hz": hz,
        "capture_cadence_steps": cadence,
        "duration_steps_by_tank": {
            tank_id: int(duration_steps_by_tank[tank_id])
            for tank_id in base["physical_sequence"]["dispense_order"]
        },
        "pour_end_steps": [int(pour["end_step"]) for pour in runtime_pours],
        "fill_end_step": previous_end_step,
        "no_inter_pour_settle": True,
    }
    for offset, steps in zip(offsets, offset_steps, strict=True):
        label = str(offset).replace(".", "p")
        timing[f"fill_plus_{label}_step"] = previous_end_step + steps
    return {
        **deepcopy(base),
        "schema_version": 1,
        "artifact_type": "physx_f2_runtime_leading_residual_schedule",
        "pours": runtime_pours,
        "timing": timing,
        "runtime_event_contract": {
            "event_count": sum(
                int(pour["emission_chunk_count"]) for pour in runtime_pours
            ),
            "event_ids_unique": True,
            "one_source_identity_per_event": True,
            "segment_time_allocation": (
                "nearest_integer_duration_proportional_to_particle_count_at_"
                "constant_nominal_emission_rate"
            ),
            "events_at_source_boundary_are_not_merged": True,
        },
    }


def build_recipe8_runtime_leading_residual_schedule(
    *,
    residual_volume_ml: object,
    effective_particle_volume_ml: object | None = None,
) -> dict[str, object]:
    """Return the 120 Hz runtime schedule for real recipe 8 qualification."""

    return build_runtime_leading_residual_schedule(
        current_recipe_id="video5-recipe-8",
        previous_recipe_id="video5-recipe-7",
        programmed_ml=RECIPE8_PROGRAMMED_ML,
        dispense_order=RECIPE8_DISPENSE_ORDER,
        previous_final_tank_id=RECIPE8_PREVIOUS_FINAL_TANK_ID,
        residual_volume_ml=residual_volume_ml,
        duration_steps_by_tank=RECIPE8_DURATION_STEPS_120HZ,
        physics_hz=RECIPE8_PHYSICS_HZ,
        capture_cadence_steps=RECIPE8_CAPTURE_CADENCE_STEPS,
        final_capture_offsets_s=(0.3, 3.0),
        effective_particle_volume_ml=effective_particle_volume_ml,
    )

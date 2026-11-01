"""Pure planning for synthetic residual particles in the F2 PBD simulation.

The residual schedule is a prescribed synthetic latent variable. Particle
motion and later diffusion can be simulated, but this planner is not a pipe
model, a calibrated residue-volume model, or evidence of real carryover.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Real


COMPONENT_ORDER = ("t200", "t400", "t100")
FINAL_DISPENSE_TANK = "t100"
NOMINAL_PARTICLE_VOLUME_ML = Fraction(27, 1000)


def _validated_programmed_ml(
    value: Mapping[str, object], *, role: str
) -> dict[str, Fraction]:
    if not isinstance(value, Mapping) or set(value) != set(COMPONENT_ORDER):
        raise ValueError(f"{role} recipe must contain exactly {COMPONENT_ORDER}")
    result: dict[str, Fraction] = {}
    for tank in COMPONENT_ORDER:
        amount = value[tank]
        if isinstance(amount, bool) or not isinstance(amount, Real):
            raise TypeError(f"{role} {tank} programmed mL must be numeric")
        if not math.isfinite(float(amount)) or float(amount) < 0.0:
            raise ValueError(f"{role} {tank} programmed mL must be finite and non-negative")
        result[tank] = Fraction(str(amount))
    if sum(result.values()) <= 0:
        raise ValueError(f"{role} recipe total must be positive")
    return result


def _nearest_particle_count(amount_ml: Fraction) -> int:
    quotient = amount_ml / NOMINAL_PARTICLE_VOLUME_ML
    return int(quotient + Fraction(1, 2))


def _largest_remainder_counts(
    total: int, weights: Mapping[str, Fraction]
) -> dict[str, int]:
    if total == 0:
        return {tank: 0 for tank in COMPONENT_ORDER}
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("residual source recipe total must be positive")
    quotas = {tank: Fraction(total) * weights[tank] / weight_total for tank in COMPONENT_ORDER}
    counts = {tank: int(quotas[tank]) for tank in COMPONENT_ORDER}
    remaining = total - sum(counts.values())
    ranked = sorted(
        COMPONENT_ORDER,
        key=lambda tank: (-(quotas[tank] - counts[tank]), COMPONENT_ORDER.index(tank)),
    )
    for tank in ranked[:remaining]:
        counts[tank] += 1
    return counts


def build_residual_particle_plan(
    *,
    current_programmed_ml: Mapping[str, object],
    previous_programmed_ml: Mapping[str, object],
    residual_particle_count: int,
) -> dict[str, object]:
    """Return a deterministic prior-recipe slug plus conserved current delivery.

    The programmed current recipe is unchanged. The synthetic shared-path model
    assumes the residual slug displaces an equal count of the last current fluid
    from delivery into the vial; that retained current fluid is not simulated.
    """

    current = _validated_programmed_ml(current_programmed_ml, role="current")
    previous = _validated_programmed_ml(previous_programmed_ml, role="previous")
    if isinstance(residual_particle_count, bool) or not isinstance(
        residual_particle_count, int
    ):
        raise TypeError("residual particle count must be an integer")
    if residual_particle_count < 0:
        raise ValueError("residual particle count must be non-negative")

    clean_counts = {
        tank: _nearest_particle_count(current[tank]) for tank in COMPONENT_ORDER
    }
    if residual_particle_count > clean_counts[FINAL_DISPENSE_TANK]:
        raise ValueError("residual particle count exceeds final-tank delivery")
    delivered_counts = dict(clean_counts)
    delivered_counts[FINAL_DISPENSE_TANK] -= residual_particle_count
    residual_counts = _largest_remainder_counts(residual_particle_count, previous)
    clean_total = sum(clean_counts.values())
    delivered_total = sum(delivered_counts.values())
    final_total = delivered_total + residual_particle_count
    if final_total != clean_total or any(value < 0 for value in delivered_counts.values()):
        raise RuntimeError("residual particle plan does not conserve the vial count")

    return {
        "schema_version": 1,
        "component_order": list(COMPONENT_ORDER),
        "residual_initialization": "shared_path_slug_before_t200",
        "residual_particle_count": residual_particle_count,
        "residual_source_component_counts": residual_counts,
        "clean_current_particle_counts": clean_counts,
        "delivered_current_particle_counts": delivered_counts,
        "displaced_current_tank": FINAL_DISPENSE_TANK,
        "displaced_current_particle_count": residual_particle_count,
        "clean_nominal_particle_count": clean_total,
        "delivered_current_particle_count": delivered_total,
        "final_vial_particle_count": final_total,
        "particle_count_conserved": final_total == clean_total,
        "residual_particle_id_range": [0, residual_particle_count],
        "current_particle_id_range": [residual_particle_count, final_total],
        "programmed_current_dispense_unchanged": True,
        "vial_delivery_displacement_model": "residual_slug_displaces_equal_final_tank_particle_count",
        "decay_schedule_is_prescribed_not_simulated": True,
        "residual_parameter_calibrated_to_real_line": False,
    }


def one_hot_residual_compositions(
    plan: Mapping[str, object],
) -> tuple[tuple[float, float, float], ...]:
    """Expand a reviewed residual plan into stable one-hot source identities."""

    if not isinstance(plan, Mapping) or plan.get("component_order") != list(COMPONENT_ORDER):
        raise ValueError("residual plan component order is invalid")
    counts = plan.get("residual_source_component_counts")
    total = plan.get("residual_particle_count")
    if not isinstance(counts, Mapping) or set(counts) != set(COMPONENT_ORDER):
        raise ValueError("residual source component counts are invalid")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("residual particle total is invalid")
    normalized: dict[str, int] = {}
    for tank in COMPONENT_ORDER:
        count = counts[tank]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("residual source component counts are invalid")
        normalized[tank] = count
    if sum(normalized.values()) != total:
        raise ValueError("residual source component counts do not sum to the total")

    rows: list[tuple[float, float, float]] = []
    for index, tank in enumerate(COMPONENT_ORDER):
        row = tuple(1.0 if component == index else 0.0 for component in range(3))
        rows.extend([row] * normalized[tank])
    return tuple(rows)

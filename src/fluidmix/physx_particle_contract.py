"""Reviewed CPU contract for F2's density-derived 2.5 mm particles.

This is the single owner of particle resolution, density, nominal mass and
the default three-pour count schedule.  It is a planning contract only; it
does not establish that a PhysX runtime used the configuration or displaced
the programmed liquid volume.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real


SPAWN_SPACING_M = 0.0025
FLUID_REST_OFFSET_M = 0.00125
PARTICLE_CONTACT_OFFSET_M = 0.002104377104
REST_OFFSET_M = 0.002083333333
SOLID_REST_OFFSET_M = REST_OFFSET_M
DENSITY_KG_M3 = 1000.0
PARTICLE_SET_MASS_OVERRIDE = None

# These exact decimal values are the density-derived values for 2.5 mm spacing;
# spelling them directly avoids platform-dependent floating-point roundoff in
# serialized contracts.
NOMINAL_PARTICLE_VOLUME_MM3 = 15.625
NOMINAL_PARTICLE_VOLUME_ML = 0.015625
NOMINAL_PARTICLE_MASS_MG = NOMINAL_PARTICLE_VOLUME_ML * DENSITY_KG_M3
NOMINAL_PARTICLE_MASS_KG = NOMINAL_PARTICLE_MASS_MG * 1.0e-6

THREE_POUR_ROWS = (
    ("t200", 10, 640, 706),
    ("t400", 10, 640, 605),
    ("t100", 40, 2560, 673),
)
MINIMUM_PARTICLE_CAPACITY = 4096


def particle_count_for_volume_ml(volume_ml: object) -> int:
    """Return the exact density-derived particle count for a positive volume."""

    if isinstance(volume_ml, bool) or not isinstance(volume_ml, Real):
        raise TypeError("volume_ml must be a real number")
    volume = float(volume_ml)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume_ml must be finite and positive")
    count = volume / NOMINAL_PARTICLE_VOLUME_ML
    rounded = round(count)
    if not math.isclose(count, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("volume_ml is not an integral nominal particle volume")
    return int(rounded)


def required_particle_capacity(particle_counts: Iterable[object] | None = None) -> int:
    """Return at least 4096 particles, or the supplied schedule maximum."""

    counts = (
        (row[2] for row in THREE_POUR_ROWS)
        if particle_counts is None
        else particle_counts
    )
    total = 0
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("particle_counts must contain integers")
        if count < 0:
            raise ValueError("particle_counts must be non-negative")
        total += count
    return max(MINIMUM_PARTICLE_CAPACITY, total)


def particle_contract_receipt() -> dict[str, object]:
    """Return serializable reviewed parameters for CPU plans and receipts."""

    return {
        "spawn_spacing_m": SPAWN_SPACING_M,
        "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
        "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
        "rest_offset_m": REST_OFFSET_M,
        "solid_rest_offset_m": SOLID_REST_OFFSET_M,
        "density_kg_m3": DENSITY_KG_M3,
        "particle_set_mass_override": PARTICLE_SET_MASS_OVERRIDE,
        "nominal_particle_volume_mm3": NOMINAL_PARTICLE_VOLUME_MM3,
        "nominal_particle_volume_ml": NOMINAL_PARTICLE_VOLUME_ML,
        "nominal_particle_mass_mg": NOMINAL_PARTICLE_MASS_MG,
        "nominal_particle_mass_kg": NOMINAL_PARTICLE_MASS_KG,
        "minimum_particle_capacity": MINIMUM_PARTICLE_CAPACITY,
        "required_particle_capacity": required_particle_capacity(),
    }

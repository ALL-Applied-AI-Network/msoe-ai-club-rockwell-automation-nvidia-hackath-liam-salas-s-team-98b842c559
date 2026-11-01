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

LEGACY_RESOLUTION_PROFILE_ID = "c20_2p5mm"
C21_RESOLUTION_PROFILE_ID = "c21_2p0mm"
C21_PARENT_PACKING_RECEIPT_SHA256 = (
    "11173728015f6c8371e7639987fc5992517d1b60271477252b9b0de0975eb934"
)
C21_LINEAR_RESOLUTION_SCALE = 0.8
C21_VOLUME_RESOLUTION_SCALE = 0.512
C21_EFFECTIVE_PARTICLE_VOLUME_ML = 0.005771707871971808

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


def particle_resolution_profile(profile_id: object) -> dict[str, object]:
    """Return one frozen particle-resolution profile without changing defaults.

    The C21 effective volume is an explicitly unvalidated cubic scaling of the
    earlier packing observation.  It is not a new measurement.
    """

    if not isinstance(profile_id, str):
        raise TypeError("profile_id must be a string")
    if profile_id == LEGACY_RESOLUTION_PROFILE_ID:
        return {
            "profile_id": profile_id,
            "spawn_spacing_m": SPAWN_SPACING_M,
            "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
            "particle_contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
            "contact_offset_m": PARTICLE_CONTACT_OFFSET_M,
            "rest_offset_m": REST_OFFSET_M,
            "solid_rest_offset_m": SOLID_REST_OFFSET_M,
            "nominal_particle_volume_ml": NOMINAL_PARTICLE_VOLUME_ML,
            "nominal_particle_mass_mg": NOMINAL_PARTICLE_MASS_MG,
            "render_radius_m": FLUID_REST_OFFSET_M,
            "linear_resolution_scale": 1.0,
            "volume_resolution_scale": 1.0,
            "effective_particle_volume_ml": None,
            "calibration_mode": "legacy_density_derived_nominal",
            "parent_packing_receipt_sha256": None,
        }
    if profile_id == C21_RESOLUTION_PROFILE_ID:
        return {
            "profile_id": profile_id,
            "spawn_spacing_m": 0.002,
            "fluid_rest_offset_m": 0.001,
            "particle_contact_offset_m": 0.0016835016832,
            "contact_offset_m": 0.0016835016832,
            "rest_offset_m": 0.0016666666664,
            "solid_rest_offset_m": 0.0016666666664,
            "nominal_particle_volume_ml": 0.008,
            "nominal_particle_mass_mg": 8.0,
            "render_radius_m": 0.001,
            "linear_resolution_scale": C21_LINEAR_RESOLUTION_SCALE,
            "volume_resolution_scale": C21_VOLUME_RESOLUTION_SCALE,
            "effective_particle_volume_ml": C21_EFFECTIVE_PARTICLE_VOLUME_ML,
            "calibration_mode": "resolution_scaled_effective_volume_hypothesis",
            "directly_measured": False,
            "parent_packing_receipt_sha256": C21_PARENT_PACKING_RECEIPT_SHA256,
        }
    raise ValueError(
        "profile_id must be one of: "
        f"{LEGACY_RESOLUTION_PROFILE_ID}, {C21_RESOLUTION_PROFILE_ID}"
    )


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

"""CPU contract for the reviewed 2.5 mm F2 particle configuration."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from twin.physx.three_pour import build_three_pour_plan
from twin.physx.particle_contract import (
    DENSITY_KG_M3,
    NOMINAL_PARTICLE_MASS_MG,
    NOMINAL_PARTICLE_VOLUME_MM3,
    PARTICLE_SET_MASS_OVERRIDE,
    REST_OFFSET_M,
    SOLID_REST_OFFSET_M,
    particle_count_for_volume_ml,
    required_particle_capacity,
)


def test_three_pour_plan_uses_the_density_derived_2p5_mm_schedule() -> None:
    plan = build_three_pour_plan()

    assert [pour["particle_count"] for pour in plan["pours"]] == [640, 640, 2560]
    assert [pour["cumulative_particle_count"] for pour in plan["pours"]] == [
        640,
        1280,
        3840,
    ]
    assert plan["fluid_rest_offset_m"] == 0.00125
    assert plan["nominal_particle_volume_ml"] == 0.015625
    assert plan["totals"]["particle_count"] == 3840


def test_particle_contract_keeps_density_mass_and_capacity_explicit() -> None:
    plan = build_three_pour_plan()

    assert DENSITY_KG_M3 == 1000.0
    assert NOMINAL_PARTICLE_VOLUME_MM3 == 15.625
    assert NOMINAL_PARTICLE_MASS_MG == 15.625
    assert PARTICLE_SET_MASS_OVERRIDE is None
    assert REST_OFFSET_M == SOLID_REST_OFFSET_M == 0.002083333333
    assert [particle_count_for_volume_ml(volume) for volume in (10, 10, 40)] == [
        640,
        640,
        2560,
    ]
    assert required_particle_capacity() == 4096
    assert plan["required_particle_capacity"] == 4096

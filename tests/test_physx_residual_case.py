"""Pure contracts for physically simulated synthetic residual-particle cases."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_residual_case import (
    COMPONENT_ORDER,
    build_residual_particle_plan,
    one_hot_residual_compositions,
)


def test_residual_slug_preserves_nominal_vial_particle_count():
    plan = build_residual_particle_plan(
        current_programmed_ml={"t100": 40, "t200": 10, "t400": 10},
        previous_programmed_ml={"t100": 5, "t200": 15, "t400": 40},
        residual_particle_count=37,
    )

    assert plan["schema_version"] == 1
    assert plan["component_order"] == list(COMPONENT_ORDER)
    assert plan["clean_current_particle_counts"] == {
        "t200": 370,
        "t400": 370,
        "t100": 1481,
    }
    assert plan["residual_source_component_counts"] == {
        "t200": 9,
        "t400": 25,
        "t100": 3,
    }
    assert plan["delivered_current_particle_counts"] == {
        "t200": 370,
        "t400": 370,
        "t100": 1444,
    }
    assert plan["displaced_current_tank"] == "t100"
    assert plan["displaced_current_particle_count"] == 37
    assert plan["clean_nominal_particle_count"] == 2221
    assert plan["delivered_current_particle_count"] == 2184
    assert plan["final_vial_particle_count"] == 2221
    assert plan["particle_count_conserved"] is True
    assert plan["residual_particle_id_range"] == [0, 37]
    assert plan["current_particle_id_range"] == [37, 2221]
    assert plan["residual_initialization"] == "shared_path_slug_before_t200"
    assert plan["decay_schedule_is_prescribed_not_simulated"] is True


def test_residual_compositions_are_one_hot_and_follow_component_counts():
    plan = build_residual_particle_plan(
        current_programmed_ml={"t100": 40, "t200": 10, "t400": 10},
        previous_programmed_ml={"t100": 5, "t200": 15, "t400": 40},
        residual_particle_count=37,
    )

    rows = one_hot_residual_compositions(plan)

    assert len(rows) == 37
    assert all(len(row) == 3 and sum(row) == 1.0 for row in rows)
    assert all(value in (0.0, 1.0) for row in rows for value in row)
    assert [sum(row[index] for row in rows) for index in range(3)] == [9.0, 25.0, 3.0]


def test_clean_case_has_no_residual_or_displacement():
    plan = build_residual_particle_plan(
        current_programmed_ml={"t100": 40, "t200": 10, "t400": 10},
        previous_programmed_ml={"t100": 5, "t200": 15, "t400": 40},
        residual_particle_count=0,
    )

    assert plan["residual_source_component_counts"] == {
        "t200": 0,
        "t400": 0,
        "t100": 0,
    }
    assert plan["delivered_current_particle_counts"] == plan[
        "clean_current_particle_counts"
    ]
    assert plan["residual_particle_id_range"] == [0, 0]
    assert plan["current_particle_id_range"] == [0, 2221]
    assert one_hot_residual_compositions(plan) == ()


@pytest.mark.parametrize(
    ("current", "previous", "residual", "error"),
    [
        (
            {"t100": 40, "t200": 10},
            {"t100": 5, "t200": 15, "t400": 40},
            1,
            ValueError,
        ),
        (
            {"t100": 40, "t200": 10, "t400": 10},
            {"t100": 0, "t200": 0, "t400": 0},
            1,
            ValueError,
        ),
        (
            {"t100": 1, "t200": 10, "t400": 10},
            {"t100": 5, "t200": 15, "t400": 40},
            38,
            ValueError,
        ),
        (
            {"t100": 40, "t200": 10, "t400": 10},
            {"t100": 5, "t200": 15, "t400": 40},
            True,
            TypeError,
        ),
    ],
)
def test_residual_plan_rejects_malformed_or_overfilled_cases(
    current, previous, residual, error
):
    with pytest.raises(error):
        build_residual_particle_plan(
            current_programmed_ml=current,
            previous_programmed_ml=previous,
            residual_particle_count=residual,
        )


def test_residual_allocation_is_deterministic_on_equal_remainders():
    plan = build_residual_particle_plan(
        current_programmed_ml={"t100": 40, "t200": 10, "t400": 10},
        previous_programmed_ml={"t100": 1, "t200": 1, "t400": 1},
        residual_particle_count=2,
    )

    assert plan["residual_source_component_counts"] == {
        "t200": 1,
        "t400": 1,
        "t100": 0,
    }

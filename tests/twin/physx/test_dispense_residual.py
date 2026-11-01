"""Contracts for residue-first sequential tank dispenses."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import twin.physx.dispense_residual as residual_schedule  # noqa: E402
from twin.physx.dispense_residual import build_leading_residual_schedule  # noqa: E402


def test_real_recipe_8_emits_predecessor_residue_then_current_tank_and_conserves_volume():
    schedule = build_leading_residual_schedule(
        current_recipe_id="video5-recipe-8",
        previous_recipe_id="video5-recipe-7",
        programmed_ml={"t100": 20, "t200": 20, "t400": 20},
        dispense_order=("t200", "t400", "t100"),
        previous_final_tank_id="t100",
        residual_volume_ml=1.0,
    )

    assert schedule["artifact_type"] == "physx_f2_leading_residual_schedule"
    assert schedule["physical_sequence"] == {
        "premixed_before_vial": False,
        "mixing_location": "vial",
        "dispense_order": ["t200", "t400", "t100"],
        "previous_final_tank_id": "t100",
    }
    assert [
        (pour["tank_id"], pour["residual_source_tank_id"])
        for pour in schedule["pours"]
    ] == [("t200", "t100"), ("t400", "t200"), ("t100", "t400")]
    assert [
        [
            (
                segment["segment_role"],
                segment["source_tank_id"],
                segment["particle_count"],
                segment["particle_id_range"],
                segment["one_hot_composition"],
            )
            for segment in pour["emission_segments"]
        ]
        for pour in schedule["pours"]
    ] == [
        [
            ("leading_residual", "t100", 64, [0, 64], [0.0, 0.0, 1.0]),
            ("current_tank", "t200", 1216, [64, 1280], [1.0, 0.0, 0.0]),
        ],
        [
            ("leading_residual", "t200", 64, [1280, 1344], [1.0, 0.0, 0.0]),
            ("current_tank", "t400", 1216, [1344, 2560], [0.0, 1.0, 0.0]),
        ],
        [
            ("leading_residual", "t400", 64, [2560, 2624], [0.0, 1.0, 0.0]),
            ("current_tank", "t100", 1216, [2624, 3840], [0.0, 0.0, 1.0]),
        ],
    ]
    assert schedule["totals"] == {
        "programmed_volume_ml": 60.0,
        "programmed_particle_count": 3840,
        "emitted_particle_count": 3840,
        "particle_count_conserved": True,
    }
    assert schedule["label_provenance"]["leading_residual_volume_ml"] == 1.0
    assert schedule["label_provenance"]["calibrated_to_real_line"] is False
    assert schedule["label_provenance"]["excluded_from_model_inputs"] is True
    assert "leading_residual_volume_ml" not in schedule["model_input_record"]


def test_recipe8_runtime_schedule_has_source_pure_events_and_frozen_120hz_timing():
    schedule = residual_schedule.build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=1.0,
    )

    assert schedule["artifact_type"] == "physx_f2_runtime_leading_residual_schedule"
    assert schedule["timing"] == {
        "physics_hz": 120,
        "capture_cadence_steps": 12,
        "duration_steps_by_tank": {"t200": 1842, "t400": 1440, "t100": 1164},
        "pour_end_steps": [1842, 3282, 4446],
        "fill_end_step": 4446,
        "fill_plus_0p3_step": 4482,
        "fill_plus_3p0_step": 4806,
        "no_inter_pour_settle": True,
    }
    assert [
        (pour["tank_id"], pour["start_step"], pour["end_step"])
        for pour in schedule["pours"]
    ] == [
        ("t200", 1, 1842),
        ("t400", 1843, 3282),
        ("t100", 3283, 4446),
    ]
    assert [
        [
            (
                segment["segment_role"],
                segment["source_tank_id"],
                segment["start_step"],
                segment["end_step"],
                segment["particle_count"],
            )
            for segment in pour["emission_segments"]
        ]
        for pour in schedule["pours"]
    ] == [
        [
            ("leading_residual", "t100", 1, 92, 64),
            ("current_tank", "t200", 93, 1842, 1216),
        ],
        [
            ("leading_residual", "t200", 1843, 1914, 64),
            ("current_tank", "t400", 1915, 3282, 1216),
        ],
        [
            ("leading_residual", "t400", 3283, 3340, 64),
            ("current_tank", "t100", 3341, 4446, 1216),
        ],
    ]

    events = [
        event
        for pour in schedule["pours"]
        for event in pour["emission_schedule"]
    ]
    assert len(events) == 372
    assert len({event["event_id"] for event in events}) == len(events)
    assert [events[0][key] for key in ("physics_step", "particle_count")] == [12, 8]
    assert events[-1]["physics_step"] == 4446
    assert all(
        event["particle_composition"]
        == [
            float(event["source_tank_id"] == "t200"),
            float(event["source_tank_id"] == "t400"),
            float(event["source_tank_id"] == "t100"),
        ]
        for event in events
    )
    assert all(event["tank_id"] == event["source_tank_id"] for event in events)
    assert all(
        len({event["source_tank_id"] for event in segment["emission_schedule"]})
        == 1
        for pour in schedule["pours"]
        for segment in pour["emission_segments"]
    )
    assert schedule["totals"]["emitted_particle_count"] == 3840
    assert schedule["totals"]["particle_count_conserved"] is True
    assert schedule["label_provenance"]["leading_residual_volume_ml"] == 1.0
    assert "leading_residual_volume_ml" not in schedule["model_input_record"]


def test_recipe8_clean_runtime_schedule_emits_no_zero_count_residual_events():
    schedule = residual_schedule.build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=0.0,
    )

    assert all(
        pour["emission_segments"][0]["duration_steps"] == 0
        and pour["emission_segments"][0]["emission_schedule"] == []
        and pour["emission_segments"][1]["duration_steps"]
        == pour["duration_steps"]
        for pour in schedule["pours"]
    )
    assert all(
        event["segment_role"] == "current_tank"
        and event["source_tank_id"] == pour["tank_id"]
        for pour in schedule["pours"]
        for event in pour["emission_schedule"]
    )
    assert schedule["totals"]["emitted_particle_count"] == 3840


def test_recipe8_measured_packing_mode_scales_counts_and_records_provenance():
    schedule = residual_schedule.build_recipe8_runtime_leading_residual_schedule(
        residual_volume_ml=1.0,
        effective_particle_volume_ml=0.011272866937444935,
    )

    assert [pour["programmed_particle_count"] for pour in schedule["pours"]] == [
        1774,
        1774,
        1774,
    ]
    assert [
        (
            pour["leading_residual_particle_count"],
            pour["current_tank_particle_count"],
            pour["particle_id_range"],
        )
        for pour in schedule["pours"]
    ] == [
        (89, 1685, [0, 1774]),
        (89, 1685, [1774, 3548]),
        (89, 1685, [3548, 5322]),
    ]
    assert schedule["totals"]["programmed_particle_count"] == 5322
    assert schedule["totals"]["emitted_particle_count"] == 5322
    assert schedule["totals"]["particle_count_conserved"] is True
    assert schedule["particle_count_calibration"] == {
        "mode": "measured_packing_effective_particle_volume",
        "source_receipt_sha256": (
            "11173728015f6c8371e7639987fc5992517d1b60271477252b9b0de0975eb934"
        ),
        "source_particle_count": 640,
        "source_programmed_volume_ml": 10.0,
        "source_represented_volume_ml": 7.214634839964758,
        "source_solver_position_iterations": 16,
        "source_post_emission_settle_s": 3.0,
        "effective_particle_volume_ml": 0.011272866937444935,
        "particles_per_ml": 88.70857835448346,
        "count_rounding_rule": "nearest_integer_half_up",
        "scaling_beyond_source_10ml_validated": False,
        "interpretation": (
            "renderer_packing_scaling_from_one_10ml_diagnostic_not_real_fluid_"
            "density_or_sim_to_real_calibration"
        ),
    }
    assert schedule["label_provenance"]["leading_residual_particle_count_per_dispense"] == 89
    assert schedule["label_provenance"]["excluded_from_model_inputs"] is True
    assert "leading_residual_volume_ml" not in schedule["model_input_record"]
    assert "effective_particle_volume_ml" not in schedule["model_input_record"]


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (True, TypeError, "must be a real number or None"),
        (0.0, ValueError, "must be finite and positive"),
        (0.01, ValueError, "must equal the measured-packing value"),
    ],
)
def test_recipe8_rejects_unattested_effective_particle_volume(value, error_type, message):
    with pytest.raises(error_type, match=message):
        residual_schedule.build_recipe8_runtime_leading_residual_schedule(
            residual_volume_ml=1.0,
            effective_particle_volume_ml=value,
        )


def test_runtime_schedule_rejects_a_pour_too_short_for_two_nonempty_segments():
    with pytest.raises(ValueError, match="one physics step per nonempty segment"):
        residual_schedule.build_runtime_leading_residual_schedule(
            current_recipe_id="video5-recipe-8",
            previous_recipe_id="video5-recipe-7",
            programmed_ml={"t100": 20, "t200": 20, "t400": 20},
            dispense_order=("t200", "t400", "t100"),
            previous_final_tank_id="t100",
            residual_volume_ml=1.0,
            duration_steps_by_tank={"t200": 1, "t400": 1440, "t100": 1164},
            physics_hz=120,
            capture_cadence_steps=12,
        )


def test_zero_residual_is_a_clean_conserved_schedule_with_no_hidden_particles():
    schedule = build_leading_residual_schedule(
        current_recipe_id="video5-recipe-5",
        previous_recipe_id="video5-recipe-5",
        programmed_ml={"t100": 10, "t200": 10, "t400": 10},
        dispense_order=("t200", "t400", "t100"),
        previous_final_tank_id="t100",
        residual_volume_ml=0.0,
    )

    assert schedule["totals"]["programmed_particle_count"] == 1920
    assert schedule["totals"]["emitted_particle_count"] == 1920
    assert all(
        pour["emission_segments"][0]["particle_count"] == 0
        and pour["emission_segments"][1]["particle_count"] == 640
        for pour in schedule["pours"]
    )


def test_residual_volume_must_not_exceed_the_smallest_programmed_dispense():
    with pytest.raises(
        ValueError,
        match="residual_volume_ml must not exceed any programmed dispense volume",
    ):
        build_leading_residual_schedule(
            current_recipe_id="video5-recipe-5",
            previous_recipe_id="video5-recipe-14",
            programmed_ml={"t100": 10, "t200": 10, "t400": 10},
            dispense_order=("t200", "t400", "t100"),
            previous_final_tank_id="t100",
            residual_volume_ml=10.015625,
        )


def test_residual_equal_to_dispense_is_allowed_by_the_contract():
    schedule = build_leading_residual_schedule(
        current_recipe_id="video5-recipe-5",
        previous_recipe_id="video5-recipe-14",
        programmed_ml={"t100": 10, "t200": 10, "t400": 10},
        dispense_order=("t200", "t400", "t100"),
        previous_final_tank_id="t100",
        residual_volume_ml=10.0,
    )

    assert all(pour["current_tank_particle_count"] == 0 for pour in schedule["pours"])
    assert schedule["totals"]["particle_count_conserved"] is True


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"dispense_order": ("t200", "t200")}, ValueError, "must be unique"),
        ({"previous_final_tank_id": "t300"}, ValueError, "must be one of"),
        ({"residual_volume_ml": -1.0}, ValueError, "finite and non-negative"),
        ({"residual_volume_ml": True}, TypeError, "must be a real number"),
        ({"programmed_ml": {"t200": 20, "t400": 20}}, ValueError, "missing t100"),
    ],
)
def test_invalid_schedule_inputs_are_rejected(overrides, error_type, message):
    kwargs = {
        "current_recipe_id": "video5-recipe-8",
        "previous_recipe_id": "video5-recipe-7",
        "programmed_ml": {"t100": 20, "t200": 20, "t400": 20},
        "dispense_order": ("t200", "t400", "t100"),
        "previous_final_tank_id": "t100",
        "residual_volume_ml": 1.0,
    }
    kwargs.update(overrides)

    with pytest.raises(error_type, match=message):
        build_leading_residual_schedule(**kwargs)

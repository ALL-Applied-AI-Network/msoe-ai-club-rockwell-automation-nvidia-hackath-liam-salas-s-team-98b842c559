"""Contracts for residue-first sequential tank dispenses."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_dispense_residual import build_leading_residual_schedule  # noqa: E402


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

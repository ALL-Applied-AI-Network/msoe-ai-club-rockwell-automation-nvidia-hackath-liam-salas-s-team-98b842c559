from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.physx_run_variation import (
    BASE_TANK_TRANSMISSION_1CM,
    active_dispense_order,
    build_run_design,
    build_run_palette,
    validate_whole_run_partitioning,
)


def test_recipe_order_skips_only_zero_volume_tanks():
    assert BASE_TANK_TRANSMISSION_1CM == {
        "t200": (0.95, 0.39, 0.26),
        "t400": (0.14, 0.61, 0.95),
        "t100": (1.0, 1.0, 1.0),
    }
    assert active_dispense_order(
        {"t100": 0, "t200": 20, "t400": 10}
    ) == ("t200", "t400")
    assert active_dispense_order(
        {"t100": 30, "t200": 0, "t400": 0}
    ) == ("t100",)

    with pytest.raises(ValueError, match="at least one"):
        active_dispense_order({"t100": 0, "t200": 0, "t400": 0})
    with pytest.raises(ValueError, match="non-negative"):
        active_dispense_order({"t100": 0, "t200": -1, "t400": 10})
    with pytest.raises(ValueError, match="exactly"):
        active_dispense_order({"t100": 10, "t200": 10})


def test_run_palette_is_deterministic_fixed_clear_t100_and_varies_by_run():
    run_a = build_run_palette(run_id="RUN-A", run_seed=47001)
    run_a_again = build_run_palette(run_id="RUN-A", run_seed=47001)
    run_b = build_run_palette(run_id="RUN-B", run_seed=47002)

    assert run_a == run_a_again
    assert run_a["transmission_1cm"]["t100"] == [1.0, 1.0, 1.0]
    assert run_a["transmission_1cm"] != run_b["transmission_1cm"]
    assert run_a["palette_sha256"] != run_b["palette_sha256"]
    assert set(run_a["variation_by_tank"]) == {"t200", "t400"}
    for tank in ("t200", "t400"):
        varied = run_a["transmission_1cm"][tank]
        assert len(varied) == 3
        assert all(0.0 < channel <= 1.0 for channel in varied)
        assert 0.85 <= run_a["variation_by_tank"][tank]["strength_scale"] <= 1.15
        assert all(
            0.95 <= scale <= 1.05
            for scale in run_a["variation_by_tank"][tank][
                "channel_extinction_scales"
            ]
        )
    assert run_a["interpretation"] == (
        "uncalibrated_synthetic_appearance_variation_not_physical_calibration"
    )


def test_run_design_applies_one_palette_and_partition_to_every_vial():
    design = build_run_design(
        run_id="TRAIN-RUN-01",
        run_seed=48001,
        split_role="train",
        previous_recipe={
            "recipe_id": "PREV-A",
            "programmed_ml": {"t100": 0, "t200": 25, "t400": 35},
        },
        recipe_sequence=(
            {
                "recipe_id": "R-A",
                "programmed_ml": {"t100": 0, "t200": 30, "t400": 0},
                "vial_count": 3,
            },
            {
                "recipe_id": "R-B",
                "programmed_ml": {"t100": 20, "t200": 0, "t400": 20},
                "vial_count": 3,
            },
        ),
    )

    assert design["previous_recipe"]["dispense_order"] == ["t200", "t400"]
    assert [row["dispense_order"] for row in design["recipes"]] == [
        ["t200"],
        ["t400", "t100"],
    ]
    assert len(design["cases"]) == 6
    assert {case["run_id"] for case in design["cases"]} == {"TRAIN-RUN-01"}
    assert {case["split_role"] for case in design["cases"]} == {"train"}
    assert {case["palette_sha256"] for case in design["cases"]} == {
        design["palette"]["palette_sha256"]
    }
    assert [case["vial_ordinal_in_recipe"] for case in design["cases"]] == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]


def test_whole_run_validator_rejects_split_or_palette_leakage():
    train = build_run_design(
        run_id="RUN-TRAIN",
        run_seed=49001,
        split_role="train",
        previous_recipe={
            "recipe_id": "P1",
            "programmed_ml": {"t100": 10, "t200": 0, "t400": 20},
        },
        recipe_sequence=(
            {
                "recipe_id": "R1",
                "programmed_ml": {"t100": 10, "t200": 10, "t400": 10},
                "vial_count": 3,
            },
        ),
    )
    evaluation = build_run_design(
        run_id="RUN-EVAL",
        run_seed=49002,
        split_role="evaluation",
        previous_recipe={
            "recipe_id": "P2",
            "programmed_ml": {"t100": 0, "t200": 20, "t400": 10},
        },
        recipe_sequence=(
            {
                "recipe_id": "R2",
                "programmed_ml": {"t100": 20, "t200": 20, "t400": 20},
                "vial_count": 3,
            },
        ),
    )

    validate_whole_run_partitioning([train, evaluation])

    split_leak = deepcopy(train)
    split_leak["cases"][1]["split_role"] = "evaluation"
    with pytest.raises(ValueError, match="single split"):
        validate_whole_run_partitioning([split_leak, evaluation])

    palette_leak = deepcopy(evaluation)
    palette_leak["palette"] = deepcopy(train["palette"])
    for case in palette_leak["cases"]:
        case["palette_sha256"] = train["palette"]["palette_sha256"]
    with pytest.raises(ValueError, match="palette reused across runs"):
        validate_whole_run_partitioning([train, palette_leak])

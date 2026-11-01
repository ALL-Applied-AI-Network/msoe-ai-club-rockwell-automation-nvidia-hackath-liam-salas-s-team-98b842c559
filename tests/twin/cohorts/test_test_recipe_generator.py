"""Properties of the seeded WP1 test-recipe generator."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import cli.twin.cohorts.make_splits as make_splits  # noqa: E402
from twin.cohorts.test_recipe_generator import (  # noqa: E402
    DEFAULT_GENERATOR_SEED,
    build_test_recipe_plan,
    validate_test_recipe_plan,
)


TANKS = ("t100", "t200", "t400")


def _fractions(recipe: dict[str, int]) -> dict[str, float]:
    total = sum(recipe.values())
    return {tank: recipe[tank] / total for tank in TANKS}


def _near(
    left: dict[str, int],
    right: dict[str, int],
    *,
    fraction_limit: float,
) -> bool:
    left_fraction = _fractions(left)
    right_fraction = _fractions(right)
    return all(abs(left[tank] - right[tank]) <= 3 for tank in TANKS) or all(
        abs(left_fraction[tank] - right_fraction[tank]) <= fraction_limit
        for tank in TANKS
    )


def _base_rows() -> list[dict[str, str]]:
    with make_splits.RECIPES_CSV.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_default_plan_is_seeded_new_leakage_safe_and_visibly_separated():
    plan = build_test_recipe_plan(repo_root=ROOT)
    assert plan == build_test_recipe_plan(repo_root=ROOT)
    assert validate_test_recipe_plan(plan, repo_root=ROOT) == plan
    assert plan["artifact_type"] == "wp1-test-recipe-plan-v3"
    assert plan["split_version"] == "inn-7-splits-v3"
    assert plan["parent_split_version"] == "inn-7-splits-v2"
    assert plan["generator_seed"] == DEFAULT_GENERATOR_SEED
    assert [run["run_id"] for run in plan["runs"]] == ["test-run01", "test-run02"]
    assert [run["preset_name"] for run in plan["runs"]] == [
        "amber_teal",
        "crimson_indigo",
    ]

    base = _base_rows()
    base_volumes = {
        tuple(int(row[f"{tank}_ml"]) for tank in TANKS) for row in base
    }
    train_val = [
        {tank: int(row[f"{tank}_ml"]) for tank in TANKS}
        for row in base
        if row["role"] in {"train", "val"}
    ]
    real = list(make_splits.real_recipes().values())
    used_cells = {row["cell"] for row in base if row["role"] != "buffer"}
    test_cells = {row["cell"] for row in base if row["role"] == "test"}

    recipes = [recipe for run in plan["runs"] for recipe in run["recipes"]]
    assert len(recipes) == 6
    assert len({recipe["sim_recipe_id"] for recipe in recipes}) == 6
    assert len({tuple(recipe["programmed_ml"].values()) for recipe in recipes}) == 6
    assert all(recipe["role"] == "test" for recipe in recipes)
    assert any(0 in recipe["programmed_ml"].values() for recipe in recipes)
    for run in plan["runs"]:
        assert [recipe["block"] for recipe in run["recipes"]] == ["A", "B", "C"]
        assert len(run["predicted_consecutive_delta_e76"]) == 2
        assert min(run["predicted_consecutive_delta_e76"]) >= 10.0
        assert any(0 in recipe["programmed_ml"].values() for recipe in run["recipes"])

    for recipe in recipes:
        programmed = recipe["programmed_ml"]
        assert tuple(programmed) == TANKS
        assert tuple(programmed.values()) not in base_volumes
        assert 30 <= sum(programmed.values()) <= 80
        assert programmed["t200"] <= 50
        assert sum(value > 0 for value in programmed.values()) >= 2
        assert all(value == 0 or 10 <= value <= 60 for value in programmed.values())
        assert any(value % 5 for value in programmed.values())
        assert recipe["ratio_cell_origin"] in {"existing_test", "previously_unused"}
        if recipe["cell"] in used_cells:
            assert recipe["cell"] in test_cells
            assert recipe["ratio_cell_origin"] == "existing_test"
        else:
            assert recipe["ratio_cell_origin"] == "previously_unused"
        assert not any(
            _near(programmed, reference, fraction_limit=0.03)
            for reference in train_val
        )
        assert not any(
            _near(programmed, reference, fraction_limit=0.05)
            for reference in real
        )


def test_split_identity_freezes_the_recorded_generator_seed():
    plan = build_test_recipe_plan(repo_root=ROOT, seed=DEFAULT_GENERATOR_SEED)
    assert plan["generator_seed"] == DEFAULT_GENERATOR_SEED
    with pytest.raises(ValueError, match="frozen"):
        build_test_recipe_plan(repo_root=ROOT, seed=DEFAULT_GENERATOR_SEED + 1)


def test_generator_rejects_frozen_parent_source_drift(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    for relative in (
        "data/sim_recipes.csv",
        "data/sim_transitions.csv",
        "data/sim_split_spec.json",
        "data/video5_vial_recipes.csv",
        "configs/tank_colour_presets.json",
    ):
        source = ROOT / relative
        destination = tmp_path / relative
        shutil.copy2(source, destination)
    with (tmp_path / "data/sim_recipes.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="source hash drift"):
        build_test_recipe_plan(repo_root=tmp_path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update(split_version="inn-7-splits-v2"), "split"),
        (
            lambda value: value["runs"][1]["recipes"][0].update(
                sim_recipe_id=value["runs"][0]["recipes"][0]["sim_recipe_id"]
            ),
            "unique",
        ),
        (
            lambda value: value["runs"][0]["recipes"][0]["programmed_ml"].update(
                t100=5
            ),
            "volume|recipe|tank",
        ),
        (
            lambda value: value["runs"][0]["recipes"][0].update(role="train"),
            "test",
        ),
        (
            lambda value: value["runs"][0].update(
                predicted_consecutive_delta_e76=[999.0, 999.0]
            ),
            "Delta|prediction|visible",
        ),
    ],
)
def test_validator_rejects_identity_rule_and_prediction_drift(mutator, match):
    plan = build_test_recipe_plan(repo_root=ROOT)
    changed = deepcopy(plan)
    mutator(changed)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_test_recipe_plan(changed, repo_root=ROOT)

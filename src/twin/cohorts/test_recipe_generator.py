"""Seeded generator for the additive WP1 test-recipe split extension.

The accepted v2 files remain byte-for-byte inputs to older artifacts.  This module
derives a source-bound v3 extension without rewriting any v2 role.  Predicted colour
distance is an uncalibrated synthetic planning aid; rendered stills remain the gate.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import math
from numbers import Real
from pathlib import Path
import random
from functools import lru_cache
from collections.abc import Mapping, Sequence

import numpy as np

from twin.colour import absorbance, expected_color, linear_to_srgb, srgb_to_lab
from ..scene.tank_colour_presets import (
    load_tank_colour_preset_catalog,
    resolve_tank_colour_preset,
)


DEFAULT_GENERATOR_SEED = 20261028
ARTIFACT_TYPE = "wp1-test-recipe-plan-v3"
SPLIT_VERSION = "inn-7-splits-v3"
PARENT_SPLIT_VERSION = "inn-7-splits-v2"
TANKS = ("t100", "t200", "t400")
RUN_PRESETS = ("amber_teal", "crimson_indigo")
MIN_TOTAL_ML, MAX_TOTAL_ML, MAX_T200_ML = 30, 80, 50
MIN_USED_TANK_ML, MAX_TANK_ML = 10, 60
VOLUME_NEAR_ML = 3
TRAIN_VAL_NEAR_FRACTION = 0.03
VIDEO5_NEAR_FRACTION = 0.05
CELL_WIDTH = 0.1
MIN_PLANNED_CONSECUTIVE_DELTA_E76 = 10.0
_EXPECTED_SOURCE_SHA256 = {
    "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
    "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
    "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
    "data/video5_vial_recipes.csv": "4baf9246f8b34b803f8e3de4ef9915375311be83f5de91edf8cb249a78254f6d",
    "configs/tank_colour_presets.json": "bd22f64c08a8942e9a1ff85ba3571d9f4b57c24ef19b62472099251bfa5c8ffb",
}
_ROOT_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "split_version", "parent_split_version",
        "generator_seed", "source_sha256", "rules", "selection", "runs",
    }
)
_RUN_KEYS = frozenset(
    {"run_id", "preset_name", "recipes", "predicted_consecutive_delta_e76"}
)
_RECIPE_KEYS = frozenset(
    {
        "sim_recipe_id", "role", "block", "cell", "ratio_cell_origin",
        "programmed_ml", "total_ml",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path.name} is missing required columns")
            return list(reader)
    except OSError as error:
        raise ValueError(f"required generator source is unavailable: {path}") from error


def _volumes(row: Mapping[str, object]) -> dict[str, int]:
    result = {}
    for tank in TANKS:
        try:
            result[tank] = int(row[f"{tank}_ml"] or 0)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {tank}_ml in recipe source") from error
    return result


def _fractions(recipe: Mapping[str, int]) -> dict[str, float]:
    total = sum(recipe.values())
    return {tank: recipe[tank] / total for tank in TANKS}


def _cell(recipe: Mapping[str, int]) -> str:
    fractions = _fractions(recipe)
    return "-".join(
        f"{round(fractions[tank] / CELL_WIDTH) * CELL_WIDTH:.1f}" for tank in TANKS
    )


def _near(
    left: Mapping[str, int], right: Mapping[str, int], *, fraction_limit: float
) -> bool:
    if all(abs(left[tank] - right[tank]) <= VOLUME_NEAR_ML for tank in TANKS):
        return True
    left_fraction, right_fraction = _fractions(left), _fractions(right)
    return all(
        abs(left_fraction[tank] - right_fraction[tank]) <= fraction_limit
        for tank in TANKS
    )


def _candidate_recipes(
    base_rows: Sequence[Mapping[str, object]],
    real_recipes: Sequence[Mapping[str, int]],
) -> list[dict[str, object]]:
    used_cells = {str(row["cell"]) for row in base_rows if row["role"] != "buffer"}
    test_cells = {str(row["cell"]) for row in base_rows if row["role"] == "test"}
    train_val = [_volumes(row) for row in base_rows if row["role"] in {"train", "val"}]
    base_volumes = {tuple(_volumes(row).values()) for row in base_rows}
    values = (0, *range(MIN_USED_TANK_ML, MAX_TANK_ML + 1))
    candidates = []
    for t100 in values:
        for t200 in values:
            if t200 > MAX_T200_ML:
                continue
            for t400 in values:
                recipe = {"t100": t100, "t200": t200, "t400": t400}
                total, key = sum(recipe.values()), tuple(recipe.values())
                if not MIN_TOTAL_ML <= total <= MAX_TOTAL_ML:
                    continue
                if sum(value > 0 for value in recipe.values()) < 2:
                    continue
                if key in base_volumes or all(value % 5 == 0 for value in key):
                    continue
                ratio_cell = _cell(recipe)
                if ratio_cell in used_cells and ratio_cell not in test_cells:
                    continue
                if any(
                    _near(recipe, reference, fraction_limit=TRAIN_VAL_NEAR_FRACTION)
                    for reference in train_val
                ):
                    continue
                if any(
                    _near(recipe, reference, fraction_limit=VIDEO5_NEAR_FRACTION)
                    for reference in real_recipes
                ):
                    continue
                candidates.append(
                    {
                        "programmed_ml": recipe,
                        "cell": ratio_cell,
                        "ratio_cell_origin": (
                            "existing_test" if ratio_cell in test_cells else "previously_unused"
                        ),
                    }
                )
    return candidates


def _candidate_labs(
    candidates: Sequence[Mapping[str, object]], repo_root: Path, preset_name: str
) -> np.ndarray:
    catalog = load_tank_colour_preset_catalog(
        repo_root / "configs" / "tank_colour_presets.json"
    )
    transmission = resolve_tank_colour_preset(
        catalog, preset_name
    ).to_json_record()["transmission_1cm"]
    tank_absorbance = {
        tank: absorbance(np.asarray(transmission[tank]), np.ones(3)) for tank in TANKS
    }
    colours = np.stack(
        [
            linear_to_srgb(
                expected_color(
                    dict(candidate["programmed_ml"]), tank_absorbance, np.ones(3)
                )
            )
            for candidate in candidates
        ]
    )
    return srgb_to_lab(colours)


def _select_run(
    candidates: Sequence[Mapping[str, object]],
    labs: np.ndarray,
    *,
    rng: random.Random,
    unavailable_volumes: set[tuple[int, int, int]],
    unavailable_cells: set[str],
) -> tuple[list[int], list[float]]:
    eligible = [
        index
        for index, candidate in enumerate(candidates)
        if tuple(candidate["programmed_ml"].values()) not in unavailable_volumes
        and candidate["cell"] not in unavailable_cells
    ]
    rng.shuffle(eligible)
    if len(eligible) < 3:
        raise ValueError("not enough eligible recipes for a distinct test run")
    rank = {index: position for position, index in enumerate(eligible)}
    best: tuple[float, int, int, int, int] | None = None
    for first in eligible[: min(64, len(eligible))]:
        second = max(
            (
                index for index in eligible
                if index != first and candidates[index]["cell"] != candidates[first]["cell"]
            ),
            key=lambda index: (
                float(np.linalg.norm(labs[first] - labs[index])), -rank[index]
            ),
        )
        skips = (
            0 in candidates[first]["programmed_ml"].values()
            or 0 in candidates[second]["programmed_ml"].values()
        )
        middle_options = [
            index for index in eligible
            if index not in {first, second}
            and candidates[index]["cell"]
            not in {candidates[first]["cell"], candidates[second]["cell"]}
            and (skips or 0 in candidates[index]["programmed_ml"].values())
        ]
        if not middle_options:
            continue
        middle = max(
            middle_options,
            key=lambda index: (
                min(
                    float(np.linalg.norm(labs[first] - labs[index])),
                    float(np.linalg.norm(labs[index] - labs[second])),
                ),
                -rank[index],
            ),
        )
        score = min(
            float(np.linalg.norm(labs[first] - labs[middle])),
            float(np.linalg.norm(labs[middle] - labs[second])),
        )
        proposal = (score, -rank[first], first, middle, second)
        if best is None or proposal > best:
            best = proposal
    if best is None:
        raise ValueError("no visibly distinct three-recipe run could be selected")
    selected = [best[2], best[3], best[4]]
    distances = [
        round(float(np.linalg.norm(labs[left] - labs[right])), 6)
        for left, right in zip(selected, selected[1:])
    ]
    if min(distances) < MIN_PLANNED_CONSECUTIVE_DELTA_E76:
        raise ValueError("selected recipes do not meet the planned visibility separation")
    return selected, distances


def _recipe_id(ordinal: int) -> str:
    return f"S{ordinal:04d}"


def _source_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "data/sim_recipes.csv": repo_root / "data" / "sim_recipes.csv",
        "data/sim_transitions.csv": repo_root / "data" / "sim_transitions.csv",
        "data/sim_split_spec.json": repo_root / "data" / "sim_split_spec.json",
        "data/video5_vial_recipes.csv": repo_root / "data" / "video5_vial_recipes.csv",
        "configs/tank_colour_presets.json": repo_root / "configs" / "tank_colour_presets.json",
    }


def _build_unvalidated(*, repo_root: Path, seed: int) -> dict[str, object]:
    if seed != DEFAULT_GENERATOR_SEED:
        raise ValueError("WP1 v3 generator seed is frozen")
    root = repo_root.resolve()
    fingerprints = tuple(
        (name, _sha256(path)) for name, path in _source_paths(root).items()
    )
    if dict(fingerprints) != _EXPECTED_SOURCE_SHA256:
        raise ValueError("test recipe generator source hash drift")
    return deepcopy(_build_cached(str(root), seed, fingerprints))


@lru_cache(maxsize=32)
def _build_cached(
    repo_root: str,
    seed: int,
    fingerprints: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    # ``fingerprints`` is intentionally part of the cache key.  It makes repeated
    # validation cheap without masking a changed source at the same path.
    del fingerprints
    return _build_fresh(repo_root=Path(repo_root), seed=seed)


def _build_fresh(*, repo_root: Path, seed: int) -> dict[str, object]:
    paths = _source_paths(repo_root)
    base_rows = _load_csv(
        paths["data/sim_recipes.csv"],
        {"sim_recipe_id", "role", "cell", *(f"{tank}_ml" for tank in TANKS)},
    )
    real_rows = _load_csv(
        paths["data/video5_vial_recipes.csv"],
        {"recipe_id", *(f"{tank}_ml" for tank in TANKS)},
    )
    candidates = _candidate_recipes(base_rows, [_volumes(row) for row in real_rows])
    rng = random.Random(seed)
    unavailable_volumes: set[tuple[int, int, int]] = set()
    unavailable_cells: set[str] = set()
    runs = []
    for run_ordinal, preset_name in enumerate(RUN_PRESETS, start=1):
        labs = _candidate_labs(candidates, repo_root, preset_name)
        selected, distances = _select_run(
            candidates,
            labs,
            rng=rng,
            unavailable_volumes=unavailable_volumes,
            unavailable_cells=unavailable_cells,
        )
        recipes = []
        for block_ordinal, (block, index) in enumerate(
            zip(("A", "B", "C"), selected, strict=True), start=1
        ):
            candidate = candidates[index]
            programmed_ml = dict(candidate["programmed_ml"])
            volume_key = tuple(programmed_ml.values())
            unavailable_volumes.add(volume_key)
            unavailable_cells.add(str(candidate["cell"]))
            recipes.append(
                {
                    "sim_recipe_id": _recipe_id(
                        len(base_rows) + (run_ordinal - 1) * 3 + block_ordinal
                    ),
                    "role": "test",
                    "block": block,
                    "cell": candidate["cell"],
                    "ratio_cell_origin": candidate["ratio_cell_origin"],
                    "programmed_ml": programmed_ml,
                    "total_ml": sum(programmed_ml.values()),
                }
            )
        runs.append(
            {
                "run_id": f"test-run{run_ordinal:02d}",
                "preset_name": preset_name,
                "recipes": recipes,
                "predicted_consecutive_delta_e76": distances,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "split_version": SPLIT_VERSION,
        "parent_split_version": PARENT_SPLIT_VERSION,
        "generator_seed": seed,
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "rules": {
            "grid_step_ml": 1,
            "tank_volume_ml": {"unused": 0, "minimum_when_used": 10, "maximum": 60},
            "total_ml": {"minimum": 30, "maximum": 80},
            "t200_maximum_ml": 50,
            "minimum_tanks_used": 2,
            "train_val_separation": {
                "fraction_per_tank": TRAIN_VAL_NEAR_FRACTION,
                "ml_per_tank": VOLUME_NEAR_ML,
            },
            "video5_separation": {
                "fraction_per_tank": VIDEO5_NEAR_FRACTION,
                "ml_per_tank": VOLUME_NEAR_ML,
            },
            "allowed_ratio_cell_origin": ["existing_test", "previously_unused"],
        },
        "selection": {
            "candidate_count": len(candidates),
            "run_count": len(RUN_PRESETS),
            "recipes_per_run": 3,
            "minimum_planned_consecutive_delta_e76": MIN_PLANNED_CONSECUTIVE_DELTA_E76,
            "colour_model": "uncalibrated_synthetic_preset_beer_lambert_planning_only",
            "render_visibility_gate_required": True,
        },
        "runs": runs,
    }


def build_test_recipe_plan(
    *, repo_root: str | Path, seed: int = DEFAULT_GENERATOR_SEED
) -> dict[str, object]:
    """Build and validate the deterministic minimum two-run WP1 recipe plan."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("generator seed must be a non-Boolean integer")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("generator seed must be in [0, 2**63 - 1]")
    if seed != DEFAULT_GENERATOR_SEED:
        raise ValueError("WP1 v3 generator seed is frozen")
    root = Path(repo_root)
    return validate_test_recipe_plan(
        _build_unvalidated(repo_root=root, seed=seed), repo_root=root
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_test_recipe_plan(
    plan: Mapping[str, object], *, repo_root: str | Path
) -> dict[str, object]:
    """Fail closed on recipe, split, source, seed, or visibility drift."""

    if not isinstance(plan, Mapping) or set(plan) != _ROOT_KEYS:
        raise ValueError("test recipe plan has unexpected fields")
    if plan["schema_version"] != 1 or plan["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("test recipe plan schema is invalid")
    if (
        plan["split_version"] != SPLIT_VERSION
        or plan["parent_split_version"] != PARENT_SPLIT_VERSION
    ):
        raise ValueError("test recipe plan split version is invalid")
    seed = plan["generator_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("generator seed must be a non-Boolean integer")
    if seed != DEFAULT_GENERATOR_SEED:
        raise ValueError("WP1 v3 generator seed is frozen")
    runs = plan["runs"]
    if not isinstance(runs, list) or len(runs) != len(RUN_PRESETS):
        raise ValueError("test recipe plan must contain exactly two runs")
    identifiers: list[str] = []
    volume_keys: list[tuple[int, int, int]] = []
    for run_ordinal, (run, preset_name) in enumerate(
        zip(runs, RUN_PRESETS, strict=True), start=1
    ):
        if not isinstance(run, Mapping) or set(run) != _RUN_KEYS:
            raise ValueError("test recipe run has unexpected fields")
        if run["run_id"] != f"test-run{run_ordinal:02d}":
            raise ValueError("test recipe run identity is invalid")
        if run["preset_name"] != preset_name:
            raise ValueError("test recipe run preset is invalid")
        recipes = run["recipes"]
        if not isinstance(recipes, list) or len(recipes) != 3:
            raise ValueError("each test recipe run must contain A, B, and C")
        if [row.get("block") for row in recipes if isinstance(row, Mapping)] != [
            "A", "B", "C"
        ]:
            raise ValueError("test recipe blocks must be A, B, and C")
        if not any(
            isinstance(row, Mapping)
            and isinstance(row.get("programmed_ml"), Mapping)
            and 0 in row["programmed_ml"].values()
            for row in recipes
        ):
            raise ValueError("each run must contain a recipe that skips a tank")
        for recipe in recipes:
            if not isinstance(recipe, Mapping) or set(recipe) != _RECIPE_KEYS:
                raise ValueError("test recipe has unexpected fields")
            if recipe["role"] != "test":
                raise ValueError("generated recipe role must be test")
            identifier = recipe["sim_recipe_id"]
            if not isinstance(identifier, str) or not identifier.startswith("S"):
                raise ValueError("generated recipe identity is invalid")
            identifiers.append(identifier)
            programmed = recipe["programmed_ml"]
            if not isinstance(programmed, Mapping) or tuple(programmed) != TANKS:
                raise ValueError("generated recipe must contain exactly the three tanks")
            values = []
            for tank in TANKS:
                value = programmed[tank]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError("generated recipe tank volume must be an integer")
                if value != 0 and not MIN_USED_TANK_ML <= value <= MAX_TANK_ML:
                    raise ValueError("generated recipe tank volume is outside plant limits")
                values.append(value)
            if not MIN_TOTAL_ML <= sum(values) <= MAX_TOTAL_ML:
                raise ValueError("generated recipe total volume is outside plant limits")
            if values[1] > MAX_T200_ML or sum(value > 0 for value in values) < 2:
                raise ValueError("generated recipe violates tank-use plant limits")
            if all(value % 5 == 0 for value in values):
                raise ValueError("generated recipe must use the finer one-mL grid")
            if recipe["total_ml"] != sum(values):
                raise ValueError("generated recipe total volume has drifted")
            normalized = dict(zip(TANKS, values, strict=True))
            if recipe["cell"] != _cell(normalized):
                raise ValueError("generated recipe ratio cell has drifted")
            if recipe["ratio_cell_origin"] not in {
                "existing_test", "previously_unused"
            }:
                raise ValueError("generated recipe ratio-cell origin is invalid")
            volume_keys.append(tuple(values))
        distances = run["predicted_consecutive_delta_e76"]
        if not isinstance(distances, list) or len(distances) != 2:
            raise ValueError("visible-colour prediction must have two switch distances")
        if any(
            _finite(value, "predicted Delta E76")
            < MIN_PLANNED_CONSECUTIVE_DELTA_E76
            for value in distances
        ):
            raise ValueError("predicted Delta E76 does not meet the visible switch gate")
    if len(set(identifiers)) != len(identifiers) or len(set(volume_keys)) != len(
        volume_keys
    ):
        raise ValueError("generated recipe identities and volumes must be unique")
    expected = _build_unvalidated(repo_root=Path(repo_root), seed=seed)
    if dict(plan) != expected:
        raise ValueError("test recipe plan differs from the seeded source-bound prediction")
    return deepcopy(expected)

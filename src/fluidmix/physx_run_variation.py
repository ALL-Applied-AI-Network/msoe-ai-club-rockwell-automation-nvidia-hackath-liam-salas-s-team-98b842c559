"""Synthetic run-level recipe and appearance variation contracts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real


BASE_TANK_TRANSMISSION_1CM = {
    "t200": (0.95, 0.39, 0.26),
    "t400": (0.14, 0.61, 0.95),
    "t100": (1.0, 1.0, 1.0),
}
DISPENSE_ORDER = ("t200", "t400", "t100")
_TANK_KEYS = frozenset(DISPENSE_ORDER)


def _normalized_volumes(programmed_ml: Mapping[str, object]) -> dict[str, float | int]:
    if not isinstance(programmed_ml, Mapping) or set(programmed_ml) != _TANK_KEYS:
        raise ValueError("programmed_ml must contain exactly t100, t200, and t400")
    normalized: dict[str, float | int] = {}
    for tank in DISPENSE_ORDER:
        value = programmed_ml[tank]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"programmed_ml[{tank}] must be a real number")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"programmed_ml[{tank}] must be finite and non-negative")
        normalized[tank] = value
    if not any(value > 0 for value in normalized.values()):
        raise ValueError("at least one tank must have positive programmed volume")
    return normalized


def active_dispense_order(programmed_ml: Mapping[str, object]) -> tuple[str, ...]:
    """Return the fixed plant order with zero-volume tanks omitted."""

    normalized = _normalized_volumes(programmed_ml)
    return tuple(tank for tank in DISPENSE_ORDER if normalized[tank] > 0)


def _unit_interval(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def build_run_palette(*, run_id: str, run_seed: int) -> dict[str, object]:
    """Build one deterministic, uncalibrated 1-cm tank palette for a run.

    T200/T400 variation is applied in extinction space.  A shared per-tank
    factor varies dye strength and small per-channel factors vary hue.  T100
    remains exactly clear and therefore only dilutes a mixture.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError("run_seed must be an integer")

    transmission: dict[str, list[float]] = {}
    variation: dict[str, dict[str, object]] = {}
    for tank in ("t200", "t400"):
        strength = 0.85 + 0.30 * _unit_interval(
            "fluidmix-run-palette-v1", run_id, run_seed, tank, "strength"
        )
        channel_scales = tuple(
            0.95
            + 0.10
            * _unit_interval(
                "fluidmix-run-palette-v1",
                run_id,
                run_seed,
                tank,
                "channel",
                channel,
            )
            for channel in range(3)
        )
        varied = []
        for base, channel_scale in zip(
            BASE_TANK_TRANSMISSION_1CM[tank], channel_scales, strict=True
        ):
            extinction = -math.log(base)
            varied.append(round(math.exp(-extinction * strength * channel_scale), 9))
        transmission[tank] = varied
        variation[tank] = {
            "strength_scale": round(strength, 9),
            "channel_extinction_scales": [
                round(scale, 9) for scale in channel_scales
            ],
        }
    transmission["t100"] = [1.0, 1.0, 1.0]
    palette_sha256 = _canonical_sha256(transmission)
    return {
        "schema_version": "fluidmix-run-palette-v1",
        "run_id": run_id,
        "run_seed": run_seed,
        "reference_depth_m": 0.01,
        "transmission_1cm": transmission,
        "variation_by_tank": variation,
        "palette_sha256": palette_sha256,
        "interpretation": (
            "uncalibrated_synthetic_appearance_variation_not_physical_calibration"
        ),
    }


def build_run_design(
    *,
    run_id: str,
    run_seed: int,
    split_role: str,
    previous_recipe: Mapping[str, object],
    recipe_sequence: tuple[Mapping[str, object], ...]
    | list[Mapping[str, object]],
) -> dict[str, object]:
    """Expand a run-level palette and recipe sequence into vial case metadata."""

    if not isinstance(split_role, str) or not split_role.strip():
        raise ValueError("split_role must be a non-empty string")
    if not isinstance(previous_recipe, Mapping):
        raise TypeError("previous_recipe must be a mapping")
    if set(previous_recipe) != {"recipe_id", "programmed_ml"}:
        raise ValueError(
            "previous_recipe must contain exactly recipe_id and programmed_ml"
        )
    previous = _normalize_recipe(previous_recipe, include_vial_count=False)
    if isinstance(recipe_sequence, (str, bytes)) or not isinstance(
        recipe_sequence, (tuple, list)
    ):
        raise TypeError("recipe_sequence must be a tuple or list")
    if not recipe_sequence:
        raise ValueError("recipe_sequence must not be empty")
    recipes = [
        _normalize_recipe(recipe, include_vial_count=True)
        for recipe in recipe_sequence
    ]
    recipe_ids = [str(recipe["recipe_id"]) for recipe in recipes]
    if len(recipe_ids) != len(set(recipe_ids)):
        raise ValueError("recipe_id must be unique within a run")

    palette = build_run_palette(run_id=run_id, run_seed=run_seed)
    cases: list[dict[str, object]] = []
    predecessor = previous
    run_vial_ordinal = 0
    for recipe in recipes:
        for vial_ordinal in range(1, int(recipe["vial_count"]) + 1):
            run_vial_ordinal += 1
            cases.append(
                {
                    "case_id": (
                        f"{run_id}:{recipe['recipe_id']}:V{vial_ordinal:02d}"
                    ),
                    "run_id": run_id,
                    "split_role": split_role,
                    "palette_sha256": palette["palette_sha256"],
                    "recipe_id": recipe["recipe_id"],
                    "previous_recipe_id": predecessor["recipe_id"],
                    "vial_ordinal_in_recipe": vial_ordinal,
                    "vial_ordinal_in_run": run_vial_ordinal,
                    "programmed_ml": dict(recipe["programmed_ml"]),
                    "dispense_order": list(recipe["dispense_order"]),
                }
            )
        predecessor = recipe
    return {
        "schema_version": "fluidmix-run-design-v1",
        "run_id": run_id,
        "run_seed": run_seed,
        "split_role": split_role,
        "palette": palette,
        "previous_recipe": previous,
        "recipes": recipes,
        "cases": cases,
    }


def _normalize_recipe(
    recipe: Mapping[str, object], *, include_vial_count: bool
) -> dict[str, object]:
    if not isinstance(recipe, Mapping):
        raise TypeError("recipe must be a mapping")
    expected = {"recipe_id", "programmed_ml"}
    if include_vial_count:
        expected.add("vial_count")
    if set(recipe) != expected:
        raise ValueError(f"recipe must contain exactly {sorted(expected)}")
    recipe_id = recipe["recipe_id"]
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise ValueError("recipe_id must be a non-empty string")
    volumes = _normalized_volumes(recipe["programmed_ml"])
    normalized: dict[str, object] = {
        "recipe_id": recipe_id,
        "programmed_ml": {
            tank: volumes[tank] for tank in ("t100", "t200", "t400")
        },
        "dispense_order": list(active_dispense_order(volumes)),
    }
    if include_vial_count:
        vial_count = recipe["vial_count"]
        if (
            isinstance(vial_count, bool)
            or not isinstance(vial_count, int)
            or vial_count <= 0
        ):
            raise ValueError("vial_count must be a positive integer")
        normalized["vial_count"] = vial_count
    return normalized


def validate_whole_run_partitioning(
    run_designs: tuple[Mapping[str, object], ...]
    | list[Mapping[str, object]],
) -> None:
    """Reject case-level split drift and palette reuse between distinct runs."""

    if isinstance(run_designs, (str, bytes)) or not isinstance(
        run_designs, (tuple, list)
    ):
        raise TypeError("run_designs must be a tuple or list")
    if not run_designs:
        raise ValueError("run_designs must not be empty")
    run_ids: set[str] = set()
    palette_owner: dict[str, str] = {}
    for design in run_designs:
        if not isinstance(design, Mapping):
            raise TypeError("each run design must be a mapping")
        run_id = design.get("run_id")
        split_role = design.get("split_role")
        palette = design.get("palette")
        cases = design.get("cases")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run design run_id is invalid")
        if run_id in run_ids:
            raise ValueError("run_id must be unique in a cohort")
        run_ids.add(run_id)
        if not isinstance(split_role, str) or not split_role:
            raise ValueError("run design split_role is invalid")
        if not isinstance(palette, Mapping):
            raise ValueError("run design palette is invalid")
        palette_hash = palette.get("palette_sha256")
        if not isinstance(palette_hash, str) or len(palette_hash) != 64:
            raise ValueError("run design palette_sha256 is invalid")
        previous_owner = palette_owner.get(palette_hash)
        if previous_owner is not None and previous_owner != run_id:
            raise ValueError("palette reused across runs")
        palette_owner[palette_hash] = run_id
        if not isinstance(cases, list) or not cases:
            raise ValueError("run design cases are invalid")
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("run design case is invalid")
            if case.get("run_id") != run_id:
                raise ValueError("every case must remain in a single run")
            if case.get("split_role") != split_role:
                raise ValueError("every run must remain in a single split")
            if case.get("palette_sha256") != palette_hash:
                raise ValueError("every case in a run must use one palette")

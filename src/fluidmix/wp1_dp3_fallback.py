"""Deterministic source-only plans for the frozen WP1 DP3 fallback cases.

This module creates render inputs only.  It does not launch Isaac, authorize compute,
accept a render, or choose the unresolved cross-case visibility aggregation rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .physx_one_run_manifest_v1 import (
    build_one_run_source_artifacts,
    rebuild_bound_one_run_artifacts,
)
from .physx_run_variation import bind_tank_colour_resolution_to_artifacts


TARGETS = {
    5: {
        "recipe_id": "S0440",
        "original_case_id": "CASE_d52026e7df33b0cb",
        "render_seed": 2_611_005,
        "nuisance_group_id": "WP1-DP3-FALLBACK-V05",
    },
    9: {
        "recipe_id": "S0441",
        "original_case_id": "CASE_76b13e1cb1e631c1",
        "render_seed": 2_611_009,
        "nuisance_group_id": "WP1-DP3-FALLBACK-V09",
    },
}
FILENAMES = (
    "source-run-spec.json",
    "render-schedule-plan.json",
    "model-input-plan.json",
    "label-provenance.json",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _target_schedule(
    plan: Mapping[str, object], target_ordinal: int
) -> Mapping[str, object]:
    rebuilt = rebuild_bound_one_run_artifacts(plan)
    if rebuilt["render-schedule-plan.json"] != plan:
        raise ValueError("frozen run-1 plan differs from its source-bound rebuild")
    schedules = plan.get("execution_schedules")
    matches = [
        row
        for row in schedules
        if isinstance(row, Mapping)
        and isinstance(row.get("case_binding"), Mapping)
        and row["case_binding"].get("vial_ordinal") == target_ordinal
    ] if isinstance(schedules, list) else []
    if len(matches) != 1:
        raise ValueError("frozen fallback target ordinal is not unique")
    expected = TARGETS[target_ordinal]
    binding = matches[0]["case_binding"]
    if (
        binding.get("case_id") != expected["original_case_id"]
        or binding.get("current_recipe_id") != expected["recipe_id"]
        or binding.get("residual_volume_ml") != 1.5
    ):
        raise ValueError("frozen fallback target identity differs")
    return matches[0]


def build_dp3_fallback_plan_artifacts(
    frozen_run1_plan: Mapping[str, object], *, target_ordinal: int
) -> dict[str, dict[str, object]]:
    """Build one real source-bound one-vial plan for ordinal 5 or 9."""

    if isinstance(target_ordinal, bool) or target_ordinal not in TARGETS:
        raise ValueError("target_ordinal must be exactly 5 or 9")
    original = _target_schedule(frozen_run1_plan, target_ordinal)
    expected = TARGETS[target_ordinal]
    model_input = original.get("model_input_record")
    source_spec = frozen_run1_plan.get("source_run_spec")
    palette = frozen_run1_plan.get("tank_colour_resolution")
    if (
        not isinstance(model_input, Mapping)
        or not isinstance(source_spec, Mapping)
        or not isinstance(palette, Mapping)
        or model_input.get("current_recipe_id") != expected["recipe_id"]
        or model_input.get("previous_recipe_id") != expected["recipe_id"]
        or model_input.get("programmed_ml")
        != model_input.get("previous_programmed_ml")
    ):
        raise ValueError("frozen same-recipe fallback context is invalid")
    recipe = {
        "recipe_id": expected["recipe_id"],
        "programmed_ml": deepcopy(model_input["programmed_ml"]),
    }
    fallback_spec = {
        "schema_version": 1,
        "artifact_type": "fluidmix-one-run-source-spec-v1",
        "run_id": f"WP1-DP3-FALLBACK-ORDINAL-{target_ordinal:02d}",
        "split_role": "test",
        "split_version": source_spec["split_version"],
        "source_sha256": deepcopy(source_spec["source_sha256"]),
        "recipes": [recipe],
        "initial_condition": "recipe_predecessor",
        "initial_predecessor_recipe_id": expected["recipe_id"],
        "vial_recipe_ids": [expected["recipe_id"]],
        "residual_volume_ml_by_vial": [3.0],
        "render_seed_by_vial": [expected["render_seed"]],
        "nuisance_group_id_by_vial": [expected["nuisance_group_id"]],
    }
    artifacts = bind_tank_colour_resolution_to_artifacts(
        build_one_run_source_artifacts(fallback_spec), palette
    )
    if rebuild_bound_one_run_artifacts(
        artifacts["render-schedule-plan.json"]
    ) != artifacts:
        raise ValueError("fallback plan does not rebuild through the real one-run contract")
    return {"source-run-spec.json": fallback_spec, **artifacts}


def load_dp3_fallback_plan_bundle(
    root: str | Path,
    *,
    target_ordinal: int,
    frozen_run1_plan: Mapping[str, object],
) -> dict[str, Any]:
    """Load one bundle and require exact equality to frozen-run-1 derivation."""

    lexical_root = Path(root)
    if lexical_root.is_symlink():
        raise ValueError("fallback plan root is missing or a symlink")
    plan_root = lexical_root.resolve()
    if not plan_root.is_dir():
        raise ValueError("fallback plan root is missing or a symlink")
    values: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for name in FILENAMES:
        path = (plan_root / name).resolve()
        try:
            path.relative_to(plan_root)
        except ValueError as error:
            raise ValueError("fallback plan path escapes its root") from error
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"fallback plan artifact is missing: {name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"fallback plan artifact is not an object: {name}")
        paths[name] = path
        values[name] = value
    expected_values = build_dp3_fallback_plan_artifacts(
        frozen_run1_plan, target_ordinal=target_ordinal
    )
    for name in FILENAMES:
        if values[name] != expected_values[name]:
            raise ValueError(
                f"fallback artifact differs from exact frozen run-1 derivation: {name}"
            )
    plan = values["render-schedule-plan.json"]
    rebuilt = rebuild_bound_one_run_artifacts(plan)
    if values["source-run-spec.json"] != plan.get("source_run_spec"):
        raise ValueError("copied fallback source spec differs from its render plan")
    for name, value in rebuilt.items():
        if values[name] != value:
            raise ValueError(f"copied fallback artifact differs from rebuild: {name}")
    target = TARGETS[target_ordinal]
    source = values["source-run-spec.json"]
    schedule = plan["execution_schedules"][0]
    model_row = values["model-input-plan.json"]["vial_records"][0]
    label = values["label-provenance.json"]["vial_labels"][0]
    batch = plan["batches"][0]
    binding = schedule["case_binding"]
    if (
        plan.get("batch_size_vials") != 1
        or source.get("initial_condition") != "recipe_predecessor"
        or source.get("initial_predecessor_recipe_id") != target["recipe_id"]
        or source.get("vial_recipe_ids") != [target["recipe_id"]]
        or source.get("residual_volume_ml_by_vial") != [3.0]
        or source.get("render_seed_by_vial") != [target["render_seed"]]
        or source.get("nuisance_group_id_by_vial")
        != [target["nuisance_group_id"]]
        or binding.get("vial_ordinal") != 1
        or binding.get("current_recipe_id") != target["recipe_id"]
        or binding.get("previous_recipe_id") != target["recipe_id"]
        or binding.get("residual_volume_ml") != 3.0
        or model_row.get("current_recipe_id") != target["recipe_id"]
        or model_row.get("previous_recipe_id") != target["recipe_id"]
        or label.get("prescribed_leading_residual_volume_ml") != 3.0
        or label.get("render_schedule_sha256") != canonical_sha256(schedule)
        or batch.get("render_seed_by_vial") != [target["render_seed"]]
        or batch.get("nuisance_group_ids") != [target["nuisance_group_id"]]
    ):
        raise ValueError("fallback plan recipe/context/residue/seed/nuisance binding differs")
    return {
        "root": plan_root,
        "paths": paths,
        "values": values,
        "schedule": schedule,
        "model_row": model_row,
        "label": label,
        "case_id": binding["case_id"],
        "render_seed": target["render_seed"],
        "nuisance_group_id": target["nuisance_group_id"],
    }


__all__ = [
    "FILENAMES",
    "TARGETS",
    "build_dp3_fallback_plan_artifacts",
    "canonical_sha256",
    "load_dp3_fallback_plan_bundle",
]

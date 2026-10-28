"""Pure dynamic one-run source contract for source-bound PhysX renders.

The embedded source specification is the complete deterministic input to this
builder.  It is synthetic planning provenance, not render authorization or a
claim that the prescribed residue represents a real line condition.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real

from .physx_dispense_residual import (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    build_runtime_leading_residual_schedule,
)
from .physx_particle_contract import (
    LEGACY_RESOLUTION_PROFILE_ID,
    particle_resolution_profile,
)
from .physx_run_variation import (
    active_dispense_order,
    bind_tank_colour_resolution_to_artifacts,
    validate_tank_colour_plan_overlay,
)


SOURCE_SPEC_ARTIFACT_TYPE = "fluidmix-one-run-source-spec-v1"
RENDER_PLAN_ARTIFACT_TYPE = "physx_f2_source_bound_one_run_render_plan_v1"
MODEL_INPUT_ARTIFACT_TYPE = "synthetic_f2_source_bound_one_run_input_plan_v1"
LABEL_PROVENANCE_ARTIFACT_TYPE = (
    "synthetic_f2_source_bound_one_run_label_provenance_v1"
)
RUNTIME_CONTRACT_KIND = "source_bound_one_run_v1"
PHYSICS_HZ = 120
CAPTURE_CADENCE_STEPS = 12
TIMING_INTERCEPT_S = 8.16571903
TIMING_COEFFICIENT_S_PER_ML = {
    "t100": 0.07633141,
    "t200": 0.35935439,
    "t400": 0.19177068,
}
_TANKS = frozenset(("t100", "t200", "t400"))
_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "split_role",
        "split_version",
        "source_sha256",
        "recipes",
        "initial_condition",
        "initial_predecessor_recipe_id",
        "vial_recipe_ids",
        "residual_volume_ml_by_vial",
        "render_seed_by_vial",
        "nuisance_group_id_by_vial",
    }
)
_RECIPE_KEYS = frozenset(("recipe_id", "programmed_ml"))
_SOURCE_HASH_KEYS = frozenset(("recipe_manifest", "split_spec"))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _lower_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def validate_one_run_source_spec(
    spec: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate and normalize one embedded dynamic source spec."""

    if not isinstance(spec, Mapping) or set(spec) != _SPEC_KEYS:
        raise ValueError("source_run_spec has unexpected fields")
    if spec["schema_version"] != 1:
        raise ValueError("source_run_spec schema_version must be 1")
    if spec["artifact_type"] != SOURCE_SPEC_ARTIFACT_TYPE:
        raise ValueError("source_run_spec artifact_type is invalid")
    run_id = _nonempty_text(spec["run_id"], "run_id")
    split_role = _nonempty_text(spec["split_role"], "split_role")
    split_version = _nonempty_text(spec["split_version"], "split_version")

    hashes = spec["source_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != _SOURCE_HASH_KEYS:
        raise ValueError("source_sha256 must contain exactly recipe_manifest and split_spec")
    normalized_hashes = {
        name: _lower_sha256(hashes[name], f"source_sha256.{name}")
        for name in ("recipe_manifest", "split_spec")
    }

    recipes = spec["recipes"]
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("recipe definitions must be a non-empty list")
    normalized_recipes: list[dict[str, object]] = []
    recipe_by_id: dict[str, dict[str, float]] = {}
    for ordinal, recipe in enumerate(recipes, start=1):
        if not isinstance(recipe, Mapping) or set(recipe) != _RECIPE_KEYS:
            raise ValueError("recipe definitions have unexpected fields")
        recipe_id = _nonempty_text(recipe["recipe_id"], f"recipes[{ordinal}].recipe_id")
        if recipe_id in recipe_by_id:
            raise ValueError("recipe definitions must have unique recipe IDs")
        volumes = recipe["programmed_ml"]
        if not isinstance(volumes, Mapping) or set(volumes) != _TANKS:
            raise ValueError("recipe programmed_ml must contain exactly t100, t200, and t400")
        normalized_volumes = {
            tank: _finite_nonnegative(
                volumes[tank], f"recipes[{ordinal}].programmed_ml.{tank}"
            )
            for tank in ("t100", "t200", "t400")
        }
        if not any(value > 0.0 for value in normalized_volumes.values()):
            raise ValueError("recipe programmed_ml must contain a positive volume")
        recipe_by_id[recipe_id] = normalized_volumes
        normalized_recipes.append(
            {"recipe_id": recipe_id, "programmed_ml": normalized_volumes}
        )

    parallel_names = (
        "vial_recipe_ids",
        "residual_volume_ml_by_vial",
        "render_seed_by_vial",
        "nuisance_group_id_by_vial",
    )
    arrays = [spec[name] for name in parallel_names]
    if (
        any(not isinstance(value, list) for value in arrays)
        or not arrays[0]
        or len({len(value) for value in arrays}) != 1
    ):
        raise ValueError("parallel vial arrays must be non-empty and equal length")
    vial_recipe_ids = [
        _nonempty_text(value, "vial recipe ID") for value in spec["vial_recipe_ids"]
    ]
    if any(recipe_id not in recipe_by_id for recipe_id in vial_recipe_ids):
        raise ValueError("every vial recipe must resolve exactly once")
    residues = [
        _finite_nonnegative(value, "residual volume")
        for value in spec["residual_volume_ml_by_vial"]
    ]
    render_seeds: list[int] = []
    for value in spec["render_seed_by_vial"]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("render seed must be a non-Boolean integer")
        render_seeds.append(value)
    nuisance_ids = [
        _nonempty_text(value, "nuisance group ID")
        for value in spec["nuisance_group_id_by_vial"]
    ]
    if len(nuisance_ids) != len(set(nuisance_ids)):
        raise ValueError("nuisance group IDs must be unique")

    initial_condition = spec["initial_condition"]
    predecessor = spec["initial_predecessor_recipe_id"]
    if initial_condition == "clean_start":
        if predecessor is not None or residues[0] != 0.0:
            raise ValueError(
                "clean start requires a null predecessor and zero vial-1 residue"
            )
        normalized_predecessor = None
    elif initial_condition == "recipe_predecessor":
        normalized_predecessor = _nonempty_text(
            predecessor, "initial_predecessor_recipe_id"
        )
        if normalized_predecessor not in recipe_by_id:
            raise ValueError("recipe predecessor must name a defined recipe")
    else:
        raise ValueError(
            "initial_condition must be clean_start or recipe_predecessor"
        )

    return {
        "schema_version": 1,
        "artifact_type": SOURCE_SPEC_ARTIFACT_TYPE,
        "run_id": run_id,
        "split_role": split_role,
        "split_version": split_version,
        "source_sha256": normalized_hashes,
        "recipes": normalized_recipes,
        "initial_condition": initial_condition,
        "initial_predecessor_recipe_id": normalized_predecessor,
        "vial_recipe_ids": vial_recipe_ids,
        "residual_volume_ml_by_vial": residues,
        "render_seed_by_vial": render_seeds,
        "nuisance_group_id_by_vial": nuisance_ids,
    }


def _duration_steps_120hz(programmed_ml: Mapping[str, object]) -> dict[str, int]:
    return {
        tank: 2
        * int(
            math.floor(
                (
                    TIMING_INTERCEPT_S
                    + TIMING_COEFFICIENT_S_PER_ML[tank] * float(programmed_ml[tank])
                )
                * 60
                + 0.5
            )
        )
        for tank in active_dispense_order(programmed_ml)
    }


def _opaque_id(namespace: str, source_hash: str, ordinal: int) -> str:
    payload = f"fluidmix-one-run-v1|{namespace}|{source_hash}|{ordinal}".encode(
        "utf-8"
    )
    return f"{namespace.upper()}_{hashlib.sha256(payload).hexdigest()[:16]}"


def build_one_run_source_artifacts(
    spec: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build the unbound render, model-input, and label artifacts from one spec."""

    normalized = validate_one_run_source_spec(spec)
    source_hash = _canonical_sha256(normalized)
    recipes = {
        row["recipe_id"]: row["programmed_ml"] for row in normalized["recipes"]
    }
    run_id = normalized["run_id"]
    split_role = normalized["split_role"]
    split_version = normalized["split_version"]
    batch_id = _opaque_id("batch", source_hash, 1)
    case_ids = [
        _opaque_id("case", source_hash, ordinal)
        for ordinal in range(1, len(normalized["vial_recipe_ids"]) + 1)
    ]
    schedules: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for ordinal, (case_id, current_id, residue) in enumerate(
        zip(
            case_ids,
            normalized["vial_recipe_ids"],
            normalized["residual_volume_ml_by_vial"],
            strict=True,
        ),
        start=1,
    ):
        previous_id = (
            normalized["initial_predecessor_recipe_id"]
            if ordinal == 1
            else normalized["vial_recipe_ids"][ordinal - 2]
        )
        current = recipes[current_id]
        previous = recipes[previous_id] if previous_id is not None else current
        order = active_dispense_order(current)
        previous_order = active_dispense_order(previous)
        schedule = build_runtime_leading_residual_schedule(
            current_recipe_id=current_id,
            previous_recipe_id=(
                previous_id if previous_id is not None else "clean_start"
            ),
            programmed_ml=current,
            dispense_order=order,
            previous_final_tank_id=previous_order[-1],
            residual_volume_ml=residue,
            duration_steps_by_tank=_duration_steps_120hz(current),
            physics_hz=PHYSICS_HZ,
            capture_cadence_steps=CAPTURE_CADENCE_STEPS,
            final_capture_offsets_s=(0.3, 3.0),
            effective_particle_volume_ml=MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
            resolution_profile_id=LEGACY_RESOLUTION_PROFILE_ID,
        )
        schedule["particle_resolution_profile"] = particle_resolution_profile(
            LEGACY_RESOLUTION_PROFILE_ID
        )
        schedule["model_input_record"] = {
            "current_recipe_id": current_id,
            "previous_recipe_id": previous_id,
            "programmed_ml": deepcopy(current),
            "previous_programmed_ml": (
                deepcopy(previous) if previous_id is not None else None
            ),
            "dispense_order": list(order),
            "previous_final_tank_id": (
                previous_order[-1] if previous_id is not None else None
            ),
        }
        schedule["case_binding"] = {
            "case_id": case_id,
            "batch_id": batch_id,
            "batch_role": "scored_target",
            "run_id": run_id,
            "split_role": split_role,
            "vial_ordinal": ordinal,
            "current_recipe_id": current_id,
            "previous_recipe_id": previous_id,
            "residual_volume_ml": residue,
            "predecessor_context": (
                normalized["initial_condition"]
                if ordinal == 1
                else "previous_vial_last_dispense"
            ),
        }
        schedule_hash = _canonical_sha256(schedule)
        schedules.append(schedule)
        model_rows.append(
            {
                "case_id": case_id,
                "batch_id": batch_id,
                "run_id": run_id,
                "vial_position": ordinal,
                "data_origin": "twin",
                "split_role": split_role,
                "split_version": split_version,
                "previous_recipe_id": previous_id,
                "current_recipe_id": current_id,
                "previous_programmed_ml": (
                    deepcopy(previous) if previous_id is not None else None
                ),
                "current_programmed_ml": deepcopy(current),
                "programmed_dispense_order": list(order),
                "stage_image_evidence_ids": [],
                "quantitative_evidence_ids": [],
            }
        )
        label_rows.append(
            {
                "case_id": case_id,
                "batch_id": batch_id,
                "run_id": run_id,
                "split_role": split_role,
                "vial_ordinal": ordinal,
                "prescribed_leading_residual_volume_ml": residue,
                "expected_disposition": "keep" if residue == 0.0 else "discard",
                "label_source": "prescribed_synthetic_latent_oracle",
                "render_schedule_sha256": schedule_hash,
                "excluded_from_model_inputs": True,
                "excluded_from_deploy_inputs": True,
            }
        )

    keep_count = sum(row["expected_disposition"] == "keep" for row in label_rows)
    discard_count = len(label_rows) - keep_count
    denominators = {
        "runs": 1,
        "batches": 1,
        "vials": len(case_ids),
        f"{split_role}_vials": len(case_ids),
        "keep_vials": keep_count,
        "discard_vials": discard_count,
        "accepted_render_artifacts": 0,
    }
    render_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": RENDER_PLAN_ARTIFACT_TYPE,
        "status": "unrendered",
        "source_run_spec": deepcopy(normalized),
        "source_run_spec_sha256": source_hash,
        "source_sha256": deepcopy(normalized["source_sha256"]),
        "split_version": split_version,
        "batch_size_vials": len(case_ids),
        "particle_resolution_profile": particle_resolution_profile(
            LEGACY_RESOLUTION_PROFILE_ID
        ),
        "runs": [
            {
                "run_id": run_id,
                "split_role": split_role,
                "initial_condition": normalized["initial_condition"],
                "initial_predecessor_recipe_id": normalized[
                    "initial_predecessor_recipe_id"
                ],
                "case_ids": list(case_ids),
            }
        ],
        "batches": [
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "partition": split_role,
                "batch_role": "scored_target",
                "vial_count": len(case_ids),
                "case_ids": list(case_ids),
                "render_seed_by_vial": list(normalized["render_seed_by_vial"]),
                "nuisance_group_ids": list(
                    normalized["nuisance_group_id_by_vial"]
                ),
                "render_status": "unrendered",
                "training_eligible": False,
            }
        ],
        "execution_schedules": schedules,
        "execution_schedule_sha256_by_case": {
            schedule["case_binding"]["case_id"]: _canonical_sha256(schedule)
            for schedule in schedules
        },
        "denominators": denominators,
        "render_contract": {
            "video_renderer": "RaytracedLighting",
            "video_fps": 10,
            "video_capture_interval_report_steps": 12,
            "stage_still_renderer": "PathTracing",
            "height_banded_particle_composition": True,
            "dedicated_stream_material": True,
        },
        "claim_boundary": (
            "synthetic construction only; embedded recipe and split hashes are "
            "provenance, prescribed residue is uncalibrated, and no physical "
            "carryover or sim-to-real validity is claimed"
        ),
    }
    model_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": MODEL_INPUT_ARTIFACT_TYPE,
        "status": "unrendered",
        "data_origin": "twin",
        "split_version": split_version,
        "vial_records": model_rows,
        "model_input_allowlist": sorted(model_rows[0]),
        "denominators": {"runs": 1, "vials": len(case_ids), "rendered_vials": 0},
        "real_images_used": False,
        "claim_boundary": "synthetic unrendered input placeholders only",
    }
    label_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": LABEL_PROVENANCE_ARTIFACT_TYPE,
        "status": "unrendered",
        "vial_labels": label_rows,
        "synthetic_label_rule": (
            "keep iff prescribed leading residue is exactly zero; otherwise discard; "
            "positive residue is an uncalibrated synthetic stress control"
        ),
        "denominators": {
            "keep_vials": keep_count,
            "discard_vials": discard_count,
        },
        "claim_boundary": (
            "labels are synthetic latent provenance and never model or deploy input"
        ),
    }
    return {
        "render-schedule-plan.json": render_plan,
        "model-input-plan.json": model_plan,
        "label-provenance.json": label_plan,
    }


def rebuild_bound_one_run_artifacts(
    observed_plan: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Rebuild and validate a fully bound artifact graph from one observed plan."""

    if not isinstance(observed_plan, Mapping):
        raise TypeError("observed one-run plan must be a mapping")
    if observed_plan.get("artifact_type") != RENDER_PLAN_ARTIFACT_TYPE:
        raise ValueError("observed one-run plan artifact_type is invalid")
    normalized = validate_one_run_source_spec(observed_plan.get("source_run_spec"))
    source_hash = _canonical_sha256(normalized)
    if observed_plan.get("source_run_spec_sha256") != source_hash:
        raise ValueError("source_run_spec_sha256 mismatch")
    resolution = validate_tank_colour_plan_overlay(observed_plan)
    if resolution is None:
        raise ValueError("source-bound one-run plan requires tank_colour_resolution")
    expected = bind_tank_colour_resolution_to_artifacts(
        build_one_run_source_artifacts(normalized), resolution
    )
    if observed_plan != expected["render-schedule-plan.json"]:
        raise ValueError("one-run plan differs from its deterministic bound rebuild")
    return expected


__all__ = [
    "LABEL_PROVENANCE_ARTIFACT_TYPE",
    "MODEL_INPUT_ARTIFACT_TYPE",
    "RENDER_PLAN_ARTIFACT_TYPE",
    "RUNTIME_CONTRACT_KIND",
    "SOURCE_SPEC_ARTIFACT_TYPE",
    "build_one_run_source_artifacts",
    "rebuild_bound_one_run_artifacts",
    "validate_one_run_source_spec",
]

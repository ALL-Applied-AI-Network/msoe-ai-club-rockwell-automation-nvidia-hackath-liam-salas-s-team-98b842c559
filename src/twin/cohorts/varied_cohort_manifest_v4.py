"""Source-bound ten-run, thirty-vial varied PhysX cohort contract (V4).

This module describes synthetic render provenance only. Recipe volumes are
rebuilt from the checked-in video-5 table; residual profiles and labels are
latent synthetic controls and are never model or deploy inputs.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

from twin.paths import REPO
from ..physx.dispense_residual import (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    build_runtime_leading_residual_schedule,
)
from ..physx.particle_contract import LEGACY_RESOLUTION_PROFILE_ID, particle_resolution_profile
from ..physx.run_variation import active_dispense_order, build_run_palette


SOURCE_RELATIVE_PATH = "data/video5_vial_recipes.csv"
PHYSICS_HZ = 120
CAPTURE_CADENCE_STEPS = 12
TIMING_INTERCEPT_S = 8.16571903
TIMING_COEFFICIENT_S_PER_ML = {
    "t100": 0.07633141,
    "t200": 0.35935439,
    "t400": 0.19177068,
}

# (run ID, run seed, split, initial predecessor recipe, current recipe,
# vial render seeds, monotone leading-residual profile in mL).
RUN_SPECS = (
    ("RUN-V4-TRAIN-01", 52001, "train", "14", "5", (520101, 520102, 520103), (0.0, 0.0, 0.0)),
    ("RUN-V4-TRAIN-02", 52002, "train", "5", "6", (520201, 520202, 520203), (9.0, 0.0, 0.0)),
    ("RUN-V4-TRAIN-03", 52003, "train", "6", "7", (520301, 520302, 520303), (9.0, 4.5, 0.0)),
    ("RUN-V4-TRAIN-04", 52004, "train", "7", "8", (520401, 520402, 520403), (9.0, 4.5, 2.25)),
    ("RUN-V4-VALIDATION-01", 52005, "validation", "8", "9", (520501, 520502, 520503), (0.0, 0.0, 0.0)),
    ("RUN-V4-VALIDATION-02", 52006, "validation", "9", "10", (520601, 520602, 520603), (6.0, 0.0, 0.0)),
    ("RUN-V4-VALIDATION-03", 52007, "validation", "10", "11", (520701, 520702, 520703), (6.0, 3.0, 1.5)),
    ("RUN-V4-EVALUATION-01", 52008, "evaluation", "11", "12", (520801, 520802, 520803), (0.0, 0.0, 0.0)),
    ("RUN-V4-EVALUATION-02", 52009, "evaluation", "12", "13", (520901, 520902, 520903), (6.0, 3.0, 0.0)),
    ("RUN-V4-EVALUATION-03", 52010, "evaluation", "13", "14", (521001, 521002, 521003), (6.0, 3.0, 1.5)),
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _opaque_id(namespace: str, *values: object) -> str:
    payload = "|".join((namespace, *(str(value) for value in values))).encode()
    return f"{namespace.upper()}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _load_real_recipe_volumes(root: Path) -> tuple[dict[str, dict[str, int]], str]:
    path = root / SOURCE_RELATIVE_PATH
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    selected_ids = {spec[3] for spec in RUN_SPECS} | {spec[4] for spec in RUN_SPECS}
    recipes: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            recipe_id = row["recipe_id"]
            if recipe_id not in selected_ids:
                continue
            volumes = {tank: int(row[f"{tank}_ml"] or 0) for tank in ("t100", "t200", "t400")}
            if recipe_id in recipes and recipes[recipe_id] != volumes:
                raise ValueError(f"recipe {recipe_id} volumes vary across video-5 rows")
            recipes[recipe_id] = volumes
    if set(recipes) != selected_ids:
        raise ValueError("selected real recipe metadata is incomplete")
    return recipes, source_sha256


def _duration_steps_120hz(programmed_ml: dict[str, int]) -> dict[str, int]:
    return {
        tank: 2 * int(math.floor((TIMING_INTERCEPT_S + TIMING_COEFFICIENT_S_PER_ML[tank] * programmed_ml[tank]) * 60 + 0.5))
        for tank in active_dispense_order(programmed_ml)
    }


def _build_schedule(
    *, case_id: str, batch_id: str, run_id: str, split_role: str,
    palette_sha256: str, vial_ordinal: int, residual_volume_ml: float,
    previous_recipe_id: str, previous_programmed_ml: dict[str, int],
    current_recipe_id: str, current_programmed_ml: dict[str, int],
) -> dict[str, object]:
    order = active_dispense_order(current_programmed_ml)
    previous_order = active_dispense_order(previous_programmed_ml)
    schedule = build_runtime_leading_residual_schedule(
        current_recipe_id=current_recipe_id,
        previous_recipe_id=previous_recipe_id,
        programmed_ml=current_programmed_ml,
        dispense_order=order,
        previous_final_tank_id=previous_order[-1],
        residual_volume_ml=residual_volume_ml,
        duration_steps_by_tank=_duration_steps_120hz(current_programmed_ml),
        physics_hz=PHYSICS_HZ,
        capture_cadence_steps=CAPTURE_CADENCE_STEPS,
        final_capture_offsets_s=(0.3, 3.0),
        effective_particle_volume_ml=MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        resolution_profile_id=LEGACY_RESOLUTION_PROFILE_ID,
    )
    schedule["particle_resolution_profile"] = particle_resolution_profile(LEGACY_RESOLUTION_PROFILE_ID)
    schedule["model_input_record"] = {
        "current_recipe_id": current_recipe_id,
        "previous_recipe_id": previous_recipe_id,
        "programmed_ml": deepcopy(current_programmed_ml),
        "previous_programmed_ml": deepcopy(previous_programmed_ml),
        "dispense_order": list(order),
        "previous_final_tank_id": previous_order[-1],
    }
    schedule["case_binding"] = {
        "case_id": case_id,
        "batch_id": batch_id,
        "batch_role": "scored_target",
        "run_id": run_id,
        "split_role": split_role,
        "palette_sha256": palette_sha256,
        "vial_ordinal": vial_ordinal,
        "current_recipe_id": current_recipe_id,
        "previous_recipe_id": previous_recipe_id,
        "residual_volume_ml": residual_volume_ml,
        "predecessor_context": "previous_recipe_last_dispense" if vial_ordinal == 1 else "previous_vial_last_dispense",
    }
    return schedule


def build_varied_cohort_artifacts(*, repo_root: str | Path = REPO) -> dict[str, dict[str, object]]:
    """Return the unrendered source-rebuilt 10-run/30-vial V4 cohort."""

    root = Path(repo_root)
    recipes, source_sha256 = _load_real_recipe_volumes(root)
    runs: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    schedules: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []

    for run_id, run_seed, split_role, initial_previous_id, current_id, seeds, profile in RUN_SPECS:
        if tuple(profile) != tuple(sorted(profile, reverse=True)):
            raise ValueError(f"run {run_id} residual profile must be monotone")
        palette = build_run_palette(run_id=run_id, run_seed=run_seed)
        batch_id = _opaque_id("batch", run_id, current_id)
        case_ids: list[str] = []
        initial_previous, current = recipes[initial_previous_id], recipes[current_id]
        for vial_ordinal, (residual_volume_ml, render_seed) in enumerate(zip(profile, seeds, strict=True), start=1):
            case_id = _opaque_id("case", run_id, current_id, vial_ordinal)
            case_ids.append(case_id)
            previous_id = initial_previous_id if vial_ordinal == 1 else current_id
            previous = initial_previous if vial_ordinal == 1 else current
            schedule = _build_schedule(
                case_id=case_id, batch_id=batch_id, run_id=run_id, split_role=split_role,
                palette_sha256=str(palette["palette_sha256"]), vial_ordinal=vial_ordinal,
                residual_volume_ml=float(residual_volume_ml), previous_recipe_id=previous_id,
                previous_programmed_ml=previous, current_recipe_id=current_id,
                current_programmed_ml=current,
            )
            schedules.append(schedule)
            model_rows.append({
                "case_id": case_id, "batch_id": batch_id, "run_id": run_id,
                "vial_position": vial_ordinal, "data_origin": "twin",
                "split_role": split_role, "split_version": "whole-run-varied-v4",
                "previous_recipe_id": previous_id, "current_recipe_id": current_id,
                "previous_programmed_ml": deepcopy(previous), "current_programmed_ml": deepcopy(current),
                "programmed_dispense_order": list(active_dispense_order(current)),
                "stage_image_evidence_ids": [], "quantitative_evidence_ids": [],
            })
            label_rows.append({
                "case_id": case_id, "batch_id": batch_id, "run_id": run_id,
                "split_role": split_role, "vial_ordinal": vial_ordinal,
                "prescribed_leading_residual_volume_ml": float(residual_volume_ml),
                "expected_disposition": "keep" if residual_volume_ml == 0.0 else "discard",
                "label_source": "prescribed_synthetic_latent_oracle",
                "render_schedule_sha256": _canonical_sha256(schedule),
                "excluded_from_model_inputs": True, "excluded_from_deploy_inputs": True,
            })
        runs.append({
            "run_id": run_id, "run_seed": run_seed, "split_role": split_role,
            "initial_previous_recipe_id": initial_previous_id, "current_recipe_id": current_id,
            "palette": palette, "case_ids": case_ids,
        })
        batches.append({
            "batch_id": batch_id, "run_id": run_id, "partition": split_role,
            "batch_role": "scored_target", "vial_count": 3, "case_ids": case_ids,
            "render_seed_by_vial": list(seeds),
            "nuisance_group_ids": [f"{run_id}-NUISANCE-V{ordinal}" for ordinal in (1, 2, 3)],
            "run_palette_id": run_id, "run_palette_seed": run_seed,
            "render_status": "unrendered", "training_eligible": False,
        })

    denominators = {
        "runs": 10, "batches": 10, "vials": 30,
        "train_vials": 12, "validation_vials": 9, "evaluation_vials": 9,
        "keep_vials": 15, "discard_vials": 15, "accepted_render_artifacts": 0,
    }
    render_plan: dict[str, object] = {
        "schema_version": 4,
        "artifact_type": "physx_f2_varied_run_residual_render_plan_v4",
        "status": "unrendered", "source_sha256": {SOURCE_RELATIVE_PATH: source_sha256},
        "batch_size_vials": 3,
        "particle_resolution_profile": particle_resolution_profile(LEGACY_RESOLUTION_PROFILE_ID),
        "runs": runs, "batches": batches, "execution_schedules": schedules,
        "execution_schedule_sha256_by_case": {
            schedule["case_binding"]["case_id"]: _canonical_sha256(schedule) for schedule in schedules
        },
        "denominators": denominators,
        "render_contract": {
            "video_renderer": "RaytracedLighting", "video_fps": 10,
            "video_capture_interval_report_steps": 12, "stage_still_renderer": "PathTracing",
            "height_banded_particle_composition": True, "dedicated_stream_material": True,
        },
        "claim_boundary": (
            "synthetic construction only; recipe metadata comes from the real schedule, "
            "but residue severity and run palette are uncalibrated and no sim-to-real "
            "or physical carryover validity is claimed"
        ),
    }
    model_plan: dict[str, object] = {
        "schema_version": 4, "artifact_type": "synthetic_varied_run_input_plan_v4",
        "status": "unrendered", "data_origin": "twin", "split_version": "whole-run-varied-v4",
        "vial_records": model_rows, "model_input_allowlist": sorted(model_rows[0]),
        "denominators": {"runs": 10, "vials": 30, "rendered_vials": 0},
        "real_images_used": False, "claim_boundary": "synthetic unrendered input placeholders only",
    }
    label_plan: dict[str, object] = {
        "schema_version": 4, "artifact_type": "synthetic_varied_run_label_provenance_v4",
        "status": "unrendered", "vial_labels": label_rows,
        "synthetic_label_rule": (
            "keep iff prescribed leading residue is exactly zero; otherwise discard; "
            "positive profiles are uncalibrated synthetic stress controls, not real thresholds"
        ),
        "denominators": {"keep_vials": 15, "discard_vials": 15},
        "claim_boundary": "labels are synthetic latent provenance and never model or deploy input",
    }
    return {
        "render-schedule-plan.json": render_plan,
        "model-input-plan.json": model_plan,
        "label-provenance.json": label_plan,
    }


__all__ = ["RUN_SPECS", "SOURCE_RELATIVE_PATH", "build_varied_cohort_artifacts"]

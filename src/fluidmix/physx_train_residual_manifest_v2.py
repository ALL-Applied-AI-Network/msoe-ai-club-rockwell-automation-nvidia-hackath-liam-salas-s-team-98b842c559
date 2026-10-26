"""Frozen train-only three-vial leading-residue render manifest.

This module prepares synthetic renderer inputs and separated scoring provenance.
It does not launch Isaac, accept a render, or establish physical carryover validity.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

from .paths import REPO
from .physx_dispense_residual import (
    MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    build_runtime_leading_residual_schedule,
)


SOURCE_SHA256 = {
    "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
    "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
    "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
}
PREVIOUS_RECIPE_ID = "S0437"
CURRENT_RECIPE_ID = "S0117"
PREVIOUS_RECIPE_ROW_SHA256 = "f51e3e40134eb0cf5d79c95e96e3543550df55b77325e6ec812ede538085bda7"
CURRENT_RECIPE_ROW_SHA256 = "ebb4419f41397f0f16d7cd69417b7b3d1b1fb0837a1d517c37acaf2081f68f67"
TRANSITION_ROW_SHA256 = "3b3db95c2ed734d397bf1b14bf2eedbab3fb73cb4d51232a04b54e46152eb31d"
DISPENSE_ORDER = ("t200", "t400", "t100")
PREVIOUS_FINAL_TANK_ID = "t100"
PHYSICS_HZ = 120
CAPTURE_CADENCE_STEPS = 12
TIMING_INTERCEPT_S = 8.16571903
TIMING_COEFFICIENT_S_PER_ML = {
    "t100": 0.07633141,
    "t200": 0.35935439,
    "t400": 0.19177068,
}
PROFILE_ROWS = (
    ("clean", (0.0, 0.0, 0.0)),
    ("first_two", (1.0, 0.5, 0.0)),
    ("all_three", (1.0, 0.5, 0.4)),
)
MATCHED_RENDER_SEEDS = (340101, 340102, 340103)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _opaque_id(namespace: str, *values: object) -> str:
    payload = "|".join((namespace, *(str(value) for value in values))).encode()
    return f"{namespace.upper()}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _load_bound_sources(root: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file() or _sha256_path(path) != expected:
            raise ValueError(f"source hash drift: {relative}")

    recipe_lines = (root / "data/sim_recipes.csv").read_text(
        encoding="utf-8"
    ).splitlines(keepends=True)
    selected: dict[str, dict[str, object]] = {}
    for line, row in zip(
        recipe_lines[1:], csv.DictReader(recipe_lines), strict=True
    ):
        recipe_id = row["sim_recipe_id"]
        if recipe_id not in {PREVIOUS_RECIPE_ID, CURRENT_RECIPE_ID}:
            continue
        selected[recipe_id] = {
            "sim_recipe_id": recipe_id,
            "role": row["role"],
            "cell": row["cell"],
            "programmed_ml": {
                tank: int(row[f"{tank}_ml"])
                for tank in ("t100", "t200", "t400")
            },
            "total_ml": int(row["total_ml"]),
            "row_sha256": hashlib.sha256(line.encode()).hexdigest(),
        }
    if set(selected) != {PREVIOUS_RECIPE_ID, CURRENT_RECIPE_ID}:
        raise ValueError("selected train recipe rows are missing")
    previous = selected[PREVIOUS_RECIPE_ID]
    current = selected[CURRENT_RECIPE_ID]
    if previous["row_sha256"] != PREVIOUS_RECIPE_ROW_SHA256:
        raise ValueError("previous train recipe row drift")
    if current["row_sha256"] != CURRENT_RECIPE_ROW_SHA256:
        raise ValueError("current train recipe row drift")
    if previous["role"] != "train" or current["role"] != "train":
        raise ValueError("selected recipe family is not wholly train")

    transition_lines = (root / "data/sim_transitions.csv").read_text(
        encoding="utf-8"
    ).splitlines(keepends=True)
    transition: dict[str, object] | None = None
    for line, row in zip(
        transition_lines[1:], csv.DictReader(transition_lines), strict=True
    ):
        if (
            row["role"] == "train"
            and row["prev_sim_recipe_id"] == PREVIOUS_RECIPE_ID
            and row["sim_recipe_id"] == CURRENT_RECIPE_ID
        ):
            transition = {
                **row,
                "row_sha256": hashlib.sha256(line.encode()).hexdigest(),
            }
            break
    if transition is None or transition["row_sha256"] != TRANSITION_ROW_SHA256:
        raise ValueError("selected train transition row drift")

    split = json.loads(
        (root / "data/sim_split_spec.json").read_text(encoding="utf-8")
    )
    train_range = split.get("render_seed_ranges", {}).get("train")
    if split.get("version") != "v2" or train_range != [0, 999999]:
        raise ValueError("train split contract drift")
    if any(not train_range[0] <= seed <= train_range[1] for seed in MATCHED_RENDER_SEEDS):
        raise ValueError("render seed leaves the train partition")
    return previous, current, transition


def _duration_steps_120hz(programmed_ml: dict[str, int]) -> dict[str, int]:
    durations: dict[str, int] = {}
    for tank in DISPENSE_ORDER:
        steps_60hz = math.floor(
            (
                TIMING_INTERCEPT_S
                + TIMING_COEFFICIENT_S_PER_ML[tank] * programmed_ml[tank]
            )
            * 60
            + 0.5
        )
        durations[tank] = 2 * int(steps_60hz)
    return durations


def _case_schedule(
    *,
    case_id: str,
    batch_id: str,
    vial_ordinal: int,
    residual_volume_ml: float,
    previous: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    schedule = build_runtime_leading_residual_schedule(
        current_recipe_id=CURRENT_RECIPE_ID,
        previous_recipe_id=PREVIOUS_RECIPE_ID,
        programmed_ml=current["programmed_ml"],
        dispense_order=DISPENSE_ORDER,
        previous_final_tank_id=PREVIOUS_FINAL_TANK_ID,
        residual_volume_ml=residual_volume_ml,
        duration_steps_by_tank=_duration_steps_120hz(current["programmed_ml"]),
        physics_hz=PHYSICS_HZ,
        capture_cadence_steps=CAPTURE_CADENCE_STEPS,
        final_capture_offsets_s=(0.3, 3.0),
        effective_particle_volume_ml=MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
    )
    schedule["case_binding"] = {
        "case_id": case_id,
        "batch_id": batch_id,
        "vial_ordinal": vial_ordinal,
        "residual_volume_ml": residual_volume_ml,
        "predecessor_context": (
            "previous_recipe_last_dispense"
            if vial_ordinal == 1
            else "previous_vial_last_dispense"
        ),
    }
    schedule["source_binding"] = {
        "previous_recipe_row_sha256": previous["row_sha256"],
        "current_recipe_row_sha256": current["row_sha256"],
        "transition_row_sha256": TRANSITION_ROW_SHA256,
    }
    return schedule


def build_train_residual_artifacts(
    *, repo_root: str | Path = REPO
) -> dict[str, dict[str, object]]:
    """Return separated unrendered plans for one train-only transition family."""

    root = Path(repo_root)
    previous, current, transition = _load_bound_sources(root)
    family_id = _opaque_id("family", PREVIOUS_RECIPE_ID, CURRENT_RECIPE_ID)
    schedules: list[dict[str, object]] = []
    render_batches: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []

    for profile_id, residual_by_vial in PROFILE_ROWS:
        batch_id = _opaque_id("batch", family_id, profile_id)
        case_ids: list[str] = []
        for vial_ordinal, (residual_volume_ml, render_seed) in enumerate(
            zip(residual_by_vial, MATCHED_RENDER_SEEDS, strict=True), start=1
        ):
            case_id = _opaque_id("case", batch_id, vial_ordinal)
            case_ids.append(case_id)
            schedule = _case_schedule(
                case_id=case_id,
                batch_id=batch_id,
                vial_ordinal=vial_ordinal,
                residual_volume_ml=residual_volume_ml,
                previous=previous,
                current=current,
            )
            schedule_hash = _canonical_sha256(schedule)
            schedules.append(schedule)
            model_rows.append(
                {
                    "case_id": case_id,
                    "batch_id": batch_id,
                    "family_id": family_id,
                    "vial_position": vial_ordinal,
                    "data_origin": "twin",
                    "split_role": "train",
                    "split_version": "inn-7-splits-v2",
                    "previous_sim_recipe_id": PREVIOUS_RECIPE_ID,
                    "current_sim_recipe_id": CURRENT_RECIPE_ID,
                    "previous_programmed_ml": deepcopy(previous["programmed_ml"]),
                    "current_programmed_ml": deepcopy(current["programmed_ml"]),
                    "programmed_dispense_order": list(DISPENSE_ORDER),
                    "stage_image_evidence_ids": [],
                    "quantitative_evidence_ids": [],
                }
            )
            disposition = "keep" if residual_volume_ml == 0.0 else "discard"
            label_rows.append(
                {
                    "case_id": case_id,
                    "batch_id": batch_id,
                    "vial_ordinal": vial_ordinal,
                    "synthetic_profile_id": profile_id,
                    "prescribed_leading_residual_volume_ml": residual_volume_ml,
                    "prescribed_leading_residual_particle_count_per_dispense": int(
                        schedule["pours"][0]["leading_residual_particle_count"]
                    ),
                    "expected_disposition": disposition,
                    "label_source": "prescribed_synthetic_latent_oracle",
                    "render_schedule_sha256": schedule_hash,
                    "excluded_from_model_inputs": True,
                    "excluded_from_deploy_inputs": True,
                }
            )
        render_batches.append(
            {
                "batch_id": batch_id,
                "family_id": family_id,
                "partition": "train",
                "vial_count": 3,
                "case_ids": case_ids,
                "render_seed_by_vial": list(MATCHED_RENDER_SEEDS),
                "nuisance_group_ids": [
                    f"{family_id}-MATCHED-V{ordinal}" for ordinal in (1, 2, 3)
                ],
                "render_status": "unrendered",
                "training_eligible": False,
            }
        )

    keep_count = sum(row["expected_disposition"] == "keep" for row in label_rows)
    discard_count = len(label_rows) - keep_count
    family = {
        "family_id": family_id,
        "partition": "train",
        "previous_sim_recipe_id": PREVIOUS_RECIPE_ID,
        "current_sim_recipe_id": CURRENT_RECIPE_ID,
        "previous_role": previous["role"],
        "current_role": current["role"],
        "transition_role": transition["role"],
        "previous_recipe_row_sha256": previous["row_sha256"],
        "current_recipe_row_sha256": current["row_sha256"],
        "transition_row_sha256": transition["row_sha256"],
    }
    denominators = {
        "families": 1,
        "batches": len(render_batches),
        "vials": len(model_rows),
        "keep_vials": keep_count,
        "discard_vials": discard_count,
        "execution_schedules": len(schedules),
        "stage_images_planned": len(schedules) * 4,
        "accepted_render_artifacts": 0,
    }
    render_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "physx_f2_train_leading_residue_render_plan_v2",
        "status": "unrendered_blocked",
        "partition": "train",
        "source_sha256": dict(SOURCE_SHA256),
        "family": family,
        "batch_size_vials": 3,
        "batches": render_batches,
        "execution_schedules": schedules,
        "execution_schedule_sha256_by_case": {
            schedule["case_binding"]["case_id"]: _canonical_sha256(schedule)
            for schedule in schedules
        },
        "denominators": denominators,
        "render_gate": {
            "render_authorized": False,
            "training_eligible": False,
            "block_reason": "accepted_real_recipe_three_stage_qualification_required",
            "accepted_qualification_receipt_sha256": None,
        },
        "claim_boundary": (
            "synthetic unrendered plan only; no physical carryover, real defect, "
            "real cause, chemistry, or sim-to-real claim"
        ),
    }
    model_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "synthetic_train_input_plan_v2",
        "status": "unrendered",
        "data_origin": "twin",
        "split_version": "inn-7-splits-v2",
        "partition": "train",
        "vial_records": model_rows,
        "model_input_allowlist": sorted(model_rows[0]),
        "denominators": {
            "families": 1,
            "batches": len(render_batches),
            "vials": len(model_rows),
            "rendered_vials": 0,
        },
        "real_data_used": False,
        "claim_boundary": "synthetic unrendered input placeholders only",
    }
    label_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "synthetic_train_label_provenance_v2",
        "status": "unrendered",
        "family_id": family_id,
        "vial_labels": label_rows,
        "synthetic_label_rule": (
            "keep iff prescribed leading residue is exactly zero; otherwise "
            "discard; this is an uncalibrated scoring oracle only"
        ),
        "denominators": {
            "keep_vials": keep_count,
            "discard_vials": discard_count,
        },
        "claim_boundary": (
            "labels describe prescribed synthetic controls and do not define a "
            "real plant rejection threshold"
        ),
    }
    return {
        "render-schedule-plan.json": render_plan,
        "model-input-plan.json": model_plan,
        "label-provenance.json": label_plan,
    }


__all__ = ["build_train_residual_artifacts"]

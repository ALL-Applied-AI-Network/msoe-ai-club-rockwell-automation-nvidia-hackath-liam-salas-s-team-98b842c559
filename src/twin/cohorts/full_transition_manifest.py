"""Source-bound WP1 collection manifest for two AAA-BBB-CCC test runs.

The collection layer never merges the child render plans: the qualified supervisor
accepts exactly one run and one batch per plan.  It binds two independently rebuildable
one-run triples, a separate label artifact per run, and a pre-render leakage manifest.
All residue, preset, and disposition data is Part-A-only synthetic provenance.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping

from fluidmix.colour.absorbance import delta_e, linear_to_srgb
from fluidmix.core.contracts._common import stable_read_path
from fluidmix.core.paths import REPO
from .one_run_manifest_v1 import (
    build_one_run_source_artifacts,
    rebuild_bound_one_run_artifacts,
    validate_one_run_source_spec,
)
from ..physx.run_variation import (
    active_dispense_order,
    bind_tank_colour_resolution_to_artifacts,
    validate_tank_colour_resolution,
)
from ..scene.tank_colour_presets import (
    load_tank_colour_preset_catalog,
    resolve_tank_colour_preset,
)
from .test_recipe_generator import build_test_recipe_plan


SOURCE_ARTIFACT_TYPE = "fluidmix-full-transition-source-spec-v1"
COLLECTION_ARTIFACT_TYPE = "fluidmix-full-transition-collection-v1"
SPLIT_VERSION = "inn-7-splits-v3"
RUN_IDS = ("WP1-TEST-RUN-01", "WP1-TEST-RUN-02")
RUN_PRESETS = ("amber_teal", "crimson_indigo")
RESIDUE_PROFILE_ID = "wp1-minimum-two-run-residue-v1"
RESIDUE_PROFILES = (
    (0.0, 0.0, 0.0, 9.0, 1.5, 0.0, 9.0, 4.5, 1.5),
    (0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 6.0, 3.0, 0.0),
)
VISIBILITY_GATE = {
    "measure_after_run_id": RUN_IDS[0],
    "first_switch_vial_ordinal": 4,
    "same_recipe_comparison_vial_ordinals": [5, 6],
    "minimum_delta_e76": 2.0,
    "low_positive_residue_ml": 1.5,
    "fallback_residue_ml": 3.0,
    "fallback_target_vial_ordinals": [5, 9],
    "fallback_mode": "fresh_additional_cases_preserve_originals",
    "automatic_manifest_mutation": False,
}
_EXPECTED_SOURCE_SHA256 = {
    "recipe_manifest": "49ab190bf018f61c868f48ddb83dd96d7e1d0ff77080e68d62953d8f8f9ef49b",
    "split_spec": "e6e4b3b8d2c0932629ffae5872a58df9f4fb1b03612b0e842e53339a4d071bfb",
}
_SOURCE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "split_role",
        "split_version",
        "source_sha256",
        "residue_profile_id",
        "visibility_gate",
        "runs",
        "excluded_from_model_inputs",
        "claim_boundary",
    }
)
_RUN_KEYS = frozenset({"source_run_spec", "tank_colour_resolution"})
_FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "tank_colour_resolution",
        "run_palette",
        "palette",
        "preset",
        "preset_name",
        "transmission_1cm",
        "transmission_overrides",
        "dye_strength_scale",
        "t100_clear_lock",
        "randomize_around_preset",
        "randomization_seed",
        "tank_strength_jitter_bounds",
        "channel_jitter_bounds",
        "tank_strength_jitter",
        "channel_jitter",
        "free_text_label",
        "tank_colour_preset_hash",
        "renderer_palette_sha256",
        "palette_sha256",
        "residual_volume_ml",
        "prescribed_leading_residual_volume_ml",
        "expected_disposition",
        "label_provenance",
        "excluded_from_model_inputs",
        "oracle",
        "particle_state",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _source_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "recipe_manifest": repo_root / "data" / "sim_recipes_v3.csv",
        "split_spec": repo_root / "data" / "sim_split_spec_v3.json",
    }


def _source_snapshot(repo_root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    raw = {
        name: stable_read_path(path, f"WP1 {name}", max_bytes=8 * 1024 * 1024)
        for name, path in _source_paths(repo_root).items()
    }
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    if hashes != _EXPECTED_SOURCE_SHA256:
        raise ValueError("approved WP1 v3 source hash drift")
    return raw, hashes


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return _source_snapshot(repo_root)[1]


def _load_v3_recipes(raw: bytes) -> dict[str, dict[str, object]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("v3 recipe manifest must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text))
    required = {
        "sim_recipe_id", "role", "cell", "t100_ml", "t200_ml", "t400_ml",
        "total_ml",
    }
    if reader.fieldnames is None or set(reader.fieldnames) != required:
        raise ValueError("v3 recipe manifest columns are invalid")
    rows = list(reader)
    result = {}
    for row in rows:
        recipe_id = row["sim_recipe_id"]
        if recipe_id in result:
            raise ValueError("v3 recipe manifest contains duplicate IDs")
        programmed = {
            tank: int(row[f"{tank}_ml"] or 0) for tank in ("t100", "t200", "t400")
        }
        if int(row["total_ml"]) != sum(programmed.values()):
            raise ValueError("v3 recipe manifest total has drifted")
        result[recipe_id] = {
            "role": row["role"],
            "programmed_ml": programmed,
        }
    return result


def _resolved_preset(repo_root: Path, preset_name: str) -> dict[str, object]:
    catalog = load_tank_colour_preset_catalog(
        repo_root / "configs" / "tank_colour_presets.json"
    )
    return resolve_tank_colour_preset(catalog, preset_name).to_json_record()


def build_default_full_transition_source_spec(
    *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Build the frozen minimum two-run source spec from recipe v3."""

    root = Path(repo_root).resolve()
    recipe_plan = build_test_recipe_plan(repo_root=root)
    hashes = _source_hashes(root)
    runs = []
    for run_ordinal, (recipe_run, run_id, preset_name, residues) in enumerate(
        zip(
            recipe_plan["runs"], RUN_IDS, RUN_PRESETS, RESIDUE_PROFILES, strict=True
        ),
        start=1,
    ):
        recipes = [
            {
                "recipe_id": row["sim_recipe_id"],
                "programmed_ml": deepcopy(row["programmed_ml"]),
            }
            for row in recipe_run["recipes"]
        ]
        recipe_ids = [row["recipe_id"] for row in recipes]
        vial_recipe_ids = [
            recipe_id for recipe_id in recipe_ids for _ in range(3)
        ]
        seed_start = 2_610_000 + (run_ordinal - 1) * 100
        source_run_spec = {
            "schema_version": 1,
            "artifact_type": "fluidmix-one-run-source-spec-v1",
            "run_id": run_id,
            "split_role": "test",
            "split_version": SPLIT_VERSION,
            "source_sha256": deepcopy(hashes),
            "recipes": recipes,
            "initial_condition": "clean_start",
            "initial_predecessor_recipe_id": None,
            "vial_recipe_ids": vial_recipe_ids,
            "residual_volume_ml_by_vial": list(residues),
            "render_seed_by_vial": [seed_start + ordinal for ordinal in range(1, 10)],
            "nuisance_group_id_by_vial": [
                f"WP1-TEST-R{run_ordinal:02d}-V{ordinal:02d}"
                for ordinal in range(1, 10)
            ],
        }
        runs.append(
            {
                "source_run_spec": source_run_spec,
                "tank_colour_resolution": _resolved_preset(root, preset_name),
            }
        )
    source = {
        "schema_version": 1,
        "artifact_type": SOURCE_ARTIFACT_TYPE,
        "split_role": "test",
        "split_version": SPLIT_VERSION,
        "source_sha256": hashes,
        "residue_profile_id": RESIDUE_PROFILE_ID,
        "visibility_gate": deepcopy(VISIBILITY_GATE),
        "runs": runs,
        "excluded_from_model_inputs": True,
        "claim_boundary": (
            "synthetic Part-A planning provenance only; prescribed residue and colours "
            "are excluded from model/deploy inputs and make no plant-validity claim"
        ),
    }
    return validate_full_transition_source_spec(source, repo_root=root)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _preset_delta_e76(
    left: Mapping[str, object], right: Mapping[str, object], tank: str
) -> float:
    left_rgb = left["transmission_1cm"][tank]
    right_rgb = right["transmission_1cm"][tank]
    return delta_e(linear_to_srgb(left_rgb), linear_to_srgb(right_rgb))


def validate_full_transition_source_spec(
    source: Mapping[str, object], *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Validate the complete Part-A-only source contract before artifact creation."""

    if not isinstance(source, Mapping) or set(source) != _SOURCE_KEYS:
        raise ValueError("full-transition source has unexpected fields")
    if source["schema_version"] != 1 or source["artifact_type"] != SOURCE_ARTIFACT_TYPE:
        raise ValueError("full-transition source schema is invalid")
    if source["split_role"] != "test" or source["split_version"] != SPLIT_VERSION:
        raise ValueError("full-transition source split identity is invalid")
    if source["residue_profile_id"] != RESIDUE_PROFILE_ID:
        raise ValueError("full-transition residue profile is invalid")
    if source["visibility_gate"] != VISIBILITY_GATE:
        raise ValueError("full-transition visibility fallback gate has drifted")
    if source["excluded_from_model_inputs"] is not True:
        raise ValueError("full-transition truth must be excluded from model inputs")
    if not isinstance(source["claim_boundary"], str) or not source[
        "claim_boundary"
    ].strip():
        raise ValueError("full-transition claim boundary must be non-empty")

    root = Path(repo_root).resolve()
    raw_sources, actual_hashes = _source_snapshot(root)
    if source["source_sha256"] != actual_hashes:
        raise ValueError("full-transition source hash mismatch")
    authoritative = _load_v3_recipes(raw_sources["recipe_manifest"])
    generated_plan = build_test_recipe_plan(repo_root=root)
    runs = source["runs"]
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("full-transition source must contain exactly two runs")

    normalized_runs = []
    all_recipe_ids: list[str] = []
    all_case_seeds: list[int] = []
    all_nuisance_ids: list[str] = []
    keep_count = 0
    discard_count = 0
    low_positive_count = 0
    palette_hashes = []
    resolutions = []
    for run_index, (run, run_id, preset_name, expected_residues) in enumerate(
        zip(runs, RUN_IDS, RUN_PRESETS, RESIDUE_PROFILES, strict=True)
    ):
        if not isinstance(run, Mapping) or set(run) != _RUN_KEYS:
            raise ValueError("full-transition run has unexpected fields")
        run_spec = validate_one_run_source_spec(run["source_run_spec"])
        if (
            run_spec["run_id"] != run_id
            or run_spec["split_role"] != "test"
            or run_spec["split_version"] != SPLIT_VERSION
            or run_spec["source_sha256"] != actual_hashes
        ):
            raise ValueError("full-transition child run identity or source hash is invalid")
        if (
            run_spec["initial_condition"] != "clean_start"
            or run_spec["initial_predecessor_recipe_id"] is not None
            or run_spec["residual_volume_ml_by_vial"][0] != 0.0
        ):
            raise ValueError("full-transition run must use a clean start")
        recipe_ids = [row["recipe_id"] for row in run_spec["recipes"]]
        if len(recipe_ids) != 3 or len(set(recipe_ids)) != 3:
            raise ValueError("each full-transition run needs three unique recipes")
        expected_sequence = [recipe_id for recipe_id in recipe_ids for _ in range(3)]
        if run_spec["vial_recipe_ids"] != expected_sequence:
            raise ValueError("full-transition sequence must be exactly AAA BBB CCC")
        expected_generated_recipes = [
            {
                "recipe_id": row["sim_recipe_id"],
                "programmed_ml": {
                    tank: float(row["programmed_ml"][tank])
                    for tank in ("t100", "t200", "t400")
                },
            }
            for row in generated_plan["runs"][run_index]["recipes"]
        ]
        if run_spec["recipes"] != expected_generated_recipes:
            raise ValueError("full-transition recipes differ from the frozen WP1 generated selection")
        for row in run_spec["recipes"]:
            authoritative_row = authoritative.get(row["recipe_id"])
            expected_programmed = {
                tank: float(authoritative_row["programmed_ml"][tank])
                for tank in ("t100", "t200", "t400")
            } if authoritative_row is not None else None
            if (
                authoritative_row is None
                or authoritative_row["role"] != "test"
                or row["programmed_ml"] != expected_programmed
            ):
                raise ValueError("full-transition recipe differs from authoritative v3 test data")
        all_recipe_ids.extend(recipe_ids)

        residues = tuple(run_spec["residual_volume_ml_by_vial"])
        if residues != expected_residues:
            raise ValueError("full-transition residue profile has drifted")
        if residues[3] <= 0.0 or residues[6] <= 0.0:
            raise ValueError("both recipe switches require positive residue")
        keep_count += sum(value == 0.0 for value in residues)
        discard_count += sum(value > 0.0 for value in residues)
        low_positive_count += sum(0.0 < value <= 1.5 for value in residues)

        seeds = run_spec["render_seed_by_vial"]
        if any(not 2_000_000 <= seed <= 2_999_999 for seed in seeds):
            raise ValueError("full-transition render seed is outside the test range")
        all_case_seeds.extend(seeds)
        all_nuisance_ids.extend(run_spec["nuisance_group_id_by_vial"])

        resolution = validate_tank_colour_resolution(run["tank_colour_resolution"])
        if resolution["preset_name"] != preset_name:
            raise ValueError("full-transition preset assignment is invalid")
        if resolution["transmission_1cm"]["t100"] != [1.0, 1.0, 1.0]:
            raise ValueError("full-transition T100 must remain clear")
        if resolution["excluded_from_model_inputs"] is not True:
            raise ValueError("full-transition preset must be excluded from model inputs")
        palette_hashes.append(resolution["palette_sha256"])
        resolutions.append(resolution)
        normalized_runs.append(
            {
                "source_run_spec": run_spec,
                "tank_colour_resolution": resolution,
            }
        )

    if len(all_recipe_ids) != len(set(all_recipe_ids)):
        raise ValueError("full-transition recipes must be globally unique")
    if len(all_case_seeds) != len(set(all_case_seeds)):
        raise ValueError("full-transition render seeds must be globally unique")
    if len(all_nuisance_ids) != len(set(all_nuisance_ids)):
        raise ValueError("full-transition nuisance IDs must be globally unique")
    if keep_count != 9 or discard_count != 9:
        raise ValueError("full-transition labels must be balanced 9 keep / 9 discard")
    if low_positive_count < 2:
        raise ValueError("full-transition profile needs at least two residues at or below 1.5 mL")
    if len(set(palette_hashes)) != 2:
        raise ValueError("full-transition runs require distinct preset palettes")
    for tank in ("t200", "t400"):
        if _preset_delta_e76(resolutions[0], resolutions[1], tank) < 10.0:
            raise ValueError("full-transition preset swatches are not visibly distinct")

    return {
        "schema_version": 1,
        "artifact_type": SOURCE_ARTIFACT_TYPE,
        "split_role": "test",
        "split_version": SPLIT_VERSION,
        "source_sha256": actual_hashes,
        "residue_profile_id": RESIDUE_PROFILE_ID,
        "visibility_gate": deepcopy(VISIBILITY_GATE),
        "runs": normalized_runs,
        "excluded_from_model_inputs": True,
        "claim_boundary": source["claim_boundary"],
    }


def _recursive_keys(value: object):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recursive_keys(child)


def _assert_model_input_boundary(model_plan: Mapping[str, object]) -> None:
    leaked = set(_recursive_keys(model_plan)) & _FORBIDDEN_MODEL_KEYS
    if leaked:
        raise ValueError(
            "full-transition model input contains Part-A truth: "
            + ",".join(sorted(leaked))
        )


def _validate_schedule_sources(
    run_spec: Mapping[str, object], render_plan: Mapping[str, object]
) -> None:
    recipes = {
        row["recipe_id"]: row["programmed_ml"] for row in run_spec["recipes"]
    }
    schedules = render_plan["execution_schedules"]
    for schedule in schedules:
        binding = schedule["case_binding"]
        current_id = binding["current_recipe_id"]
        previous_id = binding["previous_recipe_id"]
        residue = binding["residual_volume_ml"]
        order = list(active_dispense_order(recipes[current_id]))
        if schedule["physical_sequence"]["dispense_order"] != order:
            raise ValueError("full-transition schedule dispense order has drifted")
        previous_order = (
            list(active_dispense_order(recipes[previous_id]))
            if previous_id is not None
            else order
        )
        pours = schedule["pours"]
        if [pour["tank_id"] for pour in pours] != order:
            raise ValueError("full-transition pour order has drifted")
        for index, pour in enumerate(pours):
            expected_source = previous_order[-1] if index == 0 else order[index - 1]
            if pour["residual_source_tank_id"] != expected_source:
                raise ValueError("full-transition residual source chain has drifted")
            if pour["leading_residual_volume_ml"] != residue:
                raise ValueError("full-transition prescribed residue chain has drifted")


def _render_leakage_csv(
    run_records: list[tuple[str, Mapping[str, object]]]
) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=("path", "sim_recipe_id", "role", "seed"),
        lineterminator="\n",
    )
    writer.writeheader()
    for run_directory, render_plan in run_records:
        seed_by_case = {
            case_id: seed
            for case_id, seed in zip(
                render_plan["batches"][0]["case_ids"],
                render_plan["batches"][0]["render_seed_by_vial"],
                strict=True,
            )
        }
        for schedule in render_plan["execution_schedules"]:
            binding = schedule["case_binding"]
            case_id = binding["case_id"]
            for ordinal, tank in enumerate(
                schedule["physical_sequence"]["dispense_order"], start=1
            ):
                writer.writerow(
                    {
                        "path": (
                            f"{run_directory}/render-output/cases/{case_id}/"
                            f"app-output/stages/stage-{ordinal}-{tank}.png"
                        ),
                        "sim_recipe_id": binding["current_recipe_id"],
                        "role": "test",
                        "seed": seed_by_case[case_id],
                    }
                )
    return stream.getvalue()


def build_full_transition_artifacts(
    source: Mapping[str, object], *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Build two child artifact triples and their collection bindings."""

    normalized = validate_full_transition_source_spec(source, repo_root=repo_root)
    artifacts: dict[str, object] = {
        "full-transition-source-spec.json": deepcopy(normalized)
    }
    run_bindings = []
    run_records: list[tuple[str, Mapping[str, object]]] = []
    case_rows = []
    all_case_ids: list[str] = []
    all_labels = []
    for run_ordinal, run in enumerate(normalized["runs"], start=1):
        run_directory = f"run-{run_ordinal:02d}"
        bound = bind_tank_colour_resolution_to_artifacts(
            build_one_run_source_artifacts(run["source_run_spec"]),
            run["tank_colour_resolution"],
        )
        render_plan = bound["render-schedule-plan.json"]
        if rebuild_bound_one_run_artifacts(render_plan) != bound:
            raise ValueError("full-transition child plan differs from its rebuild")
        _assert_model_input_boundary(bound["model-input-plan.json"])
        _validate_schedule_sources(run["source_run_spec"], render_plan)
        for filename, value in bound.items():
            artifacts[f"{run_directory}/{filename}"] = value

        schedules = render_plan["execution_schedules"]
        case_ids = [row["case_binding"]["case_id"] for row in schedules]
        if any(case_id in all_case_ids for case_id in case_ids):
            raise ValueError("full-transition case IDs must be globally unique")
        all_case_ids.extend(case_ids)
        residues = [
            row["case_binding"]["residual_volume_ml"] for row in schedules
        ]
        highest_index = max(range(len(residues)), key=lambda index: residues[index])
        run_bindings.append(
            {
                "run_id": run["source_run_spec"]["run_id"],
                "recipe_ids": [
                    row["recipe_id"] for row in run["source_run_spec"]["recipes"]
                ],
                "output_root_relative": f"{run_directory}/render-output",
                "case_ids": case_ids,
                "render_plan_path": f"{run_directory}/render-schedule-plan.json",
                "render_plan_canonical_sha256": _canonical_sha256(render_plan),
                "model_input_plan_path": f"{run_directory}/model-input-plan.json",
                "model_input_plan_canonical_sha256": _canonical_sha256(
                    bound["model-input-plan.json"]
                ),
                "label_provenance_path": f"{run_directory}/label-provenance.json",
                "label_provenance_canonical_sha256": _canonical_sha256(
                    bound["label-provenance.json"]
                ),
                "resolved_preset_name": run["tank_colour_resolution"]["preset_name"],
                "resolved_preset_hash": run["tank_colour_resolution"][
                    "tank_colour_preset_hash"
                ],
                "highest_residue_case_id": case_ids[highest_index],
            }
        )
        run_records.append((run_directory, render_plan))
        all_labels.extend(bound["label-provenance.json"]["vial_labels"])
        case_rows.extend(
            {
                "case_id": case_id,
                "residue": residue,
                "run_ordinal": run_ordinal,
                "vial_ordinal": index + 1,
            }
            for index, (case_id, residue) in enumerate(
                zip(case_ids, residues, strict=True)
            )
        )

    keep_count = sum(row["expected_disposition"] == "keep" for row in all_labels)
    discard_count = len(all_labels) - keep_count
    low_positive_count = sum(
        0.0 < row["prescribed_leading_residual_volume_ml"] <= 1.5
        for row in all_labels
    )
    ordered_cases = sorted(
        case_rows,
        key=lambda row: (-row["residue"], row["run_ordinal"], row["vial_ordinal"]),
    )
    resolutions = [run["tank_colour_resolution"] for run in normalized["runs"]]
    preset_separation = {
        tank: round(_preset_delta_e76(resolutions[0], resolutions[1], tank), 6)
        for tank in ("t200", "t400")
    }
    collection = {
        "schema_version": 1,
        "artifact_type": COLLECTION_ARTIFACT_TYPE,
        "status": "unrendered",
        "source_spec_path": "full-transition-source-spec.json",
        "source_spec_canonical_sha256": _canonical_sha256(normalized),
        "split_role": "test",
        "split_version": SPLIT_VERSION,
        "source_sha256": deepcopy(normalized["source_sha256"]),
        "run_bindings": run_bindings,
        "render_priority_case_ids": [row["case_id"] for row in ordered_cases],
        "denominators": {
            "runs": 2,
            "vials": 18,
            "recipes": 6,
            "keep_vials": keep_count,
            "discard_vials": discard_count,
            "low_positive_residue_vials": low_positive_count,
        },
        "preset_separation_delta_e76": preset_separation,
        "visibility_gate": deepcopy(normalized["visibility_gate"]),
        "leakage_manifest_path": "leakage-manifest.csv",
        "excluded_from_model_inputs": True,
        "claim_boundary": (
            "synthetic unrendered collection planning only; child plans remain separate, "
            "and labels, residue, particles, and preset truth never enter model inputs"
        ),
    }
    if collection["denominators"] != {
        "runs": 2,
        "vials": 18,
        "recipes": 6,
        "keep_vials": 9,
        "discard_vials": 9,
        "low_positive_residue_vials": 2,
    }:
        raise ValueError("full-transition collection denominators have drifted")
    if any(value < 10.0 for value in preset_separation.values()):
        raise ValueError("full-transition collection preset separation is too small")
    artifacts["full-transition-manifest.json"] = collection
    artifacts["leakage-manifest.csv"] = _render_leakage_csv(run_records)
    return artifacts


def validate_full_transition_artifacts(
    artifacts: Mapping[str, object], *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Rebuild the complete collection and reject any child or binding mutation."""

    if not isinstance(artifacts, Mapping):
        raise TypeError("full-transition artifacts must be a mapping")
    source = artifacts.get("full-transition-source-spec.json")
    if not isinstance(source, Mapping):
        raise ValueError("full-transition source artifact is missing")
    expected = build_full_transition_artifacts(source, repo_root=repo_root)
    if dict(artifacts) != expected:
        raise ValueError("full-transition artifact set differs from its deterministic rebuild")
    return deepcopy(expected)


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)


def write_full_transition_bundle(
    source: Mapping[str, object],
    output_root: str | Path,
    *,
    repo_root: str | Path = REPO,
    owner_thread_id: str,
) -> Path:
    """Atomically publish an owner-tagged, no-clobber unrendered bundle."""

    if not isinstance(owner_thread_id, str) or not owner_thread_id.strip():
        raise ValueError("owner_thread_id must be non-empty text")
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"output root already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("output root parent must already exist")
    artifacts = build_full_transition_artifacts(source, repo_root=repo_root)
    validate_full_transition_artifacts(artifacts, repo_root=repo_root)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    try:
        _write_new(
            staging / "OWNER.txt",
            (
                f"owner_thread_id={owner_thread_id}\n"
                "purpose=WP1 full-transition source and unrendered plans\n"
                "state=unrendered\n"
            ),
        )
        for relative, value in artifacts.items():
            _write_new(staging / relative, value)
        if destination.exists():
            raise FileExistsError(f"output root appeared during publish: {destination}")
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


__all__ = [
    "COLLECTION_ARTIFACT_TYPE",
    "SOURCE_ARTIFACT_TYPE",
    "build_default_full_transition_source_spec",
    "build_full_transition_artifacts",
    "validate_full_transition_artifacts",
    "validate_full_transition_source_spec",
    "write_full_transition_bundle",
]

"""The minimum WP1 two-run manifest is source-bound and Part-B safe."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.cohorts.full_transition_manifest import (  # noqa: E402
    build_default_full_transition_source_spec,
    build_full_transition_artifacts,
    validate_full_transition_artifacts,
    validate_full_transition_source_spec,
    write_full_transition_bundle,
)
from twin.cohorts.one_run_manifest_v1 import (  # noqa: E402
    rebuild_bound_one_run_artifacts,
)


def _all_labels(artifacts):
    return [
        row
        for run in ("run-01", "run-02")
        for row in artifacts[f"{run}/label-provenance.json"]["vial_labels"]
    ]


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recursive_keys(child)


def test_default_source_is_two_clean_start_aaa_bbb_ccc_runs():
    source = build_default_full_transition_source_spec(repo_root=ROOT)
    assert validate_full_transition_source_spec(source, repo_root=ROOT) == source
    assert source["split_role"] == "test"
    assert source["split_version"] == "inn-7-splits-v3"
    assert source["excluded_from_model_inputs"] is True
    assert source["visibility_gate"] == {
        "measure_after_run_id": "WP1-TEST-RUN-01",
        "first_switch_vial_ordinal": 4,
        "same_recipe_comparison_vial_ordinals": [5, 6],
        "minimum_delta_e76": 2.0,
        "low_positive_residue_ml": 1.5,
        "fallback_residue_ml": 3.0,
        "fallback_target_vial_ordinals": [5, 9],
        "fallback_mode": "fresh_additional_cases_preserve_originals",
        "automatic_manifest_mutation": False,
    }
    assert [
        run["tank_colour_resolution"]["preset_name"] for run in source["runs"]
    ] == ["amber_teal", "crimson_indigo"]

    expected_residues = [
        [0.0, 0.0, 0.0, 9.0, 1.5, 0.0, 9.0, 4.5, 1.5],
        [0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 6.0, 3.0, 0.0],
    ]
    all_recipe_ids = []
    all_seeds = []
    for run, residues in zip(source["runs"], expected_residues, strict=True):
        spec = run["source_run_spec"]
        recipe_ids = [row["recipe_id"] for row in spec["recipes"]]
        assert len(recipe_ids) == len(set(recipe_ids)) == 3
        assert spec["vial_recipe_ids"] == [
            recipe_ids[0], recipe_ids[0], recipe_ids[0],
            recipe_ids[1], recipe_ids[1], recipe_ids[1],
            recipe_ids[2], recipe_ids[2], recipe_ids[2],
        ]
        assert spec["initial_condition"] == "clean_start"
        assert spec["initial_predecessor_recipe_id"] is None
        assert spec["residual_volume_ml_by_vial"] == residues
        assert residues[3] > 0 and residues[6] > 0
        assert any(0 in row["programmed_ml"].values() for row in spec["recipes"])
        all_recipe_ids.extend(recipe_ids)
        all_seeds.extend(spec["render_seed_by_vial"])
    assert len(set(all_recipe_ids)) == 6
    assert len(set(all_seeds)) == 18
    assert all(2_000_000 <= seed <= 2_999_999 for seed in all_seeds)


def test_builder_binds_two_rebuildable_runs_balanced_labels_and_leakage_csv(tmp_path):
    source = build_default_full_transition_source_spec(repo_root=ROOT)
    artifacts = build_full_transition_artifacts(source, repo_root=ROOT)
    assert validate_full_transition_artifacts(artifacts, repo_root=ROOT) == artifacts
    assert set(artifacts) == {
        "full-transition-source-spec.json",
        "full-transition-manifest.json",
        "run-01/render-schedule-plan.json",
        "run-01/model-input-plan.json",
        "run-01/label-provenance.json",
        "run-02/render-schedule-plan.json",
        "run-02/model-input-plan.json",
        "run-02/label-provenance.json",
        "leakage-manifest.csv",
    }
    for run in ("run-01", "run-02"):
        plan = artifacts[f"{run}/render-schedule-plan.json"]
        rebuilt = rebuild_bound_one_run_artifacts(plan)
        assert rebuilt["render-schedule-plan.json"] == plan
        assert rebuilt["model-input-plan.json"] == artifacts[
            f"{run}/model-input-plan.json"
        ]
        assert rebuilt["label-provenance.json"] == artifacts[
            f"{run}/label-provenance.json"
        ]

    manifest = artifacts["full-transition-manifest.json"]
    assert manifest["denominators"] == {
        "runs": 2,
        "vials": 18,
        "recipes": 6,
        "keep_vials": 9,
        "discard_vials": 9,
        "low_positive_residue_vials": 2,
    }
    assert manifest["preset_separation_delta_e76"]["t200"] >= 10
    assert manifest["preset_separation_delta_e76"]["t400"] >= 10
    assert len(manifest["run_bindings"]) == 2
    assert len({case for run in manifest["run_bindings"] for case in run["case_ids"]}) == 18
    assert manifest["render_priority_case_ids"][0] == manifest["run_bindings"][0][
        "highest_residue_case_id"
    ]

    labels = _all_labels(artifacts)
    assert sum(row["expected_disposition"] == "keep" for row in labels) == 9
    assert sum(row["expected_disposition"] == "discard" for row in labels) == 9
    assert all(
        (row["prescribed_leading_residual_volume_ml"] == 0)
        == (row["expected_disposition"] == "keep")
        for row in labels
    )
    forbidden = {
        "tank_colour_resolution", "run_palette", "palette", "preset",
        "residual_volume_ml", "prescribed_leading_residual_volume_ml",
        "expected_disposition", "label_provenance", "oracle", "particle_state",
    }
    for run in ("run-01", "run-02"):
        model = artifacts[f"{run}/model-input-plan.json"]
        assert forbidden.isdisjoint(_recursive_keys(model))

    leakage = tmp_path / "leakage-manifest.csv"
    leakage.write_text(artifacts["leakage-manifest.csv"])
    rows = list(csv.DictReader(io.StringIO(artifacts["leakage-manifest.csv"])))
    assert rows
    assert {row["role"] for row in rows} == {"test"}
    assert {row["sim_recipe_id"] for row in rows} == {
        "S0439", "S0440", "S0441", "S0442", "S0443", "S0444"
    }
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/fluidmix/core/check_leakage.py"),
            "--manifest", str(leakage),
            "--allow-missing",
            "--split-version", "v3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value["runs"][1]["source_run_spec"]["recipes"][0].update(
                recipe_id=value["runs"][0]["source_run_spec"]["recipes"][0]["recipe_id"]
            ),
            "unique|recipe",
        ),
        (
            lambda value: value["runs"][0]["source_run_spec"][
                "vial_recipe_ids"
            ].__setitem__(3, value["runs"][0]["source_run_spec"]["vial_recipe_ids"][0]),
            "AAA|sequence",
        ),
        (
            lambda value: value["runs"][0]["source_run_spec"][
                "residual_volume_ml_by_vial"
            ].__setitem__(3, 0.0),
            "switch|residue",
        ),
        (
            lambda value: value["runs"][1]["source_run_spec"][
                "render_seed_by_vial"
            ].__setitem__(0, value["runs"][0]["source_run_spec"]["render_seed_by_vial"][0]),
            "seed|unique",
        ),
        (
            lambda value: value["runs"][1].update(
                tank_colour_resolution=deepcopy(value["runs"][0]["tank_colour_resolution"])
            ),
            "preset|palette|distinct",
        ),
        (
            lambda value: value["source_sha256"].update(recipe_manifest="0" * 64),
            "hash|source",
        ),
        (
            lambda value: value["visibility_gate"].update(fallback_residue_ml=1.5),
            "fallback",
        ),
    ],
)
def test_source_validator_rejects_identity_sequence_label_and_gate_drift(mutator, match):
    source = build_default_full_transition_source_spec(repo_root=ROOT)
    mutator(source)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_full_transition_source_spec(source, repo_root=ROOT)


def test_artifact_validator_rejects_child_or_collection_mutation():
    artifacts = build_full_transition_artifacts(
        build_default_full_transition_source_spec(repo_root=ROOT), repo_root=ROOT
    )
    changed = deepcopy(artifacts)
    changed["run-01/label-provenance.json"]["vial_labels"][0][
        "expected_disposition"
    ] = "discard"
    with pytest.raises(ValueError, match="artifact|rebuild|differs"):
        validate_full_transition_artifacts(changed, repo_root=ROOT)


def test_source_validator_rejects_rehashed_unapproved_v3_file(tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("sim_recipes_v3.csv", "sim_split_spec_v3.json"):
        shutil.copy2(ROOT / "data" / name, tmp_path / "data" / name)
    recipe_path = tmp_path / "data/sim_recipes_v3.csv"
    with recipe_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    source = build_default_full_transition_source_spec(repo_root=ROOT)
    source["source_sha256"]["recipe_manifest"] = hashlib.sha256(
        recipe_path.read_bytes()
    ).hexdigest()
    for run in source["runs"]:
        run["source_run_spec"]["source_sha256"] = deepcopy(
            source["source_sha256"]
        )

    with pytest.raises(ValueError, match="approved|hash"):
        validate_full_transition_source_spec(source, repo_root=tmp_path)


def test_source_validator_rejects_legacy_test_recipe_substitution():
    source = build_default_full_transition_source_spec(repo_root=ROOT)
    with (ROOT / "data/sim_recipes_v3.csv").open(newline="", encoding="utf-8") as stream:
        legacy = next(
            row
            for row in csv.DictReader(stream)
            if row["role"] == "test" and row["sim_recipe_id"] not in {
                "S0439", "S0440", "S0441", "S0442", "S0443", "S0444"
            }
        )
    replacement_id = legacy["sim_recipe_id"]
    replacement_ml = {
        tank: int(legacy[f"{tank}_ml"] or 0)
        for tank in ("t100", "t200", "t400")
    }
    source["runs"][0]["source_run_spec"]["recipes"][0] = {
        "recipe_id": replacement_id,
        "programmed_ml": replacement_ml,
    }
    source["runs"][0]["source_run_spec"]["vial_recipe_ids"][:3] = [
        replacement_id,
        replacement_id,
        replacement_id,
    ]

    with pytest.raises(ValueError, match="generated|selection|WP1"):
        validate_full_transition_source_spec(source, repo_root=ROOT)


def test_writer_is_owner_tagged_atomic_and_no_clobber(tmp_path):
    output = tmp_path / "bundle"
    written = write_full_transition_bundle(
        build_default_full_transition_source_spec(repo_root=ROOT),
        output,
        repo_root=ROOT,
        owner_thread_id="01a18128-8dc4-7780-ace1-4d816b49d176",
    )
    assert written == output
    assert "01a18128-8dc4-7780-ace1-4d816b49d176" in (
        output / "OWNER.txt"
    ).read_text()
    loaded = {
        relative: (
            (output / relative).read_text()
            if relative.endswith(".csv")
            else json.loads((output / relative).read_text())
        )
        for relative in (
            "full-transition-source-spec.json",
            "full-transition-manifest.json",
            "run-01/render-schedule-plan.json",
            "run-01/model-input-plan.json",
            "run-01/label-provenance.json",
            "run-02/render-schedule-plan.json",
            "run-02/model-input-plan.json",
            "run-02/label-provenance.json",
            "leakage-manifest.csv",
        )
    }
    assert validate_full_transition_artifacts(loaded, repo_root=ROOT) == loaded
    with pytest.raises(FileExistsError):
        write_full_transition_bundle(
            build_default_full_transition_source_spec(repo_root=ROOT),
            output,
            repo_root=ROOT,
            owner_thread_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        )

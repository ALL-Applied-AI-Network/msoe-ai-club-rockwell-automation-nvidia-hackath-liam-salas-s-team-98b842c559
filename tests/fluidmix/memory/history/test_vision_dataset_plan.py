"""Tests for the separated reference-bank synthetic vision artifacts."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.memory.history.vision_dataset_plan import (  # noqa: E402
    build_vision_artifacts,
    select_peer_reference,
    validate_vision_config,
)

CONFIG_PATH = ROOT / "data" / "twin_vision_dataset_v1.json"
PROFILE_VALUES = {"none", "first_only", "first_two", "all_three"}
ONSET_VALUES = {"after_t200", "after_t400", "after_t100"}
ORACLE_KEYS = {
    "synthetic_profile_id",
    "appearance_blend_alpha",
    "synthetic_onset_id",
    "clean_stage_linear_rgb",
    "observed_stage_linear_rgb",
    "semantic_aov",
}


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_config_freezes_separate_balanced_reference_and_target_cohorts():
    config = _config()
    validate_vision_config(config)
    assert config["alpha_profiles"] == {
        "none": [0.0, 0.0, 0.0],
        "first_only": [0.30, 0.0, 0.0],
        "first_two": [0.30, 0.12, 0.0],
        "all_three": [0.30, 0.18, 0.08],
    }
    assert config["cohorts"] == ["reference_only", "scored_target"]
    assert config["reference_contract"]["minimum_distinct_batch_ids"] == 2
    assert config["reference_contract"]["primary_mode"] == "all_reference_batches"
    assert config["partition_policy"]["test_evaluation_scope"] == "reference_assisted_transductive_not_pure_recipe_generalization"


def test_config_validator_rejects_cohort_or_seed_drift():
    config = _config()
    config["cohorts"] = ["scored_target"]
    with pytest.raises(ValueError, match="cohort"):
        validate_vision_config(config)
    config = _config()
    config["recipe_families"][0]["seed_base"] = 1310000
    with pytest.raises(ValueError, match="recipe family"):
        validate_vision_config(config)


def test_render_plan_binds_local_isaac62_watchdog_and_keeps_brev601_distinct():
    config = _config()
    assert config["render_runtime"] == {
        "host_name": "spark5",
        "machine": "aarch64",
        "runtime_name": "NVIDIA Isaac Sim",
        "isaac_sim_version": "6.2.0.0",
        "kit_version": "110.4.0",
        "isaac_python": "/home/hackathon5/liamreporoot/isaacsim62-env/.venv/bin/python",
        "environment_profile": "gb10-aarch64-isaac62",
    }
    assert config["process_attempt_policy"] == {
        "max_process_attempts": 2,
        "no_output_timeout_seconds": 180,
        "startup_readiness_timeout_seconds": 420,
        "fast_crash_retry_seconds": 180,
        "startup_readiness_marker": "Simulation App Startup Complete",
    }
    assert config["future_brev_runtime"]["isaac_sim_version"] == "6.0.1.0"
    assert config["runtime_mixing_policy"] == "one_artifact_set_one_host_and_runtime_profile_never_mix"


def test_artifacts_have_exact_denominators_and_non_circular_cohorts():
    artifacts = build_vision_artifacts(CONFIG_PATH)
    render = artifacts["render-plan.json"]
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance-plan.json"]
    assert set(artifacts) == {"render-plan.json", "model-input-plan.json", "label-provenance-plan.json"}
    assert render["denominators"] == {
        "recipe_families": 4,
        "total_batches": 32,
        "reference_only_batches": 16,
        "scored_target_batches": 16,
        "vials_and_render_launches": 96,
        "stage_images_planned": 288,
    }
    assert model["denominators"] == {
        "scored_target_decisions": 48,
        "reference_availability_target_stage_records": 864,
        "cosmos_model_calls_scored": 144,
        "template_controls_scored": 48,
        "scored_attempts_total": 192,
        "multi_image_smoke_unscored": 1,
    }
    assert len(render["batches"]) == 32
    assert len(render["render_launches"]) == 96
    assert len(render["images"]) == 288
    assert len(model["decision_records"]) == 48
    assert len(model["cosmos_attempts"]) == 192
    assert len(labels["image_labels"]) == 288
    assert all(batch["cohort"] == "scored_target" for batch in model["target_batches"])
    assert all(batch["cohort"] == "reference_only" for batch in model["reference_batches"])
    assert all(
        launch["max_process_attempts"] == 2
        and launch["attempt_limit"] == 2
        and launch["replacement_attempts"] == 1
        and launch["process_attempt_policy"] == render["process_attempt_policy"]
        for launch in render["render_launches"]
    )


def test_each_family_cohort_has_balanced_profiles_and_matched_seed_vectors():
    render = build_vision_artifacts(CONFIG_PATH)["render-plan.json"]
    batches_by_group = defaultdict(list)
    for batch in render["batches"]:
        batches_by_group[(batch["family_id"], batch["cohort"])].append(batch)
    family_seed_sets = {}
    for family_id in {batch["family_id"] for batch in render["batches"]}:
        reference = batches_by_group[(family_id, "reference_only")]
        targets = batches_by_group[(family_id, "scored_target")]
        assert {batch["synthetic_profile_id"] for batch in reference} == PROFILE_VALUES
        assert {batch["synthetic_profile_id"] for batch in targets} == PROFILE_VALUES
        reference_vectors = {tuple(batch["render_seed_by_vial"]) for batch in reference}
        target_vectors = {tuple(batch["render_seed_by_vial"]) for batch in targets}
        assert len(reference_vectors) == len(target_vectors) == 1
        assert reference_vectors.isdisjoint(target_vectors)
        family_seed_sets[family_id] = set(next(iter(reference_vectors))) | set(next(iter(target_vectors)))
    family_ids = sorted(family_seed_sets)
    for left_index, left in enumerate(family_ids):
        for right in family_ids[left_index + 1 :]:
            assert family_seed_sets[left].isdisjoint(family_seed_sets[right])


def test_reference_cases_use_separate_bank_all_positions_and_never_target_batches():
    model = build_vision_artifacts(CONFIG_PATH)["model-input-plan.json"]
    target_batch_id = next(
        batch["batch_id"]
        for batch in model["target_batches"]
        if batch["family_id"] == "TR"
    )
    decision = next(record for record in model["decision_records"] if record["batch_id"] == target_batch_id and record["vial_id"] == "V1")
    cases = decision["stages"][0]["reference_cases"]
    assert len(cases) == 6
    assert [case["mode"] for case in cases] == [
        "same_batch_only",
        "one_reference_batch",
        "one_reference_batch",
        "one_reference_batch",
        "one_reference_batch",
        "all_reference_batches",
    ]
    primary = cases[-1]
    assert primary["disposition_source"] is True
    assert len(primary["candidate_peers"]) == 14
    external = [peer for peer in primary["candidate_peers"] if peer["origin"] == "reference_bank"]
    assert len(external) == 12
    assert {peer["vial_id"] for peer in external} == {"V1", "V2", "V3"}
    reference_ids = {batch["batch_id"] for batch in model["reference_batches"]}
    target_ids = {batch["batch_id"] for batch in model["target_batches"]}
    assert all(peer["batch_id"] in reference_ids and peer["batch_id"] not in target_ids for peer in external)
    assert all(peer["current_sim_recipe_id"] == "S0117" for peer in external)
    assert cases[0]["expected_validity"] == "insufficient_evidence"
    assert all(case["alternative_id"] is not None for case in cases[1:5])


def test_one_reference_alternatives_are_all_evaluated_without_winner_and_v3_is_thin():
    model = build_vision_artifacts(CONFIG_PATH)["model-input-plan.json"]
    target_batch_id = next(
        batch["batch_id"]
        for batch in model["target_batches"]
        if batch["family_id"] == "TR"
    )
    v2 = next(record for record in model["decision_records"] if record["batch_id"] == target_batch_id and record["vial_id"] == "V2")
    v3 = next(record for record in model["decision_records"] if record["batch_id"] == target_batch_id and record["vial_id"] == "V3")
    one_v2 = [case for case in v2["stages"][0]["reference_cases"] if case["mode"] == "one_reference_batch"]
    one_v3 = [case for case in v3["stages"][0]["reference_cases"] if case["mode"] == "one_reference_batch"]
    assert len(one_v2) == len(one_v3) == 4
    assert {case["alternative_id"] for case in one_v2} == {
        batch["batch_id"]
        for batch in model["reference_batches"]
        if batch["family_id"] == "TR"
    }
    assert all(len(case["candidate_peers"]) == 4 for case in one_v2)
    assert all(len(case["candidate_peers"]) == 3 and case["expected_validity"] == "insufficient_evidence" for case in one_v3)
    assert not any("winner" in key for case in one_v2 for key in case)


def test_reference_selection_requires_unique_tight_cross_batch_mode():
    mask = "a" * 64
    candidates = {
        "R1-V1": {"lab": [49.9, 1, 2], "batch_id": "R1", "origin": "reference_bank", "mask_sha256": mask},
        "R2-V2": {"lab": [50.0, 1, 2], "batch_id": "R2", "origin": "reference_bank", "mask_sha256": mask},
        "R3-V3": {"lab": [50.1, 1, 2], "batch_id": "R3", "origin": "reference_bank", "mask_sha256": mask},
    }
    reference = select_peer_reference(candidates, target_mask_sha256=mask)
    assert reference["validity"] == "valid"
    assert reference["peer_batch_count"] == 3
    assert reference["representative_peer_id"] == "R2-V2"
    tied = {
        "A": {"lab": [0, 0, 0], "batch_id": "A", "origin": "reference_bank", "mask_sha256": mask},
        "B": {"lab": [0.1, 0, 0], "batch_id": "B", "origin": "reference_bank", "mask_sha256": mask},
        "C": {"lab": [10, 0, 0], "batch_id": "C", "origin": "reference_bank", "mask_sha256": mask},
        "D": {"lab": [10.1, 0, 0], "batch_id": "D", "origin": "reference_bank", "mask_sha256": mask},
    }
    assert select_peer_reference(tied, target_mask_sha256=mask)["validity"] == "insufficient_evidence"


def test_model_input_artifact_recursively_excludes_oracles_and_profile_values():
    artifacts = build_vision_artifacts(CONFIG_PATH)
    model = artifacts["model-input-plan.json"]
    walked = list(_walk(model))
    assert ORACLE_KEYS.isdisjoint({value for value in walked if isinstance(value, str)})
    assert PROFILE_VALUES.isdisjoint({value for value in walked if isinstance(value, str)})
    assert ONSET_VALUES.isdisjoint({value for value in walked if isinstance(value, str)})
    batch_ids = [
        batch["batch_id"]
        for batch in model["target_batches"] + model["reference_batches"]
    ]
    assert len(batch_ids) == len(set(batch_ids)) == 32
    assert all(re.fullmatch(r"BATCH_[0-9A-F]{12}", batch_id) for batch_id in batch_ids)
    assert all(
        not any(token in batch_id for token in (*PROFILE_VALUES, *ONSET_VALUES, "TR", "VA", "TE", "R01", "T01"))
        for batch_id in batch_ids
    )
    assert any(key in json.dumps(artifacts["label-provenance-plan.json"]) for key in ORACLE_KEYS)


def test_test_scope_is_explicitly_reference_assisted_transductive():
    model = build_vision_artifacts(CONFIG_PATH)["model-input-plan.json"]
    policy = model["partition_policy"]
    assert policy["test_data_used_for_training_or_tuning"] is False
    assert policy["test_evaluation_scope"] == "reference_assisted_transductive_not_pure_recipe_generalization"
    assert model["decision_timing"] == "retrospective_after_complete_three_vial_batch_no_latency_claim"


def test_cli_creates_only_three_artifacts_in_new_external_directory(tmp_path):
    script = ROOT / "cli/fluidmix/memory/history" / "build_twin_vision_plan.py"
    help_result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert help_result.returncode == 0
    assert "--output-dir" in help_result.stdout
    output = tmp_path / "vision-plan"
    result = subprocess.run([sys.executable, str(script), "--output-dir", str(output)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {"render-plan.json", "model-input-plan.json", "label-provenance-plan.json"}
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    repeated = subprocess.run([sys.executable, str(script), "--output-dir", str(output)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert repeated.returncode != 0
    assert original == {path.name: path.read_bytes() for path in output.iterdir()}

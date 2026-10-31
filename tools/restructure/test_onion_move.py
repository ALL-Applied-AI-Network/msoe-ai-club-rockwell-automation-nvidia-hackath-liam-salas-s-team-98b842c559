import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).with_name("onion_move.py")
SPEC = importlib.util.spec_from_file_location("onion_move", TOOL)
onion_move = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(onion_move)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    )


def write(repo, relative, text=""):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_repo(tmp_path, files, mapping):
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative, text in files.items():
        write(repo, relative, text)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "base")
    return repo, mapping_path


def invoke(repo, mapping, *args):
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(repo), "--mapping", str(mapping), *args],
        text=True,
        capture_output=True,
    )


def test_mapping_translation_routes_leaf_directory_cli_and_test(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/deploy_input_policy.py": "POLICY = 1\n",
            "src/fluidmix/paths.py": "DATA = 1\n",
            "src/fluidmix/wp7_final_decision.py": "VALUE = 1\n",
            "src/fluidmix/stage_decision.py": "",
            "src/fluidmix/f2_varied_triage.py": "",
            "scripts/decide_wp7_varied_triads.py": "",
            "tests/test_wp7_final_decision.py": "",
        },
        {
            "core/policy": "deploy_input_policy paths",
            "decision": "wp7_final_decision stage_decision f2_varied_triage",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "src/fluidmix/core/policy.py").is_file()
    assert (repo / "src/fluidmix/core/paths.py").is_file()
    assert (repo / "src/fluidmix/decision/vial.py").is_file()
    assert (repo / "src/fluidmix/decision/stage.py").is_file()
    assert (repo / "src/fluidmix/decision/triage.py").is_file()
    assert (repo / "cli/fluidmix/decision/decide_wp7_varied_triads.py").is_file()
    assert (repo / "tests/fluidmix/decision/test_vial.py").is_file()
    assert not (repo / "src/fluidmix/core/policy/policy.py").exists()


def test_task_2_evaluation_preparation_and_agent_docs_routes_replay(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/wp7_evaluation_only_preparation.py": (
                "from fluidmix.wp7_evaluation_only_preparation import run\n"
            ),
            "scripts/prepare_wp7_evaluation_only.py": "",
            "scripts/check_agent_docs.py": "",
        },
        {
            "pipeline": "wp7_evaluation_only_preparation",
            "core/policy": "check_agent_docs",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "src/fluidmix/pipeline/evaluation_preparation.py").is_file()
    assert (repo / "cli/fluidmix/pipeline/prepare_evaluation_only.py").is_file()
    assert (repo / "cli/fluidmix/core/check_agent_docs.py").is_file()
    assert "from fluidmix.pipeline.evaluation_preparation import run" in (
        repo / "src/fluidmix/pipeline/evaluation_preparation.py"
    ).read_text()

    replay = invoke(repo, mapping)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert " -> " not in replay.stdout


def test_task_2_routes_are_explicit_tables():
    assert onion_move.SOURCE_ROUTES["wp7_evaluation_only_preparation.py"] == "pipeline"
    assert onion_move.RENAMES["wp7_evaluation_only_preparation.py"] == (
        "evaluation_preparation.py"
    )
    assert onion_move.SCRIPT_ROUTES["prepare_wp7_evaluation_only.py"] == "pipeline"
    assert onion_move.SCRIPT_ROUTES["check_agent_docs.py"] == "core/policy"
    assert onion_move.SCRIPT_RENAMES["prepare_wp7_evaluation_only.py"] == (
        "prepare_evaluation_only.py"
    )


def test_task_2_source_route_is_active_without_mapping_member(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/wp7_evaluation_only_preparation.py": (
                "from fluidmix.wp7_evaluation_only_preparation import run\n"
            ),
        },
        {"pipeline": "unlisted_pipeline_member"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    destination = repo / "src/fluidmix/pipeline/evaluation_preparation.py"
    assert destination.is_file()
    assert "from fluidmix.pipeline.evaluation_preparation import run" in destination.read_text()
    replay = invoke(repo, mapping)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert " -> " not in replay.stdout


def test_import_rewrite_changes_only_imports_and_uses_full_paths(tmp_path):
    source = """# fluidmix.wp7_final_decision stays in a comment
from fluidmix.wp7_final_decision import decide as choose
import fluidmix.stage_decision as stage
from .wp7_final_decision import (decide as relative_decide,)
TEXT = "fluidmix.wp7_final_decision"
runtime_identifier = fluidmix.wp7_final_decision
"""
    repo, mapping = make_repo(
        tmp_path,
        {"src/fluidmix/wp7_final_decision.py": source, "src/fluidmix/stage_decision.py": ""},
        {"decision": "wp7_final_decision stage_decision"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "src/fluidmix/decision/vial.py").read_text()
    assert "from fluidmix.decision.vial import decide as choose" in rewritten
    assert "import fluidmix.decision.stage as stage" in rewritten
    assert "from .vial import (decide as relative_decide,)" in rewritten
    assert "# fluidmix.wp7_final_decision stays in a comment" in rewritten
    assert 'TEXT = "fluidmix.wp7_final_decision"' in rewritten
    assert "runtime_identifier = fluidmix.wp7_final_decision" in rewritten


def test_import_rewrite_handles_package_submodule_aliases(tmp_path):
    source = """# from fluidmix import paths stays in a comment
from fluidmix import paths as absolute_paths
from . import paths as relative_paths
from fluidmix.paths import DATA as imported_symbol
TEXT = "from fluidmix import paths"
"""
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/paths.py": "DATA = 1\n",
            "src/fluidmix/policy_consumer.py": source,
        },
        {"core/policy": "paths policy_consumer"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "src/fluidmix/core/policy_consumer.py").read_text()
    assert "from fluidmix.core import paths as absolute_paths" in rewritten
    assert "from . import paths as relative_paths" in rewritten
    assert "from fluidmix.core.paths import DATA as imported_symbol" in rewritten
    assert "# from fluidmix import paths stays in a comment" in rewritten
    assert 'TEXT = "from fluidmix import paths"' in rewritten
    replay = invoke(repo, mapping)
    assert replay.returncode == 0, replay.stdout + replay.stderr


def test_perception_mask_responsibilities_use_distinct_explicit_names(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/rgb_mask_prediction.py": "",
            "src/fluidmix/predicted_mask_materializer.py": "",
            "tests/test_rgb_mask_prediction.py": "",
            "tests/test_predicted_mask_materializer.py": "",
        },
        {"perception/masks": "rgb_mask_prediction predicted_mask_materializer"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "src/fluidmix/perception/masks.py").is_file()
    assert (repo / "src/fluidmix/perception/materializer.py").is_file()
    assert (repo / "tests/fluidmix/perception/test_masks.py").is_file()
    assert (repo / "tests/fluidmix/perception/test_materializer.py").is_file()
    replay = invoke(repo, mapping)
    assert replay.returncode == 0, replay.stdout + replay.stderr


def test_representative_current_inventory_has_complete_explicit_routes(tmp_path):
    representative_script_table = {
        "audit_cnn_holdout_recovery.py": "perception/cnn",
        "audit_sixth_development_reads.py": "memory/history",
        "build_f2_varied_stage_measurements.py": "measurement",
        "materialize_empty_reference_predictions.py": "perception/masks",
        "build_physx_case_plan.py": "sim/physx",
        "run_stage_cosmos_nano_eval.py": "explanation/vlm",
        "score_v3_cosmos_nano.py": "evaluation/scoring",
    }
    representative_test_table = {
        "fixtures/inn67-materialization-protocol-amendment.json": (
            "memory/reference_bank", "fixtures/inn67-materialization-protocol-amendment.json"
        ),
        "test_build_stage_measured_evidence.py": (
            "measurement", "test_build_stage_measured_evidence.py"
        ),
        "test_carryover_pipeline_final_wiring.py": (
            "pipeline", "test_carryover_pipeline_final_wiring.py"
        ),
        "test_f2_varied_triage_cli.py": ("decision", "test_triage_cli.py"),
        "test_wp7_development_reference_store.py": (
            "memory/history", "test_development_reference_store.py"
        ),
        "test_wp7_route_preflight.py": ("pipeline", "test_route_preflight.py"),
        "test_twin_contact_sheet.py": ("sim/render", "test_contact_sheet.py"),
    }
    assert {
        name: onion_move.SCRIPT_ROUTES.get(name) for name in representative_script_table
    } == representative_script_table
    assert {
        name: onion_move.TEST_ROUTES.get(name) for name in representative_test_table
    } == representative_test_table
    assert onion_move.RENAMES["wp7_final_decision.py"] == "vial.py"

    script_routes = {
        "audit_cnn_holdout_recovery.py": "cli/fluidmix/perception/cnn",
        "audit_sixth_development_reads.py": "cli/fluidmix/memory/history",
        "build_f2_varied_stage_measurements.py": "cli/fluidmix/measurement",
        "materialize_empty_reference_predictions.py": "cli/fluidmix/perception",
        "build_physx_case_plan.py": "cli/twin/physx",
        "run_stage_cosmos_nano_eval.py": "cli/fluidmix/explanation/vlm",
        "score_v3_cosmos_nano.py": "cli/evaluation/scoring",
    }
    test_routes = {
        "fixtures/inn67-materialization-protocol-amendment.json": (
            "tests/fluidmix/memory/reference_bank/fixtures/"
            "inn67-materialization-protocol-amendment.json"
        ),
        "test_build_stage_measured_evidence.py": (
            "tests/fluidmix/measurement/test_build_stage_measured_evidence.py"
        ),
        "test_wp7_final_decision.py": "tests/fluidmix/decision/test_vial.py",
        "test_carryover_pipeline_final_wiring.py": (
            "tests/fluidmix/pipeline/test_carryover_pipeline_final_wiring.py"
        ),
        "test_f2_varied_triage_cli.py": "tests/fluidmix/decision/test_triage_cli.py",
        "test_wp7_development_reference_store.py": (
            "tests/fluidmix/memory/history/test_development_reference_store.py"
        ),
        "test_wp7_route_preflight.py": "tests/fluidmix/pipeline/test_route_preflight.py",
        "test_twin_contact_sheet.py": "tests/twin/render/test_contact_sheet.py",
    }
    files = {
        **{f"scripts/{name}": "" for name in script_routes},
        **{f"tests/{name}": "" for name in test_routes},
        "src/fluidmix/cnn_harness_recovery.py": "",
        "src/fluidmix/f2_varied_stage_measurement.py": "",
        "src/fluidmix/predicted_mask_materializer.py": "",
        "src/fluidmix/physx_case_plan.py": "",
        "src/fluidmix/stage_cosmos_runner.py": "",
        "src/fluidmix/v3_cosmos_scoring.py": "",
        "src/fluidmix/wp7_final_decision.py": "",
        "src/fluidmix/carryover_pipeline.py": "",
        "src/fluidmix/f2_varied_triage.py": "",
        "src/fluidmix/wp7_development_reference.py": "",
    }
    repo, mapping = make_repo(
        tmp_path,
        files,
        {
            "perception/cnn": "cnn_harness_recovery",
            "perception/masks": "predicted_mask_materializer",
            "measurement": "f2_varied_stage_measurement",
            "decision": "wp7_final_decision f2_varied_triage",
            "explanation/vlm": "stage_cosmos_runner",
            "memory/history": "wp7_development_reference",
            "memory/reference_bank": "inn67_reference_bank",
            "pipeline": "carryover_pipeline",
            "sim/physx": "physx_case_plan",
            "sim/render": "render_process",
            "evaluation/scoring": "v3_cosmos_scoring",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNMAPPED" not in result.stdout
    for filename, directory in script_routes.items():
        assert (repo / directory / filename).is_file()
    for filename, destination in test_routes.items():
        assert (repo / destination).is_file(), filename


def test_contracts_move_recursively_and_unmapped_has_every_extension(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/contracts/__init__.py": "",
            "src/fluidmix/contracts/nested/schema.json": "{}",
            "src/fluidmix/unknown.toml": "x=1",
            "scripts/unknown.html": "<p>x</p>",
            "tests/fixture.csv": "a,b\n",
        },
        {"core/contracts": "contracts"},
    )
    before = git(repo, "status", "--porcelain").stdout
    report_only = invoke(repo, mapping, "--report-only")
    assert report_only.returncode == 0
    assert "UNMAPPED" in report_only.stdout
    assert git(repo, "status", "--porcelain").stdout == before
    report = invoke(repo, mapping, "--dry-run")
    assert report.returncode == 2
    assert "src/fluidmix/contracts/__init__.py -> src/fluidmix/core/contracts/__init__.py" in report.stdout
    assert "src/fluidmix/contracts/nested/schema.json -> src/fluidmix/core/contracts/nested/schema.json" in report.stdout
    assert "src/fluidmix/unknown.toml" in report.stdout
    assert "scripts/unknown.html" in report.stdout
    assert "tests/fixture.csv" in report.stdout


def test_report_only_never_mutates_and_normal_replay_is_empty_noop(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {"src/fluidmix/wp7_final_decision.py": "VALUE = 1\n"},
        {"decision": "wp7_final_decision"},
    )
    before = git(repo, "status", "--porcelain").stdout
    report = invoke(repo, mapping, "--report-only")
    assert report.returncode == 0
    assert git(repo, "status", "--porcelain").stdout == before
    assert (repo / "src/fluidmix/wp7_final_decision.py").exists()
    first = invoke(repo, mapping)
    second = invoke(repo, mapping)
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert " -> " not in second.stdout
    assert "UNMAPPED" not in second.stdout
    (repo / "src/fluidmix/decision/vial.py").write_text("unrelated edit\n")
    unrelated = invoke(repo, mapping)
    assert unrelated.returncode != 0
    assert "dirty" in unrelated.stderr.lower()


def test_apply_refuses_unrelated_dirty_worktree(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {"src/fluidmix/wp7_final_decision.py": "VALUE = 1\n", "notes.txt": "clean\n"},
        {"decision": "wp7_final_decision"},
    )
    (repo / "notes.txt").write_text("dirty\n")
    result = invoke(repo, mapping)
    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()
    assert (repo / "src/fluidmix/wp7_final_decision.py").exists()


def test_empty_plan_apply_still_refuses_unrelated_dirty_worktree(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {"src/fluidmix/decision/vial.py": "VALUE = 1\n", "notes.txt": "clean\n"},
        {"decision": "wp7_final_decision"},
    )
    (repo / "notes.txt").write_text("dirty\n")
    result = invoke(repo, mapping)
    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()
    assert (repo / "src/fluidmix/decision/vial.py").is_file()


@pytest.mark.parametrize("key", ["../escape", "decision/../../escape", "/absolute"])
def test_rejects_malicious_mapping_before_creating_any_directory(tmp_path, key):
    repo, mapping = make_repo(
        tmp_path,
        {"src/fluidmix/wp7_final_decision.py": ""},
        {key: "wp7_final_decision"},
    )
    result = invoke(repo, mapping)
    assert result.returncode != 0
    assert "unsafe mapping key" in result.stderr.lower()
    assert not (tmp_path / "escape").exists()


def test_collision_is_rejected_before_any_move_or_mkdir(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {"src/fluidmix/wp7_final_decision.py": "", "src/fluidmix/decision/vial.py": "occupied\n"},
        {"decision": "wp7_final_decision"},
    )
    result = invoke(repo, mapping)
    assert result.returncode != 0
    assert "collision" in result.stderr.lower()
    assert (repo / "src/fluidmix/wp7_final_decision.py").exists()
    assert (repo / "src/fluidmix/decision/vial.py").read_text() == "occupied\n"


def test_default_mapping_resolves_to_read_only_parent_workspace():
    expected = TOOL.resolve().parents[2].parent / "overnight-ops-20261028" / "ONION-MAPPING.json"
    assert onion_move.DEFAULT_MAPPING == expected
    assert onion_move.DEFAULT_MAPPING.is_file()

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


def with_semantic_patterns(relative, source):
    """Complete a minimal fixture without changing its scenario-specific snippets."""
    missing_patterns = []
    for old, _, expected_count in onion_move.SEMANTIC_REWRITES.get(relative, ()):
        missing = expected_count - source.count(old)
        assert missing >= 0, (relative, old)
        missing_patterns.extend(old for _ in range(missing))
    if relative.endswith(".json"):
        assert not source.strip()
        return json.dumps(missing_patterns)
    for old in missing_patterns:
        assert "\n" not in old, (relative, old)
        source += f"\n# semantic fixture: {old}"
    return source


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


def test_final_runner_service_routes_existing_tree_and_tests(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/pipeline/__init__.py": "",
            "src/fluidmix/pipeline/service/__init__.py": "",
            "src/fluidmix/pipeline/service/accepted_adapter.py": "",
            "src/fluidmix/pipeline/service/production_factory.py": "",
            "tests/fluidmix/pipeline/service/test_production_factory.py": "",
            "tests/test_accepted_adapter_tracker_observer.py": "",
            "tests/test_runner_vlm_lineage_integration.py": "",
        },
        {"pipeline": "carryover_pipeline"},
    )
    result = invoke(repo, mapping, "--report-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNMAPPED" not in result.stdout


def test_final_tracker_routes_camera_measurement_and_pipeline_adapters(tmp_path):
    files = {
        "src/fluidmix/empty_reference_timing_adapter.py": "",
        "src/fluidmix/wp7_fill_step_tracker.py": "",
        "src/fluidmix/wp7_fill_step_production.py": "",
        "src/fluidmix/wp7_tracker_runtime.py": "",
        "src/fluidmix/wp7_fill_step_calibration.py": "",
        "src/fluidmix/wp7_cohort_preparation.py": "",
        "src/fluidmix/wp7_phase_capture.py": "",
        "src/fluidmix/wp7_tracker_receipt_adapter.py": "",
        "scripts/prepare_wp7_cohort.py": "",
        "scripts/prepare_wp7_development_reference_route.py": "",
        "scripts/publish_wp7_fill_step_calibration.py": "",
        "tests/test_wp7_fill_step_tracker.py": "",
        "tests/test_wp7_fill_step_calibration.py": "",
        "tests/test_wp7_cohort_preparation.py": "",
        "tests/test_wp7_tracker_receipt_adapter.py": "",
    }
    repo, mapping = make_repo(
        tmp_path,
        files,
        {
            "perception/baselines": "empty_reference_rgb_predictor",
            "perception/masks": "rgb_mask_prediction",
            "perception/tracker": "unlisted_tracker_member",
            "measurement": "stage_measurement",
            "pipeline": "carryover_pipeline",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        "src/fluidmix/perception/baselines/timing_adapter.py",
        "src/fluidmix/perception/tracker/fill_step_tracker.py",
        "src/fluidmix/perception/tracker/fill_step_production.py",
        "src/fluidmix/perception/tracker/tracker_runtime.py",
        "src/fluidmix/measurement/fill_step_calibration.py",
        "src/fluidmix/pipeline/cohort_preparation.py",
        "src/fluidmix/pipeline/phase_capture.py",
        "src/fluidmix/pipeline/tracker_receipt_adapter.py",
        "cli/fluidmix/pipeline/prepare_wp7_cohort.py",
        "cli/fluidmix/pipeline/prepare_wp7_development_reference_route.py",
        "cli/fluidmix/measurement/publish_wp7_fill_step_calibration.py",
        "tests/fluidmix/perception/tracker/test_fill_step_tracker.py",
        "tests/fluidmix/measurement/test_fill_step_calibration.py",
        "tests/fluidmix/pipeline/test_cohort_preparation.py",
        "tests/fluidmix/pipeline/test_tracker_receipt_adapter.py",
    }
    assert {path for path in expected if (repo / path).is_file()} == expected


def test_final_vlm_runtime_and_decision_routes_follow_responsibility(tmp_path):
    files = {
        "src/fluidmix/wp7_stock_runtime.py": "",
        "src/fluidmix/wp7_vlm_constrained.py": "",
        "src/fluidmix/wp7_vlm_image_support.py": "",
        "src/fluidmix/wp7_vlm_development_materializer.py": "",
        "src/fluidmix/wp7_vlm_native_roi_rematerializer.py": "",
        "src/fluidmix/wp7_vlm_production_materializer.py": "",
        "src/fluidmix/wp7_vlm_decision.py": "",
        "scripts/materialize_wp7_vlm_development_bundle.py": "",
        "scripts/rematerialize_wp7_vlm_native_roi.py": "",
        "scripts/run_wp7_vlm_model.py": "",
        "scripts/aggregate_wp7_vlm_attempts.py": "",
        "scripts/wp7_vlm_decision.py": "",
        "tests/test_wp7_vlm_development_materializer.py": "",
        "tests/test_wp7_vlm_production_materializer.py": "",
        "tests/test_wp7_vlm_decision.py": "",
        "tests/test_wp7_vlm_reasoning_upgrade.py": "",
    }
    repo, mapping = make_repo(
        tmp_path,
        files,
        {
            "explanation/vlm": "cosmos_nano_runner",
            "decision": "wp7_final_decision",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        "src/fluidmix/explanation/vlm/stock_runtime.py",
        "src/fluidmix/explanation/vlm/constrained.py",
        "src/fluidmix/explanation/vlm/image_support.py",
        "src/fluidmix/explanation/vlm/development_materializer.py",
        "src/fluidmix/explanation/vlm/native_roi_rematerializer.py",
        "src/fluidmix/explanation/vlm/production_materializer.py",
        "src/fluidmix/decision/vlm.py",
        "cli/fluidmix/explanation/vlm/materialize_wp7_vlm_development_bundle.py",
        "cli/fluidmix/explanation/vlm/rematerialize_wp7_vlm_native_roi.py",
        "cli/fluidmix/explanation/vlm/run_wp7_vlm_model.py",
        "cli/fluidmix/decision/aggregate_wp7_vlm_attempts.py",
        "cli/fluidmix/decision/wp7_vlm_decision.py",
        "tests/fluidmix/explanation/vlm/test_development_materializer.py",
        "tests/fluidmix/explanation/vlm/test_production_materializer.py",
        "tests/fluidmix/decision/test_vlm.py",
        "tests/fluidmix/explanation/vlm/test_reasoning_upgrade.py",
    }
    assert {path for path in expected if (repo / path).is_file()} == expected


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


def test_replay_splits_two_moved_package_members_with_distinct_destinations(tmp_path):
    source = """def observe():
    from fluidmix import empty_reference_timing_adapter, wp7_fill_step_production

    return empty_reference_timing_adapter, wp7_fill_step_production
"""
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/empty_reference_timing_adapter.py": "",
            "src/fluidmix/wp7_fill_step_production.py": "",
            "tests/test_wp7_fill_step_production.py": source,
        },
        {
            "perception/baselines": "empty_reference_rgb_predictor",
            "perception/masks": "rgb_mask_prediction",
        },
    )

    result = invoke(repo, mapping)

    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (
        repo / "tests/fluidmix/perception/tracker/test_fill_step_production.py"
    ).read_text()
    assert (
        "from fluidmix.perception.baselines import timing_adapter as "
        "empty_reference_timing_adapter\n"
        "    from fluidmix.perception.tracker import fill_step_production as "
        "wp7_fill_step_production"
    ) in rewritten
    assert "from fluidmix import" not in rewritten


def test_import_rewrite_keeps_cross_package_targets_absolute(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/physx_run_variation.py": (
                "from .paths import REPO\n"
            ),
            "src/fluidmix/paths.py": "REPO = 1\n",
        },
        {"sim/physx": "physx_run_variation", "core": "paths"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "src/twin/physx/run_variation.py").read_text()
    assert "from fluidmix.core.paths import REPO" in rewritten
    assert "from ...fluidmix" not in rewritten


def test_path_rewrite_updates_repo_root_and_script_destination(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "scripts/run_carryover_pipeline.py": "",
            "tests/test_helper.py": "value = 1\n",
            "tests/test_carryover_pipeline.py": (
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parents[1]\n"
                "SCRIPT = ROOT / \"scripts/run_carryover_pipeline.py\"\n"
                "SCRIPT_SPLIT = ROOT / \"scripts\" / \"run_carryover_pipeline.py\"\n"
                "sys.path.insert(0, str(ROOT / \"scripts\"))\n"
                "import run_carryover_pipeline\n"
                "import scripts.run_carryover_pipeline as package_runner\n"
                "from test_helper import value\n"
                "EXTERNAL = Path(\"/repo/scripts/run_carryover_pipeline.py\")\n"
                "OMNI = Path(\"omni/physx/scripts/particleUtils.py\")\n"
                "TEMP = tmp_path / \"scripts\"\n"
            ),
        },
            {"pipeline": "carryover_pipeline", "core": "helper"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "tests/fluidmix/pipeline/test_carryover_pipeline.py").read_text()
    assert "parents[3]" in rewritten
    assert 'ROOT / "cli/fluidmix/pipeline/run_carryover_pipeline.py"' in rewritten
    assert 'ROOT / "cli/fluidmix/pipeline" / "run_carryover_pipeline.py"' in rewritten
    assert 'sys.path.insert(0, str(ROOT / "scripts"))' not in rewritten
    assert "import cli.fluidmix.pipeline.run_carryover_pipeline" in rewritten
    assert "import cli.fluidmix.pipeline.run_carryover_pipeline as package_runner" in rewritten
    assert "from tests/fluidmix/core/test_helper import value" not in rewritten
    assert "from tests.fluidmix.core.test_helper import value" in rewritten
    assert 'Path("/repo/cli/fluidmix/pipeline/run_carryover_pipeline.py")' in rewritten
    assert 'Path("omni/physx/scripts/particleUtils.py")' in rewritten
    assert 'tmp_path / "scripts"' in rewritten


def test_rewrite_removes_only_repo_scripts_sys_path_insertions(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "scripts/check_leakage.py": "",
            "tests/test_check_leakage.py": (
                "import os\n"
                "from pathlib import Path\n"
                "import sys\n"
                "ROOT = Path(__file__).resolve().parents[1]\n"
                'sys.path.insert(0, str(ROOT / "scripts"))\n'
                'sys.path.insert(0, os.fspath(ROOT / "scripts"))\n'
                'sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))\n'
                'sys.path.insert(0, str(tmp_path / "scripts"))\n'
                'sys.path.insert(0, str(Path("/external/scripts")))\n'
                "import check_leakage\n"
            ),
        },
        {"core/policy": "check_leakage"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "tests/fluidmix/core/test_check_leakage.py").read_text()
    assert 'ROOT / "scripts"' not in rewritten
    assert 'parents[3]/"scripts"' not in rewritten
    assert 'sys.path.insert(0, str(tmp_path / "scripts"))' in rewritten
    assert 'sys.path.insert(0, str(Path("/external/scripts")))' in rewritten


def test_rewrite_updates_dynamic_module_strings_and_source_path_literals(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/rca_bench.py": (
                'RUNNER = __import__("fluidmix.rca_bench", fromlist=["run"])\n'
                'SOURCE = "src/fluidmix/rca_bench.py"\n'
                'UNRELATED = "prefix fluidmix.rca_bench suffix"\n'
            ),
            "src/fluidmix/physx_run_variation.py": (
                'SOURCE = "src/fluidmix/physx_run_variation.py"\n'
            ),
        },
        {"evaluation/rca_bench": "rca_bench", "sim/physx": "physx_run_variation"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    evaluation = (repo / "src/evaluation/rca_bench/rca_bench.py").read_text()
    twin = (repo / "src/twin/physx/run_variation.py").read_text()
    assert '__import__("evaluation.rca_bench.rca_bench"' in evaluation
    assert '"src/evaluation/rca_bench/rca_bench.py"' in evaluation
    assert '"prefix fluidmix.rca_bench suffix"' in evaluation
    assert '"src/twin/physx/run_variation.py"' in twin


def test_rewrite_maps_module_collection_literals_but_not_direct_text(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/old.py": (
                'REGISTRY = ("fluidmix.old", {"fluidmix.old"})\n'
                'TEXT = "fluidmix.old"\n'
            ),
        },
        {"decision": "old"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "src/fluidmix/decision/old.py").read_text()
    assert '"fluidmix.decision.old"' in rewritten
    assert 'TEXT = "fluidmix.old"' in rewritten


def test_rewrite_keeps_filename_literals_out_of_module_mapping(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/old.py": (
                'SCRIPTS = ("color.py", "fixlabels.py", "fluidmix.old")\n'
            ),
            "scripts/color.py": "",
            "scripts/fixlabels.py": "",
        },
        {"decision": "old", "core/policy": "color fixlabels"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "src/fluidmix/decision/old.py").read_text()
    assert '("color.py", "fixlabels.py", "fluidmix.decision.old")' in rewritten
    assert "cli.fluidmix.core.color.py" not in rewritten
    assert "cli.fluidmix.core.fixlabels.py" not in rewritten


def test_review_regressions_semantics_preserve_script_names_and_fixture_root(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "tests/test_review_regressions.py": (
                'def run_script(name, *args):\n'
                '    if name in ("color.py", "fixlabels.py") and args:\n'
                '        pass\n'
                '    return ROOT / "scripts" / name\n'
                '(tmp_path / "scripts").mkdir()\n'
                'shutil.copy2(ROOT / "scripts/check_agent_docs.py", tmp_path / "scripts/check_agent_docs.py")\n'
            ),
            "scripts/color.py": "",
            "scripts/fixlabels.py": "",
            "scripts/check_agent_docs.py": "",
        },
        {"colour": "absorbance", "core/policy": "color fixlabels check_agent_docs"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "tests/fluidmix/colour/test_review_regressions.py").read_text()
    assert 'name in ("color.py", "fixlabels.py")' in rewritten
    assert 'ROOT / "cli/fluidmix/core" / name' in rewritten
    assert '(tmp_path / "cli/fluidmix/core").mkdir(parents=True)' in rewritten
    assert 'tmp_path / "cli/fluidmix/core/check_agent_docs.py"' in rewritten


def test_sixth_development_semantics_use_moved_predictor_module_path(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/sixth_development_run.py": with_semantic_patterns(
                "src/fluidmix/sixth_development_run.py",
                'predictor_path = Path(__file__).resolve(strict=True).with_name(\n'
                '    "cnn_predictor.py"\n'
                ')\n'
            ),
            "tests/test_sixth_development_run.py": (
                'MONKEYPATCH = str(ROOT / "tests" / ".." / "src" / "fluidmix" / "sixth_development_run.py")\n'
            ),
        },
        {"memory/history": "sixth_development_run"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "tests/fluidmix/memory/history/test_sixth_development_run.py").read_text()
    assert 'ROOT / "src" / "fluidmix" / "memory" / "history" / "sixth_development_run.py"' in rewritten


def test_remaining_cli_path_semantics_are_rewritten_and_shell_is_supported(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "tests/test_contract_schemas.py": 'SCHEMA_ROOT = ROOT / "src" / "fluidmix" / "contracts"\n',
            "tests/test_cosmos_nano_inputs_runner.py": (
                '[sys.executable, str(ROOT / "scripts" / script), "--help"]\n'
            ),
            "tests/test_cosmos_usable_preflight.py": (
                '[sys.executable, str(ROOT / "scripts" / script), "--help"]\n'
            ),
            "tests/test_rigid_vial_transport.py": (
                'path = REPO_ROOT / "scripts" / f"{name}.py"\n'
            ),
            "tests/test_physx_palette_smoke_remote.py": (
                '(source / "scripts").mkdir(parents=True)\n'
                '(source / "scripts" / "physx_particle_palette_smoke.py").write_text("# identity")\n'
            ),
            "tests/test_physx_f2_remote_launcher.py": 'scripts = source / "scripts"\n',
                "scripts/run_physx_f2_remote.sh": (
                    'runner="$source_root/scripts/physx_three_pour_sequence.py"\n'
                    'test -f "$source_root/scripts/physx_three_pour_sequence.py"\n'
                ),
                "scripts/run_physx_palette_smoke_remote.sh": (
                    'runner="$source_root/scripts/physx_particle_palette_smoke.py"\n'
                    'test -f "$source_root/scripts/physx_particle_palette_smoke.py"\n'
                ),
        },
        {
            "core/contracts": "contracts",
            "explanation/vlm": "cosmos_nano_runner",
            "sim/transport": "rigid_vial_transport",
            "sim/physx": "physx_three_pour_sequence",
            "sim/probes": "physx_particle_palette_smoke",
            "evaluation/rca_bench": "run_rca_bench",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ROOT / "src" / "fluidmix" / "core" / "contracts"' in (
        repo / "tests/fluidmix/core/contracts/test_schemas.py"
    ).read_text()
    assert '"stage_cosmos_nano_reasoner.py": ROOT / "cli/fluidmix/explanation/vlm/stage_cosmos_nano_reasoner.py"' in (
        repo / "tests/fluidmix/explanation/vlm/test_cosmos_nano_inputs_runner.py"
    ).read_text()
    assert '"render_rigid_vial_transport": REPO_ROOT / "cli/twin/transport/render_rigid_vial_transport.py"' in (
        repo / "tests/twin/transport/test_rigid_vial_transport.py"
    ).read_text()
    assert 'source / "cli/twin/probes"' in (
        repo / "tests/twin/physx/test_palette_smoke_remote.py"
    ).read_text()
    assert 'source_root/cli/twin/physx/physx_three_pour_sequence.py' in (
        repo / "cli/twin/physx/run_physx_f2_remote.sh"
    ).read_text()


def test_rewrite_maps_embedded_python_imports_and_dynamic_calls(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/old.py": "VALUE = 1\n",
            "src/fluidmix/consumer.py": (
                'SOURCE = """\n'
                'from fluidmix.old import VALUE\n'
                'loaded = __import__("fluidmix.old")\n'
                'TEXT = "fluidmix.old"\n'
                '"""\n'
            ),
        },
        {"decision": "old consumer"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    rewritten = (repo / "src/fluidmix/decision/consumer.py").read_text()
    assert "from fluidmix.decision.old import VALUE" in rewritten
    assert '__import__("fluidmix.decision.old")' in rewritten
    assert 'TEXT = "fluidmix.old"' in rewritten


def test_semantic_manifest_migration_is_scoped_and_fail_closed(tmp_path):
    from textwrap import dedent

    manifest = dedent('''
def _local_name(name: str) -> str | None:
    return name if name == "fluidmix" or name.startswith("fluidmix.") else None

def parse(local, target, package_member_import):
    if local is not None and local != "fluidmix":
        pass
    if local is not None and local != "fluidmix":
        pass
    if target == "fluidmix" or package_member_import:
        pass
    if local is not None and local != "fluidmix":
        pass

def walk(name):
    if name.startswith("scripts/"):
        pass

WP7_PRODUCTION_PROJECTOR_MODULE = "fluidmix.part_b_evidence"
current_modules = {
    "fluidmix."
    + path.relative_to(root / "src" / "fluidmix")
    .with_suffix("")
    .as_posix()
    .replace("/", ".")
    for path in (root / "src" / "fluidmix").rglob("*.py")
    if path.name != "__init__.py"
}
    ''')
    test_source = dedent('''
current_modules = {
    "fluidmix."
    + path.relative_to(ROOT / "src" / "fluidmix")
    .with_suffix("")
    .as_posix()
    .replace("/", ".")
    for path in (ROOT / "src" / "fluidmix").rglob("*.py")
    if path.name != "__init__.py"
}
source = """
from fluidmix import vision_measurement
"""
module_name="fluidmix.contracts.example"
''')
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/part_boundary_manifest.py": manifest,
            "src/fluidmix/part_b_evidence.py": "",
            "src/fluidmix/vision_measurement.py": "",
            "tests/test_part_boundary.py": test_source,
        },
        {
            "storage/artifacts": "part_boundary_manifest part_b_evidence",
            "measurement": "vision_measurement",
        },
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    migrated = (repo / "src/fluidmix/storage/artifacts/part_boundary_manifest.py").read_text()
    assert 'name.split(".", 1)[0] in {"fluidmix", "twin", "evaluation"}' in migrated
    assert 'local not in {"fluidmix", "twin", "evaluation"}' in migrated
    assert 'name.startswith("cli/")' in migrated
    assert 'WP7_PRODUCTION_PROJECTOR_MODULE = "fluidmix.storage.artifacts.part_b_evidence"' in migrated
    assert 'for package in ("fluidmix", "twin", "evaluation")' in migrated
    migrated_test = (repo / "tests/fluidmix/storage/artifacts/test_part_boundary.py").read_text()
    assert 'for package in ("fluidmix", "twin", "evaluation")' in migrated_test
    assert "import fluidmix.measurement.vision_measurement" in migrated_test
    assert 'module_name="fluidmix.core.contracts.example"' in migrated_test


def test_audited_path_semantics_are_keyed_and_fail_closed():
    audited = {
        "scripts/render_capseal_structural_pilot.py",
        "src/fluidmix/inn67_predicted_store_producer.py",
        "src/fluidmix/predicted_mask_materializer.py",
        "src/fluidmix/sixth_development_run.py",
        "scripts/physx_three_pour_sequence.py",
        "tests/test_review_regressions.py",
        "tests/test_sixth_development_run.py",
    }
    assert audited <= set(onion_move.SEMANTIC_REWRITES)
    for relative in audited:
        rules = onion_move.SEMANTIC_REWRITES[relative]
        assert all(expected_count == 1 for _, _, expected_count in rules)
        source = "\n".join(old for old, _, _ in rules)
        expected = "\n".join(new for _, new, _ in rules)
        assert onion_move._semantic_rewrite(Path(relative), source) == expected
        first_old = rules[0][0]
        with pytest.raises(SystemExit, match="semantic migration drift"):
            onion_move._semantic_rewrite(
                Path(relative), source.replace(first_old, first_old + "\n" + first_old, 1)
            )


def test_frozen_identity_semantics_rewrite_complete_hash_chain_and_fail_closed():
    expected_paths = {
        "src/fluidmix/predicted_mask_materializer.py",
        "src/fluidmix/inn67_predicted_store_producer.py",
        "src/fluidmix/sixth_development_run.py",
        "src/fluidmix/inn67_bank_materializer.py",
        "src/fluidmix/wp7_development_reference.py",
        "src/fluidmix/wp7_history_preparation.py",
        "src/fluidmix/carryover_pipeline.py",
        "src/fluidmix/wp1_postrender.py",
        "src/fluidmix/transport_stage.py",
        "scripts/validate_physx_f2_artifacts.py",
        "scripts/run_physx_train_residual_cohort.py",
        "tests/test_inn67_bank_materializer.py",
        "tests/test_inn67_postrender_producer.py",
        "tests/test_inn67_predicted_store_producer.py",
        "tests/test_wp7_history_preparation.py",
        "tests/test_wp7_route_preflight.py",
        "tests/test_wp1_postrender.py",
        "tests/test_transport_stage.py",
        "tests/test_physx_train_residual_supervisor.py",
        "tests/test_splits.py",
        "tests/test_physx_one_run_manifest_v1.py",
        "tests/test_physx_varied_cohort_supervisor.py",
        "tests/test_physx_run_variation.py",
        "tests/test_test_recipe_generator.py",
        "tests/fixtures/inn67-materialization-protocol-amendment.json",
    }
    assert expected_paths <= set(onion_move.SEMANTIC_REWRITES)
    expected_hashes = {
        "ec9f4b589f088fe979f084ebff1ad51cb0fe0dd29cf12a64a7c993bab534be19",
        "e015a0d02aef6fce283f85a51ea4ab3ff9bde2fb01293a5a3d2b42ec08ad765b",
        "27dba1b66f680c4444f0c6e6e3505291ff9e7af586a91d4747f1b09a55876a0e",
        "f0ab668df8bb2f4abe8708a486ee00a913e129772a83a4d1abd2b95d28d467d7",
        "5a39a1c8de636d0d0de83440a8483c2ec6a692b1722c1a77ac6ec613bee93128",
        "2f39a4507d429902e241fff0b5709d7c646a702e8d8331a54c9c941f62ccb4bd",
        "16b8c53fc00373af9fe94fa0305b4482588a1d77a9b6fab513c9ab6e0aa72ed7",
        "deb0d7bb98ec6c45151ce6b0a6c58342a58c5903c0c09163fd33ddec9f723edb",
    }
    rewritten_hashes = set()
    for relative in expected_paths:
        rules = onion_move.SEMANTIC_REWRITES[relative]
        source = "\n".join(old * count for old, _, count in rules)
        expected = "\n".join(new * count for _, new, count in rules)
        assert onion_move._semantic_rewrite(Path(relative), source) == expected
        rewritten_hashes.update(
            value
            for value in expected_hashes
            if value in expected
        )
        first_old = rules[0][0]
        with pytest.raises(SystemExit, match="semantic migration drift"):
            onion_move._semantic_rewrite(
                Path(relative), source.replace(first_old, first_old + first_old, 1)
            )
    assert rewritten_hashes == expected_hashes
    postrender_rule = (
        "fluidmix.inn67_postrender_producer",
        "fluidmix.memory.reference_bank.postrender_producer",
        6,
    )
    assert postrender_rule in onion_move.SEMANTIC_REWRITES[
        "tests/test_inn67_postrender_producer.py"
    ]
    assert (*postrender_rule[:2], 2) in onion_move.SEMANTIC_REWRITES[
        "tests/test_inn67_predicted_store_producer.py"
    ]
    assert (
        'monkeypatch.setitem(sys.modules, "physx_three_pour_sequence", f2)',
        'monkeypatch.setitem(sys.modules, "cli.twin.physx.physx_three_pour_sequence", f2)\n'
        '    monkeypatch.setattr(__import__("cli.twin.physx", fromlist=["physx"]), '
        '"physx_three_pour_sequence", f2, raising=False)',
        1,
    ) in onion_move.SEMANTIC_REWRITES["tests/test_transport_stage.py"]
    assert (
        'ROOT = Path(__file__).resolve().parents[1]\n'
        'sys.path.insert(0, str(ROOT / "src"))',
        'ROOT = Path(__file__).resolve().parents[1]\n'
        'sys.path.insert(0, str(ROOT))\n'
        'sys.path.insert(0, str(ROOT / "src"))',
        1,
    ) in onion_move.SEMANTIC_REWRITES["scripts/validate_physx_f2_artifacts.py"]
    assert (
        "import check_leakage",
        "import cli.fluidmix.core.check_leakage as check_leakage",
        1,
    ) in onion_move.SEMANTIC_REWRITES["tests/test_splits.py"]
    assert (
        "import make_splits",
        "import cli.twin.cohorts.make_splits as make_splits",
        1,
    ) in onion_move.SEMANTIC_REWRITES["tests/test_splits.py"]
    varied_rules = set(
        onion_move.SEMANTIC_REWRITES["tests/test_physx_varied_cohort_supervisor.py"]
    )
    assert {
        (
            "fluidmix.tank_colour_presets.load_tank_colour_preset_catalog",
            "twin.scene.tank_colour_presets.load_tank_colour_preset_catalog",
            1,
        ),
        (
            "5bf92fea5a7e941081c9b56a4e937fd2cdeaa9ecc6ba77d125dff1b257076de0",
            "511d3a9012935b2d3ed86a56f57bd434989620295f8061945d621ce05b8c3e63",
            1,
        ),
        (
            "0608ab047c2e0cf733649a9b9c0224b6412f7654613dea84af014e3a347d637b",
            "dcf2f1f22ce60b0e7d2b327bd1f3650185deac2b1a7c4dbcd6e9dddbfc6a6cfc",
            1,
        ),
        (
            "a86bce8f4e80adc985f6d5501cf3f941c84a962d114d04012949b905ff9ed572",
            "a7a97c5d43efc647c4f22d787d5fc0c3ada6c1d8e8ac195b7b25004725e5a92b",
            1,
        ),
    } <= varied_rules


def test_mapped_json_fixture_receives_replayable_semantic_rewrite(tmp_path):
    old_plan = "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79"
    old_inventory = "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95"
    old_amendment = "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8"
    old_validator = "3780a83579cf807cf8f4041bdfc931f8b98c374ae8f7eaea386be04ed6ac2315"
    fixture = json.dumps(
        {
            "bound_inputs": {
                "reference_bank_plan_sha256": old_plan,
                "reference_bank_inventory_sha256": old_inventory,
                "recipe_role_amendment_file_sha256": old_amendment,
                "accepted_occlusion_validator_sha256": old_validator,
            }
        },
        indent=2,
    )
    repo, mapping = make_repo(
        tmp_path,
        {"tests/fixtures/inn67-materialization-protocol-amendment.json": fixture},
        {"memory/reference_bank": "inn67_reference_bank"},
    )
    result = invoke(repo, mapping)
    assert result.returncode == 0, result.stdout + result.stderr
    destination = (
        repo / "tests/fluidmix/memory/reference_bank/fixtures/"
        "inn67-materialization-protocol-amendment.json"
    )
    rewritten = json.loads(destination.read_text())
    assert rewritten["bound_inputs"] == {
        "reference_bank_plan_sha256": "e015a0d02aef6fce283f85a51ea4ab3ff9bde2fb01293a5a3d2b42ec08ad765b",
        "reference_bank_inventory_sha256": "f0ab668df8bb2f4abe8708a486ee00a913e129772a83a4d1abd2b95d28d467d7",
        "recipe_role_amendment_file_sha256": "5a39a1c8de636d0d0de83440a8483c2ec6a692b1722c1a77ac6ec613bee93128",
        "accepted_occlusion_validator_sha256": "16b8c53fc00373af9fe94fa0305b4482588a1d77a9b6fab513c9ab6e0aa72ed7",
    }
    destination.write_text(fixture)
    refresh = invoke(repo, mapping, "--refresh-staged-rewrites")
    assert refresh.returncode == 0, refresh.stdout + refresh.stderr
    assert json.loads(destination.read_text())["bound_inputs"] == rewritten["bound_inputs"]
    replay = invoke(repo, mapping)
    assert replay.returncode == 0, replay.stdout + replay.stderr


def test_staged_refresh_replays_semantics_for_moved_cli_files(tmp_path):
    source = (
        "from pathlib import Path\n"
        "import sys\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        'sys.path.insert(0, str(ROOT / "src"))\n'
    )
    repo, mapping = make_repo(
        tmp_path,
        {"scripts/validate_physx_f2_artifacts.py": source},
        {"sim/physx": "validate_physx_f2_artifacts"},
    )
    first = invoke(repo, mapping)
    assert first.returncode == 0, first.stdout + first.stderr
    destination = repo / "cli/twin/physx/validate_physx_f2_artifacts.py"
    destination.write_text("stale\n")
    refresh = invoke(repo, mapping, "--refresh-staged-rewrites")
    assert refresh.returncode == 0, refresh.stdout + refresh.stderr
    rewritten = destination.read_text()
    assert "parents[3]" in rewritten
    assert 'sys.path.insert(0, str(ROOT))' in rewritten


def test_cli_top_level_cli_import_adds_repo_root_before_import_and_replays(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "scripts/align_still_labels.py": (
                "from pathlib import Path\n"
                "from scripts.check_leakage import run\n"
            ),
            "scripts/check_leakage.py": "def run():\n    return 1\n",
        },
        {"core/policy": "align_still_labels check_leakage"},
    )
    first = invoke(repo, mapping)
    assert first.returncode == 0, first.stdout + first.stderr
    destination = repo / "cli/fluidmix/core/align_still_labels.py"
    rewritten = destination.read_text()
    assert "import sys" in rewritten
    root_insert = "sys.path.insert(0, str(Path(__file__).resolve().parents[3]))"
    cli_import = "from cli.fluidmix.core.check_leakage import run"
    assert rewritten.count(root_insert) == 1
    assert rewritten.index(root_insert) < rewritten.index(cli_import)

    destination.write_text("stale\n")
    refresh = invoke(repo, mapping, "--refresh-staged-rewrites")
    assert refresh.returncode == 0, refresh.stdout + refresh.stderr
    replayed = destination.read_text()
    assert replayed.count(root_insert) == 1
    assert replayed.index(root_insert) < replayed.index(cli_import)


def test_perception_mask_responsibilities_use_distinct_explicit_names(tmp_path):
    repo, mapping = make_repo(
        tmp_path,
        {
            "src/fluidmix/rgb_mask_prediction.py": "",
            "src/fluidmix/predicted_mask_materializer.py": with_semantic_patterns(
                "src/fluidmix/predicted_mask_materializer.py",
                'predictor_source_path = Path(__file__).with_name(\n'
                '    "cnn_predictor.py" if use_cnn else "empty_reference_rgb_predictor.py"\n'
                ')\n'
            ),
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
    assert (repo / "tests/fluidmix/perception/test_mask_materializer.py").is_file()
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
        "src/fluidmix/predicted_mask_materializer.py": (
            'predictor_source_path = Path(__file__).with_name(\n'
            '    "cnn_predictor.py" if use_cnn else "empty_reference_rgb_predictor.py"\n'
            ')\n'
        ),
        "src/fluidmix/physx_case_plan.py": "",
        "src/fluidmix/stage_cosmos_runner.py": "",
        "src/fluidmix/v3_cosmos_scoring.py": "",
        "src/fluidmix/wp7_final_decision.py": "",
        "src/fluidmix/carryover_pipeline.py": "",
        "src/fluidmix/f2_varied_triage.py": "",
        "src/fluidmix/wp7_development_reference.py": "",
    }
    files = {
        relative: with_semantic_patterns(relative, source)
        for relative, source in files.items()
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

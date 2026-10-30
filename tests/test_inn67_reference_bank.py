from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.inn67_reference_bank import build_reference_bank_bundle, probe_gb10_resources  # noqa: E402
from fluidmix.physx_one_run_manifest_v1 import rebuild_bound_one_run_artifacts  # noqa: E402

SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "build_inn67_reference_bank_plan_script",
    ROOT / "scripts" / "build_inn67_reference_bank_plan.py",
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
PLAN_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(PLAN_SCRIPT)

EXPECTED_RECIPES = {
    "amber_teal": [
        ("S0439", {"t100": 26.0, "t200": 28.0, "t400": 0.0}, ["t200", "t100"]),
        ("S0440", {"t100": 11.0, "t200": 31.0, "t400": 20.0}, ["t200", "t400", "t100"]),
        ("S0441", {"t100": 11.0, "t200": 11.0, "t400": 43.0}, ["t200", "t400", "t100"]),
    ],
    "crimson_indigo": [
        ("S0442", {"t100": 56.0, "t200": 24.0, "t400": 0.0}, ["t200", "t100"]),
        ("S0443", {"t100": 10.0, "t200": 50.0, "t400": 16.0}, ["t200", "t400", "t100"]),
        ("S0444", {"t100": 11.0, "t200": 10.0, "t400": 39.0}, ["t200", "t400", "t100"]),
    ],
}


def test_bundle_has_exact_denominators_and_rebuildable_subruns():
    bundle = build_reference_bank_bundle(ROOT)
    plan = bundle["reference-bank-plan.json"]
    assert plan["accepted_renderer_base"] == "24889cad3cf9d7302622a4f844fb4b2a4977d4c9"
    assert plan["denominators"] == {
        "development_batches": 3, "palette_subruns": 6, "fill_cases": 18,
        "stage_observations": 48, "reference_donor_batches": 2,
        "config_holdout_batches": 1,
    }
    assert len(plan["batches"]) == 3
    assert len(plan["subruns"]) == 6
    assert len(plan["cases"]) == 18
    assert sum(len(case["stages"]) for case in plan["cases"]) == 48
    assert [batch["experimental_role"] for batch in plan["batches"]] == [
        "development_reference_donor", "development_reference_donor",
        "development_config_holdout",
    ]
    for subrun in plan["subruns"]:
        rows = [case for case in plan["cases"] if case["subrun_id"] == subrun["subrun_id"]]
        assert [(row["recipe_id"], row["programmed_ml"], row["dispense_order"]) for row in rows] == EXPECTED_RECIPES[subrun["palette"]]
        artifact_name = subrun["render_plan_artifact"]
        rebuilt = rebuild_bound_one_run_artifacts(bundle[artifact_name])
        assert rebuilt["render-schedule-plan.json"] == bundle[artifact_name]


def test_ids_and_profile_are_development_only_contract_known_good():
    bundle = build_reference_bank_bundle(ROOT)
    plan = bundle["reference-bank-plan.json"]
    ids = [*(x["batch_id"] for x in plan["batches"]), *(x["subrun_id"] for x in plan["subruns"]), *(x["run_id"] for x in plan["subruns"]), *(x["case_id"] for x in plan["cases"])]
    assert len(ids) == len(set(ids))
    assert all(value.startswith("INN67-DEV-RB-") for value in ids)
    assert all("WP1" not in value and "VIDEO5" not in value for value in ids)
    assert {case["split_role"] for case in plan["cases"]} == {"development"}
    profile = bundle["known-good-profile.json"]
    assert profile["injected_defect"] is False
    assert profile["prescribed_leading_residual_volume_ml"] == 0.0
    assert profile["physical_truth_claimed"] is False
    assert profile["camera_lights_geometry_changed"] is False
    leakage = bundle["leakage-audit.json"]
    assert leakage["status"] == "pass"
    assert leakage["recipe_identity_overlap_with_wp1_test_is_required"] is True
    assert leakage["wp1_render_outputs_opened"] is False
    assert leakage["video5_artifacts_opened"] is False


def test_resource_probe_refuses_foreign_load_and_never_launches():
    responses = {
        ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"): (0, "NVIDIA GB10\n"),
        ("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"): (0, "4242, foreign-python\n"),
        ("free", "-g"): (0, " total used free shared buff/cache available\nMem: 120 20 80 0 20 95\n"),
    }
    calls = []
    def run(command, **kwargs):
        calls.append(tuple(command)); code, stdout = responses[tuple(command)]
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")
    result = probe_gb10_resources(run=run, process_rows=[])
    assert result["status"] == "refused_foreign_load"
    assert result["launch_performed"] is False
    assert result["foreign_gpu_processes"] == [{"pid": 4242, "process_name": "foreign-python"}]
    assert calls == list(responses)


def test_cli_publishes_frozen_no_launch_bundle(tmp_path):
    output = tmp_path / "bundle"
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_inn67_reference_bank_plan.py"), "--output-dir", str(output), "--skip-live-resource-probe"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    expected = {"reference-bank-plan.json", "known-good-profile.json", "delta-e-freeze-config.json", "leakage-audit.json", "no-access-attestation.json", "resource-preflight.json", "PLAN.md", "artifact-hashes.json"}
    assert expected <= {path.name for path in output.iterdir()}
    preflight = json.loads((output / "resource-preflight.json").read_text())
    assert preflight["status"] == "not_probed_by_explicit_cpu_option"
    assert preflight["launch_performed"] is False


@pytest.mark.parametrize("position", ["first", "middle", "final"])
def test_plan_publication_failure_cleans_owned_output_and_retry_succeeds(
    tmp_path, monkeypatch, position
):
    output = tmp_path / "bundle"
    total_writes = len(build_reference_bank_bundle(ROOT)) + 3
    fail_at = {"first": 1, "middle": total_writes // 2, "final": total_writes}[
        position
    ]
    original = Path.write_bytes
    calls = 0

    def fail_selected_write(path, body):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected plan write failure {position}")
        return original(path, body)

    monkeypatch.setattr(Path, "write_bytes", fail_selected_write)
    with pytest.raises(OSError, match="injected plan write failure"):
        PLAN_SCRIPT.main(
            ["--output-dir", str(output), "--skip-live-resource-probe"]
        )
    assert not output.exists()

    monkeypatch.setattr(Path, "write_bytes", original)
    assert PLAN_SCRIPT.main(
        ["--output-dir", str(output), "--skip-live-resource-probe"]
    ) == 0


def test_plan_publication_preserves_preexisting_caller_owned_directory(tmp_path):
    output = tmp_path / "caller-owned"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("caller-owned\n")

    with pytest.raises(SystemExit):
        PLAN_SCRIPT.main(
            ["--output-dir", str(output), "--skip-live-resource-probe"]
        )

    assert sentinel.read_text() == "caller-owned\n"
    assert list(output.iterdir()) == [sentinel]

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fluidmix.f2_stage_cohort_materializer as materializer  # noqa: E402
from fluidmix.physx_train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)

CLI_SPEC = importlib.util.spec_from_file_location(
    "materialize_f2_stage_cohort_cli",
    ROOT / "scripts" / "materialize_f2_stage_cohort.py",
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
cli = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(cli)


TEST_RENDER_CONTRACT = {
    "renderer": "PathTracing",
    "resolution": [640, 480],
    "rt_subframes": 8,
    "pathtracing_total_spp": 32,
    "warmup_updates": 41,
    "capture_stride": 312,
    "video_fps": 10.0,
    "wall_limit_seconds": 9876,
    "render_path": "count_driven_twin_liquid_mesh",
    "count_driven_twin_liquid_mesh": True,
    "report_hz": 120,
    "physics_hz": 240,
    "physics_steps_per_second": 240,
    "physics_substeps_per_report_interval": 2,
    "diffusion_updates_per_report_interval": 1,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _two_clock_timing(report_interval_count: int) -> dict[str, int]:
    return {
        "report_hz": 120,
        "report_interval_count": report_interval_count,
        "executed_report_intervals": report_interval_count,
        "physics_hz": 240,
        "physics_steps_per_second": 240,
        "physics_substeps_per_report_interval": 2,
        "physics_callback_count": report_interval_count * 2,
        "callbacks_expected": report_interval_count * 2,
        "callbacks_observed": report_interval_count * 2,
        "diffusion_updates_per_report_interval": 1,
        "diffusion_update_count": report_interval_count,
    }


def _add_runtime_particle_evidence(receipt: dict[str, object], count: int) -> None:
    authored = (
        "positions",
        "velocities",
        "orientations",
        "scales",
        "protoIndices",
        "ids",
    )
    documentation_url = (
        "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/"
        "dev_guide/schemas/physxschema.html"
    )
    receipt["plan"] = {"totals": {"particle_count": count}}
    receipt["captures"] = [{"inside_vial_interior_count": count}]
    receipt["point_instancer"] = {
        "runtime_layer_evidence": {
            "property_specs": {
                **{f"{name}_session_layer_spec": True for name in authored},
                **{f"{name}_root_layer_spec": False for name in authored},
                # PhysX owns this read-only property; a session spec is optional.
                "simulationPoints_session_layer_spec": False,
                "simulationPoints_root_layer_spec": False,
            },
            "authored_session_only_properties": list(authored),
            "read_only_runtime_property": "simulationPoints",
            "read_only_runtime_property_session_spec_required": False,
            "read_only_runtime_property_root_spec_forbidden": True,
            "read_only_runtime_property_documentation_url": documentation_url,
            "failed_properties": [],
            "final_edit_target_is_session": True,
            "passed": True,
            "simulation_points_runtime_evidence": {
                "attribute": "physxParticle:simulationPoints",
                "documented_read_only": True,
                "documentation_url": documentation_url,
                "runtime_readback_required": True,
                "runtime_readback_shape": [count, 3],
                "runtime_readback_row_count": count,
                "expected_particle_count": count,
                "particle_id_count": count,
                "containment_particle_count": count,
                "finite_xyz_rows": True,
                "row_counts_match": True,
                "passed": True,
            },
        }
    }
    timing = receipt["timing"]
    assert isinstance(timing, dict)
    timing.update(
        {
            "final_edit_target_is_session": True,
            "positions_session_layer_only": True,
            "velocities_session_layer_only": True,
            "orientations_session_layer_only": True,
            "scales_session_layer_only": True,
            "proto_indices_session_layer_only": True,
            "ids_session_layer_only": True,
            "simulation_points_root_layer_absent": True,
            "simulation_points_runtime_readback_passed": True,
        }
    )


def _source_artifacts(*, batch_limit: int | None = None) -> dict[str, object]:
    artifacts = deepcopy(build_train_residual_artifacts(repo_root=ROOT))
    plan = artifacts["render-schedule-plan.json"]
    model_plan = artifacts["model-input-plan.json"]
    if batch_limit is not None:
        kept_batches = plan["batches"][:batch_limit]
        kept_ids = {batch["batch_id"] for batch in kept_batches}
        kept_cases = {case_id for batch in kept_batches for case_id in batch["case_ids"]}
        plan["batches"] = kept_batches
        plan["execution_schedules"] = [
            row
            for row in plan["execution_schedules"]
            if row["case_binding"]["case_id"] in kept_cases
        ]
        plan["execution_schedule_sha256_by_case"] = {
            key: value
            for key, value in plan["execution_schedule_sha256_by_case"].items()
            if key in kept_cases
        }
        plan["denominators"].update(
            batches=len(kept_batches),
            vials=len(kept_cases),
            execution_schedules=len(kept_cases),
            stage_images_planned=4 * len(kept_cases),
        )
        model_plan["vial_records"] = [
            row for row in model_plan["vial_records"] if row["batch_id"] in kept_ids
        ]
        model_plan["denominators"].update(
            batches=len(kept_batches), vials=len(kept_cases)
        )
    return artifacts


def _build_completed_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_limit: int | None = None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    artifacts = _source_artifacts(batch_limit=batch_limit)
    monkeypatch.setattr(
        materializer,
        "build_train_residual_artifacts",
        lambda **_: deepcopy(artifacts),
    )
    plan = artifacts["render-schedule-plan.json"]
    model_by_case = {
        row["case_id"]: row
        for row in artifacts["model-input-plan.json"]["vial_records"]
    }
    plan_sha = _canonical_sha256(plan)
    plan_path = (tmp_path / "render-schedule-plan.json").resolve()
    _write_json(plan_path, plan)

    qualification = {
        "status": "pass",
        "render_path": TEST_RENDER_CONTRACT["render_path"],
        "qualification": {"first_stage_only": False},
        "arguments": {
            "recipe8_residual_volume_ml": 0.0,
            "solver_position_iterations": 16,
            "count_driven_twin_liquid_mesh": True,
            "resolution": "640x480",
            "capture_stride": TEST_RENDER_CONTRACT["capture_stride"],
            "video_fps": TEST_RENDER_CONTRACT["video_fps"],
            "wall_limit_seconds": TEST_RENDER_CONTRACT["wall_limit_seconds"],
        },
        "versions": {"isaac": "6.2.0.0", "physx": "110.4.4"},
        "provenance": {
            "requested_stage_sha256": "b" * 64,
            "script_sha256": "c" * 64,
        },
        "render": {
            "renderer": TEST_RENDER_CONTRACT["renderer"],
            "resolution": TEST_RENDER_CONTRACT["resolution"],
            "rt_subframes": TEST_RENDER_CONTRACT["rt_subframes"],
            "pathtracing_total_spp_requested": TEST_RENDER_CONTRACT[
                "pathtracing_total_spp"
            ],
            "pathtracing_total_spp_readback": TEST_RENDER_CONTRACT[
                "pathtracing_total_spp"
            ],
            "pathtracing_spp_requested": TEST_RENDER_CONTRACT[
                "pathtracing_total_spp"
            ],
            "pathtracing_spp_readback": TEST_RENDER_CONTRACT[
                "pathtracing_total_spp"
            ],
            "warmup_updates": TEST_RENDER_CONTRACT["warmup_updates"],
        },
        "timing": _two_clock_timing(4806),
    }
    _add_runtime_particle_evidence(qualification, 3)
    qualification_path = (tmp_path / "accepted-clean-receipt.json").resolve()
    _write_json(qualification_path, qualification)
    qualification_sha = _sha256(qualification_path)

    supervisor_root = (tmp_path / "cohort").resolve()
    cases_root = supervisor_root / "cases"
    attempts_root = supervisor_root / "supervisor-attempts"
    cases_root.mkdir(parents=True)
    attempts_root.mkdir()
    stage_sha = qualification["provenance"]["requested_stage_sha256"]
    renderer_sha = qualification["provenance"]["script_sha256"]
    seeds = {
        case_id: seed
        for batch in plan["batches"]
        for case_id, seed in zip(
            batch["case_ids"], batch["render_seed_by_vial"], strict=True
        )
    }
    schedules = plan["execution_schedules"]
    case_ids = [row["case_binding"]["case_id"] for row in schedules]
    launch = {
        "schema_version": 1,
        "artifact_type": "physx_train_residual_serial_cohort_launch",
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "plan_file_sha256": _sha256(plan_path),
        "qualification_receipt_path": str(qualification_path),
        "qualification_receipt_sha256": qualification_sha,
        "stage_path": "/source/fill_station.usda",
        "stage_sha256": stage_sha,
        "renderer_script": "/source/physx_three_pour_sequence.py",
        "renderer_script_sha256": renderer_sha,
        "python_executable": "/source/isaac/.venv/bin/python",
        "supervisor_script_sha256": "d" * 64,
        "render_contract": TEST_RENDER_CONTRACT,
        "case_ids": case_ids,
        "created_at": "2026-10-26T00:00:00+00:00",
    }
    _write_json(supervisor_root / "cohort-launch.json", launch)

    case_results = []
    for ordinal, schedule in enumerate(schedules, start=1):
        binding = schedule["case_binding"]
        case_id = binding["case_id"]
        seed = seeds[case_id]
        case_root = cases_root / case_id
        app_output = case_root / "app-output"
        app_output.mkdir(parents=True)
        receipt = {
            "status": "pass",
            "render_path": TEST_RENDER_CONTRACT["render_path"],
            "arguments": {
                "seed": seed,
                "resolution": "640x480",
                "capture_stride": TEST_RENDER_CONTRACT["capture_stride"],
                "video_fps": TEST_RENDER_CONTRACT["video_fps"],
                "wall_limit_seconds": TEST_RENDER_CONTRACT["wall_limit_seconds"],
                "count_driven_twin_liquid_mesh": True,
                "train_render_plan_sha256": plan_sha,
                "train_case_id": case_id,
                "accepted_qualification_receipt_sha256": qualification_sha,
            },
            "provenance": {
                "requested_stage_sha256": stage_sha,
                "script_sha256": renderer_sha,
            },
            "render": qualification["render"],
            "timing": _two_clock_timing(
                int(schedule["timing"]["fill_plus_3p0_step"])
            ),
            "execution_schedule": {
                "case_id": case_id,
                "batch_id": binding["batch_id"],
                "renderer_case_binding": {
                    "case_id": case_id,
                    "batch_id": binding["batch_id"],
                    "vial_ordinal": binding["vial_ordinal"],
                },
                "cohort_plan_sha256": plan_sha,
                "schedule_file_sha256": _sha256(plan_path),
                "qualification_receipt_sha256": qualification_sha,
                "model_input_record": deepcopy(model_by_case[case_id]),
            },
        }
        _add_runtime_particle_evidence(receipt, 3)
        _write_json(app_output / "receipt.json", receipt)
        case_launch = {
            "schema_version": 1,
            "artifact_type": "physx_train_residual_case_launch",
            "case_id": case_id,
            "seed": seed,
            "ordinal": ordinal,
            "started_at": "2026-10-26T00:00:00+00:00",
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha,
            "plan_file_sha256": _sha256(plan_path),
            "qualification_receipt_path": str(qualification_path),
            "qualification_receipt_sha256": qualification_sha,
            "stage_path": "/source/fill_station.usda",
            "stage_sha256": stage_sha,
            "renderer_script": "/source/physx_three_pour_sequence.py",
            "renderer_script_sha256": renderer_sha,
            "python_executable": "/source/isaac/.venv/bin/python",
            "render_contract": TEST_RENDER_CONTRACT,
            "argv": ["/source/isaac/.venv/bin/python", "physx_three_pour_sequence.py"],
        }
        _write_json(case_root / "launch-receipt.json", case_launch)
        (case_root / "exit-code.txt").write_text("0\n", encoding="utf-8")
        (case_root / "completed-at.txt").write_text(
            "2026-10-26T00:01:00+00:00\n", encoding="utf-8"
        )
        result = {
            "case_id": case_id,
            "seed": seed,
            "ordinal": ordinal,
            "outcome": "passed",
            "return_code": 0,
            "timed_out": False,
            "launch_error": None,
            "evidence": {"failures": []},
        }
        _write_json(case_root / "case-result.json", result)
        case_results.append(result)

    count = len(case_ids)
    attempt = {
        "schema_version": 1,
        "artifact_type": "physx_train_residual_serial_supervisor_attempt",
        "status": "pass",
        "started_at": "2026-10-26T00:00:00+00:00",
        "completed_at": "2026-10-26T01:00:00+00:00",
        "plan_sha256": plan_sha,
        "qualification_receipt_sha256": qualification_sha,
        "planned_case_count": count,
        "attempts_started": count,
        "processes_completed": count,
        "cases_passed_this_invocation": count,
        "cases_failed_this_invocation": 0,
        "previous_completed_cases_recognized": 0,
        "validated_completed_case_count": count,
        "case_results": case_results,
        "failure": None,
    }
    _write_json(attempts_root / "attempt-0001.json", attempt)
    return plan_path, qualification_path, supervisor_root, plan


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(key for child in value.values() for key in _all_keys(child))]
    if isinstance(value, list):
        return [key for child in value for key in _all_keys(child)]
    return []


@pytest.mark.parametrize("batch_limit", [None, 2])
def test_materializes_every_source_bound_batch_without_fixed_contract_or_count(
    tmp_path, monkeypatch, batch_limit
):
    plan_path, qualification_path, supervisor_root, plan = _build_completed_cohort(
        tmp_path, monkeypatch, batch_limit=batch_limit
    )
    output = (tmp_path / "requests").resolve()

    receipt_path = materializer.materialize_publish_requests(
        plan_path=plan_path,
        accepted_qualification_receipt=qualification_path,
        supervisor_root=supervisor_root,
        runtime_id="gmufqeanr",
        output_dir=output,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_plan_sha256"] == _canonical_sha256(plan)
    assert receipt["source_plan_file_sha256"] == _sha256(plan_path)
    assert receipt["accepted_qualification_receipt_sha256"] == _sha256(
        qualification_path
    )
    assert receipt["render_contract"] == TEST_RENDER_CONTRACT
    assert receipt["batch_count"] == len(plan["batches"])
    assert receipt["vial_receipt_count"] == len(plan["execution_schedules"])
    assert len(receipt["requests"]) == len(plan["batches"])

    by_batch = {}
    forbidden_fragments = ("residual", "profile", "label", "oracle")
    for row in receipt["requests"]:
        request_path = output / row["path"]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert row["sha256"] == _sha256(request_path)
        assert request["batch_id"] == row["batch_id"]
        assert request["runtime_id"] == "gmufqeanr"
        assert not any(
            fragment in key.lower()
            for key in _all_keys(request)
            for fragment in forbidden_fragments
        )
        assert [vial["vial_id"] for vial in request["vials"]] == ["V1", "V2", "V3"]
        for vial in request["vials"]:
            source = Path(vial["receipt_path"])
            assert vial["receipt_sha256"] == _sha256(source)
        by_batch[request["batch_id"]] = request

    assert len({row["run_id"] for row in by_batch.values()}) == len(plan["batches"])
    assert {
        batch_id: request["cohort"] for batch_id, request in by_batch.items()
    } == {batch["batch_id"]: batch["batch_role"] for batch in plan["batches"]}

    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=output,
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_rejects_qualification_representation_drift_before_writing(tmp_path, monkeypatch):
    plan_path, qualification_path, supervisor_root, _ = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["arguments"]["count_driven_twin_liquid_mesh"] = False
    _write_json(qualification_path, qualification)

    with pytest.raises(ValueError, match="qualification|representation"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda timing: timing.update(report_hz=240),
        lambda timing: timing.update(physics_steps_per_second=120),
        lambda timing: timing.update(physics_substeps_per_report_interval=1),
        lambda timing: timing.update(diffusion_updates_per_report_interval=2),
    ],
)
def test_rejects_qualification_two_clock_drift_before_writing(
    tmp_path, monkeypatch, mutation
):
    plan_path, qualification_path, supervisor_root, _ = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    mutation(qualification["timing"])
    _write_json(qualification_path, qualification)

    with pytest.raises(ValueError, match="qualification_two_clock_contract_invalid"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda timing: timing.update(
            callbacks_observed=timing["callbacks_observed"] - 1
        ),
        lambda timing: timing.update(
            diffusion_update_count=timing["diffusion_update_count"] * 2
        ),
    ],
)
def test_rejects_case_two_clock_denominator_drift(
    tmp_path, monkeypatch, mutation
):
    plan_path, qualification_path, supervisor_root, _ = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    receipt_path = next(
        (supervisor_root / "cases").glob("CASE_*/app-output/receipt.json")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt["timing"])
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="case_receipt_two_clock_contract_invalid"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence["property_specs"].update(
            simulationPoints_root_layer_spec=True
        ),
        lambda evidence: evidence["simulation_points_runtime_evidence"].update(
            finite_xyz_rows=False, passed=False
        ),
    ],
)
def test_rejects_stale_or_nonfinite_simulation_points_binding(
    tmp_path, monkeypatch, mutation
):
    plan_path, qualification_path, supervisor_root, _ = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    receipt_path = next(
        (supervisor_root / "cases").glob("CASE_*/app-output/receipt.json")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt["point_instancer"]["runtime_layer_evidence"])
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="case_runtime_particle_binding_invalid"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


def _build_publications(request_bundle: Path) -> list[Path]:
    materialization = json.loads(
        (request_bundle / "request-materialization-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    publications = []
    for row in materialization["requests"]:
        request_path = request_bundle / row["path"]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        publication = request_bundle.parent / f"published-{request['batch_id']}"
        publication.mkdir()
        context = {
            "schema_version": 1,
            "artifact_type": "stage_batch_context",
            "split_version": "inn-7-splits-v2",
            "batches": [
                {
                    key: request[key]
                    for key in (
                        "batch_id",
                        "cohort",
                        "family_id",
                        "current_recipe_id",
                        "previous_recipe_id",
                        "run_id",
                        "split_role",
                        "programmed_recipe",
                    )
                }
            ],
        }
        _write_json(publication / "stage-batch-context.json", context)
        _write_json(
            publication / "f2-stage-capture-manifest.json",
            {"artifact_type": "f2_stage_capture_manifest", "batch_id": request["batch_id"]},
        )
        _write_json(
            publication / "f2-runtime-acceptance-receipt.json",
            {"artifact_type": "f2_runtime_acceptance_receipt", "status": "pass"},
        )
        output_names = (
            "f2-runtime-acceptance-receipt.json",
            "f2-stage-capture-manifest.json",
            "stage-batch-context.json",
        )
        publication_receipt = {
            "schema_version": 1,
            "artifact_type": "f2_stage_batch_publication_receipt",
            "status": "published",
            "data_origin": "twin",
            "real_data_used": False,
            "source_publish_request_sha256": _sha256(request_path),
            "source_receipts": [
                {**vial, "verified_artifact_count": 25}
                for vial in request["vials"]
            ],
            "outputs": {
                name: {
                    "path": name,
                    "sha256": _sha256(publication / name),
                    "bytes": (publication / name).stat().st_size,
                }
                for name in output_names
            },
            "claim_boundary": "synthetic artifact routing only",
        }
        _write_json(publication / "publication-receipt.json", publication_receipt)
        publications.append(publication.resolve())
    return publications


def test_merges_all_request_bound_publication_contexts_with_hash_receipt(
    tmp_path, monkeypatch
):
    plan_path, qualification_path, supervisor_root, plan = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    request_bundle = (tmp_path / "requests").resolve()
    materializer.materialize_publish_requests(
        plan_path=plan_path,
        accepted_qualification_receipt=qualification_path,
        supervisor_root=supervisor_root,
        runtime_id="gmufqeanr",
        output_dir=request_bundle,
    )
    publication_dirs = _build_publications(request_bundle)
    output = (tmp_path / "merged").resolve()

    receipt_path = materializer.merge_publication_contexts(
        request_bundle=request_bundle,
        publication_dirs=publication_dirs,
        output_dir=output,
    )

    assert receipt_path == output / "context-merge-receipt.json"
    merged_path = output / "stage-batch-context.json"
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert set(merged) == {
        "schema_version",
        "artifact_type",
        "split_version",
        "batches",
    }
    assert merged["artifact_type"] == "stage_batch_context"
    assert len(merged["batches"]) == len(plan["batches"])
    assert [row["batch_id"] for row in merged["batches"]] == sorted(
        row["batch_id"] for row in merged["batches"]
    )
    assert len({row["run_id"] for row in merged["batches"]}) == len(plan["batches"])
    assert not any(
        fragment in key.lower()
        for key in _all_keys(merged)
        for fragment in ("residual", "profile", "label", "oracle")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "merged"
    assert receipt["batch_count"] == len(plan["batches"])
    assert receipt["merged_context"]["sha256"] == _sha256(merged_path)
    assert receipt["merged_context"]["bytes"] == merged_path.stat().st_size
    assert len(receipt["source_publications"]) == len(plan["batches"])
    for row in receipt["source_publications"]:
        assert row["publication_receipt_sha256"] == _sha256(
            Path(row["publication_receipt_path"])
        )
        assert row["source_context_sha256"] == _sha256(
            Path(row["source_context_path"])
        )

    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        materializer.merge_publication_contexts(
            request_bundle=request_bundle,
            publication_dirs=publication_dirs,
            output_dir=output,
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_rejects_completed_case_receipt_with_source_id_drift(tmp_path, monkeypatch):
    plan_path, qualification_path, supervisor_root, plan = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    receipt_path = supervisor_root / "cases" / case_id / "app-output" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arguments"]["train_case_id"] = "CASE_wrong"
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="case_receipt_binding_mismatch"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("synthetic_profile_id", "PROFILE_hidden_oracle"),
        ("current_recipe_id", "S9999"),
    ],
)
def test_rejects_case_receipt_model_input_outside_exact_safe_record(
    tmp_path, monkeypatch, field, value
):
    plan_path, qualification_path, supervisor_root, plan = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    receipt_path = supervisor_root / "cases" / case_id / "app-output" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_schedule"]["model_input_record"][field] = value
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="case_receipt_binding_mismatch"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


def test_rejects_oracle_field_even_if_source_model_plan_lists_it(
    tmp_path, monkeypatch
):
    plan_path, qualification_path, supervisor_root, _ = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    artifacts = _source_artifacts()
    model_plan = artifacts["model-input-plan.json"]
    model_plan["model_input_allowlist"].append("synthetic_profile_id")
    for row in model_plan["vial_records"]:
        row["synthetic_profile_id"] = "PROFILE_hidden_oracle"
    monkeypatch.setattr(
        materializer,
        "build_train_residual_artifacts",
        lambda **_: deepcopy(artifacts),
    )

    with pytest.raises(ValueError, match="source_model_records_invalid"):
        materializer.materialize_publish_requests(
            plan_path=plan_path,
            accepted_qualification_receipt=qualification_path,
            supervisor_root=supervisor_root,
            runtime_id="gmufqeanr",
            output_dir=(tmp_path / "requests").resolve(),
        )
    assert not (tmp_path / "requests").exists()


def test_merge_rejects_hash_consistent_context_that_drifted_from_request(
    tmp_path, monkeypatch
):
    plan_path, qualification_path, supervisor_root, _ = _build_completed_cohort(
        tmp_path, monkeypatch
    )
    request_bundle = (tmp_path / "requests").resolve()
    materializer.materialize_publish_requests(
        plan_path=plan_path,
        accepted_qualification_receipt=qualification_path,
        supervisor_root=supervisor_root,
        runtime_id="gmufqeanr",
        output_dir=request_bundle,
    )
    publications = _build_publications(request_bundle)
    context_path = publications[0] / "stage-batch-context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["batches"][0]["run_id"] = "RUN_wrong"
    _write_json(context_path, context)
    publication_path = publications[0] / "publication-receipt.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["outputs"]["stage-batch-context.json"].update(
        sha256=_sha256(context_path), bytes=context_path.stat().st_size
    )
    _write_json(publication_path, publication)

    with pytest.raises(ValueError, match="context_request_mismatch"):
        materializer.merge_publication_contexts(
            request_bundle=request_bundle,
            publication_dirs=publications,
            output_dir=(tmp_path / "merged").resolve(),
        )
    assert not (tmp_path / "merged").exists()


def test_requests_cli_exposes_the_source_bound_materializer(tmp_path, monkeypatch, capsys):
    plan_path, qualification_path, supervisor_root, plan = _build_completed_cohort(
        tmp_path, monkeypatch, batch_limit=2
    )
    output = (tmp_path / "requests").resolve()

    assert (
        cli.main(
            [
                "requests",
                "--plan",
                str(plan_path),
                "--accepted-qualification-receipt",
                str(qualification_path),
                "--supervisor-root",
                str(supervisor_root),
                "--runtime-id",
                "gmufqeanr",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert str(output / "request-materialization-receipt.json") in capsys.readouterr().out
    receipt = json.loads(
        (output / "request-materialization-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["batch_count"] == len(plan["batches"])

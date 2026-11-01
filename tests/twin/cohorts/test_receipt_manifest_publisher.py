"""Tests for publishing accepted per-vial F2 receipts into stage manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _varied_case_id(ordinal: int) -> str:
    return f"CASE_{ordinal:016x}"


def _receipt(root: Path, vial_ordinal: int) -> Path:
    root.mkdir(parents=True)
    stage_steps = (1842, 3282, 4482)
    tanks = ("t200", "t400", "t100")
    pairs = []
    frame_paths = []
    frame_hashes = []
    mask_hashes = []
    for index, (step, tank) in enumerate(zip(stage_steps, tanks, strict=True), start=1):
        rgb = root / f"stage-{index}-{tank}.png"
        mask = root / f"stage-{index}-{tank}-mask.png"
        pixels = np.full((480, 640, 3), (20 * vial_ordinal, 30 * index, 80), np.uint8)
        liquid = np.zeros((480, 640), np.uint8)
        liquid[300:400, 285:335] = 255
        Image.fromarray(pixels, mode="RGB").save(rgb)
        Image.fromarray(liquid, mode="L").save(mask)
        frame_paths.append(rgb.name)
        frame_hashes.append(_sha(rgb))
        mask_hashes.append(_sha(mask))
        pairs.append(
            {
                "physics_step": step,
                "rgb_path": rgb.name,
                "rgb_sha256": _sha(rgb),
                "mask_path": mask.name,
                "mask_sha256": _sha(mask),
            }
        )
    payload = {
        "schema_version": 20,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "render_path": "point_instancer_sphere_emitter",
        "status": "pass",
        "versions": {"isaac": "6.2.0.0", "kit": "110.4.0", "physx": "110.4.4"},
        "provenance": {
            "script_sha256": "a" * 64,
            "requested_stage_sha256": "b" * 64,
        },
        "qualification": {"first_stage_only": False},
        "arguments": {
            "physics_substeps": 1,
            "particle_resolution_profile": "c20_2p5mm",
            "recipe8_effective_particle_volume_ml": 0.011272866937444935,
        },
        "plan": {
            "pours": [
                {
                    "ordinal": ordinal,
                    "tank_id": tank,
                    "programmed_volume_ml": 20.0,
                    "end_step": end_step,
                    "stage_capture_step": stage_step,
                }
                for ordinal, (tank, end_step, stage_step) in enumerate(
                    zip(tanks, (1842, 3282, 4446), stage_steps, strict=True), start=1
                )
            ]
        },
        "recipe8_leading_residual": {
            "model_input_record": {
                "current_recipe_id": "video5-recipe-8",
                "previous_recipe_id": "video5-recipe-7",
                "dispense_order": list(tanks),
                "programmed_ml": {"t100": 20.0, "t200": 20.0, "t400": 20.0},
            }
        },
        "render": {
            "resolution": [640, 480],
            "stage_frame_physics_steps": list(stage_steps),
            "stage_frame_paths": frame_paths,
            "stage_frame_sha256": frame_hashes,
        },
        "semantics": {"pairs": pairs},
        "timing": {
            "report_hz": 120,
            "report_interval_count": 4806,
            "executed_report_intervals": 4806,
            "physics_hz": 120,
            "physics_steps_per_second": 120,
            "physics_substeps_per_report_interval": 1,
            "physics_callback_count": 4806,
            "callbacks_expected": 4806,
            "callbacks_observed": 4806,
            "diffusion_updates_per_report_interval": 1,
            "diffusion_update_count": 4806,
            "pour_end_steps": [1842, 3282, 4446],
            "primary_final_capture_step": 4482,
            "primary_final_offset_s": 0.3,
        },
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _request(tmp_path: Path) -> dict[str, object]:
    vials = []
    for ordinal, vial_id in enumerate(("V1", "V2", "V3"), start=1):
        receipt = _receipt(tmp_path / vial_id, ordinal)
        vials.append(
            {
                "vial_id": vial_id,
                "receipt_path": str(receipt),
                "receipt_sha256": _sha(receipt),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "f2_stage_batch_publish_request",
        "data_origin": "twin",
        "provider": "brev",
        "runtime_name": "NVIDIA Isaac Sim",
        "runtime_id": "gmufqeanr",
        "family_id": "FAMILY-RECIPE8-QUAL",
        "batch_id": "BATCH-RECIPE8-QUAL",
        "run_id": "RUN-RECIPE8-QUAL-01",
        "cohort": "scored_target",
        "current_recipe_id": "video5-recipe-8",
        "previous_recipe_id": "video5-recipe-7",
        "split_role": "val",
        "split_version": "inn-7-splits-v2",
        "programmed_recipe": {
            "dispense_order": ["t200", "t400", "t100"],
            "previous_ml_by_tank": {"t100": 20, "t200": 20, "t400": 20},
            "current_ml_by_tank": {"t100": 20, "t200": 20, "t400": 20},
        },
        "vials": vials,
    }


def _varied_request(
    tmp_path: Path, *, three_stage: bool = False
) -> dict[str, object]:
    current_recipe_id = "8" if three_stage else "13"
    current_programmed_ml = (
        {"t100": 20, "t200": 20, "t400": 20}
        if three_stage
        else {"t100": 0, "t200": 50, "t400": 20}
    )
    dispense_order = (
        ["t200", "t400", "t100"] if three_stage else ["t200", "t400"]
    )
    vials = []
    for ordinal, vial_id in enumerate(("V1", "V2", "V3"), start=1):
        receipt = _receipt(tmp_path / f"varied-{vial_id}", ordinal)
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        previous_recipe_id = "7" if ordinal == 1 else current_recipe_id
        previous_programmed_ml = (
            {"t100": 20, "t200": 0, "t400": 50}
            if ordinal == 1
            else current_programmed_ml
        )
        payload["arguments"]["realtime_video"] = True
        if not three_stage:
            payload["plan"]["pours"] = payload["plan"]["pours"][:2]
            payload["plan"]["pours"][1]["end_step"] = 3246
            payload["plan"]["pours"][1]["stage_capture_step"] = 3282
        payload["plan"]["execution_schedule"] = {
            "model_input_record": {
                "batch_id": "BATCH-VARIED-VALIDATION",
                "case_id": _varied_case_id(ordinal),
                "current_programmed_ml": current_programmed_ml,
                "current_recipe_id": current_recipe_id,
                "data_origin": "twin",
                "previous_programmed_ml": previous_programmed_ml,
                "previous_recipe_id": previous_recipe_id,
                "programmed_dispense_order": dispense_order,
                "quantitative_evidence_ids": [],
                "run_id": "RUN-VALIDATION-01",
                "split_role": "validation",
                "split_version": "whole-run-varied-v3",
                "stage_image_evidence_ids": [],
                "vial_position": ordinal,
            },
            "renderer_case_binding": {
                "batch_id": "BATCH-VARIED-VALIDATION",
                "case_id": _varied_case_id(ordinal),
                "vial_ordinal": ordinal,
            },
        }
        if not three_stage:
            payload["render"]["stage_frame_physics_steps"] = [1842, 3282]
            payload["render"]["stage_frame_paths"] = payload["render"][
                "stage_frame_paths"
            ][:2]
            payload["render"]["stage_frame_sha256"] = payload["render"][
                "stage_frame_sha256"
            ][:2]
            payload["semantics"]["pairs"] = payload["semantics"]["pairs"][:2]
            payload["timing"]["pour_end_steps"] = [1842, 3246]
            payload["timing"]["primary_final_capture_step"] = 3282
            payload["timing"]["primary_final_offset_s"] = 0.3
        receipt.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        vials.append(
            {
                "case_id": _varied_case_id(ordinal),
                "vial_id": vial_id,
                "vial_position": ordinal,
                "previous_recipe_id": previous_recipe_id,
                "previous_ml_by_tank": previous_programmed_ml,
                "receipt_path": str(receipt),
                "receipt_sha256": _sha(receipt),
            }
        )
    return {
        "schema_version": 3,
        "artifact_type": "f2_varied_stage_batch_publish_request",
        "data_origin": "twin",
        "provider": "brev",
        "runtime_name": "NVIDIA Isaac Sim",
        "runtime_id": "brev-isaac62-varied-v3",
        "batch_id": "BATCH-VARIED-VALIDATION",
        "run_id": "RUN-VALIDATION-01",
        "cohort": "scored_target",
        "current_recipe_id": current_recipe_id,
        "split_role": "validation",
        "split_version": "whole-run-varied-v3",
        "programmed_recipe": {
            "dispense_order": dispense_order,
            "current_ml_by_tank": current_programmed_ml,
        },
        "vials": vials,
    }


def _bind_v4_request_to_source_cohort(
    request: dict[str, object], tmp_path: Path
) -> dict[str, object]:
    from twin.cohorts.varied_cohort_manifest_v4 import (
        build_varied_cohort_artifacts,
    )

    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    render_plan = artifacts["render-schedule-plan.json"]
    model_plan = artifacts["model-input-plan.json"]
    rows = [
        row
        for row in model_plan["vial_records"]
        if row["current_recipe_id"] == "8"
    ]
    rows.sort(key=lambda row: row["vial_position"])
    assert len(rows) == 3
    request.update(
        {
            "schema_version": 4,
            "batch_id": rows[0]["batch_id"],
            "run_id": rows[0]["run_id"],
            "current_recipe_id": rows[0]["current_recipe_id"],
            "split_role": rows[0]["split_role"],
            "split_version": "whole-run-varied-v4",
            "runtime_id": "brev-isaac62-varied-v4",
            "programmed_recipe": {
                "dispense_order": rows[0]["programmed_dispense_order"],
                "current_ml_by_tank": rows[0]["current_programmed_ml"],
            },
        }
    )
    for vial, row in zip(request["vials"], rows, strict=True):
        vial.update(
            {
                "case_id": row["case_id"],
                "previous_recipe_id": row["previous_recipe_id"],
                "previous_ml_by_tank": row["previous_programmed_ml"],
            }
        )
        receipt = Path(vial["receipt_path"])
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload["plan"]["execution_schedule"] = {
            "model_input_record": row,
            "renderer_case_binding": {
                "batch_id": row["batch_id"],
                "case_id": row["case_id"],
                "vial_ordinal": row["vial_position"],
            },
        }
        receipt.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        vial["receipt_sha256"] = _sha(receipt)

    case_ids = [
        schedule["case_binding"]["case_id"]
        for schedule in render_plan["execution_schedules"]
    ]
    plan_sha = _canonical_sha256(render_plan)
    binding_root = tmp_path / "v4-cohort-binding"
    launch_path = binding_root / "cohort-launch.json"
    merge_path = binding_root / "shard-merge.json"
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    launch_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "physx_train_residual_serial_cohort_launch",
                "case_ids": case_ids,
                "plan_sha256": plan_sha,
                "expected_output_contract": {
                    "case_count": 30,
                    "cases": {case_id: {} for case_id in case_ids},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    merge_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "physx_train_residual_shard_merge",
                "status": "pass",
                "case_ids": case_ids,
                "merged_case_count": 30,
                "plan_sha256": plan_sha,
                "shard_count": 30,
                "source_shards": [
                    {"index": index, "case_ids": [case_id]}
                    for index, case_id in enumerate(case_ids)
                ],
                "full_resume_summary": {
                    "status": "pass",
                    "planned_case_count": 30,
                    "validated_completed_case_count": 30,
                    "plan_sha256": plan_sha,
                    "case_results": [
                        {
                            "case_id": case_id,
                            "outcome": "recognized_completed",
                        }
                        for case_id in case_ids
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    request.update(
        {
            "source_render_plan_sha256": plan_sha,
            "cohort_launch_receipt": {
                "path": str(launch_path),
                "sha256": _sha(launch_path),
            },
            "cohort_merge_receipt": {
                "path": str(merge_path),
                "sha256": _sha(merge_path),
            },
        }
    )
    return request


def test_publish_varied_v3_two_stage_batch_preserves_height_warnings(
    tmp_path, monkeypatch
):
    import twin.cohorts.receipt_manifest_publisher as publisher
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

    warning = {
        "warning_type": "particle_height_outside_geometry_volume_bounds",
        "hard_gate": False,
        "claim": "diagnostic_warning_not_a_physics_pass",
        "measured_particle_height_m": 0.09,
        "lower_bound_m": 0.12,
        "upper_bound_m": 0.14,
    }
    monkeypatch.setattr(
        publisher,
        "validate_paint_ball_receipt",
        lambda _value: (_ for _ in ()).throw(AssertionError("legacy validator called")),
    )
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {
            "status": "pass",
            "verified_artifact_count": 52,
            "failures": [],
            "warnings": [warning],
        },
    )
    output = tmp_path / "varied-published"

    record_path = publisher.publish_f2_varied_stage_batch(
        _varied_request(tmp_path), output
    )

    manifest_path = output / "f2-stage-capture-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publication = json.loads(record_path.read_text(encoding="utf-8"))
    runtime_receipt = json.loads(
        (output / "f2-runtime-acceptance-receipt.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3
    assert manifest["split_version"] == "whole-run-varied-v3"
    assert len(manifest["stage_events"]) == 2
    assert len(manifest["rows"]) == 6
    assert "family_id" not in manifest
    assert {
        row["case_id"]: row["previous_recipe_id"]
        for row in json.loads(
            (output / "stage-batch-context.json").read_text(encoding="utf-8")
        )["batches"][0]["vials"]
    } == {
        _varied_case_id(1): "7",
        _varied_case_id(2): "13",
        _varied_case_id(3): "13",
    }
    assert all(
        row["artifact_validation_warnings"] == [warning]
        for row in publication["source_receipts"]
    )
    assert runtime_receipt["schema_version"] == 3
    assert runtime_receipt["artifact_type"] == "f2_varied_runtime_acceptance_receipt"
    assert runtime_receipt["hard_gates"]["rendered_body_volume_height"] is True
    assert "volume_height" not in runtime_receipt["hard_gates"]
    assert runtime_receipt["particle_height_diagnostic"] == {
        "bounds_passed": False,
        "claim": "diagnostic_warning_not_a_physics_pass",
        "hard_gate": False,
        "physics_pass_claimed": False,
        "status": "warning",
        "vial_count": 3,
        "vials": [
            {
                "bounds_passed": False,
                "case_id": _varied_case_id(ordinal),
                "status": "warning",
                "vial_id": f"V{ordinal}",
                "warning_count": 1,
            }
            for ordinal in (1, 2, 3)
        ],
        "warned_vial_count": 3,
        "warning_count": 3,
    }
    product_bound = bind_stage_capture_manifest(manifest_path)
    twin_bound = publisher.bind_stage_capture_manifest(manifest_path)
    assert twin_bound["denominators"] == {
        "image_measurements": 6,
        "stage_statistics": 6,
    }
    for field in (
        "capture_contract_version",
        "batch_id",
        "run_id",
        "recipe_id",
        "split_role",
        "split_version",
        "stage_events",
        "source_manifest_sha256",
        "image_locators",
        "denominators",
    ):
        assert twin_bound[field] == product_bound[field]
    measurement_fields = (
        "image_id",
        "batch_id",
        "run_id",
        "vial_id",
        "recipe_id",
        "split_role",
        "split_version",
        "stage_id",
        "stage_ordinal",
        "stage_event",
        "view_id",
        "validity",
    )
    assert [
        {field: row[field] for field in measurement_fields}
        for row in twin_bound["image_measurements"]
    ] == [
        {field: row[field] for field in measurement_fields}
        for row in product_bound["image_measurements"]
    ]


def test_publish_varied_v3_three_stage_batch_has_nine_pairs(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {
            "status": "pass",
            "verified_artifact_count": 52,
            "failures": [],
            "warnings": [],
        },
    )
    output = tmp_path / "varied-three-stage-published"

    publisher.publish_f2_varied_stage_batch(
        _varied_request(tmp_path, three_stage=True), output
    )

    bound = bind_stage_capture_manifest(output / "f2-stage-capture-manifest.json")
    assert len(bound["stage_events"]) == 3
    assert bound["denominators"] == {
        "image_measurements": 9,
        "stage_statistics": 9,
    }
    runtime_receipt = json.loads(
        (output / "f2-runtime-acceptance-receipt.json").read_text(encoding="utf-8")
    )
    assert runtime_receipt["particle_height_diagnostic"]["status"] == "within_bounds"
    assert runtime_receipt["particle_height_diagnostic"]["bounds_passed"] is True
    assert runtime_receipt["particle_height_diagnostic"]["warning_count"] == 0
    assert runtime_receipt["particle_height_diagnostic"]["physics_pass_claimed"] is False
    assert runtime_receipt["particle_height_diagnostic"]["claim"] == (
        "particle_height_within_bounds_diagnostic_only"
    )


def test_publish_varied_v4_request_propagates_split_version(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _bind_v4_request_to_source_cohort(
        _varied_request(tmp_path, three_stage=True), tmp_path
    )
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {
            "status": "pass",
            "verified_artifact_count": 52,
            "failures": [],
            "warnings": [],
        },
    )

    output = tmp_path / "varied-v4-published"
    publisher.publish_f2_varied_stage_batch(request, output)

    manifest = json.loads(
        (output / "f2-stage-capture-manifest.json").read_text(encoding="utf-8")
    )
    context = json.loads(
        (output / "stage-batch-context.json").read_text(encoding="utf-8")
    )
    assert manifest["split_version"] == "whole-run-varied-v4"
    assert context["split_version"] == "whole-run-varied-v4"


def test_publish_varied_v4_rejects_a_failed_bound_merge(tmp_path):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _bind_v4_request_to_source_cohort(
        _varied_request(tmp_path, three_stage=True), tmp_path
    )
    merge_path = Path(request["cohort_merge_receipt"]["path"])
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    merge["status"] = "failed"
    merge_path.write_text(json.dumps(merge, indent=2, sort_keys=True) + "\n")
    request["cohort_merge_receipt"]["sha256"] = _sha(merge_path)

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="cohort_merge_binding_invalid",
    ):
        publisher.publish_f2_varied_stage_batch(
            request, tmp_path / "failed-merge-v4-published"
        )


def test_publish_varied_v4_rejects_a_noncanonical_batch_request(tmp_path):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _bind_v4_request_to_source_cohort(
        _varied_request(tmp_path, three_stage=True), tmp_path
    )
    request["vials"][0]["previous_recipe_id"] = "13"

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="v4_request_source_binding_invalid",
    ):
        publisher.publish_f2_varied_stage_batch(
            request, tmp_path / "noncanonical-v4-published"
        )


def test_varied_binder_rejects_particle_height_physics_pass_claim(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
    )

    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {
            "status": "pass",
            "verified_artifact_count": 52,
            "failures": [],
            "warnings": [],
        },
    )
    output = tmp_path / "varied-contradictory-runtime"
    publisher.publish_f2_varied_stage_batch(
        _varied_request(tmp_path, three_stage=True), output
    )
    runtime_path = output / "f2-runtime-acceptance-receipt.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["particle_height_diagnostic"]["physics_pass_claimed"] = True
    runtime_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output / "f2-stage-capture-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_receipt"]["sha256"] = _sha(runtime_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest_path)
    assert caught.value.reason_code == "runtime_receipt_invalid"
    with pytest.raises(publisher.CaptureManifestError) as twin_caught:
        publisher.bind_stage_capture_manifest(manifest_path)
    assert twin_caught.value.reason_code == "runtime_receipt_invalid"


def test_varied_request_rejects_zero_volume_tank_in_active_order(tmp_path):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _varied_request(tmp_path)
    request["programmed_recipe"]["dispense_order"].append("t100")

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="programmed_dispense_order_invalid",
    ):
        publisher.publish_f2_varied_stage_batch(
            request, tmp_path / "invalid-active-order"
        )


def test_varied_receipt_requires_its_own_predecessor_context(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _varied_request(tmp_path)

    def use_first_vial_predecessor(receipt):
        model = receipt["plan"]["execution_schedule"]["model_input_record"]
        model["previous_recipe_id"] = "7"
        model["previous_programmed_ml"] = {"t100": 20, "t200": 0, "t400": 50}

    _rewrite_receipt(request, 1, use_first_vial_predecessor)
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {
            "status": "pass",
            "verified_artifact_count": 52,
            "failures": [],
            "warnings": [],
        },
    )

    with pytest.raises(
        publisher.F2ReceiptPublicationError, match="receipt_context_mismatch"
    ):
        publisher.publish_f2_varied_stage_batch(
            request, tmp_path / "wrong-predecessor"
        )


def test_programmed_recipe_accepts_zero_tank_with_positive_recipe_totals():
    import twin.cohorts.receipt_manifest_publisher as publisher

    recipe = {
        "dispense_order": ["t200", "t400", "t100"],
        "previous_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
        "current_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
    }

    normalized = publisher._programmed_recipe(recipe)

    assert normalized["previous_ml_by_tank"]["t200"] == 0.0
    assert normalized["current_ml_by_tank"]["t200"] == 0.0


@pytest.mark.parametrize(
    "bad_value",
    [-1, False, float("inf"), float("nan")],
    ids=["negative", "bool", "infinite", "nan"],
)
def test_programmed_recipe_rejects_invalid_tank_volume(bad_value):
    import twin.cohorts.receipt_manifest_publisher as publisher

    recipe = {
        "dispense_order": ["t200", "t400", "t100"],
        "previous_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
        "current_ml_by_tank": {"t100": 25, "t200": bad_value, "t400": 35},
    }

    with pytest.raises(publisher.F2ReceiptPublicationError, match="t200_invalid"):
        publisher._programmed_recipe(recipe)


@pytest.mark.parametrize(
    "recipe_key", ["previous_ml_by_tank", "current_ml_by_tank"]
)
def test_programmed_recipe_rejects_all_zero_recipe(recipe_key):
    import twin.cohorts.receipt_manifest_publisher as publisher

    recipe = {
        "dispense_order": ["t200", "t400", "t100"],
        "previous_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
        "current_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
    }
    recipe[recipe_key] = {"t100": 0, "t200": 0, "t400": 0}

    with pytest.raises(publisher.F2ReceiptPublicationError, match="total_invalid"):
        publisher._programmed_recipe(recipe)


def test_publish_three_receipts_creates_bindable_nine_row_manifest_and_context(
    tmp_path, monkeypatch
):
    import twin.cohorts.receipt_manifest_publisher as publisher
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {"status": "pass", "verified_artifact_count": 9, "failures": []},
    )
    output = tmp_path / "published"

    record_path = publisher.publish_f2_stage_batch(_request(tmp_path), output)

    assert record_path == output / "publication-receipt.json"
    manifest_path = output / "f2-stage-capture-manifest.json"
    context_path = output / "stage-batch-context.json"
    runtime_path = output / "f2-runtime-acceptance-receipt.json"
    assert {path.name for path in output.iterdir()} == {
        record_path.name,
        manifest_path.name,
        context_path.name,
        runtime_path.name,
    }
    manifest = json.loads(manifest_path.read_text())
    context = json.loads(context_path.read_text())
    assert len(manifest["rows"]) == 9
    assert {row["vial_id"] for row in manifest["rows"]} == {"V1", "V2", "V3"}
    assert [event["tank_id"] for event in manifest["stage_events"]] == [
        "t200",
        "t400",
        "t100",
    ]
    assert [
        event["event_relative_timestamp_s"]
        for event in manifest["stage_events"]
    ] == pytest.approx([1842 / 120, 3282 / 120, 4482 / 120])
    assert [
        event["final_observation_offset_s"]
        for event in manifest["stage_events"]
    ] == pytest.approx([0.0, 0.0, (4482 - 4446) / 120])
    assert context["batches"][0]["batch_id"] == "BATCH-RECIPE8-QUAL"
    assert context["batches"][0]["cohort"] == "scored_target"
    assert not any(
        token in json.dumps({"manifest": manifest, "context": context}).lower()
        for token in ("residual_volume", "severity", "expected_disposition", "label_provenance")
    )
    bound = bind_stage_capture_manifest(manifest_path)
    assert bound["denominators"] == {"image_measurements": 9, "stage_statistics": 9}


def _passing_artifacts(_path):
    return {"status": "pass", "verified_artifact_count": 9, "failures": []}


def _rewrite_receipt(request, vial_index, mutation):
    entry = request["vials"][vial_index]
    path = Path(entry["receipt_path"])
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    entry["receipt_sha256"] = _sha(path)


def _rename_receipt_stage_frames(request, vial_index, frame_ids):
    entry = request["vials"][vial_index]
    receipt_path = Path(entry["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    renamed_rgb = []
    for pair, frame_id in zip(
        receipt["semantics"]["pairs"], frame_ids, strict=True
    ):
        old_rgb = receipt_path.parent / pair["rgb_path"]
        old_mask = receipt_path.parent / pair["mask_path"]
        new_rgb = receipt_path.parent / f"{frame_id}.png"
        new_mask = receipt_path.parent / f"{frame_id}-mask.png"
        old_rgb.rename(new_rgb)
        old_mask.rename(new_mask)
        pair["rgb_path"] = new_rgb.name
        pair["mask_path"] = new_mask.name
        renamed_rgb.append(new_rgb.name)
    receipt["render"]["stage_frame_paths"] = renamed_rgb
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    entry["receipt_sha256"] = _sha(receipt_path)


def test_publish_accepts_capture_local_frame_ids(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    expected = {
        "V1": ["frame_0007", "frame_0012", "frame_0017"],
        "V2": ["frame_0009", "frame_0017", "frame_0025"],
        "V3": ["frame_0007", "frame_0012", "frame_0017"],
    }
    for vial_index, vial_id in enumerate(("V1", "V2", "V3")):
        _rename_receipt_stage_frames(request, vial_index, expected[vial_id])
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    publisher.publish_f2_stage_batch(request, tmp_path / "published")

    manifest = json.loads(
        (tmp_path / "published" / "f2-stage-capture-manifest.json").read_text()
    )
    assert [event["frame_id"] for event in manifest["stage_events"]] == [
        None,
        None,
        None,
    ]
    assert {
        vial_id: [
            row["frame_id"]
            for row in manifest["rows"]
            if row["vial_id"] == vial_id
        ]
        for vial_id in expected
    } == expected


def test_twin_binder_rejects_malformed_row_frame_id_like_product(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
    )

    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)
    output = tmp_path / "published-malformed-frame"
    publisher.publish_f2_stage_batch(_request(tmp_path), output)
    manifest_path = output / "f2-stage-capture-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][0]["frame_id"] = ""
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(CaptureManifestError) as product_caught:
        bind_stage_capture_manifest(manifest_path)
    with pytest.raises(publisher.CaptureManifestError) as twin_caught:
        publisher.bind_stage_capture_manifest(manifest_path)
    assert product_caught.value.reason_code == "frame_id_invalid"
    assert twin_caught.value.reason_code == product_caught.value.reason_code


@pytest.mark.parametrize(
    "mutation",
    [
        lambda timing: timing.pop("report_hz"),
        lambda timing: timing.update(report_hz=240),
        lambda timing: timing.update(physics_hz=240),
        lambda timing: timing.update(physics_steps_per_second=240),
        lambda timing: timing.update(physics_substeps_per_report_interval=2),
        lambda timing: timing.update(physics_callback_count=9612),
        lambda timing: timing.update(callbacks_observed=9612),
        lambda timing: timing.update(diffusion_update_count=9612),
    ],
)
def test_receipt_requires_consistent_two_clock_binding(
    tmp_path, monkeypatch, mutation
):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(request, 0, lambda receipt: mutation(receipt["timing"]))
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="receipt_two_clock_contract_invalid",
    ):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arguments: arguments.update(physics_substeps=2),
        lambda arguments: arguments.update(particle_resolution_profile="c21_2p0mm"),
        lambda arguments: arguments.update(
            recipe8_effective_particle_volume_ml=0.005771707871971808
        ),
    ],
)
def test_receipt_requires_the_qualified_legacy_runtime_profile(
    tmp_path, monkeypatch, mutation
):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(request, 0, lambda receipt: mutation(receipt["arguments"]))
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="receipt_legacy_runtime_contract_invalid",
    ):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_count_driven_receipts_publish_and_bind_exact_representation(
    tmp_path, monkeypatch
):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    for index in range(3):
        _rewrite_receipt(
            request,
            index,
            lambda receipt: receipt.update(
                render_path="count_driven_twin_liquid_mesh"
            ),
        )
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    record_path = publisher.publish_f2_stage_batch(
        request, tmp_path / "count-driven-published"
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert {
        row["render_path"] for row in record["source_receipts"]
    } == {"count_driven_twin_liquid_mesh"}


def test_mixed_vial_render_representations_fail_closed(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(
        request,
        1,
        lambda receipt: receipt.update(
            render_path="count_driven_twin_liquid_mesh"
        ),
    )
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="vial_receipt_contract_mismatch",
    ):
        publisher.publish_f2_stage_batch(request, tmp_path / "mixed-published")


def test_declared_receipt_hash_mismatch_fails_before_publication(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    Path(request["vials"][0]["receipt_path"]).write_text("{}\n")
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="receipt_hash_mismatch"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")

    assert not (tmp_path / "published").exists()


def test_public_paint_ball_validator_is_mandatory(tmp_path):
    import twin.cohorts.receipt_manifest_publisher as publisher

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="receipt_substantive_validation_failed",
    ):
        publisher.publish_f2_stage_batch(
            _request(tmp_path),
            tmp_path / "published",
        )


def test_official_artifact_validator_is_mandatory(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="receipt_artifact_validation_failed",
    ):
        publisher.publish_f2_stage_batch(_request(tmp_path), tmp_path / "published")


@pytest.mark.parametrize(
    "mutation,artifact_result,reason",
    [
        (
            lambda receipt: receipt.update(status="insufficient_evidence"),
            _passing_artifacts(None),
            "receipt_not_accepted_full_run",
        ),
        (
            lambda _receipt: None,
            {
                "status": "insufficient_evidence",
                "verified_artifact_count": 8,
                "failures": ["artifact:stage-mask:missing"],
            },
            "receipt_artifact_validation_failed",
        ),
    ],
)
def test_nonpass_receipt_or_missing_artifact_gate_fails_closed(
    tmp_path, monkeypatch, mutation, artifact_result, reason
):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(request, 0, mutation)
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: artifact_result,
    )

    with pytest.raises(publisher.F2ReceiptPublicationError, match=reason):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_missing_receipt_declared_stage_pair_fails_closed(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(request, 1, lambda receipt: receipt["semantics"]["pairs"].pop(1))
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="receipt_stage_pair_missing"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


@pytest.mark.parametrize("field", ["current_recipe_id", "run_id"])
def test_receipt_recipe_or_run_context_mismatch_fails_closed(
    tmp_path, monkeypatch, field
):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)

    def mutate(receipt):
        record = receipt["recipe8_leading_residual"]["model_input_record"]
        record[field] = "WRONG"

    _rewrite_receipt(request, 2, mutate)
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="receipt_context_mismatch"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_duplicate_vial_id_is_rejected(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    request["vials"][2]["vial_id"] = "V2"
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="vial_ids_invalid"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_one_receipt_cannot_be_reused_for_two_vials(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    request["vials"][2]["receipt_path"] = request["vials"][1]["receipt_path"]
    request["vials"][2]["receipt_sha256"] = request["vials"][1][
        "receipt_sha256"
    ]
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="vial_receipts_not_distinct",
    ):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_publication_is_no_clobber(tmp_path, monkeypatch):
    import twin.cohorts.receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    output = tmp_path / "published"
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    publisher.publish_f2_stage_batch(request, output)
    before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    with pytest.raises(FileExistsError):
        publisher.publish_f2_stage_batch(request, output)

    assert before == {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }


def test_publisher_cli_is_help_safe():
    script = ROOT / "cli/twin/cohorts" / "publish_f2_stage_batch.py"

    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout
    assert "--output-dir" in completed.stdout

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.inn67_bank_materializer import (  # noqa: E402
    ACCEPTED_CNN_PREDICTOR,
    ACCEPTED_OCCLUSION_VALIDATOR_SHA256,
    MaterializationError,
    materialize_reference_bank,
)
from fluidmix.inn67_reference_bank import build_reference_bank_bundle  # noqa: E402


def _bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value if isinstance(value, bytes) else _bytes(value)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _write_unsorted(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _amendment() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "inn67_pre_access_protocol_amendment",
        "status": "frozen_before_first_render",
        "owner_id": "01a18128-8dc4-7780-ace1-4d816b49d176",
        "decision_author": "Liam",
        "decision_time": "2026-10-30T06:20:00-05:00",
        "recorded_before_any_inn67_render": True,
        "supersedes_clause": "no_shared_recipe_role",
        "override_scope": "INN-67 development reference bank only",
        "what_changed": {
            "recipe_ids_and_run_presets_may_match_wp1_test": True,
            "recipe_ids": ["S0439", "S0440", "S0441", "S0442", "S0443", "S0444"],
            "meaning": "Clean development renders may use the same recipe identities, programmed volumes, tank order and palette preset as WP1 test so they act as earlier clean same-line batches.",
        },
        "separation_that_remains_exact": {
            "shared_case_ids": False,
            "shared_run_ids": False,
            "shared_batch_ids": False,
            "shared_image_or_video_hashes": False,
            "shared_source_records": False,
            "wp1_test_bytes_opened_to_build_bank": False,
            "video5_bytes_opened_to_build_bank": False,
            "bank_partition": "development",
            "wp1_partition": "test",
            "video5_status": "sealed",
        },
        "effective_delta_e_freeze_scope": {
            "reference_donor_batches": ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"],
            "configuration_freeze_target_batches": ["INN67-DEV-RB-B03"],
            "wp1_test_used_for_configuration": False,
            "video5_used_for_configuration": False,
            "note": "Batches 1 and 2 supply known-good references; the configuration is selected and frozen using development batch 3 only, before any WP1 or video-5 access.",
        },
        "reason": "Liam approved the plant-history analogue so every first-after-changeover vial has legitimate earlier same-recipe references from independent clean development batches.",
        "bound_freeze_inputs": {
            "reference_bank_plan_sha256": "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79",
            "reference_bank_inventory_sha256": "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95",
            "known_good_profile_sha256": "47bc0b6f483a121a28871b3eb74ec27efb79996b83d03d8c45d6f5b0b9cdeb7c",
            "delta_e_protocol_sha256": "8b875c93f2d9d61871dd03a187ada847239b3c2e608a18f167ceb2d1646431d6",
            "leakage_audit_sha256": "26b850654e326d46d46ec65bfe74e4356270e7a84608b4576b86e4bc9bdc4c0e",
            "no_access_attestation_sha256": "0ccd7d31632463ceb9c63d6103616846ad28c94d1aeff98af88bceb5b4db3731",
        },
        "render_or_test_access_performed": False,
        "receipt_sha256": "3cc4212f84a6a011b0d3dd579cb6cf817285d3a946598908756f6c95c9f349db",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    bundle = build_reference_bank_bundle(ROOT)
    plan_root = tmp_path / "plan"
    hashes: dict[str, str] = {}
    keep = {
        "reference-bank-plan.json",
        "known-good-profile.json",
        "delta-e-freeze-config.json",
        "leakage-audit.json",
        "no-access-attestation.json",
    }
    for relative, value in bundle.items():
        if relative in keep or relative.endswith("/render-schedule-plan.json"):
            hashes[relative] = _write(plan_root / relative, value)
    hashes["PLAN.md"] = "7bcaa8d7a59cc23b5fb9ab930e7bbcc8ba1dfa65af5d4762709593215e99e06f"
    hashes["resource-preflight.json"] = "fe72d39f7933bc1afcfafcc8d50cc0a79bde3e4b26fa15775d70f4ef73462902"
    for relative, value in bundle.items():
        if relative.endswith("/label-provenance.json") or relative.endswith("/model-input-plan.json"):
            hashes[relative] = hashlib.sha256(_bytes(value)).hexdigest()
    inventory = {
        "schema_version": 1,
        "artifact_type": "inn67_reference_bank_plan_hash_inventory",
        "files": hashes,
    }
    assert _write(plan_root / "artifact-hashes.json", inventory) == (
        "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95"
    )

    amendment_path = tmp_path / "pre-access-protocol-amendment.json"
    assert _write_unsorted(amendment_path, _amendment()) == (
        "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8"
    )
    materialization_amendment_path = tmp_path / "materialization-protocol-amendment.json"
    assert _write(
        materialization_amendment_path,
        (ROOT / "tests/fixtures/inn67-materialization-protocol-amendment.json").read_bytes(),
    ) == "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c"
    plan = bundle["reference-bank-plan.json"]
    profile_hash = hashes["known-good-profile.json"]

    render_inventory = {
        "schema_version": 1,
        "artifact_type": "inn67_accepted_render_inventory",
        "status": "accepted",
        "reference_bank_plan_sha256": hashes["reference-bank-plan.json"],
        "amendment_receipt_sha256": _amendment()["receipt_sha256"],
        "subruns": [],
    }
    receipt_by_case: dict[str, tuple[str, str]] = {}
    for subrun in plan["subruns"]:
        subrun_id = subrun["subrun_id"]
        root = tmp_path / "renders" / subrun_id
        cases = [row for row in plan["cases"] if row["subrun_id"] == subrun_id]
        case_rows = []
        supervisor_cases = []
        render_plan_hash = hashes[subrun["render_plan_artifact"]]
        for case in cases:
            renderer_case = case["renderer_case_id"]
            receipt = {
                "schema_version": 2,
                "artifact_type": "physx_three_pour_receipt",
                "status": "pass",
                "case_id": renderer_case,
                "source_plan_sha256": render_plan_hash,
                "known_good_profile_sha256": profile_hash,
            }
            receipt_path = root / "cases" / renderer_case / "app-output" / "receipt.json"
            receipt_hash = _write(receipt_path, receipt)
            validation = {
                "schema_version": 1,
                "artifact_type": "inn67_occlusion_aware_artifact_validation",
                "status": "pass",
                "failures": [],
                "case_id": renderer_case,
                "source_render_receipt_sha256": receipt_hash,
                "source_render_plan_sha256": render_plan_hash,
                "known_good_profile_sha256": profile_hash,
                "validator_implementation_sha256": ACCEPTED_OCCLUSION_VALIDATOR_SHA256,
                "validator_result": {
                    "status": "pass",
                    "verified_artifact_count": 1,
                    "failures": [],
                    "warnings": [],
                    "offline_revalidations": [],
                },
            }
            validation_path = root / "validation" / f"{renderer_case}.json"
            validation_hash = _write(validation_path, validation)
            receipt_by_case[case["case_id"]] = (receipt_hash, validation_hash)
            case_rows.append(
                {
                    "renderer_case_id": renderer_case,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": receipt_hash,
                    "validation_receipt_path": str(validation_path),
                    "validation_receipt_sha256": validation_hash,
                }
            )
            supervisor_cases.append(
                {
                    "case_id": renderer_case,
                    "outcome": "passed",
                    "evidence": {"artifact_validation": {"status": "pass", "failures": []}},
                }
            )
        supervisor = {
            "schema_version": 1,
            "artifact_type": "physx_train_residual_serial_cohort_attempt",
            "status": "pass",
            "plan_sha256": render_plan_hash,
            "planned_case_count": 3,
            "validated_completed_case_count": 3,
            "case_results": supervisor_cases,
        }
        supervisor_path = root / "supervisor-attempts" / "attempt-0001.json"
        supervisor_hash = _write(supervisor_path, supervisor)
        render_inventory["subruns"].append(
            {
                "subrun_id": subrun_id,
                "render_root": str(root),
                "render_plan_sha256": render_plan_hash,
                "supervisor_receipt_path": str(supervisor_path),
                "supervisor_receipt_sha256": supervisor_hash,
                "cases": case_rows,
            }
        )
    render_inventory_path = tmp_path / "render-inventory.json"
    _write(render_inventory_path, render_inventory)

    store_path = tmp_path / "store" / "evidence.sqlite3"
    store_hash = _write(store_path, b"fixture-store-v1")
    records = []
    cursor = 0
    for case in sorted(plan["cases"], key=lambda row: (row["batch_id"], row["subrun_id"], row["recipe_id"])):
        receipt_hash, validation_hash = receipt_by_case[case["case_id"]]
        batch_number = int(case["batch_id"][-2:])
        for stage in case["stages"]:
            records.append(
                {
                    "evidence_id": f"E-{case['case_id']}-{stage['ordinal']}",
                    "event_cursor": cursor,
                    "batch_id": case["batch_id"],
                    "run_id": case["run_id"],
                    "case_id": case["case_id"],
                    "recipe_id": case["recipe_id"],
                    "split_role": "development",
                    "stage_id": f"dispense-{stage['ordinal']}-{stage['tank_id']}",
                    "stage_ordinal": stage["ordinal"],
                    "tank_id": stage["tank_id"],
                    "rgb_sha256": hashlib.sha256(f"rgb-{cursor}".encode()).hexdigest(),
                    "mask_sha256": hashlib.sha256(f"mask-{cursor}".encode()).hexdigest(),
                    "mask_source": "predicted",
                    "median_lab": [
                        50.0 + int(case["recipe_id"][-2:]) / 10.0,
                        float(stage["ordinal"]),
                        {1: -0.25, 2: 0.25, 3: 0.0}[batch_number],
                    ],
                    "source_render_receipt_sha256": receipt_hash,
                    "source_validation_receipt_sha256": validation_hash,
                    "source_render_plan_sha256": hashes[
                        next(
                            item["render_plan_artifact"]
                            for item in plan["subruns"]
                            if item["subrun_id"] == case["subrun_id"]
                        )
                    ],
                    "known_good_profile_sha256": profile_hash,
                }
            )
            cursor += 1
    records_hash = _canonical(records)
    population = {
        "schema_version": 1,
        "artifact_type": "inn67_predicted_evidence_store_population",
        "status": "pass",
        "store_sha256": store_hash,
        "predictor": ACCEPTED_CNN_PREDICTOR,
        "record_count": 48,
        "records_sha256": records_hash,
        "evidence_ids": [row["evidence_id"] for row in records],
    }
    population_path = tmp_path / "store" / "population-receipt.json"
    population_hash = _write(population_path, population)
    export = {
        "schema_version": 1,
        "artifact_type": "inn67_predicted_store_export",
        "status": "accepted",
        "predictor": ACCEPTED_CNN_PREDICTOR,
        "store": {
            "database_path": str(store_path),
            "database_sha256": store_hash,
            "population_receipt_path": str(population_path),
            "population_receipt_sha256": population_hash,
        },
        "records": records,
    }
    export_path = tmp_path / "predicted-store-export.json"
    _write(export_path, export)
    return (
        plan_root,
        amendment_path,
        materialization_amendment_path,
        render_inventory_path,
        export_path,
    )


def test_materializes_two_donor_bank_and_freezes_only_from_b03(tmp_path: Path):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)
    output = tmp_path / "output"
    result = materialize_reference_bank(
        plan_root=plan_root,
        amendment_path=amendment,
        materialization_amendment_path=materialization_amendment,
        render_inventory_path=renders,
        predicted_store_export_path=predicted,
        output_dir=output,
    )

    assert result["status"] == "published"
    bank = json.loads((output / "development-reference-bank.json").read_text())
    frozen = json.loads((output / "delta-e-frozen-config.json").read_text())
    assert bank["denominators"] == {
        "donor_batches": 2,
        "donor_records": 32,
        "configuration_targets": 16,
        "peers_per_configuration_target": 2,
    }
    assert {row["batch_id"] for row in bank["records"]} == {
        "INN67-DEV-RB-B01", "INN67-DEV-RB-B02"
    }
    assert len(bank["history_bindings"]) == 16
    assert all(len(row["earlier_peer_evidence_ids"]) == 2 for row in bank["history_bindings"])
    assert frozen["selection_input_batch_ids"] == ["INN67-DEV-RB-B03"]
    assert frozen["reference_geometry_batch_ids"] == ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"]
    assert frozen["config_id"] == "v3-relative-diagnostic-inherited-stage-v1"
    assert frozen["tight_peer_pair_max_delta_e76"] == 2.0
    assert frozen["target_outlier_margin_delta_e76"] == 1.0
    assert frozen["decision_rule"] == (
        "peer_delta = dE76(B01,B02); threshold = peer_delta/2 + 1.0; "
        "require peer_delta <= 2.0 and B03 target_to_center_delta_e76 <= threshold"
    )
    assert len(frozen["recipe_stage_configurations"]) == 16
    assert all(
        row["consistent_with_two_peer_mode"] is True
        and row["relative_status"] == "consistent_with_two_peer_mode"
        and row["peer_pair_delta_e76"] <= 2.0
        and row["target_to_reference_center_delta_e76"] <= row["threshold_delta_e76"]
        for row in frozen["recipe_stage_configurations"]
    )
    receipt = json.loads((output / "materialization-receipt.json").read_text())
    assert receipt["provenance"]["materialization_protocol_amendment_sha256"] == (
        "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c"
    )
    with pytest.raises(MaterializationError, match="output_directory_exists"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=output,
        )


def test_materialization_protocol_amendment_drift_fails_closed(tmp_path: Path):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)
    value = json.loads(materialization_amendment.read_text())
    value["frozen_rule"]["peer_pair_delta_e76_max"] = 999.0
    _write(materialization_amendment, value)

    with pytest.raises(MaterializationError, match="materialization_amendment_hash_mismatch"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value["records"][0].update(mask_source="oracle"), "mask_source_not_predicted"),
        (lambda value: value["records"][0].update(recipe_id="S9999"), "record_plan_binding_mismatch"),
        (lambda value: value["records"][0].update(split_role="test"), "forbidden_test_or_video5_locator"),
    ],
)
def test_fails_closed_for_untrusted_or_mixed_store_rows(tmp_path: Path, mutation, code: str):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)
    value = json.loads(predicted.read_text())
    mutation(value)
    population_path = Path(value["store"]["population_receipt_path"])
    population = json.loads(population_path.read_text())
    population["records_sha256"] = _canonical(value["records"])
    value["store"]["population_receipt_sha256"] = _write(population_path, population)
    _write(predicted, value)
    with pytest.raises(MaterializationError, match=code):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "output",
        )


def test_missing_subrun_and_profile_drift_fail_before_publication(tmp_path: Path):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)
    inventory = json.loads(renders.read_text())
    inventory["subruns"].pop()
    _write(renders, inventory)
    with pytest.raises(MaterializationError, match="render_subruns_denominator_invalid"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "missing-output",
        )

    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path / "profile")
    inventory = json.loads(renders.read_text())
    case = inventory["subruns"][0]["cases"][0]
    receipt_path = Path(case["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["known_good_profile_sha256"] = "0" * 64
    case["receipt_sha256"] = _write(receipt_path, receipt)
    _write(renders, inventory)
    with pytest.raises(MaterializationError, match="case_receipt_not_accepted"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "profile-output",
        )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(artifact_type="not_the_accepted_wrapper"), "validation_receipt_not_accepted"),
        (lambda value: value.update(schema_version=999), "validation_receipt_not_accepted"),
        (lambda value: value.update(failures=["not_accepted"]), "validation_receipt_not_accepted"),
        (lambda value: value.update(case_id="wrong-case"), "validation_receipt_not_accepted"),
        (lambda value: value.update(untrusted_extra=True), "validation_receipt_not_accepted"),
        (lambda value: value.update(source_render_receipt_sha256="0" * 64), "validation_receipt_not_accepted"),
        (lambda value: value.update(source_render_plan_sha256="0" * 64), "validation_receipt_not_accepted"),
        (lambda value: value.update(known_good_profile_sha256="0" * 64), "validation_receipt_not_accepted"),
        (lambda value: value.update(validator_implementation_sha256="0" * 64), "validation_receipt_not_accepted"),
    ],
)
def test_hash_consistent_wrong_validation_identity_fails_closed(
    tmp_path: Path, mutation, code: str
):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)
    inventory = json.loads(renders.read_text())
    case = inventory["subruns"][0]["cases"][0]
    validation_path = Path(case["validation_receipt_path"])
    validation = json.loads(validation_path.read_text())
    mutation(validation)
    case["validation_receipt_sha256"] = _write(validation_path, validation)
    _write(renders, inventory)

    with pytest.raises(MaterializationError, match=code):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update(status="insufficient_evidence"),
        lambda result: result.update(failures=["height_gate_failed"]),
        lambda result: result.update(warnings=[{"not_warning_type": "missing"}]),
        lambda result: result.update(
            offline_revalidations=[
                {"revalidation_type": "current_carrier_occlusion_mask_revalidation", "passed": False}
            ]
        ),
    ],
)
def test_nonpassing_or_malformed_embedded_validator_result_fails_closed(
    tmp_path: Path, mutation
):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)
    inventory = json.loads(renders.read_text())
    case = inventory["subruns"][0]["cases"][0]
    validation_path = Path(case["validation_receipt_path"])
    validation = json.loads(validation_path.read_text())
    mutation(validation["validator_result"])
    case["validation_receipt_sha256"] = _write(validation_path, validation)
    _write(renders, inventory)

    with pytest.raises(MaterializationError, match="validation_receipt_not_accepted"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "output",
        )


def _rewrite_records(export_path: Path, mutation) -> None:
    export = json.loads(export_path.read_text())
    mutation(export["records"])
    population_path = Path(export["store"]["population_receipt_path"])
    population = json.loads(population_path.read_text())
    population["records_sha256"] = _canonical(export["records"])
    export["store"]["population_receipt_sha256"] = _write(population_path, population)
    _write(export_path, export)


def test_extreme_donor_disagreement_fails_closed(tmp_path: Path):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)

    def split_donors(records):
        for row in records:
            if row["batch_id"] == "INN67-DEV-RB-B02":
                row["median_lab"][0] += 100.0

    _rewrite_records(predicted, split_donors)
    with pytest.raises(MaterializationError, match="donor_pair_not_tight"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "output",
        )


def test_b03_target_outside_independently_derived_threshold_fails_closed(tmp_path: Path):
    plan_root, amendment, materialization_amendment, renders, predicted = _fixture(tmp_path)

    def move_target(records):
        target = next(row for row in records if row["batch_id"] == "INN67-DEV-RB-B03")
        target["median_lab"][0] += 5.0

    _rewrite_records(predicted, move_target)
    with pytest.raises(MaterializationError, match="configuration_target_not_consistent"):
        materialize_reference_bank(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=renders,
            predicted_store_export_path=predicted,
            output_dir=tmp_path / "output",
        )

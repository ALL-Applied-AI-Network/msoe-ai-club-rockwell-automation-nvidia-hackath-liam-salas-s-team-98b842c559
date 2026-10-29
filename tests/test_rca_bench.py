from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation(stage: int, role: str) -> dict[str, object]:
    target = role == "target"
    prefix = f"s{stage}-{role}"
    return {
        "slot_id": f"slot-{prefix}",
        "source_role": role if target else "cross_batch_reference",
        "source_run_id": "run-target" if target else "run-reference",
        "source_case_id": "case-target" if target else f"case-reference-{stage}",
        "source_vial_ordinal": 2 if target else 1,
        "source_block_recipe_id": "block-1",
        "source_block_position": 2 if target else 1,
        "source_recipe_id": "recipe-1",
        "source_previous_recipe_id": "recipe-0" if target else None,
        "source_split_role": "validation",
        "source_split_version": "split-v1",
        "source_time": {"production_sequence_index": 10 if target else 3, "fill_frame_start": 40 if target else 10},
        "source_stage_id": f"stage-{stage}" if target else f"reference-stage-{stage}",
        "source_tank_id": ("t200", "t100")[stage - 1],
        "source_stage_ordinal": stage,
        "source_clip_sha256": _h("target-clip") if target else _h(f"clip-{prefix}"),
        "source_evidence_record_sha256": _h(f"record-{prefix}"),
        "source_colour_regime_id": None,
        "evidence_id": f"evidence-{prefix}",
        "image_id": f"image-{prefix}",
        "rgb_sha256": _h(f"rgb-{prefix}"),
        "mask_sha256": _h(f"mask-{prefix}"),
        "mask_source": "predicted",
        "measurement_id": f"measurement-{prefix}",
        "median_lab": [40.0 + stage, 1.0, 2.0],
        "mad_lab": [0.1, 0.2, 0.3],
        "iqr_lab": [0.2, 0.3, 0.4],
        "pairwise_delta_e76": [{"other_evidence_id": f"evidence-s{stage}-{'reference' if target else 'target'}", "delta_e76": 1.25}],
    }


def model_input() -> dict[str, object]:
    references = [f"evidence-s{stage}-reference" for stage in (1, 2)]
    history_refs = []
    for value in references:
        body = {
            "schema_version": "rca-bench-history-query-receipt-v1",
            "acceptance_status": "accepted",
            "allowed_split_roles": ["validation"],
            "evidence_ids": [value],
            "source_store_sha256": _h("source-store"),
        }
        history_refs.append({"evidence_id": value, "query_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    return {
        "schema_version": "rca-bench-model-input-v1",
        "target_context": {
            "run_id": "run-target",
            "case_id": "case-target",
            "vial_ordinal": 2,
            "block_recipe_id": "block-1",
            "block_position": 2,
            "recipe_id": "recipe-1",
            "previous_recipe_id": "recipe-0",
            "recipe_ml": {"t100": 10.0, "t200": 20.0, "t400": 0.0},
            "dispense_order": ["t200", "t100"],
            "split_role": "validation",
            "split_version": "split-v1",
            "fill_frame_range": [40, 50],
            "transition_frame_ranges": {"entry": [30, 40], "exit": [50, 60]},
            "clip_sha256": _h("target-clip"),
            "decision_id": "decision-1",
            "disposition": "keep",
            "limit_codes": [],
            "evidence_ids": [f"evidence-s{stage}-{role}" for stage in (1, 2) for role in ("target", "reference")],
            "history_refs": history_refs,
        },
        "stages": [
            {
                "stage_id": f"stage-{stage}",
                "tank_id": ("t200", "t100")[stage - 1],
                "stage_ordinal": stage,
                "colour_regime_id": None,
                "observations": [_observation(stage, "target"), _observation(stage, "reference")],
            }
            for stage in (1, 2)
        ],
    }


def accepted_provenance(
    bundle: dict[str, object], *, production: bool = False
) -> dict[str, object]:
    entries = []
    source_receipts = []
    measurement_payloads = []
    for stage in bundle["stages"]:
        for observation in stage["observations"]:
            entry_body = {
                "image_id": observation["image_id"],
                "rgb_path": f"/synthetic/rgb/{observation['image_id']}.png",
                "rgb_sha256": observation["rgb_sha256"],
                "mask_path": f"/synthetic/masks/predicted/{observation['image_id']}.png",
                "mask_sha256": observation["mask_sha256"],
                "model_sha256": _h("synthetic-model"),
                "config_sha256": _h("synthetic-config"),
                "code_commit": "1" * 40,
                "threshold_source": "twin_validation_frozen",
            }
            entries.append({**entry_body, "entry_sha256": hashlib.sha256(json.dumps(entry_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
            source_receipts.append({
                "schema_version": "rca-bench-source-evidence-receipt-v1",
                "acceptance_status": "accepted",
                "evidence_id": observation["evidence_id"],
                "image_id": observation["image_id"],
                "measurement_id": observation["measurement_id"],
                "source_evidence_record_sha256": observation["source_evidence_record_sha256"],
                "source_split_role": observation["source_split_role"],
                "source_split_version": observation["source_split_version"],
            })
            measurement_payload = {
                "image_id": observation["image_id"],
                "measurement_id": observation["measurement_id"],
                "evidence_id": observation["evidence_id"],
                "rgb_sha256": observation["rgb_sha256"],
                "mask_sha256": observation["mask_sha256"],
                "validity": "valid",
                "limit_codes": [],
                "median_lab": observation["median_lab"],
                "mad_lab": observation["mad_lab"],
                "iqr_lab": observation["iqr_lab"],
            }
            measurement_payloads.append(measurement_payload)
    sorted_entries = sorted(entries, key=lambda row: row["image_id"])
    registry_hash_payload = {
        "schema_version": "trusted-predicted-mask-registry-v1",
        "entries": [
            {key: value for key, value in entry.items() if key != "entry_sha256"}
            for entry in sorted_entries
        ],
    }
    registry_sha256 = hashlib.sha256(json.dumps(registry_hash_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    entry_by_image = {entry["image_id"]: entry for entry in sorted_entries}
    measurement_receipts = []
    for measurement_payload in measurement_payloads:
        entry = entry_by_image[measurement_payload["image_id"]]
        measurement_receipts.append({
            "schema_version": "trusted-predicted-measurement-binding-v1",
            "image_id": measurement_payload["image_id"],
            "measurement_id": measurement_payload["measurement_id"],
            "evidence_id": measurement_payload["evidence_id"],
            "rgb_sha256": measurement_payload["rgb_sha256"],
            "mask_sha256": measurement_payload["mask_sha256"],
            "entry_sha256": entry["entry_sha256"],
            "registry_sha256": registry_sha256,
            "validity": "valid",
            "limit_codes": [],
            "measurement_sha256": hashlib.sha256(json.dumps(measurement_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        })
    registry_body = {
        "schema_version": "trusted-predicted-mask-registry-v1",
        "acceptance_status": "accepted",
        "entries": sorted_entries,
        "registry_sha256": registry_sha256,
    }
    queries = []
    for history in bundle["target_context"]["history_refs"]:
        body = {
            "schema_version": "rca-bench-history-query-receipt-v1",
            "acceptance_status": "accepted",
            "allowed_split_roles": ["validation"],
            "evidence_ids": [history["evidence_id"]],
            "source_store_sha256": _h("source-store"),
        }
        queries.append({**body, "query_sha256": history["query_sha256"]})
    receipt = {
        "schema_version": "rca-bench-accepted-provenance-v1",
        "purpose": "production" if production else "synthetic_contract_fixture_only",
        "production_eligible": production,
        "registry_receipt": registry_body,
        "source_evidence_receipts": sorted(source_receipts, key=lambda row: row["evidence_id"]),
        "measurement_receipts": sorted(measurement_receipts, key=lambda row: row["evidence_id"]),
        "query_receipts": sorted(queries, key=lambda row: row["query_sha256"]),
    }
    return {**receipt, "receipt_sha256": hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _deliveries(bundle: dict[str, object]) -> list[dict[str, object]]:
    result = []
    for stage in bundle["stages"]:
        for observation in stage["observations"]:
            result.append({
                "slot_id": observation["slot_id"],
                "expected_image_id": observation["image_id"],
                "expected_rgb_sha256": observation["rgb_sha256"],
                "expected_source_run_id": observation["source_run_id"],
                "expected_source_case_id": observation["source_case_id"],
                "expected_source_vial_ordinal": observation["source_vial_ordinal"],
                "expected_source_time": observation["source_time"],
                "delivered_image_id": observation["image_id"],
                "delivered_rgb_sha256": observation["rgb_sha256"],
                "delivered_source_run_id": observation["source_run_id"],
                "delivered_source_case_id": observation["source_case_id"],
                "delivered_source_vial_ordinal": observation["source_vial_ordinal"],
                "delivered_source_time": observation["source_time"],
            })
    return result


def primary_item(item_id: str = "item-001") -> dict[str, object]:
    bundle = model_input()
    prompt = (
        "Explain the decision using only cited evidence.\nEvidence JSON:\n"
        + json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return {
        "item_id": item_id,
        "source_item_id": "source-001",
        "role": "primary",
        "execution_policy": "model_attempt",
        "paired_primary_item_id": None,
        "run_id": "run-target",
        "case_id": "case-target",
        "vial_ordinal": 2,
        "split_version": "split-v1",
        "development_partition": "selection",
        "evidence_bundle": bundle,
        "evidence_bundle_sha256": _h("unused"),
        "prompt": prompt,
        "prompt_sha256": _h(prompt),
        "image_deliveries": _deliveries(bundle),
        "control_receipt": None,
    }


def plan_fixture() -> dict[str, object]:
    from fluidmix.rca_bench import canonical_sha256

    item = primary_item()
    item["evidence_bundle_sha256"] = canonical_sha256(item["evidence_bundle"])
    provenance = accepted_provenance(item["evidence_bundle"], production=True)
    return {
        "schema_version": "rca-bench-plan-v1",
        "protocol_id": "rca-bench-v1-20261028",
        "benchmark_role": "development",
        "development_partition": "selection",
        "split_version": "split-v1",
        "prompt_template_sha256": _h("prompt-template"),
        "guard_config_sha256": _h("guard"),
        "evidence_schema_sha256": _h("evidence-schema"),
        "model_input_schema_sha256": _h("model-input-schema"),
        "source_artifacts": [
            {"artifact_id": "part-b-001", "sha256": _h("part-b")},
            {"artifact_id": "accepted-provenance-001", "sha256": provenance["receipt_sha256"]},
        ],
        "planned_item_count": 1,
        "model_attempt_planned_count": 1,
        "pre_inference_validation_planned_count": 0,
        "items": [item],
    }


def plan_provenance(plan: dict[str, object]) -> dict[str, object]:
    item = plan["items"][0]
    return {
        item["source_item_id"]: accepted_provenance(
            item["evidence_bundle"], production=True
        )
    }


def test_real_development_model_attempt_rejects_synthetic_fixture_provenance() -> None:
    from fluidmix.rca_bench import ContractError, validate_plan

    plan = plan_fixture()
    item = plan["items"][0]
    synthetic = accepted_provenance(item["evidence_bundle"])
    plan["source_artifacts"][-1]["sha256"] = synthetic["receipt_sha256"]
    provenance = {item["source_item_id"]: synthetic}
    with pytest.raises(ContractError, match="development|synthetic|production"):
        validate_plan(plan, accepted_provenance_by_source_item=provenance)


def locator_fixture(plan: dict[str, object]) -> dict[str, object]:
    from fluidmix.rca_bench import canonical_sha256

    return {
        "schema_version": "rca-bench-image-locators-v1",
        "plan_sha256": canonical_sha256(plan),
        "images": [
            {
                "image_id": row["delivered_image_id"],
                "rgb_path": f"synthetic/{row['delivered_image_id']}.png",
                "rgb_sha256": row["delivered_rgb_sha256"],
            }
            for row in plan["items"][0]["image_deliveries"]
        ],
    }


def oracle_fixture(plan: dict[str, object]) -> dict[str, object]:
    from fluidmix.rca_bench import canonical_sha256

    return {
        "schema_version": "rca-bench-oracle-v1",
        "plan_sha256": canonical_sha256(plan),
        "protocol_sha256": _h("protocol"),
        "items": [oracle_item()],
    }


def freeze_fixture(plan: dict[str, object]) -> dict[str, object]:
    from fluidmix.rca_bench import canonical_sha256

    locators = locator_fixture(plan)
    oracle = oracle_fixture(plan)
    return {
        "schema_version": "rca-bench-freeze-receipt-v1",
        "protocol_sha256": _h("protocol"),
        "builder_source_sha256": _h("builder"),
        "scorer_source_sha256": _h("scorer"),
        "test_hashes": [_h("tests")],
        "evidence_schema_sha256": _h("evidence-schema"),
        "prompt_sha256": _h("prompt-template"),
        "guard_config_sha256": _h("guard"),
        "plan_sha256": canonical_sha256(plan),
        "oracle_sha256": canonical_sha256(oracle),
        "locator_sha256": canonical_sha256(locators),
        "split_ids": ["split-v1"],
        "source_artifacts": plan["source_artifacts"],
        "model_repository": "local/synthetic-model",
        "model_revision": "revision-001",
        "model_weight_sha256": _h("weights"),
        "development_partition_receipt_sha256": _h("partition"),
        "qualification_access_log_sha256": _h("access"),
        "generation_config_sha256": _h("generation"),
        "planned_item_count": plan["planned_item_count"],
        "model_attempt_planned_count": plan["model_attempt_planned_count"],
        "pre_inference_validation_planned_count": plan["pre_inference_validation_planned_count"],
        "required_abstention_primary_count": sum(
            item["role"] == "primary" and item["evidence_bundle"]["target_context"]["disposition"] == "insufficient_evidence"
            for item in plan["items"]
        ),
        "non_required_abstention_primary_count": sum(
            item["role"] == "primary" and item["evidence_bundle"]["target_context"]["disposition"] != "insufficient_evidence"
            for item in plan["items"]
        ),
        "status_count_equation": "completed+inference_error+pre_inference_rejected+pre_inference_error+not_run_after_fatal=planned_item_count",
        "fatal_outcome": "no_rerun_incomplete_fails_adoption",
        "partition_checker_bindings": {name: _h(name) for name in ("split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code")},
        "git_commit": "0" * 40,
    }


def freeze_expected(freeze: dict[str, object]) -> dict[str, object]:
    return {
        field: copy.deepcopy(freeze[field])
        for field in (
            "protocol_sha256", "builder_source_sha256", "scorer_source_sha256",
            "test_hashes", "evidence_schema_sha256", "prompt_sha256",
            "guard_config_sha256", "git_commit",
        )
    }


def test_model_input_boundary_accepts_four_slots_and_rejects_oracle_and_extra_fields() -> None:
    from fluidmix.rca_bench import ContractError, validate_model_input

    value = model_input()
    with pytest.raises(ContractError, match="accepted provenance"):
        validate_model_input(value)
    provenance = accepted_provenance(value)
    assert provenance["production_eligible"] is False
    assert validate_model_input(value, accepted_provenance=provenance) == value

    oracle = copy.deepcopy(value)
    oracle["stages"][0]["observations"][0]["mask_source"] = "oracle"
    with pytest.raises(ContractError, match="predicted"):
        validate_model_input(oracle, accepted_provenance=provenance)

    leaked = copy.deepcopy(value)
    leaked["stages"][0]["observations"][0]["fill_clip_path"] = "/tmp/leak.png"
    with pytest.raises(ContractError, match="fields"):
        validate_model_input(leaked, accepted_provenance=provenance)


def test_normal_evidence_shapes_allow_independent_dispense_order_and_nullable_first_vial_previous() -> None:
    from fluidmix.rca_bench import validate_model_input

    value = model_input()
    value["target_context"]["vial_ordinal"] = 1
    value["target_context"]["block_position"] = 1
    value["target_context"]["previous_recipe_id"] = None
    for stage in value["stages"]:
        target = stage["observations"][0]
        target["source_vial_ordinal"] = 1
        target["source_block_position"] = 1
        target["source_previous_recipe_id"] = None
    assert validate_model_input(value, accepted_provenance=accepted_provenance(value))["target_context"]["dispense_order"] == ["t200", "t100"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("source_stage_id", "stage-for-another-target"), ("source_clip_sha256", _h("another-clip"))),
)
def test_target_observation_binds_stage_and_clip_identity(field: str, replacement: str) -> None:
    from fluidmix.rca_bench import ContractError, validate_model_input

    value = model_input()
    value["stages"][0]["observations"][0][field] = replacement
    with pytest.raises(ContractError, match="target observation.*stage|clip"):
        validate_model_input(value, accepted_provenance=accepted_provenance(value))


@pytest.mark.parametrize("forbidden_role", ["buffer", "unknown"])
def test_cross_batch_history_query_rejects_forbidden_allowed_roles(forbidden_role: str) -> None:
    from fluidmix.rca_bench import ContractError, canonical_sha256, validate_model_input

    value = model_input()
    provenance = accepted_provenance(value)
    query = provenance["query_receipts"][0]
    query["allowed_split_roles"] = [forbidden_role]
    query["query_sha256"] = canonical_sha256({key: query[key] for key in query if key != "query_sha256"})
    provenance["receipt_sha256"] = canonical_sha256({key: provenance[key] for key in provenance if key != "receipt_sha256"})
    with pytest.raises(ContractError, match="buffer or unknown"):
        validate_model_input(value, accepted_provenance=provenance)


def test_cross_batch_history_query_allows_truthful_same_test_history() -> None:
    from fluidmix.rca_bench import canonical_sha256, validate_model_input

    value = model_input()
    value["target_context"]["split_role"] = "test"
    for stage in value["stages"]:
        for observation in stage["observations"]:
            observation["source_split_role"] = "test"
    for history in value["target_context"]["history_refs"]:
        query_body = {
            "schema_version": "rca-bench-history-query-receipt-v1",
            "acceptance_status": "accepted",
            "allowed_split_roles": ["test"],
            "evidence_ids": [history["evidence_id"]],
            "source_store_sha256": _h("source-store"),
        }
        history["query_sha256"] = canonical_sha256(query_body)
    provenance = accepted_provenance(value)
    for query in provenance["query_receipts"]:
        query["allowed_split_roles"] = ["test"]
    provenance["receipt_sha256"] = canonical_sha256(
        {key: provenance[key] for key in provenance if key != "receipt_sha256"}
    )

    assert validate_model_input(value, accepted_provenance=provenance) == value


def test_partition_assignment_is_frozen_first_four_hashes_and_leakage_fails_closed() -> None:
    from fluidmix.rca_bench import (
        ContractError,
        assign_development_partitions,
        validate_partition_leakage,
    )

    runs = [f"run-{index}" for index in range(6)]
    receipt = assign_development_partitions(runs)
    expected = sorted(
        runs,
        key=lambda run_id: _h(f"rca-bench-v1-20261028|grounding-qualification-run|{run_id}"),
    )
    assert receipt["grounding_qualification_run_ids"] == expected[:4]
    assert receipt["selection_run_ids"] == expected[4:]
    assert receipt["sealed_inventory_sha256"] == _h("placeholder") or len(receipt["sealed_inventory_sha256"]) == 64

    audit = {
        "benchmark_role": "development",
        "recipes": [
            {"recipe_id": "r1", "role": "validation", "ratio_cell": "1:1", "carryover_pair_id": "p1", "near_group": "n1", "is_ratio_buffer": False, "render_seed": 2001},
            {"recipe_id": "r2", "role": "validation", "ratio_cell": "2:1", "carryover_pair_id": "p2", "near_group": "n2", "is_ratio_buffer": False, "render_seed": 2002},
        ],
        "seed_ranges": {"validation": [2000, 2999]},
        "manifests": [
            {"recipe_id": "r1", "role": "validation", "path": "v/r1.png", "sha256": _h("r1")},
            {"recipe_id": "r2", "role": "validation", "path": "v/r2.png", "sha256": _h("r2")},
        ],
        "reserved_video5_recipe_ids": [],
        "development_run_ids": runs,
        "test_run_ids": [],
        "selection_run_ids": receipt["selection_run_ids"],
        "qualification_run_ids": receipt["grounding_qualification_run_ids"],
        "checker_bindings": {name: _h(name) for name in ("split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code")},
    }
    assert validate_partition_leakage(audit)["status"] == "pass"

    crossed = copy.deepcopy(audit)
    crossed["recipes"].append({**crossed["recipes"][0], "recipe_id": "r3", "role": "test", "render_seed": 3001})
    crossed["seed_ranges"]["test"] = [3000, 3999]
    with pytest.raises(ContractError, match="ratio-cell"):
        validate_partition_leakage(crossed)


def test_control_builders_preserve_boundaries_and_quarantine_id_shuffle() -> None:
    from fluidmix.rca_bench import (
        ContractError,
        build_shuffled_evidence_ids_control,
        build_shuffled_images_control,
        build_support_removed_control,
        build_unsupported_cause_probe,
        validate_natural_abstention,
        validate_pre_inference_control,
    )

    primary = primary_item()
    rebuilt = copy.deepcopy(primary["evidence_bundle"])
    rebuilt["target_context"]["disposition"] = "insufficient_evidence"
    rebuilt["target_context"]["limit_codes"] = ["support_removed"]
    provenance = accepted_provenance(primary["evidence_bundle"])
    support = build_support_removed_control(
        primary, rebuilt, item_id="control-support", accepted_provenance=provenance
    )
    assert support["role"] == "support_removed"
    assert support["prompt"] != primary["prompt"]

    shuffled_ids = build_shuffled_evidence_ids_control(primary, item_id="control-ids")
    assert shuffled_ids["execution_policy"] == "pre_inference_validation_only"
    assert shuffled_ids["evidence_bundle"] != primary["evidence_bundle"]
    rejection = validate_pre_inference_control(shuffled_ids)
    assert rejection["record_status"] == "pre_inference_rejected"
    assert rejection["rejection_code"] == "evidence_id_binding_mismatch"

    donor = copy.deepcopy(primary)
    donor["item_id"] = "item-donor"
    donor["source_item_id"] = "source-donor"
    for delivery in donor["image_deliveries"]:
        delivery["delivered_image_id"] = "donor-" + delivery["delivered_image_id"]
        delivery["delivered_rgb_sha256"] = _h(delivery["delivered_image_id"])
        delivery["delivered_source_run_id"] = "run-donor"
        delivery["delivered_source_case_id"] = "case-donor"
    shuffled_images = build_shuffled_images_control(
        primary,
        donor,
        changed_slot_ids=[primary["image_deliveries"][0]["slot_id"]],
        delta_e76_by_slot={primary["image_deliveries"][0]["slot_id"]: 2.01},
        item_id="control-images",
    )
    assert shuffled_images["prompt"] == primary["prompt"]
    assert shuffled_images["evidence_bundle"] == primary["evidence_bundle"]
    assert {row["delivered_image_id"] for row in shuffled_images["image_deliveries"]}.isdisjoint(
        {row["delivered_image_id"] for row in primary["image_deliveries"]}
    )

    probe = build_unsupported_cause_probe(primary, item_id="control-cause")
    assert "assert none without cited support" in probe["prompt"]

    natural = copy.deepcopy(primary)
    natural["evidence_bundle"]["target_context"]["disposition"] = "insufficient_evidence"
    assert validate_natural_abstention(natural, accepted_provenance=provenance) is natural
    with pytest.raises(ContractError, match="insufficient"):
        validate_natural_abstention(primary, accepted_provenance=provenance)


def test_selection_controls_are_disposition_run_stratified_and_report_missing_strata() -> None:
    from fluidmix.rca_bench import ContractError, select_deterministic_control_items

    primaries = []
    for index, (run_id, disposition) in enumerate(
        (("run-a", "keep"), ("run-b", "keep"), ("run-a", "discard"), ("run-b", "discard"))
    ):
        item = primary_item(f"item-{index + 10}")
        item["source_item_id"] = f"source-{index + 10}"
        item["run_id"] = run_id
        item["evidence_bundle"]["target_context"]["disposition"] = disposition
        primaries.append(item)
    receipt = select_deterministic_control_items(
        primaries,
        "support_removed",
        3,
        required_dispositions=["keep", "discard", "insufficient_evidence"],
    )
    assert len(receipt["selected_items"]) == 3
    assert {item["evidence_bundle"]["target_context"]["disposition"] for item in receipt["selected_items"]} == {"keep", "discard"}
    assert receipt["missing_strata"] == [
        {"disposition": "insufficient_evidence", "run_id": "run-a"},
        {"disposition": "insufficient_evidence", "run_id": "run-b"},
    ]
    assert len({item["run_id"] for item in receipt["selected_items"]}) == 2
    with pytest.raises(ContractError, match="zero eligible"):
        select_deterministic_control_items([], "support_removed", 1, required_dispositions=["keep"])


def oracle_item() -> dict[str, object]:
    return {
        "item_id": "item-001",
        "source_item_id": "source-001",
        "role": "primary",
        "paired_primary_item_id": None,
        "expected_decision": "keep",
        "required_abstention": False,
        "required_statement_ids": ["statement-1"],
        "supportable_statements": [{
            "statement_id": "statement-1",
            "kind": "observation",
            "normalized_claim": "target matches reference",
            "evidence_ids": ["evidence-s1-target"],
            "image_ids": ["image-s1-target"],
            "numeric_claims": [{"name": "delta_e76", "value": 1.25, "unit": "delta_e76", "evidence_id": "evidence-s1-target"}],
            "required": True,
        }],
        "cause_expectation": None,
        "cause_supported_by_part_b": False,
        "post_inference_label_ref": None,
    }


def response_object() -> dict[str, object]:
    statement = copy.deepcopy(oracle_item()["supportable_statements"][0])
    statement.pop("kind")
    statement.pop("required")
    return {
        "disposition": "keep",
        "abstain": False,
        "asserted_statements": [statement],
        "cause": None,
        "affected_stage_ids": [],
        "numeric_claims": copy.deepcopy(statement["numeric_claims"]),
        "evidence_ids": ["evidence-s1-target"],
        "image_ids": ["image-s1-target"],
    }


def test_strict_parser_and_atom_scorer_reject_miscitation_cause_and_advice() -> None:
    from fluidmix.rca_bench import parse_response, score_parsed_response

    parsed = parse_response(json.dumps(response_object(), sort_keys=True, separators=(",", ":")))
    score = score_parsed_response(parsed, oracle_item(), public_item=primary_item())
    assert score["supported_explanation_pass"] is True
    assert score["supported_assertion_count"] == 1

    miscited = response_object()
    miscited["asserted_statements"][0]["evidence_ids"] = ["evidence-wrong"]
    miscited["evidence_ids"] = ["evidence-wrong"]
    assert score_parsed_response(
        parse_response(json.dumps(miscited)), oracle_item(), public_item=primary_item()
    )["unsupported_assertion_count"] == 1

    cause = response_object()
    cause["cause"] = {"switch_id": "switch-1", "previous_recipe_id": "recipe-0", "affected_stage_ids": ["stage-1"]}
    cause["affected_stage_ids"] = ["stage-1"]
    assert "unsupported_cause" in score_parsed_response(
        parse_response(json.dumps(cause)), oracle_item(), public_item=primary_item()
    )["failures"]

    advice = response_object()
    advice["asserted_statements"][0]["normalized_claim"] = "flush the tank"
    assert "process_control_advice" in score_parsed_response(
        parse_response(json.dumps(advice)), oracle_item(), public_item=primary_item()
    )["failures"]

    extra = response_object()
    extra["extra_prose"] = "sounds convincing"
    with pytest.raises(Exception, match="fields"):
        parse_response(json.dumps(extra))


def test_oracle_required_atoms_and_decision_must_exactly_match_public_evidence() -> None:
    from fluidmix.rca_bench import parse_response, score_parsed_response, validate_oracle

    plan = plan_fixture()
    oracle_row = oracle_item()
    second = copy.deepcopy(oracle_row["supportable_statements"][0])
    second["statement_id"] = "statement-2"
    oracle_row["supportable_statements"].append(second)
    oracle = {
        "schema_version": "rca-bench-oracle-v1",
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "protocol_sha256": _h("protocol"),
        "items": [oracle_row],
    }
    with pytest.raises(Exception, match="all and only|required"):
        validate_oracle(
            oracle, plan, accepted_provenance_by_source_item=plan_provenance(plan)
        )

    oracle_row["required_statement_ids"].append("statement-2")
    oracle_row["expected_decision"] = "discard"
    with pytest.raises(Exception, match="public.*disposition|expected decision"):
        validate_oracle(
            oracle, plan, accepted_provenance_by_source_item=plan_provenance(plan)
        )
    with pytest.raises(Exception, match="scoring oracle decision"):
        score_parsed_response(
            parse_response(json.dumps(response_object())),
            oracle_row,
            public_item=plan["items"][0],
        )


def test_metrics_preserve_denominators_and_gate_fails_closed() -> None:
    from fluidmix.rca_bench import adoption_gate, compute_metrics, paired_transitions

    good = score_parsed_response_for_test(required_abstention=False)
    abstain = score_parsed_response_for_test(required_abstention=True)
    metrics = compute_metrics([
        {**good, "item_id": "a", "parse_success": True},
        {**abstain, "item_id": "b", "parse_success": True},
        {
            "item_id": "c", "parse_success": False, "supported_explanation_pass": False,
            "supported_assertion_count": 0, "unsupported_assertion_count": 0,
            "asserted_assertion_count": 0, "numeric_tp": 0, "numeric_fn_required": 1,
            "numeric_fp": 0, "correct_abstention": False, "unnecessary_abstention": False,
            "required_abstention": True, "cause_supported": False, "cause_correct": False,
            "unsupported_claim": False,
        },
    ])
    assert metrics["supported_explanation_pass_rate"]["numerator"] == 2
    assert metrics["supported_explanation_pass_rate"]["denominator"] == 3
    assert metrics["required_abstention_recall"]["numerator"] == 1
    assert metrics["required_abstention_recall"]["denominator"] == 2
    assert metrics["correct_cause_rate"]["status"] == "not_applicable"
    assert metrics["numeric_fidelity"]["numerator"] <= metrics["numeric_fidelity"]["denominator"]

    assert paired_transitions(
        {"a": True, "b": False}, {"a": True, "b": True}
    ) == {"both_pass": 1, "baseline_only": 0, "candidate_only": 1, "neither": 0}

    baseline = gate_metrics(pass_rate=0.50, unsupported=0.20, recall=0.80)
    candidate = gate_metrics(pass_rate=0.60, unsupported=0.20, recall=0.80)
    assert adoption_gate(baseline, candidate, grounding_classification="image_grounded", run_status="complete")["adopt"] is True
    assert adoption_gate(baseline, candidate, grounding_classification="image_sensitive_not_proven_grounded", run_status="complete")["adopt"] is False
    assert adoption_gate(baseline, candidate, grounding_classification="image_grounded", run_status="fatal_incomplete")["adopt"] is False


def score_parsed_response_for_test(*, required_abstention: bool) -> dict[str, object]:
    return {
        "supported_explanation_pass": True,
        "supported_assertion_count": 1,
        "unsupported_assertion_count": 0,
        "asserted_assertion_count": 1,
        "numeric_tp": 1,
        "numeric_fn_required": 0,
        "numeric_fp": 0,
        "correct_abstention": required_abstention,
        "unnecessary_abstention": False,
        "required_abstention": required_abstention,
        "cause_supported": False,
        "cause_correct": False,
        "unsupported_claim": False,
    }


def gate_metrics(*, pass_rate: float, unsupported: float, recall: float) -> dict[str, object]:
    return {
        "supported_explanation_pass_rate": {"numerator": int(pass_rate * 10), "denominator": 10, "rate": pass_rate, "status": "measured"},
        "unsupported_claim_rate": {"numerator": int(unsupported * 10), "denominator": 10, "rate": unsupported, "status": "measured"},
        "required_abstention_recall": {"numerator": int(recall * 10), "denominator": 10, "rate": recall, "status": "measured"},
        "unnecessary_abstention_rate": {"numerator": 0, "denominator": 1, "rate": 0.0, "status": "measured"},
    }


def test_grounding_normalizes_identifiers_and_requires_twelve_of_twelve_all_stages() -> None:
    from fluidmix.rca_bench import classify_grounding, semantic_signature

    item = primary_item()
    response = response_object()
    signature = semantic_signature(response, item)
    renamed = copy.deepcopy(response)
    renamed["asserted_statements"][0]["image_ids"] = ["replacement-image"]
    renamed["image_ids"] = ["replacement-image"]
    renamed_item = copy.deepcopy(item)
    renamed_item["image_deliveries"][0]["delivered_image_id"] = "replacement-image"
    assert semantic_signature(renamed, renamed_item) == signature

    changed = copy.deepcopy(response)
    changed["asserted_statements"][0]["normalized_claim"] = "replacement conflicts with reference"
    assert semantic_signature(changed, item) != signature

    all_pass = [
        {"pair_id": f"pair-{index}", "valid_control": True, "grounded_pair_pass": True, "stage_ordinals": [1, 2]}
        for index in range(12)
    ]
    assert classify_grounding(all_pass, required_stage_ordinals=[1, 2]) == "image_grounded"
    partial = copy.deepcopy(all_pass)
    partial[0]["grounded_pair_pass"] = False
    assert classify_grounding(partial, required_stage_ordinals=[1, 2]) == "image_sensitive_not_proven_grounded"
    none = copy.deepcopy(all_pass)
    for row in none:
        row["grounded_pair_pass"] = False
    assert classify_grounding(none, required_stage_ordinals=[1, 2]) == "not_image_grounded"
    invalid = copy.deepcopy(all_pass)
    invalid[0]["valid_control"] = False
    assert classify_grounding(invalid, required_stage_ordinals=[1, 2]) == "invalid_control"
    assert classify_grounding([], required_stage_ordinals=[1, 2], qualification_available=False) == "qualification_unavailable"


def test_grounding_rejects_disposition_only_change_without_changed_slot_semantics() -> None:
    from fluidmix.rca_bench import build_shuffled_images_control, grounding_pair_result

    primary = primary_item()
    primary["development_partition"] = "grounding_qualification"
    donor = copy.deepcopy(primary)
    donor["item_id"] = "item-donor"
    donor["source_item_id"] = "source-donor"
    donor["run_id"] = "run-donor"
    changed_slot = primary["image_deliveries"][0]["slot_id"]
    for delivery in donor["image_deliveries"]:
        delivery["delivered_image_id"] = "donor-" + delivery["delivered_image_id"]
        delivery["delivered_rgb_sha256"] = _h(delivery["delivered_image_id"])
        delivery["delivered_source_run_id"] = "run-donor"
        delivery["delivered_source_case_id"] = "case-donor"
    shuffled = build_shuffled_images_control(
        primary,
        donor,
        changed_slot_ids=[changed_slot],
        delta_e76_by_slot={changed_slot: 2.1},
        item_id="control-images-grounding",
    )
    primary_response = response_object()
    shuffled_response = copy.deepcopy(primary_response)
    shuffled_response["disposition"] = "discard"
    shuffled_oracle = oracle_item()
    shuffled_oracle["expected_decision"] = "discard"
    result = grounding_pair_result(
        json.dumps(primary_response),
        json.dumps(shuffled_response),
        primary,
        shuffled,
        oracle_item(),
        shuffled_oracle,
    )
    assert result["pixel_conditioned_change"] is False
    assert result["grounded_pair_pass"] is False


def test_journal_is_hash_chained_and_fatal_finalization_never_retries(tmp_path: Path) -> None:
    from fluidmix.rca_bench import RunLedger, finalize_fatal_run, validate_raw_count_equations

    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path)
    ledger.append("preflight_started", payload={})
    ledger.append("preflight_completed", payload={"status": "ok"})
    ledger.append("validation_started", item_id="control-ids", execution_index=0, payload={})
    rejection_record = {
        "record_index": 0, "item_id": "control-ids",
        "execution_policy": "pre_inference_validation_only",
        "record_status": "pre_inference_rejected", "attempt_index": None,
        "validation_index": 0, "model_repository": None, "model_revision": None,
        "model_weight_sha256": None, "prompt_sha256": None,
        "ordered_delivered_image_ids": None, "ordered_delivered_rgb_sha256": None,
        "generation_config_sha256": None, "response": None,
        "response_utf8_sha256": None, "elapsed_seconds": None,
        "peak_cuda_memory_allocated_bytes": None,
        "peak_cuda_memory_reserved_bytes": None, "processor_telemetry": None,
        "telemetry_unavailable_reason": None, "error_type": None, "error": None,
        "validation_stage": "part_b_intake", "rejection_code": "evidence_id_binding",
        "validator_receipt_sha256": _h("validator"), "fatal_event_sha256": None,
        "not_run_reason": None,
    }
    ledger.append(
        "validation_terminal", item_id="control-ids", execution_index=0,
        payload=rejection_record,
    )
    ledger.append("model_attempt_started", item_id="item-a", execution_index=0, payload={})
    fatal = ledger.append("fatal", payload={"fatal_code": "process_crash"})

    items = [
        {"item_id": "control-ids", "execution_policy": "pre_inference_validation_only"},
        {"item_id": "item-a", "execution_policy": "model_attempt"},
        {"item_id": "item-b", "execution_policy": "model_attempt"},
    ]
    raw = finalize_fatal_run(
        items, ledger.events, ledger.payloads,
        fatal_event_sha256=fatal["event_sha256"],
        raw_bindings={
            "plan_sha256": _h("plan"),
            "freeze_receipt_sha256": _h("freeze"),
            "model_repository": "local/synthetic-model",
            "model_revision": "revision-001",
            "model_weight_sha256": _h("weights"),
            "generation_config_sha256": _h("generation"),
        },
    )
    assert [row["record_status"] for row in raw["records"]] == [
        "pre_inference_rejected", "inference_error", "not_run_after_fatal"
    ]
    assert raw["records"][1]["error_type"] == "process_crash_unmatched_model_attempt_start"
    assert raw["records"][2]["not_run_reason"] == "fatal_run_terminated"
    validate_raw_count_equations(raw)

    with pytest.raises(Exception, match="resume|exists"):
        RunLedger(path)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["previous_event_sha256"] is None
    assert rows[-1]["previous_event_sha256"] == rows[-2]["event_sha256"]


def test_fatal_finalizer_emits_exact_validate_raw_compatible_schema(tmp_path: Path) -> None:
    from fluidmix.rca_bench import RunLedger, canonical_sha256, finalize_fatal_run, validate_raw

    plan = plan_fixture()
    freeze = freeze_fixture(plan)
    item = plan["items"][0]
    ledger = RunLedger(tmp_path / "fatal-exact-ledger.jsonl")
    ledger.append("preflight_started", payload={})
    ledger.append("preflight_completed", payload={"status": "ok"})
    ledger.append("model_attempt_started", item_id=item["item_id"], execution_index=0, payload={})
    fatal = ledger.append("fatal", payload={"fatal_code": "process_crash"})
    raw = finalize_fatal_run(
        plan["items"], ledger.events, ledger.payloads,
        fatal_event_sha256=fatal["event_sha256"],
        raw_bindings={
            "plan_sha256": canonical_sha256(plan),
            "freeze_receipt_sha256": canonical_sha256(freeze),
            "model_repository": freeze["model_repository"],
            "model_revision": freeze["model_revision"],
            "model_weight_sha256": freeze["model_weight_sha256"],
            "generation_config_sha256": freeze["generation_config_sha256"],
        },
    )
    assert set(raw) == {
        "schema_version", "plan_sha256", "freeze_receipt_sha256", "journal_terminal_sha256",
        "model_repository", "model_revision", "model_weight_sha256", "generation_config_sha256",
        "run_status", "planned_item_count", "model_attempt_planned_count",
        "pre_inference_validation_planned_count", "completed_count", "inference_error_count",
        "pre_inference_rejected_count", "pre_inference_error_count", "not_run_after_fatal_count",
        "model_not_run_after_fatal_count", "validation_not_run_after_fatal_count", "records",
    }
    assert validate_raw(
        raw, plan, freeze,
        accepted_provenance_by_source_item=plan_provenance(plan),
        ledger_events=ledger.events,
        expected_freeze_bindings=freeze_expected(freeze),
    )["run_status"] == "fatal_incomplete"

    preflight_ledger = RunLedger(tmp_path / "preflight-fatal-ledger.jsonl")
    preflight_ledger.append("preflight_started", payload={})
    preflight_fatal = preflight_ledger.append(
        "fatal", payload={"fatal_code": "process_crash_unmatched_preflight_start"}
    )
    preflight_raw = finalize_fatal_run(
        plan["items"], preflight_ledger.events, preflight_ledger.payloads,
        fatal_event_sha256=preflight_fatal["event_sha256"],
        raw_bindings={
            "plan_sha256": canonical_sha256(plan),
            "freeze_receipt_sha256": canonical_sha256(freeze),
            "model_repository": freeze["model_repository"],
            "model_revision": freeze["model_revision"],
            "model_weight_sha256": freeze["model_weight_sha256"],
            "generation_config_sha256": freeze["generation_config_sha256"],
        },
    )
    assert preflight_raw["records"][0]["record_status"] == "not_run_after_fatal"
    validate_raw(
        preflight_raw, plan, freeze,
        accepted_provenance_by_source_item=plan_provenance(plan),
        ledger_events=preflight_ledger.events,
        expected_freeze_bindings=freeze_expected(freeze),
    )


def test_frozen_synthetic_artifacts_are_strict_hash_bound_and_no_clobber(tmp_path: Path) -> None:
    from fluidmix.rca_bench import (
        canonical_sha256,
        validate_image_locators,
        validate_oracle,
        validate_plan,
        write_canonical_json_exclusive,
    )

    plan = plan_fixture()
    item = plan["items"][0]
    locators = {
        "schema_version": "rca-bench-image-locators-v1",
        "plan_sha256": canonical_sha256(plan),
        "images": [
            {"image_id": row["delivered_image_id"], "rgb_path": f"synthetic/{row['delivered_image_id']}.png", "rgb_sha256": row["delivered_rgb_sha256"]}
            for row in item["image_deliveries"]
        ],
    }
    provenance_by_source = plan_provenance(plan)
    checked_plan = validate_plan(
        plan, locators=locators,
        accepted_provenance_by_source_item=provenance_by_source,
    )
    assert checked_plan["planned_item_count"] == 1
    assert validate_image_locators(locators)["plan_sha256"] == canonical_sha256(plan)

    oracle_row = oracle_item()
    oracle = {
        "schema_version": "rca-bench-oracle-v1",
        "plan_sha256": canonical_sha256(plan),
        "protocol_sha256": _h("protocol"),
        "items": [oracle_row],
    }
    assert validate_oracle(
        oracle, plan, accepted_provenance_by_source_item=provenance_by_source
    )["items"][0]["expected_decision"] == "keep"

    leaked = copy.deepcopy(plan)
    leaked["items"][0]["evidence_bundle"]["sim_metadata"] = {"label": "hidden"}
    leaked["items"][0]["evidence_bundle_sha256"] = canonical_sha256(leaked["items"][0]["evidence_bundle"])
    with pytest.raises(Exception, match="fields|simulator"):
        validate_plan(
            leaked, locators=locators,
            accepted_provenance_by_source_item=provenance_by_source,
        )

    output = tmp_path / "fixture.json"
    write_canonical_json_exclusive(output, plan)
    with pytest.raises(Exception, match="overwrite"):
        write_canonical_json_exclusive(output, plan)


def test_freeze_and_raw_records_bind_every_image_and_count(tmp_path: Path) -> None:
    from fluidmix.rca_bench import RunLedger, canonical_sha256, validate_freeze_receipt, validate_raw

    plan = plan_fixture()
    item = plan["items"][0]
    freeze = freeze_fixture(plan)
    provenance_by_source = plan_provenance(plan)
    validate_freeze_receipt(
        freeze, plan, locators=locator_fixture(plan), oracle=oracle_fixture(plan),
        accepted_provenance_by_source_item=provenance_by_source,
        expected_bindings=freeze_expected(freeze),
    )
    response = json.dumps(response_object(), sort_keys=True, separators=(",", ":"))
    record = {
        "record_index": 0,
        "item_id": item["item_id"],
        "execution_policy": "model_attempt",
        "record_status": "completed",
        "attempt_index": 0,
        "validation_index": None,
        "model_repository": freeze["model_repository"],
        "model_revision": freeze["model_revision"],
        "model_weight_sha256": freeze["model_weight_sha256"],
        "prompt_sha256": item["prompt_sha256"],
        "ordered_delivered_image_ids": [row["delivered_image_id"] for row in item["image_deliveries"]],
        "ordered_delivered_rgb_sha256": [row["delivered_rgb_sha256"] for row in item["image_deliveries"]],
        "generation_config_sha256": freeze["generation_config_sha256"],
        "response": response,
        "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "elapsed_seconds": 0.1,
        "peak_cuda_memory_allocated_bytes": 100,
        "peak_cuda_memory_reserved_bytes": 200,
        "processor_telemetry": {"ordered_image_count": 4},
        "telemetry_unavailable_reason": None,
        "error_type": None,
        "error": None,
        "validation_stage": None,
        "rejection_code": None,
        "validator_receipt_sha256": None,
        "fatal_event_sha256": None,
        "not_run_reason": None,
    }
    ledger = RunLedger(tmp_path / "normal-ledger.jsonl")
    ledger.append("preflight_started", payload={})
    ledger.append("preflight_completed", payload={"status": "ok"})
    ledger.append("model_attempt_started", item_id=item["item_id"], execution_index=0, payload={})
    terminal = ledger.append(
        "model_attempt_terminal", item_id=item["item_id"], execution_index=0,
        payload=record,
    )
    raw = {
        "schema_version": "rca-bench-raw-v1",
        "plan_sha256": canonical_sha256(plan),
        "freeze_receipt_sha256": canonical_sha256(freeze),
        "journal_terminal_sha256": terminal["event_sha256"],
        "model_repository": freeze["model_repository"],
        "model_revision": freeze["model_revision"],
        "model_weight_sha256": freeze["model_weight_sha256"],
        "generation_config_sha256": freeze["generation_config_sha256"],
        "run_status": "complete",
        "planned_item_count": 1,
        "model_attempt_planned_count": 1,
        "pre_inference_validation_planned_count": 0,
        "completed_count": 1,
        "inference_error_count": 0,
        "pre_inference_rejected_count": 0,
        "pre_inference_error_count": 0,
        "not_run_after_fatal_count": 0,
        "model_not_run_after_fatal_count": 0,
        "validation_not_run_after_fatal_count": 0,
        "records": [record],
    }
    assert validate_raw(
        raw, plan, freeze,
        accepted_provenance_by_source_item=provenance_by_source,
        ledger_events=ledger.events,
        expected_freeze_bindings=freeze_expected(freeze),
    )["completed_count"] == 1

    broken = copy.deepcopy(raw)
    broken["records"][0]["ordered_delivered_image_ids"] = broken["records"][0]["ordered_delivered_image_ids"][:-1]
    with pytest.raises(Exception, match="ordered multi-image"):
        validate_raw(
            broken, plan, freeze,
            accepted_provenance_by_source_item=provenance_by_source,
            ledger_events=ledger.events,
            expected_freeze_bindings=freeze_expected(freeze),
        )

    lying_counts = copy.deepcopy(raw)
    lying_counts["completed_count"] = 0
    lying_counts["inference_error_count"] = 1
    with pytest.raises(Exception, match="record statuses|recomputed"):
        validate_raw(
            lying_counts, plan, freeze,
            accepted_provenance_by_source_item=provenance_by_source,
            ledger_events=ledger.events,
            expected_freeze_bindings=freeze_expected(freeze),
        )

    falsely_incomplete = copy.deepcopy(raw)
    falsely_incomplete["run_status"] = "fatal_incomplete"
    with pytest.raises(Exception, match="fatal_incomplete.*fatal|fatal.*fatal_incomplete"):
        validate_raw(
            falsely_incomplete, plan, freeze,
            accepted_provenance_by_source_item=provenance_by_source,
            ledger_events=ledger.events,
            expected_freeze_bindings=freeze_expected(freeze),
        )

    fatal = ledger.append("fatal", payload={"fatal_code": "late_process_crash"})
    falsely_complete = copy.deepcopy(raw)
    falsely_complete["journal_terminal_sha256"] = fatal["event_sha256"]
    with pytest.raises(Exception, match="complete.*fatal|fatal.*complete"):
        validate_raw(
            falsely_complete, plan, freeze,
            accepted_provenance_by_source_item=provenance_by_source,
            ledger_events=ledger.events,
            expected_freeze_bindings=freeze_expected(freeze),
        )


def test_freeze_checks_explicit_hashes_and_both_test_abstention_denominators() -> None:
    from fluidmix.rca_bench import validate_freeze_receipt

    plan = plan_fixture()
    freeze = freeze_fixture(plan)
    expected = freeze_expected(freeze)
    expected["builder_source_sha256"] = _h("wrong-builder")
    with pytest.raises(Exception, match="explicit expected"):
        validate_freeze_receipt(
            freeze, plan,
            locators=locator_fixture(plan), oracle=oracle_fixture(plan),
            accepted_provenance_by_source_item=plan_provenance(plan),
            expected_bindings=expected,
        )

    test_plan = copy.deepcopy(plan)
    test_plan["benchmark_role"] = "test"
    test_plan["development_partition"] = None
    test_plan["items"][0]["development_partition"] = None
    test_freeze = freeze_fixture(test_plan)
    with pytest.raises(Exception, match="both abstention denominator"):
        validate_freeze_receipt(
            test_freeze, test_plan,
            locators=locator_fixture(test_plan), oracle=oracle_fixture(test_plan),
            accepted_provenance_by_source_item=plan_provenance(test_plan),
            expected_bindings=freeze_expected(test_freeze),
        )


@pytest.mark.parametrize("field", ["locator_sha256", "oracle_sha256"])
def test_freeze_binds_locator_and_oracle_hashes_to_actual_artifacts(field: str) -> None:
    from fluidmix.rca_bench import ContractError, validate_freeze_receipt

    plan = plan_fixture()
    freeze = freeze_fixture(plan)
    freeze[field] = _h(f"wrong-{field}")
    with pytest.raises(ContractError, match="locator.*hash|oracle.*hash|artifact.*hash"):
        validate_freeze_receipt(
            freeze,
            plan,
            locators=locator_fixture(plan),
            oracle=oracle_fixture(plan),
            accepted_provenance_by_source_item=plan_provenance(plan),
            expected_bindings=freeze_expected(freeze),
        )

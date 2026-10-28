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
        "source_previous_recipe_id": "recipe-0",
        "source_split_role": "validation",
        "source_split_version": "split-v1",
        "source_time": {"production_sequence_index": 10 if target else 3, "fill_frame_start": 40 if target else 10},
        "source_stage_id": f"stage-{stage}" if target else f"reference-stage-{stage}",
        "source_tank_id": f"tank-{stage}",
        "source_stage_ordinal": stage,
        "source_clip_sha256": _h(f"clip-{prefix}"),
        "source_evidence_record_sha256": _h(f"record-{prefix}"),
        "source_colour_regime_id": "regime-1",
        "evidence_id": f"evidence-{prefix}",
        "image_id": f"image-{prefix}",
        "rgb_sha256": _h(f"rgb-{prefix}"),
        "mask_sha256": _h(f"mask-{prefix}"),
        "mask_source": "predicted",
        "measurement_id": f"measurement-{prefix}",
        "median_lab": [40.0 + stage, 1.0, 2.0],
        "mad_lab": [0.1, 0.2, 0.3],
        "iqr_lab": [0.2, 0.3, 0.4],
        "pairwise_delta_e76": [1.25],
    }


def model_input() -> dict[str, object]:
    references = [f"evidence-s{stage}-reference" for stage in (1, 2)]
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
            "recipe_ml": {"tank-1": 10.0, "tank-2": 20.0},
            "dispense_order": ["tank-1", "tank-2"],
            "split_role": "validation",
            "split_version": "split-v1",
            "fill_frame_range": [40, 50],
            "transition_frame_ranges": [[30, 39]],
            "clip_sha256": _h("target-clip"),
            "decision_id": "decision-1",
            "disposition": "keep",
            "limit_codes": [],
            "evidence_ids": [f"evidence-s{stage}-{role}" for stage in (1, 2) for role in ("target", "reference")],
            "history_refs": [{"evidence_id": value, "query_sha256": _h(f"query-{value}")} for value in references],
        },
        "stages": [
            {
                "stage_id": f"stage-{stage}",
                "tank_id": f"tank-{stage}",
                "stage_ordinal": stage,
                "colour_regime_id": "regime-1",
                "observations": [_observation(stage, "target"), _observation(stage, "reference")],
            }
            for stage in (1, 2)
        ],
    }


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
        "source_artifacts": [{"artifact_id": "part-b-001", "sha256": _h("part-b")}],
        "planned_item_count": 1,
        "model_attempt_planned_count": 1,
        "pre_inference_validation_planned_count": 0,
        "items": [item],
    }


def test_model_input_boundary_accepts_four_slots_and_rejects_oracle_and_extra_fields() -> None:
    from fluidmix.rca_bench import ContractError, validate_model_input

    value = model_input()
    assert validate_model_input(value) == value

    oracle = copy.deepcopy(value)
    oracle["stages"][0]["observations"][0]["mask_source"] = "oracle"
    with pytest.raises(ContractError, match="predicted"):
        validate_model_input(oracle)

    leaked = copy.deepcopy(value)
    leaked["stages"][0]["observations"][0]["fill_clip_path"] = "/tmp/leak.png"
    with pytest.raises(ContractError, match="fields"):
        validate_model_input(leaked)


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
    support = build_support_removed_control(primary, rebuilt, item_id="control-support")
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
    assert validate_natural_abstention(natural) is natural
    with pytest.raises(ContractError, match="insufficient"):
        validate_natural_abstention(primary)


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
    score = score_parsed_response(parsed, oracle_item())
    assert score["supported_explanation_pass"] is True
    assert score["supported_assertion_count"] == 1

    miscited = response_object()
    miscited["asserted_statements"][0]["evidence_ids"] = ["evidence-wrong"]
    miscited["evidence_ids"] = ["evidence-wrong"]
    assert score_parsed_response(parse_response(json.dumps(miscited)), oracle_item())["unsupported_assertion_count"] == 1

    cause = response_object()
    cause["cause"] = {"switch_id": "switch-1", "previous_recipe_id": "recipe-0", "affected_stage_ids": ["stage-1"]}
    cause["affected_stage_ids"] = ["stage-1"]
    assert "unsupported_cause" in score_parsed_response(parse_response(json.dumps(cause)), oracle_item())["failures"]

    advice = response_object()
    advice["asserted_statements"][0]["normalized_claim"] = "flush the tank"
    assert "process_control_advice" in score_parsed_response(parse_response(json.dumps(advice)), oracle_item())["failures"]

    extra = response_object()
    extra["extra_prose"] = "sounds convincing"
    with pytest.raises(Exception, match="fields"):
        parse_response(json.dumps(extra))


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


def test_journal_is_hash_chained_and_fatal_finalization_never_retries(tmp_path: Path) -> None:
    from fluidmix.rca_bench import RunLedger, finalize_fatal_run, validate_raw_count_equations

    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path)
    ledger.append("preflight_started", payload={})
    ledger.append("preflight_completed", payload={"status": "ok"})
    ledger.append("validation_started", item_id="control-ids", execution_index=0, payload={})
    ledger.append("validation_terminal", item_id="control-ids", execution_index=0, payload={"record_status": "pre_inference_rejected", "rejection_code": "evidence_id_binding"})
    ledger.append("model_attempt_started", item_id="item-a", execution_index=0, payload={})
    fatal = ledger.append("fatal", payload={"fatal_code": "process_crash"})

    items = [
        {"item_id": "control-ids", "execution_policy": "pre_inference_validation_only"},
        {"item_id": "item-a", "execution_policy": "model_attempt"},
        {"item_id": "item-b", "execution_policy": "model_attempt"},
    ]
    raw = finalize_fatal_run(items, ledger.events, ledger.payloads, fatal_event_sha256=fatal["event_sha256"])
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
    checked_plan = validate_plan(plan, locators=locators)
    assert checked_plan["planned_item_count"] == 1
    assert validate_image_locators(locators)["plan_sha256"] == canonical_sha256(plan)

    oracle_row = oracle_item()
    oracle = {
        "schema_version": "rca-bench-oracle-v1",
        "plan_sha256": canonical_sha256(plan),
        "protocol_sha256": _h("protocol"),
        "items": [oracle_row],
    }
    assert validate_oracle(oracle, plan)["items"][0]["expected_decision"] == "keep"

    leaked = copy.deepcopy(plan)
    leaked["items"][0]["evidence_bundle"]["sim_metadata"] = {"label": "hidden"}
    leaked["items"][0]["evidence_bundle_sha256"] = canonical_sha256(leaked["items"][0]["evidence_bundle"])
    with pytest.raises(Exception, match="fields|simulator"):
        validate_plan(leaked, locators=locators)

    output = tmp_path / "fixture.json"
    write_canonical_json_exclusive(output, plan)
    with pytest.raises(Exception, match="overwrite"):
        write_canonical_json_exclusive(output, plan)


def test_freeze_and_raw_records_bind_every_image_and_count() -> None:
    from fluidmix.rca_bench import canonical_sha256, validate_freeze_receipt, validate_raw

    plan = plan_fixture()
    item = plan["items"][0]
    freeze = {
        "schema_version": "rca-bench-freeze-receipt-v1",
        "protocol_sha256": _h("protocol"),
        "builder_source_sha256": _h("builder"),
        "scorer_source_sha256": _h("scorer"),
        "test_hashes": [_h("tests")],
        "evidence_schema_sha256": _h("evidence-schema"),
        "prompt_sha256": _h("prompt-template"),
        "guard_config_sha256": _h("guard"),
        "plan_sha256": canonical_sha256(plan),
        "oracle_sha256": _h("oracle-artifact"),
        "locator_sha256": _h("locator-artifact"),
        "split_ids": ["split-v1"],
        "source_artifacts": plan["source_artifacts"],
        "model_repository": "local/synthetic-model",
        "model_revision": "revision-001",
        "model_weight_sha256": _h("weights"),
        "development_partition_receipt_sha256": _h("partition"),
        "qualification_access_log_sha256": _h("access"),
        "generation_config_sha256": _h("generation"),
        "planned_item_count": 1,
        "model_attempt_planned_count": 1,
        "pre_inference_validation_planned_count": 0,
        "status_count_equation": "completed+inference_error+pre_inference_rejected+pre_inference_error+not_run_after_fatal=planned_item_count",
        "fatal_outcome": "no_rerun_incomplete_fails_adoption",
        "partition_checker_bindings": {name: _h(name) for name in ("split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code")},
        "git_commit": "0" * 40,
    }
    validate_freeze_receipt(freeze, plan)
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
    raw = {
        "schema_version": "rca-bench-raw-v1",
        "plan_sha256": canonical_sha256(plan),
        "freeze_receipt_sha256": canonical_sha256(freeze),
        "journal_terminal_sha256": _h("terminal"),
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
    assert validate_raw(raw, plan, freeze)["completed_count"] == 1
    broken = copy.deepcopy(raw)
    broken["records"][0]["ordered_delivered_image_ids"] = broken["records"][0]["ordered_delivered_image_ids"][:-1]
    with pytest.raises(Exception, match="ordered multi-image"):
        validate_raw(broken, plan, freeze)

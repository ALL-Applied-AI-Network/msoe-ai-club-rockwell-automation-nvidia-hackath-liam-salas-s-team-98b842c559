from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.evaluation.rca_bench.test_rca_bench import accepted_provenance, model_input, primary_item  # noqa: E402


def _rehash_provenance_records(
    records: list[dict[str, object]], provenance: dict[str, object]
) -> None:
    from fluidmix.storage.sqlite.evidence_store import canonical_source_record_sha256

    entries = {
        row["image_id"]: row
        for row in provenance["registry_receipt"]["entries"]
    }
    receipts = {
        row["evidence_id"]: row for row in provenance["source_evidence_receipts"]
    }
    artifact_root = Path("/synthetic")
    for record in records:
        entry = entries[record["image_id"]]
        locator = {
            "rgb_path": Path(entry["rgb_path"]).relative_to(artifact_root).as_posix(),
            "rgb_sha256": entry["rgb_sha256"],
            "mask_path": Path(entry["mask_path"]).relative_to(artifact_root).as_posix(),
            "mask_sha256": entry["mask_sha256"],
        }
        receipts[record["evidence_id"]]["source_evidence_record_sha256"] = (
            canonical_source_record_sha256(record, locator)
        )
    unsigned = {key: value for key, value in provenance.items() if key != "receipt_sha256"}
    provenance["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _records(bundle: dict[str, object]) -> list[dict[str, object]]:
    context = bundle["target_context"]
    result = []
    for stage in bundle["stages"]:
        target, reference = stage["observations"]
        for observation in (target, reference):
            result.append({
                "run_id": observation["source_run_id"],
                "case_id": observation["source_case_id"],
                "vial_ordinal": observation["source_vial_ordinal"],
                "block_recipe_id": observation["source_block_recipe_id"],
                "block_position": observation["source_block_position"],
                "recipe_id": observation["source_recipe_id"],
                "previous_recipe_id": observation["source_previous_recipe_id"],
                "recipe_ml": copy.deepcopy(context["recipe_ml"]),
                "dispense_order": copy.deepcopy(context["dispense_order"]),
                "split_role": observation["source_split_role"],
                "split_version": observation["source_split_version"],
                "stage_id": observation["source_stage_id"],
                "tank_id": observation["source_tank_id"],
                "stage_ordinal": observation["source_stage_ordinal"],
                "fill_clip_path": f"clips/{observation['source_case_id']}.mp4",
                "fill_frame_range": copy.deepcopy(context["fill_frame_range"]),
                "transition_frame_ranges": copy.deepcopy(context["transition_frame_ranges"]),
                "clip_sha256": observation["source_clip_sha256"],
                "image_id": observation["image_id"],
                "rgb_sha256": observation["rgb_sha256"],
                "mask_sha256": observation["mask_sha256"],
                "mask_source": observation["mask_source"],
                "measurement_id": observation["measurement_id"],
                "median_lab": copy.deepcopy(observation["median_lab"]),
                "mad_lab": copy.deepcopy(observation["mad_lab"]),
                "iqr_lab": copy.deepcopy(observation["iqr_lab"]),
                "pairwise_delta_e76": copy.deepcopy(observation["pairwise_delta_e76"]),
                "colour_regime_id": observation["source_colour_regime_id"],
                "evidence_id": observation["evidence_id"],
                "decision_id": context["decision_id"],
                "disposition": context["disposition"],
                "limit_codes": copy.deepcopy(context["limit_codes"]),
                "history_refs": copy.deepcopy(context["history_refs"]),
                "explanation_id": None,
                "model_revision": None,
                "prompt_sha256": None,
                "image_ids_sent": [],
                "guard_result": None,
            })
    return result


def _spec(bundle: dict[str, object], source: str = "source-001") -> dict[str, object]:
    targets = [stage["observations"][0] for stage in bundle["stages"]]
    references = [stage["observations"][1] for stage in bundle["stages"]]
    return {
        "source_item_id": source,
        "target_evidence_ids": [row["evidence_id"] for row in targets],
        "reference_evidence_ids": [row["evidence_id"] for row in references],
        "reference_roles": [row["source_role"] for row in references],
    }


def _partition_audit(
    *, benchmark_role: str, development_run_ids: list[str], test_run_ids: list[str],
    selection_run_ids: list[str], qualification_run_ids: list[str], recipe_role: str,
    recipe_id: str = "recipe-1",
) -> dict[str, object]:
    seed = 2001 if recipe_role == "validation" else 3001
    return {
        "benchmark_role": benchmark_role,
        "recipes": [{
            "recipe_id": recipe_id, "role": recipe_role,
            "ratio_cell": f"ratio-{recipe_role}",
            "carryover_pair_id": f"pair-{recipe_role}",
            "near_group": f"near-{recipe_role}", "is_ratio_buffer": False,
            "render_seed": seed,
        }],
        "seed_ranges": {recipe_role: [2000, 2999] if recipe_role == "validation" else [3000, 3999]},
        "manifests": [{
            "recipe_id": recipe_id, "role": recipe_role,
            "path": f"{recipe_role}/{recipe_id}.json",
            "sha256": hashlib.sha256(f"{recipe_role}-{recipe_id}".encode()).hexdigest(),
        }],
        "reserved_video5_recipe_ids": [],
        "development_run_ids": development_run_ids,
        "test_run_ids": test_run_ids,
        "selection_run_ids": selection_run_ids,
        "qualification_run_ids": qualification_run_ids,
        "checker_bindings": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in ("split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code")
        },
    }


def _wp1_source_provenance(items: list[dict], proofs: dict[str, dict]) -> dict[str, object]:
    from evaluation.rca_bench.rca_bench import canonical_sha256

    body = {
        "schema_version": "rca-bench-wp1-source-provenance-v1",
        "acceptance_status": "accepted",
        "aggregate_acceptance_sha256": hashlib.sha256(b"wp1-aggregate").hexdigest(),
        "source_receipt_sha256_by_source_item": {
            source_id: proof["receipt_sha256"] for source_id, proof in sorted(proofs.items())
        },
        "test_run_ids": sorted({item["run_id"] for item in items}),
        "test_recipe_ids": sorted({
            item["evidence_bundle"]["target_context"]["recipe_id"] for item in items
        }),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _qualification_primary(run: int, vial: int) -> tuple[dict, dict]:
    from evaluation.rca_bench.rca_bench import canonical_bytes, canonical_sha256

    item = copy.deepcopy(primary_item(f"item-{run}-{vial}"))
    item["source_item_id"] = f"source-{run}-{vial}"
    item["run_id"] = f"run-{run}"
    item["case_id"] = f"case-{run}-{vial}"
    item["vial_ordinal"] = vial
    item["development_partition"] = "grounding_qualification"
    bundle = item["evidence_bundle"]
    context = bundle["target_context"]
    context["run_id"] = item["run_id"]
    context["case_id"] = item["case_id"]
    context["vial_ordinal"] = vial
    all_ids = []
    history = []
    for stage in bundle["stages"]:
        for observation in stage["observations"]:
            suffix = f"{run}-{vial}-{stage['stage_ordinal']}-{observation['source_role']}"
            if observation["source_role"] == "target":
                observation["source_run_id"] = item["run_id"]
                observation["source_case_id"] = item["case_id"]
                observation["source_vial_ordinal"] = vial
                observation["median_lab"][0] = float(run * 10 + vial)
            else:
                observation["source_run_id"] = f"reference-{run}-{vial}"
                observation["source_case_id"] = f"reference-case-{run}-{vial}-{stage['stage_ordinal']}"
                observation["median_lab"][0] = float(run * 10 + vial + 1)
            observation["evidence_id"] = f"evidence-{suffix}"
            observation["image_id"] = f"image-{suffix}"
            observation["measurement_id"] = f"measurement-{suffix}"
            observation["rgb_sha256"] = hashlib.sha256(f"rgb-{suffix}".encode()).hexdigest()
            observation["mask_sha256"] = hashlib.sha256(f"mask-{suffix}".encode()).hexdigest()
            observation["source_evidence_record_sha256"] = hashlib.sha256(f"record-{suffix}".encode()).hexdigest()
            observation["pairwise_delta_e76"] = []
            all_ids.append(observation["evidence_id"])
            if observation["source_role"] == "cross_batch_reference":
                body = {
                    "schema_version": "rca-bench-history-query-receipt-v1",
                    "acceptance_status": "accepted",
                    "allowed_split_roles": ["validation"],
                    "evidence_ids": [observation["evidence_id"]],
                    "source_store_sha256": hashlib.sha256(b"source-store").hexdigest(),
                }
                query_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                history.append({"evidence_id": observation["evidence_id"], "query_sha256": query_hash})
        target, reference = stage["observations"]
        target["pairwise_delta_e76"] = [
            {"other_evidence_id": reference["evidence_id"], "delta_e76": 1.25}
        ]
        reference["pairwise_delta_e76"] = [
            {"other_evidence_id": target["evidence_id"], "delta_e76": 1.25}
        ]
    context["evidence_ids"] = all_ids
    context["history_refs"] = sorted(history, key=lambda row: row["evidence_id"])
    item["image_deliveries"] = []
    for stage in bundle["stages"]:
        for observation in stage["observations"]:
            row = {"slot_id": observation["slot_id"]}
            for prefix in ("expected", "delivered"):
                row.update({
                    f"{prefix}_image_id": observation["image_id"],
                    f"{prefix}_rgb_sha256": observation["rgb_sha256"],
                    f"{prefix}_source_run_id": observation["source_run_id"],
                    f"{prefix}_source_case_id": observation["source_case_id"],
                    f"{prefix}_source_vial_ordinal": observation["source_vial_ordinal"],
                    f"{prefix}_source_time": copy.deepcopy(observation["source_time"]),
                })
            item["image_deliveries"].append(row)
    item["evidence_bundle_sha256"] = canonical_sha256(bundle)
    item["prompt"] = "Explain.\nEvidence JSON:\n" + canonical_bytes(bundle).decode()
    item["prompt_sha256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
    return item, accepted_provenance(bundle, production=True)


def _wp1_test_primary(run: int, vial: int, disposition: str) -> tuple[dict, dict]:
    from evaluation.rca_bench.rca_bench import canonical_bytes, canonical_sha256

    item, _ = _qualification_primary(run, vial)
    item["development_partition"] = None
    context = item["evidence_bundle"]["target_context"]
    context["split_role"] = "test"
    context["recipe_id"] = "recipe-test"
    context["previous_recipe_id"] = "recipe-test-previous"
    context["disposition"] = disposition
    histories = []
    for stage in item["evidence_bundle"]["stages"]:
        for observation in stage["observations"]:
            observation["source_split_role"] = "test"
            observation["source_recipe_id"] = "recipe-test"
            if observation["source_role"] == "target":
                observation["source_previous_recipe_id"] = "recipe-test-previous"
        reference = stage["observations"][1]
        body = {
            "schema_version": "rca-bench-history-query-receipt-v1",
            "acceptance_status": "accepted", "allowed_split_roles": ["test"],
            "evidence_ids": [reference["evidence_id"]],
            "source_store_sha256": hashlib.sha256(b"source-store").hexdigest(),
        }
        histories.append({
            "evidence_id": reference["evidence_id"],
            "query_sha256": canonical_sha256(body),
        })
    context["history_refs"] = sorted(histories, key=lambda row: row["evidence_id"])
    item["evidence_bundle_sha256"] = canonical_sha256(item["evidence_bundle"])
    item["prompt"] = "Explain.\nEvidence JSON:\n" + canonical_bytes(item["evidence_bundle"]).decode()
    item["prompt_sha256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
    proof = accepted_provenance(item["evidence_bundle"], production=True)
    for query in proof["query_receipts"]:
        query["allowed_split_roles"] = ["test"]
    proof["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in proof.items() if key != "receipt_sha256"}
    )
    return item, proof


def test_project_primary_from_decided_records_and_reject_leakage() -> None:
    from evaluation.rca_bench.rca_bench import ContractError, validate_model_input
    from evaluation.rca_bench.rca_bench_input_bundle import project_primary_item

    bundle = model_input()
    records = _records(bundle)
    sequence = {
        row["evidence_id"]: next(
            observation["source_time"]["production_sequence_index"]
            for stage in bundle["stages"]
            for observation in stage["observations"]
            if observation["evidence_id"] == row["evidence_id"]
        )
        for row in records
    }
    provenance = accepted_provenance(bundle, production=True)
    _rehash_provenance_records(records, provenance)
    item = project_primary_item(
        records, _spec(bundle), sequence, provenance,
        item_id="item-001", development_partition="selection",
        prompt_template="Explain using only cited evidence.",
    )
    assert validate_model_input(item["evidence_bundle"], accepted_provenance=provenance)
    assert len(item["image_deliveries"]) == 4
    assert "fill_clip_path" not in json.dumps(item["evidence_bundle"])

    records[0]["label"] = "hidden"
    with pytest.raises(ContractError, match="record|field|forbidden|schema"):
        project_primary_item(
            records, _spec(bundle), sequence, provenance,
            item_id="item-001", development_partition="selection",
            prompt_template="Explain using only cited evidence.",
        )


def test_project_rejects_changed_record_with_stale_accepted_receipt() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import project_primary_item

    bundle = model_input()
    records = _records(bundle)
    provenance = accepted_provenance(bundle, production=True)
    _rehash_provenance_records(records, provenance)
    sequence = {
        row["evidence_id"]: (20 if row["run_id"] == "run-target" else 10)
        for row in records
    }
    for record in records:
        if record["run_id"] == "run-target":
            record["disposition"] = "discard"
    with pytest.raises(ContractError, match="source record.*hash|receipt.*record"):
        project_primary_item(
            records, _spec(bundle), sequence, provenance,
            item_id="item-stale", development_partition="selection",
            prompt_template="Explain.",
        )


def test_project_rejects_oracle_future_and_cross_role_reference() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import project_primary_item

    bundle = model_input()
    records = _records(bundle)
    sequence = {
        row["evidence_id"]: (20 if row["run_id"] == "run-target" else 10)
        for row in records
    }
    provenance = accepted_provenance(bundle, production=True)
    _rehash_provenance_records(records, provenance)

    oracle = copy.deepcopy(records)
    oracle[0]["mask_source"] = "oracle"
    with pytest.raises((ContractError, ValueError), match="oracle|mask_source"):
        project_primary_item(oracle, _spec(bundle), sequence, provenance,
                             item_id="item-001", development_partition="selection",
                             prompt_template="Explain.")

    future = dict(sequence)
    future[_spec(bundle)["reference_evidence_ids"][0]] = 21
    with pytest.raises(ContractError, match="earlier|future|sequence"):
        project_primary_item(records, _spec(bundle), future, provenance,
                             item_id="item-001", development_partition="selection",
                             prompt_template="Explain.")

    crossed = copy.deepcopy(records)
    reference_id = _spec(bundle)["reference_evidence_ids"][0]
    next(row for row in crossed if row["evidence_id"] == reference_id)["split_version"] = "other-split"
    with pytest.raises((ContractError, ValueError), match="split|source record|receipt"):
        project_primary_item(crossed, _spec(bundle), sequence, provenance,
                             item_id="item-001", development_partition="selection",
                             prompt_template="Explain.")


def test_freeze_helper_binds_model_and_count_equations_without_oracle_bytes() -> None:
    from evaluation.rca_bench.rca_bench import canonical_sha256
    from evaluation.rca_bench.rca_bench_input_bundle import build_model_freeze_receipt
    from tests.evaluation.rca_bench.test_rca_bench import freeze_expected, plan_fixture, plan_provenance

    plan = plan_fixture()
    locators = {
        "schema_version": "rca-bench-image-locators-v1",
        "plan_sha256": canonical_sha256(plan),
        "images": [
            {"image_id": row["delivered_image_id"], "rgb_path": f"/rgb/{row['delivered_image_id']}", "rgb_sha256": row["delivered_rgb_sha256"]}
            for row in plan["items"][0]["image_deliveries"]
        ],
    }
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
        "protocol", "builder_source", "scorer_source", "evidence_schema", "prompt", "guard_config",
        "development_partition_receipt", "qualification_access_log", "generation_config", "oracle",
        "split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code",
    )}
    hashes["protocol"] = "6a6465048e3839bdcce60c39686f6edf382dfff0b6d23493d0d9b4a4e4d3a4f3"
    hashes["prompt"] = plan["prompt_template_sha256"]
    hashes["guard_config"] = plan["guard_config_sha256"]
    hashes["evidence_schema"] = plan["evidence_schema_sha256"]
    receipt = build_model_freeze_receipt(
        plan, locators,
        model_repository="local/model", model_revision="revision-1",
        model_weight_sha256=hashlib.sha256(b"weights").hexdigest(),
        oracle_sha256=hashes["oracle"], split_ids=["split-v1"],
        test_hashes=[hashlib.sha256(b"tests").hexdigest()], git_commit="0" * 40,
        hashes=hashes,
        accepted_provenance_by_source_item=plan_provenance(plan),
    )
    assert receipt["planned_item_count"] == 1
    assert receipt["model_repository"] == "local/model"
    assert receipt["oracle_sha256"] == hashes["oracle"]
    assert freeze_expected(receipt)["protocol_sha256"] == hashes["protocol"]


def test_qualification_is_exactly_first_four_complete_runs_and_12_pairs() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import build_development_partition_artifacts

    rows = [_qualification_primary(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    primaries = [row[0] for row in rows]
    provenance = {item["source_item_id"]: proof for item, proof in rows}
    result = build_development_partition_artifacts(primaries, provenance)
    assert result["partition_receipt"]["grounding_qualification_run_ids"]
    assert result["qualification"]["pair_count"] == 12
    assert result["qualification"]["model_attempt_count"] == 24
    assert len(result["qualification_plan_items"]) == 24
    assert result == build_development_partition_artifacts(primaries, provenance)

    with pytest.raises(ContractError, match="six|eligible|complete"):
        build_development_partition_artifacts(primaries[:-3], provenance)
    incomplete = [item for item in primaries if not (item["run_id"] == "run-6" and item["vial_ordinal"] == 3)]
    with pytest.raises(ContractError, match="complete|three"):
        build_development_partition_artifacts(incomplete, provenance)


def test_development_partition_rejects_test_split_records() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import build_development_partition_artifacts

    rows = [_qualification_primary(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    primaries = [row[0] for row in rows]
    provenance = {item["source_item_id"]: proof for item, proof in rows}
    for item in primaries:
        item["evidence_bundle"]["target_context"]["split_role"] = "test"
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                observation["source_split_role"] = "test"

    with pytest.raises(ContractError, match="development.*split|test split"):
        build_development_partition_artifacts(primaries, provenance)


def test_qualification_rejects_donor_delta_e_threshold_and_access_log_is_sealed() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import build_development_partition_artifacts

    rows = [_qualification_primary(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    for item, _ in rows:
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                observation["median_lab"] = [1.0, 1.0, 1.0]
        from evaluation.rca_bench.rca_bench import canonical_bytes, canonical_sha256
        item["evidence_bundle_sha256"] = canonical_sha256(item["evidence_bundle"])
        item["prompt"] = "Explain.\nEvidence JSON:\n" + canonical_bytes(item["evidence_bundle"]).decode()
        item["prompt_sha256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
    provenance = {item["source_item_id"]: accepted_provenance(item["evidence_bundle"], production=True) for item, _ in rows}
    with pytest.raises(ContractError, match="donor|Delta|eligible"):
        build_development_partition_artifacts([row[0] for row in rows], provenance)


def test_development_reference_reuse_is_limited_to_one_frozen_partition() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import (
        assign_development_partitions,
        build_development_partition_artifacts,
    )

    rows = [_qualification_primary(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    primaries = [row[0] for row in rows]
    provenance = {item["source_item_id"]: proof for item, proof in rows}
    partition = assign_development_partitions(sorted({item["run_id"] for item in primaries}))
    selection_runs = partition["selection_run_ids"]
    same_partition = [item for item in primaries if item["run_id"] == selection_runs[0]]
    for stage, donor_stage in zip(
        same_partition[1]["evidence_bundle"]["stages"],
        same_partition[0]["evidence_bundle"]["stages"],
        strict=True,
    ):
        stage["observations"][1] = copy.deepcopy(donor_stage["observations"][1])
    build_development_partition_artifacts(primaries, provenance)

    qualification_run = partition["grounding_qualification_run_ids"][0]
    cross_partition = next(item for item in primaries if item["run_id"] == qualification_run)
    cross_partition["evidence_bundle"]["stages"][0]["observations"][1] = copy.deepcopy(
        same_partition[0]["evidence_bundle"]["stages"][0]["observations"][1]
    )
    with pytest.raises(ContractError, match="cannot cross frozen run partitions"):
        build_development_partition_artifacts(primaries, provenance)


def test_test_plan_is_wp1_primary_only_and_requires_both_abstention_classes() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import build_test_public_plan

    first, first_proof = _qualification_primary(1, 1)
    second, second_proof = _qualification_primary(1, 2)
    for item in (first, second):
        item["development_partition"] = None
        item["evidence_bundle"]["target_context"]["split_role"] = "test"
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                observation["source_split_role"] = "test"
        histories = []
        for stage in item["evidence_bundle"]["stages"]:
            reference = stage["observations"][1]
            body = {
                "schema_version": "rca-bench-history-query-receipt-v1",
                "acceptance_status": "accepted", "allowed_split_roles": ["test"],
                "evidence_ids": [reference["evidence_id"]],
                "source_store_sha256": hashlib.sha256(b"source-store").hexdigest(),
            }
            histories.append({
                "evidence_id": reference["evidence_id"],
                "query_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            })
        item["evidence_bundle"]["target_context"]["history_refs"] = sorted(histories, key=lambda row: row["evidence_id"])
        from evaluation.rca_bench.rca_bench import canonical_bytes, canonical_sha256
        item["evidence_bundle_sha256"] = canonical_sha256(item["evidence_bundle"])
        item["prompt"] = "Explain.\nEvidence JSON:\n" + canonical_bytes(item["evidence_bundle"]).decode()
        item["prompt_sha256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
    second["evidence_bundle"]["target_context"]["disposition"] = "insufficient_evidence"
    from evaluation.rca_bench.rca_bench import canonical_bytes, canonical_sha256
    second["evidence_bundle_sha256"] = canonical_sha256(second["evidence_bundle"])
    second["prompt"] = "Explain.\nEvidence JSON:\n" + canonical_bytes(second["evidence_bundle"]).decode()
    second["prompt_sha256"] = hashlib.sha256(second["prompt"].encode()).hexdigest()
    proofs = {
        first["source_item_id"]: accepted_provenance(first["evidence_bundle"], production=True),
        second["source_item_id"]: accepted_provenance(second["evidence_bundle"], production=True),
    }
    for proof in proofs.values():
        for query in proof["query_receipts"]:
            query["allowed_split_roles"] = ["test"]
        unsigned = {key: value for key, value in proof.items() if key != "receipt_sha256"}
        proof["receipt_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = {
        "split_version": "split-v1", "prompt_template_sha256": hashlib.sha256(b"p").hexdigest(),
        "guard_config_sha256": hashlib.sha256(b"g").hexdigest(),
        "evidence_schema_sha256": hashlib.sha256(b"e").hexdigest(),
        "model_input_schema_sha256": hashlib.sha256(b"m").hexdigest(),
        "source_artifacts": [],
    }
    audit = _partition_audit(
        benchmark_role="test", development_run_ids=[], test_run_ids=["run-1"],
        selection_run_ids=[], qualification_run_ids=[], recipe_role="test",
    )
    wp1 = _wp1_source_provenance([first, second], proofs)
    plan = build_test_public_plan(
        [first, second], proofs, metadata=metadata,
        wp1_source_provenance=wp1, partition_leakage_audit=audit,
    )
    assert plan["benchmark_role"] == "test" and all(row["role"] == "primary" for row in plan["items"])
    with pytest.raises(ContractError, match="both abstention"):
        build_test_public_plan(
            [first], {first["source_item_id"]: proofs[first["source_item_id"]]},
            metadata=metadata,
            wp1_source_provenance=_wp1_source_provenance(
                [first], {first["source_item_id"]: proofs[first["source_item_id"]]}
            ),
            partition_leakage_audit=audit,
        )
    rejected_wp1 = copy.deepcopy(wp1)
    rejected_wp1["acceptance_status"] = "rejected"
    with pytest.raises(ContractError, match="WP1|accepted"):
        build_test_public_plan(
            [first, second], proofs, metadata=metadata,
            wp1_source_provenance=rejected_wp1, partition_leakage_audit=audit,
        )


def test_development_plans_have_non_vacuous_controls_and_publication_is_no_clobber(tmp_path: Path) -> None:
    from evaluation.rca_bench.rca_bench import ContractError, canonical_sha256
    from evaluation.rca_bench.rca_bench_input_bundle import (
        build_development_partition_artifacts,
        build_development_public_plans,
        publish_pre_inference_bundle,
    )

    rows = [_qualification_primary(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    primaries = [row[0] for row in rows]
    provenance = {item["source_item_id"]: proof for item, proof in rows}
    partitioned = build_development_partition_artifacts(primaries, provenance)
    metadata = {
        "split_version": "split-v1", "prompt_template_sha256": hashlib.sha256(b"p").hexdigest(),
        "guard_config_sha256": hashlib.sha256(b"g").hexdigest(),
        "evidence_schema_sha256": hashlib.sha256(b"e").hexdigest(),
        "model_input_schema_sha256": hashlib.sha256(b"m").hexdigest(),
        "source_artifacts": [{"artifact_id": "part-b-records", "sha256": hashlib.sha256(b"records").hexdigest()}],
    }
    partition = partitioned["partition_receipt"]
    audit = _partition_audit(
        benchmark_role="development",
        development_run_ids=partition["eligible_run_ids"], test_run_ids=[],
        selection_run_ids=partition["selection_run_ids"],
        qualification_run_ids=partition["grounding_qualification_run_ids"],
        recipe_role="validation",
    )
    plans = build_development_public_plans(
        partitioned, provenance, metadata=metadata, partition_leakage_audit=audit,
    )
    roles = [item["role"] for item in plans["selection_plan"]["items"]]
    assert {"support_removed", "shuffled_evidence_ids", "unsupported_cause_probe"} <= set(roles)
    assert plans["qualification_plan"]["planned_item_count"] == 24
    support_control = next(
        item for item in plans["selection_plan"]["items"]
        if item["role"] == "support_removed"
    )
    details = support_control["control_receipt"]["details"]
    assert set(details) == {
        "limit_code", "removed_support_links", "removed_pair_count",
    }
    assert details["removed_pair_count"] == len(details["removed_support_links"])
    assert details["removed_pair_count"] >= 1
    primary = next(
        item for item in plans["selection_plan"]["items"]
        if item["item_id"] == support_control["paired_primary_item_id"]
    )
    assert primary["evidence_bundle"]["stages"] != support_control["evidence_bundle"]["stages"]
    assert all(
        observation["pairwise_delta_e76"] == []
        for stage in support_control["evidence_bundle"]["stages"]
        for observation in stage["observations"]
    )
    assert (
        support_control["evidence_bundle"]["target_context"]["history_refs"]
        == primary["evidence_bundle"]["target_context"]["history_refs"]
    )
    assert support_control["image_deliveries"] == primary["image_deliveries"]

    from evaluation.rca_bench.rca_bench_input_bundle import build_selection_controls
    vacuous = {
        item["source_item_id"]: {
            **copy.deepcopy(item["evidence_bundle"]),
            "target_context": {
                **copy.deepcopy(item["evidence_bundle"]["target_context"]),
                "disposition": "insufficient_evidence",
                "limit_codes": ["support_removed"],
            },
        }
        for item in partitioned["selection_primaries"]
    }
    with pytest.raises(ContractError, match="caller-supplied|trusted builder"):
        build_selection_controls(
            partitioned["selection_primaries"], provenance,
            support_removed_bundles_by_source_item=vacuous,
        )

    artifacts = {
        "plan": plans["qualification_plan"],
        "locators": {"schema_version": "rca-bench-image-locators-v1", "plan_sha256": canonical_sha256(plans["qualification_plan"]), "images": []},
        "accepted_provenance": plans["qualification_provenance"],
        "eligible_inventory": partitioned["eligible_inventory"],
        "partition_receipt": partitioned["partition_receipt"],
        "qualification": partitioned["qualification"],
        "qualification_access_log": partitioned["qualification_access_log"],
    }
    output = tmp_path / "published"
    receipt = publish_pre_inference_bundle(artifacts, output_dir=output)
    assert receipt["inventory_sha256"]
    with pytest.raises(ContractError, match="overwrite"):
        publish_pre_inference_bundle(artifacts, output_dir=output)


def test_public_builder_and_cli_cannot_reach_oracle_or_model_execution() -> None:
    from fluidmix.storage.artifacts.part_boundary_manifest import parse_local_imports

    root = Path(__file__).resolve().parents[3]
    module_source = (root / "src/evaluation/rca_bench/rca_bench_input_bundle.py").read_text()
    cli_source = (root / "cli/evaluation/rca_bench/build_rca_bench_inputs.py").read_text()
    imports = parse_local_imports(module_source, module_name="fluidmix.rca_bench_input_bundle")
    assert "fluidmix.rca_bench_oracle_builder" not in imports
    lowered = cli_source.lower()
    assert "transformers" not in lowered
    assert "run_model_phase" not in cli_source
    assert "build_hidden_oracle" not in cli_source
    assert "--oracle" not in cli_source.replace("--oracle-sha256", "")


def test_cross_role_identity_or_hash_duplication_is_rejected() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import validate_cross_role_isolation
    from tests.evaluation.rca_bench.test_rca_bench import plan_fixture

    development = plan_fixture()
    test = copy.deepcopy(development)
    test["benchmark_role"] = "test"
    test["development_partition"] = None
    test["items"][0]["development_partition"] = None
    with pytest.raises(ContractError, match="publication|provenance"):
        validate_cross_role_isolation([development], test)


def test_cross_role_isolation_rejects_empty_malformed_plans_before_audit() -> None:
    from evaluation.rca_bench.rca_bench import ContractError
    from evaluation.rca_bench.rca_bench_input_bundle import validate_cross_role_isolation

    with pytest.raises(ContractError, match="plan|provenance|publication|development"):
        validate_cross_role_isolation(
            [{"benchmark_role": "development"}],
            {"benchmark_role": "test", "source_artifacts": [], "items": []},
        )


def test_cross_role_isolation_binds_full_published_plans_receipts_and_audits() -> None:
    from evaluation.rca_bench.rca_bench import ContractError, canonical_sha256
    from evaluation.rca_bench.rca_bench_input_bundle import (
        build_development_partition_artifacts,
        build_development_public_plans,
        build_test_public_plan,
        validate_cross_role_isolation,
    )

    rows = [_qualification_primary(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    development_provenance = {item["source_item_id"]: proof for item, proof in rows}
    partitioned = build_development_partition_artifacts(
        [item for item, _ in rows], development_provenance
    )
    partition = partitioned["partition_receipt"]
    development_audit = _partition_audit(
        benchmark_role="development",
        development_run_ids=partition["eligible_run_ids"], test_run_ids=[],
        selection_run_ids=partition["selection_run_ids"],
        qualification_run_ids=partition["grounding_qualification_run_ids"],
        recipe_role="validation",
    )
    metadata = {
        "split_version": "split-v1", "prompt_template_sha256": hashlib.sha256(b"p").hexdigest(),
        "guard_config_sha256": hashlib.sha256(b"g").hexdigest(),
        "evidence_schema_sha256": hashlib.sha256(b"e").hexdigest(),
        "model_input_schema_sha256": hashlib.sha256(b"m").hexdigest(),
        "source_artifacts": [{"artifact_id": "part-b-records", "sha256": hashlib.sha256(b"records").hexdigest()}],
    }
    development = build_development_public_plans(
        partitioned, development_provenance, metadata=metadata,
        partition_leakage_audit=development_audit,
    )
    test_rows = [
        _wp1_test_primary(101, 1, "keep"),
        _wp1_test_primary(101, 2, "insufficient_evidence"),
    ]
    test_items = [item for item, _ in test_rows]
    test_provenance = {item["source_item_id"]: proof for item, proof in test_rows}
    wp1 = _wp1_source_provenance(test_items, test_provenance)
    test_audit = copy.deepcopy(development_audit)
    test_audit["benchmark_role"] = "test"
    test_audit["test_run_ids"] = ["run-101"]
    test_audit["recipes"].append({
        "recipe_id": "recipe-test", "role": "test", "ratio_cell": "ratio-test",
        "carryover_pair_id": "pair-test", "near_group": "near-test",
        "is_ratio_buffer": False, "render_seed": 3001,
    })
    test_audit["seed_ranges"]["test"] = [3000, 3999]
    test_audit["manifests"].append({
        "recipe_id": "recipe-test", "role": "test", "path": "test/recipe-test.json",
        "sha256": hashlib.sha256(b"test-recipe").hexdigest(),
    })
    test_plan = build_test_public_plan(
        test_items, test_provenance, metadata={**metadata, "source_artifacts": []},
        wp1_source_provenance=wp1, partition_leakage_audit=test_audit,
    )
    kwargs = {
        "development_provenance_by_partition": {
            "selection": development["selection_provenance"],
            "grounding_qualification": development["qualification_provenance"],
        },
        "test_provenance": test_provenance,
        "development_partition_receipt": partition,
        "development_publication_receipt": development["publication_receipt"],
        "development_partition_leakage_audit": development_audit,
        "test_partition_leakage_audit": test_audit,
        "wp1_source_provenance": wp1,
    }
    receipt = validate_cross_role_isolation(
        [development["selection_plan"], development["qualification_plan"]],
        test_plan, **kwargs,
    )
    assert receipt["status"] == "pass" and receipt["overlap_count"] == 0

    stale = copy.deepcopy(development["publication_receipt"])
    stale["selection_plan_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="publication receipt hashes"):
        validate_cross_role_isolation(
            [development["selection_plan"], development["qualification_plan"]],
            test_plan, **{**kwargs, "development_publication_receipt": stale},
        )

    def publication_for(selection_plan: dict, qualification_plan: dict) -> dict:
        body = {
            "schema_version": "rca-bench-development-publication-v1",
            "selection_plan_sha256": canonical_sha256(selection_plan),
            "selection_provenance_sha256": canonical_sha256(
                development["selection_provenance"]
            ),
            "qualification_plan_sha256": canonical_sha256(qualification_plan),
            "qualification_provenance_sha256": canonical_sha256(
                development["qualification_provenance"]
            ),
            "development_partition_receipt_sha256": canonical_sha256(partition),
            "partition_leakage_receipt_sha256": canonical_sha256(
                development["partition_leakage_receipt"]
            ),
        }
        return {**body, "receipt_sha256": canonical_sha256(body)}

    control_free = copy.deepcopy(development["selection_plan"])
    control_free["items"] = [
        item for item in control_free["items"] if item["role"] == "primary"
    ]
    control_free["planned_item_count"] = len(control_free["items"])
    control_free["model_attempt_planned_count"] = len(control_free["items"])
    control_free["pre_inference_validation_planned_count"] = 0
    with pytest.raises(ContractError, match="selection.*control|denominator"):
        validate_cross_role_isolation(
            [control_free, development["qualification_plan"]], test_plan,
            **{
                **kwargs,
                "development_publication_receipt": publication_for(
                    control_free, development["qualification_plan"]
                ),
            },
        )

    primary_only = copy.deepcopy(development["qualification_plan"])
    qualification_by_id = {
        item["item_id"]: item for item in primary_only["items"]
    }
    primary_only["items"] = [
        (
            {
                **copy.deepcopy(qualification_by_id[item["paired_primary_item_id"]]),
                "item_id": item["item_id"],
            }
            if item["role"] == "shuffled_images"
            else item
        )
        for item in primary_only["items"]
    ]
    with pytest.raises(ContractError, match="qualification.*pair|denominator"):
        validate_cross_role_isolation(
            [development["selection_plan"], primary_only], test_plan,
            **{
                **kwargs,
                "development_publication_receipt": publication_for(
                    development["selection_plan"], primary_only
                ),
            },
        )


def test_shuffled_delivery_noop_and_delta_at_threshold_are_rejected() -> None:
    from evaluation.rca_bench.rca_bench import ContractError, build_shuffled_images_control

    primary = primary_item()
    donor = copy.deepcopy(primary)
    donor["source_item_id"] = "source-donor"
    donor["run_id"] = "run-donor"
    slot = primary["image_deliveries"][0]["slot_id"]
    with pytest.raises(ContractError, match="strictly greater"):
        build_shuffled_images_control(
            primary, donor, changed_slot_ids=[slot], delta_e76_by_slot={slot: 2.0},
            item_id="control-threshold",
        )
    with pytest.raises(ContractError, match="disjoint|unique"):
        build_shuffled_images_control(
            primary, donor, changed_slot_ids=[slot], delta_e76_by_slot={slot: 2.1},
            item_id="control-noop",
        )


def test_qualification_builder_rejects_wrong_twelve_pair_denominator() -> None:
    from evaluation.rca_bench.rca_bench import ContractError, build_grounding_qualification

    rows = [_qualification_primary(run, vial)[0] for run in range(1, 5) for vial in range(1, 4)]
    with pytest.raises(ContractError, match="exactly four runs|three primary|12"):
        build_grounding_qualification(rows[:-1], {})


def test_hidden_oracle_construction_is_a_separate_exact_plan_boundary() -> None:
    from evaluation.rca_bench.rca_bench_oracle_builder import build_hidden_oracle
    from tests.evaluation.rca_bench.test_rca_bench import oracle_item, plan_fixture, plan_provenance

    plan = plan_fixture()
    oracle = build_hidden_oracle(
        plan, [oracle_item()],
        protocol_sha256="6a6465048e3839bdcce60c39686f6edf382dfff0b6d23493d0d9b4a4e4d3a4f3",
        accepted_provenance_by_source_item=plan_provenance(plan),
    )
    assert oracle["items"][0]["item_id"] == plan["items"][0]["item_id"]

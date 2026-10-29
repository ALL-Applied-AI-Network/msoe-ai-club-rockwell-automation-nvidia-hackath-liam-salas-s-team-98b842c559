from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rca_bench import (  # noqa: E402
    accepted_provenance,
    freeze_expected,
    freeze_fixture,
    locator_fixture,
    oracle_fixture,
    plan_fixture,
    primary_item,
    plan_provenance,
)


def _canonical_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def _freeze_source_paths(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """Create explicit, tiny source inputs used by real-freeze controller tests."""
    sources: dict[str, object] = {}
    for name in (
        "protocol", "builder_source", "scorer_source", "evidence_schema", "prompt",
        "guard_config", "test_source", "split_spec", "recipes", "transitions",
        "real_reserved", "manifests", "checker_code",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"{name}-v1\n", encoding="utf-8")
        sources[name] = path
    sources["tests"] = [sources.pop("test_source")]
    sources["partitions"] = {
        name: sources.pop(name)
        for name in (
            "split_spec", "recipes", "transitions", "real_reserved", "manifests",
            "checker_code",
        )
    }
    sources["git_commit"] = "0" * 40
    generation_path = tmp_path / "generation.json"
    _canonical_write(
        generation_path,
        {"mode": "deterministic_template", "temperature": 0},
    )
    return sources, generation_path


def _real_plan_locators(tmp_path: Path, sources: dict[str, object]) -> tuple[dict, dict, dict]:
    from fluidmix.rca_bench import canonical_sha256, sha256_file

    plan = plan_fixture()
    item = plan["items"][0]
    rgb_root = tmp_path / "rgb-real"
    rgb_root.mkdir()
    rgb_paths = {}
    for delivery in item["image_deliveries"]:
        path = rgb_root / f"{delivery['delivered_image_id']}.png"
        path.write_bytes(f"rgb:{delivery['delivered_image_id']}".encode())
        digest = sha256_file(path)
        delivery["expected_rgb_sha256"] = digest
        delivery["delivered_rgb_sha256"] = digest
        rgb_paths[delivery["delivered_image_id"]] = path
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                if observation["image_id"] == delivery["delivered_image_id"]:
                    observation["rgb_sha256"] = digest
    item["evidence_bundle_sha256"] = canonical_sha256(item["evidence_bundle"])
    item["prompt"] = (
        "Explain the decision using only cited evidence.\nEvidence JSON:\n"
        + json.dumps(item["evidence_bundle"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    item["prompt_sha256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
    provenance = plan_provenance(plan)
    plan["source_artifacts"][-1]["sha256"] = provenance[item["source_item_id"]]["receipt_sha256"]
    plan["prompt_template_sha256"] = sha256_file(sources["prompt"])
    plan["guard_config_sha256"] = sha256_file(sources["guard_config"])
    plan["evidence_schema_sha256"] = sha256_file(sources["evidence_schema"])
    locators = {
        "schema_version": "rca-bench-image-locators-v1",
        "plan_sha256": canonical_sha256(plan),
        "images": [
            {
                "image_id": delivery["delivered_image_id"],
                "rgb_path": str(rgb_paths[delivery["delivered_image_id"]]),
                "rgb_sha256": delivery["delivered_rgb_sha256"],
            }
            for delivery in item["image_deliveries"]
        ],
    }
    return plan, locators, provenance


def _mark_template_freeze(freeze: dict, config: dict | None = None) -> None:
    from fluidmix.rca_bench import canonical_sha256
    from fluidmix.rca_bench_controller import template_weight_binding_sha256

    config = config or {"mode": "deterministic_template", "temperature": 0}
    freeze["model_repository"] = "deterministic-template"
    freeze["model_revision"] = "rca-bench-template-v1"
    freeze["generation_config_sha256"] = canonical_sha256(config)
    freeze["model_weight_sha256"] = template_weight_binding_sha256(config)


def _real_template_runtime(tmp_path: Path, count: int = 1) -> tuple[dict, dict, dict, dict, dict, Path]:
    import copy

    from fluidmix.rca_bench import canonical_sha256
    from fluidmix.rca_bench_controller import compute_actual_freeze_bindings

    sources, generation_path = _freeze_source_paths(tmp_path)
    plan, locators, provenance = _real_plan_locators(tmp_path, sources)
    if count > 1:
        base = plan["items"][0]
        items = []
        provenance = {}
        for index in range(count):
            item = copy.deepcopy(base)
            item["item_id"] = f"item-{index}"
            item["source_item_id"] = f"source-{index}"
            items.append(item)
            provenance[item["source_item_id"]] = accepted_provenance(
                item["evidence_bundle"], production=True
            )
        plan["items"] = items
        plan["planned_item_count"] = count
        plan["model_attempt_planned_count"] = count
        locators["plan_sha256"] = canonical_sha256(plan)
    config = json.loads(generation_path.read_text())
    actual = compute_actual_freeze_bindings(
        protocol_path=sources["protocol"], builder_source_path=sources["builder_source"],
        scorer_source_path=sources["scorer_source"], test_paths=sources["tests"],
        evidence_schema_path=sources["evidence_schema"], prompt_path=sources["prompt"],
        guard_config_path=sources["guard_config"], partition_paths=sources["partitions"],
        git_commit=sources["git_commit"], generation_config=config,
    )
    freeze = freeze_fixture({**plan, "items": [plan["items"][0]]})
    freeze.update(actual)
    freeze.update({
        "plan_sha256": canonical_sha256(plan),
        "locator_sha256": canonical_sha256(locators),
        "planned_item_count": count,
        "model_attempt_planned_count": count,
        "required_abstention_primary_count": 0,
        "non_required_abstention_primary_count": count,
    })
    _mark_template_freeze(freeze, config)
    return plan, locators, provenance, freeze, sources, generation_path


def test_build_and_publish_locators_verify_rgb_bytes_and_refuse_clobber(tmp_path: Path) -> None:
    from fluidmix.rca_bench_controller import (
        build_public_plan,
        build_image_locators,
        publish_public_artifacts,
    )
    from fluidmix.rca_bench import ContractError, canonical_sha256

    item = primary_item()
    rgb_root = tmp_path / "rgb"
    rgb_root.mkdir()
    paths = {}
    for delivery in item["image_deliveries"]:
        path = rgb_root / f"{delivery['delivered_image_id']}.png"
        path.write_bytes(f"rgb:{delivery['delivered_image_id']}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        delivery["expected_rgb_sha256"] = digest
        delivery["delivered_rgb_sha256"] = digest
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                if observation["image_id"] == delivery["delivered_image_id"]:
                    observation["rgb_sha256"] = digest
        paths[delivery["delivered_image_id"]] = path
    item["prompt"] = (
        "Explain the decision using only cited evidence.\nEvidence JSON:\n"
        + json.dumps(item["evidence_bundle"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    item["prompt_sha256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
    provenance = {item["source_item_id"]: accepted_provenance(item["evidence_bundle"], production=True)}
    plan = build_public_plan(
        [item],
        benchmark_role="development",
        development_partition="selection",
        split_version="split-v1",
        prompt_template_sha256=hashlib.sha256(b"prompt").hexdigest(),
        guard_config_sha256=hashlib.sha256(b"guard").hexdigest(),
        evidence_schema_sha256=hashlib.sha256(b"evidence").hexdigest(),
        model_input_schema_sha256=hashlib.sha256(b"input").hexdigest(),
        source_artifacts=[
            {"artifact_id": "part-b", "sha256": hashlib.sha256(b"part-b").hexdigest()},
            {"artifact_id": "accepted", "sha256": provenance[item["source_item_id"]]["receipt_sha256"]},
        ],
        accepted_provenance_by_source_item=provenance,
    )
    locators = build_image_locators(plan, rgb_paths=paths, verify_files=True)
    assert locators["plan_sha256"] == canonical_sha256(plan)

    plan_path = tmp_path / "rca-bench-plan-v1.json"
    locator_path = tmp_path / "rca-bench-image-locators-v1.json"
    publish_public_artifacts(plan, locators, plan_path=plan_path, locator_path=locator_path,
                             accepted_provenance_by_source_item=provenance)
    with pytest.raises(ContractError, match="overwrite|exists"):
        publish_public_artifacts(plan, locators, plan_path=plan_path, locator_path=locator_path,
                                 accepted_provenance_by_source_item=provenance)

    paths[item["image_deliveries"][0]["delivered_image_id"]].write_bytes(b"tampered")
    with pytest.raises(ContractError, match="hash|mismatch|unreadable"):
        build_image_locators(plan, rgb_paths=paths, verify_files=True)


def test_oracle_publisher_is_plan_bound_and_freeze_publisher_accepts_digest_only(tmp_path: Path) -> None:
    from fluidmix.rca_bench import ContractError, canonical_sha256, validate_freeze_receipt
    from fluidmix.rca_bench_controller import (
        publish_oracle_artifact,
        publish_freeze_digest_only,
        validate_freeze_bindings,
    )

    plan = plan_fixture()
    provenance = plan_provenance(plan)
    locators = locator_fixture(plan)
    oracle = oracle_fixture(plan)
    oracle_path = tmp_path / "oracle.json"
    published_oracle = publish_oracle_artifact(
        oracle, plan, path=oracle_path,
        accepted_provenance_by_source_item=provenance,
    )
    assert published_oracle["plan_sha256"] == canonical_sha256(plan)
    assert oracle_path.exists()

    freeze = freeze_fixture(plan)
    _mark_template_freeze(freeze)
    validate_freeze_bindings(
        freeze, plan,
        accepted_provenance_by_source_item=provenance,
        expected_bindings=freeze_expected(freeze),
    )
    freeze_path = tmp_path / "freeze.json"
    publish_freeze_digest_only(
        freeze, plan, locators,
        oracle_sha256=canonical_sha256(oracle),
        path=freeze_path,
        accepted_provenance_by_source_item=provenance,
        expected_bindings=freeze_expected(freeze),
    )
    assert freeze_path.exists()
    validate_freeze_receipt(
        freeze, plan, locators=locators, oracle=oracle,
        accepted_provenance_by_source_item=provenance,
        expected_bindings=freeze_expected(freeze),
    )

    wrong_oracle = dict(oracle)
    wrong_oracle["plan_sha256"] = canonical_sha256({"different": True})
    with pytest.raises(ContractError, match="plan hash|public plan"):
        publish_oracle_artifact(
            wrong_oracle, plan, path=tmp_path / "wrong-oracle.json",
            accepted_provenance_by_source_item=provenance,
        )


def test_actual_freeze_bindings_rehash_files_and_canonical_template_config(tmp_path: Path) -> None:
    from fluidmix.rca_bench import ContractError
    from fluidmix.rca_bench_controller import (
        compute_actual_freeze_bindings,
        validate_actual_freeze_bindings,
    )

    files = {}
    for name in ("protocol", "builder", "scorer", "evidence", "prompt", "guard", "split"):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"{name}-v1\n", encoding="utf-8")
        files[name] = path
    bindings = compute_actual_freeze_bindings(
        protocol_path=files["protocol"],
        builder_source_path=files["builder"],
        scorer_source_path=files["scorer"],
        test_paths=[files["split"]],
        evidence_schema_path=files["evidence"],
        prompt_path=files["prompt"],
        guard_config_path=files["guard"],
        partition_paths={name: files["split"] for name in (
            "split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code"
        )},
        git_commit="0" * 40,
        generation_config={"mode": "deterministic_template", "temperature": 0},
    )
    assert bindings["generation_config_sha256"]
    freeze = {**freeze_fixture(plan_fixture()), **bindings}
    validate_actual_freeze_bindings(freeze, bindings)
    changed_config = compute_actual_freeze_bindings(
        protocol_path=files["protocol"],
        builder_source_path=files["builder"],
        scorer_source_path=files["scorer"],
        test_paths=[files["split"]],
        evidence_schema_path=files["evidence"],
        prompt_path=files["prompt"],
        guard_config_path=files["guard"],
        partition_paths={name: files["split"] for name in (
            "split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code"
        )},
        git_commit="0" * 40,
        generation_config={"mode": "deterministic_template", "temperature": 1},
    )
    with pytest.raises(ContractError, match="generation_config|actual"):
        validate_actual_freeze_bindings(freeze, changed_config)
    files["guard"].write_text("guard-tampered\n", encoding="utf-8")
    with pytest.raises(ContractError, match="guard|actual|hash"):
        changed = compute_actual_freeze_bindings(
            protocol_path=files["protocol"],
            builder_source_path=files["builder"],
            scorer_source_path=files["scorer"],
            test_paths=[files["split"]],
            evidence_schema_path=files["evidence"],
            prompt_path=files["prompt"],
            guard_config_path=files["guard"],
            partition_paths={name: files["split"] for name in (
                "split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code"
            )},
            git_commit="0" * 40,
            generation_config={"mode": "deterministic_template", "temperature": 0},
        )
        validate_actual_freeze_bindings(freeze, changed)


def test_template_projection_freezes_response_and_sorted_positional_limit_atoms() -> None:
    import json

    from fluidmix.rca_bench import deterministic_template
    from fluidmix.rca_bench_controller import template_projection

    item = primary_item()
    item["evidence_bundle"]["target_context"]["limit_codes"] = ["limit_a", "limit_z"]
    item["prompt"] = (
        "Explain the decision using only cited evidence.\nEvidence JSON:\n"
        + json.dumps(item["evidence_bundle"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    item["prompt_sha256"] = __import__("hashlib").sha256(item["prompt"].encode()).hexdigest()
    provenance = accepted_provenance(item["evidence_bundle"])
    projection = template_projection(item, accepted_provenance=provenance)
    assert projection["response"] == deterministic_template(item, accepted_provenance=provenance)
    assert [atom["statement_id"] for atom in projection["supportable_statements"]] == [
        "template-decision", "template-limit-1", "template-limit-2"
    ]
    assert [atom["normalized_claim"] for atom in projection["supportable_statements"][1:]] == [
        "evidence limit limit_a", "evidence limit limit_z"
    ]


def test_ledger_phase_machine_requires_validation_first_and_terminal_order(tmp_path: Path) -> None:
    from fluidmix.rca_bench import ContractError
    from fluidmix.rca_bench_controller import ControllerLedger, validate_ledger_state

    items = [
        {"item_id": "primary-a", "execution_policy": "model_attempt"},
        {"item_id": "ids-control", "execution_policy": "pre_inference_validation_only"},
        {"item_id": "primary-b", "execution_policy": "model_attempt"},
    ]
    ledger = ControllerLedger(tmp_path / "run.jsonl", items)
    ledger.append("preflight_started", payload={})
    ledger.append("preflight_completed", payload={})
    ledger.append("validation_started", item_id="ids-control", execution_index=0, payload={})
    with pytest.raises(ContractError, match="terminal|incomplete"):
        validate_ledger_state(ledger.events, items)
    with pytest.raises(ContractError, match="terminal|validation"):
        ledger.append("model_attempt_started", item_id="primary-a", execution_index=0, payload={})
    ledger.append("validation_terminal", item_id="ids-control", execution_index=0, payload={})
    ledger.append("model_attempt_started", item_id="primary-a", execution_index=0, payload={})
    ledger.append("model_attempt_terminal", item_id="primary-a", execution_index=0, payload={})
    ledger.append("model_attempt_started", item_id="primary-b", execution_index=1, payload={})
    ledger.append("model_attempt_terminal", item_id="primary-b", execution_index=1, payload={})
    assert validate_ledger_state(ledger.events, items)
    with pytest.raises(ContractError, match="terminal|fatal|phase|out of order"):
        ledger.append("validation_started", item_id="ids-control", execution_index=1, payload={})

    fatal_ledger = ControllerLedger(tmp_path / "fatal.jsonl", items)
    fatal_ledger.append("preflight_started", payload={})
    fatal_ledger.append("fatal", payload={"fatal_code": "test"})
    with pytest.raises(ContractError, match="fatal|terminal"):
        fatal_ledger.append("preflight_completed", payload={})


def test_finalize_fatal_reopens_from_hash_addressed_sidecars_after_memory_loss(tmp_path: Path) -> None:
    from fluidmix.rca_bench import canonical_sha256
    from fluidmix.rca_bench_controller import ControllerLedger, finalize_fatal_reopen

    items = [{"item_id": "primary-a", "execution_policy": "model_attempt"}]
    ledger = ControllerLedger(tmp_path / "fatal.jsonl", items)
    ledger.append("preflight_started", payload={})
    ledger.append("preflight_completed", payload={})
    ledger.append("model_attempt_started", item_id="primary-a", execution_index=0, payload={})
    fatal = ledger.append("fatal", payload={"fatal_code": "process_crash"})
    ledger.payloads.clear()
    raw = finalize_fatal_reopen(
        ledger.path,
        items=items,
        raw_bindings={
            "plan_sha256": canonical_sha256({"plan": True}),
            "freeze_receipt_sha256": canonical_sha256({"freeze": True}),
            "model_repository": "deterministic-template",
            "model_revision": "rca-bench-template-v1",
            "model_weight_sha256": canonical_sha256({"template": True}),
            "generation_config_sha256": canonical_sha256({"mode": "template"}),
        },
        fatal_event_sha256=fatal["event_sha256"],
    )
    assert raw["run_status"] == "fatal_incomplete"
    assert raw["records"][0]["record_status"] == "inference_error"
    assert (tmp_path / "fatal.jsonl.payloads" / f"{fatal['payload_sha256']}.json").exists()


@pytest.mark.parametrize("tamper", ["delete", "change"])
def test_finalize_fatal_rejects_missing_or_changed_payload_sidecar(
    tmp_path: Path, tamper: str
) -> None:
    from fluidmix.rca_bench import ContractError, canonical_sha256
    from fluidmix.rca_bench_controller import ControllerLedger, finalize_fatal_reopen

    items = [{"item_id": "primary-a", "execution_policy": "model_attempt"}]
    ledger = ControllerLedger(tmp_path / f"{tamper}.jsonl", items)
    ledger.append("preflight_started", payload={})
    fatal = ledger.append("fatal", payload={"fatal_code": "process_crash"})
    sidecar = ledger.sidecar_dir / f"{fatal['payload_sha256']}.json"
    if tamper == "delete":
        sidecar.unlink()
    else:
        sidecar.chmod(0o600)
        sidecar.write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="sidecar"):
        finalize_fatal_reopen(
            ledger.path,
            items=items,
            raw_bindings={
                "plan_sha256": canonical_sha256({"plan": True}),
                "freeze_receipt_sha256": canonical_sha256({"freeze": True}),
                "model_repository": "deterministic-template",
                "model_revision": "rca-bench-template-v1",
                "model_weight_sha256": canonical_sha256({"template": True}),
                "generation_config_sha256": canonical_sha256({"mode": "template"}),
            },
            fatal_event_sha256=fatal["event_sha256"],
        )


def test_template_baseline_writes_complete_cpu_raw_without_oracle_or_weights(tmp_path: Path) -> None:
    from fluidmix.rca_bench import canonical_sha256
    from fluidmix.rca_bench_controller import run_template_baseline

    plan, locators, provenance, freeze, sources, generation_path = _real_template_runtime(tmp_path)
    raw = run_template_baseline(
        plan,
        locators,
        freeze,
        accepted_provenance_by_source_item=provenance,
        freeze_source_paths=sources,
        generation_config_path=generation_path,
        ledger_path=tmp_path / "template.jsonl",
        raw_path=tmp_path / "raw.json",
    )
    assert raw["run_status"] == "complete"
    assert raw["completed_count"] == 1
    assert raw["records"][0]["response"]
    assert raw["records"][0]["processor_telemetry"]["ordered_image_count"] == 4
    assert raw["records"][0]["generation_config_sha256"] == freeze["generation_config_sha256"]
    assert raw["plan_sha256"] == canonical_sha256(plan)


def _template_oracle(plan: dict[str, object], provenance: dict[str, object]) -> dict[str, object]:
    from fluidmix.rca_bench import canonical_sha256, template_projection

    item = plan["items"][0]
    projection = template_projection(
        item, accepted_provenance=provenance[item["source_item_id"]]
    )
    context = item["evidence_bundle"]["target_context"]
    row = {
        "item_id": item["item_id"],
        "source_item_id": item["source_item_id"],
        "role": item["role"],
        "paired_primary_item_id": item["paired_primary_item_id"],
        "expected_decision": context["disposition"],
        "required_abstention": context["disposition"] == "insufficient_evidence",
        "required_statement_ids": projection["required_statement_ids"],
        "supportable_statements": projection["supportable_statements"],
        "cause_expectation": None,
        "cause_supported_by_part_b": False,
        "post_inference_label_ref": None,
    }
    return {
        "schema_version": "rca-bench-oracle-v1",
        "plan_sha256": canonical_sha256(plan),
        "protocol_sha256": __import__("hashlib").sha256(b"protocol").hexdigest(),
        "items": [row],
    }


def _multi_template_plan() -> tuple[dict[str, object], dict[str, object]]:
    import copy

    plan = plan_fixture()
    items = []
    for index in range(3):
        item = copy.deepcopy(plan["items"][0])
        item["item_id"] = f"item-{index}"
        item["source_item_id"] = f"source-{index}"
        items.append(item)
    plan["items"] = items
    plan["planned_item_count"] = 3
    plan["model_attempt_planned_count"] = 3
    return plan, {item["source_item_id"]: accepted_provenance(item["evidence_bundle"], production=True)
                  for item in items}


def _multi_template_oracle(plan: dict[str, object], provenance: dict[str, object]) -> dict[str, object]:
    import copy

    oracle = _template_oracle({**plan, "items": [plan["items"][0]]},
                              {plan["items"][0]["source_item_id"]: provenance[plan["items"][0]["source_item_id"]]})
    rows = []
    for item in plan["items"]:
        row = copy.deepcopy(oracle["items"][0])
        row["item_id"] = item["item_id"]
        row["source_item_id"] = item["source_item_id"]
        rows.append(row)
    oracle["plan_sha256"] = __import__("fluidmix.rca_bench", fromlist=["canonical_sha256"]).canonical_sha256(plan)
    oracle["items"] = rows
    return oracle


def test_template_score_reconstructs_exact_items_metrics_and_not_evaluated_adoption(tmp_path: Path) -> None:
    from fluidmix.rca_bench import ContractError, canonical_sha256
    from fluidmix.rca_bench_controller import (
        build_score_artifact,
        run_template_baseline,
        validate_score_artifact,
    )

    plan, locators, provenance, freeze, sources, generation_path = _real_template_runtime(tmp_path)
    oracle = _template_oracle(plan, provenance)
    freeze["oracle_sha256"] = canonical_sha256(oracle)
    raw = run_template_baseline(
        plan, locators, freeze,
        accepted_provenance_by_source_item=provenance,
        freeze_source_paths=sources,
        generation_config_path=generation_path,
        ledger_path=tmp_path / "score.jsonl",
        raw_path=tmp_path / "raw.json",
    )
    score = build_score_artifact(
        plan, locators, oracle, raw, freeze,
        accepted_provenance_by_source_item=provenance,
        ledger_events=__import__("json").loads("[" + ",".join(
            (tmp_path / "score.jsonl").read_text().splitlines()
        ) + "]"),
        expected_bindings=freeze_expected(freeze),
        method_id="deterministic-template",
    )
    assert score["adoption"]["status"] == "not_evaluated"
    assert score["final_metrics"]["primary_item_count"] == 1
    assert score["items"][0]["score_status"] == "completed"
    validate_score_artifact(
        score, plan=plan, locators=locators, oracle=oracle, raw=raw,
        freeze_receipt=freeze, accepted_provenance_by_source_item=provenance,
        ledger_events=__import__("json").loads("[" + ",".join(
            (tmp_path / "score.jsonl").read_text().splitlines()
        ) + "]"),
        expected_bindings=freeze_expected(freeze),
    )
    tampered = __import__("copy").deepcopy(score)
    tampered["final_metrics"]["primary_item_count"] = 99
    with pytest.raises(ContractError, match="metric|recomputed|primary"):
        validate_score_artifact(
            tampered, plan=plan, locators=locators, oracle=oracle, raw=raw,
            freeze_receipt=freeze, accepted_provenance_by_source_item=provenance,
            ledger_events=__import__("json").loads("[" + ",".join(
                (tmp_path / "score.jsonl").read_text().splitlines()
            ) + "]"),
            expected_bindings=freeze_expected(freeze),
        )


@pytest.mark.parametrize("fatal_before", [0, 1, 2])
def test_primary_fatal_positions_keep_zero_failure_projection_and_denominator(
    tmp_path: Path, fatal_before: int
) -> None:
    import json

    from fluidmix.rca_bench import canonical_sha256
    from fluidmix.rca_bench_controller import (
        build_score_artifact,
        run_template_baseline,
    )

    plan, locators, provenance, freeze, sources, generation_path = _real_template_runtime(
        tmp_path, count=3
    )
    oracle = _multi_template_oracle(plan, provenance)
    freeze["oracle_sha256"] = canonical_sha256(oracle)
    raw = run_template_baseline(
        plan, locators, freeze,
        accepted_provenance_by_source_item=provenance,
        freeze_source_paths=sources,
        generation_config_path=generation_path,
        ledger_path=tmp_path / f"fatal-{fatal_before}.jsonl",
        raw_path=tmp_path / f"fatal-{fatal_before}.raw.json",
        fatal_before_attempt=fatal_before,
    )
    events = [json.loads(line) for line in (tmp_path / f"fatal-{fatal_before}.jsonl").read_text().splitlines()]
    score = build_score_artifact(
        plan, locators, oracle, raw, freeze,
        accepted_provenance_by_source_item=provenance,
        ledger_events=events,
        expected_bindings=freeze_expected(freeze),
        method_id="deterministic-template",
    )
    assert score["final_metrics"]["primary_item_count"] == 3
    assert score["raw_metrics"]["primary_item_count"] == 3
    if fatal_before < 3:
        failed = score["items"][fatal_before]
        assert failed["score_status"] == "not_run_after_fatal"
        assert failed["raw_item_score"] == failed["final_item_score"]
        assert failed["raw_item_score"]["failures"] == ["fatal_run_terminated"]
        assert failed["raw_item_score"]["numeric_tp"] == 0
        assert failed["raw_item_score"]["parse_success"] is False
    assert score["adoption"]["status"] == "not_evaluated"

    with pytest.raises(Exception, match="method|identity|spoof"):
        build_score_artifact(
            plan, locators, oracle, raw, freeze,
            accepted_provenance_by_source_item=provenance,
            ledger_events=events,
            expected_bindings=freeze_expected(freeze),
            method_id="candidate-model",
        )


def test_controller_cli_exposes_sealed_phase_commands() -> None:
    import subprocess
    import sys

    script = Path(__file__).parents[1] / "scripts" / "run_rca_bench.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        "publish-public", "publish-oracle", "publish-freeze",
        "run-template", "finalize-fatal", "score",
    ):
        assert command in result.stdout


def test_parseable_semantic_failure_scores_raw_before_fallback() -> None:
    from fluidmix.rca_bench import canonical_sha256, template_projection
    from fluidmix.rca_bench_controller import reconstruct_score_items

    plan = plan_fixture()
    provenance = plan_provenance(plan)
    item = plan["items"][0]
    response = json.loads(
        template_projection(
            item, accepted_provenance=provenance[item["source_item_id"]]
        )["response"]
    )
    response["disposition"] = "discard" if response["disposition"] == "keep" else "keep"
    raw_response = json.dumps(response, sort_keys=True, separators=(",", ":"))
    raw = {
        "records": [{
            "record_index": 0,
            "item_id": item["item_id"],
            "record_status": "completed",
            "response": raw_response,
            "response_utf8_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
            "error_type": None,
            "validation_stage": None,
            "rejection_code": None,
            "fatal_event_sha256": None,
            "not_run_reason": None,
        }]
    }
    oracle = _template_oracle(plan, provenance)
    row = reconstruct_score_items(
        plan, raw, oracle, accepted_provenance_by_source_item=provenance
    )[0]
    assert row["raw_parse_success"] is True
    assert row["raw_item_score"]["supported_explanation_pass"] is False
    assert row["final_guard_result"] == "fallback"
    assert row["final_item_score"]["supported_explanation_pass"] is True


def test_invalid_actual_freeze_fatalizes_before_template_call(tmp_path: Path, monkeypatch) -> None:
    from fluidmix.rca_bench import ContractError
    import fluidmix.rca_bench_controller as controller

    sources, generation_path = _freeze_source_paths(tmp_path)
    plan, locators, provenance = _real_plan_locators(tmp_path, sources)
    freeze = freeze_fixture(plan)  # deliberately stale and not the template sentinel
    calls = []
    monkeypatch.setattr(
        controller, "template_projection", lambda *args, **kwargs: calls.append(True)
    )
    with pytest.raises(ContractError, match="freeze|template|actual"):
        controller.run_template_baseline(
            plan, locators, freeze,
            accepted_provenance_by_source_item=provenance,
            freeze_source_paths=sources,
            generation_config_path=generation_path,
            ledger_path=tmp_path / "stale.jsonl",
            raw_path=tmp_path / "stale.raw.json",
        )
    assert calls == []
    events = [json.loads(line) for line in (tmp_path / "stale.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["preflight_started", "fatal"]


def test_fatal_candidate_rejects_spoofed_template_method_label(monkeypatch) -> None:
    from fluidmix.rca_bench import ContractError
    import fluidmix.rca_bench_controller as controller

    plan = plan_fixture()
    provenance = plan_provenance(plan)
    oracle = oracle_fixture(plan)
    raw = {
        "run_status": "fatal_incomplete",
        "records": [{
            "record_index": 0, "item_id": plan["items"][0]["item_id"],
            "record_status": "not_run_after_fatal", "response_utf8_sha256": None,
            "error_type": None, "validation_stage": None, "rejection_code": None,
            "fatal_event_sha256": "f" * 64, "not_run_reason": "fatal_run_terminated",
        }],
    }
    freeze = freeze_fixture(plan)
    monkeypatch.setattr(controller, "validate_score_artifact", lambda *args, **kwargs: {})
    with pytest.raises(ContractError, match="method|identity|spoof"):
        controller.build_score_artifact(
            plan, locator_fixture(plan), oracle, raw, freeze,
            accepted_provenance_by_source_item=provenance,
            ledger_events=[], expected_bindings=freeze_expected(freeze),
            method_id="deterministic-template",
        )


def test_invalid_raw_is_rejected_before_any_oracle_scoring(monkeypatch) -> None:
    from fluidmix.rca_bench import ContractError, template_projection
    import fluidmix.rca_bench as contracts
    from fluidmix.rca_bench_controller import build_score_artifact

    plan = plan_fixture()
    provenance = plan_provenance(plan)
    item = plan["items"][0]
    response = template_projection(
        item, accepted_provenance=provenance[item["source_item_id"]]
    )["response"]
    invalid_raw = {
        "run_status": "complete",
        "journal_terminal_sha256": "0" * 64,
        "records": [{
            "record_index": 0, "item_id": item["item_id"], "record_status": "completed",
            "response": response,
            "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "error_type": None, "validation_stage": None, "rejection_code": None,
            "fatal_event_sha256": None, "not_run_reason": None,
        }],
    }
    scoring_calls = []
    original = contracts.score_parsed_response
    monkeypatch.setattr(
        contracts, "score_parsed_response",
        lambda *args, **kwargs: (scoring_calls.append(True), original(*args, **kwargs))[1],
    )
    freeze = freeze_fixture(plan)
    with pytest.raises(ContractError):
        build_score_artifact(
            plan, locator_fixture(plan), oracle_fixture(plan), invalid_raw, freeze,
            accepted_provenance_by_source_item=provenance,
            ledger_events=[], expected_bindings=freeze_expected(freeze),
            method_id=freeze["model_repository"],
        )
    assert scoring_calls == []


def test_public_cli_pair_failure_leaves_no_partial_artifact(tmp_path: Path) -> None:
    import subprocess

    plan = plan_fixture()
    items_path = tmp_path / "items.json"
    sources_path = tmp_path / "source-artifacts.json"
    provenance_path = tmp_path / "provenance.json"
    rgb_map_path = tmp_path / "rgb-map.json"
    _canonical_write(items_path, plan["items"])
    _canonical_write(sources_path, plan["source_artifacts"])
    _canonical_write(provenance_path, plan_provenance(plan))
    _canonical_write(rgb_map_path, {"unknown-image": "/missing.png"})
    plan_out = tmp_path / "plan.json"
    locator_out = tmp_path / "locators.json"
    script = Path(__file__).parents[1] / "scripts" / "run_rca_bench.py"
    result = subprocess.run(
        [sys.executable, str(script), "publish-public", "--items", str(items_path),
         "--source-artifacts", str(sources_path), "--provenance", str(provenance_path),
         "--rgb-map", str(rgb_map_path), "--benchmark-role", "development",
         "--development-partition", "selection", "--split-version", "split-v1",
         "--prompt-template-sha256", plan["prompt_template_sha256"],
         "--guard-config-sha256", plan["guard_config_sha256"],
         "--evidence-schema-sha256", plan["evidence_schema_sha256"],
         "--model-input-schema-sha256", plan["model_input_schema_sha256"],
         "--plan-out", str(plan_out), "--locators-out", str(locator_out)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" not in result.stderr
    assert not plan_out.exists() and not locator_out.exists()


def test_finalize_fatal_cli_dispatch_imports_only_reconstruction_capability(tmp_path: Path) -> None:
    import subprocess

    script = Path(__file__).parents[1] / "scripts" / "run_rca_bench.py"
    plan = tmp_path / "plan.json"
    bindings = tmp_path / "bindings.json"
    _canonical_write(plan, {"items": []})
    _canonical_write(bindings, {
        "plan_sha256": "a" * 64, "freeze_receipt_sha256": "b" * 64,
        "model_repository": "deterministic-template",
        "model_revision": "rca-bench-template-v1",
        "model_weight_sha256": "c" * 64, "generation_config_sha256": "d" * 64,
    })
    probe = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('rca_cli', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.main(sys.argv[2:])
except Exception:
    pass
print(json.dumps(sorted(name for name in sys.modules if name.startswith('fluidmix.rca_bench'))))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(script), "finalize-fatal",
         "--ledger", str(tmp_path / "missing.jsonl"), "--plan", str(plan),
         "--raw-bindings", str(bindings), "--fatal-event-sha256", "e" * 64,
         "--raw", str(tmp_path / "raw.json")],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == ["fluidmix.rca_bench_fatal"]


@pytest.mark.parametrize("tamper", ["valid", "source", "checker", "config"])
def test_full_freeze_cli_rehashes_sources_checkers_and_config(
    tmp_path: Path, tamper: str,
) -> None:
    """The real CLI receives paths, never a caller-produced expected-hash JSON file."""
    import subprocess

    from fluidmix.rca_bench import canonical_sha256
    from fluidmix.rca_bench_controller import (
        compute_actual_freeze_bindings,
        template_weight_binding_sha256,
    )

    sources, generation_path = _freeze_source_paths(tmp_path)
    plan, locators, provenance = _real_plan_locators(tmp_path, sources)
    oracle = oracle_fixture(plan)
    oracle["protocol_sha256"] = hashlib.sha256(Path(sources["protocol"]).read_bytes()).hexdigest()
    config = json.loads(generation_path.read_text())
    actual = compute_actual_freeze_bindings(
        protocol_path=sources["protocol"], builder_source_path=sources["builder_source"],
        scorer_source_path=sources["scorer_source"], test_paths=sources["tests"],
        evidence_schema_path=sources["evidence_schema"], prompt_path=sources["prompt"],
        guard_config_path=sources["guard_config"], partition_paths=sources["partitions"],
        git_commit=sources["git_commit"], generation_config=config,
    )
    freeze = freeze_fixture(plan)
    freeze.update(actual)
    freeze.update({
        "plan_sha256": canonical_sha256(plan), "locator_sha256": canonical_sha256(locators),
        "oracle_sha256": canonical_sha256(oracle), "model_repository": "deterministic-template",
        "model_revision": "rca-bench-template-v1",
        "model_weight_sha256": template_weight_binding_sha256(config),
    })
    paths = {}
    for name, value in (("receipt", freeze), ("plan", plan), ("locators", locators),
                        ("oracle", oracle), ("provenance", provenance)):
        paths[name] = tmp_path / f"{name}.json"
        _canonical_write(paths[name], value)
    if tamper == "source":
        Path(sources["builder_source"]).write_text("builder_source-v2\n")
    elif tamper == "checker":
        Path(sources["partitions"]["checker_code"]).write_text("checker_code-v2\n")
    elif tamper == "config":
        _canonical_write(
            generation_path,
            {"mode": "deterministic_template", "temperature": 1},
        )
    script = Path(__file__).parents[1] / "scripts" / "run_rca_bench.py"
    args = [
        sys.executable, str(script), "publish-freeze", "--receipt", str(paths["receipt"]),
        "--plan", str(paths["plan"]), "--locators", str(paths["locators"]),
        "--oracle", str(paths["oracle"]), "--provenance", str(paths["provenance"]),
        "--protocol", str(sources["protocol"]), "--builder-source", str(sources["builder_source"]),
        "--scorer-source", str(sources["scorer_source"]), "--test-source", str(sources["tests"][0]),
        "--evidence-schema", str(sources["evidence_schema"]), "--prompt", str(sources["prompt"]),
        "--guard-config", str(sources["guard_config"]), "--generation-config", str(generation_path),
        "--git-commit", str(sources["git_commit"]),
    ]
    for name, path in sources["partitions"].items():
        args += [f"--{name.replace('_', '-')}", str(path)]
    output = tmp_path / "freeze-out.json"
    args += ["--out", str(output)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert "expected-bindings" not in " ".join(args)
    if tamper == "valid":
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text()) == freeze
    else:
        assert result.returncode != 0
        assert not output.exists()

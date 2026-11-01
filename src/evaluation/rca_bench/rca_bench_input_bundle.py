"""Oracle-free, CPU-only construction of frozen RCA-bench public inputs.

Only already accepted Part-B records and their out-of-band prediction/history receipts
enter this module. It has no model loader and no hidden-oracle API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any

from fluidmix.core.contracts.evidence import validate_evidence_record_v1, validate_evidence_sequence_v1
from fluidmix.storage.sqlite.evidence_store import canonical_source_record_sha256
from .rca_bench import (
    assign_development_partitions,
    build_grounding_qualification,
    build_shuffled_evidence_ids_control,
    build_support_removed_control,
    build_unsupported_cause_probe,
    ContractError,
    FREEZE_SCHEMA,
    MODEL_INPUT_SCHEMA,
    canonical_bytes,
    canonical_sha256,
    validate_freeze_bindings,
    validate_image_locators,
    validate_model_input,
    validate_partition_leakage,
    validate_plan,
    write_canonical_json_exclusive,
)
from .rca_bench_controller import build_image_locators, build_public_plan


_SPEC_FIELDS = {
    "source_item_id", "target_evidence_ids", "reference_evidence_ids", "reference_roles",
}
FROZEN_PROTOCOL_SHA256 = "6a6465048e3839bdcce60c39686f6edf382dfff0b6d23493d0d9b4a4e4d3a4f3"


def _exact_mapping(value: object, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError(f"{role} fields are invalid")
    return value


def _ordered_ids(value: object, role: str) -> list[str]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ContractError(f"{role} must be an ordered text sequence")
    result = list(value)
    if len(result) not in {2, 3} or len(result) != len(set(result)):
        raise ContractError(f"{role} must contain two or three unique IDs")
    return result


def _source_receipt_hashes(provenance: Mapping[str, Any]) -> dict[str, str]:
    receipts = provenance.get("source_evidence_receipts")
    if not isinstance(receipts, list):
        raise ContractError("accepted provenance source receipts are missing")
    result: dict[str, str] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ContractError("accepted provenance source receipt is invalid")
        evidence_id = receipt.get("evidence_id")
        digest = receipt.get("source_evidence_record_sha256")
        if not isinstance(evidence_id, str) or not isinstance(digest, str):
            raise ContractError("accepted provenance source receipt identity is invalid")
        if evidence_id in result:
            raise ContractError("accepted provenance source receipt is duplicated")
        result[evidence_id] = digest
    return result


def _verify_source_record_receipts(
    records_by_evidence_id: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    receipt_hashes: Mapping[str, str],
) -> None:
    registry = provenance.get("registry_receipt")
    entries = registry.get("entries") if isinstance(registry, Mapping) else None
    if not isinstance(entries, list):
        raise ContractError("accepted provenance predicted-mask registry is missing")
    by_image: dict[str, Mapping[str, Any]] = {}
    registry_paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("image_id"), str):
            raise ContractError("accepted provenance registry entry is invalid")
        image_id = str(entry["image_id"])
        if image_id in by_image:
            raise ContractError("accepted provenance registry image is duplicated")
        by_image[image_id] = entry
        for field in ("rgb_path", "mask_path"):
            path = Path(str(entry.get(field)))
            if not path.is_absolute():
                raise ContractError("accepted provenance registry path is not absolute")
            registry_paths.append(path)
    try:
        artifact_root = Path(os.path.commonpath(registry_paths))
    except (TypeError, ValueError) as exc:
        raise ContractError("accepted provenance registry has no common artifact root") from exc
    if artifact_root in registry_paths:
        artifact_root = artifact_root.parent
    for evidence_id, record in records_by_evidence_id.items():
        entry = by_image.get(str(record["image_id"]))
        if entry is None:
            raise ContractError("accepted provenance registry lacks a source record image")
        locator = {
            "rgb_path": Path(str(entry["rgb_path"])).relative_to(artifact_root).as_posix(),
            "rgb_sha256": entry.get("rgb_sha256"),
            "mask_path": Path(str(entry["mask_path"])).relative_to(artifact_root).as_posix(),
            "mask_sha256": entry.get("mask_sha256"),
        }
        try:
            observed = canonical_source_record_sha256(record, locator)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"accepted source record locator is invalid: {exc}") from exc
        if receipt_hashes.get(evidence_id) != observed:
            raise ContractError("accepted source record hash differs from its receipt")


def _observation(
    record: Mapping[str, Any], *, source_role: str, slot_id: str,
    production_sequence_index: int, source_record_sha256: str,
    allowed_evidence_ids: set[str],
) -> dict[str, Any]:
    if type(production_sequence_index) is not int or production_sequence_index < 0:
        raise ContractError("production sequence index must be a non-negative integer")
    pairs = [
        copy.deepcopy(pair)
        for pair in record["pairwise_delta_e76"]
        if pair["other_evidence_id"] in allowed_evidence_ids
    ]
    pairs.sort(key=lambda pair: pair["other_evidence_id"])
    return {
        "slot_id": slot_id,
        "source_role": source_role,
        "source_run_id": record["run_id"],
        "source_case_id": record["case_id"],
        "source_vial_ordinal": record["vial_ordinal"],
        "source_block_recipe_id": record["block_recipe_id"],
        "source_block_position": record["block_position"],
        "source_recipe_id": record["recipe_id"],
        "source_previous_recipe_id": record["previous_recipe_id"],
        "source_split_role": record["split_role"],
        "source_split_version": record["split_version"],
        "source_time": {
            "production_sequence_index": production_sequence_index,
            "fill_frame_start": record["fill_frame_range"][0],
        },
        "source_stage_id": record["stage_id"],
        "source_tank_id": record["tank_id"],
        "source_stage_ordinal": record["stage_ordinal"],
        "source_clip_sha256": record["clip_sha256"],
        "source_evidence_record_sha256": source_record_sha256,
        "source_colour_regime_id": record["colour_regime_id"],
        "evidence_id": record["evidence_id"],
        "image_id": record["image_id"],
        "rgb_sha256": record["rgb_sha256"],
        "mask_sha256": record["mask_sha256"],
        "mask_source": record["mask_source"],
        "measurement_id": record["measurement_id"],
        "median_lab": copy.deepcopy(record["median_lab"]),
        "mad_lab": copy.deepcopy(record["mad_lab"]),
        "iqr_lab": copy.deepcopy(record["iqr_lab"]),
        "pairwise_delta_e76": pairs,
    }


def _deliveries(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in bundle["stages"]:
        for observation in stage["observations"]:
            common = {
                "image_id": observation["image_id"],
                "rgb_sha256": observation["rgb_sha256"],
                "source_run_id": observation["source_run_id"],
                "source_case_id": observation["source_case_id"],
                "source_vial_ordinal": observation["source_vial_ordinal"],
                "source_time": copy.deepcopy(observation["source_time"]),
            }
            row = {"slot_id": observation["slot_id"]}
            for prefix in ("expected", "delivered"):
                row.update({f"{prefix}_{key}": copy.deepcopy(value) for key, value in common.items()})
            rows.append(row)
    return rows


def project_primary_item(
    records: Sequence[Mapping[str, Any]],
    projection_spec: Mapping[str, Any],
    production_sequence_by_evidence_id: Mapping[str, int],
    accepted_provenance: Mapping[str, Any],
    *, item_id: str, development_partition: str | None, prompt_template: str,
) -> dict[str, Any]:
    """Project one target vial plus role-aligned earlier references into a public item."""
    spec = _exact_mapping(projection_spec, _SPEC_FIELDS, "projection specification")
    target_ids = _ordered_ids(spec["target_evidence_ids"], "target evidence IDs")
    reference_ids = _ordered_ids(spec["reference_evidence_ids"], "reference evidence IDs")
    roles = (
        list(spec["reference_roles"])
        if isinstance(spec["reference_roles"], Sequence)
        and not isinstance(spec["reference_roles"], (str, bytes))
        else []
    )
    if len(reference_ids) != len(target_ids) or len(roles) != len(target_ids):
        raise ContractError("projection requires one reference role/record per target stage")
    if any(role not in {"same_run_reference", "cross_batch_reference"} for role in roles):
        raise ContractError("reference role is invalid")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ContractError("accepted records must be a sequence")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in records:
        try:
            checked = validate_evidence_record_v1(raw, phase="decided", mode="production")
        except (TypeError, ValueError) as exc:
            raise ContractError(f"accepted evidence record is invalid: {exc}") from exc
        evidence_id = str(checked["evidence_id"])
        if evidence_id in by_id:
            raise ContractError("accepted evidence record ID is duplicated")
        by_id[evidence_id] = checked
    wanted = set(target_ids + reference_ids)
    if set(by_id) != wanted:
        raise ContractError("accepted record set must exactly cover target/reference evidence IDs")
    targets = [by_id[evidence_id] for evidence_id in target_ids]
    try:
        targets = validate_evidence_sequence_v1(targets, phase="decided", mode="production")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"target evidence sequence is invalid: {exc}") from exc
    references = [by_id[evidence_id] for evidence_id in reference_ids]
    source_hashes = _source_receipt_hashes(accepted_provenance)
    if set(source_hashes) != wanted:
        raise ContractError("accepted source receipts must exactly cover projected evidence")
    _verify_source_record_receipts(by_id, accepted_provenance, source_hashes)
    if set(production_sequence_by_evidence_id) != wanted:
        raise ContractError("production sequence map must exactly cover projected evidence")

    stages: list[dict[str, Any]] = []
    for target, reference, role in zip(targets, references, roles, strict=True):
        ordinal = int(target["stage_ordinal"])
        target_observation = _observation(
            target, source_role="target", slot_id=f"stage-{ordinal}-target",
            production_sequence_index=production_sequence_by_evidence_id[target["evidence_id"]],
            source_record_sha256=source_hashes[target["evidence_id"]],
            allowed_evidence_ids=wanted,
        )
        reference_observation = _observation(
            reference, source_role=role, slot_id=f"stage-{ordinal}-reference",
            production_sequence_index=production_sequence_by_evidence_id[reference["evidence_id"]],
            source_record_sha256=source_hashes[reference["evidence_id"]],
            allowed_evidence_ids=wanted,
        )
        stages.append({
            "stage_id": target["stage_id"], "tank_id": target["tank_id"],
            "stage_ordinal": ordinal, "colour_regime_id": target["colour_regime_id"],
            "observations": [target_observation, reference_observation],
        })
    first = targets[0]
    bundle = {
        "schema_version": MODEL_INPUT_SCHEMA,
        "target_context": {
            "run_id": first["run_id"], "case_id": first["case_id"],
            "vial_ordinal": first["vial_ordinal"], "block_recipe_id": first["block_recipe_id"],
            "block_position": first["block_position"], "recipe_id": first["recipe_id"],
            "previous_recipe_id": first["previous_recipe_id"],
            "recipe_ml": copy.deepcopy(first["recipe_ml"]),
            "dispense_order": copy.deepcopy(first["dispense_order"]),
            "split_role": first["split_role"], "split_version": first["split_version"],
            "fill_frame_range": copy.deepcopy(first["fill_frame_range"]),
            "transition_frame_ranges": copy.deepcopy(first["transition_frame_ranges"]),
            "clip_sha256": first["clip_sha256"], "decision_id": first["decision_id"],
            "disposition": first["disposition"],
            "limit_codes": copy.deepcopy(first["limit_codes"]),
            "evidence_ids": [row["evidence_id"] for stage in stages for row in stage["observations"]],
            "history_refs": copy.deepcopy(first["history_refs"]),
        },
        "stages": stages,
    }
    checked_bundle = validate_model_input(bundle, accepted_provenance=accepted_provenance)
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ContractError("prompt template must be nonempty text")
    prompt = prompt_template + "\nEvidence JSON:\n" + canonical_bytes(checked_bundle).decode("utf-8")
    return {
        "item_id": item_id, "source_item_id": spec["source_item_id"], "role": "primary",
        "execution_policy": "model_attempt", "paired_primary_item_id": None,
        "run_id": first["run_id"], "case_id": first["case_id"],
        "vial_ordinal": first["vial_ordinal"], "split_version": first["split_version"],
        "development_partition": development_partition,
        "evidence_bundle": checked_bundle,
        "evidence_bundle_sha256": canonical_sha256(checked_bundle),
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_deliveries": _deliveries(checked_bundle), "control_receipt": None,
    }


def build_model_freeze_receipt(
    plan: Mapping[str, Any], locators: Mapping[str, Any], *,
    model_repository: str, model_revision: str, model_weight_sha256: str,
    oracle_sha256: str, split_ids: Sequence[str], test_hashes: Sequence[str],
    git_commit: str, hashes: Mapping[str, str],
    accepted_provenance_by_source_item: Mapping[str, object],
) -> dict[str, Any]:
    """Build one per-model digest-only freeze without opening hidden oracle bytes."""
    required = {
        "protocol", "builder_source", "scorer_source", "evidence_schema", "prompt",
        "guard_config", "development_partition_receipt", "qualification_access_log",
        "generation_config", "oracle", "split_spec", "recipes", "transitions",
        "real_reserved", "manifests", "checker_code",
    }
    if set(hashes) != required or hashes["oracle"] != oracle_sha256:
        raise ContractError("freeze hashes must exactly bind every frozen input and oracle digest")
    if hashes["protocol"] != FROZEN_PROTOCOL_SHA256:
        raise ContractError("freeze protocol hash differs from frozen RCA-bench v1")
    checked_plan = validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )
    checked_locators = validate_image_locators(locators)
    if checked_locators["plan_sha256"] != canonical_sha256(checked_plan):
        raise ContractError("freeze locators target another plan")
    primaries = [item for item in checked_plan["items"] if item["role"] == "primary"]
    required_abstention = sum(
        item["evidence_bundle"]["target_context"]["disposition"] == "insufficient_evidence"
        for item in primaries
    )
    receipt = {
        "schema_version": FREEZE_SCHEMA,
        "protocol_sha256": hashes["protocol"],
        "builder_source_sha256": hashes["builder_source"],
        "scorer_source_sha256": hashes["scorer_source"],
        "test_hashes": list(test_hashes),
        "evidence_schema_sha256": hashes["evidence_schema"],
        "prompt_sha256": hashes["prompt"],
        "guard_config_sha256": hashes["guard_config"],
        "plan_sha256": canonical_sha256(checked_plan),
        "oracle_sha256": oracle_sha256,
        "locator_sha256": canonical_sha256(checked_locators),
        "split_ids": list(split_ids), "source_artifacts": copy.deepcopy(checked_plan["source_artifacts"]),
        "model_repository": model_repository, "model_revision": model_revision,
        "model_weight_sha256": model_weight_sha256,
        "development_partition_receipt_sha256": hashes["development_partition_receipt"],
        "qualification_access_log_sha256": hashes["qualification_access_log"],
        "generation_config_sha256": hashes["generation_config"],
        "planned_item_count": checked_plan["planned_item_count"],
        "model_attempt_planned_count": checked_plan["model_attempt_planned_count"],
        "pre_inference_validation_planned_count": checked_plan["pre_inference_validation_planned_count"],
        "required_abstention_primary_count": required_abstention,
        "non_required_abstention_primary_count": len(primaries) - required_abstention,
        "status_count_equation": (
            "completed+inference_error+pre_inference_rejected+pre_inference_error+"
            "not_run_after_fatal=planned_item_count"
        ),
        "fatal_outcome": "no_rerun_incomplete_fails_adoption",
        "partition_checker_bindings": {
            name: hashes[name]
            for name in ("split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code")
        },
        "git_commit": git_commit,
    }
    expected = {
        "protocol_sha256": hashes["protocol"],
        "builder_source_sha256": hashes["builder_source"],
        "scorer_source_sha256": hashes["scorer_source"],
        "test_hashes": list(test_hashes),
        "evidence_schema_sha256": hashes["evidence_schema"],
        "prompt_sha256": hashes["prompt"],
        "guard_config_sha256": hashes["guard_config"], "git_commit": git_commit,
    }
    return validate_freeze_bindings(
        receipt, checked_plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected,
    )


def _lab_by_slot(item: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for stage in item["evidence_bundle"]["stages"]:
        for observation in stage["observations"]:
            values = observation["median_lab"]
            result[str(observation["slot_id"])] = tuple(float(value) for value in values)
    return result


def _pair_delta_map(items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    import math

    labs = {str(item["source_item_id"]): _lab_by_slot(item) for item in items}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for target in items:
        target_id = str(target["source_item_id"])
        result[target_id] = {}
        for donor in items:
            donor_id = str(donor["source_item_id"])
            if donor_id == target_id:
                continue
            if set(labs[target_id]) != set(labs[donor_id]):
                continue
            result[target_id][donor_id] = {
                slot: math.sqrt(sum((left - right) ** 2 for left, right in zip(labs[target_id][slot], labs[donor_id][slot], strict=True)))
                for slot in labs[target_id]
            }
    return result


def build_development_partition_artifacts(
    primaries: Sequence[Mapping[str, Any]],
    accepted_provenance_by_source_item: Mapping[str, object],
) -> dict[str, Any]:
    """Freeze complete development runs and construct the exact 12 qualification pairs."""
    if isinstance(primaries, (str, bytes)) or not isinstance(primaries, Sequence):
        raise ContractError("development primaries must be a sequence")
    rows = [copy.deepcopy(dict(item)) for item in primaries]
    _require_development_item_split_roles(rows)
    by_run: dict[str, list[dict[str, Any]]] = {}
    source_ids: list[str] = []
    image_partitions: dict[str, set[str]] = {}
    evidence_partitions: dict[str, set[str]] = {}
    for item in rows:
        if item.get("role") != "primary" or item.get("execution_policy") != "model_attempt":
            raise ContractError("development inventory accepts only primary model attempts")
        source_id = str(item.get("source_item_id"))
        source_ids.append(source_id)
        if source_id not in accepted_provenance_by_source_item:
            raise ContractError("development primary lacks accepted provenance")
        by_run.setdefault(str(item.get("run_id")), []).append(item)
    if len(source_ids) != len(set(source_ids)):
        raise ContractError("eligible development source IDs must be unique")
    if any(len(run_items) != 3 or {item.get("vial_ordinal") for item in run_items} != {1, 2, 3} for run_items in by_run.values()):
        raise ContractError("every eligible development run must be complete with three vials")
    if set(accepted_provenance_by_source_item) != {str(item["source_item_id"]) for item in rows}:
        raise ContractError("development provenance must exactly cover eligible primaries")
    partition = assign_development_partitions(sorted(by_run))
    qualification_runs = set(partition["grounding_qualification_run_ids"])
    for item in rows:
        frozen_partition = (
            "grounding_qualification"
            if item["run_id"] in qualification_runs
            else "selection"
        )
        for stage in item.get("evidence_bundle", {}).get("stages", []):
            for observation in stage.get("observations", []):
                image_partitions.setdefault(
                    str(observation.get("image_id")), set()
                ).add(frozen_partition)
                evidence_partitions.setdefault(
                    str(observation.get("evidence_id")), set()
                ).add(frozen_partition)
    if any(len(parts) != 1 for parts in image_partitions.values()) or any(
        len(parts) != 1 for parts in evidence_partitions.values()
    ):
        raise ContractError(
            "eligible development images/evidence cannot cross frozen run partitions"
        )
    qualification_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for item in rows:
        destination = qualification_rows if item["run_id"] in qualification_runs else selection_rows
        item["development_partition"] = (
            "grounding_qualification" if item["run_id"] in qualification_runs else "selection"
        )
        destination.append(item)
    qualification_rows.sort(key=lambda item: (str(item["run_id"]), int(item["vial_ordinal"])))
    selection_rows.sort(key=lambda item: (str(item["run_id"]), int(item["vial_ordinal"])))
    qualification = build_grounding_qualification(
        qualification_rows, _pair_delta_map(qualification_rows)
    )
    if qualification["pair_count"] != 12 or qualification["model_attempt_count"] != 24:
        raise ContractError("qualification denominator must remain exactly 12 pairs/24 attempts")
    access_body = {
        "schema_version": "rca-bench-qualification-access-log-v1",
        "access_state": "sealed_before_model_use",
        "qualification_sha256": qualification["qualification_sha256"],
        "pair_receipt_sha256": [pair["pair_receipt_sha256"] for pair in qualification["pairs"]],
        "pair_count": 12,
    }
    access_log = {**access_body, "access_log_sha256": canonical_sha256(access_body)}
    inventory_body = {
        "schema_version": "rca-bench-eligible-run-inventory-v1",
        "eligible_run_ids": partition["eligible_run_ids"],
        "sealed_inventory_sha256": partition["sealed_inventory_sha256"],
        "primary_source_item_ids": sorted(str(item["source_item_id"]) for item in rows),
    }
    inventory = {**inventory_body, "inventory_receipt_sha256": canonical_sha256(inventory_body)}
    return {
        "partition_receipt": partition,
        "eligible_inventory": inventory,
        "selection_primaries": selection_rows,
        "qualification": qualification,
        "qualification_plan_items": qualification["items"],
        "qualification_access_log": access_log,
    }


def _control_item_id(role: str, source_item_id: str) -> str:
    digest = hashlib.sha256(f"{role}|{source_item_id}".encode("utf-8")).hexdigest()[:24]
    return f"control-{role}-{digest}"


def _trusted_support_removed_control(
    primary: Mapping[str, Any], *, item_id: str, accepted_provenance: object
) -> dict[str, Any]:
    """Remove every explicit pairwise support link while freezing query/image evidence."""
    bundle = copy.deepcopy(primary["evidence_bundle"])
    removed_links: list[dict[str, str]] = []
    for stage in bundle["stages"]:
        for observation in stage["observations"]:
            for pair in observation["pairwise_delta_e76"]:
                removed_links.append({
                    "stage_id": str(stage["stage_id"]),
                    "evidence_id": str(observation["evidence_id"]),
                    "other_evidence_id": str(pair["other_evidence_id"]),
                })
            observation["pairwise_delta_e76"] = []
    if not removed_links or bundle["stages"] == primary["evidence_bundle"]["stages"]:
        raise ContractError("support removal did not change cited stage evidence")
    if any(
        observation["pairwise_delta_e76"]
        for stage in bundle["stages"] for observation in stage["observations"]
    ):
        raise ContractError("support removal left cited stage evidence")
    context = bundle["target_context"]
    context["disposition"] = "insufficient_evidence"
    context["limit_codes"] = sorted(set(context["limit_codes"]) | {"support_removed"})
    result = build_support_removed_control(
        primary, bundle, item_id=item_id, accepted_provenance=accepted_provenance
    )
    details = {
        "limit_code": "support_removed",
        "removed_support_links": removed_links,
        "removed_pair_count": len(removed_links),
    }
    receipt = {
        "transformation_id": "rca-bench-v1-support-removed",
        "source_item_sha256": canonical_sha256(primary),
        "details": details,
    }
    receipt["transformation_sha256"] = canonical_sha256(receipt)
    result["control_receipt"] = receipt
    return result


def build_selection_controls(
    selection_primaries: Sequence[Mapping[str, Any]],
    accepted_provenance_by_source_item: Mapping[str, object],
    *, support_removed_bundles_by_source_item: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Add one deterministic, non-vacuous control of every frozen selection type."""
    if support_removed_bundles_by_source_item is not None:
        raise ContractError("caller-supplied support removal is forbidden; use the trusted builder")
    rows = [copy.deepcopy(dict(item)) for item in selection_primaries]
    if not rows or any(item.get("development_partition") != "selection" for item in rows):
        raise ContractError("selection controls require frozen selection primaries")
    eligible_support = [
        item for item in rows
        if item["evidence_bundle"]["target_context"]["disposition"] != "insufficient_evidence"
        and any(
            target["pairwise_delta_e76"] or reference["pairwise_delta_e76"]
            for target, reference in (
                stage["observations"] for stage in item["evidence_bundle"]["stages"]
            )
        )
    ]
    if not eligible_support:
        raise ContractError("support-removed control has zero independently supported eligible items")
    support_primary = min(
        eligible_support,
        key=lambda item: hashlib.sha256(
            f"rca-bench-v1-20261028|support_removed|{item['source_item_id']}".encode()
        ).hexdigest(),
    )
    source_id = str(support_primary["source_item_id"])
    proof = accepted_provenance_by_source_item.get(source_id)
    support_control = _trusted_support_removed_control(
        support_primary, item_id=_control_item_id("support", source_id),
        accepted_provenance=proof,
    )
    id_primary = min(
        rows,
        key=lambda item: hashlib.sha256(
            f"rca-bench-v1-20261028|shuffled_evidence_ids|{item['source_item_id']}".encode()
        ).hexdigest(),
    )
    id_control = build_shuffled_evidence_ids_control(
        id_primary, item_id=_control_item_id("ids", str(id_primary["source_item_id"]))
    )
    cause_primary = min(
        rows,
        key=lambda item: hashlib.sha256(
            f"rca-bench-v1-20261028|unsupported_cause_probe|{item['source_item_id']}".encode()
        ).hexdigest(),
    )
    cause_control = build_unsupported_cause_probe(
        cause_primary, item_id=_control_item_id("cause", str(cause_primary["source_item_id"]))
    )
    return [*rows, support_control, id_control, cause_control]


_PLAN_METADATA_FIELDS = {
    "split_version", "prompt_template_sha256", "guard_config_sha256",
    "evidence_schema_sha256", "model_input_schema_sha256", "source_artifacts",
}


def _source_artifacts(
    metadata: Mapping[str, Any], provenance: Mapping[str, object]
) -> list[dict[str, str]]:
    base = metadata["source_artifacts"]
    if not isinstance(base, Sequence) or isinstance(base, (str, bytes)):
        raise ContractError("source artifacts must be a sequence")
    rows = [copy.deepcopy(dict(row)) for row in base]
    for index, source_id in enumerate(sorted(provenance), start=1):
        proof = provenance[source_id]
        if not isinstance(proof, Mapping) or not isinstance(proof.get("receipt_sha256"), str):
            raise ContractError("accepted provenance receipt is invalid")
        rows.append({
            "artifact_id": f"accepted-provenance-{index:04d}",
            "sha256": str(proof["receipt_sha256"]),
        })
    return rows


def build_test_public_plan(
    primaries: Sequence[Mapping[str, Any]],
    accepted_provenance_by_source_item: Mapping[str, object],
    *, metadata: Mapping[str, Any], wp1_source_provenance: Mapping[str, Any],
    partition_leakage_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a WP1-only primary plan and require both frozen abstention classes."""
    meta = _exact_mapping(metadata, _PLAN_METADATA_FIELDS, "plan metadata")
    rows = [copy.deepcopy(dict(item)) for item in primaries]
    if not rows or any(item.get("role") != "primary" for item in rows):
        raise ContractError("test plans contain primaries only; controls are forbidden")
    for item in rows:
        item["development_partition"] = None
        context = item.get("evidence_bundle", {}).get("target_context", {})
        if context.get("split_role") != "test" or any(
            observation.get("source_split_role") != "test"
            for stage in item.get("evidence_bundle", {}).get("stages", [])
            for observation in stage.get("observations", [])
        ):
            raise ContractError("WP1 test items and references must remain in the frozen test split")
    required = sum(
        item["evidence_bundle"]["target_context"]["disposition"] == "insufficient_evidence"
        for item in rows
    )
    if required < 1 or len(rows) - required < 1:
        raise ContractError("test plan requires both abstention denominator classes")
    wp1_fields = {
        "schema_version", "acceptance_status", "aggregate_acceptance_sha256",
        "source_receipt_sha256_by_source_item", "test_run_ids", "test_recipe_ids",
        "receipt_sha256",
    }
    wp1 = _exact_mapping(wp1_source_provenance, wp1_fields, "WP1 source provenance")
    unsigned_wp1 = {key: wp1[key] for key in wp1_fields - {"receipt_sha256"}}
    proof_hashes = {
        source_id: proof.get("receipt_sha256") if isinstance(proof, Mapping) else None
        for source_id, proof in accepted_provenance_by_source_item.items()
    }
    run_ids = sorted({str(item["run_id"]) for item in rows})
    recipe_ids = sorted({
        str(item["evidence_bundle"]["target_context"]["recipe_id"]) for item in rows
    })
    if (
        wp1["schema_version"] != "rca-bench-wp1-source-provenance-v1"
        or wp1["acceptance_status"] != "accepted"
        or canonical_sha256(unsigned_wp1) != wp1["receipt_sha256"]
        or wp1["source_receipt_sha256_by_source_item"] != proof_hashes
        or wp1["test_run_ids"] != run_ids
        or wp1["test_recipe_ids"] != recipe_ids
    ):
        raise ContractError("accepted WP1 aggregate/source provenance binding is invalid")
    aggregate_sha = wp1["aggregate_acceptance_sha256"]
    if not isinstance(aggregate_sha, str) or len(aggregate_sha) != 64:
        raise ContractError("accepted WP1 aggregate hash is invalid")
    audit_receipt = validate_partition_leakage(partition_leakage_audit)
    if (
        audit_receipt["benchmark_role"] != "test"
        or sorted(partition_leakage_audit["test_run_ids"]) != run_ids
        or sorted(
            row["recipe_id"] for row in partition_leakage_audit["recipes"]
            if row["role"] == "test"
        ) != recipe_ids
    ):
        raise ContractError("test plan differs from its frozen partition-leakage audit")
    source_meta = dict(meta)
    source_meta["source_artifacts"] = [
        *copy.deepcopy(list(meta["source_artifacts"])),
        {"artifact_id": "wp1-aggregate-acceptance", "sha256": aggregate_sha},
        {"artifact_id": "wp1-source-provenance", "sha256": wp1["receipt_sha256"]},
        {"artifact_id": "test-partition-leakage-audit", "sha256": canonical_sha256(audit_receipt)},
    ]
    plan = build_public_plan(
        rows, benchmark_role="test", development_partition=None,
        split_version=str(meta["split_version"]),
        prompt_template_sha256=str(meta["prompt_template_sha256"]),
        guard_config_sha256=str(meta["guard_config_sha256"]),
        evidence_schema_sha256=str(meta["evidence_schema_sha256"]),
        model_input_schema_sha256=str(meta["model_input_schema_sha256"]),
        source_artifacts=_source_artifacts(source_meta, accepted_provenance_by_source_item),
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    return validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )


def _plan_frozen_identities(plan: Mapping[str, Any]) -> dict[str, set[str]]:
    result = {
        name: set()
        for name in (
            "run", "recipe", "evidence", "source_record", "image", "rgb_hash",
            "mask_hash",
        )
    }
    for item in plan["items"]:
        result["run"].add(str(item["run_id"]))
        context = item["evidence_bundle"]["target_context"]
        result["recipe"].add(str(context["recipe_id"]))
        if context["previous_recipe_id"] is not None:
            result["recipe"].add(str(context["previous_recipe_id"]))
        for stage in item["evidence_bundle"]["stages"]:
            for observation in stage["observations"]:
                result["run"].add(str(observation["source_run_id"]))
                result["recipe"].add(str(observation["source_recipe_id"]))
                if observation["source_previous_recipe_id"] is not None:
                    result["recipe"].add(str(observation["source_previous_recipe_id"]))
                result["evidence"].add(str(observation["evidence_id"]))
                result["source_record"].add(
                    str(observation["source_evidence_record_sha256"])
                )
                result["image"].add(str(observation["image_id"]))
                result["rgb_hash"].add(str(observation["rgb_sha256"]))
                result["mask_hash"].add(str(observation["mask_sha256"]))
    return result


def _require_development_item_split_roles(items: Sequence[Mapping[str, Any]]) -> None:
    """Keep every development target/reference in one non-test frozen split."""
    for item in items:
        bundle = item.get("evidence_bundle")
        context = bundle.get("target_context") if isinstance(bundle, Mapping) else None
        stages = bundle.get("stages") if isinstance(bundle, Mapping) else None
        if not isinstance(context, Mapping) or not isinstance(stages, list):
            raise ContractError("development item lacks a valid evidence bundle")
        target_role = context.get("split_role")
        observation_roles = [
            observation.get("source_split_role")
            for stage in stages if isinstance(stage, Mapping)
            for observation in stage.get("observations", [])
            if isinstance(observation, Mapping)
        ]
        if (
            target_role == "test"
            or not observation_roles
            or any(role != target_role for role in observation_roles)
        ):
            raise ContractError(
                "development evidence must remain in one non-test frozen split"
            )


def _require_plan_recipe_split_roles(
    plans: Sequence[Mapping[str, Any]], partition_leakage_audit: Mapping[str, Any],
    *, benchmark_role: str,
) -> None:
    """Bind each plan's actual Part-B split role to its audited recipe role."""
    recipe_roles = {
        str(row["recipe_id"]): str(row["role"])
        for row in partition_leakage_audit["recipes"]
    }
    for plan in plans:
        for item in plan["items"]:
            context = item["evidence_bundle"]["target_context"]
            expected_role = recipe_roles.get(str(context["recipe_id"]))
            observed_roles = {
                str(context["split_role"]),
                *(
                    str(observation["source_split_role"])
                    for stage in item["evidence_bundle"]["stages"]
                    for observation in stage["observations"]
                ),
            }
            if (
                expected_role is None
                or observed_roles != {expected_role}
                or (benchmark_role == "development" and expected_role == "test")
                or (benchmark_role == "test" and expected_role != "test")
            ):
                raise ContractError(
                    f"{benchmark_role} evidence split roles differ from the partition audit"
                )


def _require_development_plan_denominators(
    selection: Mapping[str, Any], qualification: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> None:
    """Enforce the frozen selection controls and 12 paired qualification attempts."""
    selection_runs = set(partition["selection_run_ids"])
    selection_primaries = [
        item for item in selection["items"] if item["role"] == "primary"
    ]
    expected_primary_count = 3 * len(selection_runs)
    selection_roles = Counter(str(item["role"]) for item in selection["items"])
    expected_selection_roles = Counter({
        "primary": expected_primary_count,
        "support_removed": 1,
        "shuffled_evidence_ids": 1,
        "unsupported_cause_probe": 1,
    })
    primary_counts_by_run = Counter(
        str(item["run_id"]) for item in selection_primaries
    )
    support_items = [
        item for item in selection["items"] if item["role"] == "support_removed"
    ]
    if (
        len(selection_runs) < 2
        or selection_roles != expected_selection_roles
        or set(primary_counts_by_run) != selection_runs
        or any(count != 3 for count in primary_counts_by_run.values())
        or selection["planned_item_count"] != expected_primary_count + 3
        or selection["model_attempt_planned_count"] != expected_primary_count + 2
        or selection["pre_inference_validation_planned_count"] != 1
        or any(
            observation["pairwise_delta_e76"]
            for item in support_items
            for stage in item["evidence_bundle"]["stages"]
            for observation in stage["observations"]
        )
    ):
        raise ContractError("selection control denominator is invalid")

    qualification_roles = Counter(
        str(item["role"]) for item in qualification["items"]
    )
    qualification_primaries = {
        str(item["item_id"]): item
        for item in qualification["items"] if item["role"] == "primary"
    }
    shuffled = [
        item for item in qualification["items"] if item["role"] == "shuffled_images"
    ]
    paired_primary_ids = [str(item["paired_primary_item_id"]) for item in shuffled]
    qualification_primary_runs = Counter(
        str(item["run_id"]) for item in qualification_primaries.values()
    )
    qualification_runs = set(partition["grounding_qualification_run_ids"])
    if (
        qualification_roles != Counter({"primary": 12, "shuffled_images": 12})
        or len(qualification_primaries) != 12
        or len({str(item["source_item_id"]) for item in qualification_primaries.values()}) != 12
        or len(set(paired_primary_ids)) != 12
        or set(paired_primary_ids) != set(qualification_primaries)
        or set(qualification_primary_runs) != qualification_runs
        or any(count != 3 for count in qualification_primary_runs.values())
        or qualification["planned_item_count"] != 24
        or qualification["model_attempt_planned_count"] != 24
        or qualification["pre_inference_validation_planned_count"] != 0
    ):
        raise ContractError("qualification pair denominator is invalid")


def _require_disjoint_frozen_identities(
    left: Mapping[str, set[str]], right: Mapping[str, set[str]], *, label: str
) -> None:
    for identity_class in left:
        overlap = left[identity_class] & right[identity_class]
        if overlap:
            raise ContractError(
                f"{label} cross-role duplication in frozen {identity_class} identities"
            )


def validate_cross_role_isolation(
    development_plans: Sequence[Mapping[str, Any]], test_plan: Mapping[str, Any], *,
    development_provenance_by_partition: Mapping[str, Mapping[str, object]] | None = None,
    test_provenance: Mapping[str, object] | None = None,
    development_partition_receipt: Mapping[str, Any] | None = None,
    development_publication_receipt: Mapping[str, Any] | None = None,
    development_partition_leakage_audit: Mapping[str, Any] | None = None,
    test_partition_leakage_audit: Mapping[str, Any] | None = None,
    wp1_source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate published plans/provenance/audits, then reject every frozen overlap."""
    if (
        not isinstance(development_plans, Sequence)
        or isinstance(development_plans, (str, bytes))
        or len(development_plans) != 2
        or development_provenance_by_partition is None
        or test_provenance is None
        or development_partition_receipt is None
        or development_publication_receipt is None
        or development_partition_leakage_audit is None
        or test_partition_leakage_audit is None
        or wp1_source_provenance is None
    ):
        raise ContractError("complete development publication, plans, and provenance are required")
    plans_by_partition: dict[str, dict[str, Any]] = {}
    for raw_plan in development_plans:
        partition = raw_plan.get("development_partition")
        if partition not in {"selection", "grounding_qualification"}:
            raise ContractError("development publication has an invalid plan partition")
        proof = development_provenance_by_partition.get(str(partition))
        if not isinstance(proof, Mapping):
            raise ContractError("development plan lacks accepted provenance")
        checked = validate_plan(raw_plan, accepted_provenance_by_source_item=proof)
        if checked["benchmark_role"] != "development" or not checked["items"]:
            raise ContractError("development plan is empty or has the wrong role")
        _require_development_item_split_roles(checked["items"])
        plans_by_partition[str(partition)] = checked
    if set(plans_by_partition) != {"selection", "grounding_qualification"}:
        raise ContractError("selection and qualification plans are both required")
    checked_test = validate_plan(
        test_plan, accepted_provenance_by_source_item=test_provenance
    )
    if checked_test["benchmark_role"] != "test" or not checked_test["items"]:
        raise ContractError("test plan is empty or has the wrong role")
    if any(item["role"] != "primary" for item in checked_test["items"]):
        raise ContractError("test plan contains controls")
    test_abstention = [
        item["evidence_bundle"]["target_context"]["disposition"] == "insufficient_evidence"
        for item in checked_test["items"]
    ]
    if not any(test_abstention) or all(test_abstention):
        raise ContractError("test plan lacks both abstention denominator classes")

    dev_audit_receipt = validate_partition_leakage(development_partition_leakage_audit)
    test_audit_receipt = validate_partition_leakage(test_partition_leakage_audit)
    if dev_audit_receipt["benchmark_role"] != "development" or test_audit_receipt["benchmark_role"] != "test":
        raise ContractError("partition-leakage audit roles are invalid")
    _require_plan_recipe_split_roles(
        list(plans_by_partition.values()), development_partition_leakage_audit,
        benchmark_role="development",
    )
    _require_plan_recipe_split_roles(
        [checked_test], test_partition_leakage_audit, benchmark_role="test",
    )
    partition_fields = {
        "schema_version", "protocol_id", "eligible_run_ids",
        "sealed_inventory_sha256", "grounding_qualification_run_ids",
        "selection_run_ids", "partition_receipt_sha256",
    }
    partition = _exact_mapping(
        development_partition_receipt, partition_fields, "development partition receipt"
    )
    if (
        partition["schema_version"] != "rca-bench-development-partition-v1"
        or partition["protocol_id"] != "rca-bench-v1-20261028"
        or len(partition["eligible_run_ids"]) < 6
        or len(partition["grounding_qualification_run_ids"]) != 4
        or partition["partition_receipt_sha256"] != canonical_sha256(
        {key: partition[key] for key in partition_fields - {"partition_receipt_sha256"}}
        )
    ):
        raise ContractError("development partition receipt hash mismatch")
    target_selection_runs = sorted({str(item["run_id"]) for item in plans_by_partition["selection"]["items"]})
    target_qualification_runs = sorted({str(item["run_id"]) for item in plans_by_partition["grounding_qualification"]["items"]})
    qualification = plans_by_partition["grounding_qualification"]
    _require_development_plan_denominators(
        plans_by_partition["selection"], qualification, partition,
    )
    development_recipe_ids = {
        str(item["evidence_bundle"]["target_context"]["recipe_id"])
        for plan in plans_by_partition.values() for item in plan["items"]
    }
    test_recipe_ids = {
        str(item["evidence_bundle"]["target_context"]["recipe_id"])
        for item in checked_test["items"]
    }
    audited_development_recipes = {
        str(row["recipe_id"])
        for row in test_partition_leakage_audit["recipes"]
        if row["role"] != "test"
    }
    audited_test_recipes = {
        str(row["recipe_id"])
        for row in test_partition_leakage_audit["recipes"]
        if row["role"] == "test"
    }
    if (
        target_selection_runs != sorted(partition["selection_run_ids"])
        or target_qualification_runs != sorted(partition["grounding_qualification_run_ids"])
        or sorted(development_partition_leakage_audit["selection_run_ids"]) != target_selection_runs
        or sorted(development_partition_leakage_audit["qualification_run_ids"]) != target_qualification_runs
        or sorted(test_partition_leakage_audit["selection_run_ids"]) != target_selection_runs
        or sorted(test_partition_leakage_audit["qualification_run_ids"]) != target_qualification_runs
        or sorted(development_partition_leakage_audit["development_run_ids"])
        != sorted(partition["eligible_run_ids"])
        or development_partition_leakage_audit["test_run_ids"] != []
        or sorted(test_partition_leakage_audit["development_run_ids"])
        != sorted(partition["eligible_run_ids"])
        or sorted(test_partition_leakage_audit["test_run_ids"])
        != sorted({str(item["run_id"]) for item in checked_test["items"]})
        or audited_development_recipes != development_recipe_ids
        or audited_test_recipes != test_recipe_ids
    ):
        raise ContractError("published development plans differ from partition receipts")

    publication_fields = {
        "schema_version", "selection_plan_sha256", "selection_provenance_sha256",
        "qualification_plan_sha256", "qualification_provenance_sha256",
        "development_partition_receipt_sha256", "partition_leakage_receipt_sha256",
        "receipt_sha256",
    }
    publication = _exact_mapping(
        development_publication_receipt, publication_fields,
        "development publication receipt",
    )
    expected_publication = {
        "schema_version": "rca-bench-development-publication-v1",
        "selection_plan_sha256": canonical_sha256(plans_by_partition["selection"]),
        "selection_provenance_sha256": canonical_sha256(
            development_provenance_by_partition["selection"]
        ),
        "qualification_plan_sha256": canonical_sha256(
            plans_by_partition["grounding_qualification"]
        ),
        "qualification_provenance_sha256": canonical_sha256(
            development_provenance_by_partition["grounding_qualification"]
        ),
        "development_partition_receipt_sha256": canonical_sha256(partition),
        "partition_leakage_receipt_sha256": canonical_sha256(dev_audit_receipt),
    }
    if (
        {key: publication[key] for key in publication_fields - {"receipt_sha256"}}
        != expected_publication
        or publication["receipt_sha256"] != canonical_sha256(expected_publication)
    ):
        raise ContractError("development publication receipt hashes do not bind supplied plans")

    wp1_fields = {
        "schema_version", "acceptance_status", "aggregate_acceptance_sha256",
        "source_receipt_sha256_by_source_item", "test_run_ids", "test_recipe_ids",
        "receipt_sha256",
    }
    wp1 = _exact_mapping(wp1_source_provenance, wp1_fields, "WP1 source provenance")
    if (
        wp1["schema_version"] != "rca-bench-wp1-source-provenance-v1"
        or wp1["acceptance_status"] != "accepted"
        or wp1["receipt_sha256"] != canonical_sha256(
            {key: wp1[key] for key in wp1_fields - {"receipt_sha256"}}
        )
        or wp1["source_receipt_sha256_by_source_item"]
        != {
            source_id: proof.get("receipt_sha256") if isinstance(proof, Mapping) else None
            for source_id, proof in test_provenance.items()
        }
        or wp1["test_run_ids"]
        != sorted({str(item["run_id"]) for item in checked_test["items"]})
        or wp1["test_recipe_ids"] != sorted(test_recipe_ids)
    ):
        raise ContractError("test plan lacks accepted WP1 aggregate/source provenance")

    selection_ids = _plan_frozen_identities(plans_by_partition["selection"])
    qualification_ids = _plan_frozen_identities(
        plans_by_partition["grounding_qualification"]
    )
    _require_disjoint_frozen_identities(
        {key: value for key, value in selection_ids.items() if key != "recipe"},
        {key: value for key, value in qualification_ids.items() if key != "recipe"},
        label="selection/qualification",
    )
    development_ids = {
        name: selection_ids[name] | qualification_ids[name] for name in selection_ids
    }
    _require_disjoint_frozen_identities(
        development_ids, _plan_frozen_identities(checked_test), label="development/test"
    )
    body = {
        "schema_version": "rca-bench-cross-role-isolation-v1",
        "development_publication_receipt_sha256": publication["receipt_sha256"],
        "development_plan_sha256": sorted(
            canonical_sha256(plan) for plan in plans_by_partition.values()
        ),
        "test_plan_sha256": canonical_sha256(checked_test),
        "test_provenance_sha256": canonical_sha256(test_provenance),
        "wp1_source_provenance_sha256": wp1["receipt_sha256"],
        "development_partition_leakage_receipt_sha256": canonical_sha256(dev_audit_receipt),
        "test_partition_leakage_receipt_sha256": canonical_sha256(test_audit_receipt),
        "overlap_count": 0,
        "status": "pass",
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_development_public_plans(
    partition_artifacts: Mapping[str, Any],
    accepted_provenance_by_source_item: Mapping[str, object],
    *, metadata: Mapping[str, Any], partition_leakage_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the separate selection and grounding-qualification public plans."""
    meta = _exact_mapping(metadata, _PLAN_METADATA_FIELDS, "plan metadata")
    selection_primaries = partition_artifacts.get("selection_primaries")
    qualification_items = partition_artifacts.get("qualification_plan_items")
    if not isinstance(selection_primaries, list) or not isinstance(qualification_items, list):
        raise ContractError("development partition artifacts are incomplete")
    selection_items = build_selection_controls(
        selection_primaries, accepted_provenance_by_source_item,
    )
    selection_sources = {str(item["source_item_id"]) for item in selection_items}
    qualification_sources = {str(item["source_item_id"]) for item in qualification_items}
    selection_provenance = {
        key: accepted_provenance_by_source_item[key] for key in sorted(selection_sources)
    }
    qualification_provenance = {
        key: accepted_provenance_by_source_item[key] for key in sorted(qualification_sources)
    }
    common = {
        "split_version": str(meta["split_version"]),
        "prompt_template_sha256": str(meta["prompt_template_sha256"]),
        "guard_config_sha256": str(meta["guard_config_sha256"]),
        "evidence_schema_sha256": str(meta["evidence_schema_sha256"]),
        "model_input_schema_sha256": str(meta["model_input_schema_sha256"]),
    }
    selection_plan = build_public_plan(
        selection_items, benchmark_role="development", development_partition="selection",
        source_artifacts=_source_artifacts(meta, selection_provenance),
        accepted_provenance_by_source_item=selection_provenance, **common,
    )
    qualification_plan = build_public_plan(
        qualification_items, benchmark_role="development",
        development_partition="grounding_qualification",
        source_artifacts=_source_artifacts(meta, qualification_provenance),
        accepted_provenance_by_source_item=qualification_provenance, **common,
    )
    if (
        qualification_plan["planned_item_count"] != 24
        or qualification_plan["model_attempt_planned_count"] != 24
        or qualification_plan["pre_inference_validation_planned_count"] != 0
    ):
        raise ContractError("qualification public plan must contain exactly 24 model attempts")
    selection_plan = validate_plan(
        selection_plan, accepted_provenance_by_source_item=selection_provenance
    )
    qualification_plan = validate_plan(
        qualification_plan, accepted_provenance_by_source_item=qualification_provenance
    )
    audit_receipt = validate_partition_leakage(partition_leakage_audit)
    _require_plan_recipe_split_roles(
        [selection_plan, qualification_plan], partition_leakage_audit,
        benchmark_role="development",
    )
    partition = partition_artifacts.get("partition_receipt")
    planned_recipe_ids = {
        str(item["evidence_bundle"]["target_context"]["recipe_id"])
        for plan in (selection_plan, qualification_plan) for item in plan["items"]
    }
    audited_recipe_ids = {
        str(row["recipe_id"]) for row in partition_leakage_audit["recipes"]
        if row["role"] != "test"
    }
    if not isinstance(partition, Mapping) or (
        audit_receipt["benchmark_role"] != "development"
        or sorted(partition_leakage_audit["development_run_ids"])
        != sorted(partition["eligible_run_ids"])
        or sorted(partition_leakage_audit["selection_run_ids"])
        != sorted(partition["selection_run_ids"])
        or sorted(partition_leakage_audit["qualification_run_ids"])
        != sorted(partition["grounding_qualification_run_ids"])
        or audited_recipe_ids != planned_recipe_ids
    ):
        raise ContractError("development plans differ from the frozen partition-leakage audit")
    _require_development_plan_denominators(
        selection_plan, qualification_plan, partition,
    )
    result = {
        "selection_plan": selection_plan,
        "selection_provenance": selection_provenance,
        "qualification_plan": qualification_plan,
        "qualification_provenance": qualification_provenance,
        "partition_leakage_audit": copy.deepcopy(dict(partition_leakage_audit)),
        "partition_leakage_receipt": audit_receipt,
    }
    publication_body = {
        "schema_version": "rca-bench-development-publication-v1",
        "selection_plan_sha256": canonical_sha256(selection_plan),
        "selection_provenance_sha256": canonical_sha256(selection_provenance),
        "qualification_plan_sha256": canonical_sha256(qualification_plan),
        "qualification_provenance_sha256": canonical_sha256(qualification_provenance),
        "development_partition_receipt_sha256": canonical_sha256(partition),
        "partition_leakage_receipt_sha256": canonical_sha256(audit_receipt),
    }
    result["publication_receipt"] = {
        **publication_body, "receipt_sha256": canonical_sha256(publication_body)
    }
    return result


def build_verified_image_locators(
    plan: Mapping[str, Any], accepted_provenance_by_source_item: Mapping[str, object]
) -> dict[str, Any]:
    """Derive locator paths only from accepted registries and verify exact RGB bytes."""
    paths: dict[str, str] = {}
    for proof in accepted_provenance_by_source_item.values():
        if not isinstance(proof, Mapping):
            raise ContractError("accepted provenance is invalid")
        registry = proof.get("registry_receipt")
        if not isinstance(registry, Mapping) or not isinstance(registry.get("entries"), list):
            raise ContractError("accepted predicted-mask registry is missing")
        for entry in registry["entries"]:
            if not isinstance(entry, Mapping):
                raise ContractError("accepted predicted-mask registry entry is invalid")
            image_id = str(entry.get("image_id"))
            path = str(entry.get("rgb_path"))
            previous = paths.setdefault(image_id, path)
            if previous != path:
                raise ContractError("accepted RGB image ID has conflicting locator paths")
    delivered = {
        str(delivery["delivered_image_id"])
        for item in plan["items"] for delivery in item["image_deliveries"]
    }
    if set(paths).intersection(delivered) != delivered:
        raise ContractError("accepted registries do not cover every delivered RGB image")
    return build_image_locators(
        plan, rgb_paths={image_id: paths[image_id] for image_id in delivered}, verify_files=True
    )


def validate_development_artifact_bindings(
    plans: Mapping[str, Any], partition_artifacts: Mapping[str, Any]
) -> None:
    """Require the published qualification plan and all pre-model seals to agree."""
    qualification = partition_artifacts.get("qualification")
    access = partition_artifacts.get("qualification_access_log")
    partition = partition_artifacts.get("partition_receipt")
    inventory = partition_artifacts.get("eligible_inventory")
    if not all(isinstance(value, Mapping) for value in (qualification, access, partition, inventory)):
        raise ContractError("development seal artifacts are incomplete")
    if (
        qualification.get("items") != plans["qualification_plan"].get("items")
        or qualification.get("run_ids") != sorted(partition.get("grounding_qualification_run_ids", []))
        or qualification.get("qualification_sha256") != access.get("qualification_sha256")
        or access.get("pair_count") != 12
        or access.get("pair_receipt_sha256")
        != [pair.get("pair_receipt_sha256") for pair in qualification.get("pairs", [])]
        or inventory.get("eligible_run_ids") != partition.get("eligible_run_ids")
        or inventory.get("sealed_inventory_sha256") != partition.get("sealed_inventory_sha256")
    ):
        raise ContractError("development qualification plan/seal/inventory binding mismatch")


def _rename_directory_durable(staging: Path, destination: Path) -> None:
    os.rename(staging, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_PUBLIC_FILENAMES = {
    "plan": "rca-bench-plan-v1.json",
    "locators": "rca-bench-image-locators-v1.json",
    "accepted_provenance": "rca-bench-accepted-provenance-v1.json",
    "eligible_inventory": "rca-bench-eligible-run-inventory-v1.json",
    "partition_receipt": "rca-bench-development-partition-v1.json",
    "qualification": "rca-bench-grounding-qualification-v1.json",
    "qualification_access_log": "rca-bench-qualification-access-log-v1.json",
}


def publish_pre_inference_bundle(
    artifacts: Mapping[str, object], *, output_dir: str | Path,
) -> dict[str, Any]:
    """Exclusively publish a complete public bundle and its hash inventory."""
    if set(artifacts) != set(_PUBLIC_FILENAMES):
        raise ContractError("pre-inference publication must contain every public artifact")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise ContractError("refusing to overwrite an existing pre-inference bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ContractError("pre-inference output parent must be a real directory")
    staging = destination.parent / f".{destination.name}.incomplete"
    if staging.exists() or staging.is_symlink():
        raise ContractError("pre-inference staging path is occupied")
    staging.mkdir(mode=0o700)
    try:
        inventory_rows = []
        for key, filename in sorted(_PUBLIC_FILENAMES.items()):
            value = artifacts[key]
            write_canonical_json_exclusive(staging / filename, value)
            inventory_rows.append({"artifact": key, "filename": filename, "sha256": canonical_sha256(value)})
        inventory_body = {
            "schema_version": "rca-bench-pre-inference-inventory-v1",
            "artifacts": inventory_rows,
        }
        inventory = {**inventory_body, "inventory_sha256": canonical_sha256(inventory_body)}
        write_canonical_json_exclusive(staging / "rca-bench-pre-inference-inventory-v1.json", inventory)
        _rename_directory_durable(staging, destination)
        return inventory
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def publish_plan_input_bundle(
    plan: Mapping[str, Any], locators: Mapping[str, Any],
    accepted_provenance_by_source_item: Mapping[str, object], *, output_dir: str | Path,
    extra_artifacts: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Publish the controller's three public inputs as one exclusive directory."""
    checked_plan = validate_plan(
        plan, locators=locators,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    checked_locators = validate_image_locators(locators, verify_files=True)
    values = {
        "rca-bench-plan-v1.json": checked_plan,
        "rca-bench-image-locators-v1.json": checked_locators,
        "rca-bench-accepted-provenance-v1.json": dict(accepted_provenance_by_source_item),
    }
    if extra_artifacts is not None:
        for filename, value in extra_artifacts.items():
            if (
                not isinstance(filename, str)
                or not filename.endswith(".json")
                or Path(filename).name != filename
                or filename in values
            ):
                raise ContractError("extra plan input artifact filename is invalid")
            values[filename] = value
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise ContractError("refusing to overwrite an existing plan input bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete"
    if staging.exists() or staging.is_symlink():
        raise ContractError("plan input staging path is occupied")
    staging.mkdir(mode=0o700)
    try:
        rows = []
        for filename, value in sorted(values.items()):
            write_canonical_json_exclusive(staging / filename, value)
            rows.append({"filename": filename, "sha256": canonical_sha256(value)})
        body = {"schema_version": "rca-bench-plan-input-inventory-v1", "artifacts": rows}
        inventory = {**body, "inventory_sha256": canonical_sha256(body)}
        write_canonical_json_exclusive(staging / "rca-bench-plan-input-inventory-v1.json", inventory)
        _rename_directory_durable(staging, destination)
        return inventory
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def publish_development_input_bundles(
    plans: Mapping[str, Any], partition_artifacts: Mapping[str, Any], *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish selection, qualification, and their pre-model seals as one unit."""
    required_plans = {
        "selection_plan", "selection_provenance",
        "qualification_plan", "qualification_provenance",
        "partition_leakage_audit", "partition_leakage_receipt",
        "publication_receipt",
    }
    if set(plans) != required_plans:
        raise ContractError("development plans are incomplete")
    validate_development_artifact_bindings(plans, partition_artifacts)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise ContractError("refusing to overwrite existing development input bundles")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete"
    if staging.exists() or staging.is_symlink():
        raise ContractError("development input staging path is occupied")
    staging.mkdir(mode=0o700)
    try:
        selection_locators = build_verified_image_locators(
            plans["selection_plan"], plans["selection_provenance"]
        )
        qualification_locators = build_verified_image_locators(
            plans["qualification_plan"], plans["qualification_provenance"]
        )
        selection_inventory = publish_plan_input_bundle(
            plans["selection_plan"], selection_locators, plans["selection_provenance"],
            output_dir=staging / "selection",
        )
        qualification_inventory = publish_plan_input_bundle(
            plans["qualification_plan"], qualification_locators,
            plans["qualification_provenance"], output_dir=staging / "grounding-qualification",
        )
        shared = {
            "rca-bench-development-partition-v1.json": partition_artifacts["partition_receipt"],
            "rca-bench-eligible-run-inventory-v1.json": partition_artifacts["eligible_inventory"],
            "rca-bench-grounding-qualification-v1.json": partition_artifacts["qualification"],
            "rca-bench-qualification-access-log-v1.json": partition_artifacts["qualification_access_log"],
            "rca-bench-development-partition-leakage-input-v1.json": plans["partition_leakage_audit"],
            "rca-bench-development-partition-leakage-audit-v1.json": plans["partition_leakage_receipt"],
            "rca-bench-development-publication-v1.json": plans["publication_receipt"],
        }
        for filename, value in shared.items():
            write_canonical_json_exclusive(staging / filename, value)
        body = {
            "schema_version": "rca-bench-development-input-inventory-v1",
            "selection_inventory_sha256": selection_inventory["inventory_sha256"],
            "qualification_inventory_sha256": qualification_inventory["inventory_sha256"],
            "shared_artifacts": [
                {"filename": filename, "sha256": canonical_sha256(value)}
                for filename, value in sorted(shared.items())
            ],
        }
        inventory = {**body, "inventory_sha256": canonical_sha256(body)}
        write_canonical_json_exclusive(staging / "rca-bench-development-input-inventory-v1.json", inventory)
        _rename_directory_durable(staging, destination)
        return inventory
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


__all__ = [
    "build_development_partition_artifacts",
    "build_development_public_plans",
    "build_model_freeze_receipt",
    "build_selection_controls",
    "build_test_public_plan",
    "build_verified_image_locators",
    "FROZEN_PROTOCOL_SHA256",
    "project_primary_item",
    "publish_development_input_bundles",
    "publish_plan_input_bundle",
    "publish_pre_inference_bundle",
    "validate_cross_role_isolation",
    "validate_development_artifact_bindings",
]

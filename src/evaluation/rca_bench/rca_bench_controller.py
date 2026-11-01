"""CPU-only orchestration for the frozen RCA-bench contracts.

This module deliberately does not load models or open hidden oracle data.  The first
slice owns only public plan/locator construction and exclusive publication; later slices
add the sealed ledger, template, and post-seal score phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .rca_bench import (
    ContractError,
    PLAN_SCHEMA,
    PROTOCOL_ID,
    canonical_bytes,
    canonical_sha256,
    compute_metrics,
    finalize_fatal_run,
    guard_response,
    RAW_SCHEMA,
    SCORE_SCHEMA,
    score_parsed_response,
    parse_response,
    validate_pre_inference_control,
    validate_raw,
    sha256_file,
    template_projection,
    validate_image_locators,
    validate_freeze_bindings,
    validate_freeze_receipt,
    validate_oracle,
    validate_plan,
    validate_score,
    RunLedger,
    write_canonical_json_exclusive,
)


_TEMPLATE_REPOSITORY = "deterministic-template"
_TEMPLATE_REVISION = "rca-bench-template-v1"


def template_weight_binding_sha256(generation_config: Mapping[str, Any]) -> str:
    """Bind the weight-less template identity to the exact canonical config."""
    return canonical_sha256({
        "model_repository": _TEMPLATE_REPOSITORY,
        "model_revision": _TEMPLATE_REVISION,
        "generation_config_sha256": canonical_sha256(generation_config),
    })


def _template_weight_from_receipt(freeze_receipt: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "model_repository": _TEMPLATE_REPOSITORY,
        "model_revision": _TEMPLATE_REVISION,
        "generation_config_sha256": freeze_receipt.get("generation_config_sha256"),
    })


def derive_method_class(freeze_receipt: Mapping[str, Any]) -> str:
    """Derive template/candidate class solely from the validated frozen identity."""
    repository = freeze_receipt.get("model_repository")
    revision = freeze_receipt.get("model_revision")
    mentions_template = repository == _TEMPLATE_REPOSITORY or revision == _TEMPLATE_REVISION
    if mentions_template:
        if (
            repository != _TEMPLATE_REPOSITORY
            or revision != _TEMPLATE_REVISION
            or freeze_receipt.get("model_weight_sha256")
            != _template_weight_from_receipt(freeze_receipt)
        ):
            raise ContractError("frozen template identity/weight binding is inconsistent")
        return "template"
    return "candidate"


def compute_actual_freeze_bindings(
    *,
    protocol_path: str | Path,
    builder_source_path: str | Path,
    scorer_source_path: str | Path,
    test_paths: Sequence[str | Path],
    evidence_schema_path: str | Path,
    prompt_path: str | Path,
    guard_config_path: str | Path,
    partition_paths: Mapping[str, str | Path],
    git_commit: str,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the actual frozen files and one canonical generation config object."""
    required_partition_names = {
        "split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code"
    }
    if set(partition_paths) != required_partition_names:
        raise ContractError("partition freeze paths must name every checker input")
    return {
        "protocol_sha256": sha256_file(protocol_path),
        "builder_source_sha256": sha256_file(builder_source_path),
        "scorer_source_sha256": sha256_file(scorer_source_path),
        "test_hashes": [sha256_file(path) for path in test_paths],
        "evidence_schema_sha256": sha256_file(evidence_schema_path),
        "prompt_sha256": sha256_file(prompt_path),
        "guard_config_sha256": sha256_file(guard_config_path),
        "partition_checker_bindings": {
            name: sha256_file(partition_paths[name]) for name in sorted(required_partition_names)
        },
        "git_commit": git_commit,
        "generation_config_sha256": canonical_sha256(generation_config),
    }


def validate_actual_freeze_bindings(
    freeze_receipt: Mapping[str, Any],
    actual_bindings: Mapping[str, Any],
) -> None:
    """Compare a freeze receipt with hashes freshly computed from source bytes."""
    fields = (
        "protocol_sha256", "builder_source_sha256", "scorer_source_sha256",
        "test_hashes", "evidence_schema_sha256", "prompt_sha256",
        "guard_config_sha256", "partition_checker_bindings", "git_commit",
        "generation_config_sha256",
    )
    for field in fields:
        if freeze_receipt.get(field) != actual_bindings.get(field):
            raise ContractError(f"freeze {field} differs from actual bytes/config")


def _read_canonical_generation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("generation config must be readable canonical JSON") from exc
    if not isinstance(value, dict) or config_path.read_bytes() != canonical_bytes(value):
        raise ContractError("generation config bytes are not the canonical frozen JSON object")
    return value


def _compute_bindings_from_paths(
    source_paths: Mapping[str, Any], generation_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "protocol", "builder_source", "scorer_source", "tests", "evidence_schema",
        "prompt", "guard_config", "partitions", "git_commit",
    }
    if set(source_paths) != required:
        raise ContractError("freeze source paths must explicitly name every frozen input")
    tests = source_paths["tests"]
    partitions = source_paths["partitions"]
    if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)):
        raise ContractError("freeze test source paths must be a sequence")
    if not isinstance(partitions, Mapping):
        raise ContractError("freeze partition source paths must be a mapping")
    config = _read_canonical_generation_config(generation_config_path)
    actual = compute_actual_freeze_bindings(
        protocol_path=source_paths["protocol"],
        builder_source_path=source_paths["builder_source"],
        scorer_source_path=source_paths["scorer_source"],
        test_paths=tests,
        evidence_schema_path=source_paths["evidence_schema"],
        prompt_path=source_paths["prompt"],
        guard_config_path=source_paths["guard_config"],
        partition_paths=partitions,
        git_commit=str(source_paths["git_commit"]),
        generation_config=config,
    )
    return actual, config


def _expected_binding_subset(bindings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(bindings[field])
        for field in (
            "protocol_sha256", "builder_source_sha256", "scorer_source_sha256",
            "test_hashes", "evidence_schema_sha256", "prompt_sha256",
            "guard_config_sha256", "git_commit",
        )
    }


def validate_template_run_preflight(
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    freeze_source_paths: Mapping[str, Any],
    generation_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform every oracle-free real-byte check before template execution."""
    actual, config = _compute_bindings_from_paths(
        freeze_source_paths, generation_config_path
    )
    checked_plan = validate_plan(
        plan,
        locators=locators,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    checked_locators = validate_image_locators(locators, verify_files=True)
    checked_freeze = validate_freeze_bindings(
        freeze_receipt,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=_expected_binding_subset(actual),
    )
    validate_actual_freeze_bindings(checked_freeze, actual)
    if checked_freeze["locator_sha256"] != canonical_sha256(checked_locators):
        raise ContractError("freeze locator hash differs from actual locator artifact")
    if checked_freeze["generation_config_sha256"] != canonical_sha256(config):
        raise ContractError("freeze generation config differs from canonical config bytes")
    if (
        checked_freeze["model_repository"] != _TEMPLATE_REPOSITORY
        or checked_freeze["model_revision"] != _TEMPLATE_REVISION
        or checked_freeze["model_weight_sha256"] != template_weight_binding_sha256(config)
    ):
        raise ContractError("template freeze sentinel or weight binding is invalid")
    return checked_plan, actual


def publish_full_freeze(
    freeze_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    path: str | Path,
    accepted_provenance_by_source_item: Mapping[str, object],
    freeze_source_paths: Mapping[str, Any],
    generation_config_path: str | Path,
) -> dict[str, Any]:
    """Validate actual bytes and the full hidden binding, then publish digests only."""
    actual, config = _compute_bindings_from_paths(
        freeze_source_paths, generation_config_path
    )
    validate_image_locators(locators, verify_files=True)
    checked = validate_freeze_receipt(
        freeze_receipt,
        plan,
        locators=locators,
        oracle=oracle,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=_expected_binding_subset(actual),
    )
    validate_actual_freeze_bindings(checked, actual)
    if oracle.get("protocol_sha256") != checked["protocol_sha256"]:
        raise ContractError("oracle protocol hash differs from actual frozen protocol")
    if (
        checked["model_repository"] != _TEMPLATE_REPOSITORY
        or checked["model_revision"] != _TEMPLATE_REVISION
        or checked["model_weight_sha256"] != template_weight_binding_sha256(config)
    ):
        raise ContractError("template freeze sentinel or weight binding is invalid")
    return publish_freeze_digest_only(
        checked,
        plan,
        locators,
        oracle_sha256=canonical_sha256(oracle),
        path=path,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=_expected_binding_subset(actual),
    )


def build_public_plan(
    items: Sequence[Mapping[str, Any]],
    *,
    benchmark_role: str,
    development_partition: str | None,
    split_version: str,
    prompt_template_sha256: str,
    guard_config_sha256: str,
    evidence_schema_sha256: str,
    model_input_schema_sha256: str,
    source_artifacts: Sequence[Mapping[str, Any]],
    accepted_provenance_by_source_item: Mapping[str, object],
) -> dict[str, Any]:
    """Construct and validate a public plan from already projected Part B items."""
    checked_items: list[dict[str, Any]] = []
    for raw_item in items:
        item = copy.deepcopy(dict(raw_item))
        item["development_partition"] = development_partition
        item["evidence_bundle_sha256"] = canonical_sha256(item["evidence_bundle"])
        checked_items.append(item)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "benchmark_role": benchmark_role,
        "development_partition": development_partition,
        "split_version": split_version,
        "prompt_template_sha256": prompt_template_sha256,
        "guard_config_sha256": guard_config_sha256,
        "evidence_schema_sha256": evidence_schema_sha256,
        "model_input_schema_sha256": model_input_schema_sha256,
        "source_artifacts": copy.deepcopy(list(source_artifacts)),
        "planned_item_count": len(checked_items),
        "model_attempt_planned_count": sum(
            item["execution_policy"] == "model_attempt" for item in checked_items
        ),
        "pre_inference_validation_planned_count": sum(
            item["execution_policy"] == "pre_inference_validation_only"
            for item in checked_items
        ),
        "items": checked_items,
    }
    return validate_plan(
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )


def build_image_locators(
    plan: Mapping[str, Any],
    *,
    rgb_paths: Mapping[str, str | Path],
    verify_files: bool,
) -> dict[str, Any]:
    """Build the locator map from the plan's delivered RGB identities."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in plan["items"]:
        for delivery in item["image_deliveries"]:
            image_id = str(delivery["delivered_image_id"])
            if image_id in seen:
                continue
            seen.add(image_id)
            path = Path(rgb_paths[image_id])
            if verify_files:
                if path.is_symlink() or not path.is_file():
                    raise ContractError(f"RGB locator is unreadable: {path}")
                digest = sha256_file(path)
                if digest != delivery["delivered_rgb_sha256"]:
                    raise ContractError(f"RGB locator hash mismatch: {image_id}")
            rows.append(
                {
                    "image_id": image_id,
                    "rgb_path": str(path),
                    "rgb_sha256": str(delivery["delivered_rgb_sha256"]),
                }
            )
    locator = {
        "schema_version": "rca-bench-image-locators-v1",
        "plan_sha256": canonical_sha256(plan),
        "images": rows,
    }
    return validate_image_locators(locator, verify_files=verify_files)


def publish_public_artifacts(
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    *,
    plan_path: str | Path,
    locator_path: str | Path,
    accepted_provenance_by_source_item: Mapping[str, object],
) -> None:
    """Validate and publish the plan/locator pair without a partial final state."""
    checked_plan = validate_plan(
        plan,
        locators=locators,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    checked_locators = validate_image_locators(locators, verify_files=True)
    if checked_locators["plan_sha256"] != canonical_sha256(checked_plan):
        raise ContractError("locator plan hash differs from public plan")
    outputs = (Path(plan_path), Path(locator_path))
    if outputs[0] == outputs[1]:
        raise ContractError("public plan and locator outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ContractError("refusing to overwrite an existing public artifact pair")
    staged = tuple(
        path.with_name(f".{path.name}.pair-incomplete") for path in outputs
    )
    linked: list[Path] = []
    try:
        write_canonical_json_exclusive(staged[0], checked_plan)
        write_canonical_json_exclusive(staged[1], checked_locators)
        for source, target in zip(staged, outputs, strict=True):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.parent.is_symlink() or not target.parent.is_dir():
                raise ContractError("public output parent must be a real directory")
            os.link(source, target, follow_symlinks=False)
            linked.append(target)
        for parent in {path.parent for path in outputs}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        for target in linked:
            if target.exists() and not target.is_symlink():
                target.unlink()
        raise
    finally:
        for temporary in staged:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()


def publish_oracle_artifact(
    oracle: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    path: str | Path,
    accepted_provenance_by_source_item: Mapping[str, object],
) -> dict[str, Any]:
    """Validate and publish one hidden oracle bound to the exact public plan."""
    checked = validate_oracle(
        oracle,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    if [row["item_id"] for row in checked["items"]] != [
        item["item_id"] for item in plan["items"]
    ]:
        raise ContractError("oracle item order must exactly match the public plan")
    write_canonical_json_exclusive(path, checked)
    return checked


def publish_freeze_digest_only(
    freeze_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    *,
    oracle_sha256: str,
    path: str | Path,
    accepted_provenance_by_source_item: Mapping[str, object],
    expected_bindings: Mapping[str, object],
) -> dict[str, Any]:
    """Publish a structurally validated receipt with only an oracle digest."""
    checked = validate_freeze_bindings(
        freeze_receipt,
        plan,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected_bindings,
    )
    checked_locators = validate_image_locators(locators)
    if checked_locators["plan_sha256"] != canonical_sha256(plan):
        raise ContractError("freeze locator plan hash differs from public plan")
    if checked["locator_sha256"] != canonical_sha256(locators):
        raise ContractError("freeze locator hash differs from actual locators")
    if checked["oracle_sha256"] != oracle_sha256:
        raise ContractError("freeze oracle digest differs from supplied oracle digest")
    write_canonical_json_exclusive(path, checked)
    return checked


def _ledger_sequences(items: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    validation_ids = [
        str(item["item_id"])
        for item in items
        if item.get("execution_policy") == "pre_inference_validation_only"
    ]
    model_ids = [
        str(item["item_id"])
        for item in items
        if item.get("execution_policy") == "model_attempt"
    ]
    if len(set(validation_ids)) != len(validation_ids) or len(set(model_ids)) != len(model_ids):
        raise ContractError("ledger item IDs must be unique")
    return validation_ids, model_ids


def validate_ledger_state(
    events: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]],
    *, allow_in_progress: bool = False,
) -> bool:
    """Validate the frozen preflight/validation/model/fatal event state machine."""
    validation_ids, model_ids = _ledger_sequences(items)
    if not events:
        raise ContractError("ledger must contain a preflight event")
    if events[0].get("event_type") != "preflight_started":
        raise ContractError("ledger must start with one preflight_started event")
    if sum(event.get("event_type") == "preflight_started" for event in events) != 1:
        raise ContractError("ledger preflight_started must occur exactly once")
    fatal_positions = [index for index, event in enumerate(events) if event.get("event_type") == "fatal"]
    if len(fatal_positions) > 1 or (fatal_positions and fatal_positions[0] != len(events) - 1):
        raise ContractError("fatal must be one terminal final event")
    if len(events) == 1:
        if allow_in_progress:
            return True
        raise ContractError("preflight must complete or terminate fatally")
    if events[1].get("event_type") == "fatal":
        return True
    if events[1].get("event_type") != "preflight_completed":
        raise ContractError("preflight_completed must follow preflight_started")
    if sum(event.get("event_type") == "preflight_completed" for event in events) != 1:
        raise ContractError("ledger preflight_completed must occur exactly once")

    index = 2
    completed_validations = 0
    for expected_index, item_id in enumerate(validation_ids):
        if index >= len(events) or events[index].get("event_type") != "validation_started":
            break
        event = events[index]
        if event.get("item_id") != item_id or event.get("execution_index") != expected_index:
            raise ContractError("validation rows are out of frozen order")
        if index + 1 >= len(events) or events[index + 1].get("event_type") != "validation_terminal":
            if index + 1 == len(events) and not fatal_positions:
                if allow_in_progress:
                    return True
                raise ContractError("validation start must be followed by its terminal")
            raise ContractError("validation start must be followed by its terminal")
        terminal = events[index + 1]
        if terminal.get("item_id") != item_id or terminal.get("execution_index") != expected_index:
            raise ContractError("validation terminal does not match its start")
        index += 2
        completed_validations += 1
    if completed_validations != len(validation_ids):
        if fatal_positions and index == fatal_positions[0]:
            return True
        if index < len(events) and events[index].get("event_type") == "model_attempt_started":
            raise ContractError("all validation rows must terminate before model starts")
        if index == len(events):
            if allow_in_progress:
                return True
            raise ContractError("validation rows are incomplete")
        raise ContractError("validation rows are incomplete or out of order")

    for expected_index, item_id in enumerate(model_ids):
        if index >= len(events):
            if allow_in_progress:
                return True
            raise ContractError("model rows are incomplete")
        if events[index].get("event_type") != "model_attempt_started":
            if events[index].get("event_type") == "fatal" and index == len(events) - 1:
                return True
            raise ContractError("model rows are out of frozen order")
        event = events[index]
        if event.get("item_id") != item_id or event.get("execution_index") != expected_index:
            raise ContractError("model rows are out of frozen order")
        if index + 1 >= len(events) or events[index + 1].get("event_type") != "model_attempt_terminal":
            if index + 1 == len(events) and not fatal_positions:
                if allow_in_progress:
                    return True
                raise ContractError("model start must be followed by its terminal")
            raise ContractError("model start must be followed by its terminal")
        terminal = events[index + 1]
        if terminal.get("item_id") != item_id or terminal.get("execution_index") != expected_index:
            raise ContractError("model terminal does not match its start")
        index += 2
    if index < len(events):
        if events[index].get("event_type") == "fatal" and index == len(events) - 1:
            return True
        raise ContractError("unexpected event after frozen model sequence")
    return True


class ControllerLedger:
    """Phase-checked ledger that persists payload sidecars before each event."""

    def __init__(self, path: str | Path, items: Sequence[Mapping[str, Any]]):
        self._items = [dict(item) for item in items]
        self._ledger = RunLedger(path)
        self._sidecar_dir = Path(path).with_name(Path(path).name + ".payloads")
        self._sidecar_dir.mkdir(parents=True, exist_ok=False)

    @property
    def path(self) -> Path:
        return self._ledger.path

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._ledger.events

    @property
    def payloads(self) -> dict[str, object]:
        return self._ledger.payloads

    @property
    def sidecar_dir(self) -> Path:
        return self._sidecar_dir

    def _write_sidecar(self, payload: object) -> str:
        digest = canonical_sha256(payload)
        sidecar = self._sidecar_dir / f"{digest}.json"
        if sidecar.exists():
            if sidecar.read_bytes() != canonical_bytes(payload):
                raise ContractError("payload sidecar hash collision or tamper")
        else:
            write_canonical_json_exclusive(sidecar, payload)
        return digest

    def append(self, event_type: str, *, item_id: str | None = None,
               execution_index: int | None = None, payload: object) -> dict[str, Any]:
        candidate = {
            "event_type": event_type,
            "item_id": item_id,
            "execution_index": execution_index,
        }
        self._check_append(candidate)
        self._write_sidecar(payload)
        return self._ledger.append(
            event_type,
            item_id=item_id,
            execution_index=execution_index,
            payload=payload,
        )

    def _check_append(self, candidate: Mapping[str, Any]) -> None:
        event_type = candidate["event_type"]
        item_id = candidate["item_id"]
        execution_index = candidate["execution_index"]
        if self.events and self.events[-1].get("event_type") == "fatal":
            raise ContractError("no event is allowed after fatal")
        if not self.events:
            if event_type != "preflight_started":
                raise ContractError("ledger must start with preflight_started")
            return
        if event_type == "fatal":
            return
        if event_type == "preflight_started":
            raise ContractError("preflight_started must occur once")
        if len(self.events) == 1:
            if event_type != "preflight_completed":
                raise ContractError("preflight_completed must follow preflight_started")
            return
        try:
            validate_ledger_state(
                [*self.events, dict(candidate)], self._items,
                allow_in_progress=True,
            )
        except ContractError as exc:
            # A start without its terminal is a valid in-progress append state.
            if event_type in {"validation_started", "model_attempt_started"}:
                self._check_start_transition(item_id, execution_index, event_type)
                return
            raise exc

    def _check_start_transition(self, item_id: str | None, execution_index: int | None,
                                event_type: str) -> None:
        validation_ids, model_ids = _ledger_sequences(self._items)
        if event_type == "validation_started":
            expected = validation_ids
            start_type, terminal_type = "validation_started", "validation_terminal"
        else:
            expected = model_ids
            start_type, terminal_type = "model_attempt_started", "model_attempt_terminal"
            if any(event.get("event_type") == "validation_started" and not any(
                later.get("event_type") == "validation_terminal"
                and later.get("item_id") == event.get("item_id")
                for later in self.events
            ) for event in self.events):
                raise ContractError("all validation rows must terminate before model starts")
        started = [event for event in self.events if event.get("event_type") == start_type]
        terminals = [event for event in self.events if event.get("event_type") == terminal_type]
        if len(started) != len(terminals) or len(started) >= len(expected):
            raise ContractError("ledger start is duplicate or out of order")
        if item_id != expected[len(started)] or execution_index != len(started):
            raise ContractError("ledger start is out of frozen order")


def finalize_fatal_reopen(
    ledger_path: str | Path,
    *,
    items: Sequence[Mapping[str, Any]],
    raw_bindings: Mapping[str, Any],
    fatal_event_sha256: str,
    raw_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reconstruct a fatal raw artifact from journal and payload sidecars only."""
    journal = Path(ledger_path)
    sidecar_dir = journal.with_name(journal.name + ".payloads")
    if not journal.is_file() or not sidecar_dir.is_dir():
        raise ContractError("fatal reopen requires the journal and payload sidecars")
    events: list[dict[str, Any]] = []
    payloads: dict[str, object] = {}
    for line in journal.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        events.append(event)
        payload_sha256 = event["payload_sha256"]
        sidecar = sidecar_dir / f"{payload_sha256}.json"
        if not sidecar.is_file():
            raise ContractError("ledger payload sidecar is missing")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if canonical_sha256(payload) != payload_sha256:
            raise ContractError("ledger payload sidecar hash mismatch")
        payloads[payload_sha256] = payload
    raw = finalize_fatal_run(
        items,
        events,
        payloads,
        fatal_event_sha256=fatal_event_sha256,
        raw_bindings=raw_bindings,
    )
    if raw_path is not None:
        write_canonical_json_exclusive(raw_path, raw)
    return raw


_RAW_RECORD_FIELDS = {
    "record_index", "item_id", "execution_policy", "record_status", "attempt_index",
    "validation_index", "model_repository", "model_revision", "model_weight_sha256",
    "prompt_sha256", "ordered_delivered_image_ids", "ordered_delivered_rgb_sha256",
    "generation_config_sha256", "response", "response_utf8_sha256", "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
    "processor_telemetry", "telemetry_unavailable_reason", "error_type", "error",
    "validation_stage", "rejection_code", "validator_receipt_sha256",
    "fatal_event_sha256", "not_run_reason",
}


def _blank_raw_record(record_index: int, item: Mapping[str, Any], status: str) -> dict[str, Any]:
    record = {field: None for field in _RAW_RECORD_FIELDS}
    record.update({
        "record_index": record_index,
        "item_id": item["item_id"],
        "execution_policy": item["execution_policy"],
        "record_status": status,
    })
    return record


def _raw_bindings(
    plan: Mapping[str, Any], freeze_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "plan_sha256": canonical_sha256(plan),
        "freeze_receipt_sha256": canonical_sha256(freeze_receipt),
        "model_repository": freeze_receipt["model_repository"],
        "model_revision": freeze_receipt["model_revision"],
        "model_weight_sha256": freeze_receipt["model_weight_sha256"],
        "generation_config_sha256": freeze_receipt["generation_config_sha256"],
    }


def _fatal_model_run(
    ledger: ControllerLedger,
    *,
    checked_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    raw_path: str | Path,
    fatal_code: str,
    exc: BaseException,
    fatal_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"fatal_code": fatal_code, "error_type": type(exc).__name__}
    if fatal_payload is not None:
        payload.update(copy.deepcopy(dict(fatal_payload)))
    fatal = ledger.append(
        "fatal",
        payload=payload,
    )
    return finalize_fatal_reopen(
        ledger.path,
        items=checked_plan["items"],
        raw_bindings=_raw_bindings(plan, freeze_receipt),
        fatal_event_sha256=fatal["event_sha256"],
        raw_path=raw_path,
    )


def _complete_raw(
    *,
    plan: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    ledger: ControllerLedger,
    items: Sequence[Mapping[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = (
        "completed", "inference_error", "pre_inference_rejected",
        "pre_inference_error", "not_run_after_fatal",
    )
    counts = {
        status: sum(record["record_status"] == status for record in records)
        for status in statuses
    }
    model_items = [item for item in items if item["execution_policy"] == "model_attempt"]
    validation_items = [
        item for item in items
        if item["execution_policy"] == "pre_inference_validation_only"
    ]
    return {
        "schema_version": RAW_SCHEMA,
        **_raw_bindings(plan, freeze_receipt),
        "journal_terminal_sha256": ledger.events[-1]["event_sha256"],
        "run_status": "complete",
        "planned_item_count": len(records),
        "model_attempt_planned_count": len(model_items),
        "pre_inference_validation_planned_count": len(validation_items),
        "completed_count": counts["completed"],
        "inference_error_count": counts["inference_error"],
        "pre_inference_rejected_count": counts["pre_inference_rejected"],
        "pre_inference_error_count": counts["pre_inference_error"],
        "not_run_after_fatal_count": counts["not_run_after_fatal"],
        "model_not_run_after_fatal_count": sum(
            record["record_status"] == "not_run_after_fatal"
            and record["execution_policy"] == "model_attempt"
            for record in records
        ),
        "validation_not_run_after_fatal_count": sum(
            record["record_status"] == "not_run_after_fatal"
            and record["execution_policy"] == "pre_inference_validation_only"
            for record in records
        ),
        "records": records,
    }


def _stock_max_new_tokens(config: Mapping[str, Any]) -> int:
    """Validate the frozen greedy/batch-one policy and return its token cap."""
    if config.get("do_sample") not in (None, False):
        raise ContractError("stock generation config must use greedy decoding")
    if config.get("temperature") not in (None, 0, 0.0):
        raise ContractError("stock generation config temperature must be zero or absent")
    if config.get("batch_size") not in (None, 1):
        raise ContractError("stock generation config batch size must be one")
    value = config.get("max_new_tokens", 384)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("stock generation max_new_tokens must be a positive integer")
    return value


_AUTO_LOAD_TELEMETRY_TORCH = object()


def _finite_nonnegative_number(value: object, role: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{role} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ContractError(f"{role} must be a finite nonnegative number")
    return value


def _measure_runtime_load(
    runtime_loader: Any,
    repository: str,
    revision: str,
    snapshot_path: str | Path,
    *,
    torch_module: Any,
    clock: Any,
    telemetry: dict[str, Any],
) -> Any:
    """Load once while isolating load wall time and CUDA peak counters."""
    cuda = None if torch_module is None else getattr(torch_module, "cuda", None)
    cuda_available = bool(
        cuda is not None and getattr(cuda, "is_available", lambda: False)()
    )
    unavailable_reason = None
    if cuda_available:
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
    else:
        unavailable_reason = "torch_unavailable" if torch_module is None else "cuda_unavailable"
    started = clock()
    try:
        return runtime_loader(repository, revision, snapshot_path)
    finally:
        if cuda_available:
            cuda.synchronize()
        finished = clock()
        elapsed = round(float(finished - started), 6)
        _finite_nonnegative_number(elapsed, "model load elapsed seconds")
        allocated = reserved = None
        if cuda_available:
            allocated = cuda.max_memory_allocated()
            reserved = cuda.max_memory_reserved()
            _finite_nonnegative_number(allocated, "model load peak allocated bytes")
            _finite_nonnegative_number(reserved, "model load peak reserved bytes")
            if not isinstance(allocated, int) or not isinstance(reserved, int):
                raise ContractError("model load CUDA peak bytes must be nonnegative integers")
        telemetry.update({
            "model_load_seconds": elapsed,
            "model_load_peak_cuda_memory_allocated_bytes": allocated,
            "model_load_peak_cuda_memory_reserved_bytes": reserved,
            "model_load_telemetry_unavailable_reason": unavailable_reason,
        })


def run_model_phase(
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    freeze_source_paths: Mapping[str, Any],
    generation_config_path: str | Path,
    model_snapshot_path: str | Path,
    ledger_path: str | Path,
    raw_path: str | Path,
    runtime_loader: Any = None,
    runtime_invoker: Any = None,
    load_telemetry_torch_module: Any = _AUTO_LOAD_TELEMETRY_TORCH,
    load_clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Run one exact stock Qwen snapshot without opening hidden oracle data."""
    from .model_bakeoff_runner import (
        ImageInput,
        invoke_stock_runtime,
        load_stock_runtime,
        snapshot_inventory,
        validate_model_selection,
    )

    raw_output = Path(raw_path)
    if raw_output.exists() or raw_output.is_symlink():
        raise ContractError(f"refusing to overwrite occupied raw output: {raw_output}")
    if raw_output.parent.exists() and (
        raw_output.parent.is_symlink() or not raw_output.parent.is_dir()
    ):
        raise ContractError("raw output parent must be a real directory")
    checked_plan = validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )
    ledger = ControllerLedger(ledger_path, checked_plan["items"])
    ledger.append("preflight_started", payload={"phase": "stock_model"})
    runtime_loader = runtime_loader or load_stock_runtime
    runtime_invoker = runtime_invoker or invoke_stock_runtime
    try:
        actual, generation_config = _compute_bindings_from_paths(
            freeze_source_paths, generation_config_path
        )
        checked_plan = validate_plan(
            plan, locators=locators,
            accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        )
        checked_locators = validate_image_locators(locators, verify_files=True)
        checked_freeze = validate_freeze_bindings(
            freeze_receipt,
            plan,
            accepted_provenance_by_source_item=accepted_provenance_by_source_item,
            expected_bindings=_expected_binding_subset(actual),
        )
        validate_actual_freeze_bindings(checked_freeze, actual)
        if checked_freeze["locator_sha256"] != canonical_sha256(checked_locators):
            raise ContractError("freeze locator hash differs from actual locator artifact")
        if checked_freeze["generation_config_sha256"] != canonical_sha256(generation_config):
            raise ContractError("freeze generation config differs from canonical config bytes")
        max_new_tokens = _stock_max_new_tokens(generation_config)
        spec = validate_model_selection(
            checked_freeze["model_repository"], checked_freeze["model_revision"]
        )
        if not spec.stock_runtime or spec.trust_remote_code:
            raise ContractError("run-model accepts only an exact stock no-remote-code model")
        inventory = snapshot_inventory(model_snapshot_path)
        if inventory["sha256"] != checked_freeze["model_weight_sha256"]:
            raise ContractError("model snapshot inventory differs from frozen model weight binding")
        locator_by_id = {row["image_id"]: row for row in checked_locators["images"]}
        effective_bindings = _expected_binding_subset(actual)
    except Exception as exc:
        return _fatal_model_run(
            ledger, checked_plan=checked_plan, plan=plan,
            freeze_receipt=freeze_receipt, raw_path=raw_path,
            fatal_code="preflight_error", exc=exc,
        )

    ledger.append(
        "preflight_completed",
        payload={
            "status": "ok",
            "snapshot_inventory_sha256": inventory["sha256"],
        },
    )
    records_by_id: dict[str, dict[str, Any]] = {}
    validation_items = [
        item for item in checked_plan["items"]
        if item["execution_policy"] == "pre_inference_validation_only"
    ]
    model_items = [
        item for item in checked_plan["items"]
        if item["execution_policy"] == "model_attempt"
    ]
    for validation_index, item in enumerate(validation_items):
        ledger.append(
            "validation_started", item_id=item["item_id"],
            execution_index=validation_index, payload={"item_id": item["item_id"]},
        )
        unexpectedly_accepted = False
        try:
            receipt = validate_pre_inference_control(item)
            if receipt.get("record_status") != "pre_inference_rejected":
                unexpectedly_accepted = True
                raise ContractError("validation control was unexpectedly accepted")
            record = _blank_raw_record(
                checked_plan["items"].index(item), item, "pre_inference_rejected"
            )
            record.update({"validation_index": validation_index, **receipt})
            ledger.append(
                "validation_terminal", item_id=item["item_id"],
                execution_index=validation_index, payload=record,
            )
        except Exception as exc:
            error_type = (
                "control_not_rejected"
                if unexpectedly_accepted
                else "validator_error"
            )
            record = _blank_raw_record(
                checked_plan["items"].index(item), item, "pre_inference_error"
            )
            record.update({
                "validation_index": validation_index,
                "error_type": error_type,
                "error": str(exc)[:4000],
            })
            ledger.append(
                "validation_terminal", item_id=item["item_id"],
                execution_index=validation_index, payload=record,
            )
            records_by_id[item["item_id"]] = record
            return _fatal_model_run(
                ledger, checked_plan=checked_plan, plan=plan,
                freeze_receipt=freeze_receipt, raw_path=raw_path,
                fatal_code="pre_inference_validation_failed", exc=exc,
            )
        records_by_id[item["item_id"]] = record

    if load_telemetry_torch_module is _AUTO_LOAD_TELEMETRY_TORCH:
        try:
            import torch as load_telemetry_torch_module
        except Exception:
            load_telemetry_torch_module = None
    load_telemetry: dict[str, Any] = {}
    try:
        runtime = _measure_runtime_load(
            runtime_loader,
            checked_freeze["model_repository"],
            checked_freeze["model_revision"],
            model_snapshot_path,
            torch_module=load_telemetry_torch_module,
            clock=load_clock,
            telemetry=load_telemetry,
        )
        if runtime.inventory != inventory:
            raise ContractError("loaded runtime inventory differs from preflight inventory")
        if runtime.model_class != "Qwen3VLForConditionalGeneration":
            raise ContractError("loaded runtime model class differs from stock Qwen3-VL")
    except Exception as exc:
        return _fatal_model_run(
            ledger, checked_plan=checked_plan, plan=plan,
            freeze_receipt=freeze_receipt, raw_path=raw_path,
            fatal_code="model_load_error", exc=exc,
            fatal_payload={"model_load_telemetry": load_telemetry},
        )

    runtime_provenance = {
        "snapshot_inventory": inventory["files"],
        "snapshot_inventory_sha256": inventory["sha256"],
        "processor_class": runtime.processor_class,
        "model_class": runtime.model_class,
        "trust_remote_code": False,
        "local_files_only": True,
        **load_telemetry,
    }
    for attempt_index, item in enumerate(model_items):
        ledger.append(
            "model_attempt_started", item_id=item["item_id"],
            execution_index=attempt_index,
            payload={
                "item_id": item["item_id"],
                "model_load_telemetry": load_telemetry,
            },
        )
        deliveries = item["image_deliveries"]
        images = tuple(
            ImageInput(
                path=Path(locator_by_id[delivery["delivered_image_id"]]["rgb_path"]),
                sha256=delivery["delivered_rgb_sha256"],
                image_id=delivery["delivered_image_id"],
            )
            for delivery in deliveries
        )
        started = time.monotonic()
        try:
            outcome = runtime_invoker(
                runtime, item["prompt"], images, max_new_tokens=max_new_tokens
            )
            if not isinstance(outcome, Mapping):
                raise ContractError("stock runtime outcome must be a mapping")
            if outcome.get("status") == "completed":
                response = outcome.get("response_raw")
                if not isinstance(response, str):
                    raise ContractError("stock runtime completed without a text response")
                status = "completed"
                error_type = error = None
            elif outcome.get("status") == "error":
                response = None
                status = "inference_error"
                error_type = str(outcome.get("error_type") or "RuntimeError")
                error = str(outcome.get("error") or "stock runtime inference error")[:4000]
            else:
                raise ContractError("stock runtime outcome status is invalid")
        except Exception as exc:
            outcome = {}
            response = None
            status = "inference_error"
            error_type, error = type(exc).__name__, str(exc)[:4000]
        telemetry = dict(outcome.get("processor_telemetry") or {})
        telemetry.update(runtime_provenance)
        record = _blank_raw_record(
            checked_plan["items"].index(item), item, status
        )
        record.update({
            "attempt_index": attempt_index,
            "model_repository": freeze_receipt["model_repository"],
            "model_revision": freeze_receipt["model_revision"],
            "model_weight_sha256": freeze_receipt["model_weight_sha256"],
            "prompt_sha256": item["prompt_sha256"],
            "ordered_delivered_image_ids": [
                delivery["delivered_image_id"] for delivery in deliveries
            ],
            "ordered_delivered_rgb_sha256": [
                delivery["delivered_rgb_sha256"] for delivery in deliveries
            ],
            "generation_config_sha256": freeze_receipt["generation_config_sha256"],
            "response": response,
            "response_utf8_sha256": (
                hashlib.sha256(response.encode()).hexdigest() if response is not None else None
            ),
            "elapsed_seconds": outcome.get(
                "elapsed_seconds", round(time.monotonic() - started, 6)
            ),
            "peak_cuda_memory_allocated_bytes": outcome.get(
                "peak_cuda_memory_allocated_bytes"
            ),
            "peak_cuda_memory_reserved_bytes": outcome.get(
                "peak_cuda_memory_reserved_bytes"
            ),
            "processor_telemetry": telemetry,
            "telemetry_unavailable_reason": outcome.get(
                "telemetry_unavailable_reason",
                "runtime_exception_before_complete_telemetry" if status == "inference_error"
                else "cuda_unavailable",
            ),
            "error_type": error_type,
            "error": error,
        })
        ledger.append(
            "model_attempt_terminal", item_id=item["item_id"],
            execution_index=attempt_index, payload=record,
        )
        records_by_id[item["item_id"]] = record

    records = [records_by_id[item["item_id"]] for item in checked_plan["items"]]
    raw = _complete_raw(
        plan=plan, freeze_receipt=freeze_receipt, ledger=ledger,
        items=checked_plan["items"], records=records,
    )
    try:
        validate_raw(
            raw, plan, freeze_receipt,
            accepted_provenance_by_source_item=accepted_provenance_by_source_item,
            ledger_events=ledger.events,
            expected_freeze_bindings=effective_bindings,
        )
        write_canonical_json_exclusive(raw_path, raw)
    except Exception as exc:
        ledger.append(
            "fatal", payload={"fatal_code": "raw_seal_failed", "error_type": type(exc).__name__}
        )
        raise
    return raw


def run_template_baseline(
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    expected_bindings: Mapping[str, object] | None = None,
    freeze_source_paths: Mapping[str, Any] | None = None,
    generation_config_path: str | Path | None = None,
    ledger_path: str | Path,
    raw_path: str | Path,
    fatal_before_attempt: int | None = None,
) -> dict[str, Any]:
    """Run the deterministic template through the sealed CPU ledger path."""
    checked_plan = validate_plan(
        plan, accepted_provenance_by_source_item=accepted_provenance_by_source_item
    )
    ledger = ControllerLedger(ledger_path, checked_plan["items"])
    ledger.append("preflight_started", payload={"phase": "template"})
    try:
        if freeze_source_paths is None or generation_config_path is None:
            raise ContractError(
                "template run requires explicit frozen source paths and canonical generation config"
            )
        checked_plan, actual = validate_template_run_preflight(
            plan,
            locators,
            freeze_receipt,
            accepted_provenance_by_source_item=accepted_provenance_by_source_item,
            freeze_source_paths=freeze_source_paths,
            generation_config_path=generation_config_path,
        )
        effective_bindings = _expected_binding_subset(actual)
    except Exception as exc:
        ledger.append(
            "fatal",
            payload={"fatal_code": "preflight_validation_failed", "error_type": type(exc).__name__},
        )
        raise
    try:
        ledger.append("preflight_completed", payload={"status": "ok"})
    except Exception as exc:
        if not ledger.events or ledger.events[-1].get("event_type") != "fatal":
            ledger.append(
                "fatal",
                payload={"fatal_code": "preflight_completion_failed", "error_type": type(exc).__name__},
            )
        raise
    records_by_id: dict[str, dict[str, Any]] = {}
    validation_index = 0
    attempt_index = 0
    validation_items = [
        item for item in checked_plan["items"]
        if item["execution_policy"] == "pre_inference_validation_only"
    ]
    model_items = [
        item for item in checked_plan["items"]
        if item["execution_policy"] == "model_attempt"
    ]
    for item in validation_items:
        ledger.append(
            "validation_started", item_id=item["item_id"],
            execution_index=validation_index, payload={"item_id": item["item_id"]},
        )
        try:
            receipt = validate_pre_inference_control(item)
        except Exception as exc:
            ledger.append(
                "fatal",
                payload={"fatal_code": "validation_failed", "error_type": type(exc).__name__},
            )
            raise
        record = _blank_raw_record(
            checked_plan["items"].index(item), item, "pre_inference_rejected"
        )
        record.update({"validation_index": validation_index, **receipt})
        try:
            ledger.append(
                "validation_terminal", item_id=item["item_id"],
                execution_index=validation_index, payload=record,
            )
        except Exception as exc:
            ledger.append(
                "fatal",
                payload={"fatal_code": "validation_terminal_failed", "error_type": type(exc).__name__},
            )
            raise
        records_by_id[item["item_id"]] = record
        validation_index += 1
    for item in model_items:
        if fatal_before_attempt is not None and attempt_index == fatal_before_attempt:
            fatal = ledger.append(
                "fatal",
                payload={"fatal_code": "injected_template_fatal"},
            )
            return finalize_fatal_reopen(
                ledger.path,
                items=checked_plan["items"],
                raw_bindings={
                    "plan_sha256": canonical_sha256(plan),
                    "freeze_receipt_sha256": canonical_sha256(freeze_receipt),
                    "model_repository": freeze_receipt["model_repository"],
                    "model_revision": freeze_receipt["model_revision"],
                    "model_weight_sha256": freeze_receipt["model_weight_sha256"],
                    "generation_config_sha256": freeze_receipt["generation_config_sha256"],
                },
                fatal_event_sha256=fatal["event_sha256"],
                raw_path=raw_path,
            )
        ledger.append(
            "model_attempt_started", item_id=item["item_id"],
            execution_index=attempt_index, payload={"item_id": item["item_id"]},
        )
        started = time.monotonic()
        try:
            response = template_projection(
                item,
                accepted_provenance=accepted_provenance_by_source_item[item["source_item_id"]],
            )["response"]
        except Exception as exc:
            ledger.append(
                "fatal",
                payload={"fatal_code": "template_attempt_failed", "error_type": type(exc).__name__},
            )
            raise
        elapsed = time.monotonic() - started
        record = _blank_raw_record(
            checked_plan["items"].index(item), item, "completed"
        )
        record.update({
            "attempt_index": attempt_index,
            "model_repository": freeze_receipt["model_repository"],
            "model_revision": freeze_receipt["model_revision"],
            "model_weight_sha256": freeze_receipt["model_weight_sha256"],
            "prompt_sha256": item["prompt_sha256"],
            "ordered_delivered_image_ids": [
                delivery["delivered_image_id"] for delivery in item["image_deliveries"]
            ],
            "ordered_delivered_rgb_sha256": [
                delivery["delivered_rgb_sha256"] for delivery in item["image_deliveries"]
            ],
            "generation_config_sha256": freeze_receipt["generation_config_sha256"],
            "response": response,
            "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "elapsed_seconds": elapsed,
            "processor_telemetry": {
                "ordered_image_count": len(item["image_deliveries"]),
                "runner": "deterministic_template",
            },
            "telemetry_unavailable_reason": "cpu_template_no_cuda",
        })
        try:
            ledger.append(
                "model_attempt_terminal", item_id=item["item_id"],
                execution_index=attempt_index, payload=record,
            )
        except Exception as exc:
            ledger.append(
                "fatal",
                payload={"fatal_code": "model_terminal_failed", "error_type": type(exc).__name__},
            )
            raise
        records_by_id[item["item_id"]] = record
        attempt_index += 1
    records = [records_by_id[item["item_id"]] for item in checked_plan["items"]]
    counts = {status: sum(record["record_status"] == status for record in records)
              for status in ("completed", "inference_error", "pre_inference_rejected",
                             "pre_inference_error", "not_run_after_fatal")}
    raw = {
        "schema_version": RAW_SCHEMA,
        "plan_sha256": canonical_sha256(plan),
        "freeze_receipt_sha256": canonical_sha256(freeze_receipt),
        "journal_terminal_sha256": ledger.events[-1]["event_sha256"],
        "model_repository": freeze_receipt["model_repository"],
        "model_revision": freeze_receipt["model_revision"],
        "model_weight_sha256": freeze_receipt["model_weight_sha256"],
        "generation_config_sha256": freeze_receipt["generation_config_sha256"],
        "run_status": "complete",
        "planned_item_count": len(records),
        "model_attempt_planned_count": len(model_items),
        "pre_inference_validation_planned_count": len(validation_items),
        "completed_count": counts["completed"],
        "inference_error_count": counts["inference_error"],
        "pre_inference_rejected_count": counts["pre_inference_rejected"],
        "pre_inference_error_count": counts["pre_inference_error"],
        "not_run_after_fatal_count": counts["not_run_after_fatal"],
        "model_not_run_after_fatal_count": sum(
            record["record_status"] == "not_run_after_fatal"
            and record["execution_policy"] == "model_attempt" for record in records
        ),
        "validation_not_run_after_fatal_count": sum(
            record["record_status"] == "not_run_after_fatal"
            and record["execution_policy"] == "pre_inference_validation_only"
            for record in records
        ),
        "records": records,
    }
    try:
        validate_raw(
            raw,
            plan,
            freeze_receipt,
            accepted_provenance_by_source_item=accepted_provenance_by_source_item,
            ledger_events=ledger.events,
            expected_freeze_bindings=effective_bindings,
        )
        write_canonical_json_exclusive(raw_path, raw)
    except Exception as exc:
        ledger.append(
            "fatal",
            payload={"fatal_code": "raw_seal_failed", "error_type": type(exc).__name__},
        )
        raise
    return raw


_SCORE_ITEM_FIELDS = {
    "record_index", "item_id", "execution_policy", "role", "metric_scope",
    "oracle_item_sha256", "raw_record_status", "score_status", "raw_response_sha256",
    "raw_parse_success", "raw_guard_result", "raw_item_score", "final_guard_result",
    "final_item_score", "fallback_reasons", "error_type", "validation_stage",
    "rejection_code", "fatal_event_sha256", "not_run_reason",
}
_METRIC_SCORE_FIELDS = {
    "failures", "supported_explanation_pass", "supported_assertion_count",
    "unsupported_assertion_count", "asserted_assertion_count", "numeric_tp",
    "numeric_fn_required", "numeric_fp", "correct_abstention",
    "unnecessary_abstention", "required_abstention", "cause_supported",
    "cause_correct", "unsupported_claim", "parse_success", "guard_result",
    "fallback_reasons",
}


def _failure_projection(oracle_item: Mapping[str, Any], code: str) -> dict[str, Any]:
    required_numeric = sum(
        len(atom["numeric_claims"])
        for atom in oracle_item["supportable_statements"]
        if atom["required"]
    )
    return {
        "failures": [code],
        "supported_explanation_pass": False,
        "supported_assertion_count": 0,
        "unsupported_assertion_count": 0,
        "asserted_assertion_count": 0,
        "numeric_tp": 0,
        "numeric_fn_required": required_numeric,
        "numeric_fp": 0,
        "correct_abstention": False,
        "unnecessary_abstention": False,
        "required_abstention": bool(oracle_item["required_abstention"]),
        "cause_supported": bool(oracle_item["cause_supported_by_part_b"]),
        "cause_correct": False,
        "unsupported_claim": False,
        "parse_success": False,
        "guard_result": None,
        "fallback_reasons": [],
    }


def _metric_projection(score: Mapping[str, Any], *, guard_result: str | None,
                       fallback_reasons: Sequence[str] = ()) -> dict[str, Any]:
    projection = {field: copy.deepcopy(score.get(field)) for field in _METRIC_SCORE_FIELDS}
    projection["guard_result"] = guard_result
    projection["fallback_reasons"] = list(fallback_reasons)
    if projection["failures"] is None:
        projection["failures"] = []
    return projection


def reconstruct_score_items(
    plan: Mapping[str, Any],
    raw: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Reconstruct exact score rows from the frozen plan, raw, and oracle."""
    plan_items = list(plan["items"])
    raw_records = list(raw["records"])
    oracle_rows = list(oracle["items"])
    if len(raw_records) != len(plan_items) or len(oracle_rows) != len(plan_items):
        raise ContractError("score reconstruction requires one raw/oracle row per plan item")
    if [row["item_id"] for row in oracle_rows] != [
        item["item_id"] for item in plan_items
    ]:
        raise ContractError("oracle score item order differs from the frozen plan")
    oracle_by_id = {row["item_id"]: row for row in oracle_rows}
    if len(oracle_by_id) != len(oracle_rows):
        raise ContractError("oracle score item IDs must be unique")
    result: list[dict[str, Any]] = []
    for index, (item, record) in enumerate(zip(plan_items, raw_records, strict=True)):
        if record["record_index"] != index or record["item_id"] != item["item_id"]:
            raise ContractError("raw score identity/order differs from plan")
        oracle_item = oracle_by_id.get(item["item_id"])
        if oracle_item is None:
            raise ContractError("oracle is missing a plan item")
        if any(oracle_item[field] != item[field] for field in (
            "source_item_id", "role", "paired_primary_item_id"
        )):
            raise ContractError("oracle score identity differs from plan")
        policy = item["execution_policy"]
        metric_scope = "primary" if item["role"] == "primary" else (
            "validation_only" if policy == "pre_inference_validation_only" else "control"
        )
        common = {
            "record_index": index,
            "item_id": item["item_id"],
            "execution_policy": policy,
            "role": item["role"],
            "metric_scope": metric_scope,
            "oracle_item_sha256": canonical_sha256(oracle_item),
            "raw_record_status": record["record_status"],
            "raw_response_sha256": record.get("response_utf8_sha256"),
            "raw_parse_success": None,
            "raw_guard_result": None,
            "raw_item_score": None,
            "final_guard_result": None,
            "final_item_score": None,
            "fallback_reasons": [],
            "error_type": record.get("error_type"),
            "validation_stage": record.get("validation_stage"),
            "rejection_code": record.get("rejection_code"),
            "fatal_event_sha256": record.get("fatal_event_sha256"),
            "not_run_reason": record.get("not_run_reason"),
        }
        if record["record_status"] == "completed":
            try:
                parsed = parse_response(record["response"])
            except ContractError:
                raw_score = _failure_projection(oracle_item, "parse_failure")
                common["raw_parse_success"] = False
                fallback_raw = template_projection(
                    item,
                    accepted_provenance=accepted_provenance_by_source_item[item["source_item_id"]],
                )["response"]
                guarded = guard_response(
                    record["response"], oracle_item,
                    public_item=item,
                    fallback_raw=fallback_raw,
                )
                final_score = _metric_projection(
                    guarded["score"],
                    guard_result=guarded["guard_result"],
                    fallback_reasons=guarded["limit_codes"],
                )
                common.update({
                    "score_status": "parse_failure",
                    "raw_guard_result": "rejected",
                    "raw_item_score": raw_score,
                    "final_guard_result": guarded["guard_result"],
                    "final_item_score": final_score,
                    "fallback_reasons": list(guarded["limit_codes"]),
                })
            else:
                raw_score = _metric_projection(
                    score_parsed_response(parsed, oracle_item, public_item=item),
                    guard_result=None,
                )
                final_guard = guard_response(
                    record["response"], oracle_item,
                    public_item=item,
                    fallback_raw=template_projection(
                        item,
                        accepted_provenance=accepted_provenance_by_source_item[item["source_item_id"]],
                    )["response"],
                )
                common.update({
                    "score_status": "completed",
                    "raw_parse_success": True,
                    "raw_guard_result": final_guard["guard_result"],
                    "raw_item_score": raw_score,
                    "final_guard_result": final_guard["guard_result"],
                    "final_item_score": _metric_projection(
                        final_guard["score"],
                        guard_result=final_guard["guard_result"],
                        fallback_reasons=final_guard["limit_codes"],
                    ),
                    "fallback_reasons": list(final_guard["limit_codes"]),
                })
        elif record["record_status"] == "inference_error":
            failure = _failure_projection(oracle_item, "inference_error")
            common.update({
                "score_status": "inference_error",
                "raw_response_sha256": None,
                "raw_item_score": failure if metric_scope == "primary" else None,
                "final_item_score": failure if metric_scope == "primary" else None,
            })
        elif record["record_status"] in {"pre_inference_rejected", "pre_inference_error"}:
            common["score_status"] = "validation_only"
            common["raw_response_sha256"] = None
        elif record["record_status"] == "not_run_after_fatal":
            common["score_status"] = "not_run_after_fatal"
            common["raw_response_sha256"] = None
            if metric_scope == "primary":
                failure = _failure_projection(oracle_item, "fatal_run_terminated")
                common["raw_item_score"] = failure
                common["final_item_score"] = copy.deepcopy(failure)
        else:
            raise ContractError("raw score status is invalid")
        checked = {**common}
        if set(checked) != _SCORE_ITEM_FIELDS:
            raise ContractError("score item fields are invalid")
        result.append(checked)
    return result


def build_score_artifact(
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    oracle: Mapping[str, Any],
    raw: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    *,
    accepted_provenance_by_source_item: Mapping[str, object],
    ledger_events: Sequence[Mapping[str, Any]],
    expected_bindings: Mapping[str, object],
    method_id: str,
) -> dict[str, Any]:
    method_class = derive_method_class(freeze_receipt)
    if method_id != freeze_receipt.get("model_repository"):
        raise ContractError("score method ID is spoofed or differs from frozen identity")
    validate_plan(
        plan,
        locators=locators,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    validate_image_locators(locators)
    validate_freeze_receipt(
        freeze_receipt,
        plan,
        locators=locators,
        oracle=oracle,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        expected_bindings=expected_bindings,
    )
    validate_raw(
        raw,
        plan,
        freeze_receipt,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        ledger_events=ledger_events,
        expected_freeze_bindings=expected_bindings,
    )
    items = reconstruct_score_items(
        plan, raw, oracle,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    primary_rows = [row for row in items if row["metric_scope"] == "primary"]
    raw_metric_rows = [{**row["raw_item_score"], "item_id": row["item_id"]}
                       for row in primary_rows]
    final_metric_rows = [{**row["final_item_score"], "item_id": row["item_id"]}
                         for row in primary_rows]
    if method_class == "candidate" and raw.get("run_status") == "fatal_incomplete":
        adoption = {
            "status": "not_adopted",
            "reason": "fatal_incomplete",
            "adopt": False,
        }
    elif method_class == "template":
        adoption = {
            "status": "not_evaluated",
            "reason": "template_baseline_only",
            "fallback_metric_ceiling": "fallback rows reuse deterministic template output",
        }
    else:
        adoption = {
            "status": "not_evaluated",
            "reason": "candidate_adoption_gate_pending",
            "adopt": False,
        }
    score = {
        "schema_version": SCORE_SCHEMA,
        "plan_sha256": canonical_sha256(plan),
        "locator_sha256": canonical_sha256(locators),
        "oracle_sha256": canonical_sha256(oracle),
        "raw_sha256": canonical_sha256(raw),
        "freeze_receipt_sha256": canonical_sha256(freeze_receipt),
        "method_id": method_id,
        "raw_metrics": compute_metrics(raw_metric_rows),
        "final_metrics": compute_metrics(final_metric_rows),
        "grounding": {"status": "qualification_unavailable", "reason": "template_baseline_only"},
        "adoption": adoption,
        "items": items,
    }
    validate_score_artifact(
        score,
        plan=plan,
        locators=locators,
        oracle=oracle,
        raw=raw,
        freeze_receipt=freeze_receipt,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        ledger_events=ledger_events,
        expected_bindings=expected_bindings,
    )
    return score


def validate_score_artifact(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    locators: Mapping[str, Any],
    oracle: Mapping[str, Any],
    raw: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    accepted_provenance_by_source_item: Mapping[str, object],
    ledger_events: Sequence[Mapping[str, Any]],
    expected_bindings: Mapping[str, object],
) -> dict[str, Any]:
    validate_score(
        value,
        plan=plan,
        locators=locators,
        oracle=oracle,
        raw=raw,
        freeze_receipt=freeze_receipt,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
        ledger_events=ledger_events,
        expected_freeze_bindings=expected_bindings,
    )
    expected_items = reconstruct_score_items(
        plan, raw, oracle,
        accepted_provenance_by_source_item=accepted_provenance_by_source_item,
    )
    if value["items"] != expected_items:
        raise ContractError("score items differ from deterministic raw/oracle reconstruction")
    primary_rows = [row for row in expected_items if row["metric_scope"] == "primary"]
    expected_raw_metrics = compute_metrics([
        {**row["raw_item_score"], "item_id": row["item_id"]} for row in primary_rows
    ])
    expected_final_metrics = compute_metrics([
        {**row["final_item_score"], "item_id": row["item_id"]} for row in primary_rows
    ])
    if value["raw_metrics"] != expected_raw_metrics or value["final_metrics"] != expected_final_metrics:
        raise ContractError("score metrics differ from deterministic item projections")
    method_class = derive_method_class(freeze_receipt)
    if value["method_id"] != freeze_receipt.get("model_repository"):
        raise ContractError("score method ID differs from validated frozen identity")
    if method_class == "candidate" and raw.get("run_status") == "fatal_incomplete":
        if value["adoption"] != {
            "status": "not_adopted",
            "reason": "fatal_incomplete",
            "adopt": False,
        }:
            raise ContractError("fatal candidate adoption must be an explicit non-adoption sentinel")
    elif method_class == "template":
        adoption = value["adoption"]
        if adoption.get("status") != "not_evaluated" or adoption.get("reason") != "template_baseline_only":
            raise ContractError("template baseline adoption must be not_evaluated")
    elif value["adoption"].get("status") != "not_evaluated" or value["adoption"].get("adopt") is not False:
        raise ContractError("candidate adoption gate must remain unevaluated")
    return dict(value)

"""Reconstruction-only fatal journal capability.

This module intentionally imports only the standard library.  In particular it has no
oracle, freeze-validator, scorer, image processor, or model capability.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class FatalReconstructionError(ValueError):
    """A sealed journal/sidecar/raw-binding input is invalid."""


_EVENT_FIELDS = {
    "event_index", "previous_event_sha256", "event_type", "item_id", "execution_index",
    "payload_sha256", "recorded_at_utc", "event_sha256",
}
_EVENT_TYPES = {
    "preflight_started", "preflight_completed", "validation_started",
    "validation_terminal", "model_attempt_started", "model_attempt_terminal", "fatal",
}
_RAW_FIELDS = {
    "record_index", "item_id", "execution_policy", "record_status", "attempt_index",
    "validation_index", "model_repository", "model_revision", "model_weight_sha256",
    "prompt_sha256", "ordered_delivered_image_ids", "ordered_delivered_rgb_sha256",
    "generation_config_sha256", "response", "response_utf8_sha256", "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
    "processor_telemetry", "telemetry_unavailable_reason", "error_type", "error",
    "validation_stage", "rejection_code", "validator_receipt_sha256",
    "fatal_event_sha256", "not_run_reason",
}


def _bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FatalReconstructionError("value is not canonical finite JSON") from exc


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _digest(value: object, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise FatalReconstructionError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _events(journal: Path) -> list[dict[str, Any]]:
    result = []
    previous = None
    for index, line in enumerate(journal.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FatalReconstructionError("journal row is not JSON") from exc
        if not isinstance(row, dict) or set(row) != _EVENT_FIELDS:
            raise FatalReconstructionError("journal row fields are invalid")
        if row["event_index"] != index or row["previous_event_sha256"] != previous:
            raise FatalReconstructionError("journal index/hash chain is broken")
        if row["event_type"] not in _EVENT_TYPES:
            raise FatalReconstructionError("journal event type is invalid")
        _digest(row["payload_sha256"], "payload hash")
        _digest(row["event_sha256"], "event hash")
        body = {key: row[key] for key in _EVENT_FIELDS - {"event_sha256"}}
        if _sha(body) != row["event_sha256"]:
            raise FatalReconstructionError("journal event hash is invalid")
        previous = row["event_sha256"]
        result.append(row)
    return result


def _empty(index: int, item: Mapping[str, Any], status: str) -> dict[str, Any]:
    record = {field: None for field in _RAW_FIELDS}
    record.update({
        "record_index": index,
        "item_id": item["item_id"],
        "execution_policy": item["execution_policy"],
        "record_status": status,
    })
    return record


def _publish(path: str | Path, value: object) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FatalReconstructionError("refusing to overwrite fatal raw output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        output.chmod(0o444)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def finalize_fatal_reopen(
    ledger_path: str | Path,
    *,
    items: Sequence[Mapping[str, Any]],
    raw_bindings: Mapping[str, Any],
    fatal_event_sha256: str,
    raw_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reconstruct terminal raw rows using only the journal, sidecars, and bindings."""
    journal = Path(ledger_path)
    sidecars = journal.with_name(journal.name + ".payloads")
    if not journal.is_file() or not sidecars.is_dir():
        raise FatalReconstructionError("fatal reopen requires journal and sidecars")
    events = _events(journal)
    _digest(fatal_event_sha256, "fatal event hash")
    positions = [
        index for index, event in enumerate(events)
        if event["event_type"] == "fatal" and event["event_sha256"] == fatal_event_sha256
    ]
    if positions != [len(events) - 1]:
        raise FatalReconstructionError("one bound terminal fatal event is required")
    required_bindings = {
        "plan_sha256", "freeze_receipt_sha256", "model_repository", "model_revision",
        "model_weight_sha256", "generation_config_sha256",
    }
    if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != required_bindings:
        raise FatalReconstructionError("fatal raw binding fields are invalid")
    for field in (
        "plan_sha256", "freeze_receipt_sha256", "model_weight_sha256",
        "generation_config_sha256",
    ):
        _digest(raw_bindings[field], field)
    payloads = {}
    for event in events:
        sidecar = sidecars / f"{event['payload_sha256']}.json"
        if not sidecar.is_file():
            raise FatalReconstructionError("ledger payload sidecar is missing")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if _sha(payload) != event["payload_sha256"]:
            raise FatalReconstructionError("ledger payload sidecar hash mismatch")
        payloads[event["payload_sha256"]] = payload
    starts = {}
    terminals = {}
    for event in events[:-1]:
        item_id = event["item_id"]
        if event["event_type"] in {"validation_started", "model_attempt_started"}:
            if item_id in starts or item_id in terminals:
                raise FatalReconstructionError("journal retries an item")
            starts[item_id] = event
        elif event["event_type"] in {"validation_terminal", "model_attempt_terminal"}:
            if item_id not in starts or item_id in terminals:
                raise FatalReconstructionError("terminal lacks exactly one start")
            terminals[item_id] = event
    records = []
    for index, item in enumerate(items):
        item_id = item.get("item_id")
        policy = item.get("execution_policy")
        if not isinstance(item_id, str) or policy not in {
            "model_attempt", "pre_inference_validation_only",
        }:
            raise FatalReconstructionError("plan item identity/policy is invalid")
        terminal = terminals.get(item_id)
        start = starts.get(item_id)
        if terminal is not None:
            record = payloads[terminal["payload_sha256"]]
            if not isinstance(record, dict) or set(record) != _RAW_FIELDS:
                raise FatalReconstructionError("terminal sidecar is not an exact raw record")
            if record["record_index"] != index or record["item_id"] != item_id:
                raise FatalReconstructionError("terminal raw identity differs from plan")
            records.append(copy.deepcopy(record))
        elif start is not None and policy == "model_attempt" and start["event_type"] == "model_attempt_started":
            record = _empty(index, item, "inference_error")
            record.update({
                "attempt_index": start["execution_index"],
                "model_repository": raw_bindings["model_repository"],
                "model_revision": raw_bindings["model_revision"],
                "model_weight_sha256": raw_bindings["model_weight_sha256"],
                "prompt_sha256": item.get("prompt_sha256"),
                "ordered_delivered_image_ids": [row["delivered_image_id"] for row in item.get("image_deliveries", [])],
                "ordered_delivered_rgb_sha256": [row["delivered_rgb_sha256"] for row in item.get("image_deliveries", [])],
                "generation_config_sha256": raw_bindings["generation_config_sha256"],
                "error_type": "process_crash_unmatched_model_attempt_start",
                "error": "process ended after model_attempt_started without a terminal event",
                "telemetry_unavailable_reason": "process_crash_before_terminal_measurement",
            })
            records.append(record)
        elif start is not None and policy == "pre_inference_validation_only" and start["event_type"] == "validation_started":
            record = _empty(index, item, "pre_inference_error")
            record.update({
                "validation_index": start["execution_index"],
                "error_type": "process_crash_unmatched_validation_start",
                "error": "process ended after validation_started without a terminal event",
            })
            records.append(record)
        elif start is not None:
            raise FatalReconstructionError("journal start contradicts item policy")
        else:
            record = _empty(index, item, "not_run_after_fatal")
            record.update({
                "fatal_event_sha256": fatal_event_sha256,
                "not_run_reason": "fatal_run_terminated",
            })
            records.append(record)
    counts = Counter(record["record_status"] for record in records)
    raw = {
        "schema_version": "rca-bench-raw-v1",
        **dict(raw_bindings),
        "journal_terminal_sha256": events[-1]["event_sha256"],
        "run_status": "fatal_incomplete",
        "planned_item_count": len(records),
        "model_attempt_planned_count": sum(item["execution_policy"] == "model_attempt" for item in items),
        "pre_inference_validation_planned_count": sum(item["execution_policy"] == "pre_inference_validation_only" for item in items),
        "completed_count": counts["completed"],
        "inference_error_count": counts["inference_error"],
        "pre_inference_rejected_count": counts["pre_inference_rejected"],
        "pre_inference_error_count": counts["pre_inference_error"],
        "not_run_after_fatal_count": counts["not_run_after_fatal"],
        "model_not_run_after_fatal_count": sum(record["record_status"] == "not_run_after_fatal" and record["execution_policy"] == "model_attempt" for record in records),
        "validation_not_run_after_fatal_count": sum(record["record_status"] == "not_run_after_fatal" and record["execution_policy"] == "pre_inference_validation_only" for record in records),
        "records": records,
    }
    if raw_path is not None:
        _publish(raw_path, raw)
    return raw

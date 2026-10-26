"""Publish accepted per-vial F2 receipts into downstream stage manifests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from types import ModuleType

from fluidmix.deploy_input_policy import validate_deploy_input_fields
from fluidmix.paint_ball_emitter import (
    EMITTER_RENDER_PATH,
    F2_REQUIRED_ACCEPTANCE_GATES,
    PAINT_BALL_RECEIPT_SCHEMA_VERSION,
    build_f2_runtime_acceptance_receipt,
    validate_paint_ball_receipt,
)
from fluidmix.render_process import hash_regular_file, load_strict_json_snapshot
from fluidmix.twin_video import _atomic_rename_noreplace, _write_private_json


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TANKS = ("t100", "t200", "t400")
_VIALS = ("V1", "V2", "V3")
_SPLITS = frozenset(("train", "val", "test", "buffer"))
_REQUEST_KEYS = {
    "schema_version",
    "artifact_type",
    "data_origin",
    "provider",
    "runtime_name",
    "runtime_id",
    "family_id",
    "batch_id",
    "run_id",
    "cohort",
    "current_recipe_id",
    "previous_recipe_id",
    "split_role",
    "split_version",
    "programmed_recipe",
    "vials",
}
_PROGRAMMED_KEYS = {
    "dispense_order",
    "previous_ml_by_tank",
    "current_ml_by_tank",
}
_VIAL_KEYS = {"vial_id", "receipt_path", "receipt_sha256"}
_ARTIFACT_VALIDATOR_MODULE: ModuleType | None = None


class F2ReceiptPublicationError(ValueError):
    """The per-vial receipt set could not be published safely."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise F2ReceiptPublicationError(f"{field}_invalid")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2ReceiptPublicationError(f"{field}_invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise F2ReceiptPublicationError(f"{field}_invalid")
    return result


def _load_request(value: object) -> tuple[Mapping[str, object], Path, str]:
    if isinstance(value, Mapping):
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return value, Path.cwd().resolve(), hashlib.sha256(encoded).hexdigest()
    if not isinstance(value, (str, os.PathLike)):
        raise F2ReceiptPublicationError("publish_request_invalid")
    path = Path(value).expanduser().absolute()
    try:
        payload, snapshot = load_strict_json_snapshot(path)
    except (OSError, ValueError) as error:
        raise F2ReceiptPublicationError("publish_request_invalid") from error
    if not isinstance(payload, Mapping):
        raise F2ReceiptPublicationError("publish_request_invalid")
    return payload, path.parent.resolve(), str(snapshot["sha256"])


def _programmed_recipe(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PROGRAMMED_KEYS:
        raise F2ReceiptPublicationError("programmed_recipe_invalid")
    order = value.get("dispense_order")
    if (
        not isinstance(order, list)
        or len(order) != 3
        or tuple(sorted(order)) != _TANKS
    ):
        raise F2ReceiptPublicationError("programmed_dispense_order_invalid")
    result: dict[str, object] = {"dispense_order": list(order)}
    for key in ("previous_ml_by_tank", "current_ml_by_tank"):
        rows = value.get(key)
        if not isinstance(rows, Mapping) or set(rows) != set(_TANKS):
            raise F2ReceiptPublicationError(f"{key}_invalid")
        result[key] = {
            tank: _positive_number(rows[tank], f"{key}.{tank}") for tank in _TANKS
        }
    return result


def _validate_request(payload: Mapping[str, object]) -> dict[str, object]:
    if set(payload) != _REQUEST_KEYS:
        raise F2ReceiptPublicationError("publish_request_schema_invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "f2_stage_batch_publish_request"
        or payload.get("data_origin") != "twin"
        or payload.get("provider") != "brev"
        or payload.get("runtime_name") != "NVIDIA Isaac Sim"
        or payload.get("split_version") != "inn-7-splits-v2"
        or payload.get("cohort") not in {"reference_only", "scored_target"}
        or payload.get("split_role") not in _SPLITS
    ):
        raise F2ReceiptPublicationError("publish_request_schema_invalid")
    previous = payload.get("previous_recipe_id")
    if previous is not None:
        previous = _text(previous, "previous_recipe_id")
    if payload.get("cohort") == "scored_target" and previous is None:
        raise F2ReceiptPublicationError("target_previous_recipe_missing")
    vials = payload.get("vials")
    if not isinstance(vials, list) or len(vials) != 3:
        raise F2ReceiptPublicationError("vial_receipts_invalid")
    normalized_vials: list[dict[str, str]] = []
    for raw in vials:
        if not isinstance(raw, Mapping) or set(raw) != _VIAL_KEYS:
            raise F2ReceiptPublicationError("vial_receipt_schema_invalid")
        vial_id = _text(raw.get("vial_id"), "vial_id")
        path = _text(raw.get("receipt_path"), "receipt_path")
        sha = raw.get("receipt_sha256")
        if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
            raise F2ReceiptPublicationError("receipt_sha256_invalid")
        normalized_vials.append(
            {"vial_id": vial_id, "receipt_path": path, "receipt_sha256": sha}
        )
    if sorted(row["vial_id"] for row in normalized_vials) != list(_VIALS):
        raise F2ReceiptPublicationError("vial_ids_invalid")
    if (
        len({row["receipt_path"] for row in normalized_vials}) != 3
        or len({row["receipt_sha256"] for row in normalized_vials}) != 3
    ):
        raise F2ReceiptPublicationError("vial_receipts_not_distinct")
    return {
        "runtime_id": _text(payload.get("runtime_id"), "runtime_id"),
        "family_id": _text(payload.get("family_id"), "family_id"),
        "batch_id": _text(payload.get("batch_id"), "batch_id"),
        "run_id": _text(payload.get("run_id"), "run_id"),
        "cohort": str(payload["cohort"]),
        "current_recipe_id": _text(payload.get("current_recipe_id"), "current_recipe_id"),
        "previous_recipe_id": previous,
        "split_role": str(payload["split_role"]),
        "programmed_recipe": _programmed_recipe(payload.get("programmed_recipe")),
        "vials": sorted(normalized_vials, key=lambda row: row["vial_id"]),
    }


def _load_artifact_validator_module() -> ModuleType:
    global _ARTIFACT_VALIDATOR_MODULE
    if _ARTIFACT_VALIDATOR_MODULE is not None:
        return _ARTIFACT_VALIDATOR_MODULE
    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_physx_f2_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "_fluidmix_validate_physx_f2_artifacts", script
    )
    if spec is None or spec.loader is None:
        raise F2ReceiptPublicationError("artifact_validator_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ARTIFACT_VALIDATOR_MODULE = module
    return module


def _official_artifact_validator(app_output: Path) -> Mapping[str, object]:
    result = _load_artifact_validator_module().validate_output(app_output)
    if not isinstance(result, Mapping):
        raise F2ReceiptPublicationError("artifact_validation_invalid")
    return result


def _resolved_receipt_path(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise F2ReceiptPublicationError("receipt_missing") from error
    if resolved != path or path.name != "receipt.json" or not path.is_file():
        raise F2ReceiptPublicationError("receipt_path_invalid")
    return path


def _normalized_model_record(receipt: Mapping[str, object]) -> dict[str, object]:
    plan = receipt.get("plan")
    plan = plan if isinstance(plan, Mapping) else {}
    execution = plan.get("execution_schedule")
    execution = execution if isinstance(execution, Mapping) else {}
    model = execution.get("model_input_record")
    binding = execution.get("renderer_case_binding")
    if not isinstance(model, Mapping):
        residual = receipt.get("recipe8_leading_residual")
        residual = residual if isinstance(residual, Mapping) else {}
        model = residual.get("model_input_record")
        binding = None
    if not isinstance(model, Mapping):
        raise F2ReceiptPublicationError("receipt_model_input_record_missing")
    return {
        "current_recipe_id": model.get(
            "current_recipe_id", model.get("current_sim_recipe_id")
        ),
        "previous_recipe_id": model.get(
            "previous_recipe_id", model.get("previous_sim_recipe_id")
        ),
        "dispense_order": model.get(
            "dispense_order", model.get("programmed_dispense_order")
        ),
        "current_ml_by_tank": model.get(
            "programmed_ml", model.get("current_programmed_ml")
        ),
        "previous_ml_by_tank": model.get("previous_programmed_ml"),
        "family_id": model.get("family_id"),
        "split_role": model.get("split_role"),
        "split_version": model.get("split_version"),
        "data_origin": model.get("data_origin"),
        "run_id": model.get("run_id"),
        "batch_id": binding.get("batch_id") if isinstance(binding, Mapping) else None,
        "vial_ordinal": (
            binding.get("vial_ordinal") if isinstance(binding, Mapping) else None
        ),
    }


def _same_number_map(left: object, right: Mapping[str, object]) -> bool:
    if not isinstance(left, Mapping) or set(left) != set(_TANKS):
        return False
    return all(
        not isinstance(left[tank], bool)
        and isinstance(left[tank], (int, float))
        and math.isclose(
            float(left[tank]), float(right[tank]), rel_tol=0.0, abs_tol=1e-12
        )
        for tank in _TANKS
    )


def _verify_model_record(
    record: Mapping[str, object], request: Mapping[str, object], *, vial_ordinal: int
) -> None:
    recipe = request["programmed_recipe"]
    assert isinstance(recipe, Mapping)
    checks = (
        record.get("current_recipe_id") == request["current_recipe_id"],
        record.get("previous_recipe_id") == request["previous_recipe_id"],
        record.get("dispense_order") == recipe["dispense_order"],
        _same_number_map(
            record.get("current_ml_by_tank"), recipe["current_ml_by_tank"]
        ),
        record.get("family_id") in {None, request["family_id"]},
        record.get("split_role") in {None, request["split_role"]},
        record.get("split_version") in {None, "inn-7-splits-v2"},
        record.get("data_origin") in {None, "twin"},
        record.get("run_id") in {None, request["run_id"]},
        record.get("batch_id") in {None, request["batch_id"]},
        record.get("vial_ordinal") in {None, vial_ordinal},
    )
    previous = record.get("previous_ml_by_tank")
    if previous is not None:
        checks = (*checks, _same_number_map(previous, recipe["previous_ml_by_tank"]))
    if not all(checks):
        raise F2ReceiptPublicationError("receipt_context_mismatch")


def _artifact_locator(
    root: Path, pair: Mapping[str, object], prefix: str
) -> dict[str, str]:
    path_value = pair.get(f"{prefix}_path")
    expected = pair.get(f"{prefix}_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise F2ReceiptPublicationError(f"stage_{prefix}_path_invalid")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise F2ReceiptPublicationError(f"stage_{prefix}_hash_invalid")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    path = path.absolute()
    if path.suffix.lower() != ".png":
        raise F2ReceiptPublicationError(f"stage_{prefix}_not_png")
    try:
        resolved = path.resolve(strict=True)
        snapshot = hash_regular_file(path)
    except (OSError, ValueError) as error:
        raise F2ReceiptPublicationError(f"stage_{prefix}_missing") from error
    if resolved != path or snapshot["sha256"] != expected:
        raise F2ReceiptPublicationError(f"stage_{prefix}_hash_mismatch")
    return {"path": str(path), "sha256": expected}


def _extract_receipt(
    item: Mapping[str, str],
    *,
    base: Path,
    request: Mapping[str, object],
    vial_ordinal: int,
    artifact_validator: Callable[[Path], Mapping[str, object]],
) -> dict[str, object]:
    path = _resolved_receipt_path(item["receipt_path"], base)
    try:
        receipt_value, snapshot = load_strict_json_snapshot(
            path, max_bytes=64 * 1024 * 1024
        )
    except (OSError, ValueError) as error:
        raise F2ReceiptPublicationError("receipt_unreadable") from error
    if snapshot["sha256"] != item["receipt_sha256"]:
        raise F2ReceiptPublicationError("receipt_hash_mismatch")
    if not isinstance(receipt_value, Mapping):
        raise F2ReceiptPublicationError("receipt_malformed")
    receipt = receipt_value
    qualification = receipt.get("qualification")
    if (
        receipt.get("schema_version") != PAINT_BALL_RECEIPT_SCHEMA_VERSION
        or receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter"
        or receipt.get("render_path") != EMITTER_RENDER_PATH
        or receipt.get("status") != "pass"
        or not isinstance(qualification, Mapping)
        or qualification.get("first_stage_only") is not False
    ):
        raise F2ReceiptPublicationError("receipt_not_accepted_full_run")
    try:
        receipt_failures = tuple(validate_paint_ball_receipt(receipt))
    except Exception as error:
        raise F2ReceiptPublicationError("receipt_substantive_validation_error") from error
    if receipt_failures:
        raise F2ReceiptPublicationError(
            "receipt_substantive_validation_failed:" + ",".join(receipt_failures)
        )
    try:
        validation = artifact_validator(path.parent)
    except Exception as error:
        raise F2ReceiptPublicationError("receipt_artifact_validation_error") from error
    if (
        validation.get("status") != "pass"
        or validation.get("failures") != []
        or isinstance(validation.get("verified_artifact_count"), bool)
        or not isinstance(validation.get("verified_artifact_count"), int)
        or validation["verified_artifact_count"] <= 0
    ):
        raise F2ReceiptPublicationError("receipt_artifact_validation_failed")

    versions = receipt.get("versions")
    provenance = receipt.get("provenance")
    render = receipt.get("render")
    semantics = receipt.get("semantics")
    timing = receipt.get("timing")
    plan = receipt.get("plan")
    if not all(
        isinstance(value, Mapping)
        for value in (versions, provenance, render, semantics, timing, plan)
    ):
        raise F2ReceiptPublicationError("receipt_contract_missing")
    if versions.get("isaac") != "6.2.0.0":
        raise F2ReceiptPublicationError("receipt_runtime_mismatch")
    script_sha = provenance.get("script_sha256")
    stage_sha = provenance.get("requested_stage_sha256")
    if (
        not isinstance(script_sha, str)
        or _SHA256.fullmatch(script_sha) is None
        or not isinstance(stage_sha, str)
        or _SHA256.fullmatch(stage_sha) is None
    ):
        raise F2ReceiptPublicationError("receipt_source_hash_invalid")
    _verify_model_record(
        _normalized_model_record(receipt), request, vial_ordinal=vial_ordinal
    )

    steps = render.get("stage_frame_physics_steps")
    stage_hashes = render.get("stage_frame_sha256")
    pours = plan.get("pours")
    pairs = semantics.get("pairs")
    physics_hz = timing.get("physics_hz")
    if (
        not isinstance(steps, list)
        or len(steps) != 3
        or not isinstance(stage_hashes, list)
        or len(stage_hashes) != 3
        or not isinstance(pours, list)
        or len(pours) != 3
        or not isinstance(pairs, list)
        or isinstance(physics_hz, bool)
        or not isinstance(physics_hz, int)
        or physics_hz <= 0
    ):
        raise F2ReceiptPublicationError("receipt_stage_contract_invalid")
    pair_by_step: dict[int, Mapping[str, object]] = {}
    for pair in pairs:
        if isinstance(pair, Mapping) and isinstance(pair.get("physics_step"), int):
            step = int(pair["physics_step"])
            if step in pair_by_step:
                raise F2ReceiptPublicationError("receipt_stage_pair_duplicate")
            pair_by_step[step] = pair
    events: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for ordinal, (raw_step, raw_pour, stage_hash) in enumerate(
        zip(steps, pours, stage_hashes, strict=True), start=1
    ):
        if (
            isinstance(raw_step, bool)
            or not isinstance(raw_step, int)
            or not isinstance(raw_pour, Mapping)
            or raw_pour.get("ordinal") != ordinal
            or raw_pour.get("tank_id") not in _TANKS
            or raw_pour.get("stage_capture_step") != raw_step
            or isinstance(raw_pour.get("end_step"), bool)
            or not isinstance(raw_pour.get("end_step"), int)
            or raw_pour["end_step"] > raw_step
        ):
            raise F2ReceiptPublicationError("receipt_stage_contract_invalid")
        tank = str(raw_pour["tank_id"])
        pair = pair_by_step.get(raw_step)
        if pair is None or pair.get("rgb_sha256") != stage_hash:
            raise F2ReceiptPublicationError("receipt_stage_pair_missing")
        rgb = _artifact_locator(path.parent, pair, "rgb")
        mask = _artifact_locator(path.parent, pair, "mask")
        event = {
            "event_id": f"dispense-{ordinal}-{tank}",
            "stage_ordinal": ordinal,
            "tank_id": tank,
            "tank_order": ordinal,
            "physics_step": raw_step,
            "frame_id": Path(rgb["path"]).stem,
            "event_relative_timestamp_s": raw_step / physics_hz,
            "final_observation_offset_s": (
                raw_step - raw_pour["end_step"]
            ) / physics_hz,
        }
        events.append(event)
        rows.append(
            {
                "vial_id": item["vial_id"],
                **{
                    key: event[key]
                    for key in event
                    if key not in {"tank_id", "tank_order"}
                },
                "view_id": "vial_fill",
                "verified": True,
                "rgb": rgb,
                "liquid_mask": mask,
            }
        )
    recipe = request["programmed_recipe"]
    assert isinstance(recipe, Mapping)
    if [event["tank_id"] for event in events] != recipe["dispense_order"]:
        raise F2ReceiptPublicationError("receipt_stage_order_mismatch")
    return {
        "vial_id": item["vial_id"],
        "receipt_path": str(path),
        "receipt_sha256": item["receipt_sha256"],
        "verified_artifact_count": validation["verified_artifact_count"],
        "script_sha256": script_sha,
        "stage_sha256": stage_sha,
        "isaac_version": versions["isaac"],
        "physx_version": versions.get("physx"),
        "events": events,
        "rows": rows,
    }


def _write_publication(
    output: Path,
    *,
    runtime_receipt: Mapping[str, object],
    manifest: Mapping[str, object],
    context: Mapping[str, object],
    source_receipts: list[Mapping[str, object]],
    request_sha256: str,
) -> Path:
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if not output.name or output.name in {".", ".."}:
        raise F2ReceiptPublicationError("output_directory_invalid")
    if os.path.lexists(output):
        raise FileExistsError(output)
    partial: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent)
    )
    os.chmod(partial, 0o700)
    try:
        values = {
            "f2-runtime-acceptance-receipt.json": runtime_receipt,
            "f2-stage-capture-manifest.json": manifest,
            "stage-batch-context.json": context,
        }
        outputs: dict[str, dict[str, object]] = {}
        for name, value in values.items():
            path = partial / name
            _write_private_json(path, value)
            outputs[name] = {"path": name, **hash_regular_file(path)}
        from fluidmix.capture_manifest_adapter import bind_stage_capture_manifest

        bound = bind_stage_capture_manifest(partial / "f2-stage-capture-manifest.json")
        if bound.get("denominators") != {
            "image_measurements": 9,
            "stage_statistics": 9,
        }:
            raise F2ReceiptPublicationError("published_manifest_binding_failed")
        record = {
            "schema_version": 1,
            "artifact_type": "f2_stage_batch_publication_receipt",
            "status": "published",
            "data_origin": "twin",
            "real_data_used": False,
            "source_publish_request_sha256": request_sha256,
            "source_receipts": source_receipts,
            "outputs": outputs,
            "claim_boundary": (
                "synthetic artifact routing only; no physical accuracy or sim-to-real claim"
            ),
        }
        _write_private_json(partial / "publication-receipt.json", record)
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
        return output / "publication-receipt.json"
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == parent:
                shutil.rmtree(partial)


def publish_f2_stage_batch(
    request: object,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Publish one three-vial synthetic batch after fail-closed validation."""

    payload, request_base, request_sha = _load_request(request)
    try:
        validate_deploy_input_fields(payload, require_allowlist=False)
    except ValueError as error:
        raise F2ReceiptPublicationError("forbidden_oracle_field") from error
    normalized = _validate_request(payload)
    source_receipts = [
        _extract_receipt(
            item,
            base=request_base,
            request=normalized,
            vial_ordinal=ordinal,
            artifact_validator=_official_artifact_validator,
        )
        for ordinal, item in enumerate(normalized["vials"], start=1)
    ]
    shared_fields = {
        (
            row["script_sha256"],
            row["stage_sha256"],
            row["isaac_version"],
            row["physx_version"],
            json.dumps(row["events"], sort_keys=True),
        )
        for row in source_receipts
    }
    if len(shared_fields) != 1:
        raise F2ReceiptPublicationError("vial_receipt_contract_mismatch")
    runtime_binding = {
        "provider": "brev",
        "runtime_name": "NVIDIA Isaac Sim",
        "isaac_sim_version": "6.2.0.0",
        "runtime_id": normalized["runtime_id"],
        "source_sha256": source_receipts[0]["script_sha256"],
    }
    aggregate_validation = {
        "status": "pass",
        "verified_artifact_count": sum(
            int(row["verified_artifact_count"]) for row in source_receipts
        ),
        "failures": [],
    }
    runtime_receipt = build_f2_runtime_acceptance_receipt(
        runtime_binding=runtime_binding,
        artifact_validation=aggregate_validation,
        gate_evidence={name: True for name in F2_REQUIRED_ACCEPTANCE_GATES},
    )
    runtime_bytes = (
        json.dumps(runtime_receipt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    events = source_receipts[0]["events"]
    manifest = {
        "schema_version": 2,
        "artifact_type": "f2_stage_capture_manifest",
        "status": "accepted",
        "data_origin": "twin",
        "family_id": normalized["family_id"],
        "batch_id": normalized["batch_id"],
        "run_id": normalized["run_id"],
        "recipe_id": normalized["current_recipe_id"],
        "split_role": normalized["split_role"],
        "split_version": "inn-7-splits-v2",
        "view_id": "vial_fill",
        "runtime_binding": runtime_binding,
        "runtime_receipt": {
            "path": "f2-runtime-acceptance-receipt.json",
            "sha256": runtime_sha,
        },
        "stage_events": events,
        "rows": [row for receipt in source_receipts for row in receipt["rows"]],
    }
    context = {
        "schema_version": 1,
        "artifact_type": "stage_batch_context",
        "split_version": "inn-7-splits-v2",
        "batches": [
            {
                "batch_id": normalized["batch_id"],
                "cohort": normalized["cohort"],
                "family_id": normalized["family_id"],
                "current_recipe_id": normalized["current_recipe_id"],
                "previous_recipe_id": normalized["previous_recipe_id"],
                "run_id": normalized["run_id"],
                "split_role": normalized["split_role"],
                "programmed_recipe": normalized["programmed_recipe"],
            }
        ],
    }
    try:
        validate_deploy_input_fields(manifest, require_allowlist=False)
        validate_deploy_input_fields(context)
    except ValueError as error:
        raise F2ReceiptPublicationError("forbidden_oracle_field") from error
    provenance = [
        {
            key: row[key]
            for key in (
                "vial_id",
                "receipt_path",
                "receipt_sha256",
                "verified_artifact_count",
            )
        }
        for row in source_receipts
    ]
    return _write_publication(
        Path(output_dir).expanduser(),
        runtime_receipt=runtime_receipt,
        manifest=manifest,
        context=context,
        source_receipts=provenance,
        request_sha256=request_sha,
    )


__all__ = ["F2ReceiptPublicationError", "publish_f2_stage_batch"]

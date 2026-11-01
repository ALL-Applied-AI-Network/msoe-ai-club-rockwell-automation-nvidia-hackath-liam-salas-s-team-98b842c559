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

from fluidmix.core.policy import validate_deploy_input_fields
from twin.physx.paint_ball_emitter import (
    COUNT_DRIVEN_RENDER_PATH,
    EMITTER_RENDER_PATH,
    F2_REQUIRED_ACCEPTANCE_GATES,
    PAINT_BALL_RECEIPT_SCHEMA_VERSION,
    build_f2_runtime_acceptance_receipt,
    validate_paint_ball_receipt,
)
from twin.render.render_process import hash_regular_file, load_strict_json_snapshot
from twin.cohorts.varied_cohort_manifest_v4 import (
    build_varied_cohort_artifacts as build_varied_cohort_artifacts_v4,
)
from twin.render.twin_video import _atomic_rename_noreplace, _write_private_json


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"CASE_[0-9a-f]{16}\Z")
_TANKS = ("t100", "t200", "t400")
_VIALS = ("V1", "V2", "V3")
_SPLITS = frozenset(("train", "val", "test", "buffer"))
_LEGACY_RESOLUTION_PROFILE = "c20_2p5mm"
_LEGACY_EFFECTIVE_PARTICLE_VOLUME_ML = 0.011272866937444935
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
_VARIED_REQUEST_KEYS = {
    "schema_version",
    "artifact_type",
    "data_origin",
    "provider",
    "runtime_name",
    "runtime_id",
    "batch_id",
    "run_id",
    "cohort",
    "current_recipe_id",
    "split_role",
    "split_version",
    "programmed_recipe",
    "vials",
}
_VARIED_V4_COHORT_BINDING_KEYS = {
    "source_render_plan_sha256",
    "cohort_launch_receipt",
    "cohort_merge_receipt",
}
_ARTIFACT_LOCATOR_KEYS = {"path", "sha256"}
_VARIED_PROGRAMMED_KEYS = {"dispense_order", "current_ml_by_tank"}
_VARIED_VIAL_KEYS = {
    "case_id",
    "vial_id",
    "vial_position",
    "previous_recipe_id",
    "previous_ml_by_tank",
    "receipt_path",
    "receipt_sha256",
}
_VARIED_SPLITS = frozenset(("train", "validation", "evaluation"))
_VARIED_SPLIT_VERSION = "whole-run-varied-v3"
_VARIED_CONTRACTS: dict[str, dict[str, object]] = {
    "whole-run-varied-v3": {"schema_version": 3},
    "whole-run-varied-v4": {"schema_version": 4},
}
_FIXED_DISPENSE_ORDER = ("t200", "t400", "t100")
_VARIED_RUNTIME_HARD_GATES = (
    F2_REQUIRED_ACCEPTANCE_GATES - {"volume_height"}
) | {"rendered_body_volume_height"}
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


def _nonnegative_recipe_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2ReceiptPublicationError(f"{field}_invalid")
    try:
        result = float(value)
    except OverflowError as error:
        raise F2ReceiptPublicationError(f"{field}_invalid") from error
    if not math.isfinite(result) or result < 0:
        raise F2ReceiptPublicationError(f"{field}_invalid")
    return result


def _two_clock_binding(timing: Mapping[str, object]) -> dict[str, int]:
    """Bind the qualified one-PhysX-step-per-report clock."""

    fields = {
        name: timing.get(name)
        for name in (
            "report_hz",
            "report_interval_count",
            "executed_report_intervals",
            "physics_hz",
            "physics_steps_per_second",
            "physics_substeps_per_report_interval",
            "physics_callback_count",
            "callbacks_expected",
            "callbacks_observed",
            "diffusion_updates_per_report_interval",
            "diffusion_update_count",
        )
    }
    if any(type(value) is not int for value in fields.values()):
        raise F2ReceiptPublicationError("receipt_two_clock_contract_invalid")
    report_count = fields["report_interval_count"]
    callback_count = report_count
    if (
        fields["report_hz"] != 120
        or report_count <= 0
        or fields["executed_report_intervals"] != report_count
        or fields["physics_hz"] != 120
        or fields["physics_steps_per_second"] != 120
        or fields["physics_substeps_per_report_interval"] != 1
        or fields["physics_callback_count"] != callback_count
        or fields["callbacks_expected"] != callback_count
        or fields["callbacks_observed"] != callback_count
        or fields["diffusion_updates_per_report_interval"] != 1
        or fields["diffusion_update_count"] != report_count
    ):
        raise F2ReceiptPublicationError("receipt_two_clock_contract_invalid")
    return {name: int(value) for name, value in fields.items()}


def _legacy_runtime_binding(arguments: object) -> None:
    """Reject C21 or another unqualified particle contract at publication."""

    if not isinstance(arguments, Mapping):
        raise F2ReceiptPublicationError("receipt_legacy_runtime_contract_invalid")
    effective_volume = arguments.get("recipe8_effective_particle_volume_ml")
    if (
        arguments.get("physics_substeps") != 1
        or arguments.get("particle_resolution_profile")
        != _LEGACY_RESOLUTION_PROFILE
        or isinstance(effective_volume, bool)
        or not isinstance(effective_volume, (int, float))
        or not math.isclose(
            float(effective_volume),
            _LEGACY_EFFECTIVE_PARTICLE_VOLUME_ML,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        )
    ):
        raise F2ReceiptPublicationError("receipt_legacy_runtime_contract_invalid")


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
        normalized = {
            tank: _nonnegative_recipe_number(rows[tank], f"{key}.{tank}")
            for tank in _TANKS
        }
        if sum(normalized.values()) <= 0:
            raise F2ReceiptPublicationError(f"{key}_total_invalid")
        result[key] = normalized
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


def _varied_programmed_recipe(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _VARIED_PROGRAMMED_KEYS:
        raise F2ReceiptPublicationError("programmed_recipe_invalid")
    rows = value.get("current_ml_by_tank")
    if not isinstance(rows, Mapping) or set(rows) != set(_TANKS):
        raise F2ReceiptPublicationError("current_ml_by_tank_invalid")
    current = {
        tank: _nonnegative_recipe_number(
            rows[tank], f"current_ml_by_tank.{tank}"
        )
        for tank in _TANKS
    }
    if sum(current.values()) <= 0:
        raise F2ReceiptPublicationError("current_ml_by_tank_total_invalid")
    expected_order = [tank for tank in _FIXED_DISPENSE_ORDER if current[tank] > 0]
    order = value.get("dispense_order")
    if not isinstance(order, list) or order != expected_order or len(order) not in {2, 3}:
        raise F2ReceiptPublicationError("programmed_dispense_order_invalid")
    return {"dispense_order": list(order), "current_ml_by_tank": current}


def _varied_previous_ml(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_TANKS):
        raise F2ReceiptPublicationError("previous_ml_by_tank_invalid")
    result = {
        tank: _nonnegative_recipe_number(
            value[tank], f"previous_ml_by_tank.{tank}"
        )
        for tank in _TANKS
    }
    if sum(result.values()) <= 0:
        raise F2ReceiptPublicationError("previous_ml_by_tank_total_invalid")
    return result


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _v4_source_contract() -> dict[str, object]:
    artifacts = build_varied_cohort_artifacts_v4()
    render_plan = artifacts["render-schedule-plan.json"]
    model_plan = artifacts["model-input-plan.json"]
    case_ids = [
        schedule["case_binding"]["case_id"]
        for schedule in render_plan["execution_schedules"]
    ]
    return {
        "render_plan_sha256": _canonical_sha256(render_plan),
        "case_ids": case_ids,
        "model_records": model_plan["vial_records"],
    }


def _normalize_v4_cohort_binding(payload: Mapping[str, object]) -> dict[str, object]:
    source_sha = payload.get("source_render_plan_sha256")
    if not isinstance(source_sha, str) or _SHA256.fullmatch(source_sha) is None:
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
    result: dict[str, object] = {"source_render_plan_sha256": source_sha}
    for field in ("cohort_launch_receipt", "cohort_merge_receipt"):
        locator = payload.get(field)
        if not isinstance(locator, Mapping) or set(locator) != _ARTIFACT_LOCATOR_KEYS:
            raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
        path = locator.get("path")
        sha = locator.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(sha, str)
            or _SHA256.fullmatch(sha) is None
        ):
            raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
        result[field] = {"path": path, "sha256": sha}
    return result


def _validate_v4_request_source_binding(request: Mapping[str, object]) -> None:
    expected = _v4_source_contract()
    rows = [
        row
        for row in expected["model_records"]
        if row["batch_id"] == request["batch_id"]
    ]
    rows.sort(key=lambda row: int(row["vial_position"]))
    if len(rows) != 3:
        raise F2ReceiptPublicationError("v4_request_source_binding_invalid")
    top_level_checks = (
        request["run_id"] == rows[0]["run_id"],
        request["current_recipe_id"] == rows[0]["current_recipe_id"],
        request["split_role"] == rows[0]["split_role"],
        request["programmed_recipe"]["dispense_order"]
        == rows[0]["programmed_dispense_order"],
        request["programmed_recipe"]["current_ml_by_tank"]
        == rows[0]["current_programmed_ml"],
    )
    actual_vials = request["vials"]
    vial_checks = all(
        actual["case_id"] == source["case_id"]
        and actual["vial_position"] == source["vial_position"]
        and actual["previous_recipe_id"] == source["previous_recipe_id"]
        and actual["previous_ml_by_tank"] == source["previous_programmed_ml"]
        for actual, source in zip(actual_vials, rows, strict=True)
    )
    if not all(top_level_checks) or not vial_checks:
        raise F2ReceiptPublicationError("v4_request_source_binding_invalid")


def _validate_varied_request(payload: Mapping[str, object]) -> dict[str, object]:
    split_version = payload.get("split_version")
    contract = _VARIED_CONTRACTS.get(str(split_version))
    expected_keys = (
        _VARIED_REQUEST_KEYS | _VARIED_V4_COHORT_BINDING_KEYS
        if split_version == "whole-run-varied-v4"
        else _VARIED_REQUEST_KEYS
    )
    if (
        contract is None
        or set(payload) != expected_keys
        or payload.get("schema_version") != contract["schema_version"]
        or payload.get("artifact_type")
        != "f2_varied_stage_batch_publish_request"
        or payload.get("data_origin") != "twin"
        or payload.get("provider") != "brev"
        or payload.get("runtime_name") != "NVIDIA Isaac Sim"
        or payload.get("cohort") != "scored_target"
        or payload.get("split_role") not in _VARIED_SPLITS
    ):
        raise F2ReceiptPublicationError("publish_request_schema_invalid")
    vials = payload.get("vials")
    if not isinstance(vials, list) or len(vials) != 3:
        raise F2ReceiptPublicationError("vial_receipts_invalid")
    normalized_vials: list[dict[str, object]] = []
    for raw in vials:
        if not isinstance(raw, Mapping) or set(raw) != _VARIED_VIAL_KEYS:
            raise F2ReceiptPublicationError("vial_receipt_schema_invalid")
        position = raw.get("vial_position")
        sha = raw.get("receipt_sha256")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position not in {1, 2, 3}
            or not isinstance(sha, str)
            or _SHA256.fullmatch(sha) is None
        ):
            raise F2ReceiptPublicationError("vial_receipt_schema_invalid")
        case_id = _text(raw.get("case_id"), "case_id")
        if _CASE_ID.fullmatch(case_id) is None:
            raise F2ReceiptPublicationError("case_id_invalid")
        normalized_vials.append(
            {
                "case_id": case_id,
                "vial_id": _text(raw.get("vial_id"), "vial_id"),
                "vial_position": position,
                "previous_recipe_id": _text(
                    raw.get("previous_recipe_id"), "previous_recipe_id"
                ),
                "previous_ml_by_tank": _varied_previous_ml(
                    raw.get("previous_ml_by_tank")
                ),
                "receipt_path": _text(raw.get("receipt_path"), "receipt_path"),
                "receipt_sha256": sha,
            }
        )
    normalized_vials.sort(key=lambda row: int(row["vial_position"]))
    if (
        [row["vial_position"] for row in normalized_vials] != [1, 2, 3]
        or [row["vial_id"] for row in normalized_vials] != list(_VIALS)
        or len({row["case_id"] for row in normalized_vials}) != 3
        or len({row["receipt_path"] for row in normalized_vials}) != 3
        or len({row["receipt_sha256"] for row in normalized_vials}) != 3
    ):
        raise F2ReceiptPublicationError("vial_receipts_not_distinct")
    result = {
        "schema_version": int(contract["schema_version"]),
        "split_version": str(split_version),
        "runtime_id": _text(payload.get("runtime_id"), "runtime_id"),
        "batch_id": _text(payload.get("batch_id"), "batch_id"),
        "run_id": _text(payload.get("run_id"), "run_id"),
        "cohort": "scored_target",
        "current_recipe_id": _text(
            payload.get("current_recipe_id"), "current_recipe_id"
        ),
        "split_role": str(payload["split_role"]),
        "programmed_recipe": _varied_programmed_recipe(
            payload.get("programmed_recipe")
        ),
        "vials": normalized_vials,
    }
    if split_version == "whole-run-varied-v4":
        result["cohort_binding"] = _normalize_v4_cohort_binding(payload)
        _validate_v4_request_source_binding(result)
    return result


def _load_artifact_validator_module() -> ModuleType:
    global _ARTIFACT_VALIDATOR_MODULE
    if _ARTIFACT_VALIDATOR_MODULE is not None:
        return _ARTIFACT_VALIDATOR_MODULE
    script = Path(__file__).resolve().parents[3] / "cli/twin/physx" / "validate_physx_f2_artifacts.py"
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


def _load_bound_json_artifact(
    locator: Mapping[str, object], base: Path
) -> tuple[Mapping[str, object], dict[str, str]]:
    path = Path(str(locator["path"])).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.absolute()
    try:
        resolved = path.resolve(strict=True)
        payload, snapshot = load_strict_json_snapshot(path, max_bytes=64 * 1024 * 1024)
    except (OSError, ValueError) as error:
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid") from error
    if (
        resolved != path
        or not path.is_file()
        or snapshot["sha256"] != locator["sha256"]
        or not isinstance(payload, Mapping)
    ):
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
    return payload, {"path": str(path), "sha256": str(snapshot["sha256"])}


def _bind_v4_cohort_provenance(
    binding: Mapping[str, object], base: Path
) -> dict[str, object]:
    expected = _v4_source_contract()
    expected_case_ids = expected["case_ids"]
    expected_plan_sha = expected["render_plan_sha256"]
    if binding.get("source_render_plan_sha256") != expected_plan_sha:
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
    launch, launch_locator = _load_bound_json_artifact(
        binding["cohort_launch_receipt"], base
    )
    merge, merge_locator = _load_bound_json_artifact(
        binding["cohort_merge_receipt"], base
    )
    output_contract = launch.get("expected_output_contract")
    output_cases = (
        output_contract.get("cases")
        if isinstance(output_contract, Mapping)
        else None
    )
    if (
        launch.get("schema_version") != 1
        or launch.get("artifact_type")
        != "physx_train_residual_serial_cohort_launch"
        or launch.get("case_ids") != expected_case_ids
        or launch.get("plan_sha256") != expected_plan_sha
        or not isinstance(output_contract, Mapping)
        or output_contract.get("case_count") != 30
        or not isinstance(output_cases, Mapping)
        or set(output_cases) != set(expected_case_ids)
    ):
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")

    source_shards = merge.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != 30:
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
    shard_cases: dict[int, str] = {}
    for shard in source_shards:
        if not isinstance(shard, Mapping):
            raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
        index = shard.get("index")
        case_ids = shard.get("case_ids")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in shard_cases
            or not isinstance(case_ids, list)
            or len(case_ids) != 1
            or not isinstance(case_ids[0], str)
        ):
            raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
        shard_cases[index] = case_ids[0]
    if (
        sorted(shard_cases) != list(range(30))
        or [shard_cases[index] for index in range(30)] != expected_case_ids
    ):
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")

    resume = merge.get("full_resume_summary")
    case_results = resume.get("case_results") if isinstance(resume, Mapping) else None
    if not isinstance(case_results, list):
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
    recognized = [
        row.get("case_id")
        for row in case_results
        if isinstance(row, Mapping) and row.get("outcome") == "recognized_completed"
    ]
    if (
        merge.get("schema_version") != 1
        or merge.get("artifact_type") != "physx_train_residual_shard_merge"
        or merge.get("status") != "pass"
        or merge.get("case_ids") != expected_case_ids
        or merge.get("merged_case_count") != 30
        or merge.get("shard_count") != 30
        or merge.get("plan_sha256") != expected_plan_sha
        or resume.get("status") != "pass"
        or resume.get("planned_case_count") != 30
        or resume.get("validated_completed_case_count") != 30
        or resume.get("plan_sha256") != expected_plan_sha
        or recognized != expected_case_ids
        or len(case_results) != 30
    ):
        raise F2ReceiptPublicationError("cohort_merge_binding_invalid")
    return {
        "source_render_plan_sha256": expected_plan_sha,
        "cohort_launch_receipt": launch_locator,
        "cohort_merge_receipt": merge_locator,
    }


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
        "case_id": model.get(
            "case_id", binding.get("case_id") if isinstance(binding, Mapping) else None
        ),
        "batch_id": binding.get("batch_id") if isinstance(binding, Mapping) else None,
        "vial_position": model.get(
            "vial_position",
            binding.get("vial_ordinal") if isinstance(binding, Mapping) else None,
        ),
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


def _verify_varied_model_record(
    record: Mapping[str, object],
    request: Mapping[str, object],
    *,
    vial: Mapping[str, object],
) -> None:
    recipe = request["programmed_recipe"]
    assert isinstance(recipe, Mapping)
    checks = (
        record.get("current_recipe_id") == request["current_recipe_id"],
        record.get("previous_recipe_id") == vial["previous_recipe_id"],
        record.get("dispense_order") == recipe["dispense_order"],
        _same_number_map(
            record.get("current_ml_by_tank"), recipe["current_ml_by_tank"]
        ),
        _same_number_map(
            record.get("previous_ml_by_tank"), vial["previous_ml_by_tank"]
        ),
        record.get("split_role") == request["split_role"],
        record.get("split_version") == request["split_version"],
        record.get("data_origin") == "twin",
        record.get("run_id") == request["run_id"],
        record.get("batch_id") == request["batch_id"],
        record.get("case_id") == vial["case_id"],
        record.get("vial_position") == vial["vial_position"],
    )
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
    item: Mapping[str, object],
    *,
    base: Path,
    request: Mapping[str, object],
    vial_ordinal: int,
    artifact_validator: Callable[[Path], Mapping[str, object]],
    varied: bool = False,
) -> dict[str, object]:
    path = _resolved_receipt_path(str(item["receipt_path"]), base)
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
        or receipt.get("render_path")
        not in {EMITTER_RENDER_PATH, COUNT_DRIVEN_RENDER_PATH}
        or receipt.get("status") != "pass"
        or not isinstance(qualification, Mapping)
        or qualification.get("first_stage_only") is not False
    ):
        raise F2ReceiptPublicationError("receipt_not_accepted_full_run")
    if not varied:
        try:
            receipt_failures = tuple(validate_paint_ball_receipt(receipt))
        except Exception as error:
            raise F2ReceiptPublicationError(
                "receipt_substantive_validation_error"
            ) from error
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
    artifact_warnings = validation.get("warnings", [])
    if not isinstance(artifact_warnings, list) or any(
        not isinstance(warning, Mapping)
        or warning.get("hard_gate") is not False
        or warning.get("claim") != "diagnostic_warning_not_a_physics_pass"
        for warning in artifact_warnings
    ):
        raise F2ReceiptPublicationError("receipt_artifact_warnings_invalid")

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
    _legacy_runtime_binding(receipt.get("arguments"))
    script_sha = provenance.get("script_sha256")
    stage_sha = provenance.get("requested_stage_sha256")
    if (
        not isinstance(script_sha, str)
        or _SHA256.fullmatch(script_sha) is None
        or not isinstance(stage_sha, str)
        or _SHA256.fullmatch(stage_sha) is None
    ):
        raise F2ReceiptPublicationError("receipt_source_hash_invalid")
    if varied:
        arguments = receipt.get("arguments")
        if not isinstance(arguments, Mapping) or arguments.get("realtime_video") is not True:
            raise F2ReceiptPublicationError("receipt_varied_runtime_contract_invalid")
        _verify_varied_model_record(
            _normalized_model_record(receipt), request, vial=item
        )
    else:
        _verify_model_record(
            _normalized_model_record(receipt), request, vial_ordinal=vial_ordinal
        )

    steps = render.get("stage_frame_physics_steps")
    stage_hashes = render.get("stage_frame_sha256")
    pours = plan.get("pours")
    pairs = semantics.get("pairs")
    clock = _two_clock_binding(timing)
    report_hz = clock["report_hz"]
    expected_stage_count = (
        len(request["programmed_recipe"]["dispense_order"])
        if varied
        else 3
    )
    if (
        not isinstance(steps, list)
        or len(steps) != expected_stage_count
        or not isinstance(stage_hashes, list)
        or len(stage_hashes) != expected_stage_count
        or not isinstance(pours, list)
        or len(pours) != expected_stage_count
        or not isinstance(pairs, list)
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
        frame_id = Path(rgb["path"]).stem
        if not frame_id:
            raise F2ReceiptPublicationError("receipt_frame_id_invalid")
        event = {
            "event_id": f"dispense-{ordinal}-{tank}",
            "stage_ordinal": ordinal,
            "tank_id": tank,
            "tank_order": ordinal,
            "physics_step": raw_step,
            "frame_id": None,
            "event_relative_timestamp_s": raw_step / report_hz,
            "final_observation_offset_s": (
                raw_step - raw_pour["end_step"]
            ) / report_hz,
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
                "frame_id": frame_id,
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
    pour_end_steps = timing.get("pour_end_steps")
    final_step = timing.get("primary_final_capture_step")
    final_offset = timing.get("primary_final_offset_s")
    if (
        pour_end_steps != [int(pour["end_step"]) for pour in pours]
        or final_step != steps[-1]
        or isinstance(final_offset, bool)
        or not isinstance(final_offset, (int, float))
        or not math.isfinite(float(final_offset))
        or not math.isclose(
            float(final_offset),
            float(events[-1]["final_observation_offset_s"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise F2ReceiptPublicationError("receipt_stage_clock_binding_invalid")
    result = {
        "vial_id": item["vial_id"],
        "receipt_path": str(path),
        "receipt_sha256": item["receipt_sha256"],
        "verified_artifact_count": validation["verified_artifact_count"],
        "script_sha256": script_sha,
        "stage_sha256": stage_sha,
        "isaac_version": versions["isaac"],
        "physx_version": versions.get("physx"),
        "render_path": receipt["render_path"],
        "clock_binding": clock,
        "events": events,
        "rows": rows,
        "artifact_validation_warnings": [dict(warning) for warning in artifact_warnings],
    }
    if varied:
        result.update(
            {
                "case_id": item["case_id"],
                "vial_position": item["vial_position"],
            }
        )
    return result


def _write_publication(
    output: Path,
    *,
    runtime_receipt: Mapping[str, object],
    manifest: Mapping[str, object],
    context: Mapping[str, object],
    source_receipts: list[Mapping[str, object]],
    request_sha256: str,
    expected_stage_row_count: int = 9,
    publication_schema_version: int = 1,
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
        from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

        bound = bind_stage_capture_manifest(
            partial / "f2-stage-capture-manifest.json"
        )
        if bound.get("denominators") != {
            "image_measurements": expected_stage_row_count,
            "stage_statistics": expected_stage_row_count,
        }:
            raise F2ReceiptPublicationError("published_manifest_binding_failed")
        record = {
            "schema_version": publication_schema_version,
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
            row["render_path"],
            json.dumps(row["clock_binding"], sort_keys=True),
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
                "render_path",
                "clock_binding",
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


def _build_varied_runtime_acceptance_receipt(
    *,
    runtime_binding: Mapping[str, object],
    artifact_validation: Mapping[str, object],
    source_receipts: list[Mapping[str, object]],
) -> dict[str, object]:
    """Build V3 runtime evidence without conflating two height checks."""

    if (
        set(runtime_binding)
        != {
            "provider",
            "runtime_name",
            "isaac_sim_version",
            "runtime_id",
            "source_sha256",
        }
        or runtime_binding.get("provider") != "brev"
        or runtime_binding.get("runtime_name") != "NVIDIA Isaac Sim"
        or runtime_binding.get("isaac_sim_version") != "6.2.0.0"
        or not isinstance(runtime_binding.get("runtime_id"), str)
        or not runtime_binding["runtime_id"]
        or not isinstance(runtime_binding.get("source_sha256"), str)
        or _SHA256.fullmatch(str(runtime_binding["source_sha256"])) is None
    ):
        raise F2ReceiptPublicationError("runtime_binding_invalid")
    failures = artifact_validation.get("failures")
    if (
        artifact_validation.get("status") != "pass"
        or failures != []
        or isinstance(artifact_validation.get("verified_artifact_count"), bool)
        or not isinstance(artifact_validation.get("verified_artifact_count"), int)
        or artifact_validation["verified_artifact_count"] <= 0
    ):
        raise F2ReceiptPublicationError("runtime_artifact_validation_invalid")
    if len(source_receipts) != 3:
        raise F2ReceiptPublicationError("runtime_particle_height_diagnostic_invalid")
    vials: list[dict[str, object]] = []
    for row in source_receipts:
        warnings = row.get("artifact_validation_warnings")
        case_id = row.get("case_id")
        vial_id = row.get("vial_id")
        if (
            not isinstance(warnings, list)
            or not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            or vial_id not in _VIALS
        ):
            raise F2ReceiptPublicationError(
                "runtime_particle_height_diagnostic_invalid"
            )
        warning_count = len(warnings)
        bounds_passed = warning_count == 0
        vials.append(
            {
                "case_id": case_id,
                "vial_id": vial_id,
                "status": "within_bounds" if bounds_passed else "warning",
                "bounds_passed": bounds_passed,
                "warning_count": warning_count,
            }
        )
    vials.sort(key=lambda row: str(row["vial_id"]))
    if [row["vial_id"] for row in vials] != list(_VIALS):
        raise F2ReceiptPublicationError("runtime_particle_height_diagnostic_invalid")
    warning_count = sum(int(row["warning_count"]) for row in vials)
    warned_vial_count = sum(row["bounds_passed"] is False for row in vials)
    bounds_passed = warning_count == 0
    return {
        "schema_version": 3,
        "artifact_type": "f2_varied_runtime_acceptance_receipt",
        "status": "accepted",
        "provider": "brev",
        "runtime_id": runtime_binding["runtime_id"],
        "isaac_sim_version": "6.2.0.0",
        "source_sha256": runtime_binding["source_sha256"],
        "hard_gates": {
            name: True for name in sorted(_VARIED_RUNTIME_HARD_GATES)
        },
        "particle_height_diagnostic": {
            "hard_gate": False,
            "physics_pass_claimed": False,
            "status": "within_bounds" if bounds_passed else "warning",
            "bounds_passed": bounds_passed,
            "warning_count": warning_count,
            "warned_vial_count": warned_vial_count,
            "vial_count": 3,
            "claim": (
                "particle_height_within_bounds_diagnostic_only"
                if bounds_passed
                else "diagnostic_warning_not_a_physics_pass"
            ),
            "vials": vials,
        },
    }


def publish_f2_varied_stage_batch(
    request: object,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Publish one varied-run three-vial batch with two or three active stages."""

    payload, request_base, request_sha = _load_request(request)
    try:
        validate_deploy_input_fields(payload, require_allowlist=False)
    except ValueError as error:
        raise F2ReceiptPublicationError("forbidden_oracle_field") from error
    normalized = _validate_varied_request(payload)
    if normalized["split_version"] == "whole-run-varied-v4":
        normalized["cohort_binding"] = _bind_v4_cohort_provenance(
            normalized["cohort_binding"], request_base
        )
    source_receipts = [
        _extract_receipt(
            item,
            base=request_base,
            request=normalized,
            vial_ordinal=ordinal,
            artifact_validator=_official_artifact_validator,
            varied=True,
        )
        for ordinal, item in enumerate(normalized["vials"], start=1)
    ]
    shared_fields = {
        (
            row["script_sha256"],
            row["stage_sha256"],
            row["isaac_version"],
            row["physx_version"],
            row["render_path"],
            json.dumps(row["clock_binding"], sort_keys=True),
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
    runtime_receipt = _build_varied_runtime_acceptance_receipt(
        runtime_binding=runtime_binding,
        artifact_validation=aggregate_validation,
        source_receipts=source_receipts,
    )
    runtime_bytes = (
        json.dumps(runtime_receipt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    events = source_receipts[0]["events"]
    manifest = {
        "schema_version": 3,
        "artifact_type": "f2_stage_capture_manifest",
        "status": "accepted",
        "data_origin": "twin",
        "batch_id": normalized["batch_id"],
        "run_id": normalized["run_id"],
        "recipe_id": normalized["current_recipe_id"],
        "split_role": normalized["split_role"],
        "split_version": normalized["split_version"],
        "view_id": "vial_fill",
        "runtime_binding": runtime_binding,
        "runtime_receipt": {
            "path": "f2-runtime-acceptance-receipt.json",
            "sha256": runtime_sha,
        },
        "stage_events": events,
        "rows": [row for receipt in source_receipts for row in receipt["rows"]],
    }
    context_vials = [
        {
            key: item[key]
            for key in (
                "case_id",
                "vial_id",
                "vial_position",
                "previous_recipe_id",
                "previous_ml_by_tank",
            )
        }
        for item in normalized["vials"]
    ]
    context = {
        "schema_version": 3,
        "artifact_type": "stage_batch_context",
        "split_version": normalized["split_version"],
        "batches": [
            {
                "batch_id": normalized["batch_id"],
                "cohort": normalized["cohort"],
                "current_recipe_id": normalized["current_recipe_id"],
                "run_id": normalized["run_id"],
                "split_role": normalized["split_role"],
                "programmed_recipe": normalized["programmed_recipe"],
                "vials": context_vials,
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
            **{
                key: row[key]
                for key in (
                    "vial_id",
                    "receipt_path",
                    "receipt_sha256",
                    "verified_artifact_count",
                    "render_path",
                    "clock_binding",
                    "artifact_validation_warnings",
                )
            },
            "case_id": normalized["vials"][index]["case_id"],
            "vial_position": normalized["vials"][index]["vial_position"],
        }
        for index, row in enumerate(source_receipts)
    ]
    return _write_publication(
        Path(output_dir).expanduser(),
        runtime_receipt=runtime_receipt,
        manifest=manifest,
        context=context,
        source_receipts=provenance,
        request_sha256=request_sha,
        expected_stage_row_count=3 * len(events),
        publication_schema_version=3,
    )


__all__ = ["F2ReceiptPublicationError", "publish_f2_stage_batch"]

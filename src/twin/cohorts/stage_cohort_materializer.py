"""Materialize oracle-free F2 batch publication inputs from one frozen cohort."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from os import PathLike
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Sequence

from twin.policy import validate_deploy_input_fields
from .downstream_binding import (
    F2DownstreamBindingError,
    bind_runtime_particle_readback,
)
from .train_residual_manifest_v2 import build_train_residual_artifacts
from ..scene.nuisance_authoring import light_rows_match, transform_rows_match
from ..render.render_process import hash_regular_file, load_strict_json_snapshot
from ..render.twin_video import _atomic_rename_noreplace, _write_private_json


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUNTIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_LEGACY_RESOLUTION_PROFILE = "c20_2p5mm"
_LEGACY_EFFECTIVE_PARTICLE_VOLUME_ML = 0.011272866937444935
_APPROVED_QUALIFICATION_RENDERER_SHA256 = (
    "cc330a4db95162eef9e2d832c307406f0796a3f78b3ac17aa3dbd0e7e746afc0"
)
_APPROVED_COHORT_RENDERER_SHA256 = (
    "e8f89b39382a90c931ace5b95350e974ac3aad3c58bee9f09619761315cdcf0d"
)
_APPROVED_QUALIFICATION_RENDERER_COMMIT = (
    "53063639cdcf33a5be29c7980cd72ce067e35b9d"
)
_APPROVED_COHORT_RENDERER_COMMIT = "1d7395511e68db40d2eb7253b4f0bfd2bbc192b6"
_APPROVED_RENDERER_DELTA_SCOPE = [
    "train_case_argument_validation",
    "train_residual_runtime_plan_adapter",
    "receipt_bookkeeping_and_validator",
    "fill_end_boundary_schedule_metadata",
    "count_driven_gpu_memory_safety_gate",
]


class F2StageCohortMaterializationError(ValueError):
    """A frozen cohort or published context failed closed."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return value


def _direct_directory(value: str | PathLike[str], field: str) -> Path:
    candidate = Path(value).expanduser().absolute()
    try:
        lexical = os.lstat(candidate)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise F2StageCohortMaterializationError(f"{field}_invalid") from error
    if (
        stat.S_ISLNK(lexical.st_mode)
        or not stat.S_ISDIR(lexical.st_mode)
        or resolved != candidate
    ):
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return candidate


def _load_object(
    path: Path, field: str, *, max_bytes: int = 16 * 1024 * 1024
) -> tuple[dict[str, Any], dict[str, object]]:
    try:
        payload, snapshot = load_strict_json_snapshot(path, max_bytes=max_bytes)
    except (OSError, ValueError) as error:
        raise F2StageCohortMaterializationError(f"{field}_invalid") from error
    if not isinstance(payload, dict):
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return payload, snapshot


def _load_plan(
    path: str | PathLike[str],
) -> tuple[dict[str, Any], dict[str, object], dict[str, object], str]:
    candidate = Path(path).expanduser().absolute()
    plan, snapshot = _load_object(candidate, "source_plan")
    artifacts = build_train_residual_artifacts(
        repo_root=Path(__file__).resolve().parents[3]
    )
    expected = artifacts["render-schedule-plan.json"]
    expected_sha = _canonical_sha256(expected)
    if (
        plan != expected
        or _canonical_sha256(plan) != expected_sha
        or plan.get("artifact_type")
        != "physx_f2_train_leading_residue_render_plan_v2"
    ):
        raise F2StageCohortMaterializationError("source_plan_not_frozen_canonical_plan")
    return plan, snapshot, artifacts["model-input-plan.json"], expected_sha


def _batch_rows(plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    batches = plan.get("batches")
    if not isinstance(batches, list) or not batches:
        raise F2StageCohortMaterializationError("source_plan_batches_invalid")
    if not all(isinstance(row, Mapping) for row in batches):
        raise F2StageCohortMaterializationError("source_plan_batches_invalid")
    batch_ids = [row.get("batch_id") for row in batches]
    if (
        any(not isinstance(batch_id, str) or not batch_id for batch_id in batch_ids)
        or len(set(batch_ids)) != len(batch_ids)
        or any(
            row.get("batch_role") not in {"reference_only", "scored_target"}
            for row in batches
        )
    ):
        raise F2StageCohortMaterializationError("source_plan_batch_ids_invalid")
    return list(batches)


def _schedule_rows(plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    schedules = plan.get("execution_schedules")
    batches = _batch_rows(plan)
    expected_case_ids = [
        case_id
        for batch in batches
        for case_id in (
            batch.get("case_ids") if isinstance(batch.get("case_ids"), list) else []
        )
    ]
    if (
        not isinstance(schedules, list)
        or len(schedules) != 3 * len(batches)
        or len(expected_case_ids) != len(schedules)
    ):
        raise F2StageCohortMaterializationError("source_plan_schedules_invalid")
    if not all(isinstance(row, Mapping) for row in schedules):
        raise F2StageCohortMaterializationError("source_plan_schedules_invalid")
    observed_case_ids = [
        _mapping(row.get("case_binding"), "source_case_binding").get("case_id")
        for row in schedules
    ]
    if observed_case_ids != expected_case_ids or len(set(observed_case_ids)) != len(
        observed_case_ids
    ):
        raise F2StageCohortMaterializationError("source_plan_schedules_invalid")
    return list(schedules)


def _seed_by_case(plan: Mapping[str, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for batch in _batch_rows(plan):
        cases = batch.get("case_ids")
        seeds = batch.get("render_seed_by_vial")
        if (
            not isinstance(cases, list)
            or not isinstance(seeds, list)
            or len(cases) != 3
            or len(seeds) != 3
        ):
            raise F2StageCohortMaterializationError("source_plan_case_seeds_invalid")
        for case_id, seed in zip(cases, seeds, strict=True):
            if (
                not isinstance(case_id, str)
                or case_id in result
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                raise F2StageCohortMaterializationError(
                    "source_plan_case_seeds_invalid"
                )
            result[case_id] = seed
    return result


def _nuisance_case_by_case(plan: Mapping[str, object]) -> dict[str, str]:
    """Return the frozen explicit nuisance ID for every render case."""

    result: dict[str, str] = {}
    for batch in _batch_rows(plan):
        cases = batch.get("case_ids")
        nuisance_ids = batch.get("nuisance_group_ids")
        if (
            not isinstance(cases, list)
            or not isinstance(nuisance_ids, list)
            or len(cases) != 3
            or len(nuisance_ids) != 3
        ):
            raise F2StageCohortMaterializationError(
                "source_plan_nuisance_bindings_invalid"
            )
        for case_id, nuisance_id in zip(cases, nuisance_ids, strict=True):
            if (
                not isinstance(case_id, str)
                or not isinstance(nuisance_id, str)
                or not nuisance_id
                or case_id in result
                or nuisance_id in result.values()
            ):
                raise F2StageCohortMaterializationError(
                    "source_plan_nuisance_bindings_invalid"
                )
            result[case_id] = nuisance_id
    return result


def _safe_record_by_case(model_plan: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = model_plan.get("vial_records")
    allowlist = model_plan.get("model_input_allowlist")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(allowlist, list)
        or not allowlist
        or not all(isinstance(field, str) and field for field in allowlist)
        or len(set(allowlist)) != len(allowlist)
    ):
        raise F2StageCohortMaterializationError("source_model_records_invalid")
    allowed_fields = set(allowlist)
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != allowed_fields
            or not isinstance(row.get("case_id"), str)
        ):
            raise F2StageCohortMaterializationError("source_model_records_invalid")
        try:
            validate_deploy_input_fields(row, require_allowlist=False)
        except ValueError as error:
            raise F2StageCohortMaterializationError(
                "source_model_records_invalid"
            ) from error
        case_id = str(row["case_id"])
        if case_id in result:
            raise F2StageCohortMaterializationError("source_model_records_invalid")
        result[case_id] = row
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return value


def _positive_number(value: object, field: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return value


_CLOCK_INVARIANT_FIELDS = (
    "report_hz",
    "physics_hz",
    "physics_steps_per_second",
    "physics_substeps_per_report_interval",
    "diffusion_updates_per_report_interval",
)


def _two_clock_binding(value: object, field: str) -> dict[str, int]:
    timing = _mapping(value, field)
    names = (
        *_CLOCK_INVARIANT_FIELDS,
        "report_interval_count",
        "executed_report_intervals",
        "physics_callback_count",
        "callbacks_expected",
        "callbacks_observed",
        "diffusion_update_count",
    )
    fields = {name: timing.get(name) for name in names}
    if any(type(item) is not int for item in fields.values()):
        raise F2StageCohortMaterializationError(f"{field}_invalid")
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
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return {name: int(item) for name, item in fields.items()}


def _legacy_runtime_binding(arguments: Mapping[str, object], field: str) -> dict[str, object]:
    """Bind the only qualified particle profile, rather than a C21 substitute."""

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
        raise F2StageCohortMaterializationError(f"{field}_invalid")
    return {
        "particle_resolution_profile": _LEGACY_RESOLUTION_PROFILE,
        "effective_particle_volume_ml": _LEGACY_EFFECTIVE_PARTICLE_VOLUME_ML,
    }


def _runtime_particle_binding(value: object, field: str) -> dict[str, object]:
    try:
        return bind_runtime_particle_readback(value)
    except F2DownstreamBindingError as error:
        raise F2StageCohortMaterializationError(f"{field}_invalid") from error


def _qualification_binding(
    path: str | PathLike[str],
) -> tuple[dict[str, Any], dict[str, object], dict[str, object]]:
    qualification_path = Path(path).expanduser().absolute()
    receipt, snapshot = _load_object(
        qualification_path, "accepted_qualification_receipt"
    )
    arguments = _mapping(receipt.get("arguments"), "qualification_arguments")
    render = _mapping(receipt.get("render"), "qualification_render")
    versions = _mapping(receipt.get("versions"), "qualification_versions")
    legacy_runtime = _legacy_runtime_binding(
        arguments, "qualification_legacy_runtime_contract"
    )
    if versions.get("isaac") != "6.2.0.0":
        raise F2StageCohortMaterializationError(
            "qualification_legacy_runtime_contract_invalid"
        )
    clock = _two_clock_binding(
        receipt.get("timing"), "qualification_two_clock_contract"
    )
    _runtime_particle_binding(receipt, "qualification_runtime_particle_binding")
    provenance = _mapping(receipt.get("provenance"), "qualification_provenance")
    qualification = _mapping(
        receipt.get("qualification"), "qualification_result"
    )
    resolution = render.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        )
    ):
        raise F2StageCohortMaterializationError(
            "qualification_render_resolution_invalid"
        )
    total_spp = _positive_int(
        render.get("pathtracing_total_spp_readback"),
        "qualification_pathtracing_total_spp",
    )
    count_driven = arguments.get("count_driven_twin_liquid_mesh")
    render_path = receipt.get("render_path")
    if count_driven is True:
        expected_render_path = "count_driven_twin_liquid_mesh"
    elif count_driven is False:
        expected_render_path = "point_instancer"
    else:
        raise F2StageCohortMaterializationError(
            "qualification_representation_invalid"
        )
    expected_resolution_text = f"{resolution[0]}x{resolution[1]}"
    if not (
        receipt.get("status") == "pass"
        and qualification.get("first_stage_only") is False
        and arguments.get("recipe8_residual_volume_ml") == 0.0
        and arguments.get("solver_position_iterations") == 16
        and render_path == expected_render_path
        and arguments.get("resolution") == expected_resolution_text
        and render.get("renderer") == "PathTracing"
        and render.get("pathtracing_total_spp_requested") == total_spp
        and render.get("pathtracing_spp_requested") == total_spp
        and render.get("pathtracing_spp_readback") == total_spp
    ):
        raise F2StageCohortMaterializationError(
            "qualification_representation_or_render_contract_invalid"
        )
    contract = {
        "renderer": "PathTracing",
        "resolution": list(resolution),
        "rt_subframes": _positive_int(
            render.get("rt_subframes"), "qualification_rt_subframes"
        ),
        "pathtracing_total_spp": total_spp,
        "warmup_updates": _positive_int(
            render.get("warmup_updates"), "qualification_warmup_updates"
        ),
        "capture_stride": _positive_int(
            arguments.get("capture_stride"), "qualification_capture_stride"
        ),
        "video_fps": _positive_number(
            arguments.get("video_fps"), "qualification_video_fps"
        ),
        "wall_limit_seconds": _positive_int(
            arguments.get("wall_limit_seconds"),
            "qualification_wall_limit_seconds",
        ),
        "render_path": render_path,
        "count_driven_twin_liquid_mesh": count_driven,
        **legacy_runtime,
        **{field: clock[field] for field in _CLOCK_INVARIANT_FIELDS},
    }
    _sha(provenance.get("requested_stage_sha256"), "qualification_stage_sha256")
    _sha(provenance.get("script_sha256"), "qualification_renderer_sha256")
    return receipt, snapshot, contract


def _normalize_launch_render_contract(
    value: object, expected: Mapping[str, object]
) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    allowed_fields = set(expected) | {"nuisance"}
    normalized = dict(value)
    nuisance = normalized.pop("nuisance", None)
    if nuisance != "required_applied_session_layer_only" or set(value) - allowed_fields:
        return None
    expected = dict(expected)
    expected.pop("nuisance", None)
    if "physics_hz" not in normalized:
        physics_steps_per_second = normalized.get("physics_steps_per_second")
        if type(physics_steps_per_second) is not int:
            return None
        normalized["physics_hz"] = physics_steps_per_second
    if "diffusion_updates_per_report_interval" not in normalized:
        physics_substeps = normalized.get("physics_substeps_per_report_interval")
        if not (
            type(physics_substeps) is int
            and physics_substeps == 1
            and expected.get("physics_substeps_per_report_interval") == 1
            and expected.get("diffusion_updates_per_report_interval") == 1
        ):
            return None
        normalized["diffusion_updates_per_report_interval"] = 1
    return normalized if normalized == expected else None


def _cohort_binding(
    *,
    root: Path,
    plan: Mapping[str, object],
    plan_snapshot: Mapping[str, object],
    plan_sha256: str,
    qualification_snapshot: Mapping[str, object],
    qualification_receipt: Mapping[str, object],
    render_contract: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, object], list[dict[str, object]]]:
    launch_path = root / "cohort-launch.json"
    launch, launch_snapshot = _load_object(launch_path, "cohort_launch")
    schedules = _schedule_rows(plan)
    nuisance_case_ids = _nuisance_case_by_case(plan)
    case_ids = [
        _mapping(row.get("case_binding"), "source_case_binding").get("case_id")
        for row in schedules
    ]
    qualification_sha = _sha(
        launch.get("qualification_receipt_sha256"),
        "accepted_qualification_receipt_sha256",
    )
    normalized_launch_contract = _normalize_launch_render_contract(
        launch.get("render_contract"), render_contract
    )
    checks = (
        launch.get("schema_version") == 1,
        launch.get("artifact_type")
        == "physx_train_residual_serial_cohort_launch",
        launch.get("plan_sha256") == plan_sha256,
        launch.get("plan_file_sha256") == plan_snapshot.get("sha256"),
        qualification_sha == qualification_snapshot.get("sha256"),
        normalized_launch_contract is not None,
        launch.get("case_ids") == case_ids,
    )
    if not all(checks):
        raise F2StageCohortMaterializationError("cohort_launch_binding_mismatch")
    nuisance_plan = _mapping(
        launch.get("nuisance_plan"), "cohort_launch_nuisance_plan"
    )
    nuisance_receipt_path = nuisance_plan.get("receipt_path")
    nuisance_receipt_sha256 = _sha(
        nuisance_plan.get("receipt_sha256"),
        "cohort_launch_nuisance_receipt_sha256",
    )
    expected_nuisance_plan_fields = {
        "mapping_version",
        "mapping_status",
        "real_mapping_claimed",
        "session_layer_only",
        "excluded_from_model_inputs",
        "excluded_from_deploy_inputs",
        "receipt_path",
        "receipt_sha256",
        "case_id_by_case",
    }
    if (
        set(nuisance_plan) != expected_nuisance_plan_fields
        or nuisance_plan.get("mapping_version")
        != "f2-measured-percentile-to-existing-lights-v1"
        or nuisance_plan.get("mapping_status")
        != "uncalibrated_synthetic_nuisance_mapping"
        or nuisance_plan.get("real_mapping_claimed") is not False
        or nuisance_plan.get("session_layer_only") is not True
        or nuisance_plan.get("excluded_from_model_inputs") is not True
        or nuisance_plan.get("excluded_from_deploy_inputs") is not True
        or not isinstance(nuisance_receipt_path, str)
        or not Path(nuisance_receipt_path).is_absolute()
        or nuisance_plan.get("case_id_by_case") != nuisance_case_ids
        or launch.get("nuisance_receipt_path") != nuisance_receipt_path
        or launch.get("nuisance_receipt_sha256") != nuisance_receipt_sha256
        or launch.get("nuisance_case_id_by_case") != nuisance_case_ids
    ):
        raise F2StageCohortMaterializationError(
            "cohort_launch_nuisance_binding_mismatch"
        )
    for field in (
        "stage_sha256",
        "renderer_script_sha256",
        "supervisor_script_sha256",
    ):
        _sha(launch.get(field), f"cohort_launch_{field}")
    qualification_provenance = _mapping(
        qualification_receipt.get("provenance"), "qualification_provenance"
    )
    qualification_renderer_sha = qualification_provenance.get("script_sha256")
    cohort_renderer_sha = launch.get("renderer_script_sha256")
    renderer_compatible = cohort_renderer_sha == qualification_renderer_sha
    if not renderer_compatible:
        compatibility = launch.get("renderer_qualification_compatibility")
        required_compatibility_fields = {
            "mode",
            "qualification_renderer_sha256",
            "cohort_renderer_sha256",
            "qualification_renderer_commit",
            "cohort_renderer_commit",
            "declared_delta_scope",
            "qualification_render_path_changed",
        }
        renderer_compatible = (
            isinstance(compatibility, Mapping)
            and set(compatibility) == required_compatibility_fields
            and compatibility.get("mode")
            == "approved_parent_train_adapter_delta"
            and compatibility.get("qualification_renderer_sha256")
            == qualification_renderer_sha
            == _APPROVED_QUALIFICATION_RENDERER_SHA256
            and compatibility.get("cohort_renderer_sha256")
            == cohort_renderer_sha
            == _APPROVED_COHORT_RENDERER_SHA256
            and compatibility.get("qualification_renderer_commit")
            == _APPROVED_QUALIFICATION_RENDERER_COMMIT
            and compatibility.get("cohort_renderer_commit")
            == _APPROVED_COHORT_RENDERER_COMMIT
            and compatibility.get("declared_delta_scope")
            == _APPROVED_RENDERER_DELTA_SCOPE
            and compatibility.get("qualification_render_path_changed") is False
        )
    if not (
        launch.get("stage_sha256")
        == qualification_provenance.get("requested_stage_sha256")
        and renderer_compatible
    ):
        raise F2StageCohortMaterializationError(
            "cohort_launch_qualification_binding_mismatch"
        )
    attempts_root = _direct_directory(
        root / "supervisor-attempts", "supervisor_attempts"
    )
    attempt_paths = sorted(attempts_root.glob("attempt-*.json"))
    if not attempt_paths:
        raise F2StageCohortMaterializationError("supervisor_attempts_missing")
    attempts: list[dict[str, object]] = []
    latest: Mapping[str, object] | None = None
    for path in attempt_paths:
        payload, snapshot = _load_object(path, "supervisor_attempt")
        attempts.append(
            {
                "path": os.fspath(path),
                "sha256": snapshot["sha256"],
                "bytes": snapshot["bytes"],
                "status": payload.get("status"),
            }
        )
        latest = payload
    assert latest is not None
    latest_results = latest.get("case_results")
    latest_case_ids = (
        [row.get("case_id") for row in latest_results]
        if isinstance(latest_results, list)
        and all(isinstance(row, Mapping) for row in latest_results)
        else None
    )
    latest_outcomes = (
        [row.get("outcome") for row in latest_results]
        if isinstance(latest_results, list)
        and all(isinstance(row, Mapping) for row in latest_results)
        else []
    )
    if not (
        latest.get("schema_version") == 1
        and latest.get("artifact_type")
        == "physx_train_residual_serial_supervisor_attempt"
        and latest.get("status") == "pass"
        and latest.get("plan_sha256") == plan_sha256
        and latest.get("qualification_receipt_sha256") == qualification_sha
        and latest.get("planned_case_count") == len(case_ids)
        and latest.get("validated_completed_case_count") == len(case_ids)
        and latest_case_ids == case_ids
        and all(
            outcome in {"passed", "recognized_completed"}
            for outcome in latest_outcomes
        )
    ):
        raise F2StageCohortMaterializationError(
            "latest_supervisor_attempt_not_complete_pass"
        )
    return launch, launch_snapshot, attempts


def _case_receipts(
    *,
    root: Path,
    plan: Mapping[str, object],
    plan_snapshot: Mapping[str, object],
    plan_sha256: str,
    launch: Mapping[str, object],
    render_contract: Mapping[str, object],
    safe_record_by_case: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    seeds = _seed_by_case(plan)
    nuisance_case_ids = _nuisance_case_by_case(plan)
    nuisance_receipt_path = launch.get("nuisance_receipt_path")
    nuisance_receipt_sha256 = _sha(
        launch.get("nuisance_receipt_sha256"),
        "case_nuisance_receipt_sha256",
    )
    result: dict[str, dict[str, object]] = {}
    for ordinal, schedule in enumerate(_schedule_rows(plan), start=1):
        binding = _mapping(schedule.get("case_binding"), "source_case_binding")
        case_id = binding.get("case_id")
        if not isinstance(case_id, str) or case_id not in seeds:
            raise F2StageCohortMaterializationError("source_case_id_invalid")
        case_root = _direct_directory(root / "cases" / case_id, "case_root")
        try:
            exit_code = (case_root / "exit-code.txt").read_text(
                encoding="utf-8"
            ).strip()
        except (OSError, UnicodeError) as error:
            raise F2StageCohortMaterializationError("case_exit_code_invalid") from error
        case_result, case_result_snapshot = _load_object(
            case_root / "case-result.json", "case_result"
        )
        case_launch, case_launch_snapshot = _load_object(
            case_root / "launch-receipt.json", "case_launch"
        )
        receipt_path = case_root / "app-output" / "receipt.json"
        receipt, receipt_snapshot = _load_object(
            receipt_path, "case_receipt", max_bytes=64 * 1024 * 1024
        )
        seed = seeds[case_id]
        nuisance_case_id = nuisance_case_ids.get(case_id)
        if nuisance_case_id is None:
            raise F2StageCohortMaterializationError(
                f"source_nuisance_case_missing:{case_id}"
            )
        normalized_case_render_contract = _normalize_launch_render_contract(
            case_launch.get("render_contract"), render_contract
        )
        if not (
            exit_code == "0"
            and case_result.get("case_id") == case_id
            and case_result.get("seed") == seed
            and case_result.get("ordinal") == ordinal
            and case_result.get("outcome") == "passed"
            and case_result.get("return_code") == 0
            and case_result.get("timed_out") is False
            and case_launch.get("artifact_type")
            == "physx_train_residual_case_launch"
            and case_launch.get("case_id") == case_id
            and case_launch.get("seed") == seed
            and case_launch.get("ordinal") == ordinal
            and case_launch.get("plan_sha256") == plan_sha256
            and case_launch.get("plan_file_sha256") == plan_snapshot.get("sha256")
            and case_launch.get("qualification_receipt_sha256")
            == launch.get("qualification_receipt_sha256")
            and case_launch.get("stage_sha256") == launch.get("stage_sha256")
            and case_launch.get("renderer_script_sha256")
            == launch.get("renderer_script_sha256")
            and case_launch.get("nuisance_receipt_path")
            == nuisance_receipt_path
            and case_launch.get("nuisance_receipt_sha256")
            == nuisance_receipt_sha256
            and case_launch.get("nuisance_case_id") == nuisance_case_id
            and case_launch.get("nuisance_plan") == launch.get("nuisance_plan")
            and normalized_case_render_contract is not None
        ):
            raise F2StageCohortMaterializationError(
                f"completed_case_binding_mismatch:{case_id}"
            )
        arguments = _mapping(receipt.get("arguments"), "case_receipt_arguments")
        _legacy_runtime_binding(arguments, "case_legacy_runtime_contract")
        provenance = _mapping(receipt.get("provenance"), "case_receipt_provenance")
        render = _mapping(receipt.get("render"), "case_receipt_render")
        clock = _two_clock_binding(
            receipt.get("timing"), "case_receipt_two_clock_contract"
        )
        particle_binding = _runtime_particle_binding(
            receipt, "case_runtime_particle_binding"
        )
        schedule_timing = _mapping(
            schedule.get("timing"), "source_case_schedule_timing"
        )
        expected_report_count = _positive_int(
            schedule_timing.get("fill_plus_3p0_step"),
            "source_case_schedule_final_report_step",
        )
        if (
            clock["report_interval_count"] != expected_report_count
            or any(
                clock[field] != render_contract.get(field)
                for field in _CLOCK_INVARIANT_FIELDS
            )
        ):
            raise F2StageCohortMaterializationError(
                "case_receipt_two_clock_contract_invalid"
            )
        execution = _mapping(
            receipt.get("execution_schedule"), "case_receipt_execution_schedule"
        )
        renderer_binding = _mapping(
            execution.get("renderer_case_binding"),
            "case_receipt_renderer_case_binding",
        )
        expected_model = safe_record_by_case.get(case_id)
        if expected_model is None:
            raise F2StageCohortMaterializationError(
                f"source_case_record_missing:{case_id}"
            )
        if not (
            receipt.get("status") == "pass"
            and arguments.get("seed") == seed
            and arguments.get("train_render_plan_sha256") == plan_sha256
            and arguments.get("train_case_id") == case_id
            and arguments.get("accepted_qualification_receipt_sha256")
            == launch.get("qualification_receipt_sha256")
            and arguments.get("capture_stride")
            == render_contract.get("capture_stride")
            and arguments.get("video_fps") == render_contract.get("video_fps")
            and arguments.get("wall_limit_seconds")
            == render_contract.get("wall_limit_seconds")
            and arguments.get("resolution")
            == (
                f"{render_contract['resolution'][0]}x"
                f"{render_contract['resolution'][1]}"
            )
            and arguments.get("count_driven_twin_liquid_mesh")
            is render_contract.get("count_driven_twin_liquid_mesh")
            and receipt.get("render_path") == render_contract.get("render_path")
            and provenance.get("requested_stage_sha256")
            == launch.get("stage_sha256")
            and provenance.get("script_sha256")
            == launch.get("renderer_script_sha256")
            and render.get("renderer") == render_contract.get("renderer")
            and render.get("resolution") == render_contract.get("resolution")
            and render.get("rt_subframes") == render_contract.get("rt_subframes")
            and render.get("pathtracing_total_spp_requested")
            == render_contract.get("pathtracing_total_spp")
            and render.get("pathtracing_total_spp_readback")
            == render_contract.get("pathtracing_total_spp")
            and render.get("pathtracing_spp_requested")
            == render_contract.get("pathtracing_total_spp")
            and render.get("pathtracing_spp_readback")
            == render_contract.get("pathtracing_total_spp")
            and render.get("warmup_updates")
            == render_contract.get("warmup_updates")
            and execution.get("case_id") == case_id
            and execution.get("batch_id") == binding.get("batch_id")
            and execution.get("cohort_plan_sha256") == plan_sha256
            and execution.get("schedule_file_sha256") == plan_snapshot.get("sha256")
            and execution.get("qualification_receipt_sha256")
            == launch.get("qualification_receipt_sha256")
            and execution.get("model_input_record") == expected_model
            and renderer_binding.get("case_id") == case_id
            and renderer_binding.get("batch_id") == binding.get("batch_id")
            and renderer_binding.get("vial_ordinal") == binding.get("vial_ordinal")
        ):
            raise F2StageCohortMaterializationError(
                f"case_receipt_binding_mismatch:{case_id}"
            )
        nuisance_binding = _mapping(
            receipt.get("nuisance_profile_binding"),
            "case_nuisance_profile_binding",
        )
        nuisance_runtime = _mapping(
            receipt.get("nuisance_runtime"), "case_nuisance_runtime"
        )
        if not (
            arguments.get("nuisance_receipt") == nuisance_receipt_path
            and arguments.get("nuisance_receipt_sha256") == nuisance_receipt_sha256
            and arguments.get("nuisance_case_id") == nuisance_case_id
            and nuisance_binding.get("status") == "measured"
            and nuisance_binding.get("receipt_path") == nuisance_receipt_path
            and nuisance_binding.get("receipt_sha256") == nuisance_receipt_sha256
            and nuisance_binding.get("physics_parameters_modified") is False
            and nuisance_runtime.get("status") == "applied"
            and nuisance_runtime.get("session_layer_only") is True
            and nuisance_runtime.get("source_receipt_sha256")
            == nuisance_receipt_sha256
            and (
                nuisance_runtime.get("source_sha256") == nuisance_receipt_sha256
                if "source_sha256" in nuisance_runtime
                else True
            )
            and nuisance_runtime.get("case_id") == nuisance_case_id
            and nuisance_runtime.get("seed") == seed
            and light_rows_match(
                nuisance_runtime.get("authored_light_properties"),
                nuisance_runtime.get("readback_light_properties"),
            )
            and transform_rows_match(
                nuisance_runtime.get("authored_transforms"),
                nuisance_runtime.get("readback_transforms"),
            )
            and all(
                nuisance_runtime.get(field) is not False
                for field in ("camera_pose_unchanged", "camera_transform_unchanged")
                if field in nuisance_runtime
            )
            and all(
                nuisance_runtime.get(field) is not False
                for field in (
                    "non_light_scene_unchanged",
                    "non_light_geometry_unchanged",
                )
                if field in nuisance_runtime
            )
        ):
            raise F2StageCohortMaterializationError(
                f"case_nuisance_contract_mismatch:{case_id}"
            )
        result[case_id] = {
            "path": os.fspath(receipt_path),
            "sha256": receipt_snapshot["sha256"],
            "bytes": receipt_snapshot["bytes"],
            "case_launch_sha256": case_launch_snapshot["sha256"],
            "case_result_sha256": case_result_snapshot["sha256"],
            "runtime_particle_binding": particle_binding,
        }
    return result


def _run_id(plan_sha256: str, batch_id: str) -> str:
    encoded = f"f2-stage-run-v1\0{plan_sha256}\0{batch_id}".encode("utf-8")
    return f"RUN_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _safe_output(value: str | PathLike[str]) -> tuple[Path, Path]:
    candidate = Path(value).expanduser().absolute()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(candidate)
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise F2StageCohortMaterializationError("output_parent_invalid") from error
    if not parent.is_dir() or candidate.name in {"", ".", ".."}:
        raise F2StageCohortMaterializationError("output_path_invalid")
    return candidate, parent


def _publish_directory(
    output: Path,
    parent: Path,
    values: Mapping[str, object],
) -> None:
    partial: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent)
    )
    os.chmod(partial, 0o700)
    try:
        for name, value in values.items():
            _write_private_json(partial / name, value)
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == parent:
                shutil.rmtree(partial)


def materialize_publish_requests(
    *,
    plan_path: str | PathLike[str],
    accepted_qualification_receipt: str | PathLike[str],
    supervisor_root: str | PathLike[str],
    runtime_id: str,
    output_dir: str | PathLike[str],
) -> Path:
    """Write one hash-bound ``f2_stage_batch`` request per source-plan batch."""

    if not isinstance(runtime_id, str) or _RUNTIME_ID.fullmatch(runtime_id) is None:
        raise F2StageCohortMaterializationError("runtime_id_invalid")
    plan, plan_snapshot, model_plan, plan_sha256 = _load_plan(plan_path)
    qualification, qualification_snapshot, render_contract = _qualification_binding(
        accepted_qualification_receipt
    )
    render_contract["nuisance"] = "required_applied_session_layer_only"
    root = _direct_directory(supervisor_root, "supervisor_root")
    launch, launch_snapshot, attempts = _cohort_binding(
        root=root,
        plan=plan,
        plan_snapshot=plan_snapshot,
        plan_sha256=plan_sha256,
        qualification_snapshot=qualification_snapshot,
        qualification_receipt=qualification,
        render_contract=render_contract,
    )
    safe_records = _safe_record_by_case(model_plan)
    expected_case_ids = set(_seed_by_case(plan))
    if set(safe_records) != expected_case_ids:
        raise F2StageCohortMaterializationError("source_case_denominator_mismatch")
    receipt_by_case = _case_receipts(
        root=root,
        plan=plan,
        plan_snapshot=plan_snapshot,
        plan_sha256=plan_sha256,
        launch=launch,
        render_contract=render_contract,
        safe_record_by_case=safe_records,
    )
    if set(receipt_by_case) != expected_case_ids:
        raise F2StageCohortMaterializationError("source_case_denominator_mismatch")
    output, parent = _safe_output(output_dir)

    requests: dict[str, object] = {}
    receipt_rows: list[dict[str, object]] = []
    run_ids: set[str] = set()
    for batch in _batch_rows(plan):
        batch_id = batch.get("batch_id")
        family_id = batch.get("family_id")
        case_ids = batch.get("case_ids")
        batch_role = batch.get("batch_role")
        if (
            not isinstance(batch_id, str)
            or not isinstance(family_id, str)
            or not isinstance(case_ids, list)
            or len(case_ids) != 3
            or batch_role not in {"reference_only", "scored_target"}
        ):
            raise F2StageCohortMaterializationError("source_plan_batch_invalid")
        rows = []
        for case_id in case_ids:
            if not isinstance(case_id, str) or case_id not in safe_records:
                raise F2StageCohortMaterializationError("source_case_record_missing")
            rows.append(safe_records[case_id])
        if any(
            row.get("batch_id") != batch_id
            or row.get("family_id") != family_id
            or row.get("data_origin") != "twin"
            or row.get("split_role") != "train"
            or row.get("split_version") != "inn-7-splits-v2"
            for row in rows
        ):
            raise F2StageCohortMaterializationError(
                "source_case_record_batch_binding_mismatch"
            )
        rows.sort(key=lambda row: int(row["vial_position"]))
        if [row.get("vial_position") for row in rows] != [1, 2, 3]:
            raise F2StageCohortMaterializationError("source_vial_positions_invalid")
        shared_fields = (
            "current_sim_recipe_id",
            "previous_sim_recipe_id",
            "previous_programmed_ml",
            "current_programmed_ml",
            "programmed_dispense_order",
        )
        if any(
            any(row.get(field) != rows[0].get(field) for field in shared_fields)
            for row in rows[1:]
        ):
            raise F2StageCohortMaterializationError(
                "source_case_programmed_recipe_mismatch"
            )
        run_id = _run_id(plan_sha256, batch_id)
        if run_id in run_ids:
            raise F2StageCohortMaterializationError("derived_run_id_collision")
        run_ids.add(run_id)
        request = {
            "schema_version": 1,
            "artifact_type": "f2_stage_batch_publish_request",
            "data_origin": "twin",
            "provider": "brev",
            "runtime_name": "NVIDIA Isaac Sim",
            "runtime_id": runtime_id,
            "family_id": family_id,
            "batch_id": batch_id,
            "run_id": run_id,
            "cohort": batch_role,
            "current_recipe_id": rows[0]["current_sim_recipe_id"],
            "previous_recipe_id": rows[0]["previous_sim_recipe_id"],
            "split_role": "train",
            "split_version": "inn-7-splits-v2",
            "programmed_recipe": {
                "dispense_order": rows[0]["programmed_dispense_order"],
                "previous_ml_by_tank": rows[0]["previous_programmed_ml"],
                "current_ml_by_tank": rows[0]["current_programmed_ml"],
            },
            "vials": [
                {
                    "vial_id": f"V{ordinal}",
                    "receipt_path": receipt_by_case[str(row["case_id"])]["path"],
                    "receipt_sha256": receipt_by_case[str(row["case_id"])][
                        "sha256"
                    ],
                }
                for ordinal, row in enumerate(rows, start=1)
            ],
        }
        try:
            validate_deploy_input_fields(request, require_allowlist=False)
        except ValueError as error:
            raise F2StageCohortMaterializationError(
                "publish_request_contains_forbidden_field"
            ) from error
        name = f"publish-request-{batch_id}.json"
        encoded = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        requests[name] = request
        receipt_rows.append(
            {
                "path": name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
                "batch_id": batch_id,
                "run_id": run_id,
                "source_case_ids": [str(row["case_id"]) for row in rows],
                "source_receipts": [
                    {
                        "vial_id": f"V{ordinal}",
                        **receipt_by_case[str(row["case_id"])],
                    }
                    for ordinal, row in enumerate(rows, start=1)
                ],
            }
        )
    batch_count = len(_batch_rows(plan))
    vial_count = len(_schedule_rows(plan))
    if len(requests) != batch_count or len(run_ids) != batch_count:
        raise F2StageCohortMaterializationError("publish_request_denominator_invalid")
    receipt_rows.sort(key=lambda row: str(row["batch_id"]))
    materialization_receipt = {
        "schema_version": 1,
        "artifact_type": "f2_stage_cohort_request_materialization_receipt",
        "status": "materialized",
        "source_plan_sha256": plan_sha256,
        "source_plan_file_sha256": plan_snapshot["sha256"],
        "source_plan_bytes": plan_snapshot["bytes"],
        "source_family_id": plan["family"]["family_id"],
        "source_cohort_launch_sha256": launch_snapshot["sha256"],
        "accepted_qualification_receipt_sha256": launch[
            "qualification_receipt_sha256"
        ],
        "accepted_qualification_receipt_bytes": qualification_snapshot["bytes"],
        "render_contract": dict(render_contract),
        "runtime_id": runtime_id,
        "batch_count": batch_count,
        "vial_receipt_count": vial_count,
        "supervisor_attempts": attempts,
        "requests": receipt_rows,
        "claim_boundary": (
            "synthetic artifact routing only; source simulation settings and "
            "dispositions are excluded from publication requests"
        ),
    }
    values = {**requests, "request-materialization-receipt.json": materialization_receipt}
    _publish_directory(output, parent, values)
    return output / "request-materialization-receipt.json"


def merge_publication_contexts(
    *,
    request_bundle: str | PathLike[str],
    publication_dirs: Sequence[str | PathLike[str]],
    output_dir: str | PathLike[str],
) -> Path:
    """Merge every request-bound publication context into one strict context."""

    output, parent = _safe_output(output_dir)
    bundle = _direct_directory(request_bundle, "request_bundle")
    materialization_path = bundle / "request-materialization-receipt.json"
    materialization, materialization_snapshot = _load_object(
        materialization_path, "request_materialization_receipt"
    )
    rows = materialization.get("requests")
    if not (
        materialization.get("schema_version") == 1
        and materialization.get("artifact_type")
        == "f2_stage_cohort_request_materialization_receipt"
        and materialization.get("status") == "materialized"
        and isinstance(rows, list)
        and rows
        and materialization.get("batch_count") == len(rows)
        and materialization.get("vial_receipt_count") == 3 * len(rows)
    ):
        raise F2StageCohortMaterializationError(
            "request_materialization_receipt_invalid"
        )
    source_plan_sha = _sha(
        materialization.get("source_plan_sha256"), "source_plan_sha256"
    )
    requests_by_sha: dict[str, dict[str, object]] = {}
    request_batch_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise F2StageCohortMaterializationError(
                "request_materialization_row_invalid"
            )
        name = row.get("path")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("publish-request-")
            or not name.endswith(".json")
        ):
            raise F2StageCohortMaterializationError(
                "request_materialization_path_invalid"
            )
        request_path = bundle / name
        request, snapshot = _load_object(request_path, "publish_request")
        request_sha = _sha(row.get("sha256"), "publish_request_sha256")
        if (
            snapshot.get("sha256") != request_sha
            or snapshot.get("bytes") != row.get("bytes")
            or request.get("schema_version") != 1
            or request.get("artifact_type") != "f2_stage_batch_publish_request"
            or request.get("data_origin") != "twin"
            or request.get("provider") != "brev"
            or request.get("runtime_name") != "NVIDIA Isaac Sim"
            or request.get("split_version") != "inn-7-splits-v2"
            or request.get("batch_id") != row.get("batch_id")
            or request.get("run_id") != row.get("run_id")
            or request.get("batch_id") in request_batch_ids
            or request_sha in requests_by_sha
        ):
            raise F2StageCohortMaterializationError("publish_request_binding_invalid")
        vials = request.get("vials")
        if (
            not isinstance(vials, list)
            or [vial.get("vial_id") for vial in vials if isinstance(vial, Mapping)]
            != ["V1", "V2", "V3"]
        ):
            raise F2StageCohortMaterializationError("publish_request_vials_invalid")
        try:
            validate_deploy_input_fields(request, require_allowlist=False)
        except ValueError as error:
            raise F2StageCohortMaterializationError(
                "publish_request_contains_forbidden_field"
            ) from error
        request_batch_ids.add(str(request["batch_id"]))
        requests_by_sha[request_sha] = {
            "path": request_path,
            "snapshot": snapshot,
            "payload": request,
        }

    if (
        not isinstance(publication_dirs, Sequence)
        or isinstance(publication_dirs, (str, bytes))
        or len(publication_dirs) != len(requests_by_sha)
    ):
        raise F2StageCohortMaterializationError("publication_directory_count_invalid")
    publication_roots = [
        _direct_directory(value, "publication_directory") for value in publication_dirs
    ]
    if len(set(publication_roots)) != len(publication_roots):
        raise F2StageCohortMaterializationError("publication_directories_not_distinct")

    expected_output_names = {
        "f2-runtime-acceptance-receipt.json",
        "f2-stage-capture-manifest.json",
        "stage-batch-context.json",
    }
    consumed_request_hashes: set[str] = set()
    context_rows: list[dict[str, object]] = []
    publication_rows: list[dict[str, object]] = []
    for publication_root in publication_roots:
        publication_path = publication_root / "publication-receipt.json"
        publication, publication_snapshot = _load_object(
            publication_path, "publication_receipt"
        )
        request_sha = _sha(
            publication.get("source_publish_request_sha256"),
            "source_publish_request_sha256",
        )
        request_record = requests_by_sha.get(request_sha)
        if request_record is None or request_sha in consumed_request_hashes:
            raise F2StageCohortMaterializationError(
                "publication_request_binding_invalid"
            )
        request = _mapping(request_record["payload"], "publish_request")
        if not (
            publication.get("schema_version") == 1
            and publication.get("artifact_type")
            == "f2_stage_batch_publication_receipt"
            and publication.get("status") == "published"
            and publication.get("data_origin") == "twin"
            and publication.get("real_data_used") is False
        ):
            raise F2StageCohortMaterializationError("publication_receipt_invalid")
        source_receipts = publication.get("source_receipts")
        request_vials = request.get("vials")
        if not isinstance(source_receipts, list) or not isinstance(request_vials, list):
            raise F2StageCohortMaterializationError(
                "publication_source_receipts_invalid"
            )
        expected_receipts = {
            (
                vial.get("vial_id"),
                vial.get("receipt_path"),
                vial.get("receipt_sha256"),
            )
            for vial in request_vials
            if isinstance(vial, Mapping)
        }
        observed_receipts = {
            (
                vial.get("vial_id"),
                vial.get("receipt_path"),
                vial.get("receipt_sha256"),
            )
            for vial in source_receipts
            if isinstance(vial, Mapping)
            and isinstance(vial.get("verified_artifact_count"), int)
            and not isinstance(vial.get("verified_artifact_count"), bool)
            and int(vial["verified_artifact_count"]) > 0
        }
        if (
            len(source_receipts) != 3
            or len(expected_receipts) != 3
            or observed_receipts != expected_receipts
        ):
            raise F2StageCohortMaterializationError(
                "publication_source_receipts_invalid"
            )
        outputs = publication.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != expected_output_names:
            raise F2StageCohortMaterializationError("publication_outputs_invalid")
        output_snapshots: dict[str, dict[str, object]] = {}
        for name in sorted(expected_output_names):
            record = outputs.get(name)
            if not isinstance(record, Mapping) or record.get("path") != name:
                raise F2StageCohortMaterializationError(
                    "publication_output_binding_invalid"
                )
            actual = hash_regular_file(publication_root / name)
            if (
                record.get("sha256") != actual.get("sha256")
                or record.get("bytes") != actual.get("bytes")
            ):
                raise F2StageCohortMaterializationError(
                    "publication_output_hash_mismatch"
                )
            output_snapshots[name] = actual

        context_path = publication_root / "stage-batch-context.json"
        context, context_snapshot = _load_object(context_path, "stage_batch_context")
        expected_batch = {
            key: request[key]
            for key in (
                "batch_id",
                "cohort",
                "family_id",
                "current_recipe_id",
                "previous_recipe_id",
                "run_id",
                "split_role",
                "programmed_recipe",
            )
        }
        if context != {
            "schema_version": 1,
            "artifact_type": "stage_batch_context",
            "split_version": "inn-7-splits-v2",
            "batches": [expected_batch],
        }:
            raise F2StageCohortMaterializationError(
                "publication_context_request_mismatch"
            )
        try:
            validate_deploy_input_fields(context)
        except ValueError as error:
            raise F2StageCohortMaterializationError(
                "publication_context_contains_forbidden_field"
            ) from error
        consumed_request_hashes.add(request_sha)
        context_rows.append(expected_batch)
        publication_rows.append(
            {
                "batch_id": request["batch_id"],
                "request_path": os.fspath(request_record["path"]),
                "request_sha256": request_sha,
                "publication_receipt_path": os.fspath(publication_path),
                "publication_receipt_sha256": publication_snapshot["sha256"],
                "publication_receipt_bytes": publication_snapshot["bytes"],
                "source_context_path": os.fspath(context_path),
                "source_context_sha256": context_snapshot["sha256"],
                "source_context_bytes": context_snapshot["bytes"],
                "capture_manifest_sha256": output_snapshots[
                    "f2-stage-capture-manifest.json"
                ]["sha256"],
                "runtime_acceptance_sha256": output_snapshots[
                    "f2-runtime-acceptance-receipt.json"
                ]["sha256"],
            }
        )
    if consumed_request_hashes != set(requests_by_sha):
        raise F2StageCohortMaterializationError("publication_request_set_mismatch")

    context_rows.sort(key=lambda row: str(row["batch_id"]))
    publication_rows.sort(key=lambda row: str(row["batch_id"]))
    merged_context = {
        "schema_version": 1,
        "artifact_type": "stage_batch_context",
        "split_version": "inn-7-splits-v2",
        "batches": context_rows,
    }
    try:
        validate_deploy_input_fields(merged_context)
    except ValueError as error:
        raise F2StageCohortMaterializationError(
            "merged_context_contains_forbidden_field"
        ) from error
    merged_bytes = (
        json.dumps(merged_context, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    merge_receipt = {
        "schema_version": 1,
        "artifact_type": "f2_stage_cohort_context_merge_receipt",
        "status": "merged",
        "source_plan_sha256": source_plan_sha,
        "request_materialization_receipt_path": os.fspath(materialization_path),
        "request_materialization_receipt_sha256": materialization_snapshot[
            "sha256"
        ],
        "request_materialization_receipt_bytes": materialization_snapshot["bytes"],
        "batch_count": len(context_rows),
        "source_publications": publication_rows,
        "merged_context": {
            "path": "stage-batch-context.json",
            "sha256": hashlib.sha256(merged_bytes).hexdigest(),
            "bytes": len(merged_bytes),
        },
        "claim_boundary": (
            "synthetic artifact routing only; the merged context contains no "
            "simulation settings or disposition fields"
        ),
    }
    _publish_directory(
        output,
        parent,
        {
            "stage-batch-context.json": merged_context,
            "context-merge-receipt.json": merge_receipt,
        },
    )
    return output / "context-merge-receipt.json"

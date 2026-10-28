#!/usr/bin/env python3
"""Run all or one deterministic shard of the source-bound PhysX train cohort."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))
sys.path.insert(0, os.fspath(ROOT / "scripts"))

import physx_three_pour_sequence as sequence  # noqa: E402
from fluidmix.nuisance_authoring import (  # noqa: E402
    light_rows_match,
    transform_rows_match,
)
from fluidmix.physx_train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)
from fluidmix.physx_varied_cohort_manifest_v3 import (  # noqa: E402
    build_varied_cohort_artifacts,
)
from fluidmix.physx_run_variation import (  # noqa: E402
    run_palette_from_tank_colour_resolution,
    validate_tank_colour_plan_overlay,
)


EXPECTED_PLAN_SHA256 = "37ee900a40206ee956c074697dbbf9ae677e685858ba0b80bf64c3f18c124e8e"
LEGACY_PLAN_ARTIFACT_TYPE = "physx_f2_train_leading_residue_render_plan_v2"
VARIED_V3_PLAN_ARTIFACT_TYPE = "physx_f2_varied_run_residual_render_plan_v3"
VARIED_V4_PLAN_ARTIFACT_TYPE = "physx_f2_varied_run_residual_render_plan_v4"
# Kept as a compatibility alias for callers that used the old single-varied
# constant.  All source selection below uses the explicit contract table.
VARIED_PLAN_ARTIFACT_TYPE = VARIED_V3_PLAN_ARTIFACT_TYPE
VARIED_PLAN_ARTIFACT_TYPES = frozenset(
    {VARIED_V3_PLAN_ARTIFACT_TYPE, VARIED_V4_PLAN_ARTIFACT_TYPE}
)


def _build_v4_source_artifacts(*, repo_root: Path) -> dict[str, dict[str, object]]:
    """Load the additive V4 builder lazily while Task 1 is in flight."""

    try:
        from fluidmix.physx_varied_cohort_manifest_v4 import (
            build_varied_cohort_artifacts,
        )
    except ModuleNotFoundError as error:
        raise ValueError("V4 source-bound manifest builder is unavailable") from error
    return build_varied_cohort_artifacts(repo_root=repo_root)


SOURCE_PLAN_CONTRACTS: dict[str, dict[str, Any]] = {
    LEGACY_PLAN_ARTIFACT_TYPE: {
        "builder": build_train_residual_artifacts,
        "case_count": 18,
        "run_count": None,
        "split_version": "inn-7-splits-v2",
        "contract_kind": None,
        "realtime_video": False,
    },
    VARIED_V3_PLAN_ARTIFACT_TYPE: {
        "builder": build_varied_cohort_artifacts,
        "case_count": 18,
        "run_count": 6,
        "split_version": "whole-run-varied-v3",
        "contract_kind": "varied_run_residual_v3",
        "realtime_video": True,
    },
    VARIED_V4_PLAN_ARTIFACT_TYPE: {
        "builder": _build_v4_source_artifacts,
        "case_count": 30,
        "run_count": 10,
        "split_version": "whole-run-varied-v4",
        "contract_kind": "varied_run_residual_v4",
        "realtime_video": True,
    },
}
QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT = {
    "renderer": "PathTracing",
    "resolution": [640, 480],
    "rt_subframes": 64,
    "pathtracing_total_spp": 64,
    "warmup_updates": 60,
    "capture_stride": sequence.F2_SAFE_CAPTURE_STRIDE_STEPS,
    "video_fps": sequence.F2_VIDEO_PLAYBACK_FPS,
    "wall_limit_seconds": sequence.F2_WALL_LIMIT_SECONDS,
    "render_path": "count_driven_twin_liquid_mesh",
    "count_driven_twin_liquid_mesh": True,
    "report_hz": 120,
    "physics_steps_per_second": 120,
    "physics_substeps_per_report_interval": 1,
    "particle_resolution_profile": "c20_2p5mm",
    "effective_particle_volume_ml": 0.011272866937444935,
    # The receipt path and per-case IDs are recorded beside this stable render
    # contract because they are bound at invocation time.
    "nuisance": "required_applied_session_layer_only",
}
APPROVED_PARENT_RENDERER_COMPATIBILITY = {
    "mode": "approved_parent_train_adapter_delta",
    "qualification_renderer_sha256": (
        "2e357b0b09a1fce901889b030291e59ed1540dc7b322fb3d203432d3fd9f6b3b"
    ),
    "cohort_renderer_sha256": (
        "cfd446262c91ddb4cbfc8a34659711565206faac14c657867a75b311ac982038"
    ),
    "qualification_renderer_commit": "3903789c78d1b4accc241a413db964657e143648",
    "cohort_renderer_commit": "bound_by_cohort_renderer_sha256",
    "declared_delta_scope": [
        "additive_v4_source_plan_adapter",
        "additive_tank_colour_resolution_adapter",
    ],
    "qualification_render_path_changed": False,
}
OUTER_TIMEOUT_SECONDS = (
    QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["wall_limit_seconds"] + 600
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{description} must be an existing absolute file")

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{description} contains non-finite value {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return value


def _write_json_new(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o640,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _write_text_new(path: Path, value: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o640,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _load_bound_plan(path: Path, declared_sha256: str) -> dict[str, Any]:
    observed = _read_json_object(path, "train render plan")
    artifact_type = observed.get("artifact_type")
    contract = SOURCE_PLAN_CONTRACTS.get(artifact_type)
    if contract is None:
        raise ValueError("train render plan artifact type is not source-bound")
    if (
        artifact_type == LEGACY_PLAN_ARTIFACT_TYPE
        and declared_sha256 != EXPECTED_PLAN_SHA256
    ):
        raise ValueError(
            "train render plan SHA-256 is not the frozen source-bound plan"
        )
    if _canonical_sha256(observed) != declared_sha256:
        raise ValueError("train render plan canonical SHA-256 mismatch")
    tank_colour_resolution = validate_tank_colour_plan_overlay(observed)
    expected = contract["builder"](repo_root=ROOT)["render-schedule-plan.json"]
    if tank_colour_resolution is not None:
        expected = deepcopy(expected)
        expected["tank_colour_resolution"] = deepcopy(tank_colour_resolution)
        expected["run_palette"] = run_palette_from_tank_colour_resolution(
            tank_colour_resolution
        )
    if observed != expected:
        raise ValueError("train render plan differs from the source-bound plan")
    schedules = observed.get("execution_schedules")
    if not isinstance(schedules, list) or len(schedules) != contract["case_count"]:
        raise ValueError(
            "train render plan must contain exactly "
            f"{contract['case_count']} schedules"
        )
    return observed


def _case_run_palettes(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return per-case run palette launch bindings for the varied plan."""

    tank_colour_resolution = validate_tank_colour_plan_overlay(plan)
    if tank_colour_resolution is not None:
        palette = run_palette_from_tank_colour_resolution(tank_colour_resolution)
        case_ids = [
            row.get("case_binding", {}).get("case_id")
            for row in plan.get("execution_schedules", [])
            if isinstance(row, dict)
        ]
        if (
            not case_ids
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
        ):
            raise ValueError("tank-colour plan has invalid case bindings")
        return {
            case_id: {
                "tank_colour_resolution": deepcopy(tank_colour_resolution),
                "palette_sha256": palette["palette_sha256"],
            }
            for case_id in case_ids
        }
    if plan.get("artifact_type") not in VARIED_PLAN_ARTIFACT_TYPES:
        return {}
    palette_by_run: dict[str, dict[str, Any]] = {}
    for run in plan.get("runs", []):
        if not isinstance(run, dict) or not isinstance(run.get("palette"), dict):
            raise ValueError("varied run palette provenance is invalid")
        run_id = run.get("run_id")
        run_seed = run.get("run_seed")
        palette_sha256 = run["palette"].get("palette_sha256")
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in palette_by_run
            or isinstance(run_seed, bool)
            or not isinstance(run_seed, int)
            or not isinstance(palette_sha256, str)
            or len(palette_sha256) != 64
        ):
            raise ValueError("varied run palette provenance is invalid")
        palette_by_run[run_id] = {
            "run_palette_id": run_id,
            "run_palette_seed": run_seed,
            "palette_sha256": palette_sha256,
        }
    result: dict[str, dict[str, Any]] = {}
    for batch in plan.get("batches", []):
        if not isinstance(batch, dict):
            raise ValueError("varied render batch is invalid")
        case_ids = batch.get("case_ids")
        palette_id = batch.get("run_palette_id")
        palette_seed = batch.get("run_palette_seed")
        if (
            not isinstance(case_ids, list)
            or not case_ids
            or not isinstance(palette_id, str)
            or not palette_id
            or isinstance(palette_seed, bool)
            or not isinstance(palette_seed, int)
        ):
            raise ValueError("varied render batch lacks a run palette binding")
        expected_palette = palette_by_run.get(palette_id)
        if (
            expected_palette is None
            or expected_palette["run_palette_seed"] != palette_seed
        ):
            raise ValueError("varied batch palette differs from its run")
        for case_id in case_ids:
            if not isinstance(case_id, str) or case_id in result:
                raise ValueError("varied render plan has invalid case palette bindings")
            result[case_id] = dict(expected_palette)
    schedule_ids = {
        row["case_binding"]["case_id"]
        for row in plan.get("execution_schedules", [])
    }
    if set(result) != schedule_ids:
        raise ValueError("varied schedule and run palette bindings differ")
    return result


def _case_seeds(plan: dict[str, Any]) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for batch in plan["batches"]:
        case_ids = batch.get("case_ids")
        render_seeds = batch.get("render_seed_by_vial")
        if (
            not isinstance(case_ids, list)
            or not isinstance(render_seeds, list)
            or len(case_ids) != len(render_seeds)
        ):
            raise ValueError("train render batch has invalid case/seed bindings")
        for case_id, seed in zip(case_ids, render_seeds, strict=True):
            if (
                not isinstance(case_id, str)
                or not case_id.startswith("CASE_")
                or case_id in seeds
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                raise ValueError("train render batch has invalid case/seed values")
            seeds[case_id] = seed
    schedule_ids = [row["case_binding"]["case_id"] for row in plan["execution_schedules"]]
    if set(schedule_ids) != set(seeds) or len(schedule_ids) != len(seeds):
        raise ValueError("train schedules and batch seed bindings differ")
    return seeds


def _case_nuisance_ids(plan: dict[str, Any]) -> dict[str, str]:
    """Bind every planned render case to its explicit nuisance group ID."""

    result: dict[str, str] = {}
    for batch in plan.get("batches", []):
        case_ids = batch.get("case_ids")
        nuisance_ids = batch.get("nuisance_group_ids")
        if (
            not isinstance(case_ids, list)
            or not isinstance(nuisance_ids, list)
            or len(case_ids) != len(nuisance_ids)
        ):
            raise ValueError("train render batch has invalid nuisance bindings")
        for case_id, nuisance_id in zip(case_ids, nuisance_ids, strict=True):
            if (
                not isinstance(case_id, str)
                or not isinstance(nuisance_id, str)
                or not nuisance_id
                or case_id in result
                or nuisance_id in result.values()
            ):
                raise ValueError("train render batch has invalid nuisance IDs")
            result[case_id] = nuisance_id
    schedule_ids = [
        row["case_binding"]["case_id"]
        for row in plan.get("execution_schedules", [])
    ]
    if set(schedule_ids) != set(result) or len(schedule_ids) != len(result):
        raise ValueError("train schedules and nuisance bindings differ")
    return result


def _source_plan_denominators(plan: dict[str, Any]) -> dict[str, Any]:
    """Read and cross-check counts from the selected source-bound render plan."""

    artifact_type = plan.get("artifact_type")
    contract = SOURCE_PLAN_CONTRACTS.get(artifact_type)
    if contract is None:
        raise ValueError("train render plan artifact type is not source-bound")
    schedules = plan.get("execution_schedules")
    if not isinstance(schedules, list):
        raise ValueError("train render plan schedules are invalid")
    case_ids: list[str] = []
    keep_count = 0
    for schedule in schedules:
        binding = schedule.get("case_binding") if isinstance(schedule, dict) else None
        case_id = binding.get("case_id") if isinstance(binding, dict) else None
        residual = binding.get("residual_volume_ml") if isinstance(binding, dict) else None
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
            or isinstance(residual, bool)
            or not isinstance(residual, (int, float))
        ):
            raise ValueError("train render plan case denominator bindings are invalid")
        case_ids.append(case_id)
        keep_count += float(residual) == 0.0
    discard_count = len(case_ids) - keep_count
    runs = plan.get("runs")
    batches = plan.get("batches")
    run_count = len(runs) if isinstance(runs, list) else None
    batch_count = len(batches) if isinstance(batches, list) else None
    split_counts: dict[str, int] = {}
    if artifact_type in VARIED_PLAN_ARTIFACT_TYPES:
        if not isinstance(runs, list) or not isinstance(batches, list):
            raise ValueError("varied render plan runs and batches are invalid")
        run_ids = set()
        batch_case_ids: set[str] = set()
        for run in runs:
            if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
                raise ValueError("varied render plan run bindings are invalid")
            run_ids.add(run["run_id"])
        for batch in batches:
            if not isinstance(batch, dict):
                raise ValueError("varied render plan batch bindings are invalid")
            split = batch.get("partition")
            batch_run = batch.get("run_id")
            batch_case_list = batch.get("case_ids")
            if (
                not isinstance(split, str)
                or not split
                or batch_run not in run_ids
                or not isinstance(batch_case_list, list)
            ):
                raise ValueError("varied render plan batch bindings are invalid")
            split_counts[split] = split_counts.get(split, 0) + len(batch_case_list)
            for case_id in batch_case_list:
                if not isinstance(case_id, str) or case_id in batch_case_ids:
                    raise ValueError("varied render plan batch case bindings are invalid")
                batch_case_ids.add(case_id)
        if batch_case_ids != set(case_ids):
            raise ValueError("varied render plan batches and schedules differ")
        if run_count != contract["run_count"]:
            raise ValueError("varied render plan run denominator drift")
        if batch_count != run_count:
            raise ValueError("varied render plan batch denominator drift")

    declared = plan.get("denominators")
    if not isinstance(declared, dict):
        raise ValueError("train render plan denominators are missing")
    expected_declared = {
        "vials": len(case_ids),
        "keep_vials": keep_count,
        "discard_vials": discard_count,
    }
    if run_count is not None:
        expected_declared.update(runs=run_count, batches=batch_count)
    for name, expected in expected_declared.items():
        if declared.get(name) != expected:
            raise ValueError(f"train render plan denominator drift: {name}")
    if artifact_type in VARIED_PLAN_ARTIFACT_TYPES:
        for split, expected in split_counts.items():
            if declared.get(f"{split}_vials") != expected:
                raise ValueError(f"train render plan denominator drift: {split}_vials")
        expected_split_version = contract["split_version"]
        # The split version is carried in V4's top-level plan and in the V3
        # model plan; when present, it is source-bound evidence and must agree.
        if "split_version" in plan and plan["split_version"] != expected_split_version:
            raise ValueError("varied render plan split version drift")
    return {
        "run_count": run_count,
        "batch_count": batch_count,
        "split_counts": split_counts,
        "case_count": len(case_ids),
        "keep_count": keep_count,
        "discard_count": discard_count,
    }


def _case_output_contract(plan: dict[str, Any]) -> dict[str, Any]:
    contract = SOURCE_PLAN_CONTRACTS.get(plan.get("artifact_type"))
    if contract is None:
        raise ValueError("train render plan artifact type is not source-bound")
    denominators = _source_plan_denominators(plan)
    realtime_video = bool(contract["realtime_video"])
    cases: dict[str, dict[str, int]] = {}
    for saved_schedule in plan["execution_schedules"]:
        schedule = deepcopy(saved_schedule)
        case_binding = schedule.get("case_binding")
        if not isinstance(case_binding, dict):
            raise ValueError("train execution schedule lacks its case binding")
        case_id = case_binding.get("case_id")
        residual_volume_ml = case_binding.get("residual_volume_ml")
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise ValueError("train execution schedules have invalid case IDs")
        if isinstance(residual_volume_ml, bool) or not isinstance(
            residual_volume_ml, (int, float)
        ):
            raise ValueError("train execution schedule lacks its residual provenance")
        schedule["timing"]["capture_hz"] = 10
        capture_schedule = sequence.build_runtime_capture_schedule(
            schedule,
            capture_stride_steps=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
                "capture_stride"
            ],
        )
        report_interval_count = int(capture_schedule["final_plus_3_0_step"])
        if realtime_video:
            video_steps = sequence._build_realtime_video_capture_steps(
                final_step=report_interval_count,
                report_hz=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["report_hz"],
                video_fps=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["video_fps"],
            )
            source_capture_count = len(schedule["pours"])
            decoded_video_frame_count = len(video_steps)
        else:
            frame_steps = [
                int(row["physics_step"])
                for row in capture_schedule["captures"]
            ]
            video = sequence.build_time_preserving_video_repeat_plan(
                frame_steps,
                physics_hz=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["report_hz"],
                playback_fps=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["video_fps"],
            )
            source_capture_count = len(frame_steps)
            decoded_video_frame_count = int(video["video_frame_count"])
        cases[case_id] = {
            "report_interval_count": report_interval_count,
            "physics_callback_count": (
                QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
                    "physics_substeps_per_report_interval"
                ]
                * report_interval_count
            ),
            "source_capture_count": source_capture_count,
            "decoded_video_frame_count": decoded_video_frame_count,
        }
    source_capture_count = sum(
        row["source_capture_count"] for row in cases.values()
    )
    decoded_video_frame_count = sum(
        row["decoded_video_frame_count"] for row in cases.values()
    )
    legacy_denominators_drifted = (
        not realtime_video
        and (
            source_capture_count != 381
            or decoded_video_frame_count != 7074
        )
    )
    if (
        len(cases) != contract["case_count"]
        or len(cases) != denominators["case_count"]
        or denominators["keep_count"] + denominators["discard_count"] != len(cases)
        or legacy_denominators_drifted
    ):
        raise ValueError("frozen train cohort output denominator drift")
    return {
        "cases": cases,
        "case_count": len(cases),
        "keep_count": denominators["keep_count"],
        "discard_count": denominators["discard_count"],
        "run_count": denominators["run_count"],
        "batch_count": denominators["batch_count"],
        "split_counts": denominators["split_counts"],
        "source_capture_count": source_capture_count,
        "decoded_video_frame_count": decoded_video_frame_count,
    }


def _selected_schedule_rows(
    plan: dict[str, Any], *, shard_count: int, shard_index: int
) -> list[tuple[int, dict[str, Any]]]:
    """Return stable global-ordinal rows for one modulo shard."""

    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    schedule_count = plan.get("execution_schedules")
    if not isinstance(schedule_count, list) or not schedule_count:
        raise ValueError("train render plan schedules are missing")
    if shard_count < 1 or shard_count > len(schedule_count):
        raise ValueError(
            f"shard_count must be between 1 and {len(schedule_count)}"
        )
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("shard_index must be an integer")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    rows = [
        (ordinal, schedule)
        for ordinal, schedule in enumerate(plan["execution_schedules"], start=1)
        if (ordinal - 1) % shard_count == shard_index
    ]
    if not rows:
        raise ValueError("shard selection is empty")
    return rows


def _selected_output_contract(
    full_contract: dict[str, Any],
    selected_rows: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Project the frozen full-cohort denominators onto selected cases."""

    selected_cases: dict[str, dict[str, int]] = {}
    keep_count = 0
    discard_count = 0
    run_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    for _, schedule in selected_rows:
        binding = schedule["case_binding"]
        case_id = binding["case_id"]
        selected_cases[case_id] = deepcopy(full_contract["cases"][case_id])
        run_id = binding.get("run_id")
        split_role = binding.get("split_role")
        if isinstance(run_id, str):
            run_ids.add(run_id)
        if isinstance(split_role, str):
            split_counts[split_role] = split_counts.get(split_role, 0) + 1
        if float(binding["residual_volume_ml"]) == 0.0:
            keep_count += 1
        else:
            discard_count += 1
    return {
        "cases": selected_cases,
        "case_count": len(selected_cases),
        "keep_count": keep_count,
        "discard_count": discard_count,
        "run_count": len(run_ids) if run_ids else None,
        "batch_count": len(run_ids) if run_ids else None,
        "split_counts": split_counts,
        "source_capture_count": sum(
            row["source_capture_count"] for row in selected_cases.values()
        ),
        "decoded_video_frame_count": sum(
            row["decoded_video_frame_count"] for row in selected_cases.values()
        ),
    }


def _validate_paths(args: argparse.Namespace) -> None:
    for name in ("output_root", "stage", "renderer_script", "python_executable", "host_lock"):
        path = getattr(args, name)
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if not args.stage.is_file():
        raise ValueError("stage must be an existing file")
    if not args.renderer_script.is_file():
        raise ValueError("renderer_script must be an existing file")
    if not args.python_executable.is_file() or not os.access(args.python_executable, os.X_OK):
        raise ValueError("python_executable must be an executable file")
    if not args.output_root.parent.is_dir():
        raise ValueError("output_root parent must exist")
    if not args.host_lock.parent.is_dir():
        raise ValueError("host_lock parent must exist")


def _bind_inputs(args: argparse.Namespace) -> dict[str, Any]:
    _validate_paths(args)
    plan = _load_bound_plan(args.plan, args.plan_sha256)
    nuisance_binding = sequence.bind_external_nuisance_receipt(
        args.nuisance_receipt,
        args.nuisance_receipt_sha256,
    )
    nuisance_case_ids = _case_nuisance_ids(plan)
    qualification = sequence._bind_accepted_qualification_receipt(
        args.accepted_qualification_receipt,
        args.accepted_qualification_receipt_sha256,
        stage_path=args.stage,
        rt_subframes=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["rt_subframes"],
        pathtracing_total_spp=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "pathtracing_total_spp"
        ],
        count_driven_twin_liquid_mesh=True,
        particle_resolution_profile_id=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "particle_resolution_profile"
        ],
    )
    if (
        qualification.get("physics_substeps")
        != QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "physics_substeps_per_report_interval"
        ]
        or qualification.get("physics_steps_per_second")
        != QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["physics_steps_per_second"]
    ):
        raise ValueError(
            "accepted qualification timing differs from the frozen cohort contract"
        )
    renderer_sha256 = _sha256(args.renderer_script)
    qualification_renderer_sha256 = qualification["runner_script_sha256"]
    if renderer_sha256 == qualification_renderer_sha256:
        renderer_compatibility = {
            "mode": "exact_renderer_sha256",
            "qualification_renderer_sha256": qualification_renderer_sha256,
            "cohort_renderer_sha256": renderer_sha256,
            "declared_delta_scope": [],
            "qualification_render_path_changed": False,
        }
    elif (
        qualification_renderer_sha256
        == APPROVED_PARENT_RENDERER_COMPATIBILITY[
            "qualification_renderer_sha256"
        ]
        and renderer_sha256
        == APPROVED_PARENT_RENDERER_COMPATIBILITY["cohort_renderer_sha256"]
    ):
        renderer_compatibility = deepcopy(
            APPROVED_PARENT_RENDERER_COMPATIBILITY
        )
    else:
        raise ValueError("renderer script hash differs from the accepted qualification")
    run_palette_by_case = _case_run_palettes(plan)
    setattr(args, "_run_palette_by_case", run_palette_by_case)
    tank_colour_resolution = validate_tank_colour_plan_overlay(plan)
    render_contract = dict(QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT)
    source_contract = SOURCE_PLAN_CONTRACTS[plan["artifact_type"]]
    if plan.get("artifact_type") in VARIED_PLAN_ARTIFACT_TYPES:
        render_contract.update(
            realtime_video=True,
            video_renderer="RaytracedLighting",
            stage_still_renderer="PathTracing",
            height_banded_body=True,
            run_palette_required=True,
            contract_kind=source_contract["contract_kind"],
            split_version=source_contract["split_version"],
        )
    else:
        render_contract.update(
            contract_kind=source_contract["contract_kind"],
            split_version=source_contract["split_version"],
        )
    result = {
        "plan": plan,
        "case_seeds": _case_seeds(plan),
        "nuisance_case_ids": nuisance_case_ids,
        "run_palette_by_case": run_palette_by_case,
        "nuisance_binding": nuisance_binding,
        "nuisance_plan": {
            "mapping_version": "f2-measured-percentile-to-existing-lights-v1",
            "mapping_status": "uncalibrated_synthetic_nuisance_mapping",
            "real_mapping_claimed": False,
            "session_layer_only": True,
            "excluded_from_model_inputs": True,
            "excluded_from_deploy_inputs": True,
            "receipt_path": os.fspath(args.nuisance_receipt),
            "receipt_sha256": args.nuisance_receipt_sha256,
            "case_id_by_case": nuisance_case_ids,
        },
        "output_contract": _case_output_contract(plan),
        "plan_file_sha256": _sha256(args.plan),
        "qualification": qualification,
        "stage_sha256": _sha256(args.stage),
        "renderer_sha256": renderer_sha256,
        "renderer_qualification_compatibility": renderer_compatibility,
        "supervisor_sha256": _sha256(Path(__file__).resolve()),
        "render_contract": render_contract,
    }
    if tank_colour_resolution is not None:
        result["tank_colour_resolution"] = deepcopy(tank_colour_resolution)
    return result


def _ancestor_pids(pid: int) -> set[int]:
    ancestors = {pid}
    current = pid
    while current > 1:
        try:
            fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
            parent = int(fields[3])
        except (OSError, UnicodeError, ValueError, IndexError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def _find_other_isaac_processes() -> list[dict[str, Any]]:
    excluded = _ancestor_pids(os.getpid())
    markers = (
        "physx_three_pour_sequence.py",
        "isaac-sim",
        "isaacsim.exp",
        "omni.kit",
    )
    found: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if command and any(marker in command for marker in markers):
            found.append({"pid": int(entry.name), "command": command[:2048]})
    return sorted(found, key=lambda row: row["pid"])


@contextmanager
def _exclusive_host_lock(path: Path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o640)
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another cohort supervisor holds the host lock") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={_utc_now()}\n")
        handle.flush()
        yield


def _cohort_manifest(args: argparse.Namespace, binding: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "artifact_type": "physx_train_residual_serial_cohort_launch",
        "plan_path": os.fspath(args.plan),
        "plan_sha256": args.plan_sha256,
        "plan_file_sha256": binding["plan_file_sha256"],
        "qualification_receipt_path": os.fspath(args.accepted_qualification_receipt),
        "qualification_receipt_sha256": args.accepted_qualification_receipt_sha256,
        "stage_path": os.fspath(args.stage),
        "stage_sha256": binding["stage_sha256"],
        "renderer_script": os.fspath(args.renderer_script),
        "renderer_script_sha256": binding["renderer_sha256"],
        "renderer_qualification_compatibility": deepcopy(
            binding["renderer_qualification_compatibility"]
        ),
        "python_executable": os.fspath(args.python_executable),
        "supervisor_script_sha256": binding["supervisor_sha256"],
        "render_contract": dict(binding["render_contract"]),
        "nuisance_plan": deepcopy(binding["nuisance_plan"]),
        "nuisance_receipt_path": os.fspath(args.nuisance_receipt),
        "nuisance_receipt_sha256": args.nuisance_receipt_sha256,
        "nuisance_case_id_by_case": deepcopy(binding["nuisance_case_ids"]),
        "expected_output_contract": deepcopy(binding["output_contract"]),
        "case_ids": [
            row["case_binding"]["case_id"]
            for row in binding["plan"]["execution_schedules"]
        ],
        "created_at": _utc_now(),
    }
    if "tank_colour_resolution" in binding:
        manifest["tank_colour_resolution"] = deepcopy(
            binding["tank_colour_resolution"]
        )
    return manifest


_HOST_PATH_MANIFEST_FIELDS = (
    "plan_path",
    "qualification_receipt_path",
    "stage_path",
    "renderer_script",
    "python_executable",
    "nuisance_receipt_path",
)
_HOST_PATH_COMMAND_FLAGS = (
    "--out",
    "--stage",
    "--train-render-plan",
    "--accepted-qualification-receipt",
    "--nuisance-receipt",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_path_string(value: object) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def _host_relocated_manifest(
    observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any] | None:
    """Compare immutable cohort bindings while allowing host-local path relocation."""

    observed_copy = deepcopy(observed)
    expected_copy = deepcopy(expected)
    observed_copy.pop("created_at", None)
    expected_copy.pop("created_at", None)
    source_paths: dict[str, str] = {}
    merge_paths: dict[str, str] = {}
    for field in _HOST_PATH_MANIFEST_FIELDS:
        source = observed_copy.pop(field, None)
        merge = expected_copy.pop(field, None)
        if not _absolute_path_string(source) or not _absolute_path_string(merge):
            return None
        source_paths[field] = source
        merge_paths[field] = merge

    source_supervisor_sha256 = observed_copy.pop(
        "supervisor_script_sha256", None
    )
    merge_supervisor_sha256 = expected_copy.pop(
        "supervisor_script_sha256", None
    )
    if not _is_sha256(source_supervisor_sha256) or not _is_sha256(
        merge_supervisor_sha256
    ):
        return None

    for payload, paths, label in (
        (observed_copy, source_paths, "nuisance_plan_receipt_path"),
        (expected_copy, merge_paths, "nuisance_plan_receipt_path"),
    ):
        nuisance_plan = payload.get("nuisance_plan")
        if not isinstance(nuisance_plan, dict):
            return None
        nuisance_plan = deepcopy(nuisance_plan)
        receipt_path = nuisance_plan.pop("receipt_path", None)
        if not _absolute_path_string(receipt_path):
            return None
        paths[label] = receipt_path
        payload["nuisance_plan"] = nuisance_plan

    if observed_copy != expected_copy:
        return None
    return {
        "mode": "hash_bound",
        "source_paths": source_paths,
        "merge_paths": merge_paths,
        "source_supervisor_script_sha256": source_supervisor_sha256,
        "merge_supervisor_script_sha256": merge_supervisor_sha256,
        "immutable_hashes": {
            field: observed.get(field)
            for field in (
                "plan_sha256",
                "plan_file_sha256",
                "qualification_receipt_sha256",
                "stage_sha256",
                "renderer_script_sha256",
                "nuisance_receipt_sha256",
            )
        },
    }


def _nuisance_binding_matches_with_host_relocation(
    observed: object, expected: object
) -> bool:
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    observed_copy = deepcopy(observed)
    expected_copy = deepcopy(expected)
    observed_path = observed_copy.pop("receipt_path", None)
    expected_path = expected_copy.pop("receipt_path", None)
    return (
        _absolute_path_string(observed_path)
        and _absolute_path_string(expected_path)
        and observed_copy == expected_copy
    )


def _commands_match_with_host_relocation(
    observed: object, expected: list[str], *, case_id: str
) -> bool:
    if (
        not isinstance(observed, list)
        or len(observed) != len(expected)
        or not all(isinstance(value, str) for value in observed)
        or len(observed) < 2
    ):
        return False
    observed_copy = list(observed)
    expected_copy = list(expected)
    for index, token in ((0, "<python>"), (1, "<renderer>")):
        if not _absolute_path_string(observed_copy[index]) or not _absolute_path_string(
            expected_copy[index]
        ):
            return False
        observed_copy[index] = token
        expected_copy[index] = token
    for flag in _HOST_PATH_COMMAND_FLAGS:
        if observed_copy.count(flag) != 1 or expected_copy.count(flag) != 1:
            return False
        observed_index = observed_copy.index(flag) + 1
        expected_index = expected_copy.index(flag) + 1
        source_path = observed_copy[observed_index]
        merge_path = expected_copy[expected_index]
        if not _absolute_path_string(source_path) or not _absolute_path_string(
            merge_path
        ):
            return False
        if flag == "--out" and Path(source_path).parts[-3:] != (
            "cases",
            case_id,
            "app-output",
        ):
            return False
        token = f"<{flag[2:]}>"
        observed_copy[observed_index] = token
        expected_copy[expected_index] = token
    return observed_copy == expected_copy


def _case_command(
    args: argparse.Namespace,
    *,
    case_id: str,
    seed: int,
    nuisance_case_id: str,
    app_output: Path,
) -> list[str]:
    command = [
        os.fspath(args.python_executable),
        os.fspath(args.renderer_script),
        "--out",
        os.fspath(app_output),
        "--stage",
        os.fspath(args.stage),
        "--seed",
        str(seed),
        "--solver-position-iterations",
        "16",
        "--water-pbd-preset",
        "--renderer",
        "PathTracing",
        "--resolution",
        "640x480",
        "--rt-subframes",
        "64",
        "--pathtracing-total-spp",
        "64",
        "--warmup",
        "60",
        "--capture-stride",
        str(QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["capture_stride"]),
        "--video-fps",
        str(QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["video_fps"]),
        "--physics-substeps",
        str(
            QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
                "physics_substeps_per_report_interval"
            ]
        ),
        "--particle-resolution-profile",
        QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["particle_resolution_profile"],
        "--recipe8-effective-particle-volume-ml",
        str(
            QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
                "effective_particle_volume_ml"
            ]
        ),
        "--wall-limit-seconds",
        str(QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["wall_limit_seconds"]),
        "--count-driven-twin-liquid-mesh",
        "--train-render-plan",
        os.fspath(args.plan),
        "--train-render-plan-sha256",
        args.plan_sha256,
        "--train-case-id",
        case_id,
        "--accepted-qualification-receipt",
        os.fspath(args.accepted_qualification_receipt),
        "--accepted-qualification-receipt-sha256",
        args.accepted_qualification_receipt_sha256,
        "--nuisance-receipt",
        os.fspath(args.nuisance_receipt),
        "--nuisance-receipt-sha256",
        args.nuisance_receipt_sha256,
        "--nuisance-case-id",
        nuisance_case_id,
    ]
    palette_by_case = getattr(args, "_run_palette_by_case", {})
    if case_id in palette_by_case:
        palette = palette_by_case[case_id]
        command.extend(["--height-banded-body", "--realtime-video"])
        if "tank_colour_resolution" not in palette:
            command.extend(
                [
                    "--run-palette-id",
                    str(palette["run_palette_id"]),
                    "--run-palette-seed",
                    str(palette["run_palette_seed"]),
                ]
            )
    return command


def _case_receipt_failures(
    receipt_path: Path,
    *,
    args: argparse.Namespace,
    binding: dict[str, Any],
    case_id: str,
    seed: int,
    nuisance_case_id: str,
    allow_host_path_relocation: bool = False,
) -> list[str]:
    try:
        receipt = _read_json_object(receipt_path, "case render receipt")
    except ValueError as error:
        return [str(error)]
    arguments = receipt.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    provenance = receipt.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    render = receipt.get("render")
    render = render if isinstance(render, dict) else {}
    schedule = receipt.get("execution_schedule")
    schedule = schedule if isinstance(schedule, dict) else {}
    timing = receipt.get("timing")
    timing = timing if isinstance(timing, dict) else {}
    ablation = receipt.get("ablation")
    ablation = ablation if isinstance(ablation, dict) else {}
    nuisance_runtime = receipt.get("nuisance_runtime")
    nuisance_runtime = nuisance_runtime if isinstance(nuisance_runtime, dict) else {}
    expected_output = binding["output_contract"]["cases"][case_id]
    palette = binding["run_palette_by_case"].get(case_id)
    realtime_video = palette is not None
    receipt_plan = receipt.get("plan")
    receipt_plan = receipt_plan if isinstance(receipt_plan, dict) else {}
    receipt_palette = receipt_plan.get("run_palette")
    receipt_palette = (
        receipt_palette if isinstance(receipt_palette, dict) else {}
    )
    checks = {
        "status": receipt.get("status") == "pass",
        "seed": arguments.get("seed") == seed,
        "physics_substeps": arguments.get("physics_substeps")
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "physics_substeps_per_report_interval"
        ],
        "particle_resolution_profile": arguments.get(
            "particle_resolution_profile"
        )
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "particle_resolution_profile"
        ],
        "effective_particle_volume_ml": arguments.get(
            "recipe8_effective_particle_volume_ml"
        )
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "effective_particle_volume_ml"
        ],
        "plan_sha256": arguments.get("train_render_plan_sha256") == args.plan_sha256,
        "case_id": arguments.get("train_case_id") == case_id,
        "qualification_sha256": arguments.get(
            "accepted_qualification_receipt_sha256"
        )
        == args.accepted_qualification_receipt_sha256,
        "count_driven_argument": arguments.get(
            "count_driven_twin_liquid_mesh"
        )
        is True,
        "realtime_video_argument": (
            arguments.get("realtime_video") is True
            if realtime_video
            else not arguments.get("realtime_video", False)
        ),
        "height_banded_body_argument": (
            arguments.get("height_banded_body") is True
            if realtime_video
            else not arguments.get("height_banded_body", False)
        ),
        "run_palette_id_argument": (
            arguments.get("run_palette_id") == palette["run_palette_id"]
            if realtime_video
            else arguments.get("run_palette_id") is None
        ),
        "run_palette_seed_argument": (
            arguments.get("run_palette_seed") == palette["run_palette_seed"]
            if realtime_video
            else arguments.get("run_palette_seed") is None
        ),
        "run_palette_plan_binding": (
            receipt_palette.get("palette_sha256") == palette["palette_sha256"]
            if realtime_video
            else not receipt_palette
        ),
        "render_path": receipt.get("render_path")
        == "count_driven_twin_liquid_mesh",
        "capture_stride": arguments.get("capture_stride")
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["capture_stride"],
        "video_fps": arguments.get("video_fps")
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["video_fps"],
        "wall_limit_seconds": arguments.get("wall_limit_seconds")
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["wall_limit_seconds"],
        "argument_resolution": arguments.get("resolution") == "640x480",
        "argument_rt_subframes": arguments.get("rt_subframes") == 64,
        "argument_pathtracing_total_spp": arguments.get(
            "pathtracing_total_spp"
        )
        == 64,
        "nuisance_receipt_argument": (
            arguments.get("nuisance_receipt") == os.fspath(args.nuisance_receipt)
            or (
                allow_host_path_relocation
                and _absolute_path_string(arguments.get("nuisance_receipt"))
            )
        ),
        "nuisance_receipt_sha256_argument": arguments.get(
            "nuisance_receipt_sha256"
        )
        == args.nuisance_receipt_sha256,
        "nuisance_case_id_argument": arguments.get("nuisance_case_id")
        == nuisance_case_id,
        "nuisance_binding": (
            receipt.get("nuisance_profile_binding") == binding["nuisance_binding"]
            or (
                allow_host_path_relocation
                and _nuisance_binding_matches_with_host_relocation(
                    receipt.get("nuisance_profile_binding"),
                    binding["nuisance_binding"],
                )
            )
        ),
        "nuisance_runtime_applied": isinstance(
            receipt.get("nuisance_runtime"), dict
        )
        and receipt["nuisance_runtime"].get("status") == "applied",
        "nuisance_runtime_session_layer_only": isinstance(
            receipt.get("nuisance_runtime"), dict
        )
        and receipt["nuisance_runtime"].get("session_layer_only") is True,
        "nuisance_runtime_source_receipt_sha256": isinstance(
            receipt.get("nuisance_runtime"), dict
        )
        and receipt["nuisance_runtime"].get("source_receipt_sha256")
        == args.nuisance_receipt_sha256,
        "nuisance_runtime_source_sha256": (
            nuisance_runtime.get("source_sha256") == args.nuisance_receipt_sha256
            if "source_sha256" in nuisance_runtime
            else True
        ),
        "nuisance_runtime_case_id": isinstance(
            receipt.get("nuisance_runtime"), dict
        )
        and receipt["nuisance_runtime"].get("case_id") == nuisance_case_id,
        "nuisance_runtime_seed": isinstance(
            receipt.get("nuisance_runtime"), dict
        )
        and receipt["nuisance_runtime"].get("seed") == seed,
        "nuisance_camera_unchanged": all(
            nuisance_runtime.get(field) is not False
            for field in ("camera_pose_unchanged", "camera_transform_unchanged")
            if field in nuisance_runtime
        ),
        "nuisance_non_light_scene_unchanged": all(
            nuisance_runtime.get(field) is not False
            for field in (
                "non_light_scene_unchanged",
                "non_light_geometry_unchanged",
            )
            if field in nuisance_runtime
        ),
        "nuisance_runtime_authored_readback_equal": isinstance(
            receipt.get("nuisance_runtime"), dict
        )
        and light_rows_match(
            receipt["nuisance_runtime"].get("authored_light_properties"),
            receipt["nuisance_runtime"].get("readback_light_properties"),
        )
        and transform_rows_match(
            receipt["nuisance_runtime"].get("authored_transforms"),
            receipt["nuisance_runtime"].get("readback_transforms"),
        ),
        "stage_sha256": provenance.get("requested_stage_sha256")
        == binding["stage_sha256"],
        "renderer_script_sha256": provenance.get("script_sha256")
        == binding["renderer_sha256"],
        "renderer": render.get("renderer") == "PathTracing",
        "video_renderer": (
            render.get("video_renderer") == "RaytracedLighting"
            if realtime_video
            else True
        ),
        "stage_still_renderer": (
            render.get("stage_still_renderer") == "PathTracing"
            if realtime_video
            else True
        ),
        "stage_frames_not_video_subsets": (
            render.get("stage_frames_are_video_subsets") is False
            if realtime_video
            else True
        ),
        "resolution": render.get("resolution") == [640, 480],
        "rt_subframes": render.get("rt_subframes") == 64,
        "total_spp_requested": render.get("pathtracing_total_spp_requested") == 64,
        "total_spp_readback": render.get("pathtracing_total_spp_readback") == 64,
        "spp_requested": render.get("pathtracing_spp_requested") == 64,
        "spp_readback": render.get("pathtracing_spp_readback") == 64,
        "warmup": render.get("warmup_updates") == 60,
        "source_capture_count": render.get("frame_count")
        == expected_output["source_capture_count"],
        "decoded_video_frame_count": render.get("video_decoded_frame_count")
        == expected_output["decoded_video_frame_count"],
        "report_hz": timing.get("report_hz") == 120,
        "physics_steps_per_second": timing.get("physics_steps_per_second")
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["physics_steps_per_second"],
        "physics_substeps_per_report_interval": timing.get(
            "physics_substeps_per_report_interval"
        )
        == QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT[
            "physics_substeps_per_report_interval"
        ],
        "report_interval_count": timing.get("report_interval_count")
        == expected_output["report_interval_count"],
        "physics_callback_count": timing.get("physics_callback_count")
        == expected_output["physics_callback_count"],
        "diffusion_update_count": timing.get("diffusion_update_count")
        == expected_output["report_interval_count"],
        "diffusion_updates_per_report_interval": timing.get(
            "diffusion_updates_per_report_interval"
        )
        == 1,
        "authored_rows_session_only": all(
            timing.get(name) is True
            for name in (
                "positions_session_layer_only",
                "velocities_session_layer_only",
                "orientations_session_layer_only",
                "scales_session_layer_only",
                "proto_indices_session_layer_only",
                "ids_session_layer_only",
            )
        ),
        "simulation_points_root_absent": timing.get(
            "simulation_points_root_layer_absent"
        )
        is True,
        "simulation_points_runtime_valid": timing.get(
            "simulation_points_runtime_readback_passed"
        )
        is True,
        "legacy_train_provenance_without_ablation": not ablation,
        "schedule_case_id": schedule.get("case_id") == case_id,
        "schedule_plan_sha256": schedule.get("cohort_plan_sha256")
        == args.plan_sha256,
        "schedule_plan_file_sha256": schedule.get("schedule_file_sha256")
        == binding["plan_file_sha256"],
        "schedule_qualification_sha256": schedule.get(
            "qualification_receipt_sha256"
        )
        == args.accepted_qualification_receipt_sha256,
    }
    return [name for name, passed in checks.items() if not passed]


def _default_validate_artifacts(output: Path) -> dict[str, Any]:
    from validate_physx_f2_artifacts import validate_output

    return validate_output(output)


def _case_is_valid(
    case_root: Path,
    *,
    args: argparse.Namespace,
    binding: dict[str, Any],
    case_id: str,
    seed: int,
    nuisance_case_id: str,
    validate_artifacts: Callable[[Path], dict[str, Any]],
    allow_host_path_relocation: bool = False,
) -> tuple[bool, dict[str, Any]]:
    exit_code_path = case_root / "exit-code.txt"
    launch_path = case_root / "launch-receipt.json"
    receipt_path = case_root / "app-output" / "receipt.json"
    if not exit_code_path.is_file() or not launch_path.is_file() or not receipt_path.is_file():
        return False, {"failures": ["case completion files are missing"]}
    try:
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
        launch = _read_json_object(launch_path, "case launch receipt")
    except (OSError, UnicodeError, ValueError) as error:
        return False, {"failures": [f"case completion metadata invalid: {error}"]}
    launch_failures = []
    if launch.get("case_id") != case_id:
        launch_failures.append("launch_case_id")
    if launch.get("seed") != seed:
        launch_failures.append("launch_seed")
    if launch.get("plan_sha256") != args.plan_sha256:
        launch_failures.append("launch_plan_sha256")
    if launch.get("plan_file_sha256") != binding["plan_file_sha256"]:
        launch_failures.append("launch_plan_file_sha256")
    if launch.get("qualification_receipt_sha256") != args.accepted_qualification_receipt_sha256:
        launch_failures.append("launch_qualification_sha256")
    if not (
        launch.get("nuisance_receipt_path") == os.fspath(args.nuisance_receipt)
        or (
            allow_host_path_relocation
            and _absolute_path_string(launch.get("nuisance_receipt_path"))
        )
    ):
        launch_failures.append("launch_nuisance_receipt_path")
    if launch.get("nuisance_receipt_sha256") != args.nuisance_receipt_sha256:
        launch_failures.append("launch_nuisance_receipt_sha256")
    if launch.get("nuisance_case_id") != nuisance_case_id:
        launch_failures.append("launch_nuisance_case_id")
    if not (
        launch.get("nuisance_plan") == binding["nuisance_plan"]
        or (
            allow_host_path_relocation
            and _nuisance_binding_matches_with_host_relocation(
                launch.get("nuisance_plan"), binding["nuisance_plan"]
            )
        )
    ):
        launch_failures.append("launch_nuisance_plan")
    if launch.get("renderer_script_sha256") != binding["renderer_sha256"]:
        launch_failures.append("launch_renderer_sha256")
    if launch.get("renderer_qualification_compatibility") != binding[
        "renderer_qualification_compatibility"
    ]:
        launch_failures.append("launch_renderer_qualification_compatibility")
    if launch.get("stage_sha256") != binding["stage_sha256"]:
        launch_failures.append("launch_stage_sha256")
    if launch.get("render_contract") != binding["render_contract"]:
        launch_failures.append("launch_render_contract")
    expected_command = _case_command(
        args,
        case_id=case_id,
        seed=seed,
        nuisance_case_id=nuisance_case_id,
        app_output=case_root / "app-output",
    )
    if not (
        launch.get("argv") == expected_command
        or (
            allow_host_path_relocation
            and _commands_match_with_host_relocation(
                launch.get("argv"), expected_command, case_id=case_id
            )
        )
    ):
        launch_failures.append("launch_argv")
    receipt_failures = _case_receipt_failures(
        receipt_path,
        args=args,
        binding=binding,
        case_id=case_id,
        seed=seed,
        nuisance_case_id=nuisance_case_id,
        allow_host_path_relocation=allow_host_path_relocation,
    )
    observed_counts: dict[str, int | None] = {
        "source_capture_count": None,
        "decoded_video_frame_count": None,
    }
    try:
        observed_receipt = _read_json_object(
            receipt_path, "case render receipt denominator"
        )
        observed_render = observed_receipt.get("render")
        if isinstance(observed_render, dict):
            for field in observed_counts:
                receipt_field = (
                    "frame_count"
                    if field == "source_capture_count"
                    else "video_decoded_frame_count"
                )
                value = observed_render.get(receipt_field)
                if isinstance(value, int) and not isinstance(value, bool):
                    observed_counts[field] = value
    except ValueError:
        pass
    try:
        artifact_result = validate_artifacts(case_root / "app-output")
    except Exception as error:  # noqa: BLE001 - evidence failure must be recorded
        artifact_result = {"status": "insufficient_evidence", "failures": [str(error)]}
    artifact_failures = (
        list(artifact_result.get("failures", []))
        if isinstance(artifact_result, dict)
        else ["artifact validator did not return an object"]
    )
    valid = (
        exit_code == 0
        and not launch_failures
        and not receipt_failures
        and isinstance(artifact_result, dict)
        and artifact_result.get("status") == "pass"
        and not artifact_failures
    )
    return valid, {
        "exit_code": exit_code,
        "launch_failures": launch_failures,
        "receipt_failures": receipt_failures,
        "artifact_validation": artifact_result,
        "observed_counts": observed_counts,
        "failures": launch_failures + receipt_failures + artifact_failures,
    }


def _next_attempt_receipt(output_root: Path) -> Path:
    directory = output_root / "supervisor-attempts"
    directory.mkdir(mode=0o750, exist_ok=True)
    for ordinal in range(1, 10000):
        path = directory / f"attempt-{ordinal:04d}.json"
        if not path.exists():
            return path
    raise RuntimeError("supervisor attempt receipt namespace exhausted")


def merge_completed_shard_roots(
    args: argparse.Namespace,
    *,
    shard_roots: list[Path],
    expected_shard_count: int,
    run_process: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    validate_artifacts: Callable[[Path], dict[str, Any]] | None = None,
    active_processes: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Merge disjoint completed shard roots and run the full resume validator."""

    binding = _bind_inputs(args)
    validator = validate_artifacts or _default_validate_artifacts
    if args.shard_count != 1 or args.shard_index != 0 or args.resume_completed:
        raise ValueError("merge args must use the full-cohort non-resume defaults")
    selected_schedule_count = binding["output_contract"]["case_count"]
    if (
        isinstance(expected_shard_count, bool)
        or not isinstance(expected_shard_count, int)
        or expected_shard_count < 2
        or expected_shard_count > selected_schedule_count
    ):
        raise ValueError(
            "expected_shard_count must be between 2 and "
            f"{selected_schedule_count}"
        )
    if (
        binding["plan"].get("artifact_type") == VARIED_V4_PLAN_ARTIFACT_TYPE
        and expected_shard_count != selected_schedule_count
    ):
        raise ValueError(
            "V4 shard merge requires exactly one root per source schedule"
        )
    if len(shard_roots) != expected_shard_count:
        raise ValueError("shard root count differs from expected_shard_count")
    merge_receipt_path = args.output_root / "shard-merge.json"
    if merge_receipt_path.exists():
        raise ValueError("shard merge receipt already exists")

    expected_manifest = _cohort_manifest(args, binding)

    def tree_snapshot(root: Path) -> dict[str, dict[str, int | str]]:
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("shard case root must be an existing absolute directory")
        snapshot: dict[str, dict[str, int | str]] = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("shard case trees must not contain symbolic links")
            if path.is_file():
                snapshot[path.relative_to(root).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        return snapshot

    def bound_manifest(root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        path = root / "cohort-launch.json"
        payload = _read_json_object(path, "shard cohort launch manifest")
        comparable = dict(payload)
        comparable.pop("created_at", None)
        expected_comparable = dict(expected_manifest)
        expected_comparable.pop("created_at", None)
        if comparable == expected_comparable:
            relocation = {
                "mode": "exact",
                "source_paths": {
                    field: payload[field] for field in _HOST_PATH_MANIFEST_FIELDS
                },
                "merge_paths": {
                    field: expected_manifest[field]
                    for field in _HOST_PATH_MANIFEST_FIELDS
                },
                "source_supervisor_script_sha256": payload[
                    "supervisor_script_sha256"
                ],
                "merge_supervisor_script_sha256": expected_manifest[
                    "supervisor_script_sha256"
                ],
            }
        else:
            relocation = _host_relocated_manifest(payload, expected_manifest)
            if relocation is None:
                raise ValueError(
                    "shard cohort launch manifest differs from the full binding"
                )
        return payload, path, relocation

    def latest_attempt(root: Path) -> tuple[dict[str, Any], Path]:
        attempts = sorted((root / "supervisor-attempts").glob("attempt-*.json"))
        if not attempts:
            raise ValueError("shard supervisor attempt is missing")
        path = attempts[-1]
        return _read_json_object(path, "shard supervisor attempt"), path

    full_case_ids = [
        row["case_binding"]["case_id"]
        for row in binding["plan"]["execution_schedules"]
    ]
    source_by_case: dict[str, tuple[Path, int]] = {}
    validated_shards: dict[int, dict[str, Any]] = {}
    cohort_source_supervisor_sha256: str | None = None
    for raw_root in shard_roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError("shard root must be an existing directory")
        source_manifest, launch_path, host_path_relocation = bound_manifest(root)
        source_supervisor_sha256 = source_manifest["supervisor_script_sha256"]
        if cohort_source_supervisor_sha256 is None:
            cohort_source_supervisor_sha256 = source_supervisor_sha256
        elif source_supervisor_sha256 != cohort_source_supervisor_sha256:
            raise ValueError("shard source supervisor SHA-256 values differ")
        attempt, attempt_path = latest_attempt(root)
        shard = attempt.get("shard")
        if not isinstance(shard, dict):
            raise ValueError("shard supervisor attempt lacks shard provenance")
        shard_index = shard.get("index")
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or shard_index < 0
            or shard_index >= expected_shard_count
        ):
            raise ValueError("shard supervisor attempt has invalid shard_index")
        if shard_index in validated_shards:
            raise ValueError("duplicate shard_index")
        selected_rows = _selected_schedule_rows(
            binding["plan"],
            shard_count=expected_shard_count,
            shard_index=shard_index,
        )
        if (
            binding["plan"].get("artifact_type") == VARIED_V4_PLAN_ARTIFACT_TYPE
            and len(selected_rows) != 1
        ):
            raise ValueError("V4 shard merge requires one-case shard roots")
        selected_case_ids = [
            schedule["case_binding"]["case_id"]
            for _, schedule in selected_rows
        ]
        selected_output = _selected_output_contract(
            binding["output_contract"], selected_rows
        )
        case_results = attempt.get("case_results")
        result_case_ids = (
            [row.get("case_id") for row in case_results]
            if isinstance(case_results, list)
            and all(isinstance(row, dict) for row in case_results)
            else None
        )
        result_outcomes = (
            [row.get("outcome") for row in case_results]
            if result_case_ids is not None
            else []
        )
        attempt_result_by_case = (
            {row["case_id"]: row for row in case_results}
            if result_case_ids is not None
            and all(isinstance(row.get("case_id"), str) for row in case_results)
            else {}
        )
        expected_shard = {
            "index": shard_index,
            "count": expected_shard_count,
            "selection_rule": "global_ordinal_modulo_shard_count",
            "selected_case_ids": selected_case_ids,
        }
        common_attempt_binding = (
            attempt.get("schema_version") == 1
            and attempt.get("artifact_type")
            == "physx_train_residual_serial_supervisor_attempt"
            and attempt.get("plan_sha256") == args.plan_sha256
            and attempt.get("qualification_receipt_sha256")
            == args.accepted_qualification_receipt_sha256
            and shard == expected_shard
            and attempt.get("expected_output_contract") == selected_output
            and attempt.get("planned_case_count") == len(selected_case_ids)
            and result_case_ids == selected_case_ids
        )
        completed_pass_attempt = (
            attempt.get("status") == "pass"
            and attempt.get("validated_completed_case_count")
            == len(selected_case_ids)
            and attempt.get("observed_source_capture_count")
            == selected_output["source_capture_count"]
            and attempt.get("observed_decoded_video_frame_count")
            == selected_output["decoded_video_frame_count"]
            and attempt.get("failure") is None
            and all(
                outcome in {"passed", "recognized_completed"}
                for outcome in result_outcomes
            )
        )
        failure = attempt.get("failure")
        offline_height_candidate_attempt = (
            len(selected_case_ids) == 1
            and attempt.get("status") == "failed"
            and attempt.get("validated_completed_case_count") == 0
            and attempt.get("observed_source_capture_count") == 0
            and attempt.get("observed_decoded_video_frame_count") == 0
            and isinstance(failure, dict)
            and failure.get("reason") == "completed_case_failed_validation"
            and failure.get("case_id") == selected_case_ids[0]
            and result_outcomes == ["failed"]
        )
        if not (
            common_attempt_binding
            and (completed_pass_attempt or offline_height_candidate_attempt)
        ):
            raise ValueError("shard supervisor attempt is not a complete subset pass")
        actual_case_ids = sorted(
            path.name for path in (root / "cases").glob("CASE_*") if path.is_dir()
        )
        if actual_case_ids != sorted(selected_case_ids):
            raise ValueError("shard case directory set differs from its selected cases")
        offline_height_warning_revalidated: dict[str, dict[str, Any]] = {}
        for case_id in selected_case_ids:
            if case_id in source_by_case:
                raise ValueError("case appears in more than one shard")
            case_root = root / "cases" / case_id
            try:
                exit_code = (case_root / "exit-code.txt").read_text(
                    encoding="utf-8"
                ).strip()
            except (OSError, UnicodeError) as error:
                raise ValueError("shard case exit code is unreadable") from error
            result = _read_json_object(
                case_root / "case-result.json", "shard case result"
            )
            immutable_pass = (
                exit_code == "0"
                and result.get("case_id") == case_id
                and result.get("outcome") == "passed"
                and result.get("return_code") == 0
                and result.get("timed_out") is False
            )
            if not immutable_pass:
                old_evidence = result.get("evidence")
                old_artifacts = (
                    old_evidence.get("artifact_validation")
                    if isinstance(old_evidence, dict)
                    else None
                )
                recognized = attempt_result_by_case.get(case_id)
                recognized_evidence = (
                    recognized.get("evidence")
                    if isinstance(recognized, dict)
                    else None
                )
                recognized_artifacts = (
                    recognized_evidence.get("artifact_validation")
                    if isinstance(recognized_evidence, dict)
                    else None
                )
                fresh_artifacts = validator(case_root / "app-output")
                warnings = (
                    fresh_artifacts.get("warnings")
                    if isinstance(fresh_artifacts, dict)
                    else None
                )
                exact_old_height_only_failure = (
                    exit_code == "0"
                    and result.get("case_id") == case_id
                    and result.get("outcome") == "failed"
                    and result.get("return_code") == 0
                    and result.get("timed_out") is False
                    and isinstance(old_evidence, dict)
                    and old_evidence.get("exit_code") == 0
                    and old_evidence.get("launch_failures") == []
                    and old_evidence.get("receipt_failures") == []
                    and old_evidence.get("failures")
                    == ["receipt:realtime_volume_height_gate"]
                    and isinstance(old_artifacts, dict)
                    and old_artifacts.get("status") == "insufficient_evidence"
                    and old_artifacts.get("failures")
                    == ["receipt:realtime_volume_height_gate"]
                )
                latest_attempt_supports_revalidation = (
                    isinstance(recognized, dict)
                    and (
                        (
                            recognized.get("outcome") == "recognized_completed"
                            and isinstance(recognized_evidence, dict)
                            and recognized_evidence.get("exit_code") == 0
                            and recognized_evidence.get("launch_failures") == []
                            and recognized_evidence.get("receipt_failures") == []
                            and recognized_evidence.get("failures") == []
                            and recognized_artifacts == fresh_artifacts
                        )
                        or (
                            recognized.get("outcome") == "failed"
                            and recognized_evidence == old_evidence
                            and isinstance(failure, dict)
                            and failure.get("evidence") == old_evidence
                        )
                    )
                )
                finite_height_warnings = (
                    isinstance(warnings, list)
                    and bool(warnings)
                    and all(
                        isinstance(warning, dict)
                        and warning.get("warning_type")
                        == "particle_height_outside_geometry_volume_bounds"
                        and warning.get("hard_gate") is False
                        and warning.get("claim")
                        == "diagnostic_warning_not_a_physics_pass"
                        for warning in warnings
                    )
                    and fresh_artifacts.get("status") == "pass"
                    and fresh_artifacts.get("failures") == []
                )
                if (
                    exact_old_height_only_failure
                    and latest_attempt_supports_revalidation
                    and finite_height_warnings
                ):
                    result_path = case_root / "case-result.json"
                    offline_height_warning_revalidated[case_id] = {
                        "claim": "render_pass_not_a_physics_pass",
                        "latest_attempt_outcome": recognized.get("outcome"),
                        "source_case_result_path": os.fspath(result_path),
                        "source_case_result_sha256": _sha256(result_path),
                        "source_exit_code_path": os.fspath(
                            case_root / "exit-code.txt"
                        ),
                        "source_exit_code_sha256": _sha256(
                            case_root / "exit-code.txt"
                        ),
                        "warnings": deepcopy(warnings),
                    }
                else:
                    raise ValueError("shard case result is not a completed pass")
            if not immutable_pass and case_id not in offline_height_warning_revalidated:
                raise ValueError("shard case result is not a completed pass")
            source_by_case[case_id] = (case_root, shard_index)
        validated_shards[shard_index] = {
            "root": os.fspath(root),
            "case_ids": selected_case_ids,
            "execution_host": attempt.get("execution_host"),
            "launch_manifest_path": os.fspath(launch_path),
            "launch_manifest_sha256": _sha256(launch_path),
            "attempt_path": os.fspath(attempt_path),
            "attempt_sha256": _sha256(attempt_path),
            "source_supervisor_script_sha256": source_supervisor_sha256,
            "host_path_relocation": host_path_relocation,
            "offline_height_warning_revalidated_case_ids": sorted(
                offline_height_warning_revalidated
            ),
            "height_warnings_by_case": {
                case_id: record["warnings"]
                for case_id, record in offline_height_warning_revalidated.items()
            },
            "offline_revalidation_by_case": offline_height_warning_revalidated,
        }

    if set(validated_shards) != set(range(expected_shard_count)):
        raise ValueError("shard indexes are incomplete")
    if set(source_by_case) != set(full_case_ids):
        raise ValueError("shard case union is not the complete frozen cohort")
    if args.output_root.exists():
        bound_manifest(args.output_root)
    else:
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{args.output_root.name}.merge-scaffold-",
                dir=args.output_root.parent,
            )
        )
        try:
            (staging_root / "cases").mkdir(mode=0o750)
            _write_json_new(staging_root / "cohort-launch.json", expected_manifest)
            staging_root.rename(args.output_root)
        except Exception:
            if staging_root.exists():
                shutil.rmtree(staging_root)
            raise
    cases_root = args.output_root / "cases"
    if not cases_root.is_dir():
        raise ValueError("aggregate cases directory is missing")
    shard_archive_root = args.output_root / "shard-attempts"
    shard_archive_root.mkdir(mode=0o750, exist_ok=True)

    for case_id in full_case_ids:
        source, _ = source_by_case[case_id]
        target = cases_root / case_id
        source_snapshot = tree_snapshot(source)
        if target.exists():
            if tree_snapshot(target) != source_snapshot:
                raise ValueError(f"aggregate case conflicts with shard source: {case_id}")
        else:
            shutil.copytree(source, target, copy_function=shutil.copy2)
            if tree_snapshot(target) != source_snapshot:
                raise ValueError(f"aggregate case copy verification failed: {case_id}")

    for shard_index, record in sorted(validated_shards.items()):
        source_attempts = Path(record["root"]) / "supervisor-attempts"
        archive = shard_archive_root / f"shard-{shard_index:04d}"
        source_snapshot = tree_snapshot(source_attempts)
        if archive.exists():
            if tree_snapshot(archive) != source_snapshot:
                raise ValueError("archived shard attempts conflict with source")
        else:
            shutil.copytree(source_attempts, archive, copy_function=shutil.copy2)
            if tree_snapshot(archive) != source_snapshot:
                raise ValueError("archived shard attempt copy verification failed")

    resume_args = deepcopy(args)
    resume_args.resume_completed = True
    resume_args.shard_count = 1
    resume_args.shard_index = 0
    full_summary = run_supervisor(
        resume_args,
        run_process=run_process,
        validate_artifacts=validator,
        active_processes=active_processes,
        allow_host_path_relocation=True,
    )
    if full_summary.get("status") != "pass":
        raise ValueError("full resume validation failed after shard merge")
    latest_full_attempt = sorted(
        (args.output_root / "supervisor-attempts").glob("attempt-*.json")
    )[-1]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "physx_train_residual_shard_merge",
        "status": "pass",
        "created_at": _utc_now(),
        "plan_sha256": args.plan_sha256,
        "qualification_receipt_sha256": (
            args.accepted_qualification_receipt_sha256
        ),
        "source_supervisor_script_sha256": cohort_source_supervisor_sha256,
        "shard_count": expected_shard_count,
        "merged_case_count": len(full_case_ids),
        "case_ids": full_case_ids,
        "source_shards": [
            {"index": index, **record}
            for index, record in sorted(validated_shards.items())
        ],
        "full_resume_attempt": {
            "path": os.fspath(latest_full_attempt),
            "sha256": _sha256(latest_full_attempt),
        },
        "full_resume_summary": full_summary,
    }
    _write_json_new(merge_receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--accepted-qualification-receipt", required=True, type=Path)
    parser.add_argument("--accepted-qualification-receipt-sha256", required=True)
    parser.add_argument("--nuisance-receipt", required=True, type=Path)
    parser.add_argument("--nuisance-receipt-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--renderer-script", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument(
        "--host-lock",
        type=Path,
        default=Path("/tmp/fluidmix-physx-train-residual.lock"),
    )
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def run_supervisor(
    args: argparse.Namespace,
    *,
    run_process: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    validate_artifacts: Callable[[Path], dict[str, Any]] | None = None,
    active_processes: Callable[[], list[dict[str, Any]]] | None = None,
    allow_host_path_relocation: bool = False,
) -> dict[str, Any]:
    binding = _bind_inputs(args)
    selected_rows = _selected_schedule_rows(
        binding["plan"],
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    selected_output = _selected_output_contract(
        binding["output_contract"], selected_rows
    )
    validator = validate_artifacts or _default_validate_artifacts
    process_probe = active_processes or _find_other_isaac_processes
    with _exclusive_host_lock(args.host_lock):
        manifest = _cohort_manifest(args, binding)
        if args.output_root.exists():
            if not args.resume_completed:
                raise ValueError(
                    "output_root already exists; --resume-completed is required"
                )
            existing_manifest = _read_json_object(
                args.output_root / "cohort-launch.json", "cohort launch manifest"
            )
            expected_binding = dict(manifest)
            expected_binding.pop("created_at")
            observed_binding = dict(existing_manifest)
            observed_binding.pop("created_at", None)
            if not (
                observed_binding == expected_binding
                or (
                    allow_host_path_relocation
                    and _host_relocated_manifest(existing_manifest, manifest)
                    is not None
                )
            ):
                raise ValueError("existing cohort launch manifest differs from this run")
            cases_root = args.output_root / "cases"
            if not cases_root.is_dir():
                raise ValueError("existing cohort cases directory is missing")
        else:
            if args.resume_completed:
                raise ValueError("--resume-completed requires an existing output_root")
            args.output_root.mkdir(mode=0o750)
            cases_root = args.output_root / "cases"
            cases_root.mkdir(mode=0o750)
            _write_json_new(args.output_root / "cohort-launch.json", manifest)

        summary: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "physx_train_residual_serial_supervisor_attempt",
            "status": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "plan_sha256": args.plan_sha256,
            "qualification_receipt_sha256": args.accepted_qualification_receipt_sha256,
            "expected_output_contract": deepcopy(selected_output),
            "planned_case_count": len(selected_rows),
            "shard": {
                "index": args.shard_index,
                "count": args.shard_count,
                "selection_rule": "global_ordinal_modulo_shard_count",
                "selected_case_ids": [
                    schedule["case_binding"]["case_id"]
                    for _, schedule in selected_rows
                ],
            },
            "execution_host": os.uname().nodename,
            "attempts_started": 0,
            "processes_completed": 0,
            "cases_passed_this_invocation": 0,
            "cases_failed_this_invocation": 0,
            "previous_completed_cases_recognized": 0,
            "validated_completed_case_count": 0,
            "observed_source_capture_count": 0,
            "observed_decoded_video_frame_count": 0,
            "case_results": [],
            "failure": None,
        }
        attempt_receipt = _next_attempt_receipt(args.output_root)
        seeds = binding["case_seeds"]
        nuisance_case_ids = binding["nuisance_case_ids"]

        def finish(status: str, failure: dict[str, Any] | None = None) -> dict[str, Any]:
            summary["status"] = status
            summary["failure"] = failure
            summary["completed_at"] = _utc_now()
            _write_json_new(attempt_receipt, summary)
            return summary

        for ordinal, schedule in selected_rows:
            case_id = schedule["case_binding"]["case_id"]
            seed = seeds[case_id]
            case_root = cases_root / case_id
            if case_root.exists():
                valid, evidence = _case_is_valid(
                    case_root,
                    args=args,
                    binding=binding,
                    case_id=case_id,
                    seed=seed,
                    nuisance_case_id=nuisance_case_ids[case_id],
                    validate_artifacts=validator,
                    allow_host_path_relocation=allow_host_path_relocation,
                )
                if not args.resume_completed or not valid:
                    return finish(
                        "failed",
                        {
                            "reason": "existing_case_is_not_a_valid_explicit_resume",
                            "case_id": case_id,
                            "evidence": evidence,
                        },
                    )
                summary["previous_completed_cases_recognized"] += 1
                summary["validated_completed_case_count"] += 1
                summary["observed_source_capture_count"] += int(
                    evidence["observed_counts"]["source_capture_count"]
                )
                summary["observed_decoded_video_frame_count"] += int(
                    evidence["observed_counts"]["decoded_video_frame_count"]
                )
                summary["case_results"].append(
                    {
                        "case_id": case_id,
                        "seed": seed,
                        "ordinal": ordinal,
                        "outcome": "recognized_completed",
                        "evidence": evidence,
                    }
                )
                continue

            conflicts = process_probe()
            if conflicts:
                return finish(
                    "failed",
                    {
                        "reason": "other_isaac_or_kit_process_is_running",
                        "case_id": case_id,
                        "processes": conflicts,
                    },
                )

            case_root.mkdir(mode=0o750)
            app_output = case_root / "app-output"
            command = _case_command(
                args,
                case_id=case_id,
                seed=seed,
                nuisance_case_id=nuisance_case_ids[case_id],
                app_output=app_output,
            )
            launch_receipt = {
                "schema_version": 1,
                "artifact_type": "physx_train_residual_case_launch",
                "case_id": case_id,
                "seed": seed,
                "ordinal": ordinal,
                "started_at": _utc_now(),
                "plan_path": os.fspath(args.plan),
                "plan_sha256": args.plan_sha256,
                "plan_file_sha256": binding["plan_file_sha256"],
                "qualification_receipt_path": os.fspath(
                    args.accepted_qualification_receipt
                ),
                "qualification_receipt_sha256": (
                    args.accepted_qualification_receipt_sha256
                ),
                "nuisance_plan": deepcopy(binding["nuisance_plan"]),
                "nuisance_receipt_path": os.fspath(args.nuisance_receipt),
                "nuisance_receipt_sha256": args.nuisance_receipt_sha256,
                "nuisance_case_id": nuisance_case_ids[case_id],
                "stage_path": os.fspath(args.stage),
                "stage_sha256": binding["stage_sha256"],
                "renderer_script": os.fspath(args.renderer_script),
                "renderer_script_sha256": binding["renderer_sha256"],
                "renderer_qualification_compatibility": deepcopy(
                    binding["renderer_qualification_compatibility"]
                ),
                "python_executable": os.fspath(args.python_executable),
                "render_contract": dict(binding["render_contract"]),
                "execution_host": os.uname().nodename,
                "argv": command,
            }
            if "tank_colour_resolution" in binding:
                launch_receipt["tank_colour_resolution"] = deepcopy(
                    binding["tank_colour_resolution"]
                )
            _write_json_new(case_root / "launch-receipt.json", launch_receipt)
            summary["attempts_started"] += 1
            timed_out = False
            launch_error: str | None = None
            return_code: int | None = None
            environment = os.environ.copy()
            environment.update(
                OMNI_KIT_ACCEPT_EULA="YES",
                PYTHONUNBUFFERED="1",
            )
            try:
                with (case_root / "isaac.log").open("xb") as log:
                    completed = run_process(
                        command,
                        cwd=args.renderer_script.parent.parent,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=OUTER_TIMEOUT_SECONDS,
                    )
                return_code = int(completed.returncode)
                summary["processes_completed"] += 1
            except subprocess.TimeoutExpired as error:
                timed_out = True
                launch_error = str(error)
                return_code = 124
                summary["processes_completed"] += 1
            except OSError as error:
                launch_error = str(error)

            if return_code is not None:
                _write_text_new(case_root / "exit-code.txt", f"{return_code}\n")
            _write_text_new(case_root / "completed-at.txt", f"{_utc_now()}\n")
            if return_code != 0:
                case_result = {
                    "case_id": case_id,
                    "seed": seed,
                    "ordinal": ordinal,
                    "outcome": "failed",
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "launch_error": launch_error,
                    "artifact_validation": None,
                }
                _write_json_new(case_root / "case-result.json", case_result)
                summary["cases_failed_this_invocation"] += 1
                summary["case_results"].append(case_result)
                return finish(
                    "failed",
                    {
                        "reason": "renderer_process_failed",
                        "case_id": case_id,
                        "return_code": return_code,
                        "timed_out": timed_out,
                        "launch_error": launch_error,
                    },
                )

            valid, evidence = _case_is_valid(
                case_root,
                args=args,
                binding=binding,
                case_id=case_id,
                seed=seed,
                nuisance_case_id=nuisance_case_ids[case_id],
                validate_artifacts=validator,
            )
            case_result = {
                "case_id": case_id,
                "seed": seed,
                "ordinal": ordinal,
                "outcome": "passed" if valid else "failed",
                "return_code": return_code,
                "timed_out": timed_out,
                "launch_error": launch_error,
                "evidence": evidence,
            }
            _write_json_new(case_root / "case-result.json", case_result)
            summary["case_results"].append(case_result)
            if not valid:
                summary["cases_failed_this_invocation"] += 1
                return finish(
                    "failed",
                    {
                        "reason": "completed_case_failed_validation",
                        "case_id": case_id,
                        "evidence": evidence,
                    },
                )
            summary["cases_passed_this_invocation"] += 1
            summary["validated_completed_case_count"] += 1
            summary["observed_source_capture_count"] += int(
                evidence["observed_counts"]["source_capture_count"]
            )
            summary["observed_decoded_video_frame_count"] += int(
                evidence["observed_counts"]["decoded_video_frame_count"]
            )

        if summary["validated_completed_case_count"] != summary["planned_case_count"]:
            return finish(
                "failed",
                {"reason": "validated completed denominator is incomplete"},
            )
        expected_output = selected_output
        if (
            summary["observed_source_capture_count"]
            != expected_output["source_capture_count"]
            or summary["observed_decoded_video_frame_count"]
            != expected_output["decoded_video_frame_count"]
        ):
            return finish(
                "failed",
                {"reason": "validated output denominator is incomplete"},
            )
        return finish("pass")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_supervisor(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"supervisor input/infrastructure failure: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

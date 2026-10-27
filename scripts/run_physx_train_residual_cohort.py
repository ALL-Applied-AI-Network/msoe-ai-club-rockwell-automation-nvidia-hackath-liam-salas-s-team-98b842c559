#!/usr/bin/env python3
"""Run the source-bound 18-case PhysX train cohort serially."""

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
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))
sys.path.insert(0, os.fspath(ROOT / "scripts"))

import physx_three_pour_sequence as sequence  # noqa: E402


EXPECTED_PLAN_SHA256 = "231e2388f256f2118db36ed838bebbec04a73506d97b5db33b4fb87e7a6b6b6a"
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
    "physics_steps_per_second": 240,
    "physics_substeps_per_report_interval": 2,
    "particle_resolution_profile": "c21_2p0mm",
    "effective_particle_volume_ml": 0.005771707871971808,
    "nuisance": None,
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
    if declared_sha256 != EXPECTED_PLAN_SHA256:
        raise ValueError("train render plan SHA-256 is not the frozen source-bound plan")
    observed = _read_json_object(path, "train render plan")
    if _canonical_sha256(observed) != declared_sha256:
        raise ValueError("train render plan canonical SHA-256 mismatch")
    expected = sequence.build_train_residual_artifacts(repo_root=ROOT)[
        "render-schedule-plan.json"
    ]
    if observed != expected:
        raise ValueError("train render plan differs from the source-bound plan")
    schedules = observed.get("execution_schedules")
    if not isinstance(schedules, list) or len(schedules) != 18:
        raise ValueError("train render plan must contain exactly 18 schedules")
    return observed


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


def _case_output_contract(plan: dict[str, Any]) -> dict[str, Any]:
    cases: dict[str, dict[str, int]] = {}
    keep_count = 0
    discard_count = 0
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
        frame_steps = [
            int(row["physics_step"]) for row in capture_schedule["captures"]
        ]
        video = sequence.build_time_preserving_video_repeat_plan(
            frame_steps,
            physics_hz=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["report_hz"],
            playback_fps=QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT["video_fps"],
        )
        report_interval_count = int(capture_schedule["final_plus_3_0_step"])
        cases[case_id] = {
            "report_interval_count": report_interval_count,
            "physics_callback_count": 2 * report_interval_count,
            "source_capture_count": len(frame_steps),
            "decoded_video_frame_count": int(video["video_frame_count"]),
        }
        if float(residual_volume_ml) == 0.0:
            keep_count += 1
        else:
            discard_count += 1
    source_capture_count = sum(
        row["source_capture_count"] for row in cases.values()
    )
    decoded_video_frame_count = sum(
        row["decoded_video_frame_count"] for row in cases.values()
    )
    if (
        len(cases) != 18
        or keep_count != 12
        or discard_count != 6
        or source_capture_count != 444
        or decoded_video_frame_count != 8532
    ):
        raise ValueError("frozen train cohort output denominator drift")
    return {
        "cases": cases,
        "case_count": len(cases),
        "keep_count": keep_count,
        "discard_count": discard_count,
        "source_capture_count": source_capture_count,
        "decoded_video_frame_count": decoded_video_frame_count,
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
    renderer_sha256 = _sha256(args.renderer_script)
    if renderer_sha256 != qualification["runner_script_sha256"]:
        raise ValueError("renderer script hash differs from the accepted qualification")
    return {
        "plan": plan,
        "case_seeds": _case_seeds(plan),
        "output_contract": _case_output_contract(plan),
        "plan_file_sha256": _sha256(args.plan),
        "qualification": qualification,
        "stage_sha256": _sha256(args.stage),
        "renderer_sha256": renderer_sha256,
        "supervisor_sha256": _sha256(Path(__file__).resolve()),
        "render_contract": dict(QUALIFIED_COUNT_DRIVEN_RENDER_CONTRACT),
    }


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
    return {
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
        "python_executable": os.fspath(args.python_executable),
        "supervisor_script_sha256": binding["supervisor_sha256"],
        "render_contract": dict(binding["render_contract"]),
        "expected_output_contract": deepcopy(binding["output_contract"]),
        "case_ids": [
            row["case_binding"]["case_id"]
            for row in binding["plan"]["execution_schedules"]
        ],
        "created_at": _utc_now(),
    }


def _case_command(
    args: argparse.Namespace,
    *,
    case_id: str,
    seed: int,
    app_output: Path,
) -> list[str]:
    return [
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
    ]


def _case_receipt_failures(
    receipt_path: Path,
    *,
    args: argparse.Namespace,
    binding: dict[str, Any],
    case_id: str,
    seed: int,
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
    expected_output = binding["output_contract"]["cases"][case_id]
    checks = {
        "status": receipt.get("status") == "pass",
        "seed": arguments.get("seed") == seed,
        "physics_substeps": arguments.get("physics_substeps") == 2,
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
        "nuisance_arguments_absent": all(
            arguments.get(name) is None
            for name in (
                "nuisance_receipt",
                "nuisance_receipt_sha256",
                "nuisance_case_id",
            )
        ),
        "nuisance_binding_absent": receipt.get("nuisance_profile_binding") is None,
        "nuisance_runtime_absent": receipt.get("nuisance_runtime") is None,
        "stage_sha256": provenance.get("requested_stage_sha256")
        == binding["stage_sha256"],
        "renderer_script_sha256": provenance.get("script_sha256")
        == binding["renderer_sha256"],
        "renderer": render.get("renderer") == "PathTracing",
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
        "physics_steps_per_second": timing.get("physics_steps_per_second") == 240,
        "physics_substeps_per_report_interval": timing.get(
            "physics_substeps_per_report_interval"
        )
        == 2,
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
        "two_substep_train_provenance": ablation
        == {
            "id": "accepted_two_substep_train_case",
            "qualification_only": False,
            "training_or_cohort_use_authorized": True,
            "changed_variable": "physics_substeps_per_report_interval",
            "accepted_qualification_receipt_sha256": (
                args.accepted_qualification_receipt_sha256
            ),
        },
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
    validate_artifacts: Callable[[Path], dict[str, Any]],
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
    if launch.get("renderer_script_sha256") != binding["renderer_sha256"]:
        launch_failures.append("launch_renderer_sha256")
    if launch.get("stage_sha256") != binding["stage_sha256"]:
        launch_failures.append("launch_stage_sha256")
    if launch.get("render_contract") != binding["render_contract"]:
        launch_failures.append("launch_render_contract")
    expected_command = _case_command(
        args,
        case_id=case_id,
        seed=seed,
        app_output=case_root / "app-output",
    )
    if launch.get("argv") != expected_command:
        launch_failures.append("launch_argv")
    receipt_failures = _case_receipt_failures(
        receipt_path,
        args=args,
        binding=binding,
        case_id=case_id,
        seed=seed,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--accepted-qualification-receipt", required=True, type=Path)
    parser.add_argument("--accepted-qualification-receipt-sha256", required=True)
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
    return parser


def run_supervisor(
    args: argparse.Namespace,
    *,
    run_process: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    validate_artifacts: Callable[[Path], dict[str, Any]] | None = None,
    active_processes: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    binding = _bind_inputs(args)
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
            if observed_binding != expected_binding:
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
            "expected_output_contract": deepcopy(binding["output_contract"]),
            "planned_case_count": len(binding["plan"]["execution_schedules"]),
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

        def finish(status: str, failure: dict[str, Any] | None = None) -> dict[str, Any]:
            summary["status"] = status
            summary["failure"] = failure
            summary["completed_at"] = _utc_now()
            _write_json_new(attempt_receipt, summary)
            return summary

        for ordinal, schedule in enumerate(
            binding["plan"]["execution_schedules"], start=1
        ):
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
                    validate_artifacts=validator,
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
                "stage_path": os.fspath(args.stage),
                "stage_sha256": binding["stage_sha256"],
                "renderer_script": os.fspath(args.renderer_script),
                "renderer_script_sha256": binding["renderer_sha256"],
                "python_executable": os.fspath(args.python_executable),
                "render_contract": dict(binding["render_contract"]),
                "argv": command,
            }
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
        expected_output = binding["output_contract"]
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

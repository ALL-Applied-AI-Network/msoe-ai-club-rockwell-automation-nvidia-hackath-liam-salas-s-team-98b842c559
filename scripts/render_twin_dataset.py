#!/usr/bin/env python3
"""Launch one bounded, provenance-recorded Task 2A FillCam capture attempt.

This wrapper is import-safe and help-safe without Isaac Sim.  Its CPU gates do
not prove Isaac startup, GPU health, shader compilation, visual geometry, or
semantic realism.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix.render_process import (  # noqa: E402
    CHILD_ENVIRONMENT_POLICY,
    atomic_write_json,
    build_child_environment,
    create_attempt_root,
    hash_regular_file,
    hash_regular_tree,
    load_experiment_manifest,
    reverify_gb10_preload,
    run_bounded_command,
    snapshot_gb10_preload,
    validate_requested_capture_settings,
    verify_capture_summary,
)


RENDERER = ROOT / "scripts/twin_render_sequence.py"
CAPTURE_MODULE = ROOT / "src/fluidmix/render_capture.py"
PROCESS_MODULE = ROOT / "src/fluidmix/render_process.py"
ATTEMPT_SCHEMA_VERSION = 1


def timeout_seconds(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be an integer") from error
    if parsed < 1800 or parsed > 7200:
        raise argparse.ArgumentTypeError("timeout must be between 1800 and 7200 seconds")
    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--twin", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--stage", default="stations/fill_station/fill_station.usda")
    parser.add_argument("--frames", default="1:687:15")
    parser.add_argument("--subframes", type=positive_int, default=32)
    parser.add_argument("--warmup", type=positive_int, default=60)
    parser.add_argument("--max-tries", type=positive_int, default=20)
    parser.add_argument("--seed", type=nonnegative_int, default=20261024)
    parser.add_argument("--require-liquid-pixels", action="store_true")
    parser.add_argument(
        "--renderer",
        choices=("PathTracing", "RaytracedLighting"),
        default="PathTracing",
    )
    parser.add_argument("--timeout", type=timeout_seconds, default=1800)
    parser.add_argument(
        "--environment-profile",
        choices=("generic", "gb10-aarch64"),
        default="generic",
    )
    return parser


def build_render_command(
    isaac_python: str | os.PathLike[str],
    renderer_script: str | os.PathLike[str],
    twin: str | os.PathLike[str],
    capture_root: str | os.PathLike[str],
    settings: Mapping[str, object],
) -> list[str]:
    """Construct the complete fixed-FillCam Task 2A list-form argv."""

    validate_requested_capture_settings(settings)
    command = [
        os.fspath(isaac_python),
        os.fspath(renderer_script),
        "--twin",
        os.fspath(twin),
        "--stage",
        str(settings["stage"]),
        "--camera",
        "FillCam",
        "--frames",
        str(settings["frames"]),
        "--subframes",
        str(settings["subframes"]),
        "--warmup",
        str(settings["warmup"]),
        "--max-tries",
        str(settings["max_tries"]),
        "--seed",
        str(settings["seed"]),
        "--renderer",
        str(settings["renderer"]),
    ]
    if settings["require_liquid_pixels"]:
        command.append("--require-liquid-pixels")
    command.extend(["--out", os.fspath(capture_root)])
    return command


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file_snapshot(
    path: str | os.PathLike[str], *, role: str, allow_input_symlink: bool = False
) -> dict[str, object]:
    candidate = Path(path).expanduser()
    try:
        metadata = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"{role} is unavailable: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) and not allow_input_symlink:
        raise ValueError(f"{role} must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve(strict=True)
    snapshot = hash_regular_file(resolved)
    return {"path": str(resolved), **snapshot}


def _resolve_executable(value: str) -> dict[str, object]:
    candidate = Path(value).expanduser()
    if not candidate.exists() and len(candidate.parts) == 1:
        located = shutil.which(value)
        if located is None:
            raise ValueError(f"Isaac Python executable was not found: {value}")
        candidate = Path(located)
    record = _regular_file_snapshot(
        candidate, role="Isaac Python executable", allow_input_symlink=True
    )
    if not os.access(record["path"], os.X_OK):
        raise ValueError(f"Isaac Python is not executable: {record['path']}")
    return record


def _tree_digest(tree: Mapping[str, object]) -> str:
    encoded = json.dumps(tree, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_provenance() -> dict[str, object]:
    def git(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"could not inspect Git provenance: {type(error).__name__}") from error

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD", "--")
    return {
        "commit": commit,
        "dirty": bool(status or diff),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _disk_facts(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _gpu_driver_facts() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi_not_found"}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": f"nvidia-smi_error:{type(error).__name__}"}
    if completed.returncode != 0:
        return {"available": False, "reason": f"nvidia-smi_exit:{completed.returncode}"}
    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            return {"available": False, "reason": "nvidia-smi_output_malformed"}
        gpus.append(
            {"name": parts[0], "driver_version": parts[1], "memory_total_mib": parts[2]}
        )
    if not gpus:
        return {"available": False, "reason": "nvidia-smi_reported_no_gpus"}
    return {"available": True, "gpus": gpus}


def _requested_settings(args: argparse.Namespace) -> dict[str, object]:
    settings = {
        "camera": "FillCam",
        "stage": args.stage,
        "frames": args.frames,
        "subframes": args.subframes,
        "warmup": args.warmup,
        "max_tries": args.max_tries,
        "seed": args.seed,
        "require_liquid_pixels": args.require_liquid_pixels,
        "renderer": args.renderer,
    }
    validate_requested_capture_settings(settings)
    return settings


def _source_snapshot(role: str, path: Path) -> dict[str, object]:
    return _regular_file_snapshot(path, role=role)


def _reverify_file(record: dict[str, object]) -> dict[str, object]:
    after = hash_regular_file(str(record["path"]))
    return {
        **record,
        "sha256_after": after["sha256"],
        "bytes_after": after["bytes"],
        "unchanged": (
            after["sha256"] == record["sha256"] and after["bytes"] == record["bytes"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = _requested_settings(args)
        isaac_python = _resolve_executable(args.isaac_python)
        runtime_lock = _regular_file_snapshot(args.runtime_lock, role="runtime lock")
        experiment = load_experiment_manifest(args.experiment_manifest, args.run_id)
        if experiment["payload"]["seed"] != args.seed:
            raise ValueError("experiment manifest seed does not match --seed")
        gb10_preload_before = (
            snapshot_gb10_preload()
            if args.environment_profile == "gb10-aarch64"
            else None
        )
        child_environment = build_child_environment(os.environ, args.environment_profile)

        twin_candidate = Path(args.twin).expanduser()
        if twin_candidate.is_symlink():
            raise ValueError("public twin root must not be a symlink")
        twin_before = hash_regular_tree(twin_candidate)
        twin = twin_candidate.resolve(strict=True)
        requested_stage = Path(args.stage).expanduser()
        stage = (
            requested_stage if requested_stage.is_absolute() else twin / requested_stage
        ).resolve(strict=True)
        if not stage.is_relative_to(twin) or not stage.is_file():
            raise ValueError("requested stage must be a regular file inside the public twin")
        camera = (twin / "cameras/FillCam.json").resolve(strict=True)
        if not camera.is_file():
            raise ValueError("public twin is missing the reviewed FillCam camera record")

        source_before = {
            "wrapper": _source_snapshot("launcher wrapper", Path(__file__).resolve()),
            "renderer": _source_snapshot("Task 2A renderer", RENDERER),
            "capture_module": _source_snapshot("Task 2A capture module", CAPTURE_MODULE),
            "process_module": _source_snapshot("process boundary module", PROCESS_MODULE),
        }
        git = _git_provenance()
        out_parent = Path(args.out_parent).expanduser()
        if out_parent.is_symlink():
            raise ValueError("output parent must not be a symlink")
        out_parent = out_parent.resolve(strict=True)
        if not out_parent.is_dir():
            raise ValueError("output parent must be an existing directory")
        host = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "controller_python": sys.version,
            "disk_before": _disk_facts(out_parent),
            "gpu_driver": _gpu_driver_facts(),
        }
    except (OSError, ValueError) as error:
        parser.error(str(error))

    started_utc = _utc_now()
    attempt_root = create_attempt_root(out_parent, args.run_id)
    log_path = attempt_root / "render.log"
    capture_root = attempt_root / "capture"
    command = build_render_command(
        str(isaac_python["path"]), RENDERER.resolve(), twin, capture_root, settings
    )
    failures: list[str] = []
    process_result: dict[str, object] | None = None
    capture_record: dict[str, object] = {
        "root": str(capture_root),
        "verified": False,
        "inner_summary_sha256": None,
    }
    try:
        process_result = run_bounded_command(
            command,
            log_path,
            timeout_seconds=args.timeout,
            environment=child_environment,
        )
    except Exception as error:  # preserve the owned attempt and finalize its evidence
        failures.append(f"process_runner_error:{type(error).__name__}:{error}")

    if process_result is None:
        failures.append("renderer_process_result_missing")
    else:
        if process_result.get("spawn_error") is not None:
            failures.append("renderer_spawn_error")
        if process_result.get("timed_out") is True:
            failures.append("renderer_timeout")
        if process_result.get("returncode") != 0:
            failures.append("renderer_returncode_nonzero")
        if process_result.get("process_group_cleanup_required") is True:
            failures.append("renderer_process_group_cleanup_required")
        if process_result.get("process_group_quiescent") is not True:
            failures.append("renderer_process_group_not_quiescent")

    try:
        twin_after = hash_regular_tree(twin)
        twin_unchanged = twin_after == twin_before
        if not twin_unchanged:
            failures.append("public_twin_changed")
        twin_error = None
    except (OSError, ValueError) as error:
        twin_after = None
        twin_unchanged = False
        twin_error = f"{type(error).__name__}: {error}"
        failures.append("public_twin_unreadable_after")

    file_records: dict[str, dict[str, object]] = {}
    for role, before in {
        "isaac_python": isaac_python,
        "runtime_lock": runtime_lock,
        **source_before,
    }.items():
        try:
            file_records[role] = _reverify_file(before)
            if file_records[role]["unchanged"] is not True:
                failures.append(f"{role}_changed")
        except (OSError, ValueError) as error:
            file_records[role] = {**before, "unchanged": False, "after_error": str(error)}
            failures.append(f"{role}_unreadable_after")

    experiment_provenance = {
        key: value for key, value in experiment.items() if key != "payload"
    }
    try:
        experiment_after = load_experiment_manifest(
            str(experiment["path"]), args.run_id
        )
        experiment_provenance.update(
            {
                "sha256_after": experiment_after["sha256"],
                "bytes_after": experiment_after["bytes"],
                "unchanged": (
                    experiment_after["sha256"] == experiment["sha256"]
                    and experiment_after["bytes"] == experiment["bytes"]
                ),
            }
        )
        if experiment_provenance["unchanged"] is not True:
            failures.append("experiment_manifest_changed")
    except ValueError as error:
        experiment_provenance.update({"unchanged": False, "after_error": str(error)})
        failures.append("experiment_manifest_unreadable_after")

    if gb10_preload_before is None:
        gb10_preload_provenance: dict[str, object] = {
            "required": False,
            "unchanged": None,
        }
    else:
        gb10_preload_provenance = {
            "required": True,
            **reverify_gb10_preload(gb10_preload_before),
        }
        if gb10_preload_provenance["unchanged"] is not True:
            failures.append("gb10_preload_changed_or_unreadable")

    summary_path = capture_root / "summary.json"
    if summary_path.exists() and not summary_path.is_symlink():
        try:
            capture_record["inner_summary_sha256"] = hash_regular_file(summary_path)[
                "sha256"
            ]
        except ValueError as error:
            capture_record["inner_summary_hash_error"] = str(error)
    try:
        verified = verify_capture_summary(capture_root, settings)
        capture_record.update(
            {
                "verified": True,
                "inner_summary_sha256": verified["summary_sha256"],
                "artifact_count": len(verified["artifacts"]),
            }
        )
    except (OSError, ValueError) as error:
        capture_record["verification_error"] = f"{type(error).__name__}: {error}"
        failures.append("capture_verification_failed")

    status = "mechanical_gates_passed" if not failures else "failed"
    twin_record: dict[str, object] = {
        "root": str(twin),
        "files_before": twin_before,
        "tree_sha256_before": _tree_digest(twin_before),
        "files_after": twin_after,
        "tree_sha256_after": _tree_digest(twin_after) if twin_after is not None else None,
        "unchanged": twin_unchanged,
    }
    if twin_error is not None:
        twin_record["after_error"] = twin_error
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": status,
        "visual_check_status": "pending",
        "failure_reasons": failures,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "safe_command": command,
        "requested_settings": settings,
        "environment_policy": {
            "name": CHILD_ENVIRONMENT_POLICY,
            "profile": args.environment_profile,
        },
        "timeout_seconds": args.timeout,
        "process": process_result,
        "capture": capture_record,
        "provenance": {
            "git": git,
            "experiment_manifest": experiment_provenance,
            "gb10_preload": gb10_preload_provenance,
            "twin": twin_record,
            "files": file_records,
        },
        "host": {**host, "disk_after": _disk_facts(out_parent)},
        "limits": (
            "CPU lifecycle and artifact gates do not prove Isaac startup, GPU health, "
            "shader compilation, visual geometry, or semantic realism. Visual review remains pending."
        ),
    }
    atomic_write_json(attempt_root, "attempt.json", attempt)
    return 0 if status == "mechanical_gates_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

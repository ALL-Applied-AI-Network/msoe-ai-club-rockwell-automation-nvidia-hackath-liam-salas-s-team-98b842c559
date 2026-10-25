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
    CHILD_ENVIRONMENT_PROFILES,
    LOCAL_ISAAC62_ATTEMPT_POLICY,
    LOCAL_ISAAC62_PROFILE,
    atomic_write_json,
    build_child_environment,
    create_attempt_root,
    hash_regular_file,
    hash_regular_tree,
    load_experiment_manifest,
    parse_frame_spec,
    reverify_gb10_preload,
    run_bounded_command,
    run_retriable_render_command,
    snapshot_local_isaac62_install_metadata,
    snapshot_gb10_preload,
    validate_local_isaac62_manifest_binding,
    validate_requested_capture_settings,
    verify_capture_summary,
)
from fluidmix.twin_appearance import (  # noqa: E402
    appearance_command_plan,
    inspect_appearance_layer,
)
from fluidmix.twin_stage_appearance import (  # noqa: E402
    inspect_stage_appearance_layer,
    stage_appearance_command_plan,
)


RENDERER = ROOT / "scripts/twin_render_sequence.py"
CAPTURE_MODULE = ROOT / "src/fluidmix/render_capture.py"
PROCESS_MODULE = ROOT / "src/fluidmix/render_process.py"
APPEARANCE_MODULE = ROOT / "src/fluidmix/twin_appearance.py"
STAGE_APPEARANCE_MODULE = ROOT / "src/fluidmix/twin_stage_appearance.py"
ATTEMPT_SCHEMA_VERSION = 1
DEFAULT_STAGE = "stations/fill_station/fill_station.usda"


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
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument("--stage", default=DEFAULT_STAGE)
    stage_group.add_argument("--appearance-stage")
    stage_group.add_argument("--stage-appearance-stage")
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
        choices=CHILD_ENVIRONMENT_PROFILES,
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
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical_metadata = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"Isaac Python entry is unavailable: {lexical}") from error
    if stat.S_ISLNK(lexical_metadata.st_mode):
        kind = "symlink"
        link_target: str | None = os.readlink(lexical)
    elif stat.S_ISREG(lexical_metadata.st_mode):
        kind = "regular"
        link_target = None
    else:
        raise ValueError(
            f"Isaac Python entry must be a regular file or symlink: {lexical}"
        )
    try:
        resolved = lexical.resolve(strict=True)
        target_metadata = os.lstat(resolved)
    except OSError as error:
        raise ValueError(f"Isaac Python target is unavailable: {lexical}") from error
    if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(
        target_metadata.st_mode
    ):
        raise ValueError(
            f"Isaac Python target must be a regular non-symlink file: {resolved}"
        )
    if not os.access(lexical, os.X_OK) or not os.access(resolved, os.X_OK):
        raise ValueError(f"Isaac Python entry or target is not executable: {lexical}")

    def identity(metadata: os.stat_result) -> dict[str, int]:
        return {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }

    target_hash = hash_regular_file(resolved)
    try:
        lexical_after = os.lstat(lexical)
        target_after = os.lstat(resolved)
        link_target_after = (
            os.readlink(lexical) if stat.S_ISLNK(lexical_after.st_mode) else None
        )
    except OSError as error:
        raise ValueError("Isaac Python entry changed while snapshotting") from error
    if (
        identity(lexical_after) != identity(lexical_metadata)
        or identity(target_after) != identity(target_metadata)
        or link_target_after != link_target
    ):
        raise ValueError("Isaac Python entry changed while snapshotting")
    return {
        "lexical_path": str(lexical),
        "lexical_identity": {
            "kind": kind,
            "link_target": link_target,
            **identity(lexical_metadata),
        },
        "resolved_target": {
            "path": str(resolved),
            **identity(target_metadata),
            **target_hash,
        },
    }


def _reverify_executable(before: dict[str, object]) -> dict[str, object]:
    try:
        after = _resolve_executable(str(before["lexical_path"]))
    except (KeyError, ValueError) as error:
        return {
            "before": before,
            "after": None,
            "unchanged": False,
            "after_error": str(error),
        }
    return {
        "before": before,
        "after": after,
        "unchanged": before == after,
    }


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
    appearance_stage = (
        args.stage_appearance_stage
        if args.stage_appearance_stage is not None
        else args.appearance_stage
    )
    settings = {
        "camera": "FillCam",
        "stage": appearance_stage if appearance_stage is not None else args.stage,
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


def _exact_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _verify_appearance_capture_provenance(
    summary: Mapping[str, object], inspection: Mapping[str, object]
) -> None:
    try:
        inputs = summary["inputs"]
        assert isinstance(inputs, Mapping)
        stage = inputs["stage"]
        stack = inputs["non_session_usd_layer_stack"]
        assert isinstance(stage, Mapping)
        assert isinstance(stack, list)
    except (AssertionError, KeyError, TypeError) as error:
        raise ValueError("capture appearance-stage provenance is malformed") from error
    expected_stage = {
        "requested": inspection["path"],
        "resolved_path": inspection["path"],
        "sha256": inspection["stage_sha256"],
    }
    if not _exact_json_equal(dict(stage), expected_stage):
        raise ValueError("capture root-stage provenance does not match appearance input")
    expected_stack = inspection["non_session_usd_layer_stack"]
    if not isinstance(expected_stack, list) or len(stack) != len(expected_stack):
        raise ValueError("capture non-session layer-stack length does not match preflight")
    normalized = []
    for record in stack:
        if not isinstance(record, Mapping) or set(record) != {
            "ordinal",
            "identifier",
            "resolved_path",
            "sha256",
            "bytes",
        }:
            raise ValueError("capture non-session layer-stack record is malformed")
        if (
            not isinstance(record["identifier"], str)
            or record["identifier"] != record["resolved_path"]
        ):
            raise ValueError("capture non-session layer identifier is not the resolved file")
        normalized.append(
            {
                "ordinal": record["ordinal"],
                "resolved_path": record["resolved_path"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
        )
    if not _exact_json_equal(normalized, expected_stack):
        raise ValueError("capture non-session layer stack does not exactly match preflight")


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
        local_isaac62_binding = (
            validate_local_isaac62_manifest_binding(
                experiment["payload"],
                isaac_python=str(isaac_python["lexical_path"]),
                host_name=platform.node(),
                machine=platform.machine(),
            )
            if args.environment_profile == LOCAL_ISAAC62_PROFILE
            else None
        )
        local_isaac62_metadata = (
            snapshot_local_isaac62_install_metadata()
            if args.environment_profile == LOCAL_ISAAC62_PROFILE
            else None
        )
        gb10_preload_before = (
            snapshot_gb10_preload()
            if args.environment_profile in {"gb10-aarch64", LOCAL_ISAAC62_PROFILE}
            else None
        )
        child_environment = build_child_environment(os.environ, args.environment_profile)

        twin_candidate = Path(args.twin).expanduser()
        if twin_candidate.is_symlink():
            raise ValueError("public twin root must not be a symlink")
        twin_before = hash_regular_tree(twin_candidate)
        twin = twin_candidate.resolve(strict=True)
        out_parent_candidate = Path(args.out_parent).expanduser()
        if out_parent_candidate.is_symlink():
            raise ValueError("output parent must not be a symlink")
        out_parent = out_parent_candidate.resolve(strict=True)
        if not out_parent.is_dir():
            raise ValueError("output parent must be an existing directory")

        appearance_before: dict[str, object] | None = None
        appearance_inspector = None
        requested_appearance_argument = (
            args.stage_appearance_stage
            if args.stage_appearance_stage is not None
            else args.appearance_stage
        )
        if requested_appearance_argument is None:
            requested_stage = Path(args.stage).expanduser()
            stage = (
                requested_stage if requested_stage.is_absolute() else twin / requested_stage
            ).resolve(strict=True)
            if not stage.is_relative_to(twin) or not stage.is_file():
                raise ValueError(
                    "requested stage must be a regular file inside the public twin"
                )
        else:
            requested_appearance = Path(requested_appearance_argument).expanduser()
            if not requested_appearance.is_absolute():
                raise ValueError("appearance stage must be an absolute path")
            lexical_appearance = Path(os.path.abspath(requested_appearance))
            if str(requested_appearance) != str(lexical_appearance):
                raise ValueError("appearance stage must be a canonical absolute path")
            if lexical_appearance.parent != out_parent / "appearance-layers":
                raise ValueError(
                    "appearance stage must be a direct child of output-parent/appearance-layers"
                )
            if args.stage_appearance_stage is not None:
                appearance_inspector = inspect_stage_appearance_layer
                appearance_before = appearance_inspector(lexical_appearance, twin)
                required_plan = stage_appearance_command_plan(appearance_before)
                authored_time_codes = {
                    sample["time_code"] for sample in appearance_before["samples"]
                }
                requested_time_codes = set(parse_frame_spec(settings["frames"]))
                if not authored_time_codes.issubset(requested_time_codes):
                    raise ValueError(
                        "stage appearance frames must include every authored sample time"
                    )
            else:
                appearance_inspector = inspect_appearance_layer
                appearance_before = appearance_inspector(lexical_appearance, twin)
                required_plan = appearance_command_plan(appearance_before)
            supplied_plan = experiment["payload"]["command_plan"].get(
                "appearance_layer"
            )
            if not _exact_json_equal(supplied_plan, required_plan):
                raise ValueError(
                    "experiment command_plan.appearance_layer does not exactly match the layer"
                )
            stage = Path(appearance_before["path"])
            if args.stage_appearance_stage is not None:
                args.stage_appearance_stage = str(stage)
            else:
                args.appearance_stage = str(stage)

        camera = (twin / "cameras/FillCam.json").resolve(strict=True)
        if not camera.is_file():
            raise ValueError("public twin is missing the reviewed FillCam camera record")

        source_before = {
            "wrapper": _source_snapshot("launcher wrapper", Path(__file__).resolve()),
            "renderer": _source_snapshot("Task 2A renderer", RENDERER),
            "capture_module": _source_snapshot("Task 2A capture module", CAPTURE_MODULE),
            "process_module": _source_snapshot("process boundary module", PROCESS_MODULE),
            "appearance_module": _source_snapshot(
                "appearance boundary module", APPEARANCE_MODULE
            ),
            "stage_appearance_module": _source_snapshot(
                "stage appearance boundary module", STAGE_APPEARANCE_MODULE
            ),
        }
        git = _git_provenance()
        host = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "controller_python": sys.version,
            "disk_before": _disk_facts(out_parent),
            "gpu_driver": _gpu_driver_facts(),
        }
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    started_utc = _utc_now()
    attempt_root = create_attempt_root(out_parent, args.run_id)
    log_path = attempt_root / "render.log"
    capture_root = attempt_root / "capture"
    command = build_render_command(
        str(isaac_python["lexical_path"]),
        RENDERER.resolve(),
        twin,
        (
            "capture"
            if args.environment_profile == LOCAL_ISAAC62_PROFILE
            else capture_root
        ),
        settings,
    )
    failures: list[str] = []
    process_result: dict[str, object] | None = None
    process_attempts: dict[str, object] | None = None
    try:
        if args.environment_profile == LOCAL_ISAAC62_PROFILE:
            process_attempts = run_retriable_render_command(
                command,
                attempt_root,
                timeout_seconds=args.timeout,
                environment=child_environment,
                attempt_policy=LOCAL_ISAAC62_ATTEMPT_POLICY,
            )
            recorded_attempts = process_attempts["attempts"]
            if not isinstance(recorded_attempts, list) or not recorded_attempts:
                raise RuntimeError("local Isaac 6.2 process attempts are missing")
            selected = process_attempts["selected_attempt_number"]
            selected_index = (
                int(selected) - 1
                if selected is not None
                else len(recorded_attempts) - 1
            )
            selected_record = recorded_attempts[selected_index]
            if not isinstance(selected_record, Mapping):
                raise RuntimeError("local Isaac 6.2 selected attempt is malformed")
            process_result = dict(selected_record["process"])
            capture_root = Path(str(selected_record["capture_root"]))
            log_path = Path(str(process_result["log_path"]))
            if process_attempts["status"] != "process_succeeded":
                failures.append("renderer_process_attempt_policy_failed")
        else:
            process_result = run_bounded_command(
                command,
                log_path,
                timeout_seconds=args.timeout,
                environment=child_environment,
            )
    except Exception as error:  # preserve the owned attempt and finalize its evidence
        failures.append(f"process_runner_error:{type(error).__name__}:{error}")

    capture_record: dict[str, object] = {
        "root": str(capture_root),
        "verified": False,
        "inner_summary_sha256": None,
        "appearance_provenance_verified": (
            False if appearance_before is not None else None
        ),
    }

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
        "runtime_lock": runtime_lock,
        **(
            {"isaac_distribution_metadata": local_isaac62_metadata}
            if local_isaac62_metadata is not None
            else {}
        ),
        **source_before,
    }.items():
        try:
            file_records[role] = _reverify_file(before)
            if file_records[role]["unchanged"] is not True:
                failures.append(f"{role}_changed")
        except (OSError, ValueError) as error:
            file_records[role] = {**before, "unchanged": False, "after_error": str(error)}
            failures.append(f"{role}_unreadable_after")
    file_records["isaac_python"] = _reverify_executable(isaac_python)
    if file_records["isaac_python"]["unchanged"] is not True:
        if file_records["isaac_python"].get("after") is None:
            failures.append("isaac_python_unreadable_after")
        else:
            failures.append("isaac_python_changed")

    if appearance_before is None:
        appearance_provenance: dict[str, object] = {
            "required": False,
            "path": None,
            "sha256": None,
            "bytes": None,
            "sha256_after": None,
            "bytes_after": None,
            "unchanged": None,
            "inspection_before": None,
            "inspection_after": None,
        }
    else:
        appearance_provenance = {
            "required": True,
            "path": appearance_before["path"],
            "sha256": appearance_before["stage_sha256"],
            "bytes": appearance_before["stage_bytes"],
            "sha256_after": None,
            "bytes_after": None,
            "unchanged": False,
            "inspection_before": appearance_before,
            "inspection_after": None,
        }
        try:
            if appearance_inspector is None:
                raise ValueError("appearance inspector was not bound during preflight")
            appearance_after = appearance_inspector(stage, twin)
            appearance_provenance.update(
                {
                    "sha256_after": appearance_after["stage_sha256"],
                    "bytes_after": appearance_after["stage_bytes"],
                    "inspection_after": appearance_after,
                    "unchanged": appearance_after == appearance_before,
                }
            )
            if appearance_provenance["unchanged"] is not True:
                failures.append("appearance_stage_changed")
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            appearance_provenance["after_error"] = f"{type(error).__name__}: {error}"
            failures.append("appearance_stage_unreadable_after")

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
        if appearance_before is not None:
            try:
                _verify_appearance_capture_provenance(
                    verified["summary"], appearance_before
                )
                capture_record["appearance_provenance_verified"] = True
            except (TypeError, ValueError) as error:
                capture_record["appearance_provenance_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                failures.append("capture_stage_provenance_mismatch")
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
            "appearance_layer": appearance_provenance,
            "twin": twin_record,
            "files": file_records,
        },
        "host": {**host, "disk_after": _disk_facts(out_parent)},
        "limits": (
            "CPU lifecycle and artifact gates do not prove Isaac startup, GPU health, "
            "shader compilation, visual geometry, or semantic realism. Visual review remains pending."
        ),
    }
    if local_isaac62_binding is not None:
        attempt["local_isaac62_binding"] = local_isaac62_binding
        attempt["process_attempts"] = process_attempts
    atomic_write_json(attempt_root, "attempt.json", attempt)
    return 0 if status == "mechanical_gates_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

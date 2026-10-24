#!/usr/bin/env python3
"""Run the one frozen S00 FillCam recovery attempt.

This orchestrator uses argument lists rather than a shell, records every command and
has one timeout with no retry. Outputs and logs belong under the ignored output root.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.render_manifest import load_render_manifest
from fluidmix.render_runner import (
    build_child_environment,
    build_render_command,
    hash_regular_tree,
    inspect_render_output,
    parse_live_setting_attestation,
    resolve_private_new_output,
    sha256_file,
    verify_source_lock,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_command(command: list[str], log_path: Path, timeout_seconds: int, env: dict) -> dict:
    started = utc_now()
    before = time.monotonic()
    timed_out = False
    spawn_error = None
    returncode = None
    log_path = resolve_private_new_output(log_path)
    parent_metadata = log_path.parent.stat(follow_symlinks=False)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(log_path.parent, directory_flags)
    try:
        opened_parent = os.fstat(directory_fd)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        ):
            raise RuntimeError("log parent changed before exclusive open")
        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        log_fd = os.open(log_path.name, file_flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(log_fd, "w") as log:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    text=True,
                )
            except OSError as exc:
                spawn_error = f"{type(exc).__name__}: {exc}"
                log.write(spawn_error + "\n")
            else:
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        returncode = process.wait()
        read_fd = os.open(
            log_path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        with os.fdopen(read_fd, "rb") as stream:
            log_bytes = stream.read()
    finally:
        os.close(directory_fd)
    result = {
        "command": command,
        "started_utc": started,
        "ended_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - before,
        "returncode": returncode,
        "timed_out": timed_out,
        "log": str(log_path),
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "wrote_marker": b"wrote " in log_bytes,
    }
    try:
        result["live_setting_attestation"] = parse_live_setting_attestation(
            log_bytes.decode("utf-8", errors="replace")
        )
    except ValueError as exc:
        result["live_setting_attestation"] = None
        result["live_setting_attestation_error"] = str(exc)
    if spawn_error is not None:
        result["spawn_error"] = spawn_error
    return result


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve())
    raise ValueError(f"executable not found: {value}")


def create_attempt_output_root(parent: Path, attempt_id: str) -> Path:
    """Exclusively create the private output root for the immutable attempt."""
    if parent.is_symlink():
        raise ValueError(f"attempt parent must not be a symlink: {parent}")
    approved = parent.resolve(strict=True)
    if not approved.is_dir():
        raise ValueError(f"attempt parent must be a directory: {approved}")
    if approved.stat().st_mode & 0o077:
        raise ValueError(f"attempt parent must be private (mode 0700): {approved}")
    if not attempt_id or Path(attempt_id).name != attempt_id:
        raise ValueError("attempt_id must be one path component")
    root = approved / attempt_id
    root.mkdir(mode=0o700, exist_ok=False)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--isaac-python-sha256", required=True)
    parser.add_argument("--twin", required=True)
    parser.add_argument("--out-parent", required=True)
    args = parser.parse_args()

    try:
        manifest_candidate = Path(args.manifest).expanduser()
        manifest = load_render_manifest(manifest_candidate)
        manifest_path = manifest_candidate.resolve(strict=True)
        source_lock_candidate = Path(args.source_lock).expanduser()
        source_verification_before = verify_source_lock(
            source_lock_candidate,
            expected_lock_sha256=args.source_lock_sha256,
            root=ROOT,
        )
        source_lock_path = source_lock_candidate.resolve(strict=True)
        if manifest.schema_version != 2:
            raise ValueError("recovery runner requires schema version 2")
        isaac_python = resolve_executable(args.isaac_python)
        isaac_python_sha256 = sha256_file(isaac_python)
        if isaac_python_sha256 != args.isaac_python_sha256:
            raise ValueError("Isaac Python wrapper digest does not match pinned SHA-256")
        twin_candidate = Path(args.twin).expanduser()
        twin_manifest_before = hash_regular_tree(twin_candidate)
        twin = twin_candidate.resolve(strict=True)
        canonical_renderer = (twin / "tools/render_example.py").resolve(strict=True)
        recovery_renderer_source = ROOT / "scripts/render_fillcam_recovery.py"
        recovery_input_hashes = {
            "runner_sha256": sha256_file(Path(__file__)),
            "recovery_renderer_sha256": sha256_file(recovery_renderer_source),
            "manifest_loader_sha256": sha256_file(
                ROOT / "src/fluidmix/render_manifest.py"
            ),
            "render_runner_sha256": sha256_file(
                ROOT / "src/fluidmix/render_runner.py"
            ),
            "canonical_renderer_sha256": sha256_file(canonical_renderer),
            "station_sha256": sha256_file(
                twin / "stations/fill_station/fill_station.usda"
            ),
        }
        recovery_renderer = recovery_renderer_source.resolve(strict=True)
        station = (twin / "stations/fill_station/fill_station.usda").resolve(strict=True)
        out = create_attempt_output_root(
            Path(args.out_parent).expanduser(), manifest.attempt_id
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    images = out / "images"
    logs = out / "logs"
    images.mkdir(mode=0o700)
    logs.mkdir(mode=0o700)
    summary_path = out / "summary.json"
    image = images / "S00.png"
    log = logs / "S00.render.log"
    environment = build_child_environment(os.environ)

    summary = {
        "schema_version": 2,
        "attempt_id": manifest.attempt_id,
        "started_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_lock": str(source_lock_path),
        "source_lock_sha256": args.source_lock_sha256,
        "source_verification_before": source_verification_before,
        "isaac_python": isaac_python,
        "isaac_python_sha256_before": isaac_python_sha256,
        "twin": str(twin),
        "twin_manifest_before": twin_manifest_before,
        "recovery_inputs": recovery_input_hashes,
        "child_environment": environment,
        "render": None,
        "output": None,
        "visual_check_status": "pending_local_review",
        "failure_reasons": [],
        "status": "running",
    }
    save_summary(summary_path, summary)

    render_command = build_render_command(
        isaac_python,
        str(recovery_renderer),
        station,
        image,
        manifest,
        manifest.smoke,
        twin=twin,
    )
    render_result = run_command(
        render_command, log, manifest.timeout_seconds, environment
    )
    summary["render"] = render_result
    try:
        summary["output"] = inspect_render_output(image)
    except Exception as exc:
        summary["output"] = {"inspection_error": f"{type(exc).__name__}: {exc}"}
    try:
        twin_manifest_after = hash_regular_tree(twin)
        summary["twin_manifest_after"] = twin_manifest_after
        summary["source_twin_unchanged"] = twin_manifest_after == twin_manifest_before
    except (OSError, ValueError) as exc:
        summary["twin_manifest_after_error"] = str(exc)
        summary["source_twin_unchanged"] = False
    try:
        source_verification_after = verify_source_lock(
            source_lock_path,
            expected_lock_sha256=args.source_lock_sha256,
            root=ROOT,
        )
        summary["source_verification_after"] = source_verification_after
        summary["recovery_source_unchanged"] = (
            source_verification_after == source_verification_before
        )
    except (OSError, ValueError) as exc:
        summary["source_verification_after_error"] = str(exc)
        summary["recovery_source_unchanged"] = False
    try:
        isaac_python_sha256_after = sha256_file(isaac_python)
        summary["isaac_python_sha256_after"] = isaac_python_sha256_after
        summary["isaac_python_unchanged"] = (
            isaac_python_sha256_after == isaac_python_sha256
        )
    except (OSError, ValueError) as exc:
        summary["isaac_python_after_error"] = str(exc)
        summary["isaac_python_unchanged"] = False

    failures = summary["failure_reasons"]
    if render_result.get("timed_out"):
        failures.append("renderer_timeout")
    if render_result.get("returncode") != 0:
        failures.append("renderer_returncode")
    if render_result.get("wrote_marker") is not True:
        failures.append("renderer_write_marker_missing")
    if render_result.get("live_setting_attestation") is None:
        failures.append("live_setting_attestation_missing_or_invalid")
    if not summary["source_twin_unchanged"]:
        failures.append("source_twin_changed_or_unreadable")
    if not summary["recovery_source_unchanged"]:
        failures.append("recovery_source_changed_or_unreadable")
    if not summary["isaac_python_unchanged"]:
        failures.append("isaac_python_changed_or_unreadable")
    output = summary["output"]
    if output.get("decoder_passed") is not True:
        failures.append("output_decoder_failed")
    if output.get("content_gate_passed") is not True:
        failures.append("output_content_gate_failed")
    summary["status"] = "failed" if failures else "mechanical_checks_passed"
    summary["ended_utc"] = utc_now()
    save_summary(summary_path, summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

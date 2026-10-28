"""CPU-only process and provenance safeguards for external twin rendering.

These helpers can validate local files, process lifecycle, and Task 2A capture
evidence.  They cannot prove Isaac startup, GPU health, shader compilation,
visual geometry, or semantic realism.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import signal
import stat
import subprocess
import time


CHILD_ENVIRONMENT_POLICY = "explicit-runtime-allowlist-v1"
LOCAL_ISAAC62_PROFILE = "gb10-aarch64-isaac62"
LOCAL_ISAAC62_PYTHON = (
    "/home/hackathon5/liamreporoot/isaacsim62-env/.venv/bin/python"
)
LOCAL_ISAAC62_METADATA = (
    "/home/hackathon5/liamreporoot/isaacsim62-env/.venv/lib/python3.12/"
    "site-packages/isaacsim-6.2.0.0.dist-info/METADATA"
)
LOCAL_ISAAC62_RUNTIME_CONTRACT = {
    "host_name": "spark5",
    "machine": "aarch64",
    "runtime_name": "NVIDIA Isaac Sim",
    "isaac_sim_version": "6.2.0.0",
    "kit_version": "110.4.0",
    "isaac_python": LOCAL_ISAAC62_PYTHON,
    "environment_profile": LOCAL_ISAAC62_PROFILE,
}
LOCAL_ISAAC62_ATTEMPT_POLICY = {
    "max_process_attempts": 2,
    "no_output_timeout_seconds": 180,
    "startup_readiness_timeout_seconds": 420,
    "fast_crash_retry_seconds": 180,
    "startup_readiness_marker": "Simulation App Startup Complete",
}
CHILD_ENVIRONMENT_PROFILES = (
    "generic",
    "gb10-aarch64",
    LOCAL_ISAAC62_PROFILE,
)
GB10_PRELOAD_LEXICAL = "/lib/aarch64-linux-gnu/libgomp.so.1"
GB10_PRELOAD = Path(GB10_PRELOAD_LEXICAL)
GB10_PRELOAD_ALLOWED_ROOTS = (
    Path("/lib/aarch64-linux-gnu"),
    Path("/usr/lib/aarch64-linux-gnu"),
)
TERMINATION_GRACE_SECONDS = 10.0
_ALLOWED_CHILD_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_RUNTIME_DIR",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
)


def build_child_environment(source: Mapping[str, str], profile: str) -> dict[str, str]:
    """Build the reviewed explicit environment for one renderer child."""

    if profile not in CHILD_ENVIRONMENT_PROFILES:
        raise ValueError(
            f"environment profile must be one of {CHILD_ENVIRONMENT_PROFILES!r}"
        )
    child = {name: source[name] for name in _ALLOWED_CHILD_ENV if name in source}
    if not all(isinstance(name, str) and isinstance(value, str) for name, value in child.items()):
        raise TypeError("child environment names and values must be strings")
    child["OMNI_KIT_ACCEPT_EULA"] = "YES"
    child["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    if profile in {"gb10-aarch64", LOCAL_ISAAC62_PROFILE}:
        snapshot_gb10_preload()
        child["LD_PRELOAD"] = GB10_PRELOAD_LEXICAL
    return child


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_local_isaac62_manifest_binding(
    payload: Mapping[str, object],
    *,
    isaac_python: str,
    host_name: str,
    machine: str,
) -> dict[str, object]:
    """Require the exact local-GB10 runtime and two-attempt watchdog binding."""

    if not isinstance(payload, Mapping):
        raise TypeError("local Isaac 6.2 manifest must be an object")
    runtime = payload.get("render_runtime")
    policy = payload.get("process_attempt_policy")
    if not _strict_json_equal(runtime, LOCAL_ISAAC62_RUNTIME_CONTRACT):
        raise ValueError("local Isaac 6.2 render_runtime binding drifted")
    if not _strict_json_equal(policy, LOCAL_ISAAC62_ATTEMPT_POLICY):
        raise ValueError("local Isaac 6.2 process_attempt_policy binding drifted")
    if isaac_python != LOCAL_ISAAC62_PYTHON:
        raise ValueError("local Isaac 6.2 Python path is not the reviewed exact path")
    if host_name != LOCAL_ISAAC62_RUNTIME_CONTRACT["host_name"]:
        raise ValueError("local Isaac 6.2 host binding does not match this host")
    if machine != LOCAL_ISAAC62_RUNTIME_CONTRACT["machine"]:
        raise ValueError("local Isaac 6.2 machine binding does not match this machine")
    if not isinstance(runtime, dict) or not isinstance(policy, dict):
        raise ValueError("local Isaac 6.2 manifest bindings must be objects")
    return {
        "render_runtime": dict(runtime),
        "process_attempt_policy": dict(policy),
    }


def _filesystem_identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def snapshot_gb10_preload() -> dict[str, object]:
    """Snapshot the fixed lexical preload and its confined resolved target."""

    try:
        lexical_metadata = os.lstat(GB10_PRELOAD)
    except OSError as error:
        raise ValueError(
            f"reviewed GB10 preload lexical path is unavailable: {GB10_PRELOAD}"
        ) from error
    if stat.S_ISLNK(lexical_metadata.st_mode):
        kind = "symlink"
        link_target: str | None = os.readlink(GB10_PRELOAD)
    elif stat.S_ISREG(lexical_metadata.st_mode):
        kind = "regular"
        link_target = None
    else:
        raise ValueError(
            f"reviewed GB10 preload lexical path has unsupported type: {GB10_PRELOAD}"
        )
    try:
        resolved = GB10_PRELOAD.resolve(strict=True)
        target_metadata = os.lstat(resolved)
    except OSError as error:
        raise ValueError("reviewed GB10 preload target is unavailable") from error
    if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(
        target_metadata.st_mode
    ):
        raise ValueError("reviewed GB10 preload target must be a regular non-symlink file")

    allowed_roots: list[Path] = []
    for root in GB10_PRELOAD_ALLOWED_ROOTS:
        try:
            allowed = Path(root).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"allowed GB10 library root is unavailable: {root}") from error
        if not allowed.is_dir():
            raise ValueError(f"allowed GB10 library root is not a directory: {root}")
        allowed_roots.append(allowed)
    if not any(
        resolved == allowed or resolved.is_relative_to(allowed)
        for allowed in allowed_roots
    ):
        raise ValueError(
            f"reviewed GB10 preload target is outside allowed GB10 library roots: {resolved}"
        )

    lexical_identity: dict[str, object] = {
        "kind": kind,
        "link_target": link_target,
        **_filesystem_identity(lexical_metadata),
    }
    target_hash = hash_regular_file(resolved)
    try:
        lexical_after = os.lstat(GB10_PRELOAD)
        target_after = os.lstat(resolved)
        link_target_after = (
            os.readlink(GB10_PRELOAD)
            if stat.S_ISLNK(lexical_after.st_mode)
            else None
        )
    except OSError as error:
        raise ValueError("reviewed GB10 preload changed while snapshotting") from error
    if (
        _filesystem_identity(lexical_after)
        != _filesystem_identity(lexical_metadata)
        or _filesystem_identity(target_after)
        != _filesystem_identity(target_metadata)
        or link_target_after != link_target
    ):
        raise ValueError("reviewed GB10 preload changed while snapshotting")
    return {
        "lexical_path": str(GB10_PRELOAD),
        "lexical_identity": lexical_identity,
        "resolved_target": {
            "path": str(resolved),
            **_filesystem_identity(target_metadata),
            **target_hash,
        },
    }


def reverify_gb10_preload(before: Mapping[str, object]) -> dict[str, object]:
    """Re-snapshot the fixed preload and report exact link/target stability."""

    try:
        after = snapshot_gb10_preload()
    except ValueError as error:
        return {
            "before": dict(before),
            "after": None,
            "unchanged": False,
            "after_error": str(error),
        }
    return {
        "before": dict(before),
        "after": after,
        "unchanged": dict(before) == after,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_has_live_members(process_group: int) -> bool:
    """Report non-zombie Linux processes still in one process group."""

    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                stat_text = (entry / "stat").read_text(encoding="utf-8")
                closing = stat_text.rfind(")")
                fields = stat_text[closing + 2 :].split()
                state = fields[0]
                group = int(fields[2])
            except (OSError, ValueError, IndexError):
                continue
            if group == process_group and state != "Z":
                return True
        return False
    return _process_group_exists(process_group)


def run_bounded_command(
    command: list[str],
    log_path: str | os.PathLike[str],
    *,
    timeout_seconds: int | float,
    environment: Mapping[str, str],
    cwd: str | os.PathLike[str] | None = None,
    no_output_timeout_seconds: int | float | None = None,
    startup_readiness_timeout_seconds: int | float | None = None,
    startup_readiness_marker: str | None = None,
) -> dict[str, object]:
    """Run one list-form command with optional startup watchdogs and group teardown."""

    if not isinstance(command, list) or not command:
        raise TypeError("command must be a nonempty list of strings")
    if not all(isinstance(part, str) and part for part in command):
        raise TypeError("command must be a nonempty list of strings")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ValueError("command timeout must be positive")
    if not isinstance(environment, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise TypeError("command environment must map strings to strings")
    for name, value in (
        ("no-output watchdog", no_output_timeout_seconds),
        ("startup-readiness watchdog", startup_readiness_timeout_seconds),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError(f"{name} timeout must be positive")
    if startup_readiness_timeout_seconds is not None and (
        not isinstance(startup_readiness_marker, str)
        or not startup_readiness_marker
    ):
        raise ValueError("startup-readiness marker must be a nonempty string")
    if startup_readiness_marker is not None and not isinstance(
        startup_readiness_marker, str
    ):
        raise TypeError("startup-readiness marker must be a string")
    if (
        no_output_timeout_seconds is not None
        and startup_readiness_timeout_seconds is not None
        and no_output_timeout_seconds >= startup_readiness_timeout_seconds
    ):
        raise ValueError("no-output watchdog must precede startup-readiness watchdog")
    working_directory: str | None = None
    if cwd is not None:
        candidate = Path(cwd).resolve(strict=True)
        if not candidate.is_dir():
            raise ValueError("command working directory must be a directory")
        working_directory = str(candidate)

    requested_log = Path(log_path)
    parent = requested_log.parent if requested_log.parent != Path("") else Path(".")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    started_utc = _utc_now()
    started_monotonic = time.monotonic()
    timed_out = False
    watchdog_trigger: str | None = None
    spawn_error: str | None = None
    returncode: int | None = None
    termination_escalation = "none"
    process_group_cleanup_required = False
    process_group_quiescent = True
    log_identity: tuple[int, int]
    log_output_observed = False
    startup_readiness_observed = False
    marker_bytes = (
        startup_readiness_marker.encode("utf-8")
        if startup_readiness_marker is not None
        else None
    )
    digest = None
    interrupted: str | None = None
    read_fd: int | None = None
    try:
        log_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            log_fd = os.open(requested_log.name, log_flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            raise
        os.fchmod(log_fd, 0o600)
        opened_log = os.fstat(log_fd)
        log_identity = (opened_log.st_dev, opened_log.st_ino)
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        read_fd = os.open(requested_log.name, read_flags, dir_fd=parent_fd)
        with os.fdopen(log_fd, "w", encoding="utf-8") as log:
            scan_offset = 0
            scan_tail = b""

            def observe_log() -> None:
                nonlocal scan_offset, scan_tail
                nonlocal log_output_observed, startup_readiness_observed
                size = os.fstat(read_fd).st_size
                if size > 0:
                    log_output_observed = True
                while scan_offset < size:
                    chunk = os.pread(read_fd, min(65536, size - scan_offset), scan_offset)
                    if not chunk:
                        break
                    scan_offset += len(chunk)
                    if marker_bytes is not None and not startup_readiness_observed:
                        combined = scan_tail + chunk
                        if marker_bytes in combined:
                            startup_readiness_observed = True
                        keep = max(0, len(marker_bytes) - 1)
                        scan_tail = combined[-keep:] if keep else b""

            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(environment),
                    start_new_session=True,
                    text=True,
                    cwd=working_directory,
                )
            except OSError as error:
                spawn_error = f"{type(error).__name__}: {error}"
                log.write(spawn_error + "\n")
                log.flush()
            else:
                try:
                    while True:
                        observe_log()
                        returncode = process.poll()
                        if returncode is not None:
                            break
                        elapsed = time.monotonic() - started_monotonic
                        if (
                            no_output_timeout_seconds is not None
                            and elapsed >= no_output_timeout_seconds
                            and not log_output_observed
                        ):
                            watchdog_trigger = "no_log_output"
                            break
                        if (
                            startup_readiness_timeout_seconds is not None
                            and elapsed >= startup_readiness_timeout_seconds
                            and not startup_readiness_observed
                        ):
                            watchdog_trigger = "startup_readiness_missing"
                            break
                        if elapsed >= timeout_seconds:
                            timed_out = True
                            break
                        deadlines = [float(timeout_seconds)]
                        if (
                            no_output_timeout_seconds is not None
                            and not log_output_observed
                        ):
                            deadlines.append(float(no_output_timeout_seconds))
                        if (
                            startup_readiness_timeout_seconds is not None
                            and not startup_readiness_observed
                        ):
                            deadlines.append(float(startup_readiness_timeout_seconds))
                        wait_for = max(0.001, min(0.05, min(deadlines) - elapsed))
                        try:
                            returncode = process.wait(timeout=wait_for)
                        except subprocess.TimeoutExpired:
                            continue
                        break
                except KeyboardInterrupt:
                    interrupted = "KeyboardInterrupt"
                finally:
                    observe_log()
                    process_group_cleanup_required = (
                        interrupted is not None
                        or timed_out
                        or watchdog_trigger is not None
                        or _process_group_has_live_members(process.pid)
                    )
                    if process_group_cleanup_required:
                        termination_escalation = "sigterm"
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        deadline = time.monotonic() + min(
                            float(TERMINATION_GRACE_SECONDS), 10.0
                        )
                        while (
                            _process_group_has_live_members(process.pid)
                            and time.monotonic() < deadline
                        ):
                            process.poll()
                            time.sleep(0.02)
                        if _process_group_has_live_members(process.pid):
                            termination_escalation = "sigkill"
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            kill_deadline = time.monotonic() + 1.0
                            while (
                                _process_group_has_live_members(process.pid)
                                and time.monotonic() < kill_deadline
                            ):
                                process.poll()
                                time.sleep(0.02)
                    process_group_quiescent = not _process_group_has_live_members(
                        process.pid
                    )
                    if process.poll() is None:
                        try:
                            returncode = process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            returncode = None
                            process_group_quiescent = False
                    else:
                        returncode = process.returncode
        try:
            metadata = os.fstat(read_fd)
            if not stat.S_ISREG(metadata.st_mode) or (
                metadata.st_dev,
                metadata.st_ino,
            ) != log_identity:
                raise RuntimeError("render log changed identity before hashing")
            digest = hashlib.sha256() if process_group_quiescent else None
            if digest is not None:
                os.lseek(read_fd, 0, os.SEEK_SET)
                while True:
                    chunk = os.read(read_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        finally:
            os.close(read_fd)
            read_fd = None
    finally:
        if read_fd is not None:
            os.close(read_fd)
        os.close(parent_fd)

    result = {
        "command": list(command),
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "returncode": returncode,
        "timed_out": timed_out,
        "spawn_error": spawn_error,
        "termination_escalation": termination_escalation,
        "process_group_cleanup_required": process_group_cleanup_required,
        "process_group_quiescent": process_group_quiescent,
        "log_path": str(requested_log),
        "log_sha256": digest.hexdigest() if digest is not None else None,
        "log_hash_error": (
            None if digest is not None else "process_group_not_quiescent"
        ),
    }
    if working_directory is not None or startup_readiness_marker is not None:
        result.update(
            {
                "watchdog_trigger": watchdog_trigger,
                "interrupted": interrupted,
                "log_output_observed": log_output_observed,
                "startup_readiness_marker": startup_readiness_marker,
                "startup_readiness_observed": (
                    startup_readiness_observed
                    if startup_readiness_marker is not None
                    else None
                ),
                "cwd": working_directory,
            }
        )
    if interrupted is not None and startup_readiness_timeout_seconds is None:
        raise KeyboardInterrupt
    return result


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def hash_regular_file(path: str | os.PathLike[str]) -> dict[str, object]:
    """Hash one regular non-symlink file and reject changes during the read."""

    candidate = Path(path)
    try:
        lexical = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"hash input is unavailable: {candidate}") from error
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ValueError(f"hash input must be a regular non-symlink file: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(f"could not open hash input safely: {candidate}") from error
    digest = hashlib.sha256()
    count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"hash input must be a regular file: {candidate}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or count != after.st_size:
        raise ValueError(f"hash input changed while hashing: {candidate}")
    return {"sha256": digest.hexdigest(), "bytes": count}


def snapshot_local_isaac62_install_metadata() -> dict[str, object]:
    """Bind the fixed local distribution metadata without importing Isaac Sim."""

    candidate = Path(LOCAL_ISAAC62_METADATA)
    snapshot = hash_regular_file(candidate)
    if snapshot["bytes"] > 1024 * 1024:
        raise ValueError("local Isaac 6.2 distribution metadata is unexpectedly large")
    try:
        text = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("local Isaac 6.2 distribution metadata is unreadable") from error
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            break
        if ": " in line:
            key, value = line.split(": ", 1)
            headers.setdefault(key, value)
    if headers.get("Name") != "isaacsim" or headers.get("Version") != "6.2.0.0":
        raise ValueError("local Isaac 6.2 distribution metadata version drifted")
    after = hash_regular_file(candidate)
    if after != snapshot:
        raise ValueError("local Isaac 6.2 distribution metadata changed while reading")
    return {
        "path": str(candidate.resolve(strict=True)),
        **snapshot,
        "distribution_name": headers["Name"],
        "distribution_version": headers["Version"],
    }


def hash_regular_tree(
    root: str | os.PathLike[str],
) -> dict[str, dict[str, object]]:
    """Hash all files in a symlink-free tree with stable sorted relative paths."""

    candidate = Path(root)
    try:
        root_lstat = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"input tree is unavailable: {candidate}") from error
    if stat.S_ISLNK(root_lstat.st_mode):
        raise ValueError(f"input tree root must not be a symlink: {candidate}")
    if not stat.S_ISDIR(root_lstat.st_mode):
        raise ValueError(f"input tree root must be a directory: {candidate}")
    base = candidate.resolve(strict=True)
    records: list[tuple[str, dict[str, object]]] = []

    def visit(directory: Path) -> None:
        before = os.stat(directory, follow_symlinks=False)
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"input tree directory is unreadable: {directory}") from error
        initial_names = tuple(entry.name for entry in entries)
        for entry in entries:
            path = directory / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"input tree entry is unreadable: {path}") from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"input tree contains a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                records.append(
                    (path.relative_to(base).as_posix(), hash_regular_file(path))
                )
            else:
                raise ValueError(f"input tree contains a non-regular entry: {path}")
        try:
            final_names = tuple(
                sorted(entry.name for entry in os.scandir(directory))
            )
            after = os.stat(directory, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"input tree directory drifted: {directory}") from error
        if initial_names != final_names or _stat_identity(before) != _stat_identity(after):
            raise ValueError(f"input tree directory changed while hashing: {directory}")

    visit(base)
    return {relative: record for relative, record in sorted(records)}


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _reject_symlink_path_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise ValueError(f"directory path is unavailable: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"directory path contains a symlink: {current}")


def create_attempt_root(
    parent: str | os.PathLike[str], run_id: str
) -> Path:
    """Atomically create one private attempt root beneath an existing real parent."""

    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run ID must be one safe path component")
    parent_path = Path(parent).expanduser()
    _reject_symlink_path_components(parent_path)
    resolved_parent = parent_path.resolve(strict=True)
    if not resolved_parent.is_dir():
        raise ValueError(f"attempt parent must be a directory: {resolved_parent}")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(resolved_parent, flags)
    try:
        os.mkdir(run_id, 0o700, dir_fd=parent_fd)
        child_fd = os.open(run_id, flags, dir_fd=parent_fd)
        try:
            os.fchmod(child_fd, 0o700)
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)
    return resolved_parent / run_id


def _open_owned_private_directory(path: Path) -> int:
    _reject_symlink_path_components(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        os.close(descriptor)
        raise ValueError("JSON destination must be an owned mode-0700 directory")
    return descriptor


def atomic_write_json(
    directory: str | os.PathLike[str],
    file_name: str,
    payload: object,
) -> Path:
    """Publish complete JSON inside an owned attempt directory without clobbering."""

    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
        or file_name in {".", ".."}
    ):
        raise ValueError("JSON file name must be one path component")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    directory_candidate = Path(directory)
    _reject_symlink_path_components(directory_candidate)
    root = directory_candidate.resolve(strict=True)
    directory_fd = _open_owned_private_directory(root)
    temporary = f".{file_name}.tmp-{secrets.token_hex(8)}"
    temporary_created = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        temporary_created = True
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temporary,
            file_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    return root / file_name


def _validate_process_attempt_policy(
    policy: Mapping[str, object],
) -> dict[str, object]:
    expected_fields = set(LOCAL_ISAAC62_ATTEMPT_POLICY)
    if not isinstance(policy, Mapping) or set(policy) != expected_fields:
        raise ValueError("process attempt policy fields are malformed")
    max_attempts = policy["max_process_attempts"]
    if type(max_attempts) is not int or max_attempts != 2:
        raise ValueError("process attempt policy must allow exactly two attempts")
    for field in (
        "no_output_timeout_seconds",
        "startup_readiness_timeout_seconds",
        "fast_crash_retry_seconds",
    ):
        value = policy[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError(f"process attempt policy {field} must be positive")
    if policy["no_output_timeout_seconds"] >= policy[
        "startup_readiness_timeout_seconds"
    ]:
        raise ValueError("no-output watchdog must precede readiness watchdog")
    if policy["fast_crash_retry_seconds"] > policy[
        "startup_readiness_timeout_seconds"
    ]:
        raise ValueError("fast-crash window cannot exceed readiness watchdog")
    marker = policy["startup_readiness_marker"]
    if not isinstance(marker, str) or not marker:
        raise ValueError("process attempt policy readiness marker must be nonempty")
    return dict(policy)


def run_retriable_render_command(
    command: list[str],
    logical_root: str | os.PathLike[str],
    *,
    timeout_seconds: int | float,
    environment: Mapping[str, str],
    attempt_policy: Mapping[str, object],
) -> dict[str, object]:
    """Run one logical local render with at most one recorded process relaunch.

    The renderer argv is byte-for-byte identical between attempts.  Its output
    argument must be the relative path ``capture``; distinct private working
    directories isolate and preserve every attempt without changing argv.
    """

    policy = _validate_process_attempt_policy(attempt_policy)
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise TypeError("command must be a nonempty list of strings")
    if command.count("--out") != 1:
        raise ValueError("retriable render command must contain one --out argument")
    output_index = command.index("--out") + 1
    if output_index >= len(command) or command[output_index] != "capture":
        raise ValueError("retriable render command --out must be relative capture")
    if not isinstance(environment, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise TypeError("command environment must map strings to strings")

    root = Path(logical_root).resolve(strict=True)
    descriptor = _open_owned_private_directory(root)
    os.close(descriptor)
    environment_sha256 = hashlib.sha256(
        json.dumps(
            dict(environment), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    attempts: list[dict[str, object]] = []
    selected_attempt_number: int | None = None

    for attempt_number in range(1, int(policy["max_process_attempts"]) + 1):
        attempt_root = create_attempt_root(
            root, f"process-attempt-{attempt_number:02d}"
        )
        process = run_bounded_command(
            list(command),
            attempt_root / "render.log",
            timeout_seconds=timeout_seconds,
            environment=environment,
            cwd=attempt_root,
            no_output_timeout_seconds=policy["no_output_timeout_seconds"],
            startup_readiness_timeout_seconds=policy[
                "startup_readiness_timeout_seconds"
            ],
            startup_readiness_marker=str(policy["startup_readiness_marker"]),
        )
        succeeded = (
            process["spawn_error"] is None
            and process["interrupted"] is None
            and process["returncode"] == 0
            and process["timed_out"] is False
            and process["watchdog_trigger"] is None
            and process["startup_readiness_observed"] is True
            and process["process_group_cleanup_required"] is False
            and process["process_group_quiescent"] is True
        )
        if succeeded:
            failure_classification = None
        elif process["interrupted"] is not None:
            failure_classification = "operator_interrupted"
        elif process["watchdog_trigger"] == "no_log_output":
            failure_classification = "no_log_output_180s"
        elif process["watchdog_trigger"] == "startup_readiness_missing":
            failure_classification = "startup_readiness_missing_420s"
        elif process["spawn_error"] is not None:
            failure_classification = "spawn_error"
        elif process["timed_out"] is True:
            failure_classification = "overall_timeout"
        elif process["process_group_quiescent"] is not True:
            failure_classification = "process_group_not_quiescent"
        elif (
            isinstance(process["returncode"], int)
            and process["returncode"] != 0
            and process["elapsed_seconds"] <= policy["fast_crash_retry_seconds"]
        ):
            failure_classification = "fast_nonzero_exit"
        elif process["returncode"] != 0:
            failure_classification = "nonzero_exit"
        elif process["startup_readiness_observed"] is not True:
            failure_classification = "startup_readiness_missing_on_exit"
        elif process["process_group_cleanup_required"] is True:
            failure_classification = "process_group_cleanup_required"
        else:
            failure_classification = "unclassified_process_failure"

        retryable = failure_classification in {
            "no_log_output_180s",
            "startup_readiness_missing_420s",
            "fast_nonzero_exit",
        }
        retry_trigger = (
            failure_classification
            if attempt_number < int(policy["max_process_attempts"])
            and retryable
            and process["process_group_quiescent"] is True
            else None
        )
        record = {
            "schema_version": 1,
            "attempt_number": attempt_number,
            "attempt_root": str(attempt_root),
            "capture_root": str(attempt_root / "capture"),
            "command": list(command),
            "environment_sha256": environment_sha256,
            "outcome": "succeeded" if succeeded else "failed",
            "failure_classification": failure_classification,
            "retry_trigger": retry_trigger,
            "process": process,
        }
        atomic_write_json(attempt_root, "process-attempt.json", record)
        attempts.append(record)
        if succeeded:
            selected_attempt_number = attempt_number
            break
        if retry_trigger is None:
            break

    result = {
        "schema_version": 1,
        "status": (
            "process_succeeded"
            if selected_attempt_number is not None
            else "failed"
        ),
        "max_process_attempts": policy["max_process_attempts"],
        "attempt_count": len(attempts),
        "selected_attempt_number": selected_attempt_number,
        "command": list(command),
        "environment_sha256": environment_sha256,
        "attempt_policy": policy,
        "attempts": attempts,
    }
    atomic_write_json(root, "process-attempts.json", result)
    return result


REQUESTED_CAPTURE_SETTING_KEYS = frozenset(
    {
        "camera",
        "stage",
        "frames",
        "subframes",
        "warmup",
        "max_tries",
        "seed",
        "require_liquid_pixels",
        "renderer",
    }
)
CAPTURE_ARTIFACT_KINDS = frozenset({"rgb", "semantic", "instance", "liquid"})


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_strict_json_snapshot(
    path: str | os.PathLike[str], *, max_bytes: int = 16 * 1024 * 1024
) -> tuple[object, dict[str, object]]:
    """Read one bounded regular JSON file once, rejecting duplicates and drift."""

    candidate = Path(path)
    try:
        lexical = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"JSON input is unavailable: {candidate}") from error
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ValueError(f"JSON input must be a regular non-symlink file: {candidate}")
    if lexical.st_size > max_bytes:
        raise ValueError(f"JSON input exceeds {max_bytes} bytes: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    chunks: list[bytes] = []
    count = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > max_bytes:
                raise ValueError(f"JSON input exceeds {max_bytes} bytes: {candidate}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or count != after.st_size:
        raise ValueError(f"JSON input changed while reading: {candidate}")
    raw = b"".join(chunks)
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON input is invalid: {candidate}") from error
    return payload, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _parse_frame_spec(value: object) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError("capture frames requested setting must be a string")
    if not value or (":" in value and "," in value):
        raise ValueError("capture frames requested setting has invalid syntax")
    if ":" in value:
        pieces = value.split(":")
        if len(pieces) != 3:
            raise ValueError(
                "capture frames requested setting must use start:stop:step"
            )
        try:
            start, stop, step = (int(piece) for piece in pieces)
        except ValueError as error:
            raise ValueError(
                "capture frames requested setting contains a non-integer"
            ) from error
        if stop <= start or step <= 0:
            raise ValueError("capture frames requested setting has invalid bounds")
        result = tuple(range(start, stop, step))
    else:
        pieces = value.split(",")
        if any(not piece for piece in pieces):
            raise ValueError("capture frames requested list contains an empty item")
        try:
            result = tuple(int(piece) for piece in pieces)
        except ValueError as error:
            raise ValueError(
                "capture frames requested list contains a non-integer"
            ) from error
        if any(left >= right for left, right in zip(result, result[1:])):
            raise ValueError(
                "capture frames requested list must be unique and strictly increasing"
            )
    if not result:
        raise ValueError("capture frames requested setting selects no frames")
    if any(code < 1 or code > 686 for code in result):
        raise ValueError("capture frames requested setting leaves the stage range")
    return result


def parse_frame_spec(value: object) -> tuple[int, ...]:
    """Parse one validated stage time-code selection for external bindings."""

    return _parse_frame_spec(value)


def _validate_requested_capture_settings(settings: Mapping[str, object]) -> None:
    if set(settings) != REQUESTED_CAPTURE_SETTING_KEYS:
        raise ValueError("expected requested settings have the wrong fields")
    if settings["camera"] != "FillCam":
        raise ValueError("capture requested setting camera must be FillCam")
    if not isinstance(settings["stage"], str) or not settings["stage"]:
        raise ValueError("capture requested setting stage must be a nonempty string")
    _parse_frame_spec(settings["frames"])
    for key in ("subframes", "warmup", "max_tries"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"capture requested setting {key} must be a positive integer")
    seed = settings["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("capture requested setting seed must be a nonnegative integer")
    if not isinstance(settings["require_liquid_pixels"], bool):
        raise ValueError(
            "capture requested setting require_liquid_pixels must be Boolean"
        )
    if settings["renderer"] not in {"PathTracing", "RaytracedLighting"}:
        raise ValueError("capture requested setting renderer is unsupported")


def validate_requested_capture_settings(settings: Mapping[str, object]) -> None:
    """Validate the complete externally controlled Task 2A settings mapping."""

    _validate_requested_capture_settings(settings)


def capture_requested_settings(summary: Mapping[str, object]) -> dict[str, object]:
    """Extract the externally requested controls from a Task 2A summary."""

    try:
        inputs = summary["inputs"]
        render = summary["render_settings"]
        assert isinstance(inputs, Mapping)
        assert isinstance(render, Mapping)
        stage = inputs["stage"]
        camera = inputs["camera"]
        assert isinstance(stage, Mapping)
        assert isinstance(camera, Mapping)
        settings = {
            "camera": camera["name"],
            "stage": stage["requested"],
            "frames": render["frame_spec"],
            "subframes": render["rt_subframes"],
            "warmup": render["warmup_updates"],
            "max_tries": render["max_tries_per_frame"],
            "seed": render["replicator_global_seed"],
            "require_liquid_pixels": render["require_liquid_pixels"],
            "renderer": render["renderer"],
        }
    except (AssertionError, KeyError, TypeError) as error:
        raise ValueError("capture summary requested settings are malformed") from error
    _validate_requested_capture_settings(settings)
    return settings


def verify_capture_summary(
    capture_root: str | os.PathLike[str],
    expected_settings: Mapping[str, object] | None,
) -> dict[str, object]:
    """Verify a complete Task 2A schema-1 capture and every referenced artifact."""

    root_candidate = Path(capture_root)
    tree = hash_regular_tree(root_candidate)
    if "summary.json" not in tree:
        raise ValueError("capture summary.json is missing")
    root = root_candidate.resolve(strict=True)
    payload, summary_snapshot = load_strict_json_snapshot(root / "summary.json")
    if tree["summary.json"] != summary_snapshot:
        raise ValueError("capture summary changed while it was verified")
    if not isinstance(payload, Mapping):
        raise ValueError("capture summary must be a JSON object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("capture summary schema version must be exactly 1")
    if payload.get("mechanical_gates_passed") is not True:
        raise ValueError("capture summary mechanical gates did not pass")
    if payload.get("visual_check_status") != "pending":
        raise ValueError("capture summary visual check status must remain pending")

    actual_settings = capture_requested_settings(payload)
    if expected_settings is not None:
        if not isinstance(expected_settings, Mapping):
            raise TypeError("expected requested settings must be a mapping")
        _validate_requested_capture_settings(expected_settings)
        if dict(expected_settings) != actual_settings:
            raise ValueError("capture requested setting mismatch")
    expected_time_codes = list(_parse_frame_spec(actual_settings["frames"]))
    render_settings = payload.get("render_settings")
    if not isinstance(render_settings, Mapping):
        raise ValueError("capture render settings are malformed")
    resolution = render_settings.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolution)
    ):
        raise ValueError("capture raster resolution is malformed")
    if render_settings.get("time_codes") != expected_time_codes:
        raise ValueError("capture time codes do not match the requested frame setting")

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("capture frames must be a nonempty list")
    if payload.get("frame_count") != len(frames) or len(frames) != len(expected_time_codes):
        raise ValueError("capture frame denominator is inconsistent")

    referenced: dict[str, dict[str, object]] = {}
    for ordinal, (frame, expected_time_code) in enumerate(
        zip(frames, expected_time_codes, strict=True), start=1
    ):
        if not isinstance(frame, Mapping):
            raise ValueError("capture frame record is malformed")
        if frame.get("capture_ordinal") != ordinal or frame.get("time_code") != expected_time_code:
            raise ValueError("capture frame order or time code is inconsistent")
        artifacts = frame.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != CAPTURE_ARTIFACT_KINDS:
            raise ValueError("capture frame artifact kinds are incomplete or extra")
        for kind in sorted(CAPTURE_ARTIFACT_KINDS):
            record = artifacts[kind]
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
                raise ValueError("capture artifact record is malformed")
            relative_text = record["path"]
            expected_hash = record["sha256"]
            if not isinstance(relative_text, str) or "\\" in relative_text:
                raise ValueError("capture artifact path must be a relative POSIX path")
            relative = PurePosixPath(relative_text)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative.parts[0] != kind
            ):
                raise ValueError("capture artifact path is relative-path escape or malformed")
            if (
                not isinstance(expected_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            ):
                raise ValueError("capture artifact hash is malformed")
            if relative_text in referenced:
                raise ValueError("capture artifact path is referenced more than once")
            referenced[relative_text] = dict(record)

    actual_artifacts = {key: value for key, value in tree.items() if key != "summary.json"}
    if set(actual_artifacts) != set(referenced):
        raise ValueError("capture artifact tree has missing or extra files")
    for relative, expected in referenced.items():
        if actual_artifacts[relative]["sha256"] != expected["sha256"]:
            raise ValueError(f"capture artifact hash drifted: {relative}")
    if hash_regular_tree(root) != tree:
        raise ValueError("capture tree changed during verification")
    return {
        "summary": dict(payload),
        "summary_path": str(root / "summary.json"),
        "summary_sha256": summary_snapshot["sha256"],
        "artifacts": actual_artifacts,
    }


EXPERIMENT_REQUIRED_FIELDS = frozenset(
    {
        "hypothesis",
        "falsifier",
        "command_plan",
        "seed",
        "denominators",
        "stop_conditions",
    }
)


def load_experiment_manifest(
    path: str | os.PathLike[str], run_id: str
) -> dict[str, object]:
    """Load and bind one strict schema-1 experiment preregistration snapshot."""

    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run ID must be one safe path component")
    candidate = Path(path).expanduser()
    payload, snapshot = load_strict_json_snapshot(candidate, max_bytes=1024 * 1024)
    if not isinstance(payload, Mapping):
        raise ValueError("experiment manifest must be a JSON object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("experiment manifest schema_version must be exactly 1")
    if payload.get("run_id") != run_id:
        raise ValueError("experiment manifest run_id does not match --run-id")
    missing = EXPERIMENT_REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(
            f"experiment manifest is missing required fields: {sorted(missing)!r}"
        )
    for field in ("hypothesis", "falsifier"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(
                f"experiment manifest field {field!r} must be a nonblank string"
            )
    command_plan = payload["command_plan"]
    if not isinstance(command_plan, dict) or not command_plan:
        raise ValueError(
            "experiment manifest field 'command_plan' must be a nonempty object"
        )
    if any(not isinstance(key, str) or not key.strip() for key in command_plan):
        raise ValueError(
            "experiment manifest field 'command_plan' must use nonblank string keys"
        )
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "experiment manifest field 'seed' must be a nonnegative integer"
        )
    denominators = payload["denominators"]
    if not isinstance(denominators, dict) or not denominators:
        raise ValueError(
            "experiment manifest field 'denominators' must be a nonempty object"
        )
    if any(
        not isinstance(key, str)
        or not key.strip()
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for key, value in denominators.items()
    ):
        raise ValueError(
            "experiment manifest field 'denominators' must map nonblank names "
            "to positive integers"
        )
    stop_conditions = payload["stop_conditions"]
    if (
        not isinstance(stop_conditions, list)
        or not stop_conditions
        or any(
            not isinstance(condition, str) or not condition.strip()
            for condition in stop_conditions
        )
    ):
        raise ValueError(
            "experiment manifest field 'stop_conditions' must be a nonempty "
            "list of nonblank strings"
        )
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": snapshot["sha256"],
        "bytes": snapshot["bytes"],
        "payload": dict(payload),
    }

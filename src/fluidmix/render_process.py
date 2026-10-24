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
CHILD_ENVIRONMENT_PROFILES = ("generic", "gb10-aarch64")
GB10_PRELOAD = Path("/lib/aarch64-linux-gnu/libgomp.so.1")
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
    if profile == "gb10-aarch64":
        try:
            metadata = os.lstat(GB10_PRELOAD)
        except OSError as error:
            raise ValueError(
                f"reviewed GB10 preload is not a regular non-symlink file: {GB10_PRELOAD}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"reviewed GB10 preload is not a regular non-symlink file: {GB10_PRELOAD}"
            )
        child["LD_PRELOAD"] = str(GB10_PRELOAD)
    return child


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


def run_bounded_command(
    command: list[str],
    log_path: str | os.PathLike[str],
    *,
    timeout_seconds: int | float,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Run one list-form command in a new session with bounded group teardown."""

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

    requested_log = Path(log_path)
    parent = requested_log.parent if requested_log.parent != Path("") else Path(".")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    started_utc = _utc_now()
    started_monotonic = time.monotonic()
    timed_out = False
    spawn_error: str | None = None
    returncode: int | None = None
    termination_escalation = "none"
    log_identity: tuple[int, int]
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
        with os.fdopen(log_fd, "w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(environment),
                    start_new_session=True,
                    text=True,
                )
            except OSError as error:
                spawn_error = f"{type(error).__name__}: {error}"
                log.write(spawn_error + "\n")
                log.flush()
            else:
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    termination_escalation = "sigterm"
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + min(
                        float(TERMINATION_GRACE_SECONDS), 10.0
                    )
                    while _process_group_exists(process.pid) and time.monotonic() < deadline:
                        process.poll()
                        time.sleep(0.02)
                    if _process_group_exists(process.pid):
                        termination_escalation = "sigkill"
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    returncode = process.wait()
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        read_fd = os.open(requested_log.name, read_flags, dir_fd=parent_fd)
        try:
            metadata = os.fstat(read_fd)
            if not stat.S_ISREG(metadata.st_mode) or (
                metadata.st_dev,
                metadata.st_ino,
            ) != log_identity:
                raise RuntimeError("render log changed identity before hashing")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(read_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(read_fd)
    finally:
        os.close(parent_fd)

    return {
        "command": list(command),
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "returncode": returncode,
        "timed_out": timed_out,
        "spawn_error": spawn_error,
        "termination_escalation": termination_escalation,
        "log_path": str(requested_log),
        "log_sha256": digest.hexdigest(),
    }


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
    pieces = value.split(":")
    if len(pieces) != 3:
        raise ValueError("capture frames requested setting must use start:stop:step")
    try:
        start, stop, step = (int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError("capture frames requested setting contains a non-integer") from error
    if start < 0 or stop <= start or step <= 0:
        raise ValueError("capture frames requested setting has invalid bounds")
    result = tuple(range(start, stop, step))
    if not result:
        raise ValueError("capture frames requested setting selects no frames")
    return result


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


def _nonempty_experiment_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return value is not None


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
    for field in EXPERIMENT_REQUIRED_FIELDS:
        if not _nonempty_experiment_value(payload[field]):
            raise ValueError(f"experiment manifest field {field!r} must be nonempty")
    for field in ("hypothesis", "falsifier"):
        if not isinstance(payload[field], str):
            raise ValueError(f"experiment manifest field {field!r} must be a string")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("experiment manifest seed must be a nonnegative integer")
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": snapshot["sha256"],
        "bytes": snapshot["bytes"],
        "payload": dict(payload),
    }

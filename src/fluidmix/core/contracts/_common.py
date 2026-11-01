"""Shared implementation helpers; never import Part A or Part B application modules."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import sys
import unicodedata

from jsonschema import Draft202012Validator
from ..io import open_descriptor


_ROOT = Path(__file__).resolve().parent
_READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _open_directory_no_follow(path: str | Path, role: str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        directory_fd = open_descriptor(
            "/", _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError as error:
        raise ValueError(f"{role} root is unavailable") from error
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = open_descriptor(
                    component,
                    _READ_FLAGS | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(f"{role} directory is unavailable or unsafe") from error
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def load_schema(filename: str) -> dict[str, object]:
    value = json.loads((_ROOT / filename).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return value


def validate_schema(value: object, schema: dict[str, object], role: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ValueError(f"{role} schema violation at {location}: {error.message}")


def reject_nonfinite(value: object, role: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{role} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, f"{role}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{role}[{index}]")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def contained_relative_path(value: object, role: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError(f"{role} must be a contained relative POSIX path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{role} must be a contained relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{role} must be a contained relative POSIX path")
    return path


def bounded_control_free_text(value: object, role: str, *, max_length: int = 200) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_length
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError(f"{role} must be bounded nonempty control-free text")
    return value


def exact_lower_sha256(value: object, role: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{role} must be exactly 64 lowercase hexadecimal characters")
    return value


def safe_posix_component(value: object, role: str) -> str:
    bounded_control_free_text(value, role, max_length=128)
    assert isinstance(value, str)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None or value in {
        ".",
        "..",
    }:
        raise ValueError(f"{role} must be one safe POSIX component")
    return value


def relative_path_below_root(
    artifact_root: str | Path, absolute_path: object, role: str
) -> PurePosixPath:
    if not isinstance(absolute_path, str) or not Path(absolute_path).is_absolute():
        raise ValueError(f"{role} must be an absolute path")
    root = Path(os.path.abspath(os.fspath(artifact_root)))
    candidate = Path(os.path.abspath(absolute_path))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} must remain below artifact_root") from error
    return contained_relative_path(relative.as_posix(), role)


def stable_hash_relative_file(
    artifact_root: str | Path,
    relative_path: str | PurePosixPath,
    role: str,
    *,
    max_bytes: int = 8 * 1024 * 1024 * 1024,
) -> dict[str, object]:
    """Hash one regular file through no-follow directory descriptors."""

    relative = contained_relative_path(str(relative_path), role)
    directory_fd = _open_directory_no_follow(artifact_root, f"{role} artifact_root")
    file_fd = -1
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = open_descriptor(
                    component,
                    _READ_FLAGS | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(f"{role} parent is unavailable or unsafe") from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = open_descriptor(relative.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(f"{role} is unavailable or unsafe") from error
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{role} must be a regular non-symlink file")
        if before.st_size > max_bytes:
            raise ValueError(f"{role} exceeds the size bound")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(f"{role} exceeds the size bound")
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or observed != after.st_size:
            raise ValueError(f"{role} changed while reading")
        return {"sha256": digest.hexdigest(), "bytes": observed}
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def stable_read_relative_file(
    artifact_root: str | Path,
    relative_path: str | PurePosixPath,
    role: str,
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    """Read one bounded regular file through no-follow directory descriptors."""

    relative = contained_relative_path(str(relative_path), role)
    directory_fd = _open_directory_no_follow(artifact_root, f"{role} artifact_root")
    file_fd = -1
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = open_descriptor(
                    component,
                    _READ_FLAGS | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(f"{role} parent is unavailable or unsafe") from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = open_descriptor(relative.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(f"{role} is unavailable or unsafe") from error
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"{role} must be a bounded regular non-symlink file")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(f"{role} exceeds the size bound")
        after = os.fstat(file_fd)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or observed != after.st_size:
            raise ValueError(f"{role} changed while reading")
        return b"".join(chunks)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def stable_read_path(
    path: str | Path, role: str, *, max_bytes: int = 64 * 1024 * 1024
) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    sys.audit("fluidmix.stable_read_path", os.fspath(absolute))
    relative = PurePosixPath(*absolute.parts[1:])
    return stable_read_relative_file(
        "/", relative, role, max_bytes=max_bytes
    )


def strict_json_bytes(raw: bytes, role: str) -> object:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> object:
        raise ValueError(f"{role} contains non-finite value {value!r}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} must be strict UTF-8 JSON") from error


def publish_bytes_no_clobber(path: str | Path, raw: bytes, role: str) -> Path:
    destination = Path(path)
    parent = destination.parent
    if destination.name in {"", ".", ".."}:
        raise ValueError(f"{role} destination is invalid")
    directory_fd = _open_directory_no_follow(parent, f"{role} parent")
    descriptor = -1
    try:
        try:
            descriptor = open_descriptor(
                destination.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            try:
                existing_fd = open_descriptor(destination.name, _READ_FLAGS, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(f"{role} existing destination is unsafe") from error
            try:
                before = os.fstat(existing_fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > len(raw):
                    raise FileExistsError(
                        f"{role} already exists with different bytes"
                    )
                chunks: list[bytes] = []
                observed = 0
                while True:
                    chunk = os.read(existing_fd, min(1024 * 1024, len(raw) + 1 - observed))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    observed += len(chunk)
                    if observed > len(raw):
                        break
                after = os.fstat(existing_fd)
            finally:
                os.close(existing_fd)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or observed != after.st_size
            ):
                raise ValueError(f"{role} changed while reading")
            existing = b"".join(chunks)
            if existing != raw:
                raise FileExistsError(f"{role} already exists with different bytes")
            return destination
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError(f"could not write {role}")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(directory_fd)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)

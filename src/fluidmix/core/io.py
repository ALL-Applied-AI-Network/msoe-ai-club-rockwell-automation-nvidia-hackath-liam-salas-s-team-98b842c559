"""Strict local-file snapshots shared by product and evaluation code."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess


CommandProcessError = subprocess.CalledProcessError


def open_descriptor(path: object, flags: int, mode: int = 0o777, **kwargs: object) -> int:
    """Open one descriptor through the repository's explicit I/O boundary."""

    return os.open(path, flags, mode, **kwargs)


def fdopen_descriptor(descriptor: int, *args: object, **kwargs: object):
    """Wrap an owned descriptor in a file object."""

    return os.fdopen(descriptor, *args, **kwargs)


def open_path(path: str | os.PathLike[str], *args: object, **kwargs: object):
    """Open a path through the repository's explicit I/O boundary."""

    return Path(path).open(*args, **kwargs)


def open_image(source: object, *args: object, **kwargs: object):
    """Open a Pillow image through the repository's explicit I/O boundary."""

    from PIL import Image

    return Image.open(source, *args, **kwargs)


def run_subprocess(*args: object, **kwargs: object):
    """Run a child process through the repository's explicit I/O boundary."""

    return subprocess.run(*args, **kwargs)


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
    """Publish complete JSON inside an owned directory without clobbering."""

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

"""Twin-owned bounded, no-follow reads for synthetic source artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import unicodedata


_READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash path contents in chunks; follows links and does not detect concurrent drift."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_directory_no_follow(path: str | Path, role: str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        directory_fd = os.open(
            "/", _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError as error:
        raise ValueError(f"{role} root is unavailable") from error
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    _READ_FLAGS | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(
                    f"{role} directory is unavailable or unsafe"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _contained_relative_path(value: object, role: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(
            unicodedata.category(character).startswith("C") for character in value
        )
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


def _stable_read_relative_file(
    artifact_root: str | Path,
    relative_path: str | PurePosixPath,
    role: str,
    *,
    max_bytes: int,
) -> bytes:
    relative = _contained_relative_path(str(relative_path), role)
    directory_fd = _open_directory_no_follow(artifact_root, f"{role} artifact_root")
    file_fd = -1
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(
                    component,
                    _READ_FLAGS | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(f"{role} parent is unavailable or unsafe") from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
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
    # Preserve the established audit event so existing security hooks survive the move.
    sys.audit("fluidmix.stable_read_path", os.fspath(absolute))
    relative = PurePosixPath(*absolute.parts[1:])
    return _stable_read_relative_file("/", relative, role, max_bytes=max_bytes)


__all__ = ["sha256_file", "stable_read_path"]

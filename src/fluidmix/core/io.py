"""Strict local-file snapshots shared by product and evaluation code."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


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

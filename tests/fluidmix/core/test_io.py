from __future__ import annotations

import json
from pathlib import Path

import pytest

from fluidmix.core.io import hash_regular_file, load_strict_json_snapshot
from twin.render.render_process import (
    hash_regular_file as twin_hash_regular_file,
    load_strict_json_snapshot as twin_load_strict_json_snapshot,
)


def test_hash_regular_file_matches_frozen_twin_helper(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"onion-parity\x00\xff")

    assert hash_regular_file(source) == twin_hash_regular_file(source)


def test_strict_json_snapshot_matches_frozen_twin_helper(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"z": [1, 2], "a": {"value": True}}, separators=(",", ":")),
        encoding="utf-8",
    )

    assert load_strict_json_snapshot(source) == twin_load_strict_json_snapshot(source)


@pytest.mark.parametrize(
    ("contents", "max_bytes", "message"),
    [
        (b'{"duplicate":1,"duplicate":2}', 1024, "duplicate key"),
        (b"not-json", 1024, "JSON input is invalid"),
        (b'{"large":true}', 4, "JSON input exceeds 4 bytes"),
    ],
)
def test_strict_json_snapshot_rejects_the_same_invalid_inputs(
    tmp_path: Path, contents: bytes, max_bytes: int, message: str
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(contents)

    for loader in (load_strict_json_snapshot, twin_load_strict_json_snapshot):
        with pytest.raises(ValueError, match=message):
            loader(source, max_bytes=max_bytes)


def test_file_helpers_reject_symlinks_like_frozen_twin_helpers(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.symlink_to(target)

    for helper in (
        hash_regular_file,
        twin_hash_regular_file,
        load_strict_json_snapshot,
        twin_load_strict_json_snapshot,
    ):
        with pytest.raises(ValueError, match="regular non-symlink file"):
            helper(source)

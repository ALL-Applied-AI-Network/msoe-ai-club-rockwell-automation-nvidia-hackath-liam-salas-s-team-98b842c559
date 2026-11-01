#!/usr/bin/env python3
"""Build the CPU-only immutable stage Cosmos attempt-plan preflight."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.explanation.template.explanation_contract import ContractError  # noqa: E402
from twin.render.render_process import load_strict_json_snapshot  # noqa: E402
from evaluation.scoring.stage_cosmos_plan_artifact import (  # noqa: E402
    build_stage_attempt_plan_artifact,
    stage_attempt_plan_artifact_bytes,
    write_stage_attempt_plan_artifact,
)


PROMPT_PATH = ROOT / "data/cosmos_stage_prompt_v1.txt"
MAX_PROMPT_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _read_prompt() -> tuple[str, str]:
    try:
        lexical = os.lstat(PROMPT_PATH)
    except OSError as exc:
        raise ContractError("fixed stage prompt is unavailable") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise ContractError("fixed stage prompt must be a regular file")
    if lexical.st_size > MAX_PROMPT_BYTES:
        raise ContractError("fixed stage prompt exceeds bounded size")
    descriptor = os.open(PROMPT_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    count = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, min(64 * 1024, MAX_PROMPT_BYTES + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > MAX_PROMPT_BYTES:
                raise ContractError("fixed stage prompt exceeds bounded size")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or count != after.st_size:
        raise ContractError("fixed stage prompt changed while reading")
    raw = b"".join(chunks)
    try:
        prompt = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("fixed stage prompt is not UTF-8") from exc
    return prompt, hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    measured, snapshot = load_strict_json_snapshot(args.measured_evidence)
    prompt, prompt_sha = _read_prompt()
    artifact = build_stage_attempt_plan_artifact(
        measured,
        prompt,
        measured_evidence_sha256=str(snapshot["sha256"]),
        prompt_template_sha256=prompt_sha,
    )
    output = write_stage_attempt_plan_artifact(args.out, artifact)
    output_sha = hashlib.sha256(stage_attempt_plan_artifact_bytes(artifact)).hexdigest()
    print(
        f"wrote immutable stage attempt plan: {output}; "
        f"sha256={output_sha}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

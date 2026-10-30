#!/usr/bin/env python3
"""Fail closed on forbidden production reads and bind the candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path("/home/hackathon5/liamreporoot")
BASE_READS = {
    ROOT / "outputs/pipeline-20261028/wp5-predicted-store-production-01a18128/evidence.sqlite3",
    ROOT / "outputs/pipeline-20261028/wp5-predicted-store-production-01a18128/population-receipt.json",
    ROOT / (
        "outputs/pipeline-20261028/"
        "wp5-predicted-mask-materialization-production-7e43406-01a18128/"
        "prediction-locators.json"
    ),
}
FORBIDDEN = (
    "/oracle", "/semantic", "/labels/", "/liquid/", "video-5", "video5",
    "video_5", "/evaluation/", "/held-out", "/heldout", "/test/",
)
_OPENAT_PATH = re.compile(r'openat\([^,]+, ("(?:\\.|[^"\\])*")')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _selected_sources(manifest: dict[str, object]) -> set[Path]:
    result: set[Path] = set()
    for vial in manifest["vials"]:
        result.update(
            Path(str(vial[key]))
            for key in (
                "source_receipt_path", "validation_receipt_path", "prefill_rgb_path",
                "fill_clip_path",
            )
        )
        result.update(Path(str(stage["rgb_path"])) for stage in vial["stages"])
    return result


def _observed_reads(trace: Path) -> set[Path]:
    result: set[Path] = set()
    for line in trace.read_text(errors="strict").splitlines():
        if "O_RDONLY" not in line or " = -1 " in line:
            continue
        match = _OPENAT_PATH.search(line)
        if match is None:
            continue
        value = json.loads(match.group(1))
        path = Path(value)
        if path.is_absolute():
            result.add(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--python-prefix", required=True)
    args = parser.parse_args(argv)

    trace = Path(args.trace).resolve(strict=True)
    output = Path(args.output_dir).resolve(strict=True)
    worktree = Path(args.worktree).resolve(strict=True)
    python_prefix = Path(args.python_prefix).resolve(strict=True)
    manifest = json.loads((output / "source-manifest.json").read_bytes())
    selected = _selected_sources(manifest)
    observed = _observed_reads(trace)
    project_reads = {path for path in observed if path.is_relative_to(ROOT)}
    selected_observed = project_reads & selected
    missing_selected = selected - selected_observed
    forbidden = {
        path for path in project_reads
        if any(token in path.as_posix().lower() for token in FORBIDDEN)
    }
    source_plan = Path(str(manifest["source_plan_path"]))
    code_prefixes = (
        worktree,
        python_prefix,
        ROOT / "msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/.git",
    )
    output_reads = {path for path in project_reads if path.is_relative_to(output)}
    code_reads = {
        path for path in project_reads
        if any(path.is_relative_to(prefix) for prefix in code_prefixes)
    }
    base_reads = project_reads & BASE_READS
    other_reads = project_reads - selected_observed - output_reads - code_reads - base_reads
    if missing_selected or forbidden or source_plan in observed or other_reads:
        raise SystemExit(
            "read audit failed: "
            f"missing={len(missing_selected)} forbidden={len(forbidden)} "
            f"source_plan={source_plan in observed} other={len(other_reads)}"
        )

    trace_copy = output / "openat-trace.log"
    _write_new(trace_copy, trace.read_bytes())
    audit = {
        "schema_version": 1,
        "artifact_type": "wp5_sixth_development_run_open_audit",
        "status": "pass",
        "trace": {"path": trace_copy.name, "sha256": _sha256(trace_copy)},
        "policy": {
            "production_path_must_not_open_oracle_semantic_label_evaluation_test_heldout_or_video5_inputs": True,
            "source_plan_is_identity_only_and_must_not_be_opened": True,
        },
        "counts": {
            "all_absolute_reads": len(observed),
            "project_reads": len(project_reads),
            "declared_selected_source_reads": len(selected),
            "observed_selected_source_reads": len(selected_observed),
            "base_tuple_reads": len(base_reads),
            "output_reads": len(output_reads),
            "code_environment_reads": len(code_reads),
        },
        "declared_selected_source_reads": sorted(map(str, selected)),
        "observed_selected_source_reads": sorted(map(str, selected_observed)),
        "base_tuple_reads": sorted(map(str, base_reads)),
        "source_plan_opened": False,
        "forbidden_project_reads": [],
        "uncategorized_project_reads": [],
    }
    _write_new(output / "open-audit.json", _canonical(audit))

    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact-hashes.json":
            hashes[path.relative_to(output).as_posix()] = {
                "sha256": _sha256(path), "bytes": path.stat().st_size,
            }
    _write_new(
        output / "artifact-hashes.json",
        _canonical(
            {
                "schema_version": 1,
                "artifact_type": "wp5_sixth_development_run_artifact_hashes",
                "status": "complete",
                "artifacts": hashes,
            }
        ),
    )
    print(json.dumps({"status": "pass", "project_reads": len(project_reads)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

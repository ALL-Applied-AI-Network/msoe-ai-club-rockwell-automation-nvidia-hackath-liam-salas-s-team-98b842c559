#!/usr/bin/env python3
"""Build the audited Step 8 Revision 2 train/validation aggregate.

The aggregate contains directory links for approved train/validation attempts only.
Sealed-test chunks remain metadata in the combined summary and are never linked or
opened by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


SHA256_RE = re.compile(r"[0-9a-f]{64}")
CHUNK_RE = re.compile(
    r"^(?P<index>\d{2})-(?P<role>train|val|test)-(?P<start>\d{7})-(?P<end>\d{7})$"
)
COMBINED_COUNTS = {"train": 3480, "val": 360, "test": 560}


def _plan():
    result = []
    result.extend(
        (
            ("00-val-1100000-1100199", "val", 1100000, 1100199, 200),
            ("01-val-1100200-1100299", "val", 1100200, 1100299, 100),
        )
    )
    for index in range(2, 17):
        start = 100000 + (index - 2) * 200
        result.append(
            (f"{index:02d}-train-{start:07d}-{start + 199:07d}", "train", start, start + 199, 200)
        )
    result.extend(
        (
            ("17-test-2100000-2100199", "test", 2100000, 2100199, 200),
            ("18-test-2100200-2100399", "test", 2100200, 2100399, 200),
            ("19-test-2100400-2100499", "test", 2100400, 2100499, 100),
        )
    )
    step6 = tuple(result)

    v4 = []
    for index in range(8):
        start = 104000 + index * 60
        v4.append(
            (f"{index:02d}-train-{start:07d}-{start + 59:07d}", "train", start, start + 59, 60)
        )
    v4.extend(
        (
            ("08-val-1101000-1101059", "val", 1101000, 1101059, 60),
            ("09-test-2101000-2101059", "test", 2101000, 2101059, 60),
        )
    )
    return step6, tuple(v4)


STEP6_PLAN, V4_PLAN = _plan()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_hash(value, description):
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _require_component(summary_path, expected_sha256, plan, expected_counts, name):
    expected_sha256 = _require_hash(expected_sha256, f"{name} expected summary hash")
    actual_sha256 = file_sha256(summary_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{name} summary SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    summary = _read_json(summary_path, f"{name} summary")
    integrity = summary.get("integrity", {})
    if summary.get("schema_version") != 1 or integrity.get("passed") is not True:
        raise ValueError(f"{name} summary integrity must pass at schema version 1")
    if integrity.get("errors"):
        raise ValueError(f"{name} summary integrity contains errors")
    contract = summary.get("batch_contract", {})
    if contract.get("expected_chunks") != len(plan):
        raise ValueError(f"{name} expected chunk count must be {len(plan)}")
    declared_counts = summary.get("batch_contract", {}).get("expected_role_counts")
    if declared_counts != expected_counts:
        raise ValueError(
            f"{name} expected role counts must be {expected_counts}; got {declared_counts}"
        )
    chunks = summary.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(plan):
        raise ValueError(f"{name} summary must contain exactly {len(plan)} chunks")
    by_name = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(f"{name} chunk entry is not an object")
        chunk_name = str(chunk.get("chunk", ""))
        if chunk_name in by_name:
            raise ValueError(f"{name} repeats chunk {chunk_name}")
        by_name[chunk_name] = chunk
    expected_names = {item[0] for item in plan}
    if set(by_name) != expected_names:
        raise ValueError(
            f"{name} chunk set mismatch: missing={sorted(expected_names - set(by_name))}, "
            f"extra={sorted(set(by_name) - expected_names)}"
        )
    for chunk_name, role, seed_start, seed_end, frame_count in plan:
        chunk = by_name[chunk_name]
        expected = {
            "role": role,
            "seed_start": seed_start,
            "seed_end": seed_end,
            "frame_count": frame_count,
        }
        for key, value in expected.items():
            if chunk.get(key) != value:
                raise ValueError(
                    f"{name} chunk {chunk_name} {key} must be {value}; got {chunk.get(key)}"
                )
        if chunk.get("integrity_passed") is not True:
            raise ValueError(f"{name} chunk {chunk_name} integrity did not pass")
        attempt = chunk.get("attempt")
        if not isinstance(attempt, str) or not re.fullmatch(r"attempt-\d+", attempt):
            raise ValueError(f"{name} chunk {chunk_name} has invalid attempt")
        _require_hash(chunk.get("manifest_sha256"), f"{name} chunk {chunk_name} manifest hash")

    frame_count = sum(item[4] for item in plan)
    progress = summary.get("progress")
    if not isinstance(progress, dict) or progress != {
        "complete_chunks": len(plan),
        "expected_chunks": len(plan),
        "frames": frame_count,
        "expected_frames": frame_count,
        "role_counts": expected_counts,
    }:
        raise ValueError(f"{name} progress does not prove a complete batch")

    coverage = summary.get("case_coverage", {}).get("by_role")
    if not isinstance(coverage, dict) or set(coverage) != {"train", "val", "test"}:
        raise ValueError(f"{name} summary lacks exact by-role case coverage")
    for role, count in expected_counts.items():
        bands = coverage.get(role, {}).get("sampling_bands")
        if not isinstance(bands, dict) or sum(bands.values()) != count:
            raise ValueError(f"{name} {role} sampling-band coverage must total {count}")
    return summary, actual_sha256, by_name


def _require_step6_review(review_path, expected_sha256, summary, summary_sha256):
    expected_sha256 = _require_hash(expected_sha256, "Step 6 review expected hash")
    actual_sha256 = file_sha256(review_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Step 6 review SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    review = _read_json(review_path, "Step 6 exception review")
    if review.get("schema_version") != 1 or review.get("status") != "complete":
        raise ValueError("Step 6 exception review must be complete at schema version 1")
    if review.get("unresolved_count") != 0:
        raise ValueError("Step 6 exception review has unresolved items")
    if review.get("batch_summary_sha256") != summary_sha256:
        raise ValueError("Step 6 exception review is not bound to the Step 6 summary")

    inventory = summary.get("old_rule_review", {}).get("exceptions")
    reviewed = review.get("exceptions")
    if not isinstance(inventory, list) or not isinstance(reviewed, list):
        raise ValueError("Step 6 summary/review exception inventories must be lists")
    if (
        summary.get("old_rule_review", {}).get("exception_count") != len(inventory)
        or review.get("exception_count") != len(inventory)
        or review.get("reviewed_count") != len(inventory)
        or len(reviewed) != len(inventory)
    ):
        raise ValueError("Step 6 summary/review exception counts do not match")
    expected_by_seed = {int(item["seed"]): item for item in inventory}
    if len(expected_by_seed) != len(inventory):
        raise ValueError("Step 6 summary repeats an exception seed")
    excluded = []
    seen = set()
    chunks = {chunk["chunk"]: chunk for chunk in summary["chunks"]}
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Step 6 reviewed exception is not an object")
        seed = int(item.get("seed"))
        if seed in seen or seed not in expected_by_seed:
            raise ValueError(f"Step 6 review has unexpected or repeated seed {seed}")
        seen.add(seed)
        source = expected_by_seed[seed]
        for key in ("role", "chunk"):
            if item.get(key) != source.get(key):
                raise ValueError(f"Step 6 review seed {seed} {key} does not match summary")
        chunk = chunks.get(str(item.get("chunk")))
        if chunk is None or item.get("attempt") != chunk.get("attempt"):
            raise ValueError(f"Step 6 review seed {seed} attempt is not approved")
        if item.get("source", {}).get("manifest_sha256") != chunk.get("manifest_sha256"):
            raise ValueError(f"Step 6 review seed {seed} manifest hash is not approved")
        if item.get("disposition") not in ("accepted_legitimate_occlusion", "excluded"):
            raise ValueError(f"Step 6 review seed {seed} has invalid disposition")
        if not isinstance(item.get("explanation"), str) or not item["explanation"].strip():
            raise ValueError(f"Step 6 review seed {seed} lacks an explanation")
        _require_hash(item.get("overlay_sha256"), f"Step 6 review seed {seed} overlay hash")
        if item["disposition"] == "excluded" and item.get("role") in ("train", "val"):
            excluded.append(seed)
    if seen != set(expected_by_seed):
        raise ValueError("Step 6 review does not cover every inventoried exception")
    if review.get("excluded_train_validation_seeds") != sorted(excluded):
        raise ValueError("Step 6 review excluded-seed list does not match dispositions")
    excluded_by_role = {
        role: sum(expected_by_seed[seed].get("role") == role for seed in excluded)
        for role in ("train", "val")
    }
    expected_eligible = {
        role: summary["batch_contract"]["expected_role_counts"][role]
        - excluded_by_role[role]
        for role in ("train", "val")
    }
    if review.get("eligible_counts") != expected_eligible:
        raise ValueError("Step 6 review eligible counts do not match dispositions")
    return review, actual_sha256


def _remap_v4_name(name):
    match = CHUNK_RE.fullmatch(name)
    if not match or int(match.group("index")) not in range(10):
        raise ValueError(f"invalid exact-v4 chunk name: {name}")
    return f"{int(match.group('index')) + 20:02d}-{match.group('role')}-{match.group('start')}-{match.group('end')}"


def _merge_count_trees(first, second, description):
    if isinstance(first, dict) and isinstance(second, dict):
        keys = set(first) | set(second)
        return {
            key: _merge_count_trees(first.get(key, 0), second.get(key, 0), f"{description}.{key}")
            for key in sorted(keys)
        }
    if isinstance(first, bool) or isinstance(second, bool):
        raise ValueError(f"{description} contains a boolean instead of a count")
    if isinstance(first, int) and isinstance(second, int):
        return first + second
    raise ValueError(f"{description} shapes or count types do not match")


def _combined_coverage(step6, v4):
    first = step6["case_coverage"]["by_role"]
    second = v4["case_coverage"]["by_role"]
    by_role = {
        role: _merge_count_trees(first[role], second[role], f"case_coverage.{role}")
        for role in ("train", "val", "test")
    }
    total = _merge_count_trees(
        _merge_count_trees(by_role["train"], by_role["val"], "case_coverage.total"),
        by_role["test"],
        "case_coverage.total",
    )
    return {"total": total, "by_role": by_role}


def _combined_exceptions(step6, v4):
    combined = []
    seen = set()
    for component, summary in (("step6", step6), ("exact_v4", v4)):
        review = summary.get("old_rule_review")
        if not isinstance(review, dict) or not isinstance(review.get("exceptions"), list):
            raise ValueError(f"{component} summary has no exception inventory")
        if review.get("exception_count") != len(review["exceptions"]):
            raise ValueError(f"{component} exception count mismatch")
        for source in review["exceptions"]:
            if not isinstance(source, dict) or source.get("role") not in ("train", "val"):
                raise ValueError(f"{component} exception inventory contains a non-train/val item")
            item = dict(source)
            if component == "exact_v4":
                item["chunk"] = _remap_v4_name(str(item.get("chunk", "")))
            seed = int(item["seed"])
            if seed in seen:
                raise ValueError(f"duplicate old-rule exception seed {seed}")
            seen.add(seed)
            item["component"] = component
            combined.append(item)
    combined.sort(key=lambda item: (item["role"], int(item["seed"])))
    return {
        "status": "pending_combined_review",
        "policy": (
            "Before training, verify and disposition every train/validation exception "
            "against the fresh aggregate root."
        ),
        "exception_count": len(combined),
        "exceptions": combined,
        "sealed_test_records_inventoried": 0,
    }


def _safe_attempt_root(collected_root, chunk_name, chunk):
    root = Path(collected_root).resolve()
    if not root.is_dir():
        raise ValueError(f"collected root is not a directory: {collected_root}")
    chunk_root = (root / chunk_name).resolve()
    if root not in chunk_root.parents:
        raise ValueError(f"chunk path escapes collected root: {chunk_name}")
    attempt_name = chunk["attempt"]
    nested = chunk_root / attempt_name
    if nested.is_dir():
        attempt = nested.resolve()
    elif chunk.get("storage_layout") == "direct_copy_of_attempt-01" and attempt_name == "attempt-01":
        attempt = chunk_root
    else:
        raise ValueError(f"approved attempt is missing for chunk {chunk_name}")
    if root not in attempt.parents and attempt != root:
        raise ValueError(f"attempt path escapes collected root: {chunk_name}")
    state_path = (attempt / "state.txt").resolve()
    if attempt not in state_path.parents:
        raise ValueError(f"state path escapes approved attempt: {chunk_name}")
    try:
        state = state_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"could not read state for chunk {chunk_name}: {error}") from error
    if state != "COMPLETE":
        raise ValueError(f"chunk {chunk_name} attempt is not COMPLETE")
    manifest = (attempt / "output" / "manifest.jsonl").resolve()
    if attempt not in manifest.parents:
        raise ValueError(f"manifest path escapes approved attempt: {chunk_name}")
    try:
        manifest_hash = file_sha256(manifest)
    except OSError as error:
        raise ValueError(f"could not read manifest for chunk {chunk_name}: {error}") from error
    if manifest_hash != chunk["manifest_sha256"]:
        raise ValueError(f"chunk {chunk_name} manifest hash changed after component summary")
    return attempt


def _create_directory_link(path, target):
    path = Path(path).absolute()
    target = Path(target).resolve()
    if os.name == "nt":
        environment = os.environ.copy()
        environment["FLUIDMIX_AGGREGATE_LINK"] = str(path)
        environment["FLUIDMIX_AGGREGATE_TARGET"] = str(target)
        command = (
            "$ErrorActionPreference='Stop'; "
            "New-Item -ItemType Junction "
            "-Path ([WildcardPattern]::Escape($env:FLUIDMIX_AGGREGATE_LINK)) "
            "-Target $env:FLUIDMIX_AGGREGATE_TARGET | Out-Null"
        )
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise OSError(f"could not create directory junction {path}: {detail}")
    else:
        os.symlink(str(target), str(path), target_is_directory=True)
    if not path.is_dir() or path.resolve() != target:
        raise OSError(f"directory link does not resolve to its approved target: {path}")


def _is_within(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    return path == parent or parent in path.parents


def _combined_chunks(step6_by_name, v4_by_name):
    chunks = []
    for component, plan, by_name in (
        ("step6", STEP6_PLAN, step6_by_name),
        ("exact_v4", V4_PLAN, v4_by_name),
    ):
        for source_name, role, seed_start, seed_end, frame_count in plan:
            source = by_name[source_name]
            name = source_name if component == "step6" else _remap_v4_name(source_name)
            chunks.append(
                {
                    "chunk": name,
                    "source_chunk": source_name,
                    "component": component,
                    "role": role,
                    "attempt": source["attempt"],
                    "frame_count": frame_count,
                    "seed_start": seed_start,
                    "seed_end": seed_end,
                    "manifest_sha256": source["manifest_sha256"],
                    "integrity_passed": True,
                }
            )
    return chunks


def build_aggregate(
    *,
    step6_summary_path,
    step6_summary_sha256,
    step6_review_path,
    step6_review_sha256,
    step6_collected_root,
    v4_summary_path,
    v4_summary_sha256,
    v4_collected_root,
    aggregate_root,
    report_out,
):
    aggregate_root = Path(aggregate_root)
    report_out = Path(report_out)
    if aggregate_root.exists() or aggregate_root.is_symlink():
        raise FileExistsError(f"refusing to reuse aggregate root: {aggregate_root}")
    if report_out.exists() or report_out.is_symlink():
        raise FileExistsError(f"refusing to overwrite combined summary: {report_out}")
    if _is_within(report_out, aggregate_root) or _is_within(aggregate_root, report_out):
        raise ValueError("aggregate root and combined summary path must not overlap")
    for source_root in (step6_collected_root, v4_collected_root):
        if _is_within(aggregate_root, source_root):
            raise ValueError("aggregate root cannot be created inside a collected source")
        if _is_within(report_out, source_root):
            raise ValueError("combined summary cannot be written inside a collected source")

    step6, step6_hash, step6_by_name = _require_component(
        step6_summary_path,
        step6_summary_sha256,
        STEP6_PLAN,
        {"train": 3000, "val": 300, "test": 500},
        "Step 6",
    )
    _step6_review, step6_review_hash = _require_step6_review(
        step6_review_path,
        step6_review_sha256,
        step6,
        step6_hash,
    )
    v4, v4_hash, v4_by_name = _require_component(
        v4_summary_path,
        v4_summary_sha256,
        V4_PLAN,
        {"train": 480, "val": 60, "test": 60},
        "exact-v4",
    )
    chunks = _combined_chunks(step6_by_name, v4_by_name)
    if len({chunk["chunk"] for chunk in chunks}) != 30:
        raise ValueError("combined chunk names are not unique")
    seeds = []
    for chunk in chunks:
        seeds.extend(range(chunk["seed_start"], chunk["seed_end"] + 1))
    if len(seeds) != sum(COMBINED_COUNTS.values()) or len(seeds) != len(set(seeds)):
        raise ValueError("combined seed contract is not unique and complete")

    attempts = {}
    for chunk in chunks:
        if chunk["role"] == "test":
            continue
        source_root = step6_collected_root if chunk["component"] == "step6" else v4_collected_root
        source_name = chunk["source_chunk"]
        source = step6_by_name[source_name] if chunk["component"] == "step6" else v4_by_name[source_name]
        attempts[chunk["chunk"]] = _safe_attempt_root(source_root, source_name, source)
    if len(set(attempts.values())) != len(attempts):
        raise ValueError("two combined chunks resolve to the same source attempt")

    coverage = _combined_coverage(step6, v4)
    for role, expected in COMBINED_COUNTS.items():
        if sum(coverage["by_role"][role]["sampling_bands"].values()) != expected:
            raise ValueError(f"combined {role} case coverage does not total {expected}")
    exceptions = _combined_exceptions(step6, v4)

    aggregate_root.mkdir(parents=True, exist_ok=False)
    for chunk in chunks:
        if chunk["role"] == "test":
            continue
        chunk_root = aggregate_root / chunk["chunk"]
        chunk_root.mkdir()
        link = chunk_root / chunk["attempt"]
        _create_directory_link(link, attempts[chunk["chunk"]])

    report = {
        "schema_version": 1,
        "purpose": "FillCam Step 8 Revision 2 combined synthetic batch summary",
        "evidence_label": "derived_from_two_integrity_passed_component_summaries",
        "components": {
            "step6": {
                "summary_sha256": step6_hash,
                "exception_review_sha256": step6_review_hash,
                "exception_review_status": "complete",
                "expected_role_counts": {"train": 3000, "val": 300, "test": 500},
            },
            "exact_v4": {
                "summary_sha256": v4_hash,
                "expected_role_counts": {"train": 480, "val": 60, "test": 60},
            },
        },
        "batch_contract": {
            "expected_chunks": 30,
            "expected_role_counts": dict(COMBINED_COUNTS),
            "training_root_roles": ["train", "val"],
            "test_chunks_linked": 0,
            "new_chunk_index_remap": "exact-v4 00-09 -> combined 20-29",
        },
        "case_coverage": coverage,
        "old_rule_review": exceptions,
        "chunks": chunks,
        "integrity": {
            "passed": True,
            "errors": [],
            "component_summaries_passed": 2,
            "complete_chunks": 30,
            "linked_train_validation_chunks": len(attempts),
            "test_chunks_linked": 0,
            "linked_manifest_hashes_verified": len(attempts),
            "aggregate_root_contains_test_chunk": False,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step6-summary", type=Path, required=True)
    parser.add_argument("--step6-summary-sha256", required=True)
    parser.add_argument("--step6-review", type=Path, required=True)
    parser.add_argument("--step6-review-sha256", required=True)
    parser.add_argument("--step6-collected-root", type=Path, required=True)
    parser.add_argument("--v4-summary", type=Path, required=True)
    parser.add_argument("--v4-summary-sha256", required=True)
    parser.add_argument("--v4-collected-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser


def main():
    args = _parser().parse_args()
    report = build_aggregate(
        step6_summary_path=args.step6_summary,
        step6_summary_sha256=args.step6_summary_sha256,
        step6_review_path=args.step6_review,
        step6_review_sha256=args.step6_review_sha256,
        step6_collected_root=args.step6_collected_root,
        v4_summary_path=args.v4_summary,
        v4_summary_sha256=args.v4_summary_sha256,
        v4_collected_root=args.v4_collected_root,
        aggregate_root=args.aggregate_root,
        report_out=args.report_out,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "chunks": len(report["chunks"]),
                "linked_train_validation_chunks": report["integrity"][
                    "linked_train_validation_chunks"
                ],
                "test_chunks_linked": 0,
                "report": str(args.report_out.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the exact-v4 600-frame FillCam Revision 2 render batch.

The raw RGB images, masks, logs, and manifests remain below ignored output
directories. This command writes only compact integrity, provenance, coverage,
and throughput evidence. Sealed-test RGB files are byte-hashed for integrity;
this summarizer never decodes them or derives development measurements from
their pixels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import summarize_fill_level_batch as base  # noqa: E402


PURPOSE = "FillCam Step 8 Revision 2 exact-v4 batch integrity summary"
EXPECTED_RENDERER_SHA256 = (
    "8e23ece14ed5c58599981c59589d4c9f0484703fc629ad760333e0d6f09025cd"
)
EXPECTED_VARIATION_SHA256 = (
    "71aea9f84e7b6c84e7ed807d3784891715e3ddbeac5dea8f17ed291d9f52a182"
)
EXPECTED_CHECKER_SHA256 = (
    "a03442a7da4cde31dee084ffbe3375f0574b370a78829c01976d77f49246e966"
)
EXPECTED_SPEC_TREE_SHA256 = (
    "44442cd6529baf0445643bc09ac226ea924df4dc9868ea7716f6033aba76327f"
)
EXPECTED_CAMERA_MATCH_SHA256 = (
    "aeee0c5ef7e5b8b88678356f30de4d14a3126da70e4eb5c84711c674bfafc2f8"
)
EXPECTED_SOURCE_CAMERA_SHA256 = base.EXPECTED_SOURCE_CAMERA_SHA256
EXPECTED_STATION_TREE_HASHES = base.EXPECTED_STATION_TREE_HASHES
EXPECTED_INPUT_FILE_HASHES = {
    "camera-match-r2-v4-batch.json": EXPECTED_CAMERA_MATCH_SHA256,
    "check_fill_level_capture_r2_v4_batch.py": EXPECTED_CHECKER_SHA256,
    "render_fill_level_dataset_r2_v4_batch.py": EXPECTED_RENDERER_SHA256,
    "run-step8-r2-v4-batch-v1.sh": (
        "6f746b34a70500f171916f6bfd1285dadb34ddeed5629a8e4632b8802ae86c4d"
    ),
    "scripts/make_fill_level_specs.py": (
        "40f45c825dadb37719e7599116f8de2cccf1356791bfc5715460d95d7da4c3ef"
    ),
    "src/fluidmix/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/fluidmix/fill_level_variation.py": EXPECTED_VARIATION_SHA256,
    "RUN-CONTRACT.txt": (
        "993c729c4669349905b3f1d8394ea9773be30ba3be3b63c89d264031dfcdd68a"
    ),
    "INPUT_SHA256SUMS-r2-v4-batch": (
        "5decd42c8a8f786d37252bca07cb3e23321d1eada25f6ce3d7bcb485c87fa58a"
    ),
}


def _batch_plan():
    plan = []
    for index in range(8):
        start = 104_000 + 60 * index
        plan.append(
            {
                "chunk": f"{index:02d}-train-{start:07d}-{start + 59:07d}",
                "role": "train",
                "seed_start": start,
                "seed_end": start + 59,
            }
        )
    plan.extend(
        (
            {
                "chunk": "08-val-1101000-1101059",
                "role": "val",
                "seed_start": 1_101_000,
                "seed_end": 1_101_059,
            },
            {
                "chunk": "09-test-2101000-2101059",
                "role": "test",
                "seed_start": 2_101_000,
                "seed_end": 2_101_059,
            },
        )
    )
    return tuple(plan)


BATCH_PLAN = _batch_plan()
EXPECTED_COUNTS = {"train": 480, "val": 60, "test": 60}
EXPECTED_SEED_RANGES = {
    "train": [104_000, 104_479],
    "val": [1_101_000, 1_101_059],
    "test": [2_101_000, 2_101_059],
}


def _failed_report(error):
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "integrity": {"passed": False, "errors": [str(error)]},
    }


def _resolve_attempt(chunk_dir, *, allow_direct):
    """Return the one preserved attempt and its storage-layout evidence."""
    attempts = sorted(
        path for path in Path(chunk_dir).glob("attempt-*") if path.is_dir()
    )
    direct_state = Path(chunk_dir) / "state.txt"
    if attempts:
        if direct_state.exists():
            return None, None, "chunk mixes direct and nested attempt layouts"
        names = [path.name for path in attempts]
        if names != ["attempt-01"]:
            return None, None, f"expected only attempt-01; found {names}"
        return attempts[0], "nested_attempt-01", None
    if direct_state.is_file():
        if not allow_direct:
            return None, None, "direct attempt layout is approved only for chunk 00"
        return Path(chunk_dir), "direct_copy_of_attempt-01", None
    return None, None, "missing attempt-01 state evidence"


def _attempt_evidence(attempt, expected_records):
    required = {
        "state": attempt / "state.txt",
        "stored_checker": attempt / "check.json",
        "checker_stderr": attempt / "check.stderr",
        "render_log": attempt / "render.log",
    }
    errors = [
        f"missing {name}: {path}"
        for name, path in required.items()
        if not path.is_file()
    ]
    if errors:
        return {}, None, errors
    if required["state"].read_text(encoding="utf-8").strip() != "COMPLETE":
        errors.append("attempt state is not COMPLETE")
    if required["checker_stderr"].stat().st_size != 0:
        errors.append("stored checker stderr is not empty")
    try:
        stored_check = base._read_json(
            required["stored_checker"], "stored checker result"
        )
    except ValueError as error:
        errors.append(str(error))
        stored_check = None
    if stored_check is not None:
        if not stored_check.get("passed"):
            errors.append("stored checker result did not pass")
        if stored_check.get("record_count") != expected_records:
            errors.append("stored checker record_count does not match manifest")
        if stored_check.get("errors"):
            errors.append("stored checker result contains errors")
    hashes = {
        f"{name}_sha256": base.file_sha256(path)
        for name, path in required.items()
    }
    return hashes, stored_check, errors


def _output_file_integrity(output_root, records):
    """Hash declared output bytes without decoding any RGB or mask file."""
    output_root = Path(output_root).resolve()
    errors = []
    ledger = []
    seen = set()
    role_file_counts = Counter()
    for index, record in enumerate(records):
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            errors.append(f"record {index} outputs is not an object")
            continue
        for kind in ("rgb", "mask"):
            item = outputs.get(kind)
            if not isinstance(item, dict):
                errors.append(f"record {index} missing {kind} output object")
                continue
            relative_text = item.get("path")
            declared_hash = item.get("sha256")
            if not isinstance(relative_text, str) or not relative_text:
                errors.append(f"record {index} {kind} output path is invalid")
                continue
            relative = Path(relative_text)
            resolved = (output_root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or output_root not in resolved.parents
            ):
                errors.append(f"record {index} {kind} output escapes output root")
                continue
            if resolved in seen:
                errors.append(f"record {index} {kind} output file is reused")
                continue
            seen.add(resolved)
            if not isinstance(declared_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", declared_hash
            ):
                errors.append(f"record {index} {kind} output SHA-256 is invalid")
                continue
            if not resolved.is_file():
                errors.append(f"record {index} {kind} output is missing")
                continue
            actual_hash = base.file_sha256(resolved)
            if actual_hash != declared_hash:
                errors.append(f"record {index} {kind} output SHA-256 mismatch")
            role = str(record.get("role"))
            role_file_counts[role] += 1
            ledger.append(
                f"{role}\t{record.get('seed')}\t{kind}\t"
                f"{relative.as_posix()}\t{actual_hash}\n"
            )
    discovered = set()
    for kind in ("rgb", "mask"):
        kind_root = output_root / kind
        if not kind_root.is_dir():
            errors.append(f"missing output directory: {kind_root}")
            continue
        for path in kind_root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            discovered.add(resolved)
            if output_root not in resolved.parents:
                errors.append(
                    f"{kind} output tree contains a file outside output root: {path}"
                )
            elif resolved not in seen:
                errors.append(
                    f"unmanifested {kind} output file: "
                    f"{resolved.relative_to(output_root).as_posix()}"
                )
    ledger_hash = hashlib.sha256("".join(sorted(ledger)).encode()).hexdigest()
    return {
        "file_count": len(ledger),
        "discovered_file_count": len(discovered),
        "file_counts_by_role": {
            role: int(role_file_counts.get(role, 0))
            for role in EXPECTED_COUNTS
        },
        "ordered_hash_ledger_sha256": ledger_hash,
        "method": "raw file bytes SHA-256; no image decoding by this summarizer",
    }, errors


def _input_evidence(input_root):
    observed = {}
    errors = []
    for relative, expected in EXPECTED_INPUT_FILE_HASHES.items():
        path = input_root / relative
        if not path.is_file():
            errors.append(f"missing pinned input file: {relative}")
            continue
        actual = base.file_sha256(path)
        observed[relative] = actual
        if actual != expected:
            errors.append(
                f"pinned input hash mismatch for {relative}: {actual} != {expected}"
            )
    return observed, errors


def build_report(collected_root, input_root, *, validate_manifest=None):
    """Return compact evidence while enforcing the exact immutable v4 contract."""
    collected_root = Path(collected_root).expanduser().resolve()
    input_root = Path(input_root).expanduser().resolve()
    specs_root = input_root / "specs"
    validate_manifest = validate_manifest or base.capture_checker.validate_manifest
    if not collected_root.is_dir():
        return _failed_report(f"collected root does not exist: {collected_root}")
    if not input_root.is_dir():
        return _failed_report(f"input root does not exist: {input_root}")
    if not specs_root.is_dir():
        return _failed_report(f"specs root does not exist: {specs_root}")

    errors = []
    observed_inputs, input_errors = _input_evidence(input_root)
    errors.extend(input_errors)
    checker_hash = base.file_sha256(base.capture_checker.__file__)
    if checker_hash != EXPECTED_CHECKER_SHA256:
        errors.append(
            f"offline checker hash {checker_hash} does not match pinned "
            f"{EXPECTED_CHECKER_SHA256}"
        )
    measured_spec_tree_hash = base.spec_tree_sha256(specs_root)
    if measured_spec_tree_hash != EXPECTED_SPEC_TREE_SHA256:
        errors.append(
            f"spec tree hash {measured_spec_tree_hash} does not match "
            f"{EXPECTED_SPEC_TREE_SHA256}"
        )

    expected_names = {item["chunk"] for item in BATCH_PLAN}
    actual_names = {path.name for path in collected_root.iterdir() if path.is_dir()}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        errors.append(f"missing chunk directories: {missing}")
    if unexpected:
        errors.append(f"unexpected chunk directories: {unexpected}")

    chunks = []
    all_records = []
    train_validation_records = []
    total_seconds = 0.0
    role_cases = {role: {} for role in EXPECTED_COUNTS}
    total_cases = {}
    old_rule_exceptions = []
    observed_renderer_hashes = set()
    observed_variation_hashes = set()
    observed_camera_hashes = set()
    observed_source_camera_hashes = set()
    observed_station_hashes = set()

    for contract_index, contract in enumerate(BATCH_PLAN):
        chunk_name = contract["chunk"]
        chunk_dir = collected_root / chunk_name
        if not chunk_dir.is_dir():
            continue
        chunk_errors = []
        attempt, storage_layout, attempt_error = _resolve_attempt(
            chunk_dir, allow_direct=contract_index == 0
        )
        if attempt_error:
            errors.append(f"{chunk_name}: {attempt_error}")
            continue
        output = attempt / "output"
        manifest_path = output / "manifest.jsonl"
        summary_path = output / "summary.json"
        specs_path = specs_root / f"{chunk_name}.json"
        for description, path in (
            ("manifest", manifest_path),
            ("render summary", summary_path),
            ("chunk specs", specs_path),
        ):
            if not path.is_file():
                chunk_errors.append(f"missing {description}: {path}")
        if chunk_errors:
            errors.extend(f"{chunk_name}: {error}" for error in chunk_errors)
            continue
        try:
            records = base._read_records(manifest_path)
            specs = base._read_specs(specs_path)
            capture_summary = base._read_json(summary_path, "render summary")
        except ValueError as error:
            errors.append(f"{chunk_name}: {error}")
            continue

        attempt_hashes, stored_check, attempt_errors = _attempt_evidence(
            attempt, len(records)
        )
        chunk_errors.extend(attempt_errors)
        check_result = validate_manifest(manifest_path)
        if not check_result.get("passed"):
            chunk_errors.append("offline checker did not pass")
            chunk_errors.extend(
                f"offline checker: {error}"
                for error in check_result.get("errors", [])
            )
        if check_result.get("record_count") != len(records):
            chunk_errors.append("offline checker record_count does not match manifest")

        specs_hash = base.file_sha256(specs_path)
        summary_errors, manifest_hash, seconds = base._validate_summary(
            capture_summary,
            records,
            manifest_path,
            EXPECTED_RENDERER_SHA256,
            EXPECTED_VARIATION_SHA256,
            specs_hash,
            EXPECTED_CAMERA_MATCH_SHA256,
            EXPECTED_SOURCE_CAMERA_SHA256,
            EXPECTED_STATION_TREE_HASHES,
        )
        chunk_errors.extend(summary_errors)
        expected_seeds = list(
            range(contract["seed_start"], contract["seed_end"] + 1)
        )
        if [record.get("role") for record in records] != [contract["role"]] * 60:
            chunk_errors.append("manifest roles do not match chunk contract")
        if [record.get("seed") for record in records] != expected_seeds:
            chunk_errors.append("manifest seeds do not match chunk contract")
        if len(specs) != len(records):
            chunk_errors.append("spec count does not match manifest record count")
        elif not all(
            base._contains_spec(record, spec)
            for record, spec in zip(records, specs)
        ):
            chunk_errors.append("manifest records do not preserve their input specs")

        output_integrity, output_errors = _output_file_integrity(output, records)
        chunk_errors.extend(output_errors)
        renderer_hash = capture_summary.get("renderer_script_sha256")
        variation_hash = capture_summary.get("variation_module_sha256")
        camera_hash = capture_summary.get("camera_match_sha256")
        source_camera_hash = capture_summary.get(
            "source_camera_json_sha256_before"
        )
        station_hashes = capture_summary.get("source_station_tree_hashes_before")
        observed_renderer_hashes.add(renderer_hash)
        observed_variation_hashes.add(variation_hash)
        observed_camera_hashes.add(camera_hash)
        observed_source_camera_hashes.add(source_camera_hash)
        observed_station_hashes.add(json.dumps(station_hashes, sort_keys=True))
        if seconds is not None:
            total_seconds += seconds
        cases = base._case_counts(records)
        base._add_counts(role_cases[contract["role"]], cases)
        base._add_counts(total_cases, cases)
        if contract["role"] in ("train", "val"):
            train_validation_records.extend(records)
            old_rule_exceptions.extend(
                base._old_rule_exceptions(records, chunk_name)
            )
        all_records.extend(records)
        chunks.append(
            {
                "chunk": chunk_name,
                "attempt": "attempt-01",
                "storage_layout": storage_layout,
                "role": contract["role"],
                "seed_start": contract["seed_start"],
                "seed_end": contract["seed_end"],
                "frame_count": len(records),
                "seconds": seconds,
                "frames_per_hour": (
                    3600.0 * len(records) / seconds
                    if seconds is not None
                    else None
                ),
                "manifest_sha256": manifest_hash,
                "render_summary_sha256": base.file_sha256(summary_path),
                "specs_sha256": specs_hash,
                **attempt_hashes,
                "renderer_script_sha256": renderer_hash,
                "variation_module_sha256": variation_hash,
                "offline_capture_check_passed": bool(check_result.get("passed")),
                "stored_capture_check_passed": bool(
                    stored_check and stored_check.get("passed")
                ),
                "output_file_integrity": output_integrity,
                "case_coverage": cases,
                "old_rule_exception_count": len(
                    base._old_rule_exceptions(records, chunk_name)
                    if contract["role"] in ("train", "val")
                    else []
                ),
                "integrity_passed": not chunk_errors,
                "errors": chunk_errors,
            }
        )
        errors.extend(f"{chunk_name}: {error}" for error in chunk_errors)

    seed_counts = Counter(record.get("seed") for record in all_records)
    duplicates = sorted(seed for seed, count in seed_counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate seeds across manifests: {duplicates}")
    role_counts = Counter(record.get("role") for record in all_records)
    measured_counts = {
        role: int(role_counts.get(role, 0)) for role in EXPECTED_COUNTS
    }
    if measured_counts != EXPECTED_COUNTS:
        errors.append(
            f"role counts {measured_counts} do not match expected {EXPECTED_COUNTS}"
        )
    unexpected_roles = sorted(set(role_counts) - set(EXPECTED_COUNTS), key=str)
    if unexpected_roles:
        errors.append(f"unexpected roles: {unexpected_roles}")
    for description, observed, expected in (
        ("renderer", observed_renderer_hashes, {EXPECTED_RENDERER_SHA256}),
        ("variation", observed_variation_hashes, {EXPECTED_VARIATION_SHA256}),
        ("camera", observed_camera_hashes, {EXPECTED_CAMERA_MATCH_SHA256}),
        (
            "source camera",
            observed_source_camera_hashes,
            {EXPECTED_SOURCE_CAMERA_SHA256},
        ),
        (
            "source station",
            observed_station_hashes,
            {json.dumps(EXPECTED_STATION_TREE_HASHES, sort_keys=True)},
        ),
    ):
        if observed != expected:
            errors.append(f"observed {description} hashes do not equal pinned contract")

    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "evidence_label": "measured synthetic exact-v4 render batch",
        "batch_contract": {
            "expected_chunks": 10,
            "expected_role_counts": EXPECTED_COUNTS,
            "expected_seed_ranges_inclusive": EXPECTED_SEED_RANGES,
            "expected_renderer": base.EXPECTED_RENDERER,
            "expected_subframes": base.EXPECTED_SUBFRAMES,
            "expected_resolution": base.EXPECTED_RESOLUTION,
            "expected_frame": base.EXPECTED_FRAME,
            "expected_renderer_script_sha256": EXPECTED_RENDERER_SHA256,
            "expected_variation_module_sha256": EXPECTED_VARIATION_SHA256,
            "expected_checker_sha256": EXPECTED_CHECKER_SHA256,
            "expected_camera_match_sha256": EXPECTED_CAMERA_MATCH_SHA256,
            "expected_source_camera_sha256": EXPECTED_SOURCE_CAMERA_SHA256,
            "expected_station_tree_hashes": EXPECTED_STATION_TREE_HASHES,
        },
        "progress": {
            "complete_chunks": len(chunks),
            "expected_chunks": 10,
            "frames": len(all_records),
            "expected_frames": 600,
            "role_counts": measured_counts,
        },
        "provenance": {
            "input_file_sha256": observed_inputs,
            "offline_checker_sha256": checker_hash,
            "spec_tree_sha256": measured_spec_tree_hash,
            "spec_tree_hash_algorithm": (
                "SHA-256 of sorted GNU sha256sum lines: "
                "<file_sha256><two spaces><basename><newline>"
            ),
            "observed_renderer_script_sha256": sorted(
                value for value in observed_renderer_hashes if value is not None
            ),
            "observed_variation_module_sha256": sorted(
                value for value in observed_variation_hashes if value is not None
            ),
            "camera_match_sha256": (
                next(iter(observed_camera_hashes))
                if len(observed_camera_hashes) == 1
                else None
            ),
            "source_camera_json_sha256": (
                next(iter(observed_source_camera_hashes))
                if len(observed_source_camera_hashes) == 1
                else None
            ),
            "source_station_tree_hashes": (
                json.loads(next(iter(observed_station_hashes)))
                if len(observed_station_hashes) == 1
                else None
            ),
        },
        "throughput": {
            "summed_render_seconds": total_seconds,
            "frames_per_hour": (
                3600.0 * len(all_records) / total_seconds
                if total_seconds > 0.0
                else None
            ),
        },
        "sealed_test_policy": {
            "rgb_decoded_by_this_summarizer": 0,
            "integrity_only_file_hash_reads": 120,
            "old_rule_exceptions_inventoried": False,
        },
        "case_coverage": {"total": total_cases, "by_role": role_cases},
        "label_coverage_train_validation_only": base._label_coverage(
            train_validation_records
        ),
        "old_rule_review": {
            "exception_count": len(old_rule_exceptions),
            "roles_inventoried": ["train", "val"],
            "sealed_test_records_inventoried": 0,
            "status": "pending_manual_review",
            "policy": (
                "Before training, explain every listed exception with mask-on-RGB "
                "evidence and exclude any unexplained train or validation frame."
            ),
            "exceptions": old_rule_exceptions,
        },
        "chunks": chunks,
        "integrity": {
            "passed": not errors,
            "offline_checks_passed": sum(
                bool(chunk["offline_capture_check_passed"])
                for chunk in chunks
            ),
            "errors": errors,
        },
    }


def _write_json_new(path, value):
    return base._write_json_new(path, value)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = build_report(args.collected_root, args.input_root)
    report_path = _write_json_new(args.report_out, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "chunks": report.get("progress", {}).get("complete_chunks", 0),
                "frames": report.get("progress", {}).get("frames", 0),
                "integrity_passed": report["integrity"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["integrity"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

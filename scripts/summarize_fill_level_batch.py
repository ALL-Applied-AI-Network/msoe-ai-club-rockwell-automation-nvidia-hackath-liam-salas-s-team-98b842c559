"""Verify and summarize the copied Step 6 FillCam render batch.

Raw RGB images, masks, logs, and manifests stay below the ignored collected
directory.  The generated report contains only small integrity, provenance,
coverage, and throughput measurements suitable for review and Git.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import check_fill_level_capture as capture_checker  # noqa: E402


EXPECTED_RENDERER = "PathTracing"
EXPECTED_SUBFRAMES = 32
EXPECTED_RESOLUTION = [640, 480]
EXPECTED_FRAME = 686
EXPECTED_VARIATION_SHA256 = (
    "b8e404c582758de3da0051c503d28e10edfe743bd3c4c5ee1ba586c421479898"
)
APPROVED_RENDERER_SHA256 = (
    "e807f93ad5030e0aa6eea8bf4630a772baa81a48849ea9d823cba8d0c2057958",
    "0d0f03c201db8ee3b25680f5dec78025ab8ca8e154861c4a6e3a3b165bd91717",
)
DEFAULT_RENDERER_POLICY = (
    {
        "chunk_start": 0,
        "chunk_end": 2,
        "sha256": APPROVED_RENDERER_SHA256[0],
    },
    {
        "chunk_start": 3,
        "chunk_end": 19,
        "sha256": APPROVED_RENDERER_SHA256[1],
    },
)
SPEC_TREE_SHA256 = (
    "f30ada8dcf5b6ef6f20ac31c4d45a34ac24c648cf7d6067c3e7bd73753d79a7b"
)
EXPECTED_CAMERA_MATCH_SHA256 = (
    "aeee0c5ef7e5b8b88678356f30de4d14a3126da70e4eb5c84711c674bfafc2f8"
)
EXPECTED_SOURCE_CAMERA_SHA256 = (
    "11966571ffa71a979297a0c6c730c739f33d86a0578fe0b4b890dd4a7ce1c936"
)
EXPECTED_STATION_TREE_HASHES = {
    "fill_station.usda": (
        "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9"
    ),
    "geometry/fill_station.usdc": (
        "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
    ),
    "geometry/textures/color_B7C6CE.exr": (
        "7b3ff7cc6c709d4b5ae34c40b60277db472b1c7d483a744cf0990ab7023d61cc"
    ),
}
FORBIDDEN_SEEDS = frozenset((*range(8100, 8120), 16646, 17351))


def _batch_plan():
    plan = [
        {
            "chunk": "00-val-1100000-1100199",
            "role": "val",
            "seed_start": 1_100_000,
            "seed_end": 1_100_199,
        },
        {
            "chunk": "01-val-1100200-1100299",
            "role": "val",
            "seed_start": 1_100_200,
            "seed_end": 1_100_299,
        },
    ]
    for offset in range(15):
        start = 100_000 + 200 * offset
        end = start + 199
        plan.append(
            {
                "chunk": f"{offset + 2:02d}-train-{start:07d}-{end:07d}",
                "role": "train",
                "seed_start": start,
                "seed_end": end,
            }
        )
    for index, (start, end) in enumerate(
        (
            (2_100_000, 2_100_199),
            (2_100_200, 2_100_399),
            (2_100_400, 2_100_499),
        ),
        17,
    ):
        plan.append(
            {
                "chunk": f"{index:02d}-test-{start:07d}-{end:07d}",
                "role": "test",
                "seed_start": start,
                "seed_end": end,
            }
        )
    return tuple(plan)


BATCH_PLAN = _batch_plan()
EXPECTED_COUNTS = {
    role: sum(
        item["seed_end"] - item["seed_start"] + 1
        for item in BATCH_PLAN
        if item["role"] == role
    )
    for role in ("train", "val", "test")
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spec_tree_sha256(specs_root):
    """Match ``sha256sum *.json | sha256sum`` with sorted basenames."""
    specs_root = Path(specs_root)
    entries = []
    for path in sorted(specs_root.glob("*.json"), key=lambda item: item.name):
        entries.append(f"{file_sha256(path)}  {path.name}\n")
    return hashlib.sha256("".join(entries).encode()).hexdigest()


def _read_specs(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read specs {path}: {error}") from error
    if not isinstance(value, list) or not value:
        raise ValueError(f"specs must be a non-empty JSON array: {path}")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"each spec must be a JSON object: {path}")
    return value


def _contains_spec(record, spec):
    """Return whether a renderer-enriched record preserves every spec field."""
    if isinstance(spec, dict):
        return isinstance(record, dict) and all(
            key in record and _contains_spec(record[key], value)
            for key, value in spec.items()
        )
    if isinstance(spec, list):
        return (
            isinstance(record, list)
            and len(record) == len(spec)
            and all(_contains_spec(actual, expected) for actual, expected in zip(record, spec))
        )
    return record == spec


def _renderer_hash_for_chunk(chunk_name, renderer_policy):
    index = int(chunk_name.split("-", 1)[0])
    matches = [
        item["sha256"]
        for item in renderer_policy
        if item["chunk_start"] <= index <= item["chunk_end"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"chunk {index:02d} requires exactly one renderer policy; found {len(matches)}"
        )
    return matches[0]


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _read_records(manifest_path):
    records = []
    for line_number, raw_line in enumerate(Path(manifest_path).read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"manifest line {line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError(f"manifest contains no records: {manifest_path}")
    return records


def _case_counts(records):
    counts = {
        "clean": 0,
        "foam": 0,
        "sticker": 0,
        "wet_wall": 0,
        "edge_zone": 0,
        "hidden": 0,
        "zero_level": 0,
        "sampling_bands": {},
    }
    for record in records:
        foam = bool(record.get("foam", {}).get("enabled"))
        sticker = bool(record.get("sticker_pose", {}).get("present"))
        counts["clean"] += int(not foam and not sticker)
        counts["foam"] += int(foam)
        counts["sticker"] += int(sticker)
        counts["wet_wall"] += int(
            bool(record.get("wet_walls", {}).get("enabled"))
        )
        band = str(record.get("level_sampling_band", ""))
        counts["sampling_bands"][band] = counts["sampling_bands"].get(band, 0) + 1
        counts["edge_zone"] += int("edge" in band)
        counts["hidden"] += int("hidden" in band)
        try:
            counts["zero_level"] += int(float(record.get("level_ml")) <= 0.0)
        except (TypeError, ValueError):
            pass
    return counts


def _add_counts(target, source):
    for name, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(name, {})
            _add_counts(child, value)
        else:
            target[name] = target.get(name, 0) + int(value)


def _label_coverage(records):
    names = ("empty", "liquid", "foam", "blocked", "ignore")
    frames = {name: 0 for name in names}
    pixels = {name: 0 for name in names}
    for record in records:
        class_pixels = record.get("label_stats", {}).get("class_pixels", {})
        for name in names:
            try:
                count = int(class_pixels.get(name, 0))
            except (AttributeError, TypeError, ValueError):
                count = 0
            pixels[name] += count
            frames[name] += int(count > 0)
    return {
        "frames_with_class": frames,
        "pixels_by_class": pixels,
    }


def _old_rule_exceptions(records, chunk_name):
    """Inventory frames accepted only after the visibility-rule correction."""
    exceptions = []
    for record in records:
        class_pixels = record.get("label_stats", {}).get("class_pixels", {})
        try:
            level_ml = float(record.get("level_ml"))
            empty_pixels = int(class_pixels.get("empty", 0))
            liquid_pixels = int(class_pixels.get("liquid", 0))
            foam_pixels = int(class_pixels.get("foam", 0))
            blocked_pixels = int(class_pixels.get("blocked", 0))
        except (AttributeError, TypeError, ValueError):
            continue
        failed_old_rules = []
        if level_ml > 0.0 and liquid_pixels == 0:
            failed_old_rules.append("positive_level_without_visible_liquid")
        if empty_pixels == 0:
            failed_old_rules.append("no_visible_empty")
        if not failed_old_rules:
            continue

        foam = record.get("foam", {})
        sticker = record.get("sticker_pose", {})
        factors = []
        if foam_pixels > 0 and bool(foam.get("visible_in_camera", foam.get("enabled"))):
            factors.append("visible_foam_can_occlude_the_liquid_surface")
        try:
            sticker_pixels = int(
                record.get("label_stats", {}).get("sticker_blocked_pixels", 0)
            )
        except (TypeError, ValueError):
            sticker_pixels = 0
        if sticker_pixels > 0 and bool(
            sticker.get("visible_in_camera", sticker.get("present"))
        ):
            factors.append("visible_sticker_blocks_part_of_the_vial")
        projection = record.get("holder_projection", {})
        hidden_range = projection.get("twin_hidden_ml")
        if (
            isinstance(hidden_range, list)
            and len(hidden_range) == 2
            and float(hidden_range[0]) <= level_ml <= float(hidden_range[1])
        ):
            factors.append("surface_is_inside_projected_holder_hidden_range")
        if level_ml < 1.0 and blocked_pixels > 0:
            factors.append("sub_ml_volume_can_be_below_the_visible_base_edge")
        if empty_pixels == 0 and level_ml >= 70.0:
            factors.append("high_fill_can_cover_the_visible_empty_column")

        exceptions.append(
            {
                "chunk": chunk_name,
                "frame_index": record.get("frame_index"),
                "seed": record.get("seed"),
                "role": record.get("role"),
                "level_ml": level_ml,
                "sampling_band": record.get("level_sampling_band"),
                "failed_old_rules": failed_old_rules,
                "class_pixels": {
                    "empty": empty_pixels,
                    "liquid": liquid_pixels,
                    "foam": foam_pixels,
                    "blocked": blocked_pixels,
                },
                "foam": {
                    "enabled": bool(foam.get("enabled")),
                    "visible_in_camera": bool(
                        foam.get("visible_in_camera", foam.get("enabled"))
                    ),
                    "thickness_mm": foam.get("thickness_mm"),
                },
                "sticker": {
                    "present": bool(sticker.get("present")),
                    "visible_in_camera": bool(
                        sticker.get("visible_in_camera", sticker.get("present"))
                    ),
                    "blocked_pixels": sticker_pixels,
                },
                "holder_projection": {
                    "twin_hidden_ml": projection.get("twin_hidden_ml"),
                    "twin_edge_ml": projection.get("twin_edge_ml"),
                },
                "candidate_visibility_factors": factors,
                "manual_disposition": "pending_step7_review",
            }
        )
    return exceptions


def _complete_attempt(chunk_dir):
    attempts = sorted(path for path in chunk_dir.glob("attempt-*") if path.is_dir())
    complete = []
    for attempt in attempts:
        state_path = attempt / "state.txt"
        if state_path.is_file() and state_path.read_text().strip() == "COMPLETE":
            complete.append(attempt)
    if len(complete) != 1:
        return None, (
            f"{chunk_dir.name} requires exactly one COMPLETE attempt; "
            f"found {len(complete)}"
        )
    return complete[0], None


def _validate_summary(
    capture_summary,
    records,
    manifest_path,
    expected_renderer_hash,
    expected_variation_hash,
    expected_specs_hash,
    expected_camera_match_hash,
    expected_source_camera_hash,
    expected_station_tree_hashes,
):
    errors = []
    manifest_hash = file_sha256(manifest_path)
    if not capture_summary.get("passed"):
        errors.append("render summary passed is not true")
    if not capture_summary.get("integrity_passed"):
        errors.append("render summary integrity_passed is not true")
    if not capture_summary.get("source_unchanged"):
        errors.append("source_unchanged is not true")
    if capture_summary.get("frame_count") != len(records):
        errors.append("render summary frame_count does not match manifest")
    if capture_summary.get("manifest_sha256") != manifest_hash:
        errors.append("render summary manifest_sha256 does not match manifest")
    if capture_summary.get("renderer") != EXPECTED_RENDERER:
        errors.append(f"renderer is not {EXPECTED_RENDERER}")
    if capture_summary.get("subframes") != EXPECTED_SUBFRAMES:
        errors.append(f"subframes is not {EXPECTED_SUBFRAMES}")
    if capture_summary.get("resolution") != EXPECTED_RESOLUTION:
        errors.append(f"resolution is not {EXPECTED_RESOLUTION}")
    if capture_summary.get("frame") != EXPECTED_FRAME:
        errors.append(f"frame is not {EXPECTED_FRAME}")
    renderer_hash = capture_summary.get("renderer_script_sha256")
    if renderer_hash != expected_renderer_hash:
        errors.append(
            f"renderer hash {renderer_hash} does not match chunk policy "
            f"{expected_renderer_hash}"
        )
    variation_hash = capture_summary.get("variation_module_sha256")
    if variation_hash != expected_variation_hash:
        errors.append(f"variation hash is not expected: {variation_hash}")
    if capture_summary.get("specs_sha256") != expected_specs_hash:
        errors.append("render summary specs_sha256 does not match chunk specs")
    if capture_summary.get("camera_match_sha256") != expected_camera_match_hash:
        errors.append("camera match hash does not match approved input")
    if (
        capture_summary.get("source_camera_json_sha256_before")
        != expected_source_camera_hash
    ):
        errors.append("source camera hash does not match approved input")
    if (
        capture_summary.get("source_station_tree_hashes_before")
        != expected_station_tree_hashes
    ):
        errors.append("source station tree hashes do not match approved input")
    if (
        capture_summary.get("source_camera_json_sha256_before")
        != capture_summary.get("source_camera_json_sha256_after")
    ):
        errors.append("source camera hash changed during render")
    if (
        capture_summary.get("source_station_tree_hashes_before")
        != capture_summary.get("source_station_tree_hashes_after")
    ):
        errors.append("source station tree hashes changed during render")
    raw_seconds = capture_summary.get("seconds")
    try:
        seconds = float(raw_seconds)
    except (TypeError, ValueError):
        seconds = None
    if seconds is None or not math.isfinite(seconds) or seconds <= 0.0:
        errors.append("render seconds must be finite and positive")
        seconds = None
    return errors, manifest_hash, seconds


def build_report(
    collected_root,
    specs_root,
    renderer_policy=DEFAULT_RENDERER_POLICY,
    expected_variation_hash=EXPECTED_VARIATION_SHA256,
    expected_spec_tree_hash=SPEC_TREE_SHA256,
    expected_camera_match_hash=EXPECTED_CAMERA_MATCH_SHA256,
    expected_source_camera_hash=EXPECTED_SOURCE_CAMERA_SHA256,
    expected_station_tree_hashes=EXPECTED_STATION_TREE_HASHES,
    validate_manifest=None,
):
    """Return a small report while rerunning every copied manifest check."""
    collected_root = Path(collected_root).expanduser().resolve()
    specs_root = Path(specs_root).expanduser().resolve()
    renderer_policy = tuple(renderer_policy)
    validate_manifest = validate_manifest or capture_checker.validate_manifest
    errors = []
    if not collected_root.is_dir():
        return {
            "schema_version": 1,
            "purpose": "FillCam Step 6 full-batch integrity summary",
            "integrity": {
                "passed": False,
                "errors": [f"collected root does not exist: {collected_root}"],
            },
        }
    if not specs_root.is_dir():
        return {
            "schema_version": 1,
            "purpose": "FillCam Step 6 full-batch integrity summary",
            "integrity": {
                "passed": False,
                "errors": [f"specs root does not exist: {specs_root}"],
            },
        }

    measured_spec_tree_hash = spec_tree_sha256(specs_root)
    if measured_spec_tree_hash != expected_spec_tree_hash:
        errors.append(
            f"spec tree hash {measured_spec_tree_hash} does not match "
            f"{expected_spec_tree_hash}"
        )

    expected_names = {item["chunk"] for item in BATCH_PLAN}
    actual_names = {
        path.name for path in collected_root.iterdir() if path.is_dir()
    }
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        errors.append(f"missing chunk directories: {missing}")
    if unexpected:
        errors.append(f"unexpected chunk directories: {unexpected}")

    chunks = []
    all_records = []
    total_seconds = 0.0
    observed_renderer_hashes = set()
    observed_variation_hashes = set()
    observed_camera_hashes = set()
    observed_source_camera_hashes = set()
    observed_station_tree_hashes = set()
    role_cases = {role: {} for role in EXPECTED_COUNTS}
    total_cases = {}
    old_rule_exceptions = []

    for contract in BATCH_PLAN:
        chunk_name = contract["chunk"]
        chunk_dir = collected_root / chunk_name
        if not chunk_dir.is_dir():
            continue
        chunk_errors = []
        attempt, attempt_error = _complete_attempt(chunk_dir)
        if attempt_error:
            errors.append(attempt_error)
            continue
        output = attempt / "output"
        manifest_path = output / "manifest.jsonl"
        summary_path = output / "summary.json"
        remote_check_path = attempt / "check.json"
        specs_path = specs_root / f"{chunk_name}.json"
        for description, path in (
            ("manifest", manifest_path),
            ("render summary", summary_path),
            ("remote checker result", remote_check_path),
            ("chunk specs", specs_path),
        ):
            if not path.is_file():
                chunk_errors.append(f"missing {description}: {path}")
        if chunk_errors:
            errors.extend(f"{chunk_name}: {error}" for error in chunk_errors)
            continue
        try:
            records = _read_records(manifest_path)
            specs = _read_specs(specs_path)
            capture_summary = _read_json(summary_path, "render summary")
            remote_check = _read_json(remote_check_path, "remote checker result")
        except ValueError as error:
            errors.append(f"{chunk_name}: {error}")
            continue

        check_result = validate_manifest(manifest_path)
        if not check_result.get("passed"):
            chunk_errors.append("offline checker did not pass")
            chunk_errors.extend(
                f"offline checker: {error}"
                for error in check_result.get("errors", [])
            )
        if check_result.get("record_count") != len(records):
            chunk_errors.append("offline checker record_count does not match manifest")
        if not remote_check.get("passed"):
            chunk_errors.append("remote checker result did not pass")
        if remote_check.get("record_count") != len(records):
            chunk_errors.append("remote checker record_count does not match manifest")

        expected_renderer_hash = _renderer_hash_for_chunk(
            chunk_name, renderer_policy
        )
        specs_hash = file_sha256(specs_path)
        summary_errors, manifest_hash, seconds = _validate_summary(
            capture_summary,
            records,
            manifest_path,
            expected_renderer_hash,
            expected_variation_hash,
            specs_hash,
            expected_camera_match_hash,
            expected_source_camera_hash,
            expected_station_tree_hashes,
        )
        chunk_errors.extend(summary_errors)
        expected_seeds = list(
            range(contract["seed_start"], contract["seed_end"] + 1)
        )
        roles = [record.get("role") for record in records]
        seeds = [record.get("seed") for record in records]
        if roles != [contract["role"]] * len(expected_seeds):
            chunk_errors.append("manifest roles do not match chunk contract")
        if seeds != expected_seeds:
            chunk_errors.append("manifest seeds do not match chunk contract")
        if len(specs) != len(records):
            chunk_errors.append("spec count does not match manifest record count")
        elif not all(
            _contains_spec(record, spec)
            for record, spec in zip(records, specs)
        ):
            chunk_errors.append("manifest records do not preserve their input specs")

        renderer_hash = capture_summary.get("renderer_script_sha256")
        variation_hash = capture_summary.get("variation_module_sha256")
        camera_hash = capture_summary.get("camera_match_sha256")
        source_camera_hash = capture_summary.get(
            "source_camera_json_sha256_before"
        )
        station_tree_hashes = capture_summary.get(
            "source_station_tree_hashes_before"
        )
        observed_renderer_hashes.add(renderer_hash)
        observed_variation_hashes.add(variation_hash)
        observed_camera_hashes.add(camera_hash)
        observed_source_camera_hashes.add(source_camera_hash)
        observed_station_tree_hashes.add(
            json.dumps(station_tree_hashes, sort_keys=True)
        )
        if seconds is not None:
            total_seconds += seconds
        cases = _case_counts(records)
        chunk_old_rule_exceptions = _old_rule_exceptions(records, chunk_name)
        old_rule_exceptions.extend(chunk_old_rule_exceptions)
        _add_counts(role_cases[contract["role"]], cases)
        _add_counts(total_cases, cases)
        all_records.extend(records)
        chunks.append(
            {
                "chunk": chunk_name,
                "attempt": attempt.name,
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
                "render_summary_sha256": file_sha256(summary_path),
                "specs_sha256": specs_hash,
                "remote_check_sha256": file_sha256(remote_check_path),
                "renderer_script_sha256": renderer_hash,
                "variation_module_sha256": variation_hash,
                "offline_capture_check_passed": bool(check_result.get("passed")),
                "remote_capture_check_passed": bool(remote_check.get("passed")),
                "case_coverage": cases,
                "old_rule_exception_count": len(chunk_old_rule_exceptions),
                "integrity_passed": not chunk_errors,
                "errors": chunk_errors,
            }
        )
        errors.extend(f"{chunk_name}: {error}" for error in chunk_errors)

    seed_counts = Counter(record.get("seed") for record in all_records)
    duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count > 1)
    if duplicate_seeds:
        errors.append(f"duplicate seeds across manifests: {duplicate_seeds}")
    used_forbidden = sorted(set(seed_counts) & FORBIDDEN_SEEDS)
    if used_forbidden:
        errors.append(f"forbidden pilot/held-out seeds used: {used_forbidden}")
    role_counts = Counter(record.get("role") for record in all_records)
    measured_counts = {role: int(role_counts.get(role, 0)) for role in EXPECTED_COUNTS}
    if measured_counts != EXPECTED_COUNTS:
        errors.append(
            f"role counts {measured_counts} do not match expected {EXPECTED_COUNTS}"
        )
    unexpected_roles = sorted(set(role_counts) - set(EXPECTED_COUNTS), key=str)
    if unexpected_roles:
        errors.append(f"unexpected roles: {unexpected_roles}")
    if observed_variation_hashes != {expected_variation_hash}:
        errors.append(
            "observed variation hashes do not equal the approved variation hash"
        )
    if len(observed_camera_hashes) != 1 or None in observed_camera_hashes:
        errors.append("camera match hash is not one consistent non-null value")
    if (
        len(observed_source_camera_hashes) != 1
        or None in observed_source_camera_hashes
    ):
        errors.append("source camera hash is not one consistent non-null value")
    if len(observed_station_tree_hashes) != 1 or "null" in observed_station_tree_hashes:
        errors.append("source station tree hashes are not one consistent value")
    approved_renderer_hashes = {item["sha256"] for item in renderer_policy}
    if not observed_renderer_hashes.issubset(approved_renderer_hashes):
        errors.append("observed renderer hashes include an unapproved value")

    return {
        "schema_version": 1,
        "purpose": "FillCam Step 6 full-batch integrity summary",
        "evidence_label": "measured synthetic render batch",
        "batch_contract": {
            "expected_chunks": len(BATCH_PLAN),
            "expected_role_counts": EXPECTED_COUNTS,
            "expected_seed_ranges_inclusive": {
                "train": [100_000, 102_999],
                "val": [1_100_000, 1_100_299],
                "test": [2_100_000, 2_100_499],
            },
            "expected_renderer": EXPECTED_RENDERER,
            "expected_subframes": EXPECTED_SUBFRAMES,
            "expected_resolution": EXPECTED_RESOLUTION,
            "expected_frame": EXPECTED_FRAME,
            "renderer_policy": list(renderer_policy),
            "expected_variation_module_sha256": expected_variation_hash,
            "expected_camera_match_sha256": expected_camera_match_hash,
            "expected_source_camera_sha256": expected_source_camera_hash,
            "expected_station_tree_hashes": expected_station_tree_hashes,
            "forbidden_seeds": sorted(FORBIDDEN_SEEDS),
        },
        "progress": {
            "complete_chunks": len(chunks),
            "expected_chunks": len(BATCH_PLAN),
            "frames": len(all_records),
            "expected_frames": sum(EXPECTED_COUNTS.values()),
            "role_counts": measured_counts,
        },
        "provenance": {
            "offline_checker_sha256": file_sha256(capture_checker.__file__),
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
                json.loads(next(iter(observed_station_tree_hashes)))
                if len(observed_station_tree_hashes) == 1
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
        "case_coverage": {
            "total": total_cases,
            "by_role": role_cases,
        },
        "label_coverage": _label_coverage(all_records),
        "old_rule_review": {
            "exception_count": len(old_rule_exceptions),
            "status": "pending_manual_step7_review",
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
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"report output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return path


def parse_renderer_policy(text):
    span, separator, sha256 = str(text).partition("=")
    start_text, dash, end_text = span.partition("-")
    if not separator or not dash:
        raise argparse.ArgumentTypeError("renderer policy uses START-END=SHA256")
    try:
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "renderer policy chunk bounds must be integers"
        ) from error
    if start < 0 or end < start or len(sha256) != 64:
        raise argparse.ArgumentTypeError(
            "renderer policy requires 0 <= START <= END and a 64-character SHA-256"
        )
    return {"chunk_start": start, "chunk_end": end, "sha256": sha256.lower()}


def parse_station_hash(text):
    source_path, separator, sha256 = str(text).partition("=")
    if not separator or not source_path or len(sha256) != 64:
        raise argparse.ArgumentTypeError(
            "source station hash uses RELATIVE_PATH=SHA256"
        )
    try:
        int(sha256, 16)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "source station SHA-256 must contain hexadecimal characters"
        ) from error
    return source_path, sha256.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--specs-root", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument(
        "--renderer-policy",
        type=parse_renderer_policy,
        action="append",
        help="repeat START-END=SHA256 to override the approved chunk policy",
    )
    parser.add_argument(
        "--variation-sha256",
        default=EXPECTED_VARIATION_SHA256,
    )
    parser.add_argument("--spec-tree-sha256", default=SPEC_TREE_SHA256)
    parser.add_argument(
        "--camera-match-sha256",
        default=EXPECTED_CAMERA_MATCH_SHA256,
    )
    parser.add_argument(
        "--source-camera-sha256",
        default=EXPECTED_SOURCE_CAMERA_SHA256,
    )
    parser.add_argument(
        "--source-station-sha256",
        type=parse_station_hash,
        action="append",
        dest="source_station_hashes",
        help=(
            "repeat RELATIVE_PATH=SHA256 to override the approved source-station "
            "tree"
        ),
    )
    args = parser.parse_args()
    report = build_report(
        args.collected_root,
        args.specs_root,
        renderer_policy=args.renderer_policy or DEFAULT_RENDERER_POLICY,
        expected_variation_hash=args.variation_sha256,
        expected_spec_tree_hash=args.spec_tree_sha256,
        expected_camera_match_hash=args.camera_match_sha256,
        expected_source_camera_hash=args.source_camera_sha256,
        expected_station_tree_hashes=(
            dict(args.source_station_hashes)
            if args.source_station_hashes
            else EXPECTED_STATION_TREE_HASHES
        ),
    )
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

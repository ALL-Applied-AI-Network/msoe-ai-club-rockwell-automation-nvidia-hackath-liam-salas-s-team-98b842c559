"""CPU-only checks for the Step 6 full-batch integrity summary."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_fill_level_batch.py"
SPEC = importlib.util.spec_from_file_location("summarize_fill_level_batch", SCRIPT)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


CAMERA_HASH = "c" * 64
SOURCE_HASH = "d" * 64


def _fake_checker(manifest_path):
    count = sum(bool(line.strip()) for line in Path(manifest_path).read_text().splitlines())
    return {
        "passed": True,
        "errors": [],
        "record_count": count,
    }


def _write_batch(
    tmp_path,
    *,
    skip_chunk=None,
    duplicate_seed_chunk=None,
    bad_renderer_chunk=None,
    old_rule_exception_chunk=None,
):
    collected = tmp_path / "collected"
    specs_root = tmp_path / "specs"
    collected.mkdir()
    specs_root.mkdir()
    for contract in BATCH.BATCH_PLAN:
        chunk = contract["chunk"]
        specs = []
        records = []
        for offset, seed in enumerate(
            range(contract["seed_start"], contract["seed_end"] + 1)
        ):
            foam = offset % 2 == 0
            sticker = offset % 3 == 0
            sampling_band = (
                "twin_hidden_48_1_66_5"
                if offset % 4 == 0
                else "full_range_0_80"
            )
            spec = {
                "schema_version": 1,
                "seed": seed,
                "role": contract["role"],
                "level_ml": 0.0 if offset % 10 == 0 else 50.0,
                "level_sampling_band": sampling_band,
                "foam": {"enabled": foam, "thickness_mm": 4.0 if foam else 0.0},
                "sticker_pose": {"present": sticker},
                "wet_walls": {"enabled": offset % 5 == 0},
            }
            record = deepcopy(spec)
            record["foam"]["visible_in_camera"] = foam
            record["sticker_pose"]["visible_in_camera"] = sticker
            record["label_stats"] = {
                "class_pixels": {
                    "empty": 10,
                    "liquid": int(record["level_ml"] > 0),
                    "foam": int(foam),
                    "blocked": 2,
                    "ignore": 100,
                }
            }
            specs.append(spec)
            records.append(record)
        if chunk == duplicate_seed_chunk:
            records[0]["seed"] = BATCH.BATCH_PLAN[2]["seed_start"]
        if chunk == old_rule_exception_chunk:
            records[1]["label_stats"]["class_pixels"]["empty"] = 0
            records[1]["label_stats"]["class_pixels"]["liquid"] = 0

        specs_path = specs_root / f"{chunk}.json"
        specs_path.write_text(json.dumps(specs, sort_keys=True) + "\n")
        if chunk == skip_chunk:
            continue
        attempt = collected / chunk / "attempt-01"
        output = attempt / "output"
        output.mkdir(parents=True)
        (attempt / "state.txt").write_text("COMPLETE\n")
        manifest = output / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        renderer_hash = BATCH._renderer_hash_for_chunk(
            chunk, BATCH.DEFAULT_RENDERER_POLICY
        )
        if chunk == bad_renderer_chunk:
            renderer_hash = "f" * 64
        summary = {
            "passed": True,
            "integrity_passed": True,
            "source_unchanged": True,
            "frame_count": len(records),
            "manifest_sha256": BATCH.file_sha256(manifest),
            "renderer": BATCH.EXPECTED_RENDERER,
            "subframes": BATCH.EXPECTED_SUBFRAMES,
            "resolution": BATCH.EXPECTED_RESOLUTION,
            "frame": BATCH.EXPECTED_FRAME,
            "renderer_script_sha256": renderer_hash,
            "variation_module_sha256": BATCH.EXPECTED_VARIATION_SHA256,
            "specs_sha256": BATCH.file_sha256(specs_path),
            "source_camera_json_sha256_before": SOURCE_HASH,
            "source_camera_json_sha256_after": SOURCE_HASH,
            "source_station_tree_hashes_before": {"station.usd": SOURCE_HASH},
            "source_station_tree_hashes_after": {"station.usd": SOURCE_HASH},
            "camera_match_sha256": CAMERA_HASH,
            "seconds": float(len(records)),
        }
        (output / "summary.json").write_text(json.dumps(summary) + "\n")
        (attempt / "check.json").write_text(
            json.dumps({"passed": True, "record_count": len(records)}) + "\n"
        )
    return collected, specs_root


def _report(collected, specs_root, checker=_fake_checker):
    return BATCH.build_report(
        collected,
        specs_root,
        expected_spec_tree_hash=BATCH.spec_tree_sha256(specs_root),
        expected_camera_match_hash=CAMERA_HASH,
        expected_source_camera_hash=SOURCE_HASH,
        expected_station_tree_hashes={"station.usd": SOURCE_HASH},
        validate_manifest=checker,
    )


def test_valid_full_batch_passes_with_chunk_specific_renderer_policy(tmp_path):
    collected, specs_root = _write_batch(tmp_path)
    report = _report(collected, specs_root)
    assert report["integrity"]["passed"]
    assert report["progress"] == {
        "complete_chunks": 20,
        "expected_chunks": 20,
        "frames": 3800,
        "expected_frames": 3800,
        "role_counts": {"train": 3000, "val": 300, "test": 500},
    }
    assert report["provenance"]["observed_renderer_script_sha256"] == sorted(
        BATCH.APPROVED_RENDERER_SHA256
    )
    assert report["case_coverage"]["total"]["foam"] == 1900


def test_duplicate_or_missing_chunk_fails_global_contract(tmp_path):
    collected, specs_root = _write_batch(
        tmp_path,
        skip_chunk=BATCH.BATCH_PLAN[-1]["chunk"],
        duplicate_seed_chunk=BATCH.BATCH_PLAN[3]["chunk"],
    )
    report = _report(collected, specs_root)
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "missing chunk directories" in errors
    assert "duplicate seeds across manifests" in errors
    assert "role counts" in errors


def test_wrong_renderer_policy_and_multiple_complete_attempts_fail(tmp_path):
    bad_chunk = BATCH.BATCH_PLAN[3]["chunk"]
    collected, specs_root = _write_batch(
        tmp_path,
        bad_renderer_chunk=bad_chunk,
    )
    extra = collected / BATCH.BATCH_PLAN[4]["chunk"] / "attempt-99"
    extra.mkdir()
    (extra / "state.txt").write_text("COMPLETE\n")
    report = _report(collected, specs_root)
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "does not match chunk policy" in errors
    assert "requires exactly one COMPLETE attempt" in errors


def test_spec_tamper_and_offline_checker_failure_are_reported(tmp_path):
    collected, specs_root = _write_batch(tmp_path)
    first_specs = specs_root / f"{BATCH.BATCH_PLAN[0]['chunk']}.json"
    specs = json.loads(first_specs.read_text())
    specs[0]["level_ml"] = 12.5
    first_specs.write_text(json.dumps(specs) + "\n")

    def failing_checker(manifest_path):
        result = _fake_checker(manifest_path)
        result.update({"passed": False, "errors": ["tampered mask"]})
        return result

    report = _report(collected, specs_root, failing_checker)
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "specs_sha256" in errors
    assert "do not preserve their input specs" in errors
    assert "offline checker: tampered mask" in errors


def test_consistent_but_unapproved_camera_and_source_hashes_fail(tmp_path):
    collected, specs_root = _write_batch(tmp_path)
    for summary_path in collected.glob("*/attempt-*/output/summary.json"):
        summary = json.loads(summary_path.read_text())
        summary["camera_match_sha256"] = "0" * 64
        summary["source_camera_json_sha256_before"] = "1" * 64
        summary["source_camera_json_sha256_after"] = "1" * 64
        summary["source_station_tree_hashes_before"] = {
            "station.usd": "2" * 64
        }
        summary["source_station_tree_hashes_after"] = {
            "station.usd": "2" * 64
        }
        summary_path.write_text(json.dumps(summary) + "\n")
    report = _report(collected, specs_root)
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "camera match hash does not match approved input" in errors
    assert "source camera hash does not match approved input" in errors
    assert "source station tree hashes do not match approved input" in errors


def test_old_visibility_rule_exceptions_are_fully_inventoried(tmp_path):
    chunk = BATCH.BATCH_PLAN[3]["chunk"]
    collected, specs_root = _write_batch(
        tmp_path,
        old_rule_exception_chunk=chunk,
    )
    report = _report(collected, specs_root)
    assert report["integrity"]["passed"]
    assert report["old_rule_review"]["exception_count"] == 1
    exception = report["old_rule_review"]["exceptions"][0]
    assert exception["chunk"] == chunk
    assert exception["seed"] == BATCH.BATCH_PLAN[3]["seed_start"] + 1
    assert exception["failed_old_rules"] == [
        "positive_level_without_visible_liquid",
        "no_visible_empty",
    ]
    assert exception["manual_disposition"] == "pending_step7_review"


def test_checker_failure_without_detail_still_fails_chunk(tmp_path):
    collected, specs_root = _write_batch(tmp_path)

    def silent_failure(manifest_path):
        result = _fake_checker(manifest_path)
        result.update({"passed": False, "errors": []})
        return result

    report = _report(collected, specs_root, silent_failure)
    assert not report["integrity"]["passed"]
    assert any(
        "offline checker did not pass" in error
        for error in report["integrity"]["errors"]
    )


def test_invalid_render_timing_is_null_and_strict_json(tmp_path):
    collected, specs_root = _write_batch(tmp_path)
    chunk = BATCH.BATCH_PLAN[0]["chunk"]
    summary_path = collected / chunk / "attempt-01" / "output" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["seconds"] = "not-a-number"
    summary_path.write_text(json.dumps(summary) + "\n")
    report = _report(collected, specs_root)
    assert not report["integrity"]["passed"]
    assert report["chunks"][0]["seconds"] is None
    assert "render seconds must be finite and positive" in "\n".join(
        report["integrity"]["errors"]
    )
    json.dumps(report, allow_nan=False)


def test_source_station_hash_cli_parser():
    sha256 = "a" * 64
    assert BATCH.parse_station_hash(f"geometry/station.usdc={sha256}") == (
        "geometry/station.usdc",
        sha256,
    )
    with pytest.raises(BATCH.argparse.ArgumentTypeError, match="RELATIVE_PATH=SHA256"):
        BATCH.parse_station_hash("missing-hash")


def test_report_writer_refuses_to_overwrite(tmp_path):
    target = tmp_path / "report.json"
    BATCH._write_json_new(target, {"passed": True})
    with pytest.raises(FileExistsError, match="already exists"):
        BATCH._write_json_new(target, {"passed": False})

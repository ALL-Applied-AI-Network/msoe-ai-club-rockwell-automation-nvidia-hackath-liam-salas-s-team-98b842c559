"""CPU-only checks for the exact-v4 Revision 2 batch summary."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_fill_level_revision2_v4_batch.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_fill_level_revision2_v4_batch", SCRIPT
)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def _fake_checker(manifest_path):
    count = sum(
        bool(line.strip()) for line in Path(manifest_path).read_text().splitlines()
    )
    return {"passed": True, "errors": [], "record_count": count}


def _write_fixture(
    tmp_path,
    *,
    direct_first=True,
    direct_chunk=None,
    exception_chunks=(),
):
    input_root = tmp_path / "input"
    specs_root = input_root / "specs"
    collected = tmp_path / "collected"
    specs_root.mkdir(parents=True)
    collected.mkdir()
    for relative in BATCH.EXPECTED_INPUT_FILE_HASHES:
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}\n".encode())
    expected_input_hashes = {
        relative: BATCH.base.file_sha256(input_root / relative)
        for relative in BATCH.EXPECTED_INPUT_FILE_HASHES
    }

    for contract_index, contract in enumerate(BATCH.BATCH_PLAN):
        chunk = contract["chunk"]
        specs = []
        records = []
        for offset, seed in enumerate(
            range(contract["seed_start"], contract["seed_end"] + 1)
        ):
            foam = offset % 5 == 0
            sticker = offset % 7 == 0
            spec = {
                "schema_version": 1,
                "seed": seed,
                "role": contract["role"],
                "level_ml": 0.0 if offset == 0 else 50.0,
                "level_sampling_band": (
                    "twin_hidden_48_1_66_5" if offset % 4 == 0 else "full_range_0_80"
                ),
                "foam": {"enabled": foam, "thickness_mm": 4.0 if foam else 0.0},
                "sticker_pose": {"present": sticker},
                "wet_walls": {"enabled": offset % 3 == 0},
            }
            record = deepcopy(spec)
            record["frame_index"] = offset
            record["foam"]["visible_in_camera"] = foam
            record["sticker_pose"]["visible_in_camera"] = sticker
            record["label_stats"] = {
                "class_pixels": {
                    "empty": 10,
                    "liquid": int(record["level_ml"] > 0),
                    "foam": int(foam),
                    "blocked": 2,
                    "ignore": 100,
                },
                "sticker_blocked_pixels": int(sticker),
            }
            record["outputs"] = {
                "rgb": {"path": f"rgb/{offset:05d}.png"},
                "mask": {"path": f"mask/{offset:05d}.png"},
            }
            specs.append(spec)
            records.append(record)
        if chunk in exception_chunks:
            records[1]["label_stats"]["class_pixels"]["empty"] = 0
            records[1]["label_stats"]["class_pixels"]["liquid"] = 0

        specs_path = specs_root / f"{chunk}.json"
        specs_path.write_text(json.dumps(specs, sort_keys=True) + "\n")
        chunk_dir = collected / chunk
        use_direct = (direct_first and contract_index == 0) or chunk == direct_chunk
        attempt = (
            chunk_dir if use_direct else chunk_dir / contract["attempt"]
        )
        output = attempt / "output"
        (output / "rgb").mkdir(parents=True)
        (output / "mask").mkdir()
        for record in records:
            for kind in ("rgb", "mask"):
                item = record["outputs"][kind]
                path = output / item["path"]
                path.write_bytes(
                    f"{kind}:{record['role']}:{record['seed']}\n".encode()
                )
                item["sha256"] = BATCH.base.file_sha256(path)
        manifest = output / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        summary = {
            "passed": True,
            "integrity_passed": True,
            "source_unchanged": True,
            "frame_count": 60,
            "manifest_sha256": BATCH.base.file_sha256(manifest),
            "renderer": BATCH.base.EXPECTED_RENDERER,
            "subframes": BATCH.base.EXPECTED_SUBFRAMES,
            "resolution": BATCH.base.EXPECTED_RESOLUTION,
            "frame": BATCH.base.EXPECTED_FRAME,
            "renderer_script_sha256": contract["renderer_sha256"],
            "variation_module_sha256": BATCH.EXPECTED_VARIATION_SHA256,
            "specs_sha256": BATCH.base.file_sha256(specs_path),
            "camera_match_sha256": BATCH.EXPECTED_CAMERA_MATCH_SHA256,
            "source_camera_json_sha256_before": BATCH.EXPECTED_SOURCE_CAMERA_SHA256,
            "source_camera_json_sha256_after": BATCH.EXPECTED_SOURCE_CAMERA_SHA256,
            "source_station_tree_hashes_before": BATCH.EXPECTED_STATION_TREE_HASHES,
            "source_station_tree_hashes_after": BATCH.EXPECTED_STATION_TREE_HASHES,
            "seconds": 60.0,
        }
        (output / "summary.json").write_text(json.dumps(summary) + "\n")
        (attempt / "state.txt").write_text("COMPLETE\n")
        (attempt / "check.json").write_text(
            json.dumps({"passed": True, "errors": [], "record_count": 60}) + "\n"
        )
        (attempt / "check.stderr").write_text("")
        (attempt / "render.log").write_text("fixture render log\n")
    return collected, input_root, expected_input_hashes


def _report(monkeypatch, collected, input_root, expected_input_hashes):
    monkeypatch.setattr(
        BATCH, "EXPECTED_INPUT_FILE_HASHES", expected_input_hashes
    )
    monkeypatch.setattr(
        BATCH,
        "EXPECTED_SPEC_TREE_SHA256",
        BATCH.base.spec_tree_sha256(input_root / "specs"),
    )
    return BATCH.build_report(
        collected, input_root, validate_manifest=_fake_checker
    )


def test_pinned_production_contract_is_exact():
    assert BATCH.EXPECTED_COUNTS == {"train": 480, "val": 60, "test": 60}
    assert BATCH.EXPECTED_SEED_RANGES == {
        "train": [104_000, 104_479],
        "val": [1_101_000, 1_101_059],
        "test": [2_101_000, 2_101_059],
    }
    assert BATCH.EXPECTED_INPUT_FILE_HASHES[
        "render_fill_level_dataset_r2_v4_batch.py"
    ] == BATCH.EXPECTED_RENDERER_SHA256_V2
    assert BATCH.EXPECTED_INPUT_FILE_HASHES[
        "check_fill_level_capture_r2_v4_batch.py"
    ] == BATCH.EXPECTED_CHECKER_SHA256
    assert BATCH.EXPECTED_INPUT_FILE_HASHES[
        "src/fluidmix/fill_level_variation.py"
    ] == BATCH.EXPECTED_VARIATION_SHA256
    assert BATCH.EXPECTED_INPUT_FILE_HASHES[
        "INPUT_SHA256SUMS-r2-v4-batch-v2"
    ] == "42df1780a5aa6c1c4e68f3e797a3869954020736c675f75586a685a1ecdba027"
    assert BATCH.BATCH_PLAN[7]["renderer_sha256"] == (
        BATCH.EXPECTED_RENDERER_SHA256_V1
    )
    assert BATCH.BATCH_PLAN[8]["attempt"] == "attempt-02"
    assert BATCH.BATCH_PLAN[8]["renderer_sha256"] == (
        BATCH.EXPECTED_RENDERER_SHA256_V2
    )
    assert BATCH.BATCH_PLAN[9]["attempt"] == "attempt-01"
    assert BATCH.BATCH_PLAN[9]["renderer_sha256"] == (
        BATCH.EXPECTED_RENDERER_SHA256_V2
    )


def test_full_batch_accepts_preserved_direct_chunk_zero_and_excludes_test_exceptions(
    tmp_path, monkeypatch
):
    train_chunk = BATCH.BATCH_PLAN[0]["chunk"]
    test_chunk = BATCH.BATCH_PLAN[-1]["chunk"]
    collected, input_root, input_hashes = _write_fixture(
        tmp_path, exception_chunks=(train_chunk, test_chunk)
    )
    report = _report(monkeypatch, collected, input_root, input_hashes)

    assert report["integrity"]["passed"]
    assert report["progress"] == {
        "complete_chunks": 10,
        "expected_chunks": 10,
        "frames": 600,
        "expected_frames": 600,
        "role_counts": {"train": 480, "val": 60, "test": 60},
    }
    assert report["chunks"][0]["storage_layout"] == "direct_copy_of_attempt-01"
    assert report["chunks"][1]["storage_layout"] == "nested_attempt-01"
    assert report["chunks"][8]["attempt"] == "attempt-02"
    assert report["chunks"][8]["storage_layout"] == "nested_attempt-02"
    assert all(
        chunk["output_file_integrity"]["file_count"] == 120
        for chunk in report["chunks"]
    )
    assert all(
        chunk["output_file_integrity"]["discovered_file_count"] == 120
        for chunk in report["chunks"]
    )
    assert report["sealed_test_policy"] == {
        "rgb_decoded_by_this_summarizer": 0,
        "integrity_only_file_hash_reads": 120,
        "old_rule_exceptions_inventoried": False,
    }
    assert report["old_rule_review"]["exception_count"] == 1
    assert report["old_rule_review"]["exceptions"][0]["chunk"] == train_chunk
    assert report["old_rule_review"]["sealed_test_records_inventoried"] == 0
    assert report["provenance"]["input_file_sha256"] == input_hashes
    assert report["provenance"]["input_contracts"] == {
        "v1": {
            "renderer_sha256": BATCH.EXPECTED_RENDERER_SHA256_V1,
            "runner_sha256": input_hashes["run-step8-r2-v4-batch-v1.sh"],
            "checksum_manifest_sha256": input_hashes[
                "INPUT_SHA256SUMS-r2-v4-batch"
            ],
        },
        "v2": {
            "renderer_sha256": BATCH.EXPECTED_RENDERER_SHA256_V2,
            "runner_sha256": input_hashes[
                "run-step8-r2-v4-batch-retry-v2.sh"
            ],
            "retry_contract_sha256": input_hashes["RETRY-CONTRACT.txt"],
            "checksum_manifest_sha256": input_hashes[
                "INPUT_SHA256SUMS-r2-v4-batch-v2"
            ],
        },
    }


def test_retry_attempt_renderer_policy_ambiguity_and_incomplete_state_fail(
    tmp_path, monkeypatch
):
    collected, input_root, input_hashes = _write_fixture(tmp_path)
    val_chunk = BATCH.BATCH_PLAN[8]["chunk"]
    (collected / val_chunk / "attempt-03").mkdir()
    test_chunk = BATCH.BATCH_PLAN[9]["chunk"]
    (collected / test_chunk / "attempt-01" / "state.txt").write_text(
        "RUNNING\n"
    )
    train_chunk = BATCH.BATCH_PLAN[0]["chunk"]
    summary_path = collected / train_chunk / "output" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["renderer_script_sha256"] = BATCH.EXPECTED_RENDERER_SHA256_V2
    summary_path.write_text(json.dumps(summary) + "\n")

    report = _report(monkeypatch, collected, input_root, input_hashes)

    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "expected only attempt-02" in errors
    assert "attempt state is not COMPLETE" in errors
    assert "does not match chunk policy" in errors


def test_validation_retry_rejects_unapproved_attempt_name(tmp_path, monkeypatch):
    collected, input_root, input_hashes = _write_fixture(tmp_path)
    val_chunk = BATCH.BATCH_PLAN[8]["chunk"]
    (collected / val_chunk / "attempt-02").rename(
        collected / val_chunk / "attempt-01"
    )

    report = _report(monkeypatch, collected, input_root, input_hashes)

    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "expected only attempt-02; found ['attempt-01']" in errors


def test_output_bytes_spec_and_source_immutability_are_independently_checked(
    tmp_path, monkeypatch
):
    collected, input_root, input_hashes = _write_fixture(tmp_path)
    first = BATCH.BATCH_PLAN[0]["chunk"]
    (collected / first / "output" / "rgb" / "00000.png").write_bytes(b"tampered")
    second = BATCH.BATCH_PLAN[1]["chunk"]
    summary_path = collected / second / "attempt-01" / "output" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source_camera_json_sha256_after"] = "f" * 64
    summary_path.write_text(json.dumps(summary) + "\n")
    third = BATCH.BATCH_PLAN[2]["chunk"]
    specs_path = input_root / "specs" / f"{third}.json"
    specs = json.loads(specs_path.read_text())
    specs[0]["level_ml"] = 12.0
    specs_path.write_text(json.dumps(specs) + "\n")

    report = _report(monkeypatch, collected, input_root, input_hashes)
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "rgb output SHA-256 mismatch" in errors
    assert "source camera hash changed during render" in errors
    assert "specs_sha256" in errors
    assert "do not preserve their input specs" in errors


def test_attempt_checker_input_and_layout_tampering_fail(tmp_path, monkeypatch):
    collected, input_root, input_hashes = _write_fixture(
        tmp_path, direct_chunk=BATCH.BATCH_PLAN[1]["chunk"]
    )
    first = BATCH.BATCH_PLAN[0]["chunk"]
    (collected / first / "check.stderr").write_text("checker warning\n")
    (input_root / next(iter(input_hashes))).write_text("changed\n")

    report = _report(monkeypatch, collected, input_root, input_hashes)
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "stored checker stderr is not empty" in errors
    assert "pinned input hash mismatch" in errors
    assert "direct attempt layout is approved only for chunk 00" in errors


def test_unmanifested_rgb_or_mask_file_vetoes_integrity(tmp_path, monkeypatch):
    collected, input_root, input_hashes = _write_fixture(tmp_path)
    first = BATCH.BATCH_PLAN[0]["chunk"]
    (collected / first / "output" / "rgb" / "unmanifested.png").write_bytes(
        b"not declared by any manifest record"
    )

    report = _report(monkeypatch, collected, input_root, input_hashes)

    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "unmanifested rgb output file: rgb/unmanifested.png" in errors
    assert report["chunks"][0]["output_file_integrity"]["file_count"] == 120
    assert report["chunks"][0]["output_file_integrity"][
        "discovered_file_count"
    ] == 121


def test_missing_or_unexpected_chunk_and_offline_checker_failure_are_reported(
    tmp_path, monkeypatch
):
    collected, input_root, input_hashes = _write_fixture(tmp_path)
    missing = collected / BATCH.BATCH_PLAN[-1]["chunk"]
    missing.rename(collected / "99-unexpected")

    def failed_checker(manifest_path):
        result = _fake_checker(manifest_path)
        result.update({"passed": False, "errors": ["synthetic checker veto"]})
        return result

    monkeypatch.setattr(BATCH, "EXPECTED_INPUT_FILE_HASHES", input_hashes)
    monkeypatch.setattr(
        BATCH,
        "EXPECTED_SPEC_TREE_SHA256",
        BATCH.base.spec_tree_sha256(input_root / "specs"),
    )
    report = BATCH.build_report(
        collected, input_root, validate_manifest=failed_checker
    )
    assert not report["integrity"]["passed"]
    errors = "\n".join(report["integrity"]["errors"])
    assert "missing chunk directories" in errors
    assert "unexpected chunk directories" in errors
    assert "offline checker: synthetic checker veto" in errors


def test_report_writer_refuses_overwrite_and_json_is_strict(tmp_path, monkeypatch):
    collected, input_root, input_hashes = _write_fixture(tmp_path)
    report = _report(monkeypatch, collected, input_root, input_hashes)
    json.dumps(report, allow_nan=False)
    path = tmp_path / "report.json"
    BATCH._write_json_new(path, report)
    with pytest.raises(FileExistsError, match="already exists"):
        BATCH._write_json_new(path, report)

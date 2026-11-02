"""CPU-only checks for the Step 7 visibility-exception review."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_fill_level_exceptions.py"
SPEC = importlib.util.spec_from_file_location("exception_review", SCRIPT)
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fixture(tmp_path, seed=100327):
    collected = tmp_path / "collected"
    chunk = "03-train-0100200-0100399"
    attempt = collected / chunk / "attempt-02"
    output = attempt / "output"
    (output / "rgb").mkdir(parents=True)
    (output / "mask").mkdir()
    (attempt / "state.txt").write_text("COMPLETE\n", encoding="utf-8")

    rgb = np.full((12, 16, 3), (70, 90, 120), dtype=np.uint8)
    mask = np.full((12, 16), 255, dtype=np.uint8)
    mask[2:10, 4:12] = 0
    mask[7:9, 5:11] = 2
    mask[4:6, 7:10] = 3
    rgb_path = output / "rgb" / "frame.png"
    mask_path = output / "mask" / "frame.png"
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(mask_path), mask)
    identifiers, counts = np.unique(mask, return_counts=True)
    measured = {int(key): int(value) for key, value in zip(identifiers, counts)}
    class_pixels = {
        name: measured.get(identifier, 0)
        for name, identifier in REVIEW.LABEL_IDS.items()
    }
    record = {
        "seed": seed,
        "role": "train",
        "frame_index": 127,
        "level_ml": 0.06511742022173861,
        "level_sampling_band": "full_0_85",
        "foam": {
            "enabled": True,
            "visible_in_camera": True,
            "thickness_mm": 3.964286312275849,
        },
        "sticker_pose": {"present": True, "visible_in_camera": True},
        "label_stats": {
            "class_pixels": class_pixels,
            "sticker_blocked_pixels": 6,
            "vial_bbox_xyxy": [4, 2, 11, 9],
        },
        "outputs": {
            "rgb": {"path": "rgb/frame.png", "sha256": sha256(rgb_path)},
            "mask": {"path": "mask/frame.png", "sha256": sha256(mask_path)},
        },
    }
    manifest = output / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    exception = {
        "seed": seed,
        "role": "train",
        "chunk": chunk,
        "frame_index": 127,
        "level_ml": record["level_ml"],
        "sampling_band": "full_0_85",
        "failed_old_rules": ["positive_level_without_visible_liquid"],
        "class_pixels": {
            name: class_pixels[name] for name in ("empty", "liquid", "foam", "blocked")
        },
        "foam": record["foam"],
        "sticker": {
            "present": True,
            "visible_in_camera": True,
            "blocked_pixels": 6,
        },
        "holder_projection": {"twin_hidden_ml": [47.4, 65.5]},
    }
    summary = {
        "integrity": {"passed": True},
        "batch_contract": {"expected_role_counts": {"train": 1, "val": 0}},
        "old_rule_review": {"exception_count": 1, "exceptions": [exception]},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, collected, output, mask_path


def test_known_exception_is_verified_and_overlaid(tmp_path):
    summary, collected, _, _ = write_fixture(tmp_path)
    overlay_dir = tmp_path / "overlays"
    report = REVIEW.review_exceptions(summary, collected, overlay_dir)
    assert report["status"] == "complete"
    assert report["excluded_train_validation_seeds"] == []
    assert report["eligible_counts"] == {"train": 1, "val": 0}
    assert report["exceptions"][0]["disposition"] == "accepted_legitimate_occlusion"
    assert len(report["exceptions"][0]["overlays"]) == 2
    for item in report["exceptions"][0]["overlays"]:
        path = overlay_dir / item["filename"]
        assert path.is_file()
        assert sha256(path) == item["sha256"]


def test_unregistered_exception_is_excluded(tmp_path):
    summary, collected, _, _ = write_fixture(tmp_path, seed=999999)
    report = REVIEW.review_exceptions(summary, collected, tmp_path / "overlays")
    assert report["status"] == "complete"
    assert report["excluded_train_validation_seeds"] == [999999]
    assert report["eligible_counts"] == {"train": 0, "val": 0}


def test_tampered_mask_hash_is_rejected(tmp_path):
    summary, collected, _, mask_path = write_fixture(tmp_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask[0, 0] = 0
    assert cv2.imwrite(str(mask_path), mask)
    with pytest.raises(ValueError, match="mask SHA-256"):
        REVIEW.review_exceptions(summary, collected, tmp_path / "overlays")


def test_refuses_to_overwrite_overlay_directory(tmp_path):
    summary, collected, _, _ = write_fixture(tmp_path)
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    with pytest.raises(FileExistsError, match="overlay directory"):
        REVIEW.review_exceptions(summary, collected, overlay_dir)


def test_cli_refuses_to_overwrite_report(tmp_path, monkeypatch):
    summary, collected, _, _ = write_fixture(tmp_path)
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--summary",
            str(summary),
            "--collected-root",
            str(collected),
            "--overlay-dir",
            str(tmp_path / "overlays"),
            "--report-out",
            str(report),
        ],
    )
    with pytest.raises(FileExistsError, match="report"):
        REVIEW.main()

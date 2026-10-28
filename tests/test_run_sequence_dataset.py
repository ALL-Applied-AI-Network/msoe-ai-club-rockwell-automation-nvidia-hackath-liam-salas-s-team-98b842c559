from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fluidmix.physx_varied_cohort_manifest_v3 import (  # noqa: E402
    build_varied_cohort_artifacts,
)
from fluidmix.run_sequence_dataset import (  # noqa: E402
    build_dataset_index_rows,
    build_model_input_rows,
    validate_case_artifacts,
    validate_dataset_index,
)
from package_run_sequence_dataset import stitch_run_video  # noqa: E402


def _recursive_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recursive_keys(child)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fake_case_tree(tmp_path: Path) -> tuple[Path, dict]:
    case_root = tmp_path / "cases" / "CASE_1234567890abcdef"
    app = case_root / "app-output"
    (app / "frames").mkdir(parents=True)
    (app / "liquid").mkdir()
    (app / "frames" / "stage-a.png").write_bytes(b"stage-a")
    (app / "frames" / "stage-b.png").write_bytes(b"stage-b")
    (app / "liquid" / "mask-a.png").write_bytes(b"mask-a")
    (app / "liquid" / "mask-b.png").write_bytes(b"mask-b")
    (app / "three-pour.avi").write_bytes(b"video")
    receipt = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "render": {
            "video_path": "three-pour.avi",
            "video_sha256": _sha256(app / "three-pour.avi"),
            "video_decodable": True,
            "stage_frame_paths": [
                "frames/stage-a.png",
                "frames/stage-b.png",
            ],
            "stage_frame_sha256": [
                _sha256(app / "frames" / "stage-a.png"),
                _sha256(app / "frames" / "stage-b.png"),
            ],
        },
        "semantics": {
            "semantic_gate_passed": True,
            "pairs": [
                {
                    "rgb_path": "frames/stage-a.png",
                    "rgb_sha256": _sha256(app / "frames" / "stage-a.png"),
                    "mask_path": "liquid/mask-a.png",
                    "mask_sha256": _sha256(app / "liquid" / "mask-a.png"),
                },
                {
                    "rgb_path": "frames/stage-b.png",
                    "rgb_sha256": _sha256(app / "frames" / "stage-b.png"),
                    "mask_path": "liquid/mask-b.png",
                    "mask_sha256": _sha256(app / "liquid" / "mask-b.png"),
                },
            ],
        },
    }
    _write_json(app / "receipt.json", receipt)
    result = {
        "case_id": case_root.name,
        "outcome": "passed",
        "return_code": 0,
        "timed_out": False,
        "evidence": {
            "artifact_validation": {
                "status": "pass",
                "failures": [],
                "warnings": [],
            },
            "failures": [],
            "launch_failures": [],
            "receipt_failures": [],
        },
    }
    _write_json(case_root / "case-result.json", result)
    _write_json(case_root / "launch-receipt.json", {"case_id": case_root.name})
    return case_root, receipt


def test_dataset_rows_have_requested_names_balance_and_excluded_metadata():
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    first_case = artifacts["render-schedule-plan.json"]["runs"][0]["case_ids"][0]
    warnings = {
        first_case: [
            {
                "warning_type": "particle_height_outside_geometry_volume_bounds",
                "hard_gate": False,
                "claim": "diagnostic_warning_not_a_physics_pass",
            }
        ]
    }

    rows = build_dataset_index_rows(artifacts, height_warnings_by_case=warnings)

    assert len(rows) == 30
    assert Counter(row["disposition"] for row in rows) == {
        "KEEP": 15,
        "DISCARD": 15,
    }
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["disposition"]] += 1
    assert by_split == {
        "train": {"KEEP": 9, "DISCARD": 9},
        "validation": {"KEEP": 3, "DISCARD": 3},
        "evaluation": {"KEEP": 3, "DISCARD": 3},
    }
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
        assert re.fullmatch(
            r"videos/vials/(train|validation|evaluation)-run\d{2}-"
            r"vial\d{2}-(KEEP|DISCARD)-CASE_[0-9a-f]{16}\.mp4",
            row["per_vial_video_path"],
        )
        assert re.fullmatch(
            r"videos/runs/(train|validation|evaluation)-run\d{2}-"
            r"S\d{4}-S\d{4}-S\d{4}-T200[0-9A-F]{6}-"
            r"T400[0-9A-F]{6}\.mp4",
            row["run_video_path"],
        )
        assert row["metadata_excluded_from_model_inputs"] == "true"
        assert row["simulator_metadata_excluded_from_model_inputs"] == "true"
        assert json.loads(row["residue_composition_t200_t400_t100"])
        assert json.loads(row["palette_t200_rgb"])
        assert json.loads(row["palette_t400_rgb"])
        assert json.loads(row["palette_t100_rgb"]) == [1.0, 1.0, 1.0]
    assert all(len({row["split"] for row in run}) == 1 for run in by_run.values())
    assert all(len(run) == 6 for run in by_run.values())
    assert rows[0]["height_warning_count"] == "1"
    assert rows[0]["height_warnings_json"] != "[]"
    assert all(row["height_warning_count"] == "0" for row in rows[1:])


def test_model_input_rows_are_label_free_and_use_opaque_video_paths():
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    index_rows = build_dataset_index_rows(artifacts)

    model_rows = build_model_input_rows(artifacts, index_rows=index_rows)

    assert len(model_rows) == 30
    forbidden = (
        "label",
        "residual",
        "palette",
        "particle",
        "disposition",
        "warning",
        "oracle",
        "keep",
        "discard",
    )
    assert all(
        not any(fragment in key.lower() for fragment in forbidden)
        for row in model_rows
        for key in _recursive_keys(row)
    )
    assert all(
        re.fullmatch(r"model-inputs/videos/CASE_[0-9a-f]{16}\.mp4", row["video_path"])
        for row in model_rows
    )
    assert all(
        "KEEP" not in row["video_path"] and "DISCARD" not in row["video_path"]
        for row in model_rows
    )


def test_dataset_validation_fails_on_imbalance_split_crossing_or_missing_paths(
    tmp_path: Path,
):
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    rows = build_dataset_index_rows(artifacts)

    validate_dataset_index(rows)

    unbalanced = deepcopy(rows)
    unbalanced[0]["disposition"] = unbalanced[1]["disposition"]
    with pytest.raises(ValueError, match="balanced"):
        validate_dataset_index(unbalanced)

    crossed = deepcopy(rows)
    crossed[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="split"):
        validate_dataset_index(crossed)

    with pytest.raises(ValueError, match="missing package path"):
        validate_dataset_index(rows, package_root=tmp_path)


def test_case_validation_checks_hard_gates_hashes_and_unique_stage_rgb(
    tmp_path: Path,
):
    case_root, receipt = _fake_case_tree(tmp_path)

    evidence = validate_case_artifacts(case_root)

    assert evidence["case_id"] == case_root.name
    assert evidence["stage_rgb_count"] == 2
    assert evidence["stage_rgb_byte_unique"] is True
    assert evidence["hard_gates_passed"] is True

    duplicate = deepcopy(receipt)
    duplicate["render"]["stage_frame_sha256"][1] = duplicate["render"][
        "stage_frame_sha256"
    ][0]
    _write_json(case_root / "app-output" / "receipt.json", duplicate)
    with pytest.raises(ValueError, match="stage RGB captures are not byte-unique"):
        validate_case_artifacts(case_root)

    _write_json(case_root / "app-output" / "receipt.json", receipt)
    (case_root / "app-output" / "frames" / "stage-a.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        validate_case_artifacts(case_root)


def test_case_validation_only_allows_failed_legacy_result_with_bound_height_warning(
    tmp_path: Path,
):
    case_root, _ = _fake_case_tree(tmp_path)
    result_path = case_root / "case-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["outcome"] = "failed"
    result["evidence"]["artifact_validation"] = {
        "status": "insufficient_evidence",
        "failures": ["receipt:realtime_volume_height_gate"],
    }
    result["evidence"]["failures"] = ["receipt:realtime_volume_height_gate"]
    _write_json(result_path, result)
    warning = {
        "warning_type": "particle_height_outside_geometry_volume_bounds",
        "hard_gate": False,
        "claim": "diagnostic_warning_not_a_physics_pass",
    }

    with pytest.raises(ValueError, match="completed pass"):
        validate_case_artifacts(case_root)

    evidence = validate_case_artifacts(case_root, height_warnings=[warning])
    assert evidence["height_warnings"] == [warning]
    assert evidence["legacy_height_warning_revalidated"] is True


def test_run_video_stitch_preserves_every_input_frame_and_refuses_wrong_run_size(
    tmp_path: Path,
):
    inputs = []
    for clip_index in range(6):
        path = tmp_path / f"clip-{clip_index}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
        )
        assert writer.isOpened()
        for frame_index in range(3):
            frame = np.full(
                (48, 64, 3),
                (clip_index * 30, frame_index * 50, 100),
                dtype=np.uint8,
            )
            writer.write(frame)
        writer.release()
        inputs.append(path)

    receipt = stitch_run_video(inputs, tmp_path / "run.mp4")

    assert receipt["status"] == "pass"
    assert receipt["input_frame_counts"] == [3] * 6
    assert receipt["output_frame_count"] == 18
    assert receipt["fps"] == 10.0
    assert receipt["resolution"] == [64, 48]

    with pytest.raises(ValueError, match="exactly six"):
        stitch_run_video(inputs[:5], tmp_path / "short.mp4")

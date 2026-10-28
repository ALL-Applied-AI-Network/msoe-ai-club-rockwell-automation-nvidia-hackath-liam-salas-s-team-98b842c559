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
from fluidmix.physx_varied_cohort_manifest_v4 import (  # noqa: E402
    build_varied_cohort_artifacts as build_varied_cohort_artifacts_v4,
)
from fluidmix.run_sequence_dataset import (  # noqa: E402
    MODEL_INPUT_SOURCE_FIELDS,
    build_dataset_index_rows,
    build_model_input_rows,
    validate_case_artifacts,
    validate_dataset_index,
    validate_transport_pointer,
)
import package_run_sequence_dataset as packager  # noqa: E402
from package_run_sequence_dataset import _height_warnings, stitch_run_video  # noqa: E402


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


def test_height_warnings_read_accepted_v3_and_legacy_receipts(tmp_path: Path):
    warning = {
        "warning_type": "particle_height_outside_geometry_volume_bounds",
        "hard_gate": False,
        "claim": "diagnostic_warning_not_a_physics_pass",
    }
    v3_root = tmp_path / "v3"
    case_ids = [f"CASE_{index:016x}" for index in range(18)]
    _write_json(
        v3_root / "shard-merge.json",
        {
            "schema_version": 1,
            "artifact_type": "physx_train_residual_shard_merge",
            "status": "pass",
            "merged_case_count": 18,
            "case_ids": case_ids,
            "full_resume_summary": {
                "status": "pass",
                "planned_case_count": 18,
                "validated_completed_case_count": 18,
                "case_results": [
                    {
                        "case_id": case_id,
                        "outcome": "recognized_completed",
                        "evidence": {
                            "artifact_validation": {
                                "status": "pass",
                                "failures": [],
                                "warnings": [warning] if index == 0 else [],
                            }
                        },
                    }
                    for index, case_id in enumerate(case_ids)
                ],
            },
        },
    )
    legacy_root = tmp_path / "legacy"
    _write_json(
        legacy_root / "merge-receipt.json",
        {
            "status": "pass",
            "merged_case_count": 30,
            "source_shards": [
                {"height_warnings_by_case": {case_ids[0]: [warning]}}
            ],
        },
    )

    assert _height_warnings(v3_root) == {case_ids[0]: [warning]}
    assert _height_warnings(legacy_root) == {case_ids[0]: [warning]}


def test_dataset_rows_follow_source_manifest_and_exclude_simulator_metadata():
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

    assert len(rows) == 18
    assert Counter(row["disposition"] for row in rows) == {
        "KEEP": 12,
        "DISCARD": 6,
    }
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["disposition"]] += 1
    assert by_split == {
        "train": {"KEEP": 8, "DISCARD": 4},
        "validation": {"KEEP": 2, "DISCARD": 1},
        "evaluation": {"KEEP": 2, "DISCARD": 1},
    }
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
        assert re.fullmatch(
            r"videos/vials/(train|validation|evaluation)-run\d{2}-"
            r"vial\d{2}-(KEEP|DISCARD)-CASE_[0-9a-f]{16}\.mp4",
            row["per_vial_video_path"],
        )
        assert row["run_video_path"].startswith(
            f"videos/runs/{row['split']}-run{row['run_number']}-"
        )
        assert re.search(
            r"-T200[0-9A-F]{6}-T400[0-9A-F]{6}\.mp4$",
            row["run_video_path"],
        )
        assert row["metadata_excluded_from_model_inputs"] == "true"
        assert row["simulator_metadata_excluded_from_model_inputs"] == "true"
        assert json.loads(row["residue_composition_t200_t400_t100"])
        assert json.loads(row["palette_t200_rgb"])
        assert json.loads(row["palette_t400_rgb"])
        assert json.loads(row["palette_t100_rgb"]) == [1.0, 1.0, 1.0]
    assert all(len({row["split"] for row in run}) == 1 for run in by_run.values())
    assert all(len(run) == 3 for run in by_run.values())
    assert rows[0]["run_video_path"].startswith("videos/runs/train-run01-5-T200")
    assert rows[0]["height_warning_count"] == "1"
    assert rows[0]["height_warnings_json"] != "[]"
    assert all(row["height_warning_count"] == "0" for row in rows[1:])


def test_v4_dataset_rows_accept_the_explicit_v4_run_id_contract():
    artifacts = build_varied_cohort_artifacts_v4(repo_root=ROOT)

    rows = build_dataset_index_rows(artifacts)

    assert len(rows) == 30
    assert Counter(row["run_id"] for row in rows) == {
        run["run_id"]: 3
        for run in artifacts["render-schedule-plan.json"]["runs"]
    }
    assert rows[0]["run_id"] == "RUN-V4-TRAIN-01"
    assert rows[0]["run_number"] == "01"
    assert rows[0]["run_video_path"].startswith("videos/runs/train-run01-")
    assert validate_dataset_index(rows)["disposition_counts"] == {
        "KEEP": 15,
        "DISCARD": 15,
    }


@pytest.mark.parametrize("cohort_version", ["v3", "v4"])
def test_packager_cli_forwards_explicit_cohort_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cohort_version: str
):
    observed: dict[str, object] = {}

    def fake_package_dataset(**kwargs):
        observed.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(packager, "package_dataset", fake_package_dataset)

    exit_code = packager.main(
        [
            "--cohort-root",
            str(tmp_path / "cohort"),
            "--output-root",
            str(tmp_path / "package"),
            "--repo-root",
            str(ROOT),
            "--cohort-version",
            cohort_version,
            "--transport-root",
            str(tmp_path / "transport"),
        ]
    )

    assert exit_code == 0
    assert observed["cohort_version"] == cohort_version


@pytest.mark.parametrize(
    ("cohort_version", "builder", "case_count", "run_count", "split_lines"),
    [
        (
            "v3",
            build_varied_cohort_artifacts,
            18,
            6,
            (
                "- train: 12 vials (8 KEEP / 4 DISCARD)",
                "- validation: 3 vials (2 KEEP / 1 DISCARD)",
                "- evaluation: 3 vials (2 KEEP / 1 DISCARD)",
            ),
        ),
        (
            "v4",
            build_varied_cohort_artifacts_v4,
            30,
            10,
            (
                "- train: 12 vials (6 KEEP / 6 DISCARD)",
                "- validation: 9 vials (5 KEEP / 4 DISCARD)",
                "- evaluation: 9 vials (4 KEEP / 5 DISCARD)",
            ),
        ),
    ],
)
def test_package_readme_claims_are_computed_from_validated_rows(
    tmp_path: Path,
    cohort_version: str,
    builder,
    case_count: int,
    run_count: int,
    split_lines: tuple[str, ...],
):
    rows = build_dataset_index_rows(builder(repo_root=ROOT))
    validation = validate_dataset_index(rows)
    readme_path = tmp_path / f"README-{cohort_version}.md"

    packager._write_readme(
        readme_path,
        index_validation=validation,
        height_warning_count=0,
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert f"Its {case_count} video paths" in readme
    assert f"- `videos/runs/`: {run_count} complete run videos" in readme
    assert all(line in readme for line in split_lines)
    assert "Every run contains six vials" not in readme
    assert "five complete run videos" not in readme


@pytest.mark.parametrize(
    ("builder", "expected_count"),
    [
        (build_varied_cohort_artifacts, 18),
        (build_varied_cohort_artifacts_v4, 30),
    ],
)
def test_model_input_rows_are_label_free_and_use_opaque_video_paths(
    builder, expected_count: int
):
    artifacts = builder(repo_root=ROOT)
    index_rows = build_dataset_index_rows(artifacts)

    model_rows = build_model_input_rows(artifacts, index_rows=index_rows)

    assert len(model_rows) == expected_count
    assert all(
        tuple(row) == (*MODEL_INPUT_SOURCE_FIELDS, "video_path")
        for row in model_rows
    )
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


def test_model_input_rows_reject_simulation_truth_and_unknown_fields():
    leak_cases = (
        ("residue", {"volume_ml": 9.0}),
        ("composition", [1.0, 0.0, 0.0]),
        ("tank_colour_preset", "synthetic-red"),
        ("sim_metadata", {"ground_truth": True}),
        ("unexpected_context", "future-field"),
    )
    for field, value in leak_cases:
        artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
        index_rows = build_dataset_index_rows(artifacts)
        artifacts["model-input-plan.json"]["vial_records"][0][field] = value

        with pytest.raises(ValueError, match="forbidden_deploy_input_field"):
            build_model_input_rows(artifacts, index_rows=index_rows)


def test_dataset_validation_fails_on_duplicate_split_crossing_or_missing_paths(
    tmp_path: Path,
):
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    rows = build_dataset_index_rows(artifacts)

    validate_dataset_index(rows)

    duplicate = deepcopy(rows)
    duplicate[0]["case_id"] = duplicate[1]["case_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset_index(duplicate)

    crossed = deepcopy(rows)
    crossed[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="split"):
        validate_dataset_index(crossed)

    with pytest.raises(ValueError, match="missing package path"):
        validate_dataset_index(rows, package_root=tmp_path)


def test_transport_pointer_binds_run_receipt_and_full_accepted_segments(
    tmp_path: Path,
):
    run_id = "RUN-TRAIN-01"
    case_ids = ["CASE_0000000000000001", "CASE_0000000000000002"]
    records = []
    for index, mode in enumerate(("entry", "between", "exit")):
        render_dir = tmp_path / "transport" / f"segment-{index:02d}-{mode}"
        render_dir.mkdir(parents=True)
        receipt_path = render_dir / "transport-render-receipt.json"
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "artifact_type": "fluidmix_transport_stage_render_receipt_v1",
                "status": "candidate",
                "mode": mode,
                "state_indices": list(range(13)),
                "rendered_states_byte_unique": True,
                "pilot_visual_review": {"status": "accepted"},
                "checker_evaluation": {
                    "schema_version": 1,
                    "artifact_type": "fluidmix_transport_stage_checker_v1",
                    "status": "pass",
                    "pilot": False,
                    "manual_visual_review_required": False,
                    "pilot_visual_review_status": "accepted",
                    "automatic_acceptance_gates": {
                        "example_gate": True,
                        "pilot_visual_review_accepted": True,
                    },
                },
            },
        )
        records.append(
            {
                "render_dir": str(render_dir.resolve()),
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": _sha256(receipt_path),
                "mode": mode,
            }
        )
    output_video = tmp_path / "package" / "videos" / "runs" / "run.mp4"
    output_video.parent.mkdir(parents=True)
    output_video.write_bytes(b"run-video")
    run_receipt_path = output_video.with_suffix(".run.json")
    _write_json(
        run_receipt_path,
        {
            "schema_version": 2,
            "artifact_type": "rendered_transport_run_sequence_v2",
            "status": "pass",
            "run_id": run_id,
            "case_ids": case_ids,
            "transport_segments": records,
            "transport_segment_count": 3,
            "encoded_transition_sequences_byte_unique": True,
            "encoded_transition_evidence": [
                {
                    "segment_index": index,
                    "mode": mode,
                    "start_frame_index": index * 13,
                    "stop_frame_index": index * 13 + 13,
                    "frame_count": 13,
                    "decoded_bgr_sha256": [
                        hashlib.sha256(f"{index}:{frame}".encode()).hexdigest()
                        for frame in range(13)
                    ],
                    "byte_unique": True,
                }
                for index, mode in enumerate(("entry", "between", "exit"))
            ],
            "encoded_transition_seams_passed": True,
            "encoded_seam_evidence": [
                {
                    "segment_index": index,
                    "mode": mode,
                    "start_seam_mad": None if mode == "entry" else 1.0,
                    "start_seam_threshold_p99": None if mode == "entry" else 2.0,
                    "start_seam_passed": True,
                    "end_seam_mad": None if mode == "exit" else 1.0,
                    "end_seam_threshold_p99": None if mode == "exit" else 2.0,
                    "end_seam_passed": True,
                }
                for index, mode in enumerate(("entry", "between", "exit"))
            ],
            "output_video": str(output_video.resolve()),
            "output_video_sha256": _sha256(output_video),
            "qualified_fill_receipts_changed": False,
            "camera_and_lighting_changed": False,
        },
    )
    pointer_path = tmp_path / "pointer.json"
    pointer = {
        "schema_version": 2,
        "artifact_type": "rendered_transport_vial_pointer_v2",
        "status": "pass",
        "run_id": run_id,
        "case_id": case_ids[0],
        "run_receipt": {
            "path": str(run_receipt_path.resolve()),
            "sha256": _sha256(run_receipt_path),
        },
        "entry_segment": records[0],
        "outgoing_segment": records[1],
        "qualified_fill_receipt_changed": False,
    }
    _write_json(pointer_path, pointer)

    result = validate_transport_pointer(
        pointer_path,
        expected_run_id=run_id,
        expected_case_ids=case_ids,
        expected_ordinal=1,
        expected_run_video=output_video,
    )
    assert result["status"] == "pass"
    assert result["outgoing_mode"] == "between"

    pointer["outgoing_segment"] = records[2]
    _write_json(pointer_path, pointer)
    with pytest.raises(ValueError, match="outgoing segment"):
        validate_transport_pointer(
            pointer_path,
            expected_run_id=run_id,
            expected_case_ids=case_ids,
            expected_ordinal=1,
            expected_run_video=output_video,
        )


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


def test_run_video_stitch_preserves_every_input_frame_for_variable_run_sizes(
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

    receipt = stitch_run_video(inputs[:3], tmp_path / "run.mp4")

    assert receipt["status"] == "pass"
    assert receipt["input_frame_counts"] == [3] * 3
    assert receipt["output_frame_count"] == 9
    assert receipt["fps"] == 10.0
    assert receipt["resolution"] == [64, 48]

    with pytest.raises(ValueError, match="at least one"):
        stitch_run_video([], tmp_path / "empty.mp4")

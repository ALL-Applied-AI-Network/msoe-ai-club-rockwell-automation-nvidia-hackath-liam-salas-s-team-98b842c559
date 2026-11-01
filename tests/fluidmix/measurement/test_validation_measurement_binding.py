"""Tests for binding the Brev VA cohort to deterministic measurements."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

SOURCE_MODEL_SHA256 = (
    "8dbba6e29f3d63c9c6270a6804f79203b5da40b4955fc12f1f5d1ec3c88da01a"
)


def test_binding_cli_help_is_side_effect_free():
    script = ROOT / "cli/fluidmix/measurement" / "build_validation_measurement_inputs.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--source-model-plan" in result.stdout
    assert "--cohort-root" in result.stdout
    assert "--output-dir" in result.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen_model_plan(path: Path) -> dict[str, object]:
    from fluidmix.memory.history.vision_dataset_plan import CONFIG_PATH, build_vision_artifacts

    payload = build_vision_artifacts(CONFIG_PATH, repo_root=ROOT)["model-input-plan.json"]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assert _sha256(path) == SOURCE_MODEL_SHA256
    return payload


def _referenced_image_ids(decisions: list[object]) -> set[str]:
    referenced: set[str] = set()
    for decision in decisions:
        assert isinstance(decision, dict)
        for stage in decision["stages"]:
            referenced.add(stage["target_image_id"])
            for case in stage["reference_cases"]:
                for peer in case["candidate_peers"]:
                    referenced.add(peer["peer_image_id"])
    return referenced


def test_validation_slice_is_exact_allowlisted_and_source_bound(tmp_path):
    from fluidmix.measurement.validation_measurement_binding import write_validation_model_slice

    source = tmp_path / "model-input-plan.json"
    _write_frozen_model_plan(source)
    output = tmp_path / "validation-model-input.json"

    record = write_validation_model_slice(source, output)

    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert record["artifact_type"] == "model_input_plan"
    assert record["slice_family_id"] == "VA"
    assert record["source_model_input_sha256"] == SOURCE_MODEL_SHA256
    assert record["real_data_used"] is False
    assert len(record["decision_records"]) == 12
    assert len(record["image_placeholders"]) == 72
    assert len(record["reference_batches"]) == 4
    assert len(record["target_batches"]) == 4
    assert len(record["cosmos_attempts"]) == 48
    assert {row["family_id"] for row in record["decision_records"]} == {"VA"}
    assert {row["family_id"] for row in record["image_placeholders"]} == {"VA"}
    referenced = _referenced_image_ids(record["decision_records"])
    assert referenced == {row["image_id"] for row in record["image_placeholders"]}
    decision_ids = {row["decision_id"] for row in record["decision_records"]}
    assert {row["decision_id"] for row in record["cosmos_attempts"]} == decision_ids
    assert record["denominators"] == {
        "cosmos_model_calls_scored": 36,
        "multi_image_smoke_unscored": 1,
        "reference_availability_target_stage_records": 216,
        "scored_attempts_total": 48,
        "scored_target_decisions": 12,
        "template_controls_scored": 12,
    }


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _write_completed_cohort(root: Path, sliced: dict[str, object]) -> None:
    from twin.render.twin_video import VIDEO_TIME_CODES

    stage_code = {"STAGE1": 171, "STAGE2": 342, "STAGE3": 686}
    colors = {
        "STAGE1": (20, 60, 100),
        "STAGE2": (50, 80, 110),
        "STAGE3": (80, 100, 120),
    }
    by_launch: dict[str, list[dict[str, object]]] = {}
    for image in sliced["image_placeholders"]:
        launch_id = f"{image['batch_id']}-{image['vial_id']}"
        by_launch.setdefault(launch_id, []).append(image)
    assert len(by_launch) == 24

    launches = []
    run_rows = []
    for ordinal, (launch_id, images) in enumerate(by_launch.items(), start=1):
        capture = root / launch_id / "capture"
        for kind in ("rgb", "semantic", "instance", "liquid"):
            (capture / kind).mkdir(parents=True, exist_ok=True)
        image_by_time = {stage_code[row["stage_id"]]: row for row in images}
        frames = []
        for capture_ordinal, time_code in enumerate(VIDEO_TIME_CODES, start=1):
            artifacts = {}
            if time_code in image_by_time:
                stage_id = image_by_time[time_code]["stage_id"]
                rgb = capture / "rgb" / f"{time_code:04d}.png"
                mask = capture / "liquid" / f"{time_code:04d}.png"
                Image.new("RGB", (8, 8), colors[stage_id]).save(rgb)
                Image.new("L", (8, 8), 255).save(mask)
            for kind in ("rgb", "semantic", "instance", "liquid"):
                suffix = ".png" if kind in {"rgb", "liquid"} else ".npy"
                artifact = capture / kind / f"{time_code:04d}{suffix}"
                if not artifact.exists():
                    artifact.write_bytes(
                        f"synthetic-test-{launch_id}-{kind}-{time_code}".encode()
                    )
                artifacts[kind] = {
                    "path": f"{kind}/{time_code:04d}{suffix}",
                    "sha256": _sha256(artifact),
                }
            frames.append(
                {
                    "capture_ordinal": capture_ordinal,
                    "time_code": time_code,
                    "artifacts": artifacts,
                }
            )
        summary = {
            "schema_version": 1,
            "mechanical_gates_passed": True,
            "visual_check_status": "pending",
            "frame_count": 133,
            "inputs": {
                "stage": {"requested": "fill_station.usda"},
                "camera": {"name": "FillCam"},
            },
            "render_settings": {
                "renderer": "PathTracing",
                "resolution": [640, 480],
                "time_codes": list(VIDEO_TIME_CODES),
                "frame_spec": ",".join(str(code) for code in VIDEO_TIME_CODES),
                "rt_subframes": 64,
                "warmup_updates": 60,
                "max_tries_per_frame": 20,
                "replicator_global_seed": 1310000 + ordinal,
                "require_liquid_pixels": True,
            },
            "frames": frames,
        }
        summary_sha = _write_json(capture / "summary.json", summary)
        attempt = {
            "schema_version": 1,
            "run_id": launch_id,
            "status": "mechanical_gates_passed",
            "visual_check_status": "pending",
            "failure_reasons": [],
            "capture": {
                "root": str(capture.resolve()),
                "verified": True,
                "inner_summary_sha256": summary_sha,
                "appearance_provenance_verified": True,
            },
        }
        attempt_sha = _write_json(root / launch_id / "attempt.json", attempt)
        video_dir = root / "videos" / launch_id
        video_dir.mkdir(parents=True)
        video = video_dir / "sequence.mp4"
        video.write_bytes(f"review-only-{launch_id}".encode())
        video_record = {
            "schema_version": 1,
            "status": "encoded_verified",
            "capture": {"root": str(capture.resolve()), "summary_sha256": summary_sha},
            "encoding": {
                "codec_requested": "mp4v",
                "container": "mp4",
                "fps": 6.0,
                "width": 640,
                "height": 480,
            },
            "time_codes": list(VIDEO_TIME_CODES),
            "stage_frames": [
                {
                    "capture_ordinal": list(VIDEO_TIME_CODES).index(code) + 1,
                    "time_code": code,
                    "rgb_sha256": frames[list(VIDEO_TIME_CODES).index(code)][
                        "artifacts"
                    ]["rgb"]["sha256"],
                }
                for code in (171, 342, 686)
            ],
            "video": {
                "path": "sequence.mp4",
                "sha256": _sha256(video),
                "bytes": video.stat().st_size,
            },
            "verification": {
                "decoded_frame_count": 133,
                "reported_fourcc": "FMP4",
                "reported_fps": 6.0,
                "reported_width": 640,
                "reported_height": 480,
            },
            "limits": {
                "measurement_source": "hash-bound capture PNGs only",
                "video_role": "lossy review visualization only",
            },
        }
        video_record_sha = _write_json(video_dir / "video.json", video_record)
        launches.append(
            {
                "ordinal": ordinal,
                "launch_id": launch_id,
                "family_id": "VA",
                "partition": "val",
                "batch_id": images[0]["batch_id"],
                "vial_id": images[0]["vial_id"],
                "max_process_attempts": 1,
                "replacement_attempts": 0,
                "capture_root": str(capture.resolve()),
                "video_output_dir": str(video_dir.resolve()),
            }
        )
        run_rows.append(
            {
                "ordinal": ordinal,
                "launch_id": launch_id,
                "outcome": "passed",
                "render": {"returncode": 0},
                "encoder": {"returncode": 0},
                "attempt_sha256": attempt_sha,
                "video_record_sha256": video_record_sha,
            }
        )
    cohort = {
        "schema_version": 1,
        "artifact_type": "brev_validation_video_cohort",
        "status": "materialized_unrendered",
        "source_render_plan_sha256": (
            "98e73854c4817bd3e2ef2dfa0c21bd6c63a6f2d0ac83021365a0ccd665d5296b"
        ),
        "implementation_commit": "b6b0103d74aa15f07486d2465201b5c96c50d786",
        "render_runtime": {
            "provider": "Brev",
            "instance_name": "claude-isaac-smoke-1024",
            "instance_id": "m1vgz4tbs",
            "machine_type": "g6e.xlarge",
            "gpu": "L40S",
            "machine": "x86_64",
            "runtime_name": "NVIDIA Isaac Sim",
            "isaac_sim_version": "6.0.1.0",
            "environment_profile": "generic",
            "isaac_python": "/home/ubuntu/isaac/.venv/bin/python",
        },
        "family": {
            "family_id": "VA",
            "partition": "val",
            "previous_sim_recipe_id": "S0160",
            "current_sim_recipe_id": "S0380",
            "synthetic_onset_id": "after_t400",
        },
        "runtime_mixing_policy": (
            "whole cohort on one Brev host and Isaac 6.0.1; never pool RGB/Lab "
            "with Isaac 6.2 cohorts"
        ),
        "render_controls": {
            "camera": "FillCam",
            "view_id": "vial_fill",
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "time_codes": list(VIDEO_TIME_CODES),
            "frame_spec": ",".join(str(code) for code in VIDEO_TIME_CODES),
            "stage_time_codes": [171, 342, 686],
            "subframes": 64,
            "warmup": 60,
            "max_tries": 20,
            "timeout_seconds": 1800,
            "require_liquid_pixels": True,
        },
        "video_encoding": {
            "codec_request": "mp4v",
            "container": "mp4",
            "fps": 6.0,
            "expected_decoded_frames": 133,
            "use": "lossy_review_visualization_only",
            "measurement_source": "hash_bound_capture_png_and_liquid_mask_only",
        },
        "process_policy": {
            "max_process_attempts_per_launch": 1,
            "replacement_attempts": 0,
            "cohort_stop": "first_failure",
        },
        "denominators": {
            "batches": 8,
            "vials_and_render_launches": 24,
            "frames_per_launch": 133,
            "total_frames": 3192,
            "stage_frames_per_launch": 3,
            "stage_frames_total": 72,
            "videos": 24,
            "maximum_process_attempts_per_launch": 1,
        },
        "launches": launches,
    }
    cohort_sha = _write_json(root / "cohort.json", cohort)
    cohort_run = {
        "schema_version": 1,
        "status": "completed",
        "cohort": {
            "path": str((root / "cohort.json").resolve()),
            "sha256": cohort_sha,
            "bytes": (root / "cohort.json").stat().st_size,
        },
        "expected_launches": 24,
        "attempted_launches": 24,
        "completed_launches": 24,
        "failed_launch_id": None,
        "replacement_attempts": 0,
        "stop_policy": "first_failure",
        "rows": run_rows,
    }
    _write_json(root / "cohort-run.json", cohort_run)


def test_locator_binding_uses_only_verified_stage_pngs_and_drives_measurement(tmp_path):
    from fluidmix.measurement.validation_measurement_binding import (
        write_validation_image_locators,
        write_validation_model_slice,
    )
    from fluidmix.pipeline.model_plan_measurement import write_measured_evidence

    source = tmp_path / "model-input-plan.json"
    _write_frozen_model_plan(source)
    output = tmp_path / "evidence"
    output.mkdir(mode=0o700)
    sliced_path = output / "validation-model-input.json"
    sliced = write_validation_model_slice(source, sliced_path)
    cohort = tmp_path / "cohort"
    cohort.mkdir(mode=0o700)
    _write_completed_cohort(cohort, sliced)
    locator_path = output / "validation-image-locators.json"
    receipt_path = output / "validation-binding-receipt.json"

    records = write_validation_image_locators(
        sliced_path, cohort, locator_path, receipt_path
    )

    locator = records["locator"]
    receipt = records["receipt"]
    assert json.loads(locator_path.read_text()) == locator
    assert json.loads(receipt_path.read_text()) == receipt
    assert set(locator) == {"schema_version", "data_origin", "images"}
    assert locator["data_origin"] == "twin"
    assert len(locator["images"]) == 72
    assert set(locator["images"]) == {
        row["image_id"] for row in sliced["image_placeholders"]
    }
    assert all("sequence.mp4" not in row["rgb_path"] for row in locator["images"].values())
    assert all("sequence.mp4" not in row["mask_path"] for row in locator["images"].values())
    for row in locator["images"].values():
        assert _sha256(Path(row["rgb_path"])) == row["rgb_sha256"]
        assert _sha256(Path(row["mask_path"])) == row["mask_sha256"]
    assert receipt["real_data_used"] is False
    assert receipt["denominators"] == {
        "batches": 8,
        "captures": 24,
        "stage_images": 72,
        "verified_videos_not_used_for_measurement": 24,
    }
    assert receipt["measurement_source"] == "hash_bound_capture_rgb_and_liquid_mask_pngs"
    assert receipt["capture_gate_verification"] == (
        "full_summary_and_artifact_tree_rehashed"
    )
    assert receipt["video_gate_verification"] == (
        "recorded_decode_gate_and_mp4_hash_reverified_without_measurement_decode"
    )
    assert receipt["review_video_role"] == "verified_lossy_visualization_not_measurement"
    assert receipt["source_sliced_model_input_sha256"] == _sha256(sliced_path)
    assert receipt["image_locator_sha256"] == _sha256(locator_path)

    measured_path = output / "validation-measured-evidence.json"
    measured = write_measured_evidence(sliced_path, locator_path, measured_path)
    assert measured["denominators"]["image_measurements"] == 72
    assert measured["denominators"]["vial_decisions"] == 12
    assert measured["real_data_used"] is False


def test_locator_binding_rejects_pixel_tamper_without_outputs(tmp_path):
    from fluidmix.measurement.validation_measurement_binding import (
        write_validation_image_locators,
        write_validation_model_slice,
    )
    import pytest

    source = tmp_path / "model-input-plan.json"
    _write_frozen_model_plan(source)
    output = tmp_path / "evidence"
    output.mkdir(mode=0o700)
    sliced_path = output / "validation-model-input.json"
    sliced = write_validation_model_slice(source, sliced_path)
    cohort = tmp_path / "cohort"
    cohort.mkdir(mode=0o700)
    _write_completed_cohort(cohort, sliced)
    first = sliced["image_placeholders"][0]
    rgb = (
        cohort
        / f"{first['batch_id']}-{first['vial_id']}"
        / "capture"
        / "rgb"
        / f"{first['time_code']:04d}.png"
    )
    rgb.write_bytes(b"tampered")
    locator_path = output / "validation-image-locators.json"
    receipt_path = output / "validation-binding-receipt.json"

    with pytest.raises(ValueError, match="hash"):
        write_validation_image_locators(
            sliced_path, cohort, locator_path, receipt_path
        )

    assert not locator_path.exists()
    assert not receipt_path.exists()


def test_locator_binding_rejects_incomplete_cohort_without_outputs(tmp_path):
    import pytest

    from fluidmix.measurement.validation_measurement_binding import (
        write_validation_image_locators,
        write_validation_model_slice,
    )

    source = tmp_path / "model-input-plan.json"
    _write_frozen_model_plan(source)
    output = tmp_path / "evidence"
    output.mkdir(mode=0o700)
    sliced_path = output / "validation-model-input.json"
    sliced = write_validation_model_slice(source, sliced_path)
    cohort = tmp_path / "cohort"
    cohort.mkdir(mode=0o700)
    _write_completed_cohort(cohort, sliced)
    run_path = cohort / "cohort-run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["status"] = "failed"
    run["completed_launches"] = 23
    run["failed_launch_id"] = run["rows"][-1]["launch_id"]
    _write_json(run_path, run)
    locator_path = output / "validation-image-locators.json"
    receipt_path = output / "validation-binding-receipt.json"

    with pytest.raises(ValueError, match="completed 24/24"):
        write_validation_image_locators(
            sliced_path, cohort, locator_path, receipt_path
        )

    assert not locator_path.exists()
    assert not receipt_path.exists()

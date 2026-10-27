"""Tests for the frozen VA pixel-to-evidence execution boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _validation_slice(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from fluidmix.validation_measurement_binding import write_validation_model_slice
    from fluidmix.vision_dataset_plan import CONFIG_PATH, build_vision_artifacts

    source = tmp_path / "model-input-plan.json"
    plan = build_vision_artifacts(CONFIG_PATH, repo_root=ROOT)["model-input-plan.json"]
    _write_json(source, plan)
    sliced_path = tmp_path / "validation-model-input.json"
    sliced = write_validation_model_slice(source, sliced_path)
    return sliced_path, sliced


def _bound_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    sliced_path, sliced = _validation_slice(tmp_path)
    pixel_root = tmp_path / "pixels"
    pixel_root.mkdir(mode=0o700)
    rgb = np.full((3, 4, 3), (24, 61, 103), dtype=np.uint8)
    mask = np.array(
        [[0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    images: dict[str, dict[str, str]] = {}
    for placeholder in sliced["image_placeholders"]:
        image_id = placeholder["image_id"]
        rgb_path = pixel_root / f"{image_id}-rgb.png"
        mask_path = pixel_root / f"{image_id}-mask.png"
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        images[image_id] = {
            "rgb_path": str(rgb_path),
            "rgb_sha256": _sha256(rgb_path),
            "mask_path": str(mask_path),
            "mask_sha256": _sha256(mask_path),
        }
    locator_path = tmp_path / "validation-image-locators.json"
    locator_sha = _write_json(
        locator_path,
        {"schema_version": 1, "data_origin": "twin", "images": images},
    )
    receipt_path = tmp_path / "validation-binding-receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_type": "validation_image_binding_receipt",
            "status": "bound",
            "family_id": "VA",
            "partition": "val",
            "data_origin": "synthetic_twin",
            "real_data_used": False,
            "source_sliced_model_input_sha256": _sha256(sliced_path),
            "source_cohort_graph_sha256": "1" * 64,
            "source_cohort_run_sha256": "2" * 64,
            "image_locator_sha256": locator_sha,
            "measurement_source": "hash_bound_capture_rgb_and_liquid_mask_pngs",
            "capture_gate_verification": "full_summary_and_artifact_tree_rehashed",
            "video_gate_verification": (
                "recorded_decode_gate_and_mp4_hash_reverified_without_measurement_decode"
            ),
            "review_video_role": "verified_lossy_visualization_not_measurement",
            "stage_time_codes": [171, 342, 686],
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
            "denominators": {
                "batches": 8,
                "captures": 24,
                "stage_images": 72,
                "verified_videos_not_used_for_measurement": 24,
            },
            "claim_boundary": (
                "Synthetic twin pixels only; no real data, decoded-MP4 measurement, "
                "physical carryover, defect, plant-action, or sim-to-real claim."
            ),
        },
    )
    return sliced_path, locator_path, receipt_path, sliced


def test_validation_measurement_publishes_exact_va_evidence_from_pngs(tmp_path):
    from fluidmix.validation_stage_measurement import write_validation_stage_evidence

    model_path, locator_path, receipt_path, sliced = _bound_inputs(tmp_path)
    output = tmp_path / "validation-measured-evidence.json"

    measured = write_validation_stage_evidence(
        model_path, locator_path, receipt_path, output
    )

    assert json.loads(output.read_text(encoding="utf-8")) == measured
    assert measured["artifact_type"] == "measured_evidence"
    assert measured["denominators"] == {
        "image_measurements": 72,
        "valid_image_measurements": 72,
        "invalid_image_measurements": 0,
        "invalid_image_measurement_reasons": {},
        "vial_decisions": 12,
        "valid_vial_decisions": 12,
        "abstained_vial_decisions": 0,
    }
    assert {row["decision_id"] for row in measured["decision_records"]} == {
        row["decision_id"] for row in sliced["decision_records"]
    }
    assert all(row["validity"] == "valid" for row in measured["image_measurements"])
    assert measured["source_model_input_sha256"] == _sha256(model_path)
    assert measured["source_image_locators_sha256"] == _sha256(locator_path)
    assert measured["source_binding_receipt_sha256"] == _sha256(receipt_path)
    assert measured["data_origin"] == "twin"
    assert measured["real_data_used"] is False
    assert "sequence.mp4" not in json.dumps(measured)


def test_validation_measurement_rejects_pixel_hash_drift_without_output(tmp_path):
    import pytest

    from fluidmix.validation_stage_measurement import write_validation_stage_evidence

    model_path, locator_path, receipt_path, _ = _bound_inputs(tmp_path)
    locator = json.loads(locator_path.read_text(encoding="utf-8"))
    first = next(iter(locator["images"].values()))
    Path(first["rgb_path"]).write_bytes(b"tampered")
    output = tmp_path / "validation-measured-evidence.json"

    with pytest.raises(ValueError, match="failed closed"):
        write_validation_stage_evidence(
            model_path, locator_path, receipt_path, output
        )

    assert not output.exists()


def test_validation_measurement_rejects_non_twin_origin_without_output(tmp_path):
    import pytest

    from fluidmix.validation_stage_measurement import write_validation_stage_evidence

    model_path, locator_path, receipt_path, _ = _bound_inputs(tmp_path)
    locator = json.loads(locator_path.read_text(encoding="utf-8"))
    locator["data_origin"] = "real"
    _write_json(locator_path, locator)
    output = tmp_path / "validation-measured-evidence.json"

    with pytest.raises(ValueError, match="origin or schema"):
        write_validation_stage_evidence(
            model_path, locator_path, receipt_path, output
        )

    assert not output.exists()


def test_validation_measurement_never_accepts_mp4_path_as_pixels(tmp_path):
    import pytest

    from fluidmix.validation_stage_measurement import write_validation_stage_evidence

    model_path, locator_path, receipt_path, _ = _bound_inputs(tmp_path)
    locator = json.loads(locator_path.read_text(encoding="utf-8"))
    first = next(iter(locator["images"].values()))
    original = Path(first["rgb_path"])
    disguised_video = original.with_name("sequence.mp4")
    disguised_video.write_bytes(original.read_bytes())
    first["rgb_path"] = str(disguised_video)
    first["rgb_sha256"] = _sha256(disguised_video)
    locator_sha = _write_json(locator_path, locator)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["image_locator_sha256"] = locator_sha
    _write_json(receipt_path, receipt)
    output = tmp_path / "validation-measured-evidence.json"

    with pytest.raises(ValueError, match="direct PNG"):
        write_validation_stage_evidence(
            model_path, locator_path, receipt_path, output
        )

    assert not output.exists()


def test_validation_measurement_rejects_binding_receipt_drift(tmp_path):
    import pytest

    from fluidmix.validation_stage_measurement import write_validation_stage_evidence

    model_path, locator_path, receipt_path, _ = _bound_inputs(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["review_video_role"] = "measurement"
    _write_json(receipt_path, receipt)
    output = tmp_path / "validation-measured-evidence.json"

    with pytest.raises(ValueError, match="provenance drifted"):
        write_validation_stage_evidence(
            model_path, locator_path, receipt_path, output
        )

    assert not output.exists()


def test_validation_measurement_cli_help_is_side_effect_free():
    script = ROOT / "scripts" / "measure_validation_stage_evidence.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--model-plan" in result.stdout
    assert "--image-locators" in result.stdout
    assert "--binding-receipt" in result.stdout
    assert "--output" in result.stdout

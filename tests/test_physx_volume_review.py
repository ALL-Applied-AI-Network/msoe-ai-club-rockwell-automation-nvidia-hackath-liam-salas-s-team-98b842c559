from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluidmix.physx_volume_gate import (
    find_dense_surface_row,
    projected_vial_geometry_roi_mask,
    visible_height_from_surface_row_m,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "physx_volume_review",
    ROOT / "scripts/physx_volume_review.py",
)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)

GEOMETRY = {
    "center_xy_m": [5.2324003281, -1.9811999798],
    "bottom_z_m": 0.9762065645,
    "mouth_z_m": 1.1762601692,
    "circumradius_m": 0.0132029216,
    "wall_count": 12,
    "cross_section_area_m2": 0.000522951416,
}
TARGET_STEP = 1412
CROP_XYXY = [278, 208, 342, 424]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.write_text(json.dumps(receipt), encoding="utf-8")


def _write_fixture(
    tmp_path: Path,
    *,
    surface_row_px: int = 361,
    state_passed: bool = False,
) -> tuple[Path, Path, Path, dict[str, object]]:
    app_output = tmp_path / "run" / "app-output"
    rgb_path = app_output / "frames" / "target.png"
    mask_path = app_output / "liquid" / "target.png"
    rgb_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(640, dtype=np.uint16) % 256
    rgb[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
    rgb[:, :, 2] = 40
    Image.fromarray(rgb, mode="RGB").save(rgb_path)

    roi = projected_vial_geometry_roi_mask(
        (480, 640),
        center_xy_m=tuple(GEOMETRY["center_xy_m"]),
        bottom_z_m=GEOMETRY["bottom_z_m"],
        mouth_z_m=GEOMETRY["mouth_z_m"],
        circumradius_m=GEOMETRY["circumradius_m"],
        wall_count=GEOMETRY["wall_count"],
    )
    liquid = np.zeros((480, 640), dtype=np.uint8)
    liquid[surface_row_px:] = roi[surface_row_px:] * 255
    Image.fromarray(liquid, mode="L").save(mask_path)
    surface = find_dense_surface_row(liquid, geometry_roi_mask=roi)
    visible_height = visible_height_from_surface_row_m(
        surface["surface_row_px"],
        center_xy_m=tuple(GEOMETRY["center_xy_m"]),
        bottom_z_m=GEOMETRY["bottom_z_m"],
        circumradius_m=GEOMETRY["circumradius_m"],
        maximum_height_m=GEOMETRY["mouth_z_m"] - GEOMETRY["bottom_z_m"],
        wall_count=GEOMETRY["wall_count"],
    )
    expected_height = 0.019122235236979125
    visible_passed = abs(visible_height - expected_height) <= 0.006
    gate_passed = state_passed and visible_passed
    failure_reasons = []
    if not state_passed:
        failure_reasons.append("particle_height_outside_tolerance")
    if not visible_passed:
        failure_reasons.append("visible_height_outside_tolerance")
    paired_state = {
        "passed": True,
        "physics_step": TARGET_STEP,
        "positions_sha256_before": "a" * 64,
        "positions_sha256_after": "a" * 64,
    }
    target_gate = {
        "required": True,
        "state_passed": state_passed,
        "visible_passed": visible_passed,
        "passed": gate_passed,
        "failure_reasons": failure_reasons,
        "programmed_cumulative_volume_ml": 10.0,
        "expected_height_m": expected_height,
        "height_tolerance_m": 0.006,
        "particle_bulk_height_m": 0.008,
        "visible_mask_evidence": {
            "required": True,
            "passed": visible_passed,
            "status": "measured",
            **surface,
            "observed_height_m": visible_height,
            "absolute_error_m": abs(visible_height - expected_height),
        },
    }
    receipt: dict[str, object] = {
        "schema_version": 17,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "insufficient_evidence",
        "containment": {"vial_interior_bounds_m": dict(GEOMETRY)},
        "qualification": {
            "first_stage_only": True,
            "status": "qualification_gate_failed",
            "target_step": TARGET_STEP,
            "volume_state_gate_passed": state_passed,
            "volume_visible_gate_passed": visible_passed,
            "volume_gate_passed": gate_passed,
            "target_volume_evidence": target_gate,
            "paired_capture_evidence": [
                {
                    "physics_step": TARGET_STEP,
                    "rgb_path": str(rgb_path.relative_to(app_output)),
                    "rgb_sha256": _sha256(rgb_path),
                    "mask_path": str(mask_path.relative_to(app_output)),
                    "mask_sha256": _sha256(mask_path),
                    "paired_same_physics_state_rgb_semantic_capture": True,
                    "paired_capture_state_passed": True,
                    "paired_capture_state_evidence": paired_state,
                    "visible_stream_tanks": [],
                }
            ],
        },
    }
    receipt_path = app_output / "receipt.json"
    _save_receipt(receipt_path, receipt)
    return receipt_path, rgb_path, mask_path, receipt


def test_review_writes_bound_geometry_overlay_without_overriding_runner(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    output_dir = tmp_path / "review"

    metrics = review.build_volume_review(
        receipt_path=receipt_path,
        rgb_path=rgb_path,
        liquid_mask_path=mask_path,
        output_dir=output_dir,
    )

    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    hashes = json.loads(
        (output_dir / "artifact-hashes.json").read_text(encoding="utf-8")
    )
    assert metrics == saved
    assert saved["crop_xyxy"] == CROP_XYXY
    assert saved["runner_decision"] == {
        "run_status": receipt["status"],
        "qualification_status": receipt["qualification"]["status"],
        "volume_state_gate_passed": False,
        "volume_visible_gate_passed": True,
        "volume_gate_passed": False,
    }
    assert saved["review_contract"] == {
        "decision_source": "receipt_only",
        "does_not_override_runner": True,
    }
    assert saved["detected_surface"]["surface_row_px"] == 361
    assert set(saved["contours_px"]) == {
        "floor",
        "expected",
        "lower_tolerance",
        "upper_tolerance",
    }
    assert saved["paired_capture_state_evidence"] == receipt["qualification"][
        "paired_capture_evidence"
    ][0]["paired_capture_state_evidence"]
    assert Image.open(output_dir / "target-volume-overlay.png").size == (640, 480)
    assert Image.open(output_dir / "target-rgb-and-overlay-crop.png").size == (
        128,
        216,
    )
    for name in (
        "metrics.json",
        "target-volume-overlay.png",
        "target-rgb-and-overlay-crop.png",
    ):
        assert hashes["artifacts"][name]["sha256"] == _sha256(output_dir / name)


def test_review_preserves_state_pass_visible_fail_without_reinterpreting_it(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, _ = _write_fixture(
        tmp_path,
        surface_row_px=374,
        state_passed=True,
    )

    metrics = review.build_volume_review(
        receipt_path=receipt_path,
        rgb_path=rgb_path,
        liquid_mask_path=mask_path,
        output_dir=tmp_path / "review",
    )

    assert metrics["runner_decision"]["volume_state_gate_passed"] is True
    assert metrics["runner_decision"]["volume_visible_gate_passed"] is False
    assert metrics["runner_decision"]["volume_gate_passed"] is False
    assert metrics["target_volume_evidence"]["failure_reasons"] == [
        "visible_height_outside_tolerance"
    ]


def test_review_rejects_receipt_hash_mismatch_before_writing(tmp_path: Path) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    receipt["qualification"]["paired_capture_evidence"][0]["rgb_sha256"] = "0" * 64
    _save_receipt(receipt_path, receipt)
    output_dir = tmp_path / "review"

    with pytest.raises(ValueError, match="RGB hash mismatch"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_review_rejects_same_bytes_from_a_path_not_bound_by_receipt(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, _ = _write_fixture(tmp_path)
    copied_rgb = tmp_path / "unbound-target.png"
    copied_rgb.write_bytes(rgb_path.read_bytes())

    with pytest.raises(ValueError, match="RGB path mismatch"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=copied_rgb,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )


def test_review_rejects_missing_runner_volume_state(tmp_path: Path) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    del receipt["qualification"]["target_volume_evidence"]["state_passed"]
    _save_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="state_passed.*boolean"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )


def test_review_rejects_surface_row_that_contradicts_bound_mask(tmp_path: Path) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    visible = receipt["qualification"]["target_volume_evidence"][
        "visible_mask_evidence"
    ]
    visible["surface_row_px"] += 1
    _save_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="surface_row_px contradicts"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )


def test_review_rejects_tiny_non_fillcam_rgb_even_when_receipt_hash_matches(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    Image.new("RGB", (2, 2), (1, 2, 3)).save(rgb_path)
    receipt["qualification"]["paired_capture_evidence"][0]["rgb_sha256"] = _sha256(
        rgb_path
    )
    _save_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="frozen 640x480"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )


def test_review_rejects_nonfrozen_tolerance_instead_of_drawing_other_bounds(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    receipt["qualification"]["target_volume_evidence"][
        "height_tolerance_m"
    ] = 0.005
    _save_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="frozen 0.006 m"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )


def test_review_refuses_to_overwrite_an_existing_output_directory(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "review"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=output_dir,
        )

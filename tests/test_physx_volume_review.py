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
COUNT_DRIVEN_RENDER_PATH = "count_driven_twin_liquid_mesh"
PARTICLE_VOLUME_M3 = 27.0e-9
# Deliberately more lossy than the runtime local-Vec3f authoring path; this
# exercises the frozen conservative closure bound.
CONSERVATIVE_370_WORLD_FLOAT32_VOLUME_RESIDUAL_M3 = 2.5138930311876216e-11


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
    target_pair = {
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
    receipt: dict[str, object] = {
        "schema_version": 17,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "insufficient_evidence",
        "containment": {"vial_interior_bounds_m": dict(GEOMETRY)},
        "captures": [
            {
                "physics_step": TARGET_STEP,
                "volume_gate": target_gate,
            }
        ],
        "semantics": {"pairs": [target_pair]},
        "qualification": {
            "first_stage_only": True,
            "status": "qualification_gate_failed",
            "target_step": TARGET_STEP,
            "volume_state_gate_passed": state_passed,
            "volume_visible_gate_passed": visible_passed,
            "volume_gate_passed": gate_passed,
            "target_volume_evidence": target_gate,
            "paired_capture_evidence": [target_pair],
        },
    }
    receipt_path = app_output / "receipt.json"
    _save_receipt(receipt_path, receipt)
    return receipt_path, rgb_path, mask_path, receipt


def _add_count_driven_body_evidence(receipt: dict[str, object]) -> None:
    """Add independently specified A31 target-stage evidence to a test receipt."""

    target_count = 370
    target_volume_m3 = target_count * PARTICLE_VOLUME_M3
    target_height_m = target_volume_m3 / GEOMETRY["cross_section_area_m2"]
    bottom_z_m = GEOMETRY["bottom_z_m"]
    receipt["qualification"]["target_volume_evidence"][
        "expected_cumulative_particle_count"
    ] = target_count
    extent_world_m = [
        [5.2194, -1.9942, bottom_z_m],
        [5.2454, -1.9682, bottom_z_m + target_height_m],
    ]
    receipt["render_path"] = COUNT_DRIVEN_RENDER_PATH
    receipt["containment"]["capture_records"] = [
        {
            "physics_step": TARGET_STEP,
            "emitted_particle_count": target_count,
            "active_particle_count": target_count,
            "inside_vial_interior_count": target_count,
            "in_flight_above_mouth_count": 0,
            "outside_or_escaped_particle_count": 0,
            "finite_state": True,
        }
    ]
    stage_geometry = {
        "physics_step": TARGET_STEP,
        "contained_particle_count": target_count,
        "target_volume_m3": target_volume_m3,
        "target_height_m": target_height_m,
        "world_bottom_z_m": bottom_z_m,
        "world_extent_m": extent_world_m,
        "post_transform_signed_volume_m3": (
            target_volume_m3 + CONSERVATIVE_370_WORLD_FLOAT32_VOLUME_RESIDUAL_M3
        ),
        "post_transform_volume_residual_m3": (
            CONSERVATIVE_370_WORLD_FLOAT32_VOLUME_RESIDUAL_M3
        ),
        "points_sha256": "1" * 64,
        "extent_sha256": "2" * 64,
    }
    receipt["count_driven_liquid_body"] = {
        "representation": COUNT_DRIVEN_RENDER_PATH,
        "body_mesh_path": (
            "/FillStation/FillStation_LocalRoot/"
            "FluidmixF2CountDrivenLiquid/Fill_liquid"
        ),
        "source_world_bottom_z_m": bottom_z_m,
        "particle_volume_m3": PARTICLE_VOLUME_M3,
        "cross_section_area_m2": GEOMETRY["cross_section_area_m2"],
        "sphere_instances_hidden": True,
        "extent_recomputed_each_update": True,
        "semantic_label": "liquid",
        "stage_geometry": [stage_geometry],
    }
    pair = receipt["qualification"]["paired_capture_evidence"][0]
    pair["liquid_representation"] = COUNT_DRIVEN_RENDER_PATH
    paired = pair["paired_capture_state_evidence"]
    for field, digest in {
        "body_points": "3" * 64,
        "body_extent": "4" * 64,
        "body_visibility": "5" * 64,
        "body_material_color": "6" * 64,
        "stream_visibility": "7" * 64,
    }.items():
        paired[f"{field}_sha256_before"] = digest
        paired[f"{field}_sha256_after"] = digest
    receipt["semantics"].update(
        {
            "count_driven_body_semantic_bound": True,
            "point_instancer_rgb_visible": False,
            "point_instancer_semantic_visible": False,
            "hidden_particle_control": {
                "physics_step": TARGET_STEP,
                "emitted_particle_count": target_count,
                "inside_vial_interior_count": target_count,
                "body_and_stream_hidden_liquid_pixel_count": 0,
                "hidden_sphere_prototypes": True,
                "control_semantic_path": "semantic-controls/emitted-hidden.npy",
                "control_semantic_sha256": "9" * 64,
                "body_only_target_path": receipt["count_driven_liquid_body"][
                    "body_mesh_path"
                ],
                "body_only_target_ids": [7],
                "body_only_capture_step": TARGET_STEP,
                "body_only_liquid_pixel_count": 123,
                "passed": True,
            },
        }
    )


def test_count_driven_review_rejects_receipt_without_body_evidence(tmp_path: Path) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    receipt["render_path"] = COUNT_DRIVEN_RENDER_PATH
    _save_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="count-driven"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )

    assert not (tmp_path / "review").exists()


def test_count_driven_review_binds_live_count_volume_geometry_and_isolation(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(
        tmp_path,
        state_passed=True,
    )
    _add_count_driven_body_evidence(receipt)
    _save_receipt(receipt_path, receipt)

    metrics = review.build_volume_review(
        receipt_path=receipt_path,
        rgb_path=rgb_path,
        liquid_mask_path=mask_path,
        output_dir=tmp_path / "review",
    )

    body = metrics["count_driven_body_evidence"]
    assert body["physics_step"] == TARGET_STEP
    assert body["live_contained_particle_count"] == 370
    assert body["target_volume_m3"] == pytest.approx(9.99e-6)
    assert body["target_height_m"] == pytest.approx(
        9.99e-6 / GEOMETRY["cross_section_area_m2"]
    )
    assert body["post_transform_volume_residual_m3"] == pytest.approx(
        CONSERVATIVE_370_WORLD_FLOAT32_VOLUME_RESIDUAL_M3
    )
    assert body["world_bottom_z_m"] == GEOMETRY["bottom_z_m"]
    assert body["hidden_particle_liquid_pixel_count"] == 0
    assert set(body["paired_render_state_sha256"]) == {
        "body_points",
        "body_extent",
        "body_visibility",
        "body_material_color",
        "stream_visibility",
    }
    saved = json.loads(
        (tmp_path / "review" / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved["count_driven_body_evidence"] == body


@pytest.mark.parametrize(
    "case",
    (
        "duplicate_containment_row",
        "emitted_count_mismatch",
        "contained_count_mismatch",
        "target_volume_mismatch",
        "signed_volume_mismatch",
        "residual_too_large",
        "bottom_moved",
        "extent_height_mismatch",
        "spheres_visible",
        "body_semantics_missing",
        "hidden_control_has_liquid",
        "hidden_control_wrong_step",
        "hidden_control_wrong_count",
        "paired_body_hash_missing",
        "paired_body_hash_changed",
        "duplicate_stage_geometry_row",
    ),
)
def test_count_driven_review_rejects_incomplete_or_contradictory_body_evidence(
    tmp_path: Path,
    case: str,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(
        tmp_path,
        state_passed=True,
    )
    _add_count_driven_body_evidence(receipt)
    containment_rows = receipt["containment"]["capture_records"]
    body = receipt["count_driven_liquid_body"]
    stage = body["stage_geometry"][0]
    semantics = receipt["semantics"]
    paired = receipt["qualification"]["paired_capture_evidence"][0][
        "paired_capture_state_evidence"
    ]

    if case == "duplicate_containment_row":
        containment_rows.append(dict(containment_rows[0]))
    elif case == "emitted_count_mismatch":
        containment_rows[0]["emitted_particle_count"] = 369
    elif case == "contained_count_mismatch":
        stage["contained_particle_count"] = 369
    elif case == "target_volume_mismatch":
        stage["target_volume_m3"] += 1.0e-9
    elif case == "signed_volume_mismatch":
        stage["post_transform_signed_volume_m3"] += 1.0e-9
    elif case == "residual_too_large":
        stage["post_transform_volume_residual_m3"] = 3.0e-10
    elif case == "bottom_moved":
        stage["world_bottom_z_m"] += 2.0e-6
    elif case == "extent_height_mismatch":
        stage["world_extent_m"][1][2] += 2.0e-6
    elif case == "spheres_visible":
        body["sphere_instances_hidden"] = False
    elif case == "body_semantics_missing":
        semantics["count_driven_body_semantic_bound"] = False
    elif case == "hidden_control_has_liquid":
        semantics["hidden_particle_control"][
            "body_and_stream_hidden_liquid_pixel_count"
        ] = 1
    elif case == "hidden_control_wrong_step":
        semantics["hidden_particle_control"]["physics_step"] = TARGET_STEP - 1
    elif case == "hidden_control_wrong_count":
        semantics["hidden_particle_control"]["emitted_particle_count"] = 369
    elif case == "paired_body_hash_missing":
        del paired["body_points_sha256_before"]
    elif case == "paired_body_hash_changed":
        paired["body_points_sha256_after"] = "8" * 64
    elif case == "duplicate_stage_geometry_row":
        body["stage_geometry"].append(dict(stage))
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)
    _save_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="count-driven"):
        review.build_volume_review(
            receipt_path=receipt_path,
            rgb_path=rgb_path,
            liquid_mask_path=mask_path,
            output_dir=tmp_path / "review",
        )

    assert not (tmp_path / "review").exists()


def test_review_explicit_target_step_binds_full_run_semantics_and_required_capture(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    receipt["qualification"] = {
        "first_stage_only": False,
        "status": "not_run",
    }
    _save_receipt(receipt_path, receipt)

    metrics = review.build_volume_review(
        receipt_path=receipt_path,
        rgb_path=rgb_path,
        liquid_mask_path=mask_path,
        output_dir=tmp_path / "review",
        target_step=TARGET_STEP,
    )

    assert metrics["source_evidence"]["target_step"] == TARGET_STEP
    assert metrics["source_evidence"]["target_binding_source"] == (
        "semantics.pairs+captures.volume_gate"
    )
    assert metrics["runner_decision"]["qualification_status"] == "not_run"
    assert metrics["runner_decision"]["volume_gate_passed"] is False


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


def test_existing_point_instancer_review_does_not_require_count_driven_area(
    tmp_path: Path,
) -> None:
    receipt_path, rgb_path, mask_path, receipt = _write_fixture(tmp_path)
    del receipt["containment"]["vial_interior_bounds_m"]["cross_section_area_m2"]
    _save_receipt(receipt_path, receipt)

    metrics = review.build_volume_review(
        receipt_path=receipt_path,
        rgb_path=rgb_path,
        liquid_mask_path=mask_path,
        output_dir=tmp_path / "review",
    )

    assert "count_driven_body_evidence" not in metrics


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

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fluidmix.wp7_fill_step_tracker import (
    ACTIVE_TANK_CYCLE,
    CALIBRATION_SCHEMA,
    FROZEN_FILLCAM_CAMERA_SHA256,
    FROZEN_GEOMETRY_SHA256,
    FROZEN_VOLUME_CURVE_SHA256,
    FillStepTrackerError,
    FrameEvidence,
    PixelToMetreCalibration,
    RecipePour,
    TrackerFreeze,
    adapt_runtime_frames,
    build_fill_step_receipt,
    calibration_payload_sha256,
    development_support_summary,
    track_fill_step,
    validate_fill_step_receipt,
)


def _calibration() -> PixelToMetreCalibration:
    calibration = PixelToMetreCalibration(
        schema_version=CALIBRATION_SCHEMA,
        calibration_id="fillcam-projected-collider-v1",
        camera_effective_state_sha256=FROZEN_FILLCAM_CAMERA_SHA256,
        geometry_sha256=FROZEN_GEOMETRY_SHA256,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
        fill_height_px_samples=(44.257340, 68.493011),
        visible_height_m_samples=(0.019122235, 0.038244470),
        twin_geometry_equivalent_volume_ml_samples=(10.0, 20.0),
        artifact_sha256="0" * 64,
    )
    return replace(calibration, artifact_sha256=calibration_payload_sha256(calibration))


def _freeze() -> TrackerFreeze:
    return TrackerFreeze(
        stream_roi_xyxy=(300, 80, 320, 190),
        stream_rgb_change_threshold=8,
        stream_minimum_changed_pixels=600,
        stream_minimum_consecutive_frames=2,
        stream_change_delta_e76=8.0,
        tank_cluster_delta_e76=3.0,
        volume_match_tolerance_ml=0.25,
        calibration=_calibration(),
        tank_lab_centres={
            "t100": (50.0, 0.0, 0.0),
            "t200": (50.0, 20.0, 0.0),
            "t400": (50.0, 0.0, 20.0),
        },
        recipe_artifact_sha256="b" * 64,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
    )


def _frame(
    index: int,
    lab: tuple[float, float, float],
    height_px: float,
    pour_index: int,
) -> FrameEvidence:
    return FrameEvidence(index, lab, (0.2, 0.3, 0.4), height_px, pour_index)


def _two_stage_frames(second_height_px: float = 68.493011) -> list[FrameEvidence]:
    return [
        _frame(10, (50.0, 0.0, 0.0), 44.257340, 1),
        _frame(11, (50.1, 0.0, 0.0), 44.257340, 1),
        _frame(12, (50.0, 0.1, 0.0), 44.257340, 1),
        _frame(20, (50.0, 20.0, 0.0), 55.0, 2),
        _frame(21, (50.1, 20.0, 0.0), 62.0, 2),
        _frame(22, (50.0, 20.1, 0.0), second_height_px, 2),
    ]


def _pours() -> tuple[RecipePour, ...]:
    return (
        RecipePour("t100", 10.0),
        RecipePour("t200", 10.0),
        RecipePour("t400", 10.0),
    )


def test_stream_changes_map_exact_cycle_and_select_each_stage_frames() -> None:
    result = track_fill_step(
        _two_stage_frames(), _pours(), _freeze(), previous_final_tank_id="t400"
    )

    assert ACTIVE_TANK_CYCLE == ("t100", "t200", "t400")
    assert result.tank_number == 200
    assert result.tank_id == "t200"
    assert result.stream_step_index == 2
    assert result.measured_volume_ml == pytest.approx(20.0, abs=0.002)
    assert result.expected_cumulative_ml == 20.0
    assert result.fill_step == 2
    assert result.agreement is True
    assert result.tank_cluster_consistent is True
    assert [(item.fill_step, item.early_frame_index, item.mid_frame_index,
             item.end_frame_index) for item in result.selected_frames] == [
        (1, 10, 11, 12),
        (2, 20, 21, 22),
    ]
    assert result.limit_codes == ()


def test_volume_stream_disagreement_is_the_only_state_abstention() -> None:
    # The final mask height is the independently projected 10 mL surface, while
    # stream colour has advanced to step 2 (20 mL cumulative).
    result = track_fill_step(
        _two_stage_frames(second_height_px=44.257340),
        _pours(),
        _freeze(),
        previous_final_tank_id="t400",
    )

    assert result.stream_step_index == 2
    assert result.measured_volume_ml == pytest.approx(10.0, abs=0.002)
    assert result.fill_step is None
    assert result.agreement is False
    assert result.limit_codes == ("stream_volume_step_disagreement",)


def test_tank_cluster_inconsistency_is_flagged_without_changing_step_count() -> None:
    frames = _two_stage_frames()
    frames[3:] = [replace(item, stream_lab=(50.0, 27.0, 0.0)) for item in frames[3:]]

    result = track_fill_step(
        frames, _pours(), _freeze(), previous_final_tank_id="t400"
    )

    assert result.stream_step_index == 2
    assert result.tank_id == "t200"
    assert result.tank_cluster_consistent is False
    assert result.limit_codes == ("stream_lab_tank_cluster_inconsistent",)


@pytest.mark.parametrize(
    "freeze",
    [
        replace(_freeze(), calibration=None),
        replace(_freeze(), calibration=replace(_calibration(), artifact_sha256="bad")),
        replace(
            _freeze(),
            calibration=replace(_calibration(), visible_height_m_samples=(0.0, 0.1)),
        ),
        replace(
            _freeze(),
            calibration=replace(
                _calibration(), camera_effective_state_sha256="0" * 64
            ),
        ),
        replace(_freeze(), volume_curve_sha256="0" * 64),
    ],
)
def test_missing_or_drifted_pixel_to_metre_calibration_fails_closed(freeze) -> None:
    with pytest.raises(FillStepTrackerError, match="calibration|camera|curve"):
        track_fill_step(
            _two_stage_frames(), _pours(), freeze, previous_final_tank_id="t400"
        )


def test_recipe_requires_exact_three_tank_cycle_and_positive_amounts() -> None:
    with pytest.raises(FillStepTrackerError, match="exact active-tank cycle"):
        track_fill_step(
            _two_stage_frames(),
            (RecipePour("t100", 10.0), RecipePour("t300", 10.0)),
            _freeze(),
            previous_final_tank_id="t400",
        )
    with pytest.raises(FillStepTrackerError, match="positive"):
        track_fill_step(
            _two_stage_frames(),
            (RecipePour("t100", 10.0), RecipePour("t200", 0.0)),
            _freeze(),
            previous_final_tank_id="t400",
        )


def test_adapter_joins_runtime_cnn_surface_to_path_free_stream_rows() -> None:
    from fluidmix.wp7_rgb_runtime import RuntimeVisionMeasurement

    runtime = (
        RuntimeVisionMeasurement(
            image_id="IMG-1",
            case_id="CASE-1",
            frame_index=12,
            status="visible_measured",
            surface_row=361,
            visible_height_px=44,
            mask_sha256="c" * 64,
            limit_codes=(),
            citation_sha256="d" * 64,
        ),
    )
    rows = (
        {
            "image_id": "IMG-1",
            "frame_index": 12,
            "pour_index": 1,
            "median_lab": [50.0, 0.0, 0.0],
            "spread_lab": [0.2, 0.3, 0.4],
        },
    )

    adapted = adapt_runtime_frames(runtime, rows)

    assert adapted == (
        FrameEvidence(12, (50.0, 0.0, 0.0), (0.2, 0.3, 0.4), 44.0, 1),
    )
    assert not hasattr(adapted[0], "path")


def test_adapter_rejects_unbound_or_unaccepted_mask_measurement() -> None:
    from fluidmix.wp7_rgb_runtime import RuntimeVisionMeasurement

    invalid = RuntimeVisionMeasurement(
        image_id="IMG-1",
        case_id="CASE-1",
        frame_index=12,
        status="empty_mask_abstain",
        surface_row=None,
        visible_height_px=None,
        mask_sha256="c" * 64,
        limit_codes=("empty_mask",),
        citation_sha256="d" * 64,
    )
    with pytest.raises(FillStepTrackerError, match="accepted visible CNN mask"):
        adapt_runtime_frames((invalid,), ())


def test_recipe_cycle_can_start_at_recipe_bound_t200_offset() -> None:
    pours = (
        RecipePour("t200", 10.0),
        RecipePour("t400", 10.0),
        RecipePour("t100", 10.0),
    )
    freeze = replace(
        _freeze(),
        tank_lab_centres={
            "t100": (50.0, 0.0, 20.0),
            "t200": (50.0, 0.0, 0.0),
            "t400": (50.0, 20.0, 0.0),
        },
    )

    result = track_fill_step(
        _two_stage_frames(), pours, freeze, previous_final_tank_id="t100"
    )

    assert result.tank_id == "t400"
    assert [stage.tank_id for stage in result.stages] == ["t200", "t400"]


def test_residual_states_and_boundary_repeat_do_not_advance_dispense_step() -> None:
    pours = (
        RecipePour("t200", 10.0),
        RecipePour("t400", 10.0),
        RecipePour("t100", 10.0),
    )
    frames = [
        _frame(1, (50.0, 0.0, 0.0), 44.257340, 1),  # T100 residual
        _frame(2, (50.1, 0.0, 0.0), 44.257340, 1),
        _frame(3, (50.0, 0.0, 20.0), 44.257340, 1),  # one-frame noise
        _frame(4, (50.0, 20.0, 0.0), 44.257340, 1),  # reaches T200
        _frame(5, (50.1, 20.0, 0.0), 44.257340, 1),
        _frame(6, (50.0, 20.1, 0.0), 44.257340, 1),
        _frame(10, (50.0, 20.0, 0.0), 50.0, 2),  # repeated T200 residual
        _frame(11, (50.1, 20.0, 0.0), 52.0, 2),
        _frame(12, (50.0, 0.0, 20.0), 60.0, 2),  # reaches T400
        _frame(13, (50.0, 0.1, 20.0), 65.0, 2),
        _frame(14, (50.0, 0.0, 20.1), 68.493011, 2),
    ]

    result = track_fill_step(
        frames, pours, _freeze(), previous_final_tank_id="t100"
    )

    assert result.stream_step_index == 2
    assert result.tank_id == "t400"
    assert [(stage.tank_id, stage.selected_frames.early_frame_index) for stage in result.stages] == [
        ("t200", 4),
        ("t400", 12),
    ]


def test_zero_residual_first_pour_advances_when_expected_tank_is_first_state() -> None:
    pours = (
        RecipePour("t200", 10.0),
        RecipePour("t400", 10.0),
    )
    frames = [
        _frame(1, (50.0, 20.0, 0.0), 44.257340, 1),
        _frame(2, (50.1, 20.0, 0.0), 44.257340, 1),
        _frame(3, (50.0, 20.1, 0.0), 44.257340, 1),
    ]

    result = track_fill_step(
        frames, pours, _freeze(), previous_final_tank_id="t100"
    )

    assert result.stream_step_index == 1
    assert result.tank_id == "t200"
    assert result.fill_step == 1
    assert result.selected_frames[0].early_frame_index == 1


def test_previous_final_tank_must_bind_recipe_cycle_offset() -> None:
    with pytest.raises(FillStepTrackerError, match="previous final tank"):
        track_fill_step(
            _two_stage_frames(),
            _pours(),
            _freeze(),
            previous_final_tank_id="t100",
        )


def test_closed_receipt_binds_stage_images_and_self_hash() -> None:
    result = track_fill_step(
        _two_stage_frames(), _pours(), _freeze(), previous_final_tank_id="t400"
    )
    receipt = build_fill_step_receipt(
        cohort_id="COHORT-1",
        vial_id="VIAL-1",
        result=result,
        freeze=_freeze(),
        image_id_by_frame={
            10: "IMG-10", 11: "IMG-11", 12: "IMG-12",
            20: "IMG-20", 21: "IMG-21", 22: "IMG-22",
        },
        source_artifact_sha256="e" * 64,
    )

    assert receipt["claim_boundary"] == (
        "camera_twin_geometry_equivalent_not_physical_ground_truth"
    )
    assert receipt["stages"][1]["images"] == {
        "early": "IMG-20", "mid": "IMG-21", "end": "IMG-22"
    }
    assert receipt["stages"][1]["measured_volume_ml"] == receipt["stages"][1][
        "twin_geometry_equivalent_volume_ml"
    ]
    assert validate_fill_step_receipt(receipt) == receipt
    drifted = dict(receipt)
    drifted["cohort_id"] = "COHORT-2"
    with pytest.raises(FillStepTrackerError, match="hash mismatch"):
        validate_fill_step_receipt(drifted)


def test_disagreement_cannot_enter_vlm_receipt() -> None:
    result = track_fill_step(
        _two_stage_frames(second_height_px=44.257340),
        _pours(),
        _freeze(),
        previous_final_tank_id="t400",
    )
    with pytest.raises(FillStepTrackerError, match="disagreed"):
        build_fill_step_receipt(
            cohort_id="COHORT-1",
            vial_id="VIAL-1",
            result=result,
            freeze=_freeze(),
            image_id_by_frame={},
            source_artifact_sha256="e" * 64,
        )


def test_support_summary_publishes_identities_not_invented_counts() -> None:
    summary = development_support_summary(
        total_vial_ids=("V1", "V2", "V3"),
        before_supported_vial_ids=("V1",),
        after_supported_vial_ids=("V1", "V3"),
    )
    assert summary["before_supported_vial_count"] == 1
    assert summary["after_supported_vial_count"] == 2
    with pytest.raises(FillStepTrackerError, match="identities"):
        development_support_summary(
            total_vial_ids=("V1",),
            before_supported_vial_ids=(),
            after_supported_vial_ids=("V2",),
        )


def test_production_tracker_has_no_part_a_import() -> None:
    source = Path("src/fluidmix/wp7_fill_step_tracker.py").read_text(encoding="utf-8")
    assert "from .physx_volume_gate import" not in source
    assert "from .volume_height_curve import" not in source

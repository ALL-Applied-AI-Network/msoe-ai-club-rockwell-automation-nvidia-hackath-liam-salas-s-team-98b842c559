from __future__ import annotations

from dataclasses import replace
import inspect
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
    select_stage_frame_candidates,
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
) -> FrameEvidence:
    return FrameEvidence(
        index, lab, (0.2, 0.3, 0.4), height_px,
        surface_row=417.0 - height_px,
    )


def _two_stage_frames(second_height_px: float = 68.493011) -> list[FrameEvidence]:
    return [
        _frame(10, (50.0, 0.0, 0.0), 44.257340),
        _frame(11, (50.1, 0.0, 0.0), 44.257340),
        _frame(12, (50.0, 0.1, 0.0), 44.257340),
        _frame(20, (50.0, 20.0, 0.0), second_height_px),
        _frame(21, (50.1, 20.0, 0.0), second_height_px),
        _frame(22, (50.0, 20.1, 0.0), second_height_px),
    ]


def test_volume_uses_last_stable_three_frame_window_not_unstable_final_frame() -> None:
    calibration = replace(
        _calibration(),
        fill_height_px_samples=(50.0, 100.0, 200.0),
        visible_height_m_samples=(0.01, 0.02, 0.03),
        twin_geometry_equivalent_volume_ml_samples=(5.0, 10.0, 20.0),
        artifact_sha256="0" * 64,
    )
    calibration = replace(
        calibration, artifact_sha256=calibration_payload_sha256(calibration)
    )
    frames = [
        FrameEvidence(index, (50.0, 0.0, 0.0), (0.2, 0.3, 0.4), height, surface)
        for index, height, surface in (
            (10, 99.0, 318.0), (11, 100.0, 317.0),
            (12, 101.0, 316.0), (13, 200.0, 217.0),
        )
    ]

    result = track_fill_step(
        frames,
        (RecipePour("t100", 10.0),),
        replace(_freeze(), calibration=calibration),
        previous_final_tank_id="t400",
    )

    assert result.measured_volume_ml == pytest.approx(10.0)
    assert result.selected_frames[0].end_frame_index == 13


@pytest.mark.parametrize(
    ("frame_rate_hz", "debounce_frames", "stable_frames"),
    [(5.0, 6, 2), (10.0, 12, 3), (20.0, 24, 6), (30.0, 36, 9)],
)
def test_frozen_tracker_windows_preserve_their_timed_duration(
    frame_rate_hz: float, debounce_frames: int, stable_frames: int
) -> None:
    freeze = replace(_freeze(), stream_minimum_consecutive_frames=12)

    def timed_frames(stable_tail: int) -> list[FrameEvidence]:
        unstable_count = debounce_frames - stable_tail
        heights = [50.0 if index % 2 == 0 else 60.0 for index in range(unstable_count)]
        heights.extend([44.257340] * stable_tail)
        return [
            _frame(index, (50.0, 0.0, 0.0), height)
            for index, height in enumerate(heights)
        ]

    frames = timed_frames(stable_frames)

    result = track_fill_step(
        frames,
        (RecipePour("t100", 10.0),),
        freeze,
        previous_final_tank_id="t400",
        frame_rate_hz=frame_rate_hz,
    )

    assert result.fill_step == 1
    assert result.selected_frames[0].end_frame_index == debounce_frames - 1
    with pytest.raises(FillStepTrackerError, match="stable.*window"):
        track_fill_step(
            timed_frames(stable_frames - 1),
            (RecipePour("t100", 10.0),),
            freeze,
            previous_final_tank_id="t400",
            frame_rate_hz=frame_rate_hz,
        )
    with pytest.raises(FillStepTrackerError, match="never reaches"):
        track_fill_step(
            [
                _frame(index, (50.0, 0.0, 0.0), 44.257340)
                for index in range(debounce_frames - 1)
            ],
            (RecipePour("t100", 10.0),),
            freeze,
            previous_final_tank_id="t400",
            frame_rate_hz=frame_rate_hz,
        )


@pytest.mark.parametrize("frame_rate_hz", [0.0, -1.0, float("inf"), float("nan"), True])
def test_tracker_rejects_nonpositive_or_nonfinite_frame_rate(frame_rate_hz) -> None:
    with pytest.raises(FillStepTrackerError, match="frame rate"):
        track_fill_step(
            _two_stage_frames(),
            _pours(),
            _freeze(),
            previous_final_tank_id="t400",
            frame_rate_hz=frame_rate_hz,
        )


def test_stage_without_stable_three_frame_height_surface_window_fails_closed() -> None:
    frames = [
        FrameEvidence(index, (50.0, 0.0, 0.0), (0.2, 0.3, 0.4), height, surface)
        for index, height, surface in (
            (10, 44.257340, 372.0), (11, 50.0, 360.0),
            (12, 60.0, 350.0), (13, 68.493011, 340.0),
        )
    ]

    with pytest.raises(FillStepTrackerError, match="stable.*window"):
        track_fill_step(
            frames,
            (RecipePour("t100", 10.0),),
            _freeze(),
            previous_final_tank_id="t400",
        )


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


def test_stage_candidates_are_camera_state_only_and_do_not_require_cnn_height() -> None:
    frames = [replace(frame, visible_height_px=None) for frame in _two_stage_frames()]

    selected = select_stage_frame_candidates(
        frames,
        _pours(),
        _freeze(),
        previous_final_tank_id="t400",
    )

    assert [(item.fill_step, item.early_frame_index, item.mid_frame_index,
             item.end_frame_index) for item in selected] == [
        (1, 10, 11, 12),
        (2, 20, 21, 22),
    ]


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
    assert result.agreement is False
    assert result.fill_step is None
    assert result.limit_codes == ("stream_lab_tank_cluster_inconsistent",)
    with pytest.raises(FillStepTrackerError, match="disagreed"):
        build_fill_step_receipt(
            cohort_id="COHORT-1",
            vial_id="VIAL-1",
            result=result,
            freeze=_freeze(),
            image_id_by_frame={},
            source_artifact_sha256="e" * 64,
        )


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


def test_recipe_rejects_invalid_tank_and_nonpositive_amounts() -> None:
    with pytest.raises(FillStepTrackerError, match="active-tank"):
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
            "median_lab": [50.0, 0.0, 0.0],
            "spread_lab": [0.2, 0.3, 0.4],
        },
    )

    adapted = adapt_runtime_frames(runtime, rows)

    assert adapted == (
        FrameEvidence(
            12, (50.0, 0.0, 0.0), (0.2, 0.3, 0.4), 44.0,
            surface_row=361.0,
        ),
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
        _frame(1, (50.0, 0.0, 0.0), 44.257340),  # T100 residual
        _frame(2, (50.1, 0.0, 0.0), 44.257340),
        _frame(3, (50.0, 0.0, 20.0), 44.257340),  # one-frame noise
        _frame(4, (50.0, 20.0, 0.0), 44.257340),  # reaches T200
        _frame(5, (50.1, 20.0, 0.0), 44.257340),
        _frame(6, (50.0, 20.1, 0.0), 44.257340),
        _frame(10, (50.0, 20.0, 0.0), 44.257340),  # same state across boundary
        _frame(11, (50.1, 20.0, 0.0), 44.257340),
        _frame(12, (50.0, 0.0, 20.0), 68.493011),  # reaches T400
        _frame(13, (50.0, 0.1, 20.0), 68.493011),
        _frame(14, (50.0, 0.0, 20.1), 68.493011),
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
        _frame(1, (50.0, 20.0, 0.0), 44.257340),
        _frame(2, (50.1, 20.0, 0.0), 44.257340),
        _frame(3, (50.0, 20.1, 0.0), 44.257340),
    ]

    result = track_fill_step(
        frames, pours, _freeze(), previous_final_tank_id="t100"
    )

    assert result.stream_step_index == 1
    assert result.tank_id == "t200"
    assert result.fill_step == 1
    assert result.selected_frames[0].early_frame_index == 1


def test_previous_final_tank_must_be_active() -> None:
    with pytest.raises(FillStepTrackerError, match="previous final tank"):
        track_fill_step(
            _two_stage_frames(),
            _pours(),
            _freeze(),
            previous_final_tank_id="t300",
        )


@pytest.mark.parametrize(
    ("tank_order", "previous_final_tank_id", "labs"),
    [
        (("t200", "t100"), "t100", ((50.0, 20.0, 0.0), (50.0, 0.0, 0.0))),
        (("t400", "t100"), "t100", ((50.0, 0.0, 20.0), (50.0, 0.0, 0.0))),
    ],
)
def test_recipe_accepts_cyclic_subsequence_with_zero_ml_tank_omitted(
    tank_order, previous_final_tank_id, labs
) -> None:
    pours = tuple(RecipePour(tank_id, 10.0) for tank_id in tank_order)
    frames = [
        _frame(1, labs[0], 44.257340),
        _frame(2, labs[0], 44.257340),
        _frame(3, labs[0], 44.257340),
        _frame(4, labs[1], 68.493011),
        _frame(5, labs[1], 68.493011),
        _frame(6, labs[1], 68.493011),
    ]

    result = track_fill_step(
        frames,
        pours,
        _freeze(),
        previous_final_tank_id=previous_final_tank_id,
    )

    assert [stage.tank_id for stage in result.stages] == list(tank_order)
    assert result.stream_step_index == 2
    receipt = build_fill_step_receipt(
        cohort_id="COHORT-OMITTED-ZERO",
        vial_id="VIAL-1",
        result=result,
        freeze=_freeze(),
        image_id_by_frame={index: f"IMG-{index}" for index in range(1, 7)},
        source_artifact_sha256="e" * 64,
    )
    assert [stage["tank_id"] for stage in receipt["stages"]] == list(tank_order)


@pytest.mark.parametrize(
    "tank_order",
    [
        ("t100", "t100"),
        ("t100", "t400", "t200"),
        ("t100", "t300"),
    ],
)
def test_recipe_rejects_duplicate_out_of_order_or_invalid_tanks(tank_order) -> None:
    with pytest.raises(FillStepTrackerError, match="active-tank|cyclic|duplicate"):
        track_fill_step(
            _two_stage_frames(),
            tuple(RecipePour(tank_id, 10.0) for tank_id in tank_order),
            _freeze(),
            previous_final_tank_id="t400",
        )


def test_camera_evidence_contract_has_no_external_step_ordinal() -> None:
    assert "pour_index" not in inspect.signature(FrameEvidence).parameters
    from fluidmix.wp7_rgb_runtime import RuntimeVisionMeasurement

    measurement = RuntimeVisionMeasurement(
        image_id="IMG-1",
        case_id="CASE-1",
        frame_index=1,
        status="visible_measured",
        surface_row=361,
        visible_height_px=44,
        mask_sha256="c" * 64,
        limit_codes=(),
        citation_sha256="d" * 64,
    )
    with pytest.raises(FillStepTrackerError, match="ordinal"):
        adapt_runtime_frames(
            (measurement,),
            ({
                "image_id": "IMG-1",
                "frame_index": 1,
                "pour_index": 1,
                "median_lab": [50.0, 0.0, 0.0],
                "spread_lab": [0.2, 0.3, 0.4],
            },),
        )


@pytest.mark.parametrize(
    "labs",
    [
        ((50.0, 0.0, 0.0), (50.0, 0.0, 20.0)),  # skips expected T200
        ((50.0, 20.0, 0.0),),  # neither initial residual T400 nor expected T100
    ],
)
def test_skipped_or_out_of_order_stable_tank_state_fails_closed(labs) -> None:
    frames = []
    frame_index = 1
    for lab in labs:
        frames.extend(
            [
                _frame(frame_index, lab, 44.257340),
                _frame(frame_index + 1, lab, 44.257340),
            ]
        )
        frame_index += 2
    with pytest.raises(FillStepTrackerError, match="recipe order"):
        track_fill_step(
            frames, _pours(), _freeze(), previous_final_tank_id="t400"
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

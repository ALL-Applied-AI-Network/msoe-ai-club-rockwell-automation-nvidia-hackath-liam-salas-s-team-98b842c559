from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fluidmix.empty_reference_rgb_predictor import fillcam_masks
from fluidmix.wp7_fill_step_production import (
    IncrementalFillStepTracker,
    ProductionFillStepError,
    TopologyHeightMeasurement,
    TopologyFreeze,
    load_pixel_to_metre_calibration,
    load_tracker_freeze,
    load_topology_freeze,
    persist_completed_observations,
    postprocess_cnn_mask_topology,
)
from fluidmix.wp7_fill_step_tracker import (
    CALIBRATION_SCHEMA,
    FROZEN_FILLCAM_CAMERA_SHA256,
    FROZEN_GEOMETRY_SHA256,
    FROZEN_VOLUME_CURVE_SHA256,
    FrameEvidence,
    PixelToMetreCalibration,
    RecipePour,
    TrackerFreeze,
    calibration_payload_sha256,
)


def _fragmented_runtime_tracker(
    *,
    frame_rate_hz: float = 10.0,
    recipe_pours: tuple[RecipePour, ...] = (
        RecipePour("t100", 10.0),
        RecipePour("t200", 10.0),
        RecipePour("t400", 10.0),
    ),
) -> IncrementalFillStepTracker:
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = IncrementalFillStepTracker(
        freeze=replace(_freeze(), stream_minimum_consecutive_frames=12),
        topology=_topology(), recipe_pours=recipe_pours,
        previous_final_tank_id="t400",
        same_run_empty_reference_rgb=reference,
        predictor=lambda *_args: None, cohort_id="COHORT-1", case_id="CASE-1",
        vial_id="VIAL-1", vial_position=1, source_artifact_sha256="e" * 64,
        frame_rate_hz=frame_rate_hz,
    )
    tracker._replay_temporal_heights = lambda: None
    return tracker


def _seed_fragmented_stage(
    tracker: IncrementalFillStepTracker,
    *,
    tank_id: str,
    frame_count: int,
    visible_height_px: float,
    wrong_tank_at: int | None = None,
    wrong_tank_id: str = "t100",
) -> None:
    start = len(tracker._frames)
    centre = tracker.freeze.tank_lab_centres[tank_id]
    for offset in range(frame_count):
        frame_index = start + offset
        selected_centre = (
            tracker.freeze.tank_lab_centres[wrong_tank_id]
            if offset == wrong_tank_at
            else centre
        )
        # Adjacent samples exceed the frozen raw-change threshold while each
        # remains nearest to the same frozen tank centre.
        delta_l, delta_a = (
            (5.0, 0.0), (-2.5, 4.5), (-2.5, -4.5)
        )[offset % 3]
        lab = (
            selected_centre[0] + delta_l,
            selected_centre[1] + delta_a,
            selected_centre[2],
        )
        surface_row = 417.0 - visible_height_px
        tracker._frames.append(FrameEvidence(
            frame_index=frame_index,
            stream_lab=lab,
            stream_spread_lab=(float(offset % 3), 2.0, 3.0),
            visible_height_px=visible_height_px,
            surface_row=surface_row,
        ))
        tracker._captures[frame_index] = (
            np.full((480, 640, 3), frame_index, dtype=np.uint8),
            np.zeros((480, 640), dtype=bool),
        )
        tracker._height_by_frame[frame_index] = TopologyHeightMeasurement(
            "visible_measured", int(surface_row), int(visible_height_px), "cnn_mask"
        )


def test_close_recovers_ordered_stages_from_same_tank_lab_fragments() -> None:
    tracker = _fragmented_runtime_tracker()
    for tank_id, height in (("t100", 67.0), ("t200", 87.0), ("t400", 107.0)):
        _seed_fragmented_stage(
            tracker, tank_id=tank_id, frame_count=12, visible_height_px=height
        )

    observations = tracker.close()

    receipt = observations[-1]["tracker_receipt"]
    assert [stage["tank_id"] for stage in receipt["stages"]] == [
        "t100", "t200", "t400"
    ]
    assert [stage["images"] for stage in receipt["stages"]] == [
        {"early": "CASE-1-F000000", "mid": "CASE-1-F000006", "end": "CASE-1-F000011"},
        {"early": "CASE-1-F000012", "mid": "CASE-1-F000018", "end": "CASE-1-F000023"},
        {"early": "CASE-1-F000024", "mid": "CASE-1-F000030", "end": "CASE-1-F000035"},
    ]
    assert receipt["stages"][1]["stream_spread_lab"] == [1.0, 2.0, 3.0]
    assert receipt["source_artifact_sha256"] == "e" * 64


def test_close_ignores_isolated_short_wrong_tank_blip_after_stage_debounce() -> None:
    tracker = _fragmented_runtime_tracker()
    _seed_fragmented_stage(
        tracker, tank_id="t100", frame_count=12, visible_height_px=67.0
    )
    _seed_fragmented_stage(
        tracker, tank_id="t200", frame_count=25, visible_height_px=87.0,
        wrong_tank_at=18, wrong_tank_id="t100",
    )
    _seed_fragmented_stage(
        tracker, tank_id="t400", frame_count=12, visible_height_px=107.0
    )

    receipt = tracker.close()[-1]["tracker_receipt"]

    assert [stage["tank_id"] for stage in receipt["stages"]] == [
        "t100", "t200", "t400"
    ]
    assert receipt["stages"][1]["images"] == {
        "early": "CASE-1-F000012",
        "mid": "CASE-1-F000024",
        "end": "CASE-1-F000036",
    }
    centre = tracker.freeze.tank_lab_centres["t200"]
    assert receipt["stages"][1]["stream_lab"] == pytest.approx(
        (centre[0] - 2.5, centre[1], centre[2])
    )


def test_close_fails_on_sustained_wrong_recipe_order() -> None:
    tracker = _fragmented_runtime_tracker()
    _seed_fragmented_stage(
        tracker, tank_id="t100", frame_count=12, visible_height_px=67.0
    )
    _seed_fragmented_stage(
        tracker, tank_id="t400", frame_count=12, visible_height_px=87.0
    )

    with pytest.raises(ProductionFillStepError, match="do not match recipe order"):
        tracker.close()


def test_close_scales_fragmentation_debounce_to_36_frames_at_30_hz() -> None:
    short = _fragmented_runtime_tracker(
        frame_rate_hz=30.0, recipe_pours=(RecipePour("t100", 10.0),)
    )
    _seed_fragmented_stage(
        short, tank_id="t100", frame_count=35, visible_height_px=67.0
    )
    with pytest.raises(ProductionFillStepError, match="never reaches"):
        short.close()

    complete = _fragmented_runtime_tracker(
        frame_rate_hz=30.0, recipe_pours=(RecipePour("t100", 10.0),)
    )
    _seed_fragmented_stage(
        complete, tank_id="t100", frame_count=36, visible_height_px=67.0
    )
    receipt = complete.close()[-1]["tracker_receipt"]
    assert receipt["stages"][0]["images"] == {
        "early": "CASE-1-F000000",
        "mid": "CASE-1-F000018",
        "end": "CASE-1-F000035",
    }


def test_close_keeps_ordinary_standard_segmentation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fluidmix import wp7_fill_step_production

    tracker = _fragmented_runtime_tracker()
    for tank_id, height in (("t100", 67.0), ("t200", 87.0), ("t400", 107.0)):
        start = len(tracker._frames)
        _seed_fragmented_stage(
            tracker, tank_id=tank_id, frame_count=12, visible_height_px=height
        )
        centre = tracker.freeze.tank_lab_centres[tank_id]
        for index in range(start, start + 12):
            tracker._frames[index] = replace(tracker._frames[index], stream_lab=centre)

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("ordinary segmentation entered fragmentation fallback")

    monkeypatch.setattr(
        wp7_fill_step_production,
        "_recover_fragmented_recipe_frames",
        forbidden_fallback,
    )

    receipt = tracker.close()[-1]["tracker_receipt"]
    assert [stage["tank_id"] for stage in receipt["stages"]] == [
        "t100", "t200", "t400"
    ]


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration() -> PixelToMetreCalibration:
    value = PixelToMetreCalibration(
        schema_version=CALIBRATION_SCHEMA,
        calibration_id="synthetic-camera-contract",
        camera_effective_state_sha256=FROZEN_FILLCAM_CAMERA_SHA256,
        geometry_sha256=FROZEN_GEOMETRY_SHA256,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
        fill_height_px_samples=(67.0, 87.0, 107.0),
        visible_height_m_samples=(0.01, 0.02, 0.03),
        twin_geometry_equivalent_volume_ml_samples=(10.0, 20.0, 30.0),
        artifact_sha256="0" * 64,
    )
    return replace(value, artifact_sha256=calibration_payload_sha256(value))


def _lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    raw = cv2.cvtColor(
        np.asarray([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB
    )[0, 0].astype(float)
    return raw[0] * 100.0 / 255.0, raw[1] - 128.0, raw[2] - 128.0


def _freeze() -> TrackerFreeze:
    colours = ((240, 30, 30), (30, 240, 30), (30, 30, 240))
    return TrackerFreeze(
        stream_roi_xyxy=(300, 80, 320, 190),
        stream_rgb_change_threshold=8,
        stream_minimum_changed_pixels=100,
        stream_minimum_consecutive_frames=3,
        stream_change_delta_e76=8.0,
        tank_cluster_delta_e76=3.0,
        volume_match_tolerance_ml=0.1,
        calibration=_calibration(),
        tank_lab_centres={
            tank: _lab(colour)
            for tank, colour in zip(("t100", "t200", "t400"), colours, strict=True)
        },
        recipe_artifact_sha256="b" * 64,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
    )


def _topology() -> TopologyFreeze:
    return TopologyFreeze(
        artifact_sha256="a" * 64, receipt_sha256="c" * 64,
        dense_row_fraction=0.5, sustained_dense_rows=4,
        bottom_zone_start_row=392, lower_edge_support_rows=4,
        upper_sustained_dense_rows=8,
    )


def test_topology_rejects_spur_and_accepts_two_supported_regions() -> None:
    roi, clamp = fillcam_masks()
    spur = np.zeros_like(roi)
    spur[210:235, 300:303] = True
    spur[350:417] = roi[350:417]
    first = postprocess_cnn_mask_topology(spur, **_topology().parameters())
    assert first.reason_code == "lower_body_selected"
    assert first.surface_row == 350
    assert not np.any(first.mask[:350])

    crossed = np.zeros_like(roi)
    crossed[205:238] = roi[205:238]
    crossed[271:417] = roi[271:417]
    second = postprocess_cnn_mask_topology(crossed, **_topology().parameters())
    assert second.reason_code == "upper_and_lower_bodies_selected"
    assert second.surface_row == 205
    assert not np.any(second.mask & clamp)


def test_topology_loader_is_hash_bound_closed_and_recipe_free(tmp_path: Path) -> None:
    artifact = {
        "schema_version": "wp7-cnn-mask-topology-freeze-v1",
        "artifact_type": "development_train_only_topology_threshold_freeze",
        "claim_boundary": "synthetic_train_only",
        "runtime_rule_input_contract": {
            "allowed": [
                "frozen_cnn_boolean_mask", "fixed_vial_roi", "fixed_clamp_raster"
            ],
            "forbidden_runtime_inputs": ["recipe_ml", "labels", "expected_volume"],
        },
        "validation_access": {
            "validation_rgb_decoded": False,
            "validation_semantic_masks_opened": False,
            "selection_partition": "train",
        },
        "selected_config": {
            "dense_row_fraction": 0.5, "sustained_dense_rows": 4,
            "bottom_zone_start_row": 392, "lower_edge_support_rows": 4,
            "upper_sustained_dense_rows": 8,
        },
        "device": "cpu", "grid_candidate_count": 1, "grid_scores": [],
        "replay_equivalence": {},
        "rule": "fixture", "same_stage_limitation": "fixture",
        "selected_train_score": {}, "selection_objective": "fixture",
        "source_bindings": {}, "source_commit": "1" * 40,
        "threshold_grid": {}, "train_sample_count": 0,
        "train_sample_provenance": [], "train_selection_input_contract": {},
        "train_vial_count": 0, "train_vials_with_stable_window_samples": 0,
        "train_vials_without_stable_window_samples": [],
    }
    artifact["receipt_sha256"] = _canonical_hash(artifact)
    path = tmp_path / "topology.json"
    digest = _write_json(path, artifact)

    assert load_topology_freeze(path, expected_sha256=digest).dense_row_fraction == 0.5
    with pytest.raises(ProductionFillStepError, match="raw-file SHA-256"):
        load_topology_freeze(path, expected_sha256="0" * 64)
    drifted = dict(artifact)
    drifted["recipe_ml"] = {"t100": 99}
    drift_path = tmp_path / "drift.json"
    drift_digest = _write_json(drift_path, drifted)
    with pytest.raises(ProductionFillStepError, match="fields|receipt hash"):
        load_topology_freeze(drift_path, expected_sha256=drift_digest)


def test_calibration_loader_binds_raw_and_payload_hashes(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    digest = _write_json(path, asdict(_calibration()))
    assert load_pixel_to_metre_calibration(
        path, expected_sha256=digest
    ) == _calibration()
    drifted = asdict(_calibration())
    drifted["fill_height_px_samples"] = (66.0, 87.0, 107.0)
    bad = tmp_path / "bad.json"
    bad_digest = _write_json(bad, drifted)
    with pytest.raises(ProductionFillStepError, match="payload hash"):
        load_pixel_to_metre_calibration(bad, expected_sha256=bad_digest)


def test_actual_attempt04_freeze_and_republished_calibration_load_without_media(
    tmp_path: Path,
) -> None:
    root = Path("/home/hackathon5/liamreporoot/outputs/pipeline-20261028")
    topology_path = root / (
        "wp7-fill-step-topology-freeze-01a18128-attempt03/topology-freeze.json"
    )
    train_path = root / (
        "wp7-fill-step-tracker-dev-01a18128-attempt04/attempt04-train-freeze.json"
    )
    materialization_path = root / (
        "wp7-fill-step-tracker-dev-01a18128-attempt04/"
        "development-materialization-receipt.json"
    )
    if not all(path.is_file() for path in (
        topology_path, train_path, materialization_path
    )):
        pytest.skip("accepted attempt04 source-only fixtures are unavailable")
    from scripts.publish_wp7_fill_step_calibration import publish
    calibration_path = tmp_path / "calibration.json"
    publish(calibration_path)
    calibration = load_pixel_to_metre_calibration(
        calibration_path,
        expected_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    )
    topology = load_topology_freeze(
        topology_path,
        expected_sha256=(
            "62e56a9e2423b09b25cf735452ebdd36adfa056bf8f8821c3fc173dbc6415a3f"
        ),
    )
    freeze = load_tracker_freeze(
        materialization_path,
        expected_materialization_sha256=(
            "255319ca4df8e668b0bceaaf211642cabcd59983a4a023163b0bfbf5a8a323ce"
        ),
        train_freeze_path=train_path,
        expected_train_freeze_sha256=(
            "876a2c5813eba6f2cbd018a16c144e5a9d28ca9cbf1170ba4529e5c212392206"
        ),
        topology=topology,
        calibration=calibration,
        target_recipe_artifact_sha256="f" * 64,
    )
    assert freeze.calibration.artifact_sha256 == (
        "9d2c2941244ef807cf02cc4bc3107dc6f621aa68bc59b11627484fb0d2c46c04"
    )
    assert freeze.volume_match_tolerance_ml == 5.0
    assert freeze.stream_rgb_change_threshold == 6
    assert freeze.recipe_artifact_sha256 == "f" * 64


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("mask_topology_policy", "different"),
        ("maximum_height_and_surface_delta_px", 4.0),
        ("selection", "first_three_consecutive_frames_of_camera_state_tail"),
        ("window_length", 4),
    ],
)
def test_tracker_freeze_rejects_drift_in_every_stable_volume_window_field(
    tmp_path: Path, field: str, drifted: object,
) -> None:
    root = Path("/home/hackathon5/liamreporoot/outputs/pipeline-20261028")
    topology_path = root / (
        "wp7-fill-step-topology-freeze-01a18128-attempt03/topology-freeze.json"
    )
    train_path = root / (
        "wp7-fill-step-tracker-dev-01a18128-attempt04/attempt04-train-freeze.json"
    )
    materialization_path = root / (
        "wp7-fill-step-tracker-dev-01a18128-attempt04/"
        "development-materialization-receipt.json"
    )
    if not all(path.is_file() for path in (
        topology_path, train_path, materialization_path
    )):
        pytest.skip("accepted attempt04 source-only fixtures are unavailable")
    from scripts.publish_wp7_fill_step_calibration import publish
    calibration_path = tmp_path / "calibration.json"
    publish(calibration_path)
    calibration = load_pixel_to_metre_calibration(
        calibration_path,
        expected_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    )
    topology = load_topology_freeze(
        topology_path,
        expected_sha256=(
            "62e56a9e2423b09b25cf735452ebdd36adfa056bf8f8821c3fc173dbc6415a3f"
        ),
    )
    artifact = json.loads(materialization_path.read_text(encoding="utf-8"))
    artifact["train_only_freeze"]["stable_volume_window"][field] = drifted
    artifact["receipt_sha256"] = _canonical_hash({
        key: value for key, value in artifact.items() if key != "receipt_sha256"
    })
    drift_path = tmp_path / "drifted-materialization.json"
    drift_digest = _write_json(drift_path, artifact)

    with pytest.raises(ProductionFillStepError, match="stable volume window"):
        load_tracker_freeze(
            drift_path,
            expected_materialization_sha256=drift_digest,
            train_freeze_path=train_path,
            expected_train_freeze_sha256=(
                "876a2c5813eba6f2cbd018a16c144e5a9d28ca9cbf1170ba4529e5c212392206"
            ),
            topology=topology,
            calibration=calibration,
            target_recipe_artifact_sha256="f" * 64,
        )


def test_incremental_observer_emits_three_phases_per_camera_stage_and_receipt() -> None:
    roi, _clamp = fillcam_masks()
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    colours = ((240, 30, 30), (30, 240, 30), (30, 30, 240))
    surfaces = iter((350,) * 3 + (330,) * 3 + (310,) * 3)

    def predictor(_current, _reference):
        surface = next(surfaces)
        mask = np.zeros((480, 640), dtype=bool)
        mask[surface:417] = roi[surface:417]
        return SimpleNamespace(mask=mask, status="visible_measured", surface_row=surface)

    tracker = IncrementalFillStepTracker(
        freeze=_freeze(), topology=_topology(),
        recipe_pours=(
            RecipePour("t100", 10.0), RecipePour("t200", 10.0),
            RecipePour("t400", 10.0),
        ),
        previous_final_tank_id="t400",
        same_run_empty_reference_rgb=reference,
        predictor=predictor, cohort_id="COHORT-1", case_id="CASE-1",
        vial_id="VIAL-1",
        vial_position=1, source_artifact_sha256="e" * 64,
    )
    emitted = []
    for index, colour in enumerate(
        colour for selected in colours for colour in (selected,) * 3
    ):
        frame = reference.copy()
        frame[80:190, 300:320] = colour
        emitted.extend(tracker.observe(
            frame, frame_index=index, elapsed_seconds=index / 30.0
        ))
    emitted.extend(tracker.close())

    assert [(row["fill_step"], row["phase"]) for row in emitted] == [
        (step, phase) for step in (1, 2, 3) for phase in ("early", "mid", "end")
    ]
    assert emitted[-1]["closed"] is True
    receipt = emitted[-1]["tracker_receipt"]
    assert len(receipt["stages"]) == 3
    assert receipt["stages"][0]["images"] == {
        "early": "CASE-1-F000000", "mid": "CASE-1-F000001",
        "end": "CASE-1-F000002",
    }
    assert receipt["stages"][2]["twin_geometry_equivalent_volume_ml"] == 30.0


def test_selected_phase_without_height_is_strict_by_default_and_explicitly_available() -> None:
    from fluidmix.wp7_fill_step_production import TopologyHeightMeasurement

    reference = np.zeros((480, 640, 3), dtype=np.uint8)

    def build(allow: bool):
        tracker = IncrementalFillStepTracker(
            freeze=_freeze(), topology=_topology(),
            recipe_pours=(RecipePour("t100", 10.0),),
            previous_final_tank_id="t400", same_run_empty_reference_rgb=reference,
            predictor=lambda *_args: None, cohort_id="COHORT-1",
            case_id="v4-CASE_97a8c6cb8161881c", vial_id="VIAL-1",
            vial_position=1, source_artifact_sha256="e" * 64,
            allow_unavailable_selected_phase=allow,
            image_index_width=4,
        )
        empty = np.zeros((480, 640), dtype=bool)
        tracker._captures[4] = (reference.copy(), empty)
        tracker._height_by_frame[4] = TopologyHeightMeasurement(
            "bottom_zone_component_missing", None, None, "unavailable"
        )
        return tracker

    selected = SimpleNamespace(
        early_frame_index=4, mid_frame_index=4, end_frame_index=4,
    )
    stage = SimpleNamespace(
        selected_frames=selected, fill_step=1, tank_id="t100", agreement=True,
        twin_geometry_equivalent_volume_ml=10.0, expected_dispense_ml=10.0,
        expected_cumulative_ml=10.0, stream_step_index=1,
        tank_cluster_consistent=True,
    )
    result = SimpleNamespace(stages=(stage,))

    with pytest.raises(ProductionFillStepError, match="lacks accepted CNN height"):
        build(False)._emit(result, completed_count=1, closed=False)
    rows = build(True)._emit(result, completed_count=1, closed=False)
    assert rows[0]["image_id"] == "v4-CASE_97a8c6cb8161881c-F0004"
    assert rows[0]["surface_row"] is None
    assert rows[0]["visible_height_px"] is None
    assert rows[0]["limit_codes"] == ["cnn_bottom_zone_component_missing"]


def test_deferred_source_binding_is_one_time_and_required_before_close() -> None:
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = IncrementalFillStepTracker(
        freeze=_freeze(), topology=_topology(),
        recipe_pours=(RecipePour("t100", 10.0),),
        previous_final_tank_id="t400", same_run_empty_reference_rgb=reference,
        predictor=lambda *_args: None, cohort_id="COHORT-1", case_id="CASE-1",
        vial_id="VIAL-1", vial_position=1, source_artifact_sha256=None,
        defer_source_artifact_binding=True,
    )
    with pytest.raises(ProductionFillStepError, match="source artifact.*unbound"):
        tracker.close()
    tracker.bind_source_artifact_sha256("a" * 64)
    assert tracker.source_artifact_sha256 == "a" * 64
    with pytest.raises(ProductionFillStepError, match="already bound"):
        tracker.bind_source_artifact_sha256("b" * 64)


def test_incremental_observer_runs_temporal_prediction_once_at_vial_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fluidmix import empty_reference_timing_adapter, wp7_fill_step_production

    real_predict = empty_reference_timing_adapter.predict_fill_sequence_at_frame_rate
    real_track = wp7_fill_step_production.track_fill_step
    sequence_lengths: list[int] = []
    forwarded_frame_rates: list[tuple[str, float]] = []

    def counted_predict(frames, **kwargs):
        sequence_lengths.append(len(frames))
        forwarded_frame_rates.append(("predict", kwargs["frame_rate_hz"]))
        return real_predict(frames, **kwargs)

    def counted_track(*args, **kwargs):
        forwarded_frame_rates.append(("track", kwargs["frame_rate_hz"]))
        return real_track(*args, **kwargs)

    monkeypatch.setattr(
        empty_reference_timing_adapter,
        "predict_fill_sequence_at_frame_rate",
        counted_predict,
    )
    monkeypatch.setattr(wp7_fill_step_production, "track_fill_step", counted_track)
    roi, _clamp = fillcam_masks()
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    surfaces = iter((350,) * 3 + (330,) * 3 + (310,) * 3)

    def predictor(_current, _reference):
        surface = next(surfaces)
        mask = np.zeros((480, 640), dtype=bool)
        mask[surface:417] = roi[surface:417]
        return SimpleNamespace(mask=mask)

    tracker = IncrementalFillStepTracker(
        freeze=_freeze(), topology=_topology(),
        recipe_pours=(
            RecipePour("t100", 10.0), RecipePour("t200", 10.0),
            RecipePour("t400", 10.0),
        ),
        previous_final_tank_id="t400",
        same_run_empty_reference_rgb=reference,
        predictor=predictor, cohort_id="COHORT-1", case_id="CASE-1",
        vial_id="VIAL-1", vial_position=1, source_artifact_sha256="e" * 64,
        frame_rate_hz=10.0,
    )
    colours = ((240, 30, 30), (30, 240, 30), (30, 30, 240))
    for index, colour in enumerate(
        colour for selected in colours for colour in (selected,) * 3
    ):
        frame = reference.copy()
        frame[80:190, 300:320] = colour
        assert tracker.observe(
            frame, frame_index=index, elapsed_seconds=index / 30.0
        ) == ()

    assert sequence_lengths == []
    observations = tracker.close()

    assert sequence_lengths == [10]
    assert forwarded_frame_rates == [("predict", 10.0), ("track", 10.0)]
    assert len(observations) == 9
    assert observations[-1]["closed"] is True


@pytest.mark.parametrize("frame_rate_hz", [0.0, -1.0, float("inf"), float("nan"), True])
def test_incremental_tracker_rejects_invalid_frame_rate(frame_rate_hz) -> None:
    reference = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ProductionFillStepError, match="frame rate"):
        IncrementalFillStepTracker(
            freeze=_freeze(), topology=_topology(),
            recipe_pours=(RecipePour("t100", 10.0),),
            previous_final_tank_id="t400",
            same_run_empty_reference_rgb=reference,
            predictor=lambda *_args: None,
            cohort_id="COHORT-1", case_id="CASE-1", vial_id="VIAL-1",
            vial_position=1, source_artifact_sha256="e" * 64,
            frame_rate_hz=frame_rate_hz,
        )


def test_stream_segmentation_and_phase_indices_ignore_intervening_cnn_failures() -> None:
    roi, _clamp = fillcam_masks()
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    colours = ((240, 30, 30), (30, 240, 30), (30, 30, 240))

    def run(*, fail_second: bool) -> list[tuple[int, str, int]]:
        call = 0

        def predictor(_current, _reference):
            nonlocal call
            stage_offset = call % 5
            stage = call // 5
            call += 1
            mask = np.zeros((480, 640), dtype=bool)
            if not fail_second or stage_offset != 1:
                surface = (350, 330, 310)[stage]
                mask[surface:417] = roi[surface:417]
            return SimpleNamespace(mask=mask)

        tracker = IncrementalFillStepTracker(
            freeze=_freeze(), topology=_topology(),
            recipe_pours=(
                RecipePour("t100", 10.0), RecipePour("t200", 10.0),
                RecipePour("t400", 10.0),
            ),
            previous_final_tank_id="t400",
            same_run_empty_reference_rgb=reference,
            predictor=predictor, cohort_id="COHORT-1", case_id="CASE-1",
            vial_id="VIAL-1", vial_position=1, source_artifact_sha256="e" * 64,
        )
        for index, colour in enumerate(
            colour for selected in colours for colour in (selected,) * 5
        ):
            frame = reference.copy()
            frame[80:190, 300:320] = colour
            tracker.observe(frame, frame_index=index, elapsed_seconds=index / 30.0)
        return [
            (int(row["fill_step"]), str(row["phase"]), int(row["frame_index"]))
            for row in tracker.close()
        ]

    baseline = run(fail_second=False)
    degraded = run(fail_second=True)

    assert degraded == baseline
    assert baseline == [
        (step, phase, index)
        for step, indices in enumerate(((0, 2, 4), (5, 7, 9), (10, 12, 14)), 1)
        for phase, index in zip(("early", "mid", "end"), indices, strict=True)
    ]


def test_registry_is_no_clobber_and_binds_exact_phase_images(tmp_path: Path) -> None:
    roi, _ = fillcam_masks()
    mask = np.zeros((480, 640), dtype=bool)
    mask[350:417] = roi[350:417]
    rows = tuple({
        "image_id": f"IMG-{phase}", "frame_index": index, "phase": phase,
        "observation_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "prediction_mask": mask,
    } for index, phase in enumerate(("early", "mid", "end")))
    output = tmp_path / "claim"
    registry = persist_completed_observations(output_root=output, observations=rows)
    assert [registry["images"][f"IMG-{phase}"]["phase"] for phase in (
        "early", "mid", "end"
    )] == ["early", "mid", "end"]
    assert (output / "registry.json").is_file()
    with pytest.raises(ProductionFillStepError, match="already exists"):
        persist_completed_observations(output_root=output, observations=rows)


def test_mask_topology_is_recipe_invariant() -> None:
    source = Path("src/fluidmix/wp7_fill_step_production.py").read_text()
    topology_source = source[source.index("def postprocess_cnn_mask_topology"):source.index(
        "def apply_camera_hidden_band_surface"
    )]
    assert "recipe" not in topology_source
    assert "physx" not in source
    assert "volume_height_curve" not in source

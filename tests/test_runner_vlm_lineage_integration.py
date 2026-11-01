from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_wp7_vlm_decision import _cohort, _rehash_receipt, _tracker_receipts


def _visible_prediction(*_args, **_kwargs):
    return SimpleNamespace(
        status="visible_measured", mask=np.ones((10, 12), dtype=bool),
        surface_row=0, visible_height_px=10,
    )


def test_vlm_native_crop_fails_closed_on_absent_or_drifted_camera_metadata():
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    valid_context = {
        "camera_metadata": {"resolution": [12, 10], "vial_roi": [3, 2, 11, 10]}
    }
    stages = [
        {
            "vial_id": "V1", "vial_position": 1, "fill_step": 1,
            "tank_id": "t100", "evidence_id": "EV-1", "image_id": "IMG-1",
            "closed": False, "same_run_empty_reference_rgb": np.zeros((10, 12, 3), dtype=np.uint8),
            "agreement": True, "measured_volume_ml": 1.0,
        },
        {
            "vial_id": "V1", "vial_position": 1, "fill_step": 1,
            "tank_id": "t100", "evidence_id": "EV-2", "image_id": "IMG-2",
            "closed": False, "same_run_empty_reference_rgb": np.zeros((10, 12, 3), dtype=np.uint8),
            "agreement": True, "measured_volume_ml": 1.0,
            "vlm_cohort_context": valid_context,
        },
        {
            "vial_id": "V1", "vial_position": 1, "fill_step": 1,
            "tank_id": "t100", "evidence_id": "EV-3", "image_id": "IMG-3",
            "closed": False, "same_run_empty_reference_rgb": np.zeros((10, 12, 3), dtype=np.uint8),
            "agreement": True, "measured_volume_ml": 1.0,
            "vlm_cohort_context": {
                "camera_metadata": {"resolution": [12, 10], "vial_roi": [2, 2, 10, 10]}
            },
        },
    ]

    class Tracker:
        def observe(self, _rgb, **_metadata):
            return stages.pop(0)

    measure_calls = []
    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(), predictor=_visible_prediction,
        measurer=lambda *_args, **_kwargs: measure_calls.append(True) or {
            "validity": "valid", "median_lab": [50.0, 2.0, 3.0],
            "mad_lab": [1.0, 1.0, 1.0], "iqr_lab": [2.0, 2.0, 2.0],
            "limit_codes": [],
        },
        deterministic_decider=lambda _cohort: {}, vlm_bundle_builder=lambda *_args: {},
        vlm_runner=lambda _bundle: (), device="cpu", model_id="accepted-test",
    )
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    try:
        with pytest.raises(RuntimeError, match="context is unavailable"):
            adapter.process_frame(frame, frame_index=0, elapsed_seconds=0.0)
        adapter.process_frame(frame, frame_index=1, elapsed_seconds=1.0)
        with pytest.raises(ValueError, match="context drifted"):
            adapter.process_frame(frame, frame_index=2, elapsed_seconds=2.0)
    finally:
        adapter.close()
    assert len(measure_calls) == 1


def test_accepted_adapter_builds_actual_vial1_bundle_from_27_native_roi_crops(tmp_path):
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter
    from fluidmix.wp7_vlm_decision import build_vial1_bundle

    cohort, _unused_locators = _cohort(tmp_path / "fixture")
    cohort["camera_metadata"]["resolution"] = [12, 10]
    cohort["camera_metadata"]["vial_roi"] = [3, 2, 11, 10]
    cohort["stages"].append({
        **deepcopy(cohort["stages"][-1]),
        "stage_id": "stage-3",
        "stage_ordinal": 3,
        "tank_id": "t400",
        "dispense_order": 3,
    })
    cohort["deterministic_decision_receipt"]["stages"].append({
        "stage_id": "stage-3", "evidence_id": "EV-C1-V1-S3",
        "disposition": "keep", "limit_codes": [],
        "delta_e76_vial2": 0.0, "delta_e76_vial3": 0.0,
    })
    _rehash_receipt(cohort["deterministic_decision_receipt"])
    for vial in cohort["vials"]:
        position = vial["vial_position"]
        stage = deepcopy(vial["stages"][-1])
        stage["stage_id"] = "stage-3"
        stage["measurement"]["evidence_id"] = f"EV-C1-V{position}-S3"
        for phase_index, frame in enumerate(stage["frames"]):
            phase = frame["phase"]
            frame["image_id"] = f"IMG-C1-V{position}-S3-{phase}"
            frame["timestamp_seconds"] = float(phase_index + 1)
        vial["stages"].append(stage)
    receipts = _tracker_receipts(cohort)
    context = {key: deepcopy(value) for key, value in cohort.items()
               if key not in {"vials", "deterministic_decision_receipt"}}

    observations = []
    for vial, receipt in zip(cohort["vials"], receipts, strict=True):
        for stage in vial["stages"]:
            for frame in stage["frames"]:
                observations.append({
                    "vial_id": vial["vial_id"],
                    "vial_position": vial["vial_position"],
                    "fill_step": int(stage["stage_id"].removeprefix("stage-")),
                    "stage_id": stage["stage_id"],
                    "tank_id": cohort["stages"][
                        int(stage["stage_id"].removeprefix("stage-")) - 1
                    ]["tank_id"],
                    "evidence_id": stage["measurement"]["evidence_id"],
                    "image_id": frame["image_id"],
                    "closed": frame is vial["stages"][-1]["frames"][-1],
                    "same_run_empty_reference_rgb": np.zeros((10, 12, 3), dtype=np.uint8),
                    "agreement": True,
                    "measured_volume_ml": 1.0,
                    "vlm_cohort_context": context,
                    "vlm_vial": deepcopy(vial),
                    "tracker_receipt": deepcopy(receipt),
                })

    class Tracker:
        def observe(self, _rgb, **_metadata):
            return observations.pop(0)

    def predictor(*_args, **_kwargs):
        return SimpleNamespace(
            status="visible_measured", mask=np.ones((10, 12), dtype=bool),
            surface_row=0, visible_height_px=10,
        )

    measure_calls = []

    def measurer(image_id, _locator, **_kwargs):
        measure_calls.append(image_id)
        return {
            "validity": "valid", "median_lab": [50.0, 2.0, 3.0],
            "mad_lab": [1.0, 1.0, 1.0], "iqr_lab": [2.0, 2.0, 2.0],
            "limit_codes": [],
        }

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(), predictor=predictor, measurer=measurer,
        deterministic_decider=lambda _cohort: deepcopy(
            cohort["deterministic_decision_receipt"]
        ),
        vlm_bundle_builder=build_vial1_bundle, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="accepted-test",
    )
    try:
        closed = {}
        for index in range(27):
            result = adapter.process_frame(
                np.full((10, 12, 3), index, dtype=np.uint8),
                frame_index=index, elapsed_seconds=float(index),
            )[0]
            if result["closed"]:
                closed[result["vial_id"]] = result
        adapter.decide_cohort(closed)
        built = adapter.build_vlm_bundle(closed)
        assert len(built.prompt_payload["images"]) == 27
        assert len(built.image_locators) == 27
        assert len(measure_calls) == 27
        for locator in built.image_locators.values():
            assert locator.crop_box_xyxy == (3, 2, 11, 10)
            assert locator.source_native_dimensions == (12, 10)
            assert locator.crop_native_dimensions == (8, 8)
    finally:
        adapter.close()

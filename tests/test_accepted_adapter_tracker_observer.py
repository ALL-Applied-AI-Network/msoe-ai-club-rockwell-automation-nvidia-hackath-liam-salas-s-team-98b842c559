from __future__ import annotations

import numpy as np
import pytest


def test_completed_tracker_observations_propagate_phases_without_second_cnn() -> None:
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    mask = np.zeros((480, 640), dtype=bool)
    mask[350:417, 300:340] = True
    phases = ("early", "mid", "end")
    observations = tuple({
        "vial_id": "V1", "vial_position": 1, "fill_step": 1,
        "stage_id": "STAGE-V1-1", "tank_id": "t100",
        "evidence_id": f"EVID-{phase}", "image_id": f"IMG-{phase}",
        "phase": phase, "frame_index": index,
        "same_run_empty_reference_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation_rgb": np.full((480, 640, 3), index + 1, dtype=np.uint8),
        "prediction_mask": mask, "agreement": True,
        "measured_volume_ml": 10.0, "twin_geometry_equivalent_volume_ml": 10.0,
        "expected_dispense_ml": 10.0, "expected_cumulative_ml": 10.0,
        "stream_step_index": 1, "tank_cluster_consistent": True,
        "limit_codes": [], "closed": False,
        "vlm_cohort_context": {
            "camera_metadata": {
                "resolution": [640, 480], "vial_roi": [280, 60, 380, 440]
            }
        },
    } for index, phase in enumerate(phases))

    class Tracker:
        def observe(self, _rgb, **_metadata):
            return observations

    predictor_calls: list[bool] = []
    measured_ids: list[str] = []
    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        predictor=lambda *_args, **_kwargs: predictor_calls.append(True),
        measurer=lambda image_id, *_args, **_kwargs: measured_ids.append(image_id) or {
            "validity": "valid", "median_lab": [50.0, 0.0, 0.0],
            "mad_lab": [0.0, 0.0, 0.0], "iqr_lab": [0.0, 0.0, 0.0],
            "limit_codes": [],
        },
        deterministic_decider=lambda _vials: {},
        vlm_bundle_builder=lambda *_args: {}, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="observer-test",
    )
    try:
        result = adapter.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_index=99, elapsed_seconds=3.3,
        )
    finally:
        adapter.close()

    assert predictor_calls == []
    assert [row["phase"] for row in result] == ["early", "mid", "end"]
    assert measured_ids == ["IMG-early", "IMG-mid", "IMG-end"]


def test_adapter_proxies_bounded_session_completion_contract() -> None:
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    class Tracker:
        requires_complete = True
        session_complete = True

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(), deterministic_decider=lambda _: {},
        vlm_bundle_builder=lambda *_args: {}, vlm_runner=lambda _: (),
        device="cpu", model_id="bounded-test",
    )
    try:
        assert adapter.requires_complete is True
        assert adapter.session_complete is True
    finally:
        adapter.close()


def test_empty_selected_topology_fails_closed_for_stage_aggregation() -> None:
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    valid = np.zeros((480, 640), dtype=bool)
    valid[350:417, 300:340] = True
    empty = np.zeros_like(valid)
    observations = tuple({
        "vial_id": "V1", "vial_position": 1, "fill_step": 1,
        "stage_id": "STAGE-V1-1", "tank_id": "t100",
        "evidence_id": f"EVID-{phase}", "image_id": f"IMG-{phase}",
        "phase": phase, "frame_index": index,
        "same_run_empty_reference_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "prediction_mask": empty if phase == "early" else valid,
        "surface_row": None if phase == "early" else 350,
        "visible_height_px": None if phase == "early" else 67,
        "height_status": "bottom_zone_component_missing" if phase == "early" else "visible_measured",
        "vlm_cohort_context": {
            "camera_metadata": {
                "resolution": [640, 480], "vial_roi": [280, 60, 380, 440]
            }
        },
        "agreement": True, "measured_volume_ml": 10.0,
        "twin_geometry_equivalent_volume_ml": 10.0,
        "expected_dispense_ml": 10.0, "expected_cumulative_ml": 10.0,
        "stream_step_index": 1, "tank_cluster_consistent": True,
        "limit_codes": (["cnn_bottom_zone_component_missing"] if phase == "early" else []),
        "closed": False,
    } for index, phase in enumerate(("early", "mid", "end")))

    class Tracker:
        def observe(self, _rgb, **_metadata):
            return observations

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        measurer=lambda *_args, **_kwargs: {
            "validity": "valid", "median_lab": [50.0, 0.0, 0.0],
            "mad_lab": [0.0, 0.0, 0.0], "iqr_lab": [0.0, 0.0, 0.0],
            "limit_codes": [],
        },
        deterministic_decider=lambda _vials: {},
        vlm_bundle_builder=lambda *_args: {}, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="observer-test",
    )
    try:
        with pytest.raises(ValueError, match="mask is empty"):
            adapter.process_frame(
                np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=9, elapsed_seconds=0.9,
            )
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("phases", "tank_ids", "closed_index", "message"),
    [
        (("early", "end"), ("t100", "t100"), 1, "evidence is unavailable"),
        (("early", "early"), ("t100", "t100"), None, "was duplicated"),
        (("early", "mid"), ("t100", "t200"), None, "identity drifted"),
    ],
)
def test_live_stage_aggregation_rejects_incomplete_or_rebound_phases(
    phases, tank_ids, closed_index, message,
) -> None:
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    context = {
        "camera_metadata": {"resolution": [4, 4], "vial_roi": [0, 0, 4, 4]},
    }
    observations = []
    for index, (phase, tank_id) in enumerate(zip(phases, tank_ids)):
        closed = index == closed_index
        observations.append({
            "vial_id": "V1", "vial_position": 1, "fill_step": 1,
            "stage_id": "dispense-1-t100", "tank_id": tank_id,
            "evidence_id": f"EVID-{index}", "image_id": f"IMG-{index}",
            "phase": phase,
            "same_run_empty_reference_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "observation_rgb": np.full((4, 4, 3), 50 + index, dtype=np.uint8),
            "prediction_mask": np.ones((4, 4), dtype=bool),
            "vlm_cohort_context": context, "closed": closed,
            **({
                "tracker_receipt": {"vial_id": "V1"},
                "vlm_vial": {
                    "vial_id": "V1", "vial_position": 1,
                    "stages": [{"stage_id": "dispense-1-t100"}],
                },
            } if closed else {}),
        })

    class Tracker:
        def observe(self, _rgb, **_metadata):
            return tuple(observations)

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        measurer=lambda *_args, **_kwargs: {
            "validity": "valid", "median_lab": [50.0, 0.0, 0.0],
            "mad_lab": [0.0, 0.0, 0.0], "iqr_lab": [0.0, 0.0, 0.0],
            "limit_codes": [],
        },
        deterministic_decider=lambda _vials: {},
        vlm_bundle_builder=lambda *_args: {}, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="observer-test",
    )
    try:
        with pytest.raises((RuntimeError, ValueError), match=message):
            adapter.process_frame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                frame_index=1, elapsed_seconds=0.1,
            )
    finally:
        adapter.close()


def test_closed_live_cohort_uses_three_phase_masked_pixels_for_delta_e() -> None:
    from skimage.color import rgb2lab

    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter
    from fluidmix.pipeline.service.production_factory import deterministic_stage_measurements
    from fluidmix.wp7_vlm_decision import (
        build_deterministic_decision,
        canonical_sha256,
    )

    context = {
        "cohort_id": "LIVE-C1",
        "camera_metadata": {"resolution": [4, 4], "vial_roi": [0, 0, 4, 4]},
    }
    mask = np.ones((4, 4), dtype=bool)
    values_by_vial = ((20, 100, 240), (20, 90, 240), (20, 80, 240))

    class Tracker:
        def __init__(self) -> None:
            self.vial_position = 0

        def observe(self, _rgb, **_metadata):
            self.vial_position += 1
            position = self.vial_position
            vial_id = f"V{position}"
            rows = []
            for phase, value in zip(("early", "mid", "end"), values_by_vial[position - 1]):
                rows.append({
                    "vial_id": vial_id, "vial_position": position,
                    "fill_step": 1, "stage_id": "dispense-1-t100",
                    "tank_id": "t100", "evidence_id": f"EVID-{vial_id}-{phase}",
                    "image_id": f"IMG-{vial_id}-{phase}", "phase": phase,
                    "same_run_empty_reference_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "observation_rgb": np.full((4, 4, 3), value, dtype=np.uint8),
                    "prediction_mask": mask, "agreement": True,
                    "measured_volume_ml": 10.0, "vlm_cohort_context": context,
                    "closed": phase == "end",
                    **({
                        "tracker_receipt": {"vial_id": vial_id},
                        "vlm_vial": {
                            "vial_id": vial_id, "vial_position": position,
                            "stages": [{"stage_id": "dispense-1-t100"}],
                        },
                    } if phase == "end" else {}),
                })
            return tuple(rows)

    def decide(cohort):
        stage_rows = deterministic_stage_measurements(cohort)
        return build_deterministic_decision(
            cohort_id="LIVE-C1", target_vial_id="V1",
            stage_measurements=stage_rows,
            source_artifact_sha256=canonical_sha256(stage_rows),
        )

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        measurer=lambda image_id, *_args, **_kwargs: {
            "validity": "valid", "median_lab": [float(image_id.split("-")[-1] == "end"), 0.0, 0.0],
            "mad_lab": [0.0, 0.0, 0.0], "iqr_lab": [0.0, 0.0, 0.0],
            "limit_codes": [],
        },
        deterministic_decider=decide,
        vlm_bundle_builder=lambda *_args, **_kwargs: {}, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="observer-test",
    )
    try:
        public = {}
        for position in range(1, 4):
            rows = adapter.process_frame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                frame_index=position, elapsed_seconds=float(position),
            )
            assert [row["phase"] for row in rows] == ["early", "mid", "end"]
            assert rows[-1]["measurement"]["median_lab"] == [1.0, 0.0, 0.0]
            public[f"V{position}"] = rows[-1]
        decision = adapter.decide_cohort(public)
        retained = [
            adapter._vlm_vials[f"V{position}"]["stages"][0]["measurement"]
            for position in range(1, 4)
        ]
    finally:
        adapter.close()

    expected_l = [
        float(rgb2lab(np.full((1, 1, 3), value / 255.0))[0, 0, 0])
        for value in (100, 90, 80)
    ]
    assert [row["median_lab"][0] for row in retained] == pytest.approx(expected_l)
    assert decision["stages"][0]["delta_e76_vial2"] == pytest.approx(
        expected_l[0] - expected_l[1]
    )
    assert decision["stages"][0]["delta_e76_vial3"] == pytest.approx(
        expected_l[0] - expected_l[2]
    )
    assert decision["disposition"] == "discard"

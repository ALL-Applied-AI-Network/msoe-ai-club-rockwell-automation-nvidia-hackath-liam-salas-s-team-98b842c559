from __future__ import annotations

import numpy as np


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

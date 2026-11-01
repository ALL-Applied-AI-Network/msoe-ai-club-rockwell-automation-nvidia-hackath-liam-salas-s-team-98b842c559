from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


def test_live_stage_aggregation_matches_the_frozen_development_contract() -> None:
    from fluidmix.pipeline.service.accepted_adapter import _aggregate_masked_lab
    from fluidmix.wp7_vlm_development_materializer import aggregate_masked_lab

    frames = [
        np.full((4, 5, 3), (20 + index, 40 + index, 60 + index), dtype=np.uint8)
        for index in range(3)
    ]
    masks = []
    for index in range(3):
        mask = np.zeros((4, 5), dtype=bool)
        mask[index:index + 2, 1:4] = True
        masks.append(mask)

    production = _aggregate_masked_lab(frames, masks)
    frozen = aggregate_masked_lab(frames, masks)

    assert production == {
        key: frozen[key] for key in ("median_lab", "mad_lab", "iqr_lab")
    }


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
        eof_abstention_reason = "tracker_stable_tail_missing"

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(), deterministic_decider=lambda _: {},
        vlm_bundle_builder=lambda *_args: {}, vlm_runner=lambda _: (),
        device="cpu", model_id="bounded-test",
    )
    try:
        assert adapter.requires_complete is True
        assert adapter.session_complete is True
        assert adapter.eof_abstention_reason == "tracker_stable_tail_missing"
    finally:
        adapter.close()


def test_pinned_stage_stills_keep_template_measurements_across_all_visual_phases() -> None:
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    phases = ("early", "mid", "end")
    stage_ids = ("dispense-1-t200", "dispense-2-t400", "dispense-3-t100")
    tanks = ("t200", "t400", "t100")
    pinned = {
        f"V{vial}": {
            stage_id: {
                "evidence_id": f"PIN-V{vial}-S{stage}",
                "median_lab": [float(vial * 10 + stage), 1.0, 2.0],
                "mad_lab": [0.1, 0.2, 0.3],
                "iqr_lab": [0.4, 0.5, 0.6],
                "limit_codes": [],
            }
            for stage, stage_id in enumerate(stage_ids, 1)
        }
        for vial in range(1, 4)
    }
    original = deepcopy(pinned)
    context = {
        "cohort_id": "PINNED-C1",
        "camera_metadata": {"resolution": [4, 4], "vial_roi": [0, 0, 4, 4]},
    }

    class Tracker:
        def __init__(self) -> None:
            self.vial_position = 0

        def observe(self, _rgb, **_metadata):
            self.vial_position += 1
            position = self.vial_position
            vial_id = f"V{position}"
            rows = []
            for stage, (stage_id, tank) in enumerate(zip(stage_ids, tanks), 1):
                for phase_index, phase in enumerate(phases):
                    empty = position == stage == 1 and phase == "early"
                    mask = np.zeros((4, 4), dtype=bool)
                    if not empty:
                        mask[1:3, 1:3] = True
                    closed = stage == 3 and phase == "end"
                    row = {
                        "vial_id": vial_id, "vial_position": position,
                        "fill_step": stage, "stage_id": stage_id, "tank_id": tank,
                        "evidence_id": f"VIS-{vial_id}-S{stage}-{phase}",
                        "image_id": f"IMG-{vial_id}-S{stage}-{phase}", "phase": phase,
                        "same_run_empty_reference_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                        "observation_rgb": np.full(
                            (4, 4, 3), 20 + position + stage + phase_index, dtype=np.uint8,
                        ),
                        "prediction_mask": mask,
                        "height_status": (
                            "bottom_zone_component_missing" if empty else "visible_measured"
                        ),
                        "vlm_cohort_context": context, "closed": closed,
                    }
                    if closed:
                        row.update({
                            "tracker_receipt": {"vial_id": vial_id},
                            "vlm_vial": {
                                "vial_id": vial_id, "vial_position": position,
                                "stages": [
                                    {
                                        "stage_id": candidate,
                                        "frames": [
                                            {
                                                "image_id": f"IMG-{vial_id}-S{ordinal}-{name}",
                                                "phase": name,
                                            }
                                            for name in phases
                                        ],
                                        "measurement": deepcopy(pinned[vial_id][candidate]),
                                    }
                                    for ordinal, candidate in enumerate(stage_ids, 1)
                                ],
                            },
                        })
                    rows.append(row)
            return tuple(rows)

    pinned_receipt = {
        "cohort_id": "PINNED-C1", "target_vial_id": "V1",
        "disposition": "discard", "source_artifact_sha256": "87578e5c" + "0" * 56,
        "receipt_sha256": "a" * 64,
        "stages": [
            {
                "stage_id": stage_id, "evidence_id": f"PIN-V1-S{ordinal}",
                "delta_e76_vial2": float(ordinal),
                "delta_e76_vial3": float(ordinal + 1),
                "disposition": "discard", "limit_codes": [],
            }
            for ordinal, stage_id in enumerate(stage_ids, 1)
        ],
        "limit_codes": [],
    }
    decided = []
    built = []

    def decide(cohort):
        decided.append(deepcopy(cohort))
        return deepcopy(pinned_receipt)

    def build(cohort, locators, *, tracker_receipts):
        built.append((deepcopy(cohort), deepcopy(locators), deepcopy(tracker_receipts)))
        return {"cohort_id": cohort["cohort_id"]}

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        measurement_mode="pinned_development_stage_stills",
        measurer=lambda *_args, **_kwargs: {
            "validity": "valid", "median_lab": [99.0, 99.0, 99.0],
            "mad_lab": [9.0, 9.0, 9.0], "iqr_lab": [9.0, 9.0, 9.0],
            "limit_codes": [],
        },
        deterministic_decider=decide, vlm_bundle_builder=build,
        vlm_runner=lambda bundle: ({"source_id": "stub-vlm", "bundle": bundle},),
        device="cpu", model_id="observer-test",
    )
    try:
        public = {}
        for position in range(1, 4):
            rows = adapter.process_frame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                frame_index=position, elapsed_seconds=float(position),
            )
            assert len(rows) == 9
            if position == 1:
                assert rows[0]["measurement"]["validity"] == "invalid"
            assert all(
                row["measurement"]["validity"] == "valid"
                for row in (rows[1:] if position == 1 else rows)
            )
            public[f"V{position}"] = rows[-1]

        decision = adapter.decide_cohort(public)
        bundle = adapter.build_vlm_bundle(public)
        attempts = tuple(adapter.run_vlm(bundle))

        assert len(adapter._image_locators) == 27
        assert {
            vial_id: {
                stage["stage_id"]: stage["measurement"]
                for stage in adapter._vlm_vials[vial_id]["stages"]
            }
            for vial_id in ("V1", "V2", "V3")
        } == original
        assert decided[0]["vials"][0]["stages"][0]["measurement"] == original["V1"][stage_ids[0]]
        assert decision["source_artifact_sha256"] == pinned_receipt["source_artifact_sha256"]
        assert built[0][0]["deterministic_decision_receipt"] == pinned_receipt
        assert attempts[0]["source_id"] == "stub-vlm"
    finally:
        adapter.close()


def test_default_live_mode_rejects_closed_incomplete_phase_topology() -> None:
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    context = {"camera_metadata": {"resolution": [4, 4], "vial_roi": [0, 0, 4, 4]}}

    class Tracker:
        def observe(self, _rgb, **_metadata):
            return ({
                "vial_id": "V1", "vial_position": 1, "fill_step": 1,
                "stage_id": "dispense-1-t100", "tank_id": "t100",
                "evidence_id": "VIS-V1-S1-end", "image_id": "IMG-V1-S1-end",
                "phase": "end", "same_run_empty_reference_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "observation_rgb": np.ones((4, 4, 3), dtype=np.uint8),
                "prediction_mask": np.ones((4, 4), dtype=bool),
                "vlm_cohort_context": context, "closed": True,
                "tracker_receipt": {"vial_id": "V1"},
                "vlm_vial": {
                    "vial_id": "V1", "vial_position": 1,
                    "stages": [{"stage_id": "dispense-1-t100", "measurement": {}}],
                },
            },)

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        measurer=lambda *_args, **_kwargs: {
            "validity": "valid", "median_lab": [1.0, 2.0, 3.0],
            "mad_lab": [0.0, 0.0, 0.0], "iqr_lab": [0.0, 0.0, 0.0],
            "limit_codes": [],
        },
        deterministic_decider=lambda _: {}, vlm_bundle_builder=lambda *_args: {},
        vlm_runner=lambda _: (), device="cpu", model_id="observer-test",
    )
    try:
        with pytest.raises(RuntimeError, match="complete early/mid/end phase topology"):
            adapter.process_frame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                frame_index=1, elapsed_seconds=0.1,
            )
    finally:
        adapter.close()


def test_visual_phase_validity_does_not_select_an_internal_live_measurement() -> None:
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
        rows = adapter.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_index=9, elapsed_seconds=0.9,
        )
        retained = dict(adapter._measurements)
    finally:
        adapter.close()

    assert rows[0]["mask"]["status"] == "bottom_zone_component_missing"
    assert rows[0]["measurement"]["validity"] == "invalid"
    assert rows[-1]["phase"] == "end"
    assert rows[-1]["measurement"]["validity"] == "valid"
    assert retained == {}

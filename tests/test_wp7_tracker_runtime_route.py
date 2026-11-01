from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from fluidmix.empty_reference_rgb_predictor import fillcam_masks
from fluidmix.wp7_fill_step_production import TopologyFreeze
from fluidmix.wp7_fill_step_tracker import (
    CALIBRATION_SCHEMA, FROZEN_FILLCAM_CAMERA_SHA256, FROZEN_GEOMETRY_SHA256,
    FROZEN_VOLUME_CURVE_SHA256, PixelToMetreCalibration, TrackerFreeze,
    calibration_payload_sha256,
)
from fluidmix.wp7_tracker_runtime import FrozenFillStepTrackerRuntime, TrackerLineage
from fluidmix.wp7_tracker_runtime import _frame_sequence_sha256, _production_block_metadata


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _lab(rgb):
    value = cv2.cvtColor(np.asarray([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    return value[0] * 100.0 / 255.0, float(value[1]) - 128.0, float(value[2]) - 128.0


def _freeze() -> TrackerFreeze:
    calibration = PixelToMetreCalibration(
        schema_version=CALIBRATION_SCHEMA, calibration_id="synthetic-route",
        camera_effective_state_sha256=FROZEN_FILLCAM_CAMERA_SHA256,
        geometry_sha256=FROZEN_GEOMETRY_SHA256,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
        fill_height_px_samples=(67.0, 87.0, 107.0),
        visible_height_m_samples=(0.01, 0.02, 0.03),
        twin_geometry_equivalent_volume_ml_samples=(10.0, 20.0, 30.0),
        artifact_sha256="0" * 64,
    )
    calibration = replace(
        calibration, artifact_sha256=calibration_payload_sha256(calibration)
    )
    return TrackerFreeze(
        stream_roi_xyxy=(300, 80, 320, 190), stream_rgb_change_threshold=8,
        stream_minimum_changed_pixels=100, stream_minimum_consecutive_frames=3,
        stream_change_delta_e76=8.0, tank_cluster_delta_e76=3.0,
        volume_match_tolerance_ml=0.1, calibration=calibration,
        tank_lab_centres={
            "t100": _lab((240, 30, 30)), "t200": _lab((30, 240, 30)),
            "t400": _lab((30, 30, 240)),
        }, recipe_artifact_sha256="a" * 64,
        volume_curve_sha256=FROZEN_VOLUME_CURVE_SHA256,
    )


class _Entry:
    def __init__(self, image_id, row, registry_sha256):
        self.image_id = image_id; self._row = row; self.registry_sha256 = registry_sha256
        self.rgb_sha256 = row["current_rgb_sha256"]; self.mask_sha256 = row["mask_sha256"]
        self.entry_sha256 = _digest({"image_id": image_id, **row})

    def measurement_locator(self):
        return {
            "rgb_path": self._row["current_rgb_path"], "rgb_sha256": self.rgb_sha256,
            "mask_path": self._row["mask_path"], "mask_sha256": self.mask_sha256,
        }


class _Registry:
    def __init__(self, images):
        self.registry_sha256 = _digest(images)
        self._entries = {key: _Entry(key, row, self.registry_sha256) for key, row in images.items()}

    def entry(self, image_id):
        return self._entries[image_id]


class _MovableModel:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self


def test_three_vial_two_stage_route_publishes_18_phases_and_materializes_vlm(
    tmp_path: Path, monkeypatch,
) -> None:
    import fluidmix.vision_measurement as measurements
    import fluidmix.wp7_tracker_runtime as module
    import fluidmix.carryover_pipeline  # bind production imports before local doubles
    import fluidmix.wp7_rgb_runtime as rgb_runtime
    from test_wp7_vlm_decision import _runtime_files

    roi, _ = fillcam_masks()
    references = ((5, 5, 5), (6, 6, 6), (7, 7, 7))
    colours = ((240, 30, 30), (30, 240, 30))
    frames = []
    surfaces = {}
    for vial_index, reference_colour in enumerate(references):
        reference = np.full((480, 640, 3), reference_colour, dtype=np.uint8)
        frames.append(reference)
        for stage_index, colour in enumerate(colours):
            for _ in range(3):
                frame = reference.copy(); frame[80:190, 300:320] = colour
                surfaces[len(frames)] = 350 if stage_index == 0 else 330
                frames.append(frame)
    frames.append(np.full((480, 640, 3), 8, dtype=np.uint8))

    def predict(_self, current, _reference):
        index = next(i for i, frame in enumerate(frames) if frame is current)
        surface = surfaces.get(index, 350)
        mask = np.zeros((480, 640), dtype=bool); mask[surface:417] = roi[surface:417]
        return SimpleNamespace(mask=mask, status="visible_measured", surface_row=surface)

    # ndarray identity is lost at the tracker boundary, so advance deterministically.
    predicted_surfaces = iter((350,) * 3 + (330,) * 3) 
    current_vial = {"calls": 0}
    def deterministic_predict(_self, _current, _reference):
        call = current_vial["calls"]
        surface = 350 if call % 12 < 6 else 330
        # Tracker plus selected-mask replay makes twelve calls per vial stage pair.
        current_vial["calls"] += 1
        mask = np.zeros((480, 640), dtype=bool); mask[surface:417] = roi[surface:417]
        return SimpleNamespace(mask=mask, status="visible_measured", surface_row=surface)

    # Keep tracker prediction stage-aware and replay raw masks from the persisted topology.
    tracker_calls = {"value": 0}
    def stage_predict(_self, _current, _reference):
        index = tracker_calls["value"] % 6
        tracker_calls["value"] += 1
        surface = 350 if index < 3 else 330
        mask = np.zeros((480, 640), dtype=bool); mask[surface:417] = roi[surface:417]
        return SimpleNamespace(mask=mask, status="visible_measured", surface_row=surface)
    monkeypatch.setattr(FrozenFillStepTrackerRuntime, "_predict", stage_predict)
    monkeypatch.setattr(
        rgb_runtime, "select_dispense_endpoints",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy selector was invoked")
        ),
    )
    monkeypatch.setattr(module, "build_trusted_real_rgb_mask_registry_v1",
                        lambda artifact, _loaded: _Registry(artifact["images"]))

    binding_box = {}
    def fake_measured(registry, requests):
        rows = tuple(SimpleNamespace(
            image_id=image_id,
            measurement_sha256=_digest({"image_id": image_id, **request}),
        ) for image_id, request in requests.items())
        binding_box["rows"] = rows
        return SimpleNamespace(registry=registry)
    monkeypatch.setattr(measurements, "build_trusted_runtime_measured_evidence_v1", fake_measured)
    monkeypatch.setattr(measurements, "trusted_measurement_bindings_from_result_v1",
                        lambda _value: binding_box["rows"])

    transitions = []
    for index, start in enumerate((0, 7, 14, 21)):
        transitions.append({
            "segment_index": index, "start_frame_index": start,
            "stop_frame_index": start + 1, "frame_count": 1,
            "decoded_bgr_sha256": [hashlib.sha256(frames[start][..., ::-1].copy().tobytes()).hexdigest()],
        })
    capture = {
        "schema_version": "wp7-camera-capture-v1", "artifact_type": "wp7_camera_capture",
        "capture_origin": "real_camera", "video_sha256": hashlib.sha256(b"video").hexdigest(),
        "encoded_transition_evidence": transitions,
        "fill_clips": [{"case_id": f"CASE-{i}", "frame_count": 7} for i in (1, 2, 3)],
    }
    body = dict(capture); capture["receipt_sha256"] = _digest(body)
    recipe = {
        "schema_version": "carryover-recipe-sequence-v1", "split_role": "real_check",
        "split_version": "synthetic-route-v1", "runs": [{"run_id": "RUN-1", "vials": [{
            "case_id": f"CASE-{position}", "vial_ordinal": position,
            "block_recipe_id": "BLOCK-1", "block_position": position,
            "recipe_id": "RECIPE-1", "previous_recipe_id": "RECIPE-0" if position == 1 else "RECIPE-1",
            "programmed_tank_ml": {"t100": 10.0, "t200": 10.0, "t400": 0.0},
            "dispense_order": ["t100", "t200"],
        } for position in (1, 2, 3)]}]}
    _, _, _, vlm_freeze, vlm_freeze_sha = _runtime_files(tmp_path / "vlm-freeze")
    model = _MovableModel()
    runtime = FrozenFillStepTrackerRuntime(
        loaded_cnn=SimpleNamespace(model=model, lineage_paths={
            "checkpoint": __file__, "replay": __file__, "review": __file__,
        }), capture_receipt=capture, freeze=_freeze(),
        topology=TopologyFreeze("a" * 64, "b" * 64, 0.5, 4, 392, 4, 8),
        lineage=TrackerLineage(*(["c" * 64] * 8)),
        sidecar_out=tmp_path / "tracker-sidecar.json", device="cpu",
        target_recipe_raw_sha256="d" * 64,
        target_recipe_semantic_sha256=_digest(recipe),
        initial_previous_final_tank_id="t400", frame_rate_hz=30.0,
        camera_metadata={
            "resolution": [640, 480], "frame_rate_hz": 30.0,
            "exposure": "fixed", "white_balance": "fixed",
            "viewpoint_id": "fillcam", "vial_roi": [283, 194, 338, 417],
        }, vlm_freeze_path=vlm_freeze, vlm_freeze_sha256=vlm_freeze_sha,
        vlm_output_root=tmp_path / "vlm", frame_decoder=lambda _raw: frames,
    )
    output = runtime.analyze(video_bytes=b"video", recipe_sequence=recipe,
                             artifact_root=tmp_path / "artifacts")
    sidecar = json.loads((tmp_path / "tracker-sidecar.json").read_text())
    assert sidecar["selected_image_count"] == 18
    assert model.moves == ["cpu"]
    assert sidecar["expected_vial_count"] == sidecar["represented_vial_count"] == 3
    assert sidecar["accepted_receipt_count"] == 3
    assert len(sidecar["vlm_materializations"]) == 1
    assert sidecar["vlm_materializations"][0]["image_count"] == 18
    assert output.runtime_provenance["legacy_selector_used"] is False
    assert (tmp_path / "vlm" / "RUN-1:BLOCK-1" / "cohort.json").is_file()
    assert all(
        image_id.startswith("CASE-") and "-F" in image_id
        for image_id in sidecar["phase_locators"]
    )


def test_device_and_unverified_frame_rate_fail_closed(tmp_path: Path) -> None:
    common = dict(
        loaded_cnn=SimpleNamespace(model=_MovableModel()), capture_receipt={}, freeze=_freeze(),
        topology=TopologyFreeze("a" * 64, "b" * 64, 0.5, 4, 392, 4, 8),
        lineage=TrackerLineage(*(["c" * 64] * 8)), sidecar_out=tmp_path / "sidecar",
        target_recipe_raw_sha256="d" * 64, target_recipe_semantic_sha256="e" * 64,
        initial_previous_final_tank_id="t400", camera_metadata={},
        vlm_freeze_path=tmp_path / "freeze", vlm_freeze_sha256="f" * 64,
        vlm_output_root=tmp_path / "vlm", frame_decoder=lambda _: (),
    )
    import pytest
    with pytest.raises(ValueError, match="device"):
        FrozenFillStepTrackerRuntime(**common, device="gpu", frame_rate_hz=30.0)
    with pytest.raises(ValueError, match="30.0 fps"):
        FrozenFillStepTrackerRuntime(**common, device="cpu", frame_rate_hz=29.97)
    closed_camera = {
        "resolution": [640, 480], "frame_rate_hz": 30.0,
        "exposure": "fixed", "white_balance": "fixed", "viewpoint_id": "fillcam",
        "vial_roi": [283, 194, 338, 417], "forbidden_extra": True,
    }
    with pytest.raises(ValueError, match="fields are not closed"):
        FrozenFillStepTrackerRuntime(
            **{**common, "camera_metadata": closed_camera},
            device="cpu", frame_rate_hz=30.0,
        )


def test_cuda_model_moves_once_at_runtime_construction(tmp_path: Path) -> None:
    model = _MovableModel()
    FrozenFillStepTrackerRuntime(
        loaded_cnn=SimpleNamespace(model=model), capture_receipt={}, freeze=_freeze(),
        topology=TopologyFreeze("a" * 64, "b" * 64, 0.5, 4, 392, 4, 8),
        lineage=TrackerLineage(*(["c" * 64] * 8)), sidecar_out=tmp_path / "sidecar",
        device="cuda", target_recipe_raw_sha256="d" * 64,
        target_recipe_semantic_sha256="e" * 64,
        initial_previous_final_tank_id="t400", frame_rate_hz=30.0,
        camera_metadata={
            "resolution": [640, 480], "frame_rate_hz": 30.0,
            "exposure": "fixed", "white_balance": "fixed",
            "viewpoint_id": "fillcam", "vial_roi": [283, 194, 338, 417],
        }, vlm_freeze_path=tmp_path / "freeze", vlm_freeze_sha256="f" * 64,
        vlm_output_root=tmp_path / "vlm", frame_decoder=lambda _: (),
    )
    assert model.moves == ["cuda"]


def test_absolute_vials_four_to_six_bind_a_distinct_exact_cohort_recipe_hash() -> None:
    recipe = {
        "schema_version": "carryover-recipe-sequence-v1", "split_role": "real_check",
        "split_version": "two-blocks-v1", "runs": [{"run_id": "RUN", "vials": []}],
    }
    run = recipe["runs"][0]
    for ordinal in range(1, 7):
        block = 1 if ordinal <= 3 else 2
        run["vials"].append({
            "case_id": f"CASE-{ordinal}", "vial_ordinal": ordinal,
            "block_recipe_id": f"BLOCK-{block}",
            "block_position": ((ordinal - 1) % 3) + 1,
            "recipe_id": f"RECIPE-{block}", "previous_recipe_id": None,
            "programmed_tank_ml": {"t100": 10.0, "t200": 10.0, "t400": 0.0},
            "dispense_order": ["t100", "t200"],
        })
    metadata, hashes = _production_block_metadata(
        recipe, [(run, vial) for vial in run["vials"]]
    )
    assert hashes[("RUN", "BLOCK-1")] != hashes[("RUN", "BLOCK-2")]
    second = metadata[("RUN", "BLOCK-2")]
    assert second["runs"][0]["run_id"] == "RUN:BLOCK-2"
    assert [row["vial_ordinal"] for row in second["runs"][0]["vials"]] == [4, 5, 6]
    assert hashes[("RUN", "BLOCK-2")] == _digest(second)


def test_closed_tracker_abstention_codes_are_accepted_but_unknown_codes_fail() -> None:
    import pytest
    from fluidmix.carryover_pipeline import (
        RuntimeProjectionInput, _runtime_provenance_vials,
    )

    def output(code: str):
        return RuntimeProjectionInput(
            record_templates=(), trusted_adapter_result=object(), image_locators={},
            event_cursors={}, runtime_provenance={"vials": [{
                "case_id": f"CASE-{index}", "selection_status": "abstain",
                "selected_frame_indices": [], "measurement_citation_sha256s": [],
                "limit_codes": [code],
            } for index in range(1, 31)]},
        )

    accepted = _runtime_provenance_vials(output("tracker_stable_tail_missing"))
    assert len(accepted) == 30
    with pytest.raises(ValueError, match="runtime provenance"):
        _runtime_provenance_vials(output("arbitrary_runtime_exception"))


def test_incremental_vial_source_hash_matches_concatenated_frame_bytes() -> None:
    frames = tuple(
        np.full((4, 5, 3), value, dtype=np.uint8) for value in (1, 2, 3)
    )
    expected = hashlib.sha256(b"".join(frame.tobytes() for frame in frames)).hexdigest()
    assert _frame_sequence_sha256(frames) == expected

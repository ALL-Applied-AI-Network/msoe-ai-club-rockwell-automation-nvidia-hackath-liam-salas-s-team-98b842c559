import hashlib
import json

import numpy as np
import pytest


def test_vlm_freeze_exactly_pins_attempt19_onion_rebind():
    from fluidmix.pipeline.service import production_factory as module

    assert module.VLM_FREEZE.name == "freeze.json"
    assert module.VLM_FREEZE.parent.name == (
        "wp7-vlm-development-freeze-01a18128-attempt19-onion-rebind"
    )
    assert module.EXPECTED_HASHES[module.VLM_FREEZE] == (
        "3d7468cd69e110115bdfe7847e8e578b2d1d4c0b105468fd7185c04c6bd040f8"
    )
    assert module.VLM_DECISION_SOURCE.name == "decision.py"
    assert module.VLM_DECISION_SOURCE.parent.name == "vlm"


def test_vlm_freeze_is_pinned_to_onion_decision_source(monkeypatch, tmp_path):
    from fluidmix.pipeline.service import production_factory as module

    onion = tmp_path / "decision.py"
    onion.write_bytes(b"onion decision source\n")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "schema_version": "wp7-vlm-development-freeze-v13",
        "models": [
            {"repository": repository, "revision": revision}
            for repository, revision in module.MODEL_REVISIONS.items()
        ],
        "decision_source_sha256": hashlib.sha256(onion.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(module, "VLM_FREEZE", freeze)
    monkeypatch.setattr(module, "VLM_DECISION_SOURCE", onion)

    module.validate_artifacts({freeze: hashlib.sha256(freeze.read_bytes()).hexdigest()})

    onion.write_bytes(b"drifted onion decision source\n")
    with pytest.raises(module.ProductionFactoryError, match="decision source"):
        module.validate_artifacts({freeze: hashlib.sha256(freeze.read_bytes()).hexdigest()})


def test_artifact_hash_validation_fails_closed(tmp_path):
    from fluidmix.pipeline.service.production_factory import (
        ProductionFactoryError, validate_artifacts,
    )

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ProductionFactoryError, match="SHA-256 mismatch"):
        validate_artifacts({artifact: "0" * 64})


@pytest.mark.parametrize("kind", ["file", "video", "v4l2", "rtsp"])
def test_session_profile_admits_all_frontend_transports(kind):
    from fluidmix.pipeline.service.production_factory import validate_session_profile

    validate_session_profile({
        "source": {"kind": kind},
        "run_context": {"recipe_id": "5"},
    })


def test_session_profile_rejects_recipe_drift():
    from fluidmix.pipeline.service.production_factory import (
        ProductionFactoryError, validate_session_profile,
    )

    with pytest.raises(ProductionFactoryError, match="recipe 5"):
        validate_session_profile({
            "source": {"kind": "file"}, "run_context": {"recipe_id": "6"},
        })


def test_frozen_timing_is_expressed_in_seconds_and_scaled_to_source_fps():
    from fluidmix.pipeline.service.production_factory import FrozenTiming

    timing = FrozenTiming.for_source_fps(30.0)

    assert timing.canonical_fps == 10.0
    assert timing.stream_debounce_seconds == 1.2
    assert timing.stream_debounce_source_frames == 36
    assert timing.stable_tail_seconds == 0.3
    assert timing.stable_tail_source_frames == 9


def test_session_tracker_rotates_from_detected_fill_then_dwell():
    from fluidmix.pipeline.service.production_factory import SessionVialTracker

    built = []

    class Tracker:
        def __init__(self, index):
            self.index = index
            self.frames = []

        def observe(self, _frame, *, frame_index, elapsed_seconds):
            self.frames.append((frame_index, elapsed_seconds))
            return ()

        def close(self):
            return ({"fill_step": 1, "tank_id": "t200", "closed": True},)

        def bind_source_artifact_sha256(self, _digest):
            pass

    def builder(index, _reference):
        tracker = Tracker(index)
        built.append(tracker)
        return tracker

    states = iter(
        [None] + ["t200"] * 12 + ["t400"] * 12 + ["t100"] * 12 + [None] * 3
    )
    wrapper = SessionVialTracker(
        builder, auto_close=True,
        stream_state=lambda _frame, _reference: next(states),
    )
    wrapper.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})
    assert wrapper.source_profile_evidence["timing_conversions"] == {
        "stream_debounce": "12 frames / 10 Hz = 1.2 seconds",
        "stable_tail": "3 frames / 10 Hz = 0.3 seconds",
    }
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    closed = ()
    for index in range(41):
        closed = wrapper.observe(frame, frame_index=index, elapsed_seconds=index / 10)
    assert built[0].frames[0] == (0, 0.0)
    assert closed[0]["stage_id"] == "dispense-1-t200"
    assert wrapper.completed_vial_count == 1


def test_session_tracker_keeps_first_empty_reference_during_gradual_stream_onset():
    import hashlib
    from fluidmix.pipeline.service.production_factory import SessionVialTracker

    built = []

    class Tracker:
        def __init__(self, reference):
            self.reference = reference.copy()
            self.frames = []

        def observe(self, frame, *, frame_index, elapsed_seconds):
            self.frames.append((frame.copy(), frame_index, elapsed_seconds))
            return ()

    def builder(_index, reference):
        tracker = Tracker(reference)
        built.append(tracker)
        return tracker

    def state(frame, reference):
        return "t200" if int(frame[0, 0, 0]) - int(reference[0, 0, 0]) >= 4 else None

    wrapper = SessionVialTracker(builder, stream_state=state)
    wrapper.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})
    for index, value in enumerate((10, 11, 12, 13)):
        wrapper.observe(
            np.full((4, 4, 3), value, dtype=np.uint8),
            frame_index=index, elapsed_seconds=index / 10,
        )
    assert built == []

    wrapper.observe(
        np.full((4, 4, 3), 14, dtype=np.uint8),
        frame_index=4, elapsed_seconds=0.4,
    )

    assert len(built) == 1
    assert int(built[0].reference[0, 0, 0]) == 10
    assert [(index, elapsed) for _frame, index, elapsed in built[0].frames] == [(0, 0.0)]
    digest = hashlib.sha256()
    for value in (10, 11, 12, 13, 14):
        digest.update(np.full((4, 4, 3), value, dtype=np.uint8).tobytes())
    assert wrapper._window_hash.hexdigest() == digest.hexdigest()


def test_session_tracker_closes_on_topology_cleaned_empty_false_active_seam():
    import hashlib
    from types import SimpleNamespace
    from fluidmix.pipeline.service.production_factory import SessionVialTracker
    from fluidmix.perception.tracker.fill_step_production import (
        TopologyFreeze, postprocess_cnn_mask_topology,
    )

    topology = TopologyFreeze("a" * 64, "b" * 64, 0.5, 4, 392, 4, 8)
    raw_noise = np.zeros((480, 640), dtype=bool)
    raw_noise[300, 300:313] = True
    cleaned = postprocess_cnn_mask_topology(raw_noise, **topology.parameters())
    assert np.count_nonzero(raw_noise) == 13
    assert not np.any(cleaned.mask)
    assert cleaned.reason_code == "bottom_zone_component_missing"

    built = []

    class Tracker:
        def __init__(self, reference):
            self.reference = reference.copy()
            self.frames = []
            self.bound = None
            self.topology = topology
            self.predictor = lambda _rgb, _reference: SimpleNamespace(mask=raw_noise)

        def observe(self, frame, *, frame_index, elapsed_seconds):
            self.frames.append((frame.copy(), frame_index, elapsed_seconds))
            return ()

        def bind_source_artifact_sha256(self, digest):
            self.bound = digest

        def close(self):
            return ({"fill_step": 1, "tank_id": "t200", "closed": True},)

    def builder(_index, reference):
        tracker = Tracker(reference)
        built.append(tracker)
        return tracker

    states = iter(
        ["t200"] * 12 + ["t400"] * 12 + ["t100"] * 12
        + ["t100"] * 3 + ["t200"]
    )
    wrapper = SessionVialTracker(
        builder, auto_close=True,
        stream_state=lambda _frame, _reference: next(states),
    )
    wrapper.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})
    wrapper.observe(
        np.full((480, 640, 3), 10, dtype=np.uint8),
        frame_index=0, elapsed_seconds=0.0,
    )
    active_frames = []
    for index in range(1, 37):
        frame = np.full((480, 640, 3), 20 + index, dtype=np.uint8)
        active_frames.append(frame)
        wrapper.observe(frame, frame_index=index, elapsed_seconds=index / 10)

    closed = ()
    for offset, value in enumerate((80, 81, 82), start=37):
        closed = wrapper.observe(
            np.full((480, 640, 3), value, dtype=np.uint8),
            frame_index=offset, elapsed_seconds=offset / 10,
        )

    assert closed and wrapper.completed_vial_count == 1
    assert len(built[0].frames) == 36
    digest = hashlib.sha256()
    digest.update(np.full((480, 640, 3), 10, dtype=np.uint8).tobytes())
    for frame in active_frames:
        digest.update(frame.tobytes())
    assert built[0].bound == digest.hexdigest()

    prefix_digest = hashlib.sha256()
    for value in (80, 81, 82):
        prefix_digest.update(np.full((480, 640, 3), value, dtype=np.uint8).tobytes())
    assert wrapper._window_hash.hexdigest() == prefix_digest.hexdigest()

    wrapper.observe(
        np.full((480, 640, 3), 90, dtype=np.uint8),
        frame_index=40, elapsed_seconds=4.0,
    )
    assert len(built) == 2
    assert np.all(built[1].reference == 80)


def test_session_tracker_binds_buffered_seam_frames_when_seam_reopens(monkeypatch):
    import hashlib
    from fluidmix.pipeline.service import production_factory as module

    built = []

    class Tracker:
        def __init__(self):
            self.bound = None
            self.frames = []

        def observe(self, frame, *, frame_index, elapsed_seconds):
            self.frames.append((frame.copy(), frame_index, elapsed_seconds))
            return ()

        def bind_source_artifact_sha256(self, digest):
            self.bound = digest

        def close(self):
            return ({"fill_step": 1, "tank_id": "t200", "closed": True},)

    def builder(_index, _reference):
        tracker = Tracker()
        built.append(tracker)
        return tracker

    # Keep the real observe/close path, while making the seam classification
    # deterministic for this provenance-only boundary test.
    monkeypatch.setattr(
        module.SessionVialTracker,
        "_topology_cleaned_empty",
        staticmethod(lambda _tracker, frame: int(frame[0, 0, 0]) != 82),
    )
    states = iter(
        ["t200"] * 12 + ["t400"] * 12 + ["t100"] * 12 + ["t100"] * 6
    )
    wrapper = module.SessionVialTracker(
        builder,
        auto_close=True,
        stream_state=lambda _frame, _reference: next(states),
    )
    wrapper.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})

    wrapper.observe(
        np.full((480, 640, 3), 10, dtype=np.uint8),
        frame_index=0,
        elapsed_seconds=0.0,
    )
    for index in range(1, 37):
        wrapper.observe(
            np.full((480, 640, 3), 20 + index, dtype=np.uint8),
            frame_index=index,
            elapsed_seconds=index / 10,
        )

    for index, value in enumerate((80, 81, 82, 83, 84, 85), start=37):
        closed = wrapper.observe(
            np.full((480, 640, 3), value, dtype=np.uint8),
            frame_index=index,
            elapsed_seconds=index / 10,
        )

    assert closed and wrapper.completed_vial_count == 1
    digest = hashlib.sha256()
    for value in (10, *range(21, 57), 80, 81, 82):
        digest.update(np.full((480, 640, 3), value, dtype=np.uint8).tobytes())
    assert built[0].bound == digest.hexdigest()


def test_session_tracker_uses_injected_frame_rate_for_elapsed_and_vlm_timestamps():
    from fluidmix.pipeline.service.production_factory import SessionVialTracker

    observed = []

    class Tracker:
        def observe(self, _frame, *, frame_index, elapsed_seconds):
            observed.append((frame_index, elapsed_seconds))
            return ()

        def close(self):
            return ({
                "fill_step": 1, "tank_id": "t200", "closed": True,
                "tracker_receipt": {"stages": [{
                    "images": {"early": "IMG-F0002", "mid": "IMG-F0004", "end": "IMG-F0006"},
                }]},
                "vlm_vial": {"stages": [{}]},
            },)

    wrapper = SessionVialTracker(
        lambda *_args: Tracker(), stream_state=lambda *_args: "t200",
    )
    wrapper.configure_source_profile({"fps": 20.0, "resolution": [640, 480]})
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    wrapper.observe(frame, frame_index=0, elapsed_seconds=99.0)
    wrapper.observe(frame, frame_index=2, elapsed_seconds=99.1)
    closed = wrapper.close()

    assert observed == [(0, 0.0)]
    assert [row["timestamp_seconds"] for row in closed[0]["vlm_vial"]["stages"][0]["frames"]] == [
        0.1, 0.2, 0.3,
    ]


@pytest.mark.parametrize("frame_rate_hz", [None, True, 0, -1, float("nan")])
def test_session_tracker_rejects_invalid_source_frame_rate(frame_rate_hz):
    from fluidmix.pipeline.service.production_factory import (
        ProductionFactoryError, SessionVialTracker,
    )

    with pytest.raises(ProductionFactoryError, match="frame rate"):
        SessionVialTracker(lambda *_args: object()).configure_source_profile(
            {"fps": frame_rate_hz, "resolution": [640, 480]}
        )


def test_session_tracker_auto_closes_detected_windows_with_rolling_rgb_hashes(monkeypatch):
    import hashlib
    from fluidmix.pipeline.service import production_factory as module

    built = []

    class Tracker:
        def __init__(self):
            self.frames, self.bound = [], None

        def observe(self, frame, *, frame_index, elapsed_seconds):
            self.frames.append((frame.copy(), frame_index, elapsed_seconds))
            return ()

        def bind_source_artifact_sha256(self, digest):
            self.bound = digest

        def close(self):
            return ({"fill_step": 1, "tank_id": "t200", "closed": True},)

    def builder(_index, _reference):
        tracker = Tracker(); built.append(tracker); return tracker

    states = iter(
        [None] + ["t200"] * 12 + ["t400"] * 12 + ["t100"] * 12 + [None] * 3
    )
    wrapper = module.SessionVialTracker(
        builder, auto_close=True,
        stream_state=lambda _frame, _reference: next(states),
    )
    wrapper.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})
    for index in range(41):
        frame = np.full((480, 640, 3), index + 1, dtype=np.uint8)
        rows = wrapper.observe(frame, frame_index=index, elapsed_seconds=index / 10)
    digest = hashlib.sha256()
    for index in range(41):
        digest.update(np.full((480, 640, 3), index + 1, dtype=np.uint8).tobytes())
    assert [tracker.bound for tracker in built] == [digest.hexdigest()]
    assert wrapper.completed_vial_count == 1
    assert wrapper.session_complete is False
    assert wrapper.requires_complete is True


@pytest.mark.parametrize(("states", "reason"), [
    ([], "tracker_stream_state_missing"),
    (["t200"] * 12, "tracker_incomplete_or_disagreed"),
    (["t400"] * 12, "tracker_stream_order_disagrees"),
    (
        ["t200"] * 12 + ["t400"] * 12 + ["t100"] * 12 + ["t200"] * 12,
        "tracker_stream_state_count_disagrees",
    ),
    (
        ["t200"] * 12 + ["t400"] * 12 + ["t100"] * 12,
        "tracker_stable_tail_missing",
    ),
])
def test_session_tracker_classifies_incomplete_eof_from_observed_stream_states(
    states, reason,
):
    from fluidmix.pipeline.service.production_factory import SessionVialTracker

    class Tracker:
        def observe(self, _frame, **_metadata):
            return ()

    observed_states = iter(states)
    wrapper = SessionVialTracker(
        lambda *_args: Tracker(), auto_close=True,
        stream_state=lambda *_args: next(observed_states),
    )
    wrapper.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    wrapper.observe(frame, frame_index=0, elapsed_seconds=0.0)
    for index in range(len(states)):
        wrapper.observe(frame, frame_index=index + 1, elapsed_seconds=(index + 1) / 10)

    assert wrapper.eof_abstention_reason == reason


def test_vlm_runner_projects_only_per_model_attempt_rows():
    from fluidmix.pipeline.service.production_factory import model_attempt_rows

    aggregate = {
        "sources": [
            {"source_id": "deterministic", "discard_score": 1.0},
            {"source_id": "model-a", "repository": "repo/a", "revision": "abc"},
            {"source_id": "model-b", "repository": "repo/b", "revision": "def"},
            {"source_id": "model-c", "repository": "repo/c", "revision": "ghi"},
        ]
    }
    assert model_attempt_rows(aggregate) == tuple(aggregate["sources"][1:])


def test_deterministic_measurement_shape_preserves_aligned_stage_ids():
    from fluidmix.pipeline.service.production_factory import deterministic_stage_measurements

    ids = ("dispense-1-t200", "dispense-2-t400", "dispense-3-t100")
    cohort = {"vials": [{"stages": [
        {"stage_id": stage_id, "measurement": {"median_lab": [n, ordinal, 0]}}
        for ordinal, stage_id in enumerate(ids, 1)
    ]} for n in (1, 2, 3)]}
    rows = deterministic_stage_measurements(cohort)
    assert [stage[0]["stage_id"] for stage in rows] == list(ids)
    assert all({row["stage_id"] for row in stage} == {ids[index]}
               for index, stage in enumerate(rows))
    assert "stage_id" not in cohort["vials"][0]["stages"][0]["measurement"]


def test_cnn_model_uses_selected_gpu_without_mutating_loaded_cpu_model():
    from fluidmix.pipeline.service.production_factory import prepare_cnn_model

    class Model:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

    model = Model()
    prepared = prepare_cnn_model(model, "cuda")
    assert prepared is not model
    assert model.moves == []
    assert prepared.moves == ["cuda"]


def test_runner_rgb_is_preserved_as_contiguous_tracker_rgb_pixels():
    from fluidmix.pipeline.service.production_factory import normalize_tracker_frame

    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)[:, ::-1]
    normalized = normalize_tracker_frame(rgb)
    assert normalized.tolist() == [[[4, 5, 6], [1, 2, 3]]]
    assert normalized.flags.c_contiguous


def test_fillcam_view_scales_proportional_native_roi_outward():
    from fluidmix.pipeline.service.production_factory import normalize_fillcam_view

    native = np.full((600, 800, 3), 20, dtype=np.uint8)
    canonical, evidence = normalize_fillcam_view(native)

    assert canonical.shape == (480, 640, 3)
    assert evidence["native_resolution"] == [800, 600]
    assert evidence["native_vial_roi"] == [353, 242, 423, 522]
    assert evidence["canonical_vial_roi"] == [283, 194, 338, 417]


def test_fillcam_view_rejects_non_proportional_or_black_input():
    from fluidmix.pipeline.service.production_factory import (
        ProductionFactoryError, normalize_fillcam_view,
    )

    with pytest.raises(ProductionFactoryError, match="camera view"):
        normalize_fillcam_view(np.ones((480, 800, 3), dtype=np.uint8))
    with pytest.raises(ProductionFactoryError, match="camera view"):
        normalize_fillcam_view(np.zeros((480, 640, 3), dtype=np.uint8))


def test_arbitrary_file_and_live_sources_reach_same_frozen_factory(tmp_path):
    from fluidmix.pipeline.service import production_factory as module

    arbitrary = tmp_path / "arbitrary.mp4"
    arbitrary.write_bytes(b"not-the-accepted-v4-development-file")
    factory = module.DefaultProductionFactory()
    seen = []

    class FrozenFactory:
        def for_session(self, request, device):
            seen.append((request["source"]["kind"], device))
            return object()

    factory._factory = FrozenFactory()
    sources = (
        {"kind": "file", "path": str(arbitrary), "label": "arbitrary.mp4"},
        {"kind": "v4l2", "device": "/dev/video0"},
        {"kind": "rtsp", "url": "rtsp://camera.example/live"},
    )
    for index, source in enumerate(sources):
        result = factory.for_session({
            "source": source,
            "run_context": {
                "operator_run_id": f"run-{index}",
                "recipe_id": "5", "tray_id": "tray-1",
            },
        }, "cpu")
        assert result is not None
    assert seen == [("file", "cpu"), ("v4l2", "cpu"), ("rtsp", "cpu")]


def test_operator_context_builds_session_causal_cohort_without_frozen_case_ids():
    from fluidmix.pipeline.service import production_factory as module

    context, vials = module.build_session_cohort({
        "operator_run_id": "run-17", "recipe_id": "5", "tray_id": "tray-A",
    }, frame_rate_hz=25.0)

    assert context["cohort_id"] == "run-17:tray-A"
    assert context["recipe_id"] == "5"
    assert context["partition"] == "production"
    assert context["camera_metadata"]["frame_rate_hz"] == 25.0
    assert [row["vial_id"] for row in vials] == [
        "run-17:tray-A:vial-1", "run-17:tray-A:vial-2", "run-17:tray-A:vial-3",
    ]
    assert all(row["stages"] == [] for row in vials)
    assert not ({"CASE_97a8c6cb8161881c", "CASE_59038b977a73fdfb",
                 "CASE_0b872555c4f08091"} & {row["vial_id"] for row in vials})


def test_all_transports_use_live_session_measurements(tmp_path):
    from fluidmix.pipeline.service import production_factory as module

    arbitrary = tmp_path / "arbitrary.mp4"
    arbitrary.write_bytes(b"arbitrary-video-bytes")
    assert module.measurement_mode_for_source({
        "kind": "file", "path": str(arbitrary),
    }) == "generic_live"
    assert module.measurement_mode_for_source({
        "kind": "v4l2", "device": "/dev/video0",
    }) == "generic_live"

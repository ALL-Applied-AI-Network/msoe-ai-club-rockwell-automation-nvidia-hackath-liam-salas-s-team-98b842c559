import json
from pathlib import Path

import numpy as np
import pytest


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


def test_session_tracker_rotates_only_at_boundaries():
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

    def builder(index, _reference):
        tracker = Tracker(index)
        built.append(tracker)
        return tracker

    wrapper = SessionVialTracker(builder)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    wrapper.observe(frame, frame_index=0, elapsed_seconds=0.0)
    wrapper.observe(frame, frame_index=336, elapsed_seconds=33.6)
    wrapper.observe(frame, frame_index=337, elapsed_seconds=33.7)
    assert built[0].frames == [(0, 0.0), (336, 33.6)]
    assert wrapper.close()[0]["stage_id"] == "dispense-1-t200"
    wrapper.observe(frame, frame_index=337, elapsed_seconds=33.7)
    assert [row.index for row in built] == [0, 1]
    assert built[1].frames == [(0, 0.0)]


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

    wrapper = SessionVialTracker(lambda *_args: Tracker(), frame_rate_hz=20.0)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    wrapper.observe(frame, frame_index=0, elapsed_seconds=99.0)
    wrapper.observe(frame, frame_index=2, elapsed_seconds=99.1)
    closed = wrapper.close()

    assert observed == [(0, 0.0), (2, 0.1)]
    assert [row["timestamp_seconds"] for row in closed[0]["vlm_vial"]["stages"][0]["frames"]] == [
        0.1, 0.2, 0.3,
    ]


@pytest.mark.parametrize("frame_rate_hz", [None, True, 0, -1, float("nan")])
def test_session_tracker_rejects_invalid_injected_frame_rate(frame_rate_hz):
    from fluidmix.pipeline.service.production_factory import (
        ProductionFactoryError, SessionVialTracker,
    )

    with pytest.raises(ProductionFactoryError, match="frame rate"):
        SessionVialTracker(lambda *_args: object(), frame_rate_hz=frame_rate_hz)


def test_session_tracker_auto_closes_exact_windows_with_rolling_rgb_hashes(monkeypatch):
    import hashlib
    from fluidmix.pipeline.service import production_factory as module

    monkeypatch.setattr(module, "FILL_STARTS", (0, 2, 4))
    monkeypatch.setattr(module, "FILL_STOPS", (2, 4, 6))
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

    wrapper = module.SessionVialTracker(builder, auto_close=True)
    for index in range(6):
        frame = np.full((480, 640, 3), index, dtype=np.uint8)
        rows = wrapper.observe(frame, frame_index=index, elapsed_seconds=index / 10)
        assert bool(rows) is (index in {1, 3, 5})
    expected = []
    for start in (0, 2, 4):
        digest = hashlib.sha256()
        for index in range(start, start + 2):
            digest.update(np.full((480, 640, 3), index, dtype=np.uint8).tobytes())
        expected.append(digest.hexdigest())
    assert [tracker.bound for tracker in built] == expected
    assert wrapper.session_complete is True
    assert wrapper.requires_complete is True


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


def test_native_concat_decodes_to_three_exact_accepted_windows():
    from fluidmix.pipeline.service.production_factory import _boundary_hashes

    source = Path(
        "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
        "runner-production-factory-01a18128/accepted-v4-train01-cohort.mp4"
    )
    if not source.is_file():
        pytest.skip("accepted native V4 cohort profile is unavailable")
    assert _boundary_hashes(source) == (
        "d8b5850e7133786f8e120cb154cf7dd1880279a0f9f37af975a8bdadfab10496",
        "2d67573608dd89543fb0219346665d4458934aa48aac2021f5b1f2bbd1f8db0f",
        "7e31f3a5aeb18e530a6de4cf18e51405c9acf8481e5f9e70817d468d46324a5a",
    )


def test_arbitrary_file_and_live_sources_abstain_before_heavy_factory_or_media(
    tmp_path, monkeypatch,
):
    from fluidmix.pipeline.service import production_factory as module
    from fluidmix.pipeline.service.manager import SessionManager

    arbitrary = tmp_path / "arbitrary.mp4"
    arbitrary.write_bytes(b"not-the-accepted-v4-development-file")
    initialized = []

    class MustNotInitialize:
        def __init__(self):
            initialized.append(True)
            raise AssertionError("heavy accepted factory must not initialize")

    class MustNotDecode:
        def frames(self, *_args):
            raise AssertionError("unavailable source must not be opened")

    monkeypatch.setattr(module, "ProductionAdapterFactory", MustNotInitialize)
    factory = module.DefaultProductionFactory()
    manager = SessionManager(
        adapter_factory=factory, frame_source=MustNotDecode(),
        pipeline_commit="a" * 40, accepted_pipeline_commit=module.PIPELINE_COMMIT,
        development_cohort="varied-v4",
    )
    sources = (
        {"kind": "file", "path": str(arbitrary), "label": "arbitrary.mp4"},
        {"kind": "v4l2", "device": "/dev/video0"},
        {"kind": "rtsp", "url": "rtsp://camera.example/live"},
    )
    for index, source in enumerate(sources):
        for recipe_id in (None, "5"):
            if source["kind"] == "file":
                arbitrary.write_bytes(b"not-the-accepted-v4-development-file")
            created = manager.create({
                "source": source,
                "run_context": {
                    "operator_run_id": f"run-{index}-{recipe_id}",
                    "recipe_id": recipe_id, "tray_id": "tray-1",
                },
                "device": "cpu",
            })
            worker = manager._get(created["session_id"]).worker
            assert worker is not None
            worker.join(timeout=2)
            decision = manager.decisions(created["session_id"])[0]
            assert decision["disposition"] == "insufficient_evidence"
            assert decision["limit_codes"] == ["tracker_source_profile_unavailable"]
    assert initialized == []


def test_only_hash_bound_accepted_file_reaches_static_development_factory(
    tmp_path, monkeypatch,
):
    import hashlib
    from fluidmix.pipeline.service import production_factory as module

    accepted = tmp_path / "accepted.mp4"
    accepted.write_bytes(b"accepted-v4-development-bytes")
    monkeypatch.setattr(module, "ACCEPTED_UPLOAD_SHA256", hashlib.sha256(
        accepted.read_bytes()
    ).hexdigest())
    seen = []

    class AcceptedFactory:
        def for_session(self, request, device):
            seen.append((request, device))
            return object()

    default = module.DefaultProductionFactory()
    default._factory = AcceptedFactory()
    request = {
        "source": {"kind": "file", "path": str(accepted), "label": "renamed.mp4"},
        "run_context": {"operator_run_id": "run-1", "recipe_id": "5", "tray_id": "tray-1"},
    }
    result = default.for_session(request, "cpu")
    assert result is not None
    assert seen == [(request, "cpu")]


def test_measurement_mode_is_pinned_only_for_exact_accepted_static_source(
    tmp_path, monkeypatch,
):
    import hashlib
    from fluidmix.pipeline.service import production_factory as module

    accepted = tmp_path / "accepted.mp4"
    accepted.write_bytes(b"accepted-v4-development-bytes")
    arbitrary = tmp_path / "arbitrary.mp4"
    arbitrary.write_bytes(b"arbitrary-video-bytes")
    monkeypatch.setattr(
        module, "ACCEPTED_UPLOAD_SHA256",
        hashlib.sha256(accepted.read_bytes()).hexdigest(),
    )

    assert module.measurement_mode_for_source({
        "kind": "file", "path": str(accepted),
    }) == "pinned_development_stage_stills"
    assert module.measurement_mode_for_source({
        "kind": "file", "path": str(arbitrary),
    }) == "generic_live"
    assert module.measurement_mode_for_source({
        "kind": "v4l2", "device": "/dev/video0",
    }) == "generic_live"


def test_pinned_stage_measurements_reproduce_accepted_deterministic_lineage():
    from fluidmix.pipeline.service import production_factory as module

    cohort = json.loads((module.VLM_MATERIALIZATION_ROOT / "cohort.json").read_bytes())
    receipt = module.pinned_deterministic_decision(
        cohort, cohort["deterministic_decision_receipt"],
    )

    assert receipt["source_artifact_sha256"] == (
        "87578e5cd70bd90284607f8afe2bcb0f661abf6193a58207dfc28406d61a5a8a"
    )
    assert receipt == cohort["deterministic_decision_receipt"]


def test_accepted_static_context_retains_twin_train_partition_and_frame_rate(
    tmp_path, monkeypatch,
):
    from fluidmix.pipeline.service import production_factory as module

    root = tmp_path / "materialization"
    root.mkdir()
    (root / "cohort.json").write_text("""{
      "schema_version": "wp7-vlm-cohort-v1",
      "cohort_id": "DEV-STATIC",
      "partition": "twin_train",
      "camera_metadata": {"frame_rate_hz": 12.5},
      "vials": []
    }""", encoding="utf-8")
    monkeypatch.setattr(module, "VLM_MATERIALIZATION_ROOT", root)

    context, vials, frame_rate_hz = module.load_accepted_development_context()

    assert context["partition"] == "twin_train"
    assert context["cohort_id"] == "DEV-STATIC"
    assert vials == []
    assert frame_rate_hz == 12.5

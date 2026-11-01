from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

RUN_CONTEXT = {
    "operator_run_id": "run-v3-development",
    "recipe_id": None,
    "tray_id": "tray-1",
}
PIN = "cf6829b99486b41f4190874dbd701ad812648217"


def _wait_terminal(manager, session_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = manager.state(session_id)
        if state["state"] in {"failed", "stopped"}:
            return state
        time.sleep(0.01)
    raise AssertionError("session did not finish")


class ThreeVialAdapter:
    model_id = "test-cnn"
    device = "cpu"

    def process_frame(self, frame, *, frame_index, elapsed_seconds):
        if frame_index >= 4:
            return ()
        vial = (1, 1, 2, 3)[frame_index]
        closed = frame_index != 0
        discard_score = 1.0 if frame_index == 1 else 0.0
        return ({
            "vial_id": f"vial{vial}",
            "vial_position": vial,
            "fill_step": 1,
            "tank_id": "t100",
            "tracker": {"agreement": True, "measured_volume_ml": 100.0},
            "mask": {"status": "visible_measured", "pixel_count": 12},
            "measurement": {
                "validity": "valid", "median_lab": [50.0 + vial, 1.0, 2.0],
                "evidence_id": f"EV-{vial}", "measured_or_inferred": "measured",
            },
            "deterministic": {
                "disposition": "discard" if discard_score else "keep",
                "discard_score": discard_score,
                "evidence_citations": [f"EV-{vial}"], "limit_codes": [],
            },
            "closed": closed,
        },)

    def build_vlm_bundle(self, vials):
        assert set(vials) == {"vial1", "vial2", "vial3"}
        return {"target_vial_id": "vial1", "reference_vial_ids": ["vial2", "vial3"]}

    def decide_cohort(self, vials):
        assert set(vials) == {"vial1", "vial2", "vial3"}
        return {
            "disposition": "discard", "discard_score": 1.0,
            "evidence_citations": ["EV-1", "EV-2", "EV-3"],
            "limit_codes": ["same_stage_reference_comparison"],
        }

    def run_vlm(self, bundle):
        return (
            {"source_id": "vlm-a", "disposition": "discard", "discard_score": 0.8,
             "disposition_confidence": 0.8, "guard_status": "admitted",
             "repository": "org/model-a", "revision": "rev-a",
             "explanation": "Target differs from both references.",
             "evidence_citations": ["EV-1", "EV-2"], "image_citations": ["IMG-1"]},
            {"source_id": "vlm-b", "disposition": "keep", "discard_score": 0.2,
             "disposition_confidence": 0.8, "guard_status": "admitted",
             "repository": "org/model-b", "revision": "rev-b",
             "explanation": "Target agrees with the references.",
             "evidence_citations": ["EV-1", "EV-3"], "image_citations": ["IMG-2"]},
        )


class MemoryFrames:
    def frames(self, source, stop):
        for index in range(4):
            if stop.is_set():
                return
            yield np.zeros((480, 640, 3), dtype=np.uint8), index / 10


def test_session_emits_causal_events_and_equal_weight_final_decision(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    manager = SessionManager(
        adapter_factory=lambda _context, _device: ThreeVialAdapter(),
        frame_source=MemoryFrames(),
        pipeline_commit=PIN, development_cohort="varied-v3",
    )
    session = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "v3-twin.mp4")},
        "run_context": RUN_CONTEXT,
        "device": "cpu",
    }, validate_video_path=False)
    state = _wait_terminal(manager, session["session_id"])
    events = manager.events(session["session_id"])

    assert state["state"] == "stopped"
    assert [row["sequence"] for row in events] == list(range(1, len(events) + 1))
    types = [row["type"] for row in events]
    assert types.index("measuring") < types.index("vial_decision")
    assert types.index("waiting_for_references") < types.index("model_vote")
    assert types.index("model_vote") < types.index("averaged_decision")
    final = next(row for row in events if row["type"] == "averaged_decision")
    assert final["payload"]["scores"]["discard"] == pytest.approx(2 / 3)
    assert final["payload"]["disposition"] == "discard"
    assert set(manager.vials(session["session_id"])) == {"vial1", "vial2", "vial3"}
    votes = [row["payload"] for row in events if row["type"] == "model_vote"]
    assert votes[0]["model"] == {
        "model_id": "org/model-a", "revision": "rev-a", "role": "vlm",
        "source": "inferred",
    }
    assert votes[1]["confidence"] == pytest.approx(0.8)
    assert votes[1]["evidence_ids"] == ["EV-1", "EV-3", "IMG-2"]
    explanations = [row["payload"] for row in events if row["type"] == "explanation"]
    assert {row["producer"]["model_id"] for row in explanations} >= {
        "org/model-a", "org/model-b", "deterministic-template",
    }


def test_deterministic_decision_is_causal_and_rate_health_is_measured(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class CausalAdapter(ThreeVialAdapter):
        def __init__(self):
            self.decide_calls = []

        def decide_cohort(self, vials):
            self.decide_calls.append(sorted(vials))
            assert {row["measurement"]["evidence_id"] for row in vials.values()} == {
                "EV-1", "EV-2", "EV-3",
            }
            return super().decide_cohort(vials)

    adapter = CausalAdapter()
    manager = SessionManager(
        adapter_factory=lambda *_args: adapter, frame_source=MemoryFrames(),
        pipeline_commit=PIN, development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "v3.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    _wait_terminal(manager, created["session_id"])
    assert adapter.decide_calls == [["vial1", "vial2", "vial3"]]
    rows = manager.events(created["session_id"])
    decisions = [row for row in rows if row["type"] == "vial_decision"]
    assert len(decisions) == 1
    assert decisions[0]["payload"]["evidence_ids"] == ["EV-1", "EV-2", "EV-3"]
    live_rates = [row["payload"] for row in rows if row["type"] == "rate_health"][:-1]
    assert live_rates
    assert all(row["processing_fps"] > 0 for row in live_rates)
    assert any(row["source_fps"] is not None and row["source_fps"] > 0
               for row in live_rates[1:])
    assert all(row["gpu_status"] == "not_required" for row in live_rates)


def test_source_validation_and_device_policy_fail_closed(tmp_path, monkeypatch):
    from fluidmix.pipeline.service.models import validate_create_request
    from fluidmix.pipeline.service.device import select_device

    with pytest.raises(ValueError, match="regular file"):
        validate_create_request({
            "source": {"kind": "video", "path": str(tmp_path / "missing.mp4")},
            "run_context": RUN_CONTEXT, "device": "cpu",
        })
    with pytest.raises(ValueError, match="/dev/video"):
        validate_create_request({
            "source": {"kind": "v4l2", "device": "/tmp/camera"},
            "run_context": RUN_CONTEXT, "device": "cpu",
        })
    with pytest.raises(ValueError, match="rtsp"):
        validate_create_request({
            "source": {"kind": "rtsp", "url": "https://example.test/feed"},
            "run_context": RUN_CONTEXT, "device": "cpu",
        })
    with pytest.raises(ValueError, match="fields are not closed"):
        validate_create_request({
            "source": {"kind": "rtsp", "url": "rtsp://camera/live", "path": "x"},
            "run_context": RUN_CONTEXT, "device": "cpu",
        })
    assert select_device("cpu").selected == "cpu"
    monkeypatch.setenv("FLUIDMIX_GPU_FREE", "0")
    assert select_device("gpu").selected == "cpu"
    assert select_device("gpu").reason == "gpu_not_explicitly_free"


def test_http_json_endpoints_and_sse_backlog(tmp_path):
    from fluidmix.pipeline.service.http import RunnerHTTPServer
    from fluidmix.pipeline.service.manager import SessionManager
    from urllib.request import Request, urlopen

    manager = SessionManager(
        adapter_factory=lambda _context, _device: ThreeVialAdapter(),
        frame_source=MemoryFrames(),
        pipeline_commit=PIN, development_cohort="varied-v3",
    )
    server = RunnerHTTPServer(("127.0.0.1", 0), manager)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        health = json.load(urlopen(f"{base}/api/health"))
        assert health["status"] == "healthy"
        body = json.dumps({
            "source": {"kind": "v4l2", "device": "/dev/video0"},
            "run_context": RUN_CONTEXT, "device": "cpu",
        }).encode()
        created = json.load(urlopen(Request(
            f"{base}/api/sessions", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )))
        _wait_terminal(manager, created["session_id"])
        state = json.load(urlopen(f"{base}/api/sessions/{created['session_id']}"))
        assert state["state"] == "stopped"
        response = urlopen(f"{base}/api/sessions/{created['session_id']}/events")
        assert response.headers["Content-Type"].startswith("text/event-stream")
        text = response.read().decode()
        assert "event: rate_health" in text
        assert "event: averaged_decision" in text
    finally:
        server.close()


def test_authorized_varied_v3_video_reaches_accepted_adapter_seam_without_mutation():
    video = Path(
        "/home/hackathon5/liamreporoot/varied-cohort-videos-20261027/individual/"
        "03-RUN-TRAIN-01-V3-KEEP-CASE_dd06f99eacbbbfee.mp4"
    )
    if not video.is_file():
        pytest.skip("authorized Varied V3 twin video is not present")
    from fluidmix.pipeline.service.manager import SessionManager
    from fluidmix.pipeline.service.sources import IncrementalFrameSource

    before = hashlib.sha256(video.read_bytes()).hexdigest()
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    class Tracker:
        def __init__(self):
            self.index = 0

        def observe(self, rgb, **_metadata):
            vial = (1, 1, 2, 3)[self.index]
            closed = self.index != 0
            self.index += 1
            return {
                "vial_id": f"vial{vial}", "vial_position": vial,
                "fill_step": 1, "tank_id": "t100",
                "evidence_id": f"EV-video-{self.index}", "closed": closed,
                "same_run_empty_reference_rgb": np.zeros_like(rgb),
                "agreement": True, "measured_volume_ml": 100.0,
            }

    def predictor(_model, current, _reference, *, device):
        mask = np.zeros(current.shape[:2], dtype=bool)
        height, width = mask.shape
        mask[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = True
        return SimpleNamespace(
            status="visible_measured", mask=mask,
            surface_row=height // 4, visible_height_px=height // 2,
        )

    def factory(_context, device):
        return AcceptedPipelineAdapter(
            tracker=Tracker(), cnn_model=object(),
            deterministic_decider=lambda vials: {
                "disposition": "keep", "discard_score": 0.0,
                "evidence_citations": [row["measurement"]["evidence_id"]
                                       for row in vials.values()],
                "limit_codes": [],
            },
            vlm_bundle_builder=lambda vials: {
                "target_vial_id": "vial1", "reference_vial_ids": ["vial2", "vial3"]
            },
            vlm_runner=lambda _bundle: (
                {"source_id": "external-vlm-double", "discard_score": 0.0,
                 "disposition": "keep", "guard_status": "admitted"},
            ),
            device=device, model_id="external-cnn-double", predictor=predictor,
        )

    manager = SessionManager(
        adapter_factory=factory,
        frame_source=IncrementalFrameSource(max_frames=4, pace_file=False),
        pipeline_commit=PIN, development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(video)},
        "run_context": RUN_CONTEXT, "device": "cpu",
    })
    state = _wait_terminal(manager, created["session_id"])
    assert state["state"] == "stopped"
    rows = manager.events(created["session_id"])
    assert sum(row["type"] == "rate_health" for row in rows) >= 4
    assert any(row["type"] == "measuring" for row in rows)
    assert hashlib.sha256(video.read_bytes()).hexdigest() == before


def test_accepted_adapter_calls_cnn_and_sha_bound_cielab_measurement():
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    class Tracker:
        def observe(self, rgb, **_metadata):
            return {
                "vial_id": "vial1", "vial_position": 1, "fill_step": 1,
                "tank_id": "t100", "evidence_id": "EV-live-1", "closed": True,
                "same_run_empty_reference_rgb": np.zeros_like(rgb),
                "agreement": True, "measured_volume_ml": 100.0,
            }

    observed = []

    def predictor(model, current, reference, *, device):
        observed.append((model, current.shape, reference.shape, device))
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:300, 200:400] = True
        return SimpleNamespace(
            status="visible_measured", mask=mask,
            surface_row=100, visible_height_px=200,
        )

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        deterministic_decider=lambda vials: {
            "disposition": "keep", "discard_score": 0.0,
            "evidence_citations": [row["measurement"]["evidence_id"]
                                   for row in vials.values()],
            "limit_codes": [],
        },
        vlm_bundle_builder=lambda vials: {"vials": sorted(vials)},
        vlm_runner=lambda _bundle: (), device="cpu", model_id="accepted-test",
        predictor=predictor,
    )
    try:
        frame = np.full((480, 640, 3), (120, 80, 40), dtype=np.uint8)
        result = adapter.process_frame(frame, frame_index=0, elapsed_seconds=0.0)[0]
    finally:
        adapter.close()
    assert observed[0][1:] == ((480, 640, 3), (480, 640, 3), "cpu")
    assert result["measurement"]["validity"] == "valid"
    assert len(result["measurement"]["median_lab"]) == 3
    assert result["measurement"]["measured_or_inferred"] == "measured"


def test_accepted_adapter_retains_private_hash_bound_evidence_for_frozen_builder():
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    context = {
        "schema_version": "wp7-vlm-cohort-v1", "cohort_id": "C1",
        "partition": "development", "recipe_id": "R1",
        "previous_recipe_id": "R0", "camera_metadata": {}, "stages": [],
    }

    class Tracker:
        def __init__(self):
            self.index = 0

        def observe(self, rgb, **_metadata):
            self.index += 1
            vial_id = f"V{self.index}"
            image_id = f"IMG-{vial_id}-S1-end"
            return {
                "vial_id": vial_id, "vial_position": self.index, "fill_step": 1,
                "tank_id": "t100", "evidence_id": f"EV-{vial_id}", "closed": True,
                "same_run_empty_reference_rgb": np.zeros_like(rgb), "agreement": True,
                "measured_volume_ml": 1.0, "image_id": image_id,
                "vlm_cohort_context": context,
                "vlm_vial": {"vial_position": self.index, "vial_id": vial_id,
                             "stages": []},
                "tracker_receipt": {"vial_id": vial_id, "receipt_sha256": f"{self.index:064x}"},
            }

    def predictor(*_args, **_kwargs):
        mask = np.ones((8, 8), dtype=bool)
        return SimpleNamespace(status="visible_measured", mask=mask,
                               surface_row=0, visible_height_px=8)

    builder_calls = []

    def frozen_builder(cohort, locators, *, tracker_receipts):
        builder_calls.append((cohort, locators, tracker_receipts))
        assert set(locators) == {"IMG-V1-S1-end", "IMG-V2-S1-end", "IMG-V3-S1-end"}
        for locator in locators.values():
            path = Path(locator["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == locator["sha256"]
        return {"prompt_payload": {"cohort_id": cohort["cohort_id"]}}

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(), predictor=predictor,
        deterministic_decider=lambda cohort: {
            "disposition": "keep",
            "stages": [{"evidence_id": "EV-V1"}, {"evidence_id": "EV-V2"},
                       {"evidence_id": "EV-V3"}],
            "limit_codes": [], "receipt_sha256": "d" * 64,
        },
        vlm_bundle_builder=frozen_builder, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="accepted-test",
    )
    try:
        rows = {}
        for index in range(3):
            result = adapter.process_frame(
                np.full((8, 8, 3), 20 + index, dtype=np.uint8),
                frame_index=index, elapsed_seconds=float(index),
            )[0]
            assert not ({"image_id", "vlm_cohort_context", "vlm_vial",
                         "tracker_receipt"} & set(result))
            rows[result["vial_id"]] = result
        decision = adapter.decide_cohort(rows)
        assert decision["discard_score"] == 0.0
        assert decision["evidence_citations"] == ["EV-V1", "EV-V2", "EV-V3"]
        bundle = adapter.build_vlm_bundle(rows)
        assert bundle["prompt_payload"]["cohort_id"] == "C1"
        cohort, _locators, receipts = builder_calls[0]
        assert [row["vial_id"] for row in cohort["vials"]] == ["V1", "V2", "V3"]
        assert [row["vial_id"] for row in receipts] == ["V1", "V2", "V3"]
        assert "path" not in json.dumps(cohort)
    finally:
        adapter.close()


def test_v4l2_uses_pyv_device_demuxer_argument(monkeypatch):
    from fluidmix.pipeline.service.sources import IncrementalFrameSource
    import threading

    observed = []

    class Container:
        streams = SimpleNamespace(video=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            return ()

    def open_source(target, **kwargs):
        observed.append((target, kwargs))
        return Container()

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_source))
    rows = IncrementalFrameSource().frames(
        {"kind": "v4l2", "device": "/dev/video0"}, threading.Event()
    )
    assert list(rows) == []
    assert observed == [("/dev/video0", {"format": "video4linux2"})]

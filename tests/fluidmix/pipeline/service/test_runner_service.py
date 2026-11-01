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
PIN = "9913b1a1cc5559cec9242072decabc3aaf2c4142"


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
             "runtime_telemetry": {"status": "completed", "elapsed_seconds": 1.25},
             "explanation": "Target differs from both references.",
             "model_explanation": "Target differs from both references.",
             "explanation_status": "model_written", "explanation_guard_reasons": [],
             "evidence_citations": ["EV-1", "EV-2"], "image_citations": ["IMG-1"]},
            {"source_id": "vlm-b", "disposition": "keep", "discard_score": 0.2,
             "disposition_confidence": 0.8, "guard_status": "admitted",
             "repository": "org/model-b", "revision": "rev-b",
             "explanation": "Deterministic fallback: structured keep disposition.",
             "model_explanation": "Residue caused the target to agree.",
             "explanation_status": "deterministic_fallback",
             "explanation_guard_reasons": ["cause_claim"],
             "evidence_citations": ["EV-1", "EV-3"], "image_citations": ["IMG-2"]},
            {"source_id": "vlm-c", "disposition": "insufficient_evidence",
             "discard_score": None, "disposition_confidence": None,
             "guard_status": "excluded", "guard_reasons": ["parse_failed"],
             "repository": "org/model-c", "revision": "rev-c",
             "runtime_telemetry": {
                 "status": "error", "error_type": "RuntimeError",
                 "error": "RuntimeError: model invocation failed",
             }},
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
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
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
    assert "explanation_status" not in votes[1]
    assert all("runtime_telemetry" not in row for row in votes)
    final_receipt = next(
        row for row in manager.decisions(session["session_id"])
        if row["type"] == "final"
    )
    assert final_receipt["sources"][1]["runtime_telemetry"] == {
        "status": "completed", "elapsed_seconds": 1.25,
    }
    assert final_receipt["valid_source_count"] == 3
    assert final_receipt["sources"][-1]["source_id"] == "vlm-c"
    assert final_receipt["sources"][-1]["runtime_telemetry"]["status"] == "error"
    explanations = [row["payload"] for row in events if row["type"] == "explanation"]
    assert {row["producer"]["model_id"] for row in explanations} >= {
        "org/model-a", "org/model-b", "deterministic-template",
    }
    fallback = next(row for row in explanations if row["explanation_id"].endswith("vlm-b"))
    assert fallback["producer"]["source"] == "deterministic_fallback"
    assert fallback["limits"] == ["cause_claim"]


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
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
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


def test_explicit_vial_boundaries_close_three_vials_before_cohort_decision(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class BoundaryFrames:
        def frames(self, _source, _stop):
            for index in range(3):
                yield np.zeros((480, 640, 3), dtype=np.uint8), index / 10

    class BoundaryAdapter(ThreeVialAdapter):
        def __init__(self):
            self.position = 0
            self.closed = []
            frame_sha = hashlib.sha256(
                np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
            ).hexdigest()
            self.vial_boundaries = tuple({
                "schema_version": "provenance-bound-vial-boundary-v1",
                "case_id": f"case{position}",
                "vial_id": f"vial{position}",
                "vial_position": position,
                "frame_index": position - 1,
                "frame_sha256": frame_sha,
                "source_sha256": f"{position}" * 64,
            } for position in range(1, 4))

        def process_frame(self, _frame, **_metadata):
            return ()

        def close_tracker_vial(self, boundary):
            self.position += 1
            assert boundary["vial_position"] == self.position
            self.closed.append(boundary["source_sha256"])
            row = dict(super().process_frame(None, frame_index=self.position, elapsed_seconds=0)[0])
            row.update({
                "vial_id": f"vial{self.position}", "vial_position": self.position,
                "closed": True,
            })
            return (row,)

    adapter = BoundaryAdapter()
    manager = SessionManager(
        adapter_factory=lambda *_args: adapter,
        frame_source=BoundaryFrames(), pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "bounded.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    _wait_terminal(manager, created["session_id"])
    rows = manager.events(created["session_id"])
    assert adapter.closed == ["1" * 64, "2" * 64, "3" * 64]
    assert sum(row["type"] == "vial_decision" for row in rows) == 1


def test_dynamic_complete_adapter_stops_before_consuming_extra_frames(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class Frames:
        def frames(self, _source, _stop):
            for index in range(5):
                yield np.zeros((480, 640, 3), dtype=np.uint8), index / 10

    class Adapter:
        requires_complete = True

        def __init__(self):
            self.calls = 0

        @property
        def session_complete(self):
            return self.calls == 3

        def process_frame(self, _frame, **_metadata):
            self.calls += 1
            return ()

    adapter = Adapter()
    manager = SessionManager(
        adapter_factory=lambda *_args: adapter, frame_source=Frames(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "dynamic.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    state = _wait_terminal(manager, created["session_id"])
    assert state["state"] == "stopped"
    assert adapter.calls == 3


def test_dynamic_incomplete_eof_abstains_with_tracker_reason(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class Adapter:
        requires_complete = True
        session_complete = False
        eof_abstention_reason = "tracker_stream_state_missing"

        def process_frame(self, _frame, **_metadata):
            return ()

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "short.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    state = _wait_terminal(manager, created["session_id"])
    assert state["state"] == "stopped"
    decision = next(row for row in manager.events(created["session_id"])
                    if row["type"] == "vial_decision")
    assert decision["payload"]["disposition"] == "insufficient_evidence"
    assert decision["payload"]["unavailable_reason"] == "tracker_stream_state_missing"


@pytest.mark.parametrize(("message", "reason"), [
    ("camera stream never reaches a recipe tank state", "tracker_stream_state_missing"),
    ("no completed stream fill stage", "tracker_stable_tail_missing"),
    ("camera stream states exceed recipe order", "tracker_stream_state_count_disagrees"),
    ("debounced camera stream states do not match recipe order", "tracker_stream_order_disagrees"),
    ("vial closed without complete tracker agreement", "tracker_incomplete_or_disagreed"),
    ("disagreed fill stage cannot enter a VLM receipt", "tracker_incomplete_or_disagreed"),
])
def test_tracker_profile_failures_map_to_explicit_abstention(message, reason, tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class Adapter:
        requires_complete = True
        session_complete = False

        def process_frame(self, _frame, **_metadata):
            raise ValueError(message)

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "unsupported.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    assert _wait_terminal(manager, created["session_id"])["state"] == "stopped"
    decision = next(row for row in manager.events(created["session_id"])
                    if row["type"] == "vial_decision")
    assert decision["payload"]["unavailable_reason"] == reason


@pytest.mark.parametrize(("message", "reason"), [
    (
        "stable volume window is missing",
        "tracker_stable_volume_window_missing",
    ),
    (
        "CNN fill height is outside frozen calibration domain",
        "tracker_height_outside_calibration",
    ),
    (
        "stream Lab tank classification is tied",
        "tracker_tank_classification_ambiguous",
    ),
])
def test_frozen_evidence_limits_stop_with_one_explicit_abstention(
    message, reason, tmp_path,
):
    from fluidmix.pipeline.service.manager import SessionManager
    from fluidmix.wp7_fill_step_tracker import FillStepTrackerError

    class Adapter:
        requires_complete = True
        session_complete = False

        def process_frame(self, _frame, **_metadata):
            raise FillStepTrackerError(message)

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "frozen-limit.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)

    assert _wait_terminal(manager, created["session_id"])["state"] == "stopped"
    assert manager.decisions(created["session_id"]) == [{
        "type": "abstention", "vial_id": "unavailable",
        "disposition": "insufficient_evidence",
        "limit_codes": [reason], "evidence_citations": [],
    }]
    events = manager.events(created["session_id"])
    public_decisions = [row for row in events if row["type"] == "vial_decision"]
    assert len(public_decisions) == 1
    assert public_decisions[0]["payload"]["disposition"] == "insufficient_evidence"
    assert public_decisions[0]["payload"]["unavailable_reason"] == reason
    assert not any(row["type"] == "error" for row in events)


@pytest.mark.parametrize("message", [
    "stable volume window is missing ",
    "frozen stable volume window drifted",
    "calibration artifact hash mismatch",
])
def test_frozen_evidence_mapping_remains_exact_and_corruption_fails(
    message, tmp_path,
):
    from fluidmix.pipeline.service.manager import SessionManager
    from fluidmix.wp7_fill_step_tracker import FillStepTrackerError

    class Adapter:
        def process_frame(self, _frame, **_metadata):
            raise FillStepTrackerError(message)

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "hard-failure.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)

    assert _wait_terminal(manager, created["session_id"])["state"] == "failed"
    assert manager.decisions(created["session_id"]) == []
    errors = [
        row for row in manager.events(created["session_id"])
        if row["type"] == "error"
    ]
    assert len(errors) == 1
    assert errors[0]["payload"]["code"] == "source_runtime_failed"


@pytest.mark.parametrize("source", [
    {"kind": "file", "path": "/tmp/profile.mp4", "label": "profile.mp4"},
    {"kind": "v4l2", "device": "/dev/video0"},
    {"kind": "rtsp", "url": "rtsp://camera.example/live"},
])
@pytest.mark.parametrize("recipe_id", [None, "6"])
def test_unsupported_recipe_profile_abstains_before_media(source, recipe_id):
    from fluidmix.pipeline.service.manager import SessionManager
    from fluidmix.pipeline.service.production_factory import validate_session_profile

    class Factory:
        def for_session(self, request, _device):
            validate_session_profile(request)
            raise AssertionError("unsupported recipe was admitted")

    class MustNotDecode:
        def frames(self, *_args):
            raise AssertionError("media must not open for unsupported recipe")

    manager = SessionManager(
        adapter_factory=Factory(), frame_source=MustNotDecode(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v4",
    )
    created = manager.create({
        "source": source,
        "run_context": {**RUN_CONTEXT, "recipe_id": recipe_id}, "device": "cpu",
    }, validate_video_path=False)
    assert _wait_terminal(manager, created["session_id"])["state"] == "stopped"
    decision = next(row for row in manager.events(created["session_id"])
                    if row["type"] == "vial_decision")
    assert decision["payload"]["unavailable_reason"] == "unsupported_recipe_profile"


@pytest.mark.parametrize("mutation", ["duplicate", "out_of_order", "frame_hash"])
def test_vial_boundary_plan_rejects_duplicate_order_and_frame_drift(tmp_path, mutation):
    from fluidmix.pipeline.service.manager import SessionManager

    frame_sha = hashlib.sha256(
        np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
    ).hexdigest()
    rows = [{
        "schema_version": "provenance-bound-vial-boundary-v1",
        "case_id": f"case{position}", "vial_id": f"vial{position}",
        "vial_position": position, "frame_index": position - 1,
        "frame_sha256": frame_sha, "source_sha256": f"{position}" * 64,
    } for position in range(1, 4)]
    if mutation == "duplicate":
        rows[1]["frame_index"] = rows[0]["frame_index"]
    elif mutation == "out_of_order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["frame_sha256"] = "f" * 64

    adapter = ThreeVialAdapter()
    adapter.vial_boundaries = tuple(rows)
    manager = SessionManager(
        adapter_factory=lambda *_args: adapter, frame_source=MemoryFrames(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "bad.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    state = _wait_terminal(manager, created["session_id"])
    assert state["state"] == "failed"


def test_unbounded_generic_stream_never_closes_or_decides_cohort(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class OpenAdapter(ThreeVialAdapter):
        close_calls = 0

        def process_frame(self, _frame, **_metadata):
            return ()

        def close_tracker_vial(self, _boundary):
            self.close_calls += 1
            return ()

    adapter = OpenAdapter()
    manager = SessionManager(
        adapter_factory=lambda *_args: adapter,
        frame_source=MemoryFrames(), pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v3",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "generic.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)
    _wait_terminal(manager, created["session_id"])
    assert adapter.close_calls == 0
    assert not any(
        row["type"] == "vial_decision"
        for row in manager.events(created["session_id"])
    )


def test_tracker_disagreement_has_explicit_unavailable_reason():
    from fluidmix.pipeline.service.manager import SessionManager

    payload = SessionManager._tracker_payload({
        "vial_id": "vial1", "vial_position": 1, "fill_step": 1,
        "tank_id": "t100",
        "tracker": {"agreement": False, "measured_volume_ml": 100.0},
    })
    assert payload["state"] == "unavailable"
    assert payload["unavailable_reason"] == "tracker_agreement_unavailable"


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
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
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
        pipeline_commit=PIN, accepted_pipeline_commit=PIN, development_cohort="varied-v3",
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
    measure_calls = []

    def predictor(model, current, reference, *, device):
        observed.append((model, current.shape, reference.shape, device))
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:300, 200:400] = True
        return SimpleNamespace(
            status="visible_measured", mask=mask,
            surface_row=100, visible_height_px=200,
        )

    def measurer(image_id, locator, **_kwargs):
        measure_calls.append((image_id, locator))
        return {
            "validity": "valid", "median_lab": [50.0, 2.0, 3.0],
            "mad_lab": [1.0, 1.0, 1.0], "iqr_lab": [2.0, 2.0, 2.0],
            "limit_codes": [],
        }

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
        predictor=predictor, measurer=measurer,
    )
    try:
        frame = np.full((480, 640, 3), (120, 80, 40), dtype=np.uint8)
        result = adapter.process_frame(frame, frame_index=0, elapsed_seconds=0.0)[0]
    finally:
        adapter.close()
    assert observed[0][1:] == ((480, 640, 3), (480, 640, 3), "cpu")
    assert len(measure_calls) == 1
    assert result["measurement"]["validity"] == "valid"
    assert len(result["measurement"]["median_lab"]) == 3
    assert result["measurement"]["measured_or_inferred"] == "measured"


def test_accepted_adapter_retains_private_hash_bound_evidence_for_frozen_builder():
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter

    context = {
        "schema_version": "wp7-vlm-cohort-v1", "cohort_id": "C1",
        "partition": "development", "recipe_id": "R1",
        "previous_recipe_id": "R0",
        "camera_metadata": {"resolution": [8, 8], "vial_roi": [1, 1, 7, 7]},
        "stages": [],
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
        streams = SimpleNamespace(video=[SimpleNamespace(
            average_rate=10, guessed_rate=None, base_rate=None, width=640, height=480,
        )])

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


def test_live_source_fps_falls_back_to_frame_pts_delta(monkeypatch):
    from fluidmix.pipeline.service.sources import IncrementalFrameSource
    import threading

    frames = [SimpleNamespace(
        time=value,
        to_ndarray=lambda format, value=value: np.full((480, 640, 3), int(value * 10),
                                                       dtype=np.uint8),
    ) for value in (4.0, 4.04, 4.08)]

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(
            average_rate=None, guessed_rate=None, base_rate=None, width=640, height=480,
        )])
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def decode(self, _stream): return iter(frames)

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *_a, **_k: Container()))
    stop = threading.Event()
    source = IncrementalFrameSource(pace_file=False)
    iterator = source.frames({"kind": "rtsp", "url": "rtsp://camera/live"}, stop)
    first = next(iterator)
    assert first[1] == 4.0
    assert source.profile(stop) == {
        "fps": pytest.approx(25.0), "resolution": [640, 480],
        "fps_source": "decoded_frame_pts_delta",
    }
    iterator.close()


def test_profile_and_measurement_limits_map_to_honest_abstentions():
    from fluidmix.pipeline.service.manager import SessionManager

    assert SessionManager._pipeline_failure_code(
        ValueError("camera view is incompatible with frozen FillCam geometry")
    ) == "camera_view_unvalidated"
    assert SessionManager._pipeline_failure_code(
        RuntimeError("source stream FPS is unavailable")
    ) == "source_frame_rate_unavailable"
    assert SessionManager._pipeline_failure_code(
        RuntimeError("source stream profile is unavailable")
    ) == "camera_view_unvalidated"
    assert SessionManager._pipeline_failure_code(
        RuntimeError("measured VLM stage evidence is unavailable")
    ) == "measurement_stage_unavailable"


def test_invalid_source_geometry_is_a_stable_runner_abstention(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class InvalidGeometrySource:
        def frames(self, _source, _stop):
            raise RuntimeError("source stream profile is invalid")
            yield

    class Adapter:
        def process_frame(self, _frame, **_metadata):
            return ()

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(),
        frame_source=InvalidGeometrySource(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "bad-geometry.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)

    assert _wait_terminal(manager, created["session_id"])["state"] == "stopped"
    decisions = manager.decisions(created["session_id"])
    assert decisions == [{
        "type": "abstention", "vial_id": "unavailable",
        "disposition": "insufficient_evidence",
        "limit_codes": ["camera_view_unvalidated"], "evidence_citations": [],
    }]
    events = manager.events(created["session_id"])
    assert not any(row["type"] == "error" for row in events)
    assert next(row for row in events if row["type"] == "vial_decision")["payload"][
        "unavailable_reason"
    ] == "camera_view_unvalidated"


def test_unavailable_source_profile_is_a_camera_view_abstention(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class MissingProfileSource:
        def frames(self, _source, _stop):
            yield np.zeros((480, 640, 3), dtype=np.uint8), 0.0

        def profile(self, _stop):
            return None

    class Adapter:
        def configure_source_profile(self, _profile):
            raise AssertionError("missing profile must not reach the adapter")

        def process_frame(self, _frame, **_metadata):
            return ()

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(), frame_source=MissingProfileSource(),
        pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "no-profile.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)

    assert _wait_terminal(manager, created["session_id"])["state"] == "stopped"
    assert manager.decisions(created["session_id"])[0]["limit_codes"] == [
        "camera_view_unvalidated"
    ]
    assert not any(
        row["type"] == "error" for row in manager.events(created["session_id"])
    )


def test_incompatible_fillcam_profile_is_a_stable_runner_abstention(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager
    from fluidmix.pipeline.service.production_factory import SessionVialTracker

    class IncompatibleViewSource:
        def frames(self, _source, _stop):
            yield np.zeros((480, 800, 3), dtype=np.uint8), 0.0

        def profile(self, _stop):
            return {"fps": 10.0, "resolution": [800, 480]}

    class Adapter:
        def __init__(self):
            self.tracker = SessionVialTracker(lambda *_args: object())

        def configure_source_profile(self, profile):
            self.tracker.configure_source_profile(profile)

        def process_frame(self, frame, **metadata):
            return self.tracker.observe(frame, **metadata)

    manager = SessionManager(
        adapter_factory=lambda *_args: Adapter(),
        frame_source=IncompatibleViewSource(), pipeline_commit=PIN,
        accepted_pipeline_commit=PIN, development_cohort="varied-v4",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "wrong-view.mp4")},
        "run_context": RUN_CONTEXT, "device": "cpu",
    }, validate_video_path=False)

    assert _wait_terminal(manager, created["session_id"])["state"] == "stopped"
    assert manager.decisions(created["session_id"])[0]["limit_codes"] == [
        "camera_view_unvalidated"
    ]


def test_session_aware_factory_receives_validated_request_and_legacy_factory_still_works(
    tmp_path,
):
    from fluidmix.pipeline.service.manager import SessionManager

    video = tmp_path / "run.mp4"
    video.write_bytes(b"video")
    request = {
        "source": {"kind": "video", "path": str(video)},
        "run_context": {
            "operator_run_id": "run-1", "recipe_id": "recipe-1", "tray_id": "tray-1",
        },
        "device": "cpu",
    }
    seen = []

    class Adapter:
        def process_frame(self, *_args, **_kwargs):
            return ()

    class SessionAwareFactory:
        def __call__(self, *_args):
            raise AssertionError("legacy hook must not be used")

        def for_session(self, checked_request, device):
            seen.append((checked_request, device))
            return Adapter()

    class Source:
        def frames(self, *_args):
            return iter(())

    manager = SessionManager(
        adapter_factory=SessionAwareFactory(), frame_source=Source(),
        pipeline_commit="9913b1a1cc5559cec9242072decabc3aaf2c4142",
        accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    session_id = manager.create(request)["session_id"]
    manager.shutdown(timeout=2)
    assert seen == [(manager._get(session_id).request, "cpu")]

    legacy_seen = []
    legacy = SessionManager(
        adapter_factory=lambda context, device: (
            legacy_seen.append((context, device)) or Adapter()
        ),
        frame_source=Source(),
        pipeline_commit="9913b1a1cc5559cec9242072decabc3aaf2c4142",
        accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    legacy.create(request)
    legacy.shutdown(timeout=2)
    assert legacy_seen == [(request["run_context"], "cpu")]

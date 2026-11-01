from __future__ import annotations

import threading
import time

import numpy as np
import pytest


CONTEXT = {
    "operator_run_id": "run-v3-development", "recipe_id": None,
    "tray_id": "tray-1",
}
PIN = "9913b1a1cc5559cec9242072decabc3aaf2c4142"


def _manager(**kwargs):
    from fluidmix.pipeline.service.manager import SessionManager
    return SessionManager(
        pipeline_commit=PIN, accepted_pipeline_commit=PIN,
        development_cohort="varied-v3", **kwargs
    )


class NoopAdapter:
    model_id = "none"

    def process_frame(self, *_args, **_kwargs):
        return ()

    def close(self):
        return None


class EmptyFrames:
    def frames(self, _source, _stop):
        return iter(())


def _request(source):
    return {"source": source, "run_context": CONTEXT, "device": "cpu"}


def _terminal(manager, session_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = manager.state(session_id)
        if state["state"] in {"failed", "stopped"}:
            return state
        time.sleep(0.005)
    raise AssertionError("session did not terminate")


def test_rtsp_exception_never_discloses_credentials():
    from fluidmix.pipeline.service.manager import SessionManager

    secret = "alice:secret"

    class FailingFrames:
        def frames(self, source, _stop):
            raise RuntimeError(f"could not open {source['url']}")
            yield

    manager = _manager(
        adapter_factory=lambda *_args: NoopAdapter(), frame_source=FailingFrames()
    )
    created = manager.create(_request({
        "kind": "rtsp", "url": f"rtsp://{secret}@camera/live?token=also-secret"
    }))
    state = _terminal(manager, created["session_id"])
    serialized = repr(state) + repr(manager.events(created["session_id"]))
    assert secret not in serialized
    assert "also-secret" not in serialized
    assert "source_runtime_failed" in serialized


def test_cleanup_failure_publishes_only_one_atomic_terminal_failure(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class CloseFails(NoopAdapter):
        def close(self):
            raise RuntimeError("cleanup failed")

    manager = _manager(
        adapter_factory=lambda *_args: CloseFails(), frame_source=EmptyFrames()
    )
    created = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3.mp4")}), validate_video_path=False)
    state = _terminal(manager, created["session_id"])
    events = manager.events(created["session_id"])
    terminal = [row for row in events if row["type"] in {"error", "rate_health"}]
    assert state["state"] == "failed"
    assert [row["type"] for row in terminal] == ["error"]
    assert manager.terminal_sequence(created["session_id"]) == terminal[0]["sequence"]


def test_completed_status_and_terminal_event_are_one_observable_transition(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    manager = _manager(
        adapter_factory=lambda *_args: NoopAdapter(), frame_source=EmptyFrames()
    )
    created = manager.create(_request({
        "kind": "video", "path": str(tmp_path / "v3.mp4")
    }), validate_video_path=False)
    state = _terminal(manager, created["session_id"])
    rows = manager.events(created["session_id"], after=0)
    assert manager.terminal_sequence(created["session_id"]) == rows[-1]["sequence"]
    assert rows[-1]["type"] == "rate_health"


def test_active_session_history_exceeds_legacy_bound_and_sessions_are_bounded(tmp_path):

    class ManyFrames:
        def frames(self, _source, _stop):
            for index in range(8):
                yield np.zeros((2, 2, 3), dtype=np.uint8), float(index)

    manager = _manager(
        adapter_factory=lambda *_args: NoopAdapter(), frame_source=ManyFrames(),
        event_retention=4, session_retention=2,
    )
    ids = []
    for index in range(3):
        created = manager.create(_request({
            "kind": "video", "path": str(tmp_path / f"v3-{index}.mp4")
        }), validate_video_path=False)
        _terminal(manager, created["session_id"])
        ids.append(created["session_id"])
    with pytest.raises(KeyError, match="unknown session"):
        manager.state(ids[0])
    retained = manager.events(ids[-1], after=0)
    assert len(retained) == 9
    assert [row["sequence"] for row in retained] == list(range(1, 10))
    assert retained[-1]["type"] == "rate_health"


def test_stop_and_shutdown_cancel_and_join_blocked_source(tmp_path):
    from fluidmix.pipeline.service.manager import SessionManager

    class BlockingFrames:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()

        def frames(self, _source, _stop):
            self.entered.set()
            self.release.wait()
            return iter(())

        def cancel(self, _stop):
            self.release.set()

    frames = BlockingFrames()
    manager = _manager(
        adapter_factory=lambda *_args: NoopAdapter(), frame_source=frames
    )
    created = manager.create(_request({
        "kind": "video", "path": str(tmp_path / "v3.mp4")
    }), validate_video_path=False)
    assert frames.entered.wait(1)
    manager.stop(created["session_id"])
    manager.shutdown(timeout=1)
    state = _terminal(manager, created["session_id"])
    assert state["state"] == "stopped"
    assert manager.worker_alive(created["session_id"]) is False


@pytest.mark.parametrize("context", [
    {"operator_run_id": "x", "recipe_id": None, "tray_id": "t", "label": "keep"},
    {"operator_run_id": "x", "recipe_id": None, "tray_id": "t", "truth": True}
])
def test_run_context_rejects_out_of_scope_and_oracle_fields(tmp_path, context):
    from fluidmix.pipeline.service.models import validate_create_request

    with pytest.raises(ValueError):
        validate_create_request({
            "source": {"kind": "video", "path": str(tmp_path / "v3.mp4")},
            "run_context": context, "device": "cpu",
        }, validate_video_path=False)


def test_adapter_tracker_event_uses_allowlist_not_truth_fields():
    from fluidmix.pipeline.service.accepted_adapter import AcceptedPipelineAdapter
    from types import SimpleNamespace

    class Tracker:
        def observe(self, rgb, **_metadata):
            return {
                "vial_id": "vial1", "vial_position": 1, "fill_step": 1,
                "tank_id": "t100", "evidence_id": "EV-1", "closed": False,
                "same_run_empty_reference_rgb": np.zeros_like(rgb),
                "agreement": True, "measured_volume_ml": 100.0,
                "label": "discard", "oracle": {"truth": True},
            }

    def predictor(*_args, **_kwargs):
        mask = np.zeros((480, 640), dtype=bool)
        return SimpleNamespace(status="empty_mask_abstain", mask=mask,
                               surface_row=None, visible_height_px=None)

    adapter = AcceptedPipelineAdapter(
        tracker=Tracker(), cnn_model=object(),
        deterministic_decider=lambda _vials: {
            "disposition": "insufficient_evidence", "discard_score": None,
            "evidence_citations": [], "limit_codes": ["mask_empty"],
        },
        vlm_bundle_builder=lambda _vials: {}, vlm_runner=lambda _bundle: (),
        device="cpu", model_id="test", predictor=predictor,
    )
    try:
        result = adapter.process_frame(
            np.ones((480, 640, 3), dtype=np.uint8), frame_index=0,
            elapsed_seconds=0.0,
        )[0]
    finally:
        adapter.close()
    assert set(result["tracker"]) == {"agreement", "measured_volume_ml"}


def test_two_cohorts_do_not_mix_closed_vials_and_both_finalize(tmp_path):
    class TwoCohortAdapter(NoopAdapter):
        def process_frame(self, _frame, *, frame_index, **_kwargs):
            cohort = frame_index // 3 + 1
            position = frame_index % 3 + 1
            evidence = f"EV-C{cohort}-V{position}"
            return ({
                "vial_id": f"C{cohort}-V{position}", "vial_position": position,
                "fill_step": 1, "tank_id": "t100",
                "tracker": {"agreement": True, "measured_volume_ml": 100.0},
                "mask": {"status": "visible_measured", "pixel_count": 1},
                "measurement": {"validity": "valid", "median_lab": [50, 1, 2],
                                "evidence_id": evidence},
                "deterministic": {"disposition": "keep", "discard_score": 0.0,
                                  "evidence_citations": [evidence], "limit_codes": []},
                "closed": True,
            },)

        def build_vlm_bundle(self, vials):
            cohort_ids = {key.split("-")[0] for key in vials}
            assert len(cohort_ids) == 1
            return {"cohort": next(iter(cohort_ids))}

        def decide_cohort(self, vials):
            evidence = sorted(row["measurement"]["evidence_id"] for row in vials.values())
            return {"disposition": "keep", "discard_score": 0.0,
                    "evidence_citations": evidence, "limit_codes": []}

        def run_vlm(self, bundle):
            return ({"source_id": f"vlm-{bundle['cohort']}", "disposition": "keep",
                     "discard_score": 0.0, "guard_status": "admitted"},)

    class SixFrames:
        def frames(self, _source, _stop):
            for index in range(6):
                yield np.zeros((2, 2, 3), dtype=np.uint8), float(index)

    manager = _manager(adapter_factory=lambda *_args: TwoCohortAdapter(),
                       frame_source=SixFrames())
    created = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3.mp4")}),
                             validate_video_path=False)
    _terminal(manager, created["session_id"])
    finals = [row for row in manager.events(created["session_id"])
              if row["type"] == "averaged_decision"]
    assert len(finals) == 2
    assert [row["payload"]["position"]["vial_number"] for row in finals] == [
        "C1-V1", "C2-V1",
    ]
    assert set(manager.vials(created["session_id"])) == {"C2-V1", "C2-V2", "C2-V3"}


@pytest.mark.parametrize("attempt", [
    {"source_id": "ie", "disposition": "insufficient_evidence", "discard_score": 0.4,
     "guard_status": "admitted"},
    {"source_id": "bad-disposition", "disposition": "maybe", "discard_score": 0.4,
     "guard_status": "admitted"},
    {"source_id": "guarded", "disposition": "discard", "discard_score": 0.9,
     "guard_status": "excluded"},
    {"source_id": "range", "disposition": "discard", "discard_score": 1.1,
     "guard_status": "admitted"},
    {"source_id": "nan", "disposition": "discard", "discard_score": float("nan"),
     "guard_status": "admitted"},
])
def test_invalid_or_unadmitted_vlm_attempt_is_schema_valid_and_excluded(tmp_path, attempt):
    from test_runner_service import MemoryFrames, ThreeVialAdapter

    class Adapter(ThreeVialAdapter):
        def run_vlm(self, _bundle):
            return (attempt,)

    manager = _manager(adapter_factory=lambda *_args: Adapter(), frame_source=MemoryFrames())
    created = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3.mp4")}),
                             validate_video_path=False)
    _terminal(manager, created["session_id"])
    rows = manager.events(created["session_id"])
    vote = next(row for row in rows if row["type"] == "model_vote")
    assert vote["payload"]["admission"]["admitted"] is False
    assert vote["payload"]["disposition"] == "insufficient_evidence"
    assert vote["payload"]["confidence"] is None
    final = next(row for row in rows if row["type"] == "averaged_decision")
    assert final["payload"]["disposition"] == "insufficient_evidence"


def test_source_exception_after_stop_finishes_stopped_not_failed(tmp_path):
    class CancellingFrames:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()

        def frames(self, _source, _stop):
            self.entered.set()
            self.release.wait()
            raise RuntimeError("container closed during decode")
            yield

        def cancel(self, _stop):
            self.release.set()

    frames = CancellingFrames()
    manager = _manager(adapter_factory=lambda *_args: NoopAdapter(), frame_source=frames)
    created = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3.mp4")}),
                             validate_video_path=False)
    assert frames.entered.wait(1)
    manager.stop(created["session_id"])
    state = _terminal(manager, created["session_id"])
    assert state["state"] == "stopped"
    assert manager.events(created["session_id"])[-1]["type"] == "rate_health"


def test_queued_session_stop_does_not_wait_for_active_model_session(tmp_path):
    class BlockingAdapter(NoopAdapter):
        def __init__(self, entered, release):
            self.entered, self.release = entered, release

        def process_frame(self, *_args, **_kwargs):
            self.entered.set()
            self.release.wait()
            return ()

    class TwoFrames:
        def frames(self, _source, _stop):
            yield np.zeros((2, 2, 3), dtype=np.uint8), 0.0
            yield np.zeros((2, 2, 3), dtype=np.uint8), 0.1

    entered, release = threading.Event(), threading.Event()
    calls = 0

    def factory(*_args):
        nonlocal calls
        calls += 1
        return BlockingAdapter(entered, release) if calls == 1 else NoopAdapter()

    manager = _manager(adapter_factory=factory, frame_source=TwoFrames())
    first = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3-a.mp4")}),
                           validate_video_path=False)
    assert entered.wait(1)
    second = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3-b.mp4")}),
                            validate_video_path=False)
    started = time.monotonic()
    manager.stop(second["session_id"])
    second_state = _terminal(manager, second["session_id"])
    assert time.monotonic() - started < 1
    assert second_state["state"] == "stopped"
    assert calls == 1
    release.set()
    _terminal(manager, first["session_id"])


def test_insufficient_votes_publish_neutral_unavailable_scores(tmp_path):
    from test_runner_service import MemoryFrames, ThreeVialAdapter

    class Adapter(ThreeVialAdapter):
        def run_vlm(self, _bundle):
            return ()

        def decide_cohort(self, _vials):
            return {
                "disposition": "insufficient_evidence", "discard_score": None,
                "evidence_citations": ["EV-1"], "limit_codes": ["missing_reference"],
            }

    manager = _manager(adapter_factory=lambda *_args: Adapter(), frame_source=MemoryFrames())
    created = manager.create(_request({"kind": "video", "path": str(tmp_path / "v3.mp4")}),
                             validate_video_path=False)
    _terminal(manager, created["session_id"])
    final = next(row for row in manager.events(created["session_id"])
                 if row["type"] == "averaged_decision")
    assert final["payload"]["disposition"] == "insufficient_evidence"
    assert final["payload"]["scores"] == {"keep": 0.0, "check": 1.0, "discard": 0.0}
    assert final["payload"]["unavailable_reason"] == "fewer_than_two_valid_votes"


def test_gpu_fallback_rate_health_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("FLUIDMIX_GPU_FREE", raising=False)

    class OneFrame:
        def frames(self, _source, _stop):
            yield np.zeros((2, 2, 3), dtype=np.uint8), 0.0

    manager = _manager(adapter_factory=lambda *_args: NoopAdapter(), frame_source=OneFrame())
    request = _request({"kind": "video", "path": str(tmp_path / "v3.mp4")})
    request["device"] = "gpu"
    created = manager.create(request, validate_video_path=False)
    _terminal(manager, created["session_id"])
    live = manager.events(created["session_id"])[0]["payload"]
    assert live["gpu_status"] == "unavailable"
    assert live["limits"] == ["gpu_not_explicitly_free"]


def test_rate_health_reports_waiting_model_queue_depth(tmp_path):
    gate = threading.Event()

    class TwoFrames:
        def frames(self, _source, _stop):
            yield np.zeros((2, 2, 3), dtype=np.uint8), 0.0
            gate.wait()
            yield np.zeros((2, 2, 3), dtype=np.uint8), 0.1

    manager = _manager(adapter_factory=lambda *_args: NoopAdapter(), frame_source=TwoFrames())
    first = manager.create(_request({"kind": "video", "path": str(tmp_path / "a.mp4")}),
                           validate_video_path=False)
    deadline = time.monotonic() + 1
    while not manager.events(first["session_id"]):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    second = manager.create(_request({"kind": "video", "path": str(tmp_path / "b.mp4")}),
                            validate_video_path=False)
    gate.set()
    _terminal(manager, first["session_id"])
    _terminal(manager, second["session_id"])
    rates = [row["payload"] for row in manager.events(first["session_id"])
             if row["type"] == "rate_health"]
    assert any(row["queue_depth"] == 1 for row in rates)

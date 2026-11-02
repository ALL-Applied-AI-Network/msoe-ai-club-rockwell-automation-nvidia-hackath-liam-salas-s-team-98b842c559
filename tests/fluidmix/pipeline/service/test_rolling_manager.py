from __future__ import annotations

import time

import numpy as np
import pytest


PIN = "9913b1a1cc5559cec9242072decabc3aaf2c4142"
RUN_CONTEXT = {
    "operator_run_id": "rolling-run",
    "recipe_id": "5",
    "tray_id": "tray-1",
}


def _wait_terminal(manager, session_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = manager.state(session_id)
        if state["state"] in {"failed", "stopped"}:
            return state
        time.sleep(0.01)
    raise AssertionError("session did not finish")


class Frames:
    def __init__(self, values, *, fps=4.0):
        self.values = tuple(values)
        self.fps = fps

    def frames(self, _source, stop):
        for index, value in enumerate(self.values):
            if stop.is_set():
                return
            yield np.full((480, 640, 3), value, dtype=np.uint8), index / self.fps

    def profile(self, _stop):
        return {"fps": self.fps, "resolution": [640, 480]}


class CompleteAfterTwo:
    requires_complete = True
    eof_abstention_reason = "tracker_stream_state_missing"

    def __init__(self):
        self.values = []
        self.closed = False
        self.profiles = []

    @property
    def session_complete(self):
        return len(self.values) == 2

    def configure_source_profile(self, profile):
        self.profiles.append(dict(profile))

    def process_frame(self, frame, **_metadata):
        self.values.append(int(frame[0, 0, 0]))
        return ()

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, adapter_type=CompleteAfterTwo):
        self.adapter_type = adapter_type
        self.requests = []
        self.adapters = []

    def for_session(self, request, _device):
        self.requests.append(request)
        adapter = self.adapter_type()
        self.adapters.append(adapter)
        return adapter


def _run(tmp_path, monkeypatch, factory, frames, *, rolling=True):
    from fluidmix.pipeline.service.manager import SessionManager

    if rolling:
        monkeypatch.setenv("FLUIDMIX_ROLLING_COHORTS", "1")
    else:
        monkeypatch.delenv("FLUIDMIX_ROLLING_COHORTS", raising=False)
    manager = SessionManager(
        adapter_factory=factory,
        frame_source=frames,
        pipeline_commit=PIN,
        accepted_pipeline_commit=PIN,
        development_cohort="varied-v4",
    )
    created = manager.create({
        "source": {"kind": "video", "path": str(tmp_path / "rolling.mp4")},
        "run_context": dict(RUN_CONTEXT),
        "device": "cpu",
    }, validate_video_path=False)
    state = _wait_terminal(manager, created["session_id"])
    return manager, created["session_id"], state


def test_rolling_session_reuses_seam_as_reference_for_two_cohorts(
    tmp_path, monkeypatch,
):
    factory = Factory()

    manager, session_id, state = _run(
        tmp_path, monkeypatch, factory, Frames([10, 20, 30]),
    )

    assert state["state"] == "stopped"
    assert [adapter.values for adapter in factory.adapters] == [[10, 20], [20, 30]]
    assert [request["_rolling_cohort_ordinal"] for request in factory.requests] == [1, 2]
    assert all(adapter.closed for adapter in factory.adapters)
    assert manager.state(session_id)["run_context"] == RUN_CONTEXT
    assert all("_rolling_cohort_ordinal" not in request["run_context"]
               for request in factory.requests)


def test_disabled_rolling_retains_single_cohort_break(tmp_path, monkeypatch):
    factory = Factory()

    _manager, _session_id, state = _run(
        tmp_path, monkeypatch, factory, Frames([10, 20, 30]), rolling=False,
    )

    assert state["state"] == "stopped"
    assert len(factory.adapters) == 1
    assert factory.adapters[0].values == [10, 20]
    assert "_rolling_cohort_ordinal" not in factory.requests[0]


@pytest.mark.parametrize(("message", "expected_reason"), [
    ("camera stream states exceed recipe order", "unsupported_recipe_profile"),
    ("debounced camera stream states do not match recipe order",
     "unsupported_recipe_profile"),
    ("frozen profile supports only recipe 5", "unsupported_recipe_profile"),
    ("stable volume window is missing", "tracker_stable_volume_window_missing"),
])
def test_mapped_failure_abstains_resyncs_after_motion_and_stillness(
    message, expected_reason, tmp_path, monkeypatch,
):
    class FailingAdapter(CompleteAfterTwo):
        def process_frame(self, frame, **metadata):
            raise ValueError(message)

    factory = Factory()
    factory.adapter_type = FailingAdapter

    original_for_session = factory.for_session

    def for_session(request, device):
        if len(factory.adapters) == 1:
            factory.adapter_type = CompleteAfterTwo
        return original_for_session(request, device)

    factory.for_session = for_session
    manager, session_id, state = _run(
        tmp_path, monkeypatch, factory, Frames([0, 0, 100, 100, 100, 200]),
    )

    assert state["state"] == "stopped"
    decisions = manager.decisions(session_id)
    assert decisions == [{
        "type": "abstention", "vial_id": "cohort-1:unavailable",
        "disposition": "insufficient_evidence",
        "limit_codes": [expected_reason], "evidence_citations": [],
        "cohort_ordinal": 1,
    }]
    assert factory.adapters[0].closed is True
    assert factory.adapters[1].values == [100, 200]
    assert [request["_rolling_cohort_ordinal"] for request in factory.requests] == [1, 2]


def test_eof_during_resync_does_not_add_or_create_empty_cohort(
    tmp_path, monkeypatch,
):
    class FailingAdapter(CompleteAfterTwo):
        def process_frame(self, frame, **metadata):
            raise ValueError("camera stream states exceed recipe order")

    factory = Factory(FailingAdapter)
    manager, session_id, state = _run(
        tmp_path, monkeypatch, factory, Frames([0, 0, 100]),
    )

    assert state["state"] == "stopped"
    assert len(factory.adapters) == 1
    assert factory.adapters[0].closed is True
    assert len(manager.decisions(session_id)) == 1
    assert manager.decisions(session_id)[0]["limit_codes"] == [
        "unsupported_recipe_profile"
    ]


def test_repeated_rolling_abstentions_have_cohort_specific_identity(
    tmp_path, monkeypatch,
):
    class FailingAdapter(CompleteAfterTwo):
        def process_frame(self, frame, **metadata):
            raise ValueError("stable volume window is missing")

    manager, session_id, state = _run(
        tmp_path, monkeypatch, Factory(FailingAdapter),
        Frames([0, 0, 100, 100, 100, 200]),
    )

    assert state["state"] == "stopped"
    decision_events = [
        row for row in manager.events(session_id) if row["type"] == "vial_decision"
    ]
    assert [row["payload"]["decision_id"] for row in decision_events] == [
        f"decision-abstain-{session_id}-c1",
        f"decision-abstain-{session_id}-c2",
    ]
    assert [row["payload"]["position"] for row in decision_events] == [
        {"tray_index": 0, "vial_number": "cohort-1:unavailable",
         "slot_label": "cohort-1:unavailable"},
        {"tray_index": 0, "vial_number": "cohort-2:unavailable",
         "slot_label": "cohort-2:unavailable"},
    ]
    retained = manager.decisions(session_id)
    assert [row["cohort_ordinal"] for row in retained] == [1, 2]
    assert [row["vial_id"] for row in retained] == [
        "cohort-1:unavailable", "cohort-2:unavailable",
    ]


@pytest.mark.parametrize(("eof_reason", "expected_reason"), [
    ("tracker_stream_state_count_disagrees", "unsupported_recipe_profile"),
    ("tracker_stream_order_disagrees", "unsupported_recipe_profile"),
    ("tracker_stable_tail_missing", "tracker_stable_tail_missing"),
])
def test_rolling_eof_uses_the_same_failure_reason_policy(
    eof_reason, expected_reason, tmp_path, monkeypatch,
):
    class IncompleteAdapter:
        requires_complete = True
        session_complete = False
        eof_abstention_reason = eof_reason

        def process_frame(self, _frame, **_metadata):
            return ()

    manager, session_id, state = _run(
        tmp_path, monkeypatch, Factory(IncompleteAdapter), Frames([0, 0]),
    )

    assert state["state"] == "stopped"
    assert manager.decisions(session_id) == [{
        "type": "abstention", "vial_id": "cohort-1:unavailable",
        "disposition": "insufficient_evidence",
        "limit_codes": [expected_reason], "evidence_citations": [],
        "cohort_ordinal": 1,
    }]

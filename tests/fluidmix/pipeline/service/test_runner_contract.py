from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import jsonschema

from test_runner_service import MemoryFrames, RUN_CONTEXT, ThreeVialAdapter, _wait_terminal


PIN = "9913b1a1cc5559cec9242072decabc3aaf2c4142"
SUPERSEDED_PIN = "cf6829b99486b41f4190874dbd701ad812648217"


def test_factory_attestation_uses_accepted_pipeline_identity():
    from fluidmix.pipeline.service.manager import SessionManager

    accepted = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, development_cohort="varied-v3",
    )
    assert accepted.health()["status"] == "healthy"
    assert accepted.health()["limits"] == []

    superseded = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        pipeline_commit=SUPERSEDED_PIN, development_cohort="varied-v3",
    )
    assert superseded.health()["status"] == "unhealthy"
    assert "pipeline_commit_mismatch" in superseded.health()["limits"]


def _server(*, event_retention=2048):
    from fluidmix.pipeline.service.http import RunnerHTTPServer
    from fluidmix.pipeline.service.manager import SessionManager

    manager = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, development_cohort="varied-v3",
        event_retention=event_retention,
    )
    server = RunnerHTTPServer(("127.0.0.1", 0), manager)
    server.start()
    return manager, server, f"http://127.0.0.1:{server.port}"


def _create_live(base):
    body = json.dumps({
        "source": {"kind": "v4l2", "device": "/dev/video0"},
        "run_context": RUN_CONTEXT,
    }).encode()
    return json.load(urlopen(Request(
        f"{base}/api/sessions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )))


def test_discovery_shapes_pin_and_factory_attestation():
    from fluidmix.pipeline.service.manager import SessionManager

    manager, server, base = _server()
    try:
        health = json.load(urlopen(f"{base}/api/health"))
        assert set(health) == {
            "status", "service", "api_version", "pipeline_commit", "test_only",
            "inference", "active_sessions", "limits",
        }
        assert health["pipeline_commit"] == PIN
        capabilities = json.load(urlopen(f"{base}/api/capabilities"))
        assert capabilities["sources"]["file_upload"] == {
            "supported": True, "max_bytes": 536870912,
            "mime_types": ["video/mp4", "video/x-msvideo"],
        }
        assert capabilities["events"] == {"transport": "sse", "replay": True}
    finally:
        server.close()

    unattested = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        development_cohort="varied-v3",
    )
    assert unattested.health()["status"] == "unhealthy"
    assert unattested.health()["pipeline_commit"] is None
    assert "pipeline_commit_unavailable" in unattested.health()["limits"]
    try:
        unattested.create({
            "source": {"kind": "v4l2", "device": "/dev/video0"},
            "run_context": RUN_CONTEXT,
        })
    except RuntimeError as exc:
        assert "attested" in str(exc)
    else:
        raise AssertionError("unattested runner accepted a session")
    wrong_scope = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        pipeline_commit=PIN, development_cohort="varied-v2",
    )
    assert wrong_scope.health()["status"] == "unhealthy"
    assert "development_cohort_unconfigured" in wrong_scope.health()["limits"]

    malformed = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        pipeline_commit="not-a-commit", development_cohort="varied-v3",
    )
    assert malformed.health()["pipeline_commit"] is None
    assert "pipeline_commit_malformed" in malformed.health()["limits"]


def test_health_reports_actual_commit_and_diagnoses_mismatch():
    from fluidmix.pipeline.service.manager import SessionManager

    actual = "a" * 40
    manager = SessionManager(
        adapter_factory=lambda *_args: ThreeVialAdapter(), frame_source=MemoryFrames(),
        pipeline_commit=actual, development_cohort="varied-v3",
    )
    health = manager.health()
    assert health["pipeline_commit"] == actual
    assert health["status"] == "unhealthy"
    assert "pipeline_commit_mismatch" in health["limits"]


def test_events_match_exact_pinned_frontend_schema():
    api_path = Path(
        "/home/hackathon5/liamreporoot/frontend-ui-20261030/ui/docs/runner-api.md"
    )
    assert hashlib.sha256(api_path.read_bytes()).hexdigest() == (
        "68abaf410fbabb09a2e3da471e6bc6c7aec464a0dae59c64db2459a5f22128f7"
    )
    schema_path = Path(
        "/home/hackathon5/liamreporoot/frontend-ui-20261030/ui/docs/runner-event.schema.json"
    )
    raw = schema_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "3ea09963177d3960e2a49f61958da0a899a5e53cf92adfbe971e902b7b92ccdd"
    )
    schema = json.loads(raw)
    manager, server, base = _server()
    try:
        created = _create_live(base)
        _wait_terminal(manager, created["session_id"])
        rows = manager.events(created["session_id"])
        assert {row["type"] for row in rows} <= {
            "measuring", "waiting_for_references", "vial_decision", "model_vote",
            "averaged_decision", "explanation", "error", "rate_health",
        }
        for row in rows:
            jsonschema.validate(row, schema, format_checker=jsonschema.FormatChecker())
    finally:
        server.close()


def test_cors_replay_headers_and_idempotent_alarm_acknowledgement():
    manager, server, base = _server()
    try:
        try:
            urlopen(Request(f"{base}/api/health", headers={"Origin": "https://evil.test"}))
        except HTTPError as exc:
            assert exc.code == 403
            assert exc.headers.get("Access-Control-Allow-Origin") is None
            assert json.load(exc)["error"]["code"] == "origin_forbidden"
        else:
            raise AssertionError("unlisted origin was admitted")

        created = _create_live(base)
        _wait_terminal(manager, created["session_id"])
        try:
            urlopen(Request(
                f"{base}/api/sessions/{created['session_id']}/events?after=0",
                headers={"Last-Event-ID": "1"},
            ))
        except HTTPError as exc:
            assert exc.code == 400
            assert json.load(exc)["error"]["code"] == "invalid_request"
        else:
            raise AssertionError("mismatched replay cursors were accepted")

        ack_body = json.dumps({"acknowledged_by": "operator-1"}).encode()
        ack_url = (
            f"{base}/api/sessions/{created['session_id']}"
            "/alarms/alarm-final-c1-vial1/acknowledgement"
        )
        first = json.load(urlopen(Request(
            ack_url, data=ack_body, headers={"Content-Type": "application/json"},
            method="PUT",
        )))
        second = json.load(urlopen(Request(
            ack_url, data=ack_body, headers={"Content-Type": "application/json"},
            method="PUT",
        )))
        assert first == second
        assert first["state"] == "acknowledged"
    finally:
        server.close()


def test_options_and_ahead_replay_error_are_contract_envelopes():
    manager, server, base = _server(event_retention=4)
    try:
        preflight = urlopen(Request(
            f"{base}/api/health", method="OPTIONS",
            headers={"Origin": "http://localhost:5173"},
        ))
        assert preflight.status == 204
        assert preflight.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        assert "Last-Event-ID" in preflight.headers["Access-Control-Allow-Headers"]

        created = _create_live(base)
        _wait_terminal(manager, created["session_id"])
        replay = urlopen(
            f"{base}/api/sessions/{created['session_id']}/events?after=0"
        ).read().decode()
        assert "id: 1" in replay
        try:
            urlopen(f"{base}/api/sessions/{created['session_id']}/events?after=99999")
        except HTTPError as exc:
            assert exc.code == 400
            assert json.load(exc)["error"]["code"] == "cursor_ahead"
        else:
            raise AssertionError("ahead replay cursor was admitted")
    finally:
        server.close()


def test_multipart_mp4_upload_and_closed_session_shape():
    manager, server, base = _server()
    boundary = "fluidmix-test-boundary"
    parts = []
    for name, value in (
        ("source_kind", "file"),
        ("run_context", json.dumps(RUN_CONTEXT)),
    ):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
            f"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"operator.mp4\"\r\nContent-Type: video/mp4\r\n\r\n".encode()
        + b"small-video-fixture\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    try:
        created = json.load(urlopen(Request(
            f"{base}/api/sessions", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )))
        assert set(created) == {
            "session_id", "state", "source", "run_context", "created_at",
            "updated_at", "pipeline_commit", "last_sequence", "test_only",
        }
        assert created["source"]["kind"] == "file"
        assert created["source"]["label"] == "operator.mp4"
        _wait_terminal(manager, created["session_id"])
    finally:
        server.close()


def test_decisions_route_returns_retained_decisions():
    manager, server, base = _server()
    try:
        created = _create_live(base)
        _wait_terminal(manager, created["session_id"])
        rows = json.load(urlopen(
            f"{base}/api/sessions/{created['session_id']}/decisions"
        ))
        assert rows == manager.decisions(created["session_id"])
        assert any(row["type"] == "final" for row in rows)
    finally:
        server.close()


def test_vials_route_returns_current_session_vials():
    manager, server, base = _server()
    try:
        created = _create_live(base)
        _wait_terminal(manager, created["session_id"])
        rows = json.load(urlopen(
            f"{base}/api/sessions/{created['session_id']}/vials"
        ))
        assert rows == manager.vials(created["session_id"])
        assert set(rows) == {"vial1", "vial2", "vial3"}
    finally:
        server.close()

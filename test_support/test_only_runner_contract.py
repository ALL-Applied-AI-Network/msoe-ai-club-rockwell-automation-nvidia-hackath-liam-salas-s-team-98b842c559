"""Deterministic TEST-ONLY HTTP contract simulator; never capture or inference."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if os.environ.get("NODE_ENV") == "production":
    raise RuntimeError("TEST-ONLY runner contract simulator is forbidden in production")

PIN_PATH = Path(__file__).resolve().parents[1] / "ui" / "config" / "pipeline-pin.json"
COMMIT = json.loads(PIN_PATH.read_text())["expected_pipeline_commit"]
ALLOWED_ORIGINS = {"http://localhost:4173", "http://localhost:5173", "http://127.0.0.1:4173", "http://127.0.0.1:5173"}
NOW = "2026-10-31T12:00:00Z"


def _source(kind="file", label="operator-selected.avi"):
    return {"source_id": "test-source-1", "kind": kind, "label": label}


def _event(sequence, event_type, payload, source=None):
    return {"schema_version": "runner-event.v1", "event_id": f"test-event-{sequence}", "session_id": "session-1",
            "sequence": sequence, "emitted_at": NOW, "pipeline_commit": COMMIT,
            "source": source or _source(), "type": event_type, "payload": payload}


POSITION = {"tray_index": 0, "vial_number": "1", "slot_label": "A1"}
TRACKER = {"state": "tracking", "geometry_source": "pipeline", "fill_fraction": .64, "confidence": .91,
           "tank_id": "T200", "fill_step_id": "fill-1", "measured_volume_ml": 32.5,
           "volume_provenance": "measured", "unavailable_reason": None,
           "locator": {"status": "located", "bbox_normalized": [.1, .2, .3, .4],
                       "centroid_normalized": [.25, .4], "evidence_ids": ["test-locator-42"], "unavailable_reason": None}}
MEASUREMENT = {"measurement_id": "test-measurement-1", "status": "measured", "rgb": [90, 112, 130],
               "lab": [45.1, -1.2, -4.8], "delta_e": 3.2, "reference_ids": ["test-clear-1"],
               "evidence_ids": ["test-frame-42"], "unavailable_reason": None}

EVENTS = [
    _event(1, "measuring", {"position": POSITION, "tracker": TRACKER, "measurement": MEASUREMENT, "stage": "measuring"}),
    _event(2, "waiting_for_references", {"position": POSITION, "tracker": TRACKER, "missing_reference_types": ["clear"],
        "available_reference_ids": [], "reason": "Deterministic reference wait for UI testing."}),
    _event(3, "vial_decision", {"position": POSITION, "disposition": "discard", "decision_id": "test-decision-1",
        "alarm": {"alarm_id": "alarm-1", "state": "unacknowledged", "acknowledged_at": None, "acknowledged_by": None},
        "measurement": MEASUREMENT, "evidence_ids": ["test-frame-42"], "unavailable_reason": None}),
    _event(4, "model_vote", {"position": POSITION, "vote_id": "test-vote-1",
        "model": {"model_id": "fixture-model", "revision": "test-only-v1", "role": "cnn", "source": "deterministic_test_fixture"},
        "admission": {"admitted": True, "reason": "eligible"}, "disposition": "discard", "confidence": .88,
        "evidence_ids": ["test-frame-42"], "unavailable_reason": None}),
    _event(5, "averaged_decision", {"position": POSITION, "decision_id": "test-average-1", "method": "equal_weight",
        "admitted_vote_ids": ["test-vote-1", "test-vote-2"], "scores": {"keep": .05, "check": .15, "discard": .8},
        "disposition": "discard", "alarm": {"alarm_id": "alarm-1", "state": "unacknowledged", "acknowledged_at": None,
        "acknowledged_by": None}, "evidence_ids": ["test-vote-1", "test-vote-2"], "unavailable_reason": None}),
    _event(6, "explanation", {"position": POSITION, "explanation_id": "test-explanation-1",
        "plain_language": "Deterministic fixture explanation for rendering tests.",
        "limits": ["This is a test-only fixture, not an inference result."], "evidence_ids": ["test-average-1"],
        "producer": {"model_id": "fixture-explainer", "revision": "test-only-v1", "source": "deterministic_test_fixture"},
        "unavailable_reason": None}),
    _event(7, "rate_health", {"status": "healthy", "processing_fps": 12.5, "source_fps": 30,
        "queue_depth": 1, "active_model_id": "fixture-model", "gpu_free_mib": None, "gpu_status": "unavailable",
        "limits": ["No source was decoded and no inference was run."]}),
]
ERROR_EVENT = _event(8, "error", {"code": "TEST_FIXTURE_ERROR", "message": "Deterministic test-only error.",
    "stage": "fixture", "retryable": False, "evidence_ids": []})


class TestOnlyRunnerServer(ThreadingHTTPServer):
    """In-memory deterministic simulator with no source or model integrations."""

    allow_reuse_address = True

    def __init__(self, address):
        super().__init__(address, TestOnlyRunnerHandler)
        self.sessions = {}
        self.next_session = 1
        self.acknowledgements = {}


class TestOnlyRunnerHandler(BaseHTTPRequestHandler):
    server: TestOnlyRunnerServer

    def log_message(self, _format, *_args):
        return

    def _origin_allowed(self):
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS

    def _headers(self, status, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()

    def _json(self, status, value):
        encoded = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin); self.send_header("Vary", "Origin")
        self.end_headers(); self.wfile.write(encoded)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", "0")))

    def _session(self, source, run_context=None):
        session_id = f"session-{self.server.next_session}"; self.server.next_session += 1
        value = {"session_id": session_id, "state": "running", "source": source, "run_context": run_context or {},
                 "created_at": NOW, "updated_at": NOW, "pipeline_commit": COMMIT, "last_sequence": len(EVENTS), "test_only": True}
        self.server.sessions[session_id] = value
        return value

    def do_OPTIONS(self):
        if not self._origin_allowed(): self._json(403, {"error": {"code": "origin_not_allowed", "message": "Origin is not allowlisted."}}); return
        self.send_response(204)
        origin = self.headers.get("Origin")
        if origin: self.send_header("Access-Control-Allow-Origin", origin); self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Last-Event-ID")
        self.end_headers()

    def do_GET(self):
        if not self._origin_allowed(): self._json(403, {"error": {"code": "origin_not_allowed", "message": "Origin is not allowlisted."}}); return
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(200, {"status": "healthy", "service": "fluidmix-runner-contract-simulator-test-only", "api_version": "v1",
                "pipeline_commit": COMMIT, "test_only": True, "inference": False, "active_sessions": len(self.server.sessions),
                "limits": ["TEST-ONLY: no capture, decoding, or inference."]}); return
        if parsed.path == "/api/capabilities":
            self._json(200, {"test_only": True, "sources": {"file_upload": True, "v4l2": True, "rtsp": True},
                "file_upload": {"max_bytes": 536870912, "media_types": ["video/mp4", "video/x-msvideo"]},
                "events": {"transport": "sse", "replay": True}, "alarm_acknowledgement": True}); return
        match = re.fullmatch(r"/api/sessions/([^/]+)", parsed.path)
        if match and match.group(1) in self.server.sessions: self._json(200, self.server.sessions[match.group(1)]); return
        match = re.fullmatch(r"/api/sessions/([^/]+)/events", parsed.path)
        if match and match.group(1) in self.server.sessions:
            query = parse_qs(parsed.query)
            try: after = int(query.get("after", [self.headers.get("Last-Event-ID", "0")])[0])
            except ValueError: self._json(400, {"error": {"code": "invalid_cursor", "message": "after must be an integer"}}); return
            last_event_id = self.headers.get("Last-Event-ID")
            if after < 0 or after > EVENTS[-1]["sequence"] or (last_event_id is not None and int(last_event_id) != after):
                self._json(400, {"error": {"code": "invalid_cursor", "message": "Replay cursor is out of range or conflicts with Last-Event-ID."}}); return
            scenario_events = EVENTS + [ERROR_EVENT] if query.get("scenario") == ["error"] else EVENTS
            selected = [item for item in scenario_events if item["sequence"] > after]
            encoded = "".join(f'id: {item["sequence"]}\nevent: {item["type"]}\ndata: {json.dumps(item, separators=(",", ":"))}\n\n' for item in selected).encode()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            origin = self.headers.get("Origin")
            if origin in ALLOWED_ORIGINS: self.send_header("Access-Control-Allow-Origin", origin); self.send_header("Vary", "Origin")
            self.end_headers(); self.wfile.write(encoded); return
        self._json(404, {"error": {"code": "not_found", "message": "Unknown test-only resource."}})

    def do_POST(self):
        if self.path == "/api/sessions":
            content_type = self.headers.get("Content-Type", "")
            raw = self._body()
            if content_type.startswith("multipart/form-data"):
                filename = re.search(br'filename="([^"\\/]+)"', raw)
                if not filename: self._json(400, {"error": {"code": "file_required", "message": "One file is required."}}); return
                source = _source("file", filename.group(1).decode("utf-8", "replace"))
                self._json(201, self._session(source)); return
            try: body = json.loads(raw or b"{}"); source_body = body["source"]; kind = source_body["kind"]
            except (KeyError, ValueError, TypeError): self._json(400, {"error": {"code": "invalid_source", "message": "A source is required."}}); return
            if kind == "v4l2": label = source_body.get("device", "V4L2 device")
            elif kind == "rtsp": label = "RTSP stream (credentials redacted)"
            else: self._json(400, {"error": {"code": "invalid_source", "message": "Use multipart for files."}}); return
            self._json(201, self._session(_source(kind, label), body.get("run_context", ""))); return
        match = re.fullmatch(r"/api/sessions/([^/]+)/stop", self.path)
        if match and match.group(1) in self.server.sessions:
            value = self.server.sessions[match.group(1)]; value["state"] = "stopped"; self._json(200, value); return
        self._json(404, {"error": {"code": "not_found", "message": "Unknown test-only resource."}})

    def do_PUT(self):
        match = re.fullmatch(r"/api/sessions/([^/]+)/alarms/([^/]+)/acknowledgement", self.path)
        if not match or match.group(1) not in self.server.sessions or match.group(2) != "alarm-1":
            self._json(404, {"error": {"code": "not_found", "message": "Unknown alarm."}}); return
        try: operator = json.loads(self._body())["acknowledged_by"]
        except (KeyError, ValueError, TypeError): self._json(400, {"error": {"code": "operator_required", "message": "acknowledged_by is required."}}); return
        existing = self.server.acknowledgements.get("alarm-1")
        if existing:
            if existing["acknowledged_by"] != operator:
                self._json(409, {"error": {"code": "acknowledgement_conflict", "message": "Alarm was acknowledged by another operator."}}); return
            self._json(200, existing); return
        value = {"alarm_id": "alarm-1", "state": "acknowledged", "acknowledged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "acknowledged_by": operator}
        self.server.acknowledgements["alarm-1"] = value; self._json(200, value)


def start_test_only_server(host="127.0.0.1", port=0):
    """Start the TEST-ONLY simulator in a daemon thread and return its server."""
    server = TestOnlyRunnerServer((host, port))
    threading.Thread(target=server.serve_forever, name="test-only-runner-contract", daemon=True).start()
    return server

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

try:
    import jsonschema
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "ui" / "docs" / "runner-event.schema.json"
COMMIT = "7bf35aeea21c36acf1a390661bd05fea2b07b190"


def event(event_type, payload, sequence=1):
    return {"schema_version": "runner-event.v1", "event_id": f"evt-{sequence}", "session_id": "session-1",
            "sequence": sequence, "emitted_at": "2026-10-31T12:00:00Z", "pipeline_commit": COMMIT,
            "source": {"source_id": "source-1", "kind": "file", "label": "selected.avi"},
            "type": event_type, "payload": payload}


POSITION = {"tray_index": 0, "vial_number": "1", "slot_label": "A1"}
TRACKER = {"state": "tracking", "geometry_source": "pipeline", "fill_fraction": .64, "confidence": .91,
           "tank_id": "T200", "fill_step_id": "fill-1", "measured_volume_ml": 32.5,
           "volume_provenance": "measured", "unavailable_reason": None,
           "locator": {"status": "located", "bbox_normalized": [.1, .2, .3, .4],
                       "centroid_normalized": [.25, .4], "evidence_ids": ["locator-42"], "unavailable_reason": None}}
MEASUREMENT = {"measurement_id": "measurement-1", "status": "measured", "rgb": [90, 112, 130],
               "lab": [45.1, -1.2, -4.8], "delta_e": 3.2, "reference_ids": ["clear-1"],
               "evidence_ids": ["frame-42"], "unavailable_reason": None}

VALID_EVENTS = {
    "measuring": event("measuring", {"position": POSITION, "tracker": TRACKER, "measurement": MEASUREMENT, "stage": "measuring"}),
    "waiting_for_references": event("waiting_for_references", {"position": POSITION, "tracker": TRACKER,
        "missing_reference_types": ["clear"], "available_reference_ids": [], "reason": "Clear reference unavailable."}),
    "vial_decision": event("vial_decision", {"position": POSITION, "disposition": "discard", "decision_id": "decision-1",
        "alarm": {"alarm_id": "alarm-1", "state": "unacknowledged", "acknowledged_at": None, "acknowledged_by": None},
        "measurement": MEASUREMENT, "evidence_ids": ["frame-42"], "unavailable_reason": None}),
    "model_vote": event("model_vote", {"position": POSITION, "vote_id": "vote-1",
        "model": {"model_id": "fixture-model", "revision": "test-only-v1", "role": "cnn", "source": "deterministic_test_fixture"},
        "admission": {"admitted": True, "reason": "eligible"}, "disposition": "discard", "confidence": .88,
        "evidence_ids": ["frame-42"], "unavailable_reason": None}),
    "averaged_decision": event("averaged_decision", {"position": POSITION, "decision_id": "average-1", "method": "equal_weight",
        "admitted_vote_ids": ["vote-1", "vote-2"], "scores": {"keep": .05, "check": .15, "discard": .8},
        "disposition": "discard", "alarm": {"alarm_id": "alarm-1", "state": "unacknowledged", "acknowledged_at": None,
        "acknowledged_by": None}, "evidence_ids": ["vote-1", "vote-2"], "unavailable_reason": None}),
    "explanation": event("explanation", {"position": POSITION, "explanation_id": "explanation-1", "plain_language": "Fixture explanation.",
        "limits": ["Not inference."], "evidence_ids": ["average-1"],
        "producer": {"model_id": "fixture-explainer", "revision": "test-only-v1", "source": "deterministic_test_fixture"},
        "unavailable_reason": None}),
    "error": event("error", {"code": "TEST_FIXTURE_ERROR", "message": "Fixture error.", "stage": "fixture",
        "retryable": False, "evidence_ids": []}),
    "rate_health": event("rate_health", {"status": "healthy", "processing_fps": 12.5, "source_fps": 30,
        "queue_depth": 1, "active_model_id": "fixture-model", "gpu_free_mib": None, "gpu_status": "unavailable",
        "limits": ["Test-only deterministic stream."]}),
}


def validator():
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


@pytest.mark.parametrize("event_type", VALID_EVENTS)
def test_schema_accepts_every_closed_event_variant(event_type):
    validator().validate(VALID_EVENTS[event_type])


def test_schema_rejects_unknown_variants_and_fields():
    with pytest.raises(jsonschema.ValidationError): validator().validate(event("made_up", {}))
    with pytest.raises(jsonschema.ValidationError): validator().validate(dict(VALID_EVENTS["measuring"], surprise=True))
    drift = json.loads(json.dumps(VALID_EVENTS["measuring"])); drift["payload"]["surprise"] = True
    with pytest.raises(jsonschema.ValidationError): validator().validate(drift)


@pytest.mark.parametrize("event_type,field", [("measuring", "measurement"), ("waiting_for_references", "missing_reference_types"),
    ("vial_decision", "alarm"), ("model_vote", "admission"), ("averaged_decision", "scores"), ("averaged_decision", "alarm"),
    ("explanation", "limits"), ("error", "retryable"), ("rate_health", "processing_fps")])
def test_schema_requires_hmi_fields(event_type, field):
    candidate = json.loads(json.dumps(VALID_EVENTS[event_type])); del candidate["payload"][field]
    with pytest.raises(jsonschema.ValidationError): validator().validate(candidate)


@pytest.mark.parametrize("event_type", ["vial_decision", "averaged_decision"])
def test_discard_decisions_require_an_alarm_and_non_discard_decisions_forbid_one(event_type):
    discard_without_alarm = json.loads(json.dumps(VALID_EVENTS[event_type]))
    discard_without_alarm["payload"]["alarm"] = None
    with pytest.raises(jsonschema.ValidationError): validator().validate(discard_without_alarm)
    keep_with_alarm = json.loads(json.dumps(VALID_EVENTS[event_type]))
    keep_with_alarm["payload"]["disposition"] = "keep"
    with pytest.raises(jsonschema.ValidationError): validator().validate(keep_with_alarm)
    keep_with_alarm["payload"]["alarm"] = None
    validator().validate(keep_with_alarm)


@contextmanager
def running_server():
    from test_support.test_only_runner_contract import start_test_only_server
    server = start_test_only_server()
    try: yield f"http://127.0.0.1:{server.server_port}"
    finally: server.shutdown(); server.server_close()


def request(base, path, *, method="GET", body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req) as response: return response.status, dict(response.headers), response.read()


def test_test_only_server_lifecycle_sources_replay_and_acknowledgement():
    with running_server() as base:
        status, _, raw = request(base, "/api/health"); health = json.loads(raw)
        assert status == 200 and health["test_only"] is True and health["inference"] is False
        _, _, raw = request(base, "/api/capabilities")
        capabilities = json.loads(raw)
        assert capabilities["sources"] == {"file_upload": True, "v4l2": True, "rtsp": True}
        assert capabilities["file_upload"]["max_bytes"] == 536870912
        assert capabilities["file_upload"]["media_types"] == ["video/mp4", "video/x-msvideo"]
        for source in ({"kind": "v4l2", "device": "/dev/video0"}, {"kind": "rtsp", "url": "rtsp://camera/live"}):
            run_context = {"operator_run_id": "contract-test", "recipe_id": None, "tray_id": "tray-1"}
            status, _, raw = request(base, "/api/sessions", method="POST", body={"source": source, "run_context": run_context})
            session = json.loads(raw)
            assert status == 201 and session["state"] == "running" and session["run_context"] == run_context
        status, _, raw = request(base, "/api/sessions/session-1/events?after=3")
        events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ")]
        assert status == 200 and [x["sequence"] for x in events] == sorted(x["sequence"] for x in events)
        assert all(x["sequence"] > 3 for x in events)
        assert set(x["type"] for x in events) == set(VALID_EVENTS) - {"measuring", "waiting_for_references", "vial_decision", "error"}
        status, headers, raw = request(base, "/api/sessions/session-1/events?after=3", headers={"Origin": "http://localhost:4173"})
        assert status == 200 and headers["Access-Control-Allow-Origin"] == "http://localhost:4173"
        assert all(json.loads(line[6:])["type"] != "error" for line in raw.decode().splitlines() if line.startswith("data: "))
        _, _, raw = request(base, "/api/sessions/session-1/events?after=6&scenario=error")
        assert any(json.loads(line[6:])["type"] == "error" for line in raw.decode().splitlines() if line.startswith("data: "))
        for path, headers in (("/api/sessions/session-1/events?after=-1", {}),
                              ("/api/sessions/session-1/events?after=99", {}),
                              ("/api/sessions/session-1/events?after=3", {"Last-Event-ID": "2"})):
            with pytest.raises(urllib.error.HTTPError) as invalid_cursor:
                request(base, path, headers=headers)
            assert invalid_cursor.value.code == 400
        status, _, raw = request(base, "/api/sessions/session-1/alarms/alarm-1/acknowledgement", method="PUT", body={"acknowledged_by": "operator-7"})
        first_ack = json.loads(raw)
        assert status == 200 and first_ack["state"] == "acknowledged"
        status, _, raw = request(base, "/api/sessions/session-1/alarms/alarm-1/acknowledgement", method="PUT", body={"acknowledged_by": "operator-7"})
        assert status == 200 and json.loads(raw) == first_ack
        with pytest.raises(urllib.error.HTTPError) as conflict:
            request(base, "/api/sessions/session-1/alarms/alarm-1/acknowledgement", method="PUT", body={"acknowledged_by": "operator-8"})
        assert conflict.value.code == 409
        status, _, raw = request(base, "/api/sessions/session-1/stop", method="POST", body={})
        assert status == 200 and json.loads(raw)["state"] == "stopped"


def test_file_upload_is_accepted_without_decoding_source_bytes():
    boundary = "contract-boundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"source_kind\"\r\n\r\nfile\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"operator.avi\"\r\nContent-Type: video/x-msvideo\r\n\r\nnot-a-video\r\n--{boundary}--\r\n").encode()
    with running_server() as base:
        req = urllib.request.Request(base + "/api/sessions", data=body, method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req) as response: payload = json.loads(response.read())
        assert payload["source"]["kind"] == "file" and payload["source"]["label"] == "operator.avi" and payload["test_only"] is True


def test_cors_is_allowlisted_and_production_mode_refuses_server():
    with running_server() as base:
        status, headers, _ = request(base, "/api/health", method="OPTIONS", headers={"Origin": "http://localhost:4173"})
        assert status == 204 and headers["Access-Control-Allow-Origin"] == "http://localhost:4173"
        with pytest.raises(urllib.error.HTTPError) as rejected:
            request(base, "/api/health", method="OPTIONS", headers={"Origin": "https://unreviewed.example"})
        assert rejected.value.code == 403
    result = subprocess.run([sys.executable, "-c", "import test_support.test_only_runner_contract"], cwd=ROOT,
        env={**os.environ, "NODE_ENV": "production"}, text=True, capture_output=True)
    assert result.returncode != 0 and "TEST-ONLY" in result.stderr

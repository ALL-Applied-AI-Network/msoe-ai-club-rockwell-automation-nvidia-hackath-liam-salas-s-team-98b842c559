# Pipeline runner HTTP contract (v1)

This document specifies the pipeline-side service consumed by the operator HMI. The service owns
source capture, tracking and inference. This repository does not implement those production
functions. The only production integration setting is `VITE_RUNNER_BASE_URL`; there is no data-mode
switch and the HMI must fail closed when this service or a required event is unavailable.

The sole expected pipeline pin is `ui/config/pipeline-pin.json` field
`expected_pipeline_commit`. Production health, session, and every event report the service's
actual 40-character `pipeline_commit`; they do not substitute the expected value. Before accepting
a session or event, the HMI compares that attestation with the canonical pin. A missing, malformed,
or mismatched attestation fails closed, stops decision rendering, and displays both a visible
pipeline-version mismatch alarm and the expected/actual values for diagnosis.

## HTTP conventions and deployment

All JSON responses use `Content-Type: application/json`. Errors use
`{"error":{"code":"...","message":"..."}}` and an appropriate 4xx/5xx status. IDs are opaque,
non-empty strings. Timestamps are UTC RFC 3339 strings. Session and event source labels must not
expose RTSP credentials.

Same-origin deployment is preferred. When `VITE_RUNNER_BASE_URL` is cross-origin, the runner must
answer `OPTIONS` and normal requests only for a reviewed origin allowlist. It echoes the admitted
origin in `Access-Control-Allow-Origin`, includes `Vary: Origin`, and admits `GET, POST, PUT,
OPTIONS` plus `Content-Type` and `Last-Event-ID`. It must not combine credentialed requests with
`Access-Control-Allow-Origin: *`. An unlisted origin receives 403. The test-only implementation
admits only localhost development origins and never enables credentials.

## Discovery

`GET /api/health` returns:

```json
{
  "status": "healthy",
  "service": "fluidmix-pipeline-runner",
  "api_version": "v1",
  "pipeline_commit": "<actual-40-hex-service-commit>",
  "test_only": false,
  "inference": true,
  "active_sessions": 0,
  "limits": []
}
```

`GET /api/capabilities` returns a closed capability object. `sources.file_upload`, `v4l2`, and
`rtsp` declare creation support. `file_upload` reports `max_bytes` (v1 requires 536870912,
512 MiB) and the admitted MIME types `video/mp4` and `video/x-msvideo`; a larger or different
upload receives 413 or 415. `events.transport` is `sse`; `events.replay` is true;
`alarm_acknowledgement` is true. Optional server-local paths are a separate
`sources.server_local_file_path` capability and default to false; they are not the browser upload
flow.

## Session creation

The response to every accepted creation is 201 and the session representation below.

- Browser file: `POST /api/sessions` as `multipart/form-data` with exactly one `file`,
  `source_kind=file`, and optional `run_context` containing the JSON object described below. The service streams/owns the upload. A
  browser path is never accepted as a substitute. Invalid or unsupported media fails explicitly;
  it cannot select fixture decisions.
- V4L2: JSON `{"source":{"kind":"v4l2","device":"/dev/video0"},"run_context":{"operator_run_id":"run-1","recipe_id":null,"tray_id":"tray-1"}}`.
- RTSP: JSON `{"source":{"kind":"rtsp","url":"rtsp://..."},"run_context":{"operator_run_id":"run-1","recipe_id":null,"tray_id":"tray-1"}}`. The
  response label redacts user information and credentials.

Creation is asynchronous. `created` means accepted but not processing; `running` means processing;
`stopping` is transitional; `stopped` and `failed` are terminal. A session never returns to an
earlier state.

```json
{
  "session_id": "session-opaque",
  "state": "running",
  "source": {"source_id":"source-opaque","kind":"file","label":"operator-selected.avi"},
  "run_context": {"operator_run_id":"run-1","recipe_id":null,"tray_id":"tray-1"},
  "created_at": "2026-10-31T12:00:00Z",
  "updated_at": "2026-10-31T12:00:01Z",
  "pipeline_commit": "<actual-40-hex-service-commit>",
  "last_sequence": 0,
  "test_only": false
}
```

`GET /api/sessions/{session_id}` returns the current representation. `POST
/api/sessions/{session_id}/stop` is idempotent, requests resource release, and returns the updated
representation. Stopping does not delete replay history.

## Ordered event stream and reconnect

`GET /api/sessions/{session_id}/events?after=N` returns `text/event-stream`. Each message has
`id: <sequence>`, `event: <type>`, and one `data: <JSON event>` line. The first session event has
sequence 1. Sequences are strictly increasing, unique within the session, and never reused.
`after` is an exclusive non-negative cursor: replay returns every retained event with
`sequence > N` in ascending order, followed by live events. Omitting it is equivalent to zero.
`Last-Event-ID` may supply the same cursor; if both are present they must agree or the request is
400. Reconnecting after the last applied sequence therefore cannot duplicate an applied event.

The server sends SSE comments as keepalives; comments do not consume sequences. It must retain the
complete event history until the terminal session retention period expires. A cursor older than
retained history receives 409 `replay_unavailable`, and an ahead-of-stream cursor receives 400.
Clients discard a byte-identical duplicate sequence only during connection handoff; a different
event at an already applied sequence is protocol corruption and fails closed.

Events validate against [runner-event.schema.json](runner-event.schema.json). The vocabulary is
closed: `measuring`, `waiting_for_references`, `vial_decision`, `model_vote`,
`averaged_decision`, `explanation`, `error`, and `rate_health`. Every envelope identifies the
session/source, sequence, event ID, service commit and timestamp. Payloads carry the HMI fields:
physical tray position, tracker and locator status, measured/inferred/unavailable provenance,
references and evidence IDs, exact disposition, alarm state, model/revision/source, vote admission,
equal-weight scores, plain explanation and limits, errors, processing/source rate, queue, active
model and GPU health. Nullable values remain present with an explicit unavailable reason; a
missing stage is not interpreted as a favorable decision.

`vial_decision.disposition` is exactly `keep`, `check`, `discard`, or `insufficient_evidence`.
Only discard requires an alarm. `averaged_decision.method` is exactly `equal_weight`, and
`admitted_vote_ids` names the votes included. Model outputs and explanations must say whether their
source is measured, inferred, unavailable, or—in the simulator only—a deterministic test fixture.

Tracker state includes `tank_id`, pipeline-order `fill_step_id`, `measured_volume_ml` and its
measured/inferred/unavailable provenance. The locator hook is tied to the event's vial/slot and
provides normalized `[x,y,width,height]` bounding box plus `[x,y]` centroid and evidence IDs, or
explicit null geometry with `status: unavailable` and a reason. The hook cannot invent geometry.
Both vial and final averaged decisions carry `alarm`; it is non-null exactly when the disposition
is discard. The final averaged discard raises the operator alarm. A later non-discard revision may
retire it only through a new higher-sequence decision; an acknowledgement alone never retires or
changes an alarm.

## Alarm acknowledgement

`PUT /api/sessions/{session_id}/alarms/{alarm_id}/acknowledgement` with
`{"acknowledged_by":"operator-id"}` acknowledges a discard alarm. It is idempotent for the same
operator and returns:

```json
{
  "alarm_id": "alarm-1",
  "state": "acknowledged",
  "acknowledged_at": "2026-10-31T12:01:00Z",
  "acknowledged_by": "operator-id"
}
```

Acknowledgement records operator awareness only; it cannot change disposition, evidence, or model
output. Unknown/non-discard alarms return 404. Conflicting operator identity returns 409.

## Test-only contract simulator

`test_support/test_only_runner_contract.py` is a deterministic HTTP simulator for contract and HMI
tests. Every health/session payload sets `test_only: true`, every synthetic producer uses
`deterministic_test_fixture`, and health says `inference: false`. It does not decode uploads, open
V4L2/RTSP sources, import pipeline code, or claim inference. It refuses to import when
`NODE_ENV=production`, is outside `ui/src`, is not a production fallback, and is never selected by
`VITE_RUNNER_BASE_URL` in a production build. Its normal replay omits the synthetic `error`; tests
that need error rendering explicitly request `scenario=error` on the simulator-only event URL.

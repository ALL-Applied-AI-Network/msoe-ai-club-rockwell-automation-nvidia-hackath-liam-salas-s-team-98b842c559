# Live runner service

The outermost `fluidmix.pipeline.service` adapter implements the pinned runner v1
HTTP/SSE contract. Its exact API/schema hashes and approved pipeline commit are recorded
in [runner-contract-pin.json](runner-contract-pin.json). A deployment factory must attest
commit `cf6829b99486b41f4190874dbd701ad812648217` and declare `varied-v3` or `varied-v4`
development scope. Health reports that factory-supplied value as the actual commit rather
than substituting the expected pin. A missing, malformed, or mismatched actual commit keeps
health unhealthy and session creation fails closed.

## Surface

| Method and path | Result |
|---|---|
| `GET /api/health` | Closed health and pin status |
| `GET /api/capabilities` | Upload/live/SSE/acknowledgement capabilities |
| `POST /api/sessions` | Multipart MP4/AVI upload or JSON V4L2/RTSP creation |
| `GET /api/sessions/{id}` | Closed session representation |
| `GET /api/sessions/{id}/decisions` | Retained deterministic/final decision records |
| `GET /api/sessions/{id}/events?after=N` | Retained replay followed by live SSE |
| `POST /api/sessions/{id}/stop` | Idempotent owned-source cancellation |
| `PUT /api/sessions/{id}/alarms/{alarm}/acknowledgement` | Idempotent operator acknowledgement |

Uploads are streamed to service-owned temporary files, limited to 512 MiB, and removed
after processing. JSON creation never accepts a browser/local path. The exact frontend
run context contains `operator_run_id`, nullable `recipe_id`, and `tray_id`; arbitrary
simulation truth/oracle fields fail before the adapter factory. Server deployment config,
not browser input, selects the authorized V3/V4 development cohort.

Events use only the eight `runner-event.v1` types. Internal tracker/CNN/measurement steps
map to `measuring`; reference gating maps to `waiting_for_references`; deterministic and
model results map to `vial_decision` and `model_vote`; aggregation and text map to
`averaged_decision` and `explanation`. `rate_health` carries progress/terminal health and
`error` carries a scrubbed failure. Status plus the final retained event are published in
one locked transition, after adapter cleanup.

The accepted adapter retains image paths only in a service-owned temporary directory. It
binds each private RGB locator by SHA-256, joins the three closed-vial tracker receipts, and
calls the frozen builder with `cohort`, `image_locators`, and `tracker_receipts`. Paths never
enter session state or HMI events. Deterministic target-to-reference comparison runs only
after positions 2 and 3 close; the resulting receipt is then bound into the VLM cohort.
Model events preserve the repository, revision, model confidence, explanation, and cited
evidence/image IDs. The normalized discard score remains a separate aggregation value.
When fewer than two votes are admissible, the event uses `insufficient_evidence`, explicit
unavailability, and neutral `keep=0`, `check=1`, `discard=0` scores.

Replay accepts `after` or an agreeing `Last-Event-ID`, sends keepalive comments, rejects
ahead cursors, and returns `replay_unavailable` for expired cursors. Event and completed
session memory are count-bounded. Source cancellation closes the owned PyAV container;
RTSP also has bounded open/read timeouts. Server shutdown stops and joins every session
worker and reports failure rather than silently leaving one behind.

Cross-origin access admits only a configured allowlist, never wildcard credentials.
RTSP user information and query credentials never enter public source labels or public
errors. GPU selection still requires the explicit `FLUIDMIX_GPU_FREE=1` operator
assertion; the service never inspects or changes another process.

Start with a factory that reports the actual checked-out service commit:

```bash
PYTHONPATH=src python -m fluidmix.pipeline.service.http \
  --adapter-factory your_deployment.runner:create_adapter --port 8765
```

The factory callable must expose its actual `pipeline_commit` and `development_cohort`
attributes. This integration branch contains the WP7 VLM lineage from
`b5e63cb5ceb54f0f3d5688b48613ce2e7362e620` and tests incremental decoding,
`AcceptedPipelineAdapter`, SHA-bound CIELAB measurement, and the exact frozen-builder call
seam. The retained expected pin is
`cf6829b99486b41f4190874dbd701ad812648217`, so an honest factory for this candidate reports a
different actual commit and the service remains unhealthy and refuses session creation. It
must not serve until the final integration commit is reviewed and both runner and frontend
pins are updated together. The tests prove service wiring, not camera accuracy, physical
validity, or plant suitability.

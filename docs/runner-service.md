# Live runner service

The outermost `fluidmix.pipeline.service` adapter implements the pinned runner v1
HTTP/SSE contract. Its exact API/schema hashes and accepted pipeline implementation are
recorded in [runner-contract-pin.json](runner-contract-pin.json). The HTTP entry point
requires a deployment-owned `fluidmix-runner-build-attestation-v1` JSON generated after
the deployed service commit exists. Its `service_commit` is the public `pipeline_commit`;
the accepted implementation `9913b1a1cc5559cec9242072decabc3aaf2c4142` remains a separate
frozen-profile identity. Missing, malformed, or mismatched identity fails closed.

## Surface

| Method and path | Result |
|---|---|
| `GET /api/health` | Closed health and pin status |
| `GET /api/capabilities` | Upload/live/SSE/acknowledgement capabilities |
| `POST /api/sessions` | Multipart MP4/AVI upload or JSON V4L2/RTSP creation |
| `GET /api/sessions/{id}` | Closed session representation |
| `GET /api/sessions/{id}/vials` | Current closed-vial state |
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
scales the frozen FillCam ROI proportionally for a validated 4:3 native resolution, crops
with the accepted helper, and binds the crop, source pixels, dimensions, scale, and crop
coordinates by SHA-256. It joins the three causally closed-vial
tracker receipts and calls the frozen builder with `cohort`, `image_locators`, and
`tracker_receipts`. Paths never enter session state or HMI events. Missing or drifted camera
metadata fails closed. Deterministic target-to-reference comparison runs only
after positions 2 and 3 close; the resulting receipt is then bound into the VLM cohort.
Model events preserve the repository, revision, model confidence, explanation, and cited
evidence/image IDs. The normalized discard score remains a separate aggregation value.
When fewer than two votes are admissible, the event uses `insufficient_evidence`, explicit
unavailability, and neutral `keep=0`, `check=1`, `discard=0` scores.

Replay accepts `after` or an agreeing `Last-Event-ID`, sends keepalive comments, and rejects
ahead cursors. The complete event and decision history remains replayable while its session
is retained; completed session count is bounded. Source cancellation closes the owned PyAV
container; RTSP also has bounded open/read timeouts. Server shutdown stops and joins every
session worker and reports failure rather than silently leaving one behind.

Cross-origin access admits only a configured allowlist, never wildcard credentials.
RTSP user information and query credentials never enter public source labels or public
errors. GPU selection still requires the explicit `FLUIDMIX_GPU_FREE=1` operator
assertion; the service never inspects or changes another process.

The repository-native production factory wires the accepted tracker, frozen CNN, deterministic
decision configuration, VLM bundle builder, three pinned model runtimes, device selection and
model identities. The service commit is supplied by the post-commit build attestation; the
accepted frozen implementation identity remains separate. The exact CPU command that proves the
factory-attestation boundary is:

```bash
PYTHONPATH=src python -m pytest \
  tests/fluidmix/pipeline/service/test_runner_contract.py::test_factory_attestation_uses_accepted_pipeline_identity \
  -q
```

The launch still requires a post-commit build attestation and the ignored, hash-verified local
model/artifact directories documented in [runner.md](runner.md). The service tests incremental
decoding, `AcceptedPipelineAdapter`, SHA-bound CIELAB measurement, and the exact frozen-builder
call seam. The expected accepted pin is
`9913b1a1cc5559cec9242072decabc3aaf2c4142`; it is not reported as the deployed service commit.
The superseded `cf6829b` identity and every missing or malformed identity remain unavailable.

The historical `0.90` and `0.925` keep/discard live-demo projections are hybrid, superseded
evidence: both combined H.264 tracking receipts with pinned native-MJPEG stage measurements and
images. The development VLM votes remain valid for their exact bundles, but neither result is a
current one-source keep/discard result and the service never reconstructs that hybrid path. The
actual single-source H.264 path measures stage-1 DeltaE76 `1.149856 / 1.147971` and abstains in
the frozen deadband. Arbitrary sources use one frozen path and may honestly abstain with a specific
tracker, geometry or measurement reason. `tracker-freeze Option B` is HELD for Liam; no new tracker
freeze is authorized. These tests prove service wiring, not camera accuracy, physical validity, or
plant suitability.

The latest fully dynamic single-source development session (`6a2e9d63c3034680b963b953d210c74e`,
service commit `fe64833`) stopped earlier with `measurement_stage_unavailable`: one selected
phase lacked complete accepted CNN topology. Specifically, vial 1 stage 1 early `F0000` had a
zero-pixel post-topology mask (`bottom_zone_component_missing`); its mid/end and all stage-2/3
phases were valid. The frozen all-three-phase aggregate therefore retained no vial and made no
model call. This is the current honest V4 TRAIN-01 live outcome; it does not invalidate the prior
development VLM evidence for its exact bundle and is not replaced by hybrid evidence.

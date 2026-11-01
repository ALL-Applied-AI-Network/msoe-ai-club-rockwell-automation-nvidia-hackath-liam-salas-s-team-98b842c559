# Phase 3 pipeline-service HMI plan

Spec authority: Liam's Phase 3 request and correction relayed on 2026-10-31. The dashboard is the
product, not a prerecorded demo. The operator chooses a file upload, V4L2 device, or RTSP source,
while the pipeline-side service owns capture and inference. This repository defines and consumes
that service API; it must not implement or duplicate the runner.

Base: `6e9a1a714628be3e7ff58e4d3480721e6ed362cb` on
`codex/frontend-ui-20261030`.

## Global constraints

- Work only in this worktree and branch; no push. Other worktrees, model caches, and outputs are
  read-only. Do not copy or edit lead code.
- Develop and test only on V3/V4 twin videos. Runs 1–4 permit logged smoke checks only and no
  tuning. Video 5 is forbidden until an ops-announced final recording. Every later real-footage
  read must be logged in `ui-PROGRESS.md`.
- No Brev. Before GPU inference, record free memory. Never inspect, stop, or alter another user's
  process beyond read-only telemetry. Load at most one CNN/VLM model at a time and release it
  before the next.
- Mock data and the local contract server are test-only. The production UI does not import,
  fetch, or fall back to `ui-data/active-run.json`, `mock-run.json`, or any precomputed decision
  artifact.
- Every outcome, explanation, model identity, measured/inferred tag, vote, and evidence ID comes
  from the current session event stream. Missing stages fail closed and stay visibly unavailable.
- The teammate top-down vial-locator slot remains present but cannot manufacture geometry.

## Pinned pipeline boundary

The proposed service contract requires the pipeline to report its pinned commit in session and
health payloads. The accepted candidate inspected read-only is
`7bf35aeea21c36acf1a390661bd05fea2b07b190`; it contains byte-identical tracker/runtime modules
from tracker head `4070be2f73b725e19fc2ea84ee9e4991b089d1e1`. The pipeline service, not the UI, enforces
its source pin and fails closed on drift.

The current lead branch exposes low-level tracker, CNN, Lab, deterministic-decision, VLM bundle,
sequential-model, and aggregation functions but no incremental session runner. Liam assigned that
runner to the pipeline lead. The UI must not compose these functions or implement source capture.

GPU preflight observation at planning time: CUDA reported 13.46 GiB free of 121.63 GiB unified
memory; process telemetry was unavailable because `pynvml` is not installed, and the available
PyTorch warned that GB10 capability 12.1 exceeds its advertised 12.0 maximum. No inference was
launched. GPU admission and one-model-at-a-time enforcement belong to the pipeline service and
are observable through rate/health events.

## Runner API and event contract

`ui/docs/runner-api.md` is the endpoint and lifecycle specification.
`ui/docs/runner-event.schema.json` is the closed event schema. Together they cover service
capabilities and health; multipart video, V4L2, and RTSP session creation; session inspection and
stop; ordered replayable event streaming and reconnect; alarm acknowledgement; and the event
types `measuring`, `waiting_for_references`, `vial_decision`, `model_vote`,
`averaged_decision`, `explanation`, `error`, and `rate_health`.

Payloads expose monotonic sequence, source/session identity, vial/tray position, tracker state,
measurement provenance, exact disposition, model/revision and vote admission, equal-weight
average, evidence IDs, plain explanation, limits, processing rate, pipeline commit, and timestamps.
Null or unavailable data remains explicit.

One config value, `VITE_RUNNER_BASE_URL`, selects the service origin. The UI never selects a data
mode. A thin local server implements the contract only for tests; it is named, documented, and
visibly labelled test-only and cannot be selected by a production build.

## Task 1 — runner contract and test-only server (sol-medium, parallel lane A)

Own `ui/docs/runner-api.md`, `ui/docs/runner-event.schema.json`, the test-only server, and their
tests. Define endpoints, lifecycle, ordering/reconnect behavior, closed events, and HMI-required
payloads test-first. The test server may emit deterministic fixtures to exercise rendering. It
must not claim inference, decode a source, import lead code, or be a runtime fallback. Do not add a
production `server/` implementation.

Verification: JSON Schema positive/negative tests, endpoint/lifecycle tests, monotonic replay,
explicit test-only labels, and a production-build guard.

## Task 2 — ISA-101-style live HMI (sol-medium, parallel lane B)

Own `ui/src/**`, `ui/src/styles.css`, and component tests only. Replace the Phase 2 static loader
with a typed runner-contract client and source picker for upload, V4L2 and RTSP. Render a calm grey,
high-contrast, sentence-case HMI: 48 px controls, large values, physical tray-order vial strip,
neutral keep/check, amber insufficient/triangle, red discard/alarm shape, and per-discard
acknowledgement banner. Default to plain operator language; open Lab, Delta E, tracker volume,
per-model votes/average, model sources, measured/inferred tags and evidence IDs on tap. Preserve
dark theme and the empty top-down locator slot. No charts, gradients, card soup, precomputed
loader, or mock runtime fallback.

Verification: component tests for source creation, ordered event reduction, words/icons/shapes,
alarm acknowledgement, engineer disclosure, 48 px controls, tray order, dark theme, source tags,
disconnect/failure and absence of Phase 2 runtime fetches.

## Task 3 — live integration, mechanical wiring and evidence (luna/sol-low)

After Tasks 1–2 are reviewed, wire the one base-URL setting and test commands. Run browser coverage
against the test-only contract server. Once the lead service is available, run a Playwright E2E
that uploads an authorized V3/V4 `three-pour.avi`, connects to that real service, and proves
measuring, reference-wait, model/decision, average, and explanation events appear from the current
video. It must fail rather than substitute fixtures if the real service cannot decide. Add
camera/RTSP request validation without requiring hardware. Capture fresh stable 1920x1080 light
and dark screenshots and update `ui-PROGRESS.md` with commands, hashes, limits, and data access.
Update `ui-PROGRESS.md` with pins, hashes, GPU telemetry, every data access, measured commands,
limits and any unresolved `@salasl DECISION NEEDED` item.

Verification: contract tests, full Vitest/typecheck/build, real pipeline-service Playwright when
available, screenshot dimensions/inspection, source immutability hash, docs checker,
`git diff --check`, and clean status.

## Final review

Run task-scoped reviews, then one whole-branch review on sol-high. Do not claim Phase 3 complete
unless the real V3/V4 live E2E produces current-session decisions through the lead's service.


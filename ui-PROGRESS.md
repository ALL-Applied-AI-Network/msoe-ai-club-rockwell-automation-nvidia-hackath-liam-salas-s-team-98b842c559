# Phase 1 UI progress

Phase 1 is at evidence-backed review/readiness for the frozen synthetic operator page: the local mock, adapter, accessible dashboard, deterministic progressive cues, evidence drawer, and browser smoke flow are runnable together. This is a handoff, not a plant-readiness claim.

Authorized pinned inputs (received by `scripts/generate_mock_run.py` as required CLI arguments): WP1 batch `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/transport-01a18128/production-v3-c2d740d-c5z7scout-20261029/wp1-batch-01`, manifest `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/transport-01a18128/production-v3-c2d740d-c5z7scout-20261029/wp1-batch-01/simple-production-manifest.json`, SHA-256 `90418a6e9a98d456de29bcebe497f5e0db3ab2309c6942f428318534dda14a76`, and receipt `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/transport-01a18128/production-v3-c2d740d-c5z7scout-20261029/wp1-batch-01/preexec-receipt.json`, SHA-256 `d55f9490283569b9e50c29514cfa7182d673fb3d74f5f88db6a3d25ab6e1d288`; WP5 measurements `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp5-predicted-measurements-01a18128/predicted-mask-measured-evidence-bridge-audited.json`, SHA-256 `862129e99fceaade6a9f44e74e29267b24366e8ed0fb3406002d2ef1a5e52b14`; WP5 locators `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp5-predicted-mask-materialization-production-7e43406-01a18128/prediction-locators.json`, SHA-256 `3a10c45bc67718ab98270a97e4ecd043265695fa77f5e1fe0ebd9da23a0d19f4`; WP6 directory `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp6-limited-all-abstention-wp6la-01a18128-20261029`: `anomalythink-score.json` `a0dfd8aa795f1c6e3e6aad4b7b5b3c3a12f8ec956a1e976f8fdec31bbea52be3`, `cosmos-score.json` `3b986e50d1ec0a7140c9ecff54fee5f94b256234a715876e225625e8b70fd2d9`, `qwen-score.json` `3cd9ccd37af3453063f4aed803bc0c35ee3e2db5832ad10ef6c107a6df9040f9`, `template-score.json` `ce036c2af554c201d45f2a600bdde6e301f9566c76e7232324f04553722c5843`.

Derived UI artifacts: schema `ui-data/schema.json` SHA-256 `191fa16e3c20c87d87241dae06310886edae5417da396552e5bac497a58065d8`; mock JSON `18af158570bd6f35fbbe3a7f2eb43c473d2e9b785bd62e470dada8596414f5d8`; composite MP4 `099f893c5e9721b2f0e80de799a77b4b048ccd3d6c663a97647fb6538d5cb2cc`; adapter/dashboard are derived code.

Measured: 80 Vitest tests pass; Playwright reports 2 passed (main flow plus narrow viewport); six PNGs are each 1920×1080; both vial images decode with non-zero natural dimensions in Chromium; the committed MP4 fully decodes through all 490 frames with repository `ffmpeg-static`; `npm run typecheck` and `npm run build` pass. Pending vials hide decision-bearing values until their cue; top-down waiting/feed states, explanation-source labels, history citations, decision invariants, provenance access, and polite reveal announcements have focused tests. Documented: WP6 rows are 16/16, 16/16, 0/16, 0/16 with its exact disclaimer. Inferred: browser state transitions are driven from mock cue times and are not physical timing validation.

Implemented links: [`adapter.ts`](ui/src/data/adapter.ts), [`mock-run.json`](ui-data/mock-run.json), [`OperatorDashboard.tsx`](ui/src/components/OperatorDashboard.tsx), [`operator-dashboard.spec.ts`](ui/e2e/operator-dashboard.spec.ts), and screenshots [`01-live-run.png`](ui/test-results/screenshots/01-live-run.png), [`02-vial-detail.png`](ui/test-results/screenshots/02-vial-detail.png), [`03-discard-explanations.png`](ui/test-results/screenshots/03-discard-explanations.png), [`04-history-panel.png`](ui/test-results/screenshots/04-history-panel.png), [`05-model-pane.png`](ui/test-results/screenshots/05-model-pane.png), [`06-light-theme.png`](ui/test-results/screenshots/06-light-theme.png).

Commands (paste from cwd `/home/hackathon5/liamreporoot/frontend-ui-20261030`): `python3 scripts/check_agent_docs.py`. Commands (paste from cwd `/home/hackathon5/liamreporoot/frontend-ui-20261030/ui`): `PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm test`; `PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm run typecheck`; `PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm run build`; `PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers npm run e2e`; `file test-results/screenshots/*.png`; `PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH node -e "import ffmpeg from 'ffmpeg-static'; console.log(ffmpeg)"`. The last command resolves repository ffmpeg-static; Chromium 156.0.8078.4 also verified media readiness and non-zero image natural dimensions.

Limits: all decisions, thresholds, top-down geometry, timing, and media are synthetic/mock. WP5 is measurements-only; there is no plant or physical-accuracy claim. WP6 is limited all-abstention evidence with no accuracy claim. Cosmos numeric evidence remains separate from task-context/fence diagnosis. No pixel-perfect mask registration claim is made.

Phase 2 handoff: replace the static input at the app boundary through `parseDemoData` while retaining the UI contract/component surface. Preserve the top-down hook `vial_id ↔ bbox/position ↔ decision/Delta E` and its pending-state privacy behavior.

## Phase 2 Task 4 — development E2E and handoff (2026-10-31)

Liam's ruling is applied: real/source-backed data only for development, demo, and any
future final view; mock data remains available only to component/contract tests. The
approved development fixture has six runs and 18 vials, all 18 decisions measured as
`insufficient_evidence` (0 keep, 0 discard). The demo therefore shows abstentions and
does not invent a discard animation. This is derived development presentation data, not
plant evidence or a final WP7 run. The final WP7 output remains unpublished and its browser
behavior is unverified.

The sole default switch is `ui-data/source.json` → `{"source":"development"}`. The
materializer is the single preparation boundary: `cd ui && npm run prepare:data` writes
validated `ui-data/active-run.json`; `npm run dev`, `npm run build`, `npm run preview`, and
the Playwright server prepare through it without runtime source edits. The development
adapter uses an immutable, read-only schema-v3 SQLite connection and denies `sim_metadata`;
the actual final adapter is read-only and awaits a published WP7 contract.

Measured verification: Python adapter/fixture tests and Vitest pass; typecheck and
production build pass; development operator, development `?demo=1` recording, and narrow
Playwright tests pass 3/3 at the configured 1920×1080 viewport. The operator flow seeks
15.1 s and 269.9 s cue boundaries; the recording flow seeks 30.1 s and 269.9 s, avoiding
a real 270-second wait while asserting progressive selection and 18/18 abstentions.
Fresh screenshots are [`01-development-operator.png`](ui/test-results/screenshots/01-development-operator.png)
and [`02-development-recording.png`](ui/test-results/screenshots/02-development-recording.png),
both exactly 1920×1080. Visual inspection found a stable source-backed reel frame, visible
Development data label, pending operator state in the normal view, and a stable
`insufficient_evidence` detail with unavailable reference/Delta E/threshold in recording
mode; neither shows loading/error/transient controls or mock data.

Development hashes: source selector `cbfbcbc59941543862c6ba9486c079d313d4d37b44fe6a292dffda8d72a41590`;
development-records `c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9`;
fixture SQLite `f989bf2fe2b31cc04b6984f58f46254a97631e09c9c0a7b1ebdacba240282552`;
derived reel `ec609e6a200991921a25784646aa0b0bad0c4b9c7d1c3680a391f5e247b3b07f`.
The manifest retains the approved upstream database hash
`88a910ef5780fb6608f3f0911be1685b84bb2d6ef75f200dd6f5160d016455d7` and records hash
`c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9`; source inputs were
rehash-checked unchanged. No push or PR was made.

Final-switch contract update: a selected final directory supports only top-level
`evidence.sqlite3`, one top-level `wp7-final-decisions.json`/JSONL/`decisions/*.json` form,
and zero or one top-level `*.mp4`; nested core-input/media decoys are not traversed. With no
MP4, the materializer deterministically derives a clearly labelled presentation reel from
joined, hash-verified target RGB frames using the package-pinned ffmpeg executable and writes
it only to browser output assets. Its one-second cues and duration are presentation values,
not measured plant timing. Temporary final-format tests cover full decode, deterministic
hashes, source immutability, schema validation, ambiguity/symlink rejection, and unavailable
ffmpeg failure. This remains contract-tested behavior only: no real final WP7 publication has
been inspected, and final browser behavior remains unverified against one.

## Phase 3 live pipeline — decision resolved (2026-10-31)

Pinned candidate `7bf35aeea21c36acf1a390661bd05fea2b07b190` on
`codex/wp7-vlm-ensemble-01a18128` contains byte-identical tracker/runtime modules from tracker
head `4070be2f73b725e19fc2ea84ee9e4991b089d1e1`, so one commit can bind both lanes. Read-only
inspection found no public incremental session runner for a video file, V4L2 device, or RTSP
source. `track_fill_step` accepts already measured frame evidence plus recipe/calibration state;
`FrozenCNNPredictedMaskRuntime.analyze` accepts complete video bytes, a prebuilt capture receipt,
and a recipe sequence and invokes the accepted CNN on CPU; `build_vial1_bundle` requires a closed
three-vial cohort and three tracker receipts before sequential VLM inference.

The original `@salasl DECISION NEEDED` request was: expose one pinned public live boundary that accepts decoded RGB frames
or a camera/video source and operator-supplied run context, then emits completed-vial tracker,
CNN-mask, CIELAB, deterministic-decision, VLM-vote, averaged-decision, and cited-explanation
records incrementally. Please define the required recipe/cohort/reference inputs for a source that
contains only video, and expose a reviewed GPU device option for the accepted CNN rather than the
current forced `device="cpu"`. Liam resolved ownership: the lead will build that boundary on the
pipeline side. Until it exists, the UI can implement and test only the contract, client, HMI, and
test-only transport; it cannot honestly claim the operator path runs live or that a decision
exists at each vial boundary. No precomputed Phase 2 JSON will be used as a substitute.

### Phase 3 scope correction (2026-10-31)

Liam replaced the earlier direction to compose a session runner in this worktree. The incremental
runner belongs to the pipeline-side service and will be implemented by the lead. This UI worktree
will contain no production capture, CNN, measurement, tracker, deterministic-decision, VLM, or
aggregation runner. Its boundary is the versioned contract in `ui/docs/runner-api.md` and
`ui/docs/runner-event.schema.json`, plus one `VITE_RUNNER_BASE_URL` service-origin setting.

The local contract server is test-only: it may exercise event parsing and HMI state transitions,
but it is not inference, is visibly labelled, cannot be a production fallback, and cannot satisfy
the real-data acceptance gate. Final Phase 3 acceptance still requires a Playwright run that sends
an authorized V3/V4 twin video to the lead's real service and observes current-session measuring,
reference-wait, decision/vote/average, and explanation events in the UI. Until that service is
available, this work may be contract- and component-complete but not Phase-3-complete.

Follow-up request retained for the lead: the pipeline service should expose a reviewed CNN
`device=` option and incremental `run_session` boundary, report the pinned source commit and
processing rate/health, and apply the causal rule that vial 1 posts after reference vials 2 and 3
close. No real-footage input has been accessed in this Phase 3 UI work.

### Development review rule (2026-11-01)

Liam waived independent review for development runs and iterations across the VLM, runner-service,
tracker, and UI lanes. While the runner contract and HMI are still being proven, focused tests plus
the frontend lead's own check are sufficient. Independent review becomes mandatory only after the
real-service V3/V4 path is verified working and before any video-5 or runs 1–4 access, publication
or integration-branch merge, or final-report/submission use. No push is authorized.

### Pipeline attestation pin (2026-11-01)

Liam replaced the earlier candidate pin with reviewed integration commit
`cf6829b99486b41f4190874dbd701ad812648217` (`Bind WP7 VLM generation configuration`). The
expected value is moving into one canonical UI JSON config value and will change once more to the
final integration/onion commit before recording. `/api/health` remains the service attestation:
the HMI must show that reported commit, compare health/session/event commits against the expected
value, and warn and fail closed on any mismatch. It must never silently accept drift.

## Phase 3 Task 3 — live HMI mechanical integration (2026-11-01)

Removed all Phase 2 `prepare:data` calls and Vite `publicDir` exposure. Playwright launches the
explicit TEST-ONLY contract simulator and uploads only an undecoded dummy fixture. The flow covers
event stages, alarm acknowledgement, locator, engineer detail, test-only label and commit.
Stable 1920×1080 screenshots are recorded under `ui/test-results/screenshots/phase3-test-only-*`.

The separate `real-runner.spec.ts` requires explicit `REAL_RUNNER_BASE_URL` and `REAL_TWIN_VIDEO`;
missing values fail clearly with no fixture fallback. No video was accessed. Read-only inspection
of `/home/hackathon5/liamreporoot/pipeline-runner-service-01a18128` at clean `59d674f` found only
the documented `your_deployment.runner:create_adapter` placeholder and no reviewed Varied V3/V4
deployment factory/start command.

@salasl DECISION NEEDED: provide the reviewed Varied V3/V4 deployment factory and exact start
command before running the authorized real E2E. This is contract-complete, not real-service complete.

Task 3 lead-check correction: default `npm run e2e` now selects only the TEST-ONLY project;
`npm run e2e:real` requires both real-runner environment variables before launching the separate
real config. The real spec waits for healthy service verification and a canonical 40-hex commit.
The E2E now requires the actual terminal lifecycle (`Run completed`) and asserts no error or
reconnect text on the unmodified DOM. The current simulator branch still reports sessions as
`running` and closes its normal SSE before the contract-lane terminal update; screenshots are
therefore blocked pending that contract change. No production UI code was changed.

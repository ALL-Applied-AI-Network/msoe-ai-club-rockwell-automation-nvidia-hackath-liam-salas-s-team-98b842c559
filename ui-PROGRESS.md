# UI progress

## Phase 4 — ISA-101 four-screen redesign (2026-11-01)

Spec: `overnight-ops-20261028/notes-draft/UI-RESEARCH-AND-PROPOSAL.md` §3–4. Light grey by
default, optional dark theme; colour only for discard (red ✖), can't assess (amber ?),
running (blue dashed) and alarms; ≥48 px touch targets; 1920×1080 target.

**What changed (all under `ui/`).** `src/app/AppShell.tsx` replaces `LiveHmi`: tabs L1 Line
overview, L2 Inspection station (home), L3 Vial detail, L4 Diagnostics (only with the
Engineer toggle), an Alarms (n) drawer, and honesty banners. `src/screens/station` plays the
uploaded file locally with overlays (ROI outline, HUD with tank / stage / mL from event
history, approximate fill tick), the current-vial card with rule/Qwen/Cosmos/AnomalyThink
agreement and the equal-weight average, Agree / Override… (logged, never changes the
decision), tray strip and counts. `src/screens/detail`: stages × vials evidence grid (crops,
Lab swatches, rule ΔE76 with 1.0/2.0 bands), one card per source (vote, confidence,
agreement, model-written text or labelled fallback with the rejected text behind a
disclosure, observations), tracker timeline. `src/screens/overview`: KPIs and a log-scale
ΔE-per-vial trend with bands and the recipe-start marker. `src/screens/diagnostics`: service
and pin, model runtimes, event log, raw JSON per vial, operator-record ledger.
`src/screens/alarms`: ISA-18.2-style list with priority, time, vial and acknowledge state.

**Real vs limited.**
- Real, from the runner API: the event stream, plus `GET /api/sessions/{id}/decisions` for
  rule stage ΔE76 and per-model records (both exist on the live runner).
- The live runner (`10c106f`) on `accepted-v4-train01-cohort.mp4` gives one honest
  abstention: `insufficient_evidence` / `measurement_stage_unavailable`, no votes and no
  per-vial evidence. The UI shows it as amber "Can't assess" with a plain reason.
- The rich discard result is the recorded 2344f3c frozen-cohort hybrid session `e7383eef`.
  `ui/tools/replay_runner.py` replays it, along with its evidence bundle and 27 crops. The
  replay runs no model, and the UI marks it with a "Recorded replay, not live" banner.
- Crops, bundle median Lab and the ROI box need the optional endpoints
  `/evidence-bundle` and `/evidence/images/{id}`. Only the replay serves them today, so
  against the live runner the UI says "Crop not exposed by this runner". Adding them to the
  runner (new files in `src/fluidmix/pipeline/service/`) is a follow-up; it wasn't done here.
- No liquid mask or locator geometry is streamed, so both are labelled rather than drawn.
- `physics_cited` is parsed but empty: "Physics fields not provided yet (VLM schema v2
  pending)". The model text, guard reasons and fallbacks come from today's decision records.
- Operator agree/override records live in browser localStorage only, and the UI says so.
- The pin stays `2344f3c`. Starting against `10c106f` needs `VITE_DEV_ALLOWED_COMMITS` and
  shows an "Unpinned development service" banner.
- Bug fixed: the client called `fetch` unbound (Illegal invocation in browsers). The old
  e2e test hid this with an init script.
- The dev video has its own burnt-in caption ("V1 KEEP"); the system result is discard.
  That disagreement is in the twin data, not the UI.

**Independent review (one pass, opus) and what was done.** Fixed:
- The Detail screen crashed when the average abstained.
- The rule's discard alarm stayed up after the average kept the vial.
- Error and service alarms aged out of the capped logs.
- The override form could record against a newly selected vial.
- The dev pin override applied in production builds. It is now dev-only, attaching a
  session needs verified health, and the session commit must equal the health commit.
- A stale decisions fetch could overwrite a newer one.
- The station explanation is now attributed to its model and that model's vote.
- A "Provisional: rule only" label shows while votes are pending.
- Replay crash paths are closed, and receipts are released in event order.
- EventSource reconnect failed: it sent `Last-Event-ID` next to `after=0`, which the runner
  rejects. The client now closes the stream and resubscribes from the last applied sequence.

Not fixed:
- Sessions with more than one cohort map vials by `tray_index` and use a single bundle.
  The dev video has one cohort.
- The alarm drawer has no focus trap.
- `getHealth` rejects an `unhealthy` status.

**Measured.** 60 Vitest tests pass; `npm run typecheck` and `npm run build` pass;
Playwright `e2e` (test-only simulator, all screens + drawer) 1/1; `e2e:screens` 1/1 against
the live runner and 1/1 against the replay. Screenshots (light and dark, 1920×1080):
`ui/test-results/screenshots/redesign-{live-10c106f,replay-2344f3c}-{L1-overview,L2-station,L3-detail,L4-diagnostics,alarms}-{light,dark}-1920x1080.png`
and `redesign-test-only-{light,dark}-1920x1080.png`.

**Run** (from `ui/`, `export PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH`; `npm ci` once):

    npm test && npm run typecheck && npm run build
    PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers npm run e2e
    # live runner on :8765 (must admit origin http://127.0.0.1:5173)
    VITE_RUNNER_BASE_URL=http://127.0.0.1:8765 VITE_DEV_ALLOWED_COMMITS=10c106fbd38e3da34d902abb7d920c5a4041440e npm run dev -- --host 127.0.0.1 --port 5173
    # recorded replay (development), from the repository root:
    O=/home/hackathon5/liamreporoot/outputs/pipeline-20261028
    python3 ui/tools/replay_runner.py --events $O/runner-production-factory-01a18128/free-explanations-e2e-20261101T173805Z/events.sse \
      --decisions $O/runner-production-factory-01a18128/free-explanations-e2e-20261101T173805Z/decisions.json \
      --bundle $O/wp7-vlm-dev-bundle-materializer-01a18128-attempt15/bundle.json --port 8775
    VITE_RUNNER_BASE_URL=http://127.0.0.1:8775 npm run dev -- --host 127.0.0.1 --port 5174
    # screenshots: SCREENS_RUNNER_BASE_URL=... SCREENS_LABEL=... SCREENS_VIDEO=<dev video> npm run e2e:screens

Older phase notes follow; their commands for `LiveHmi`, `e2e:real` and the Phase 3 spec are superseded.

## Phase 1 UI progress

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
reconnect text on the unmodified DOM. Contract-simulator commit `ce10baf` supplies that terminal
lifecycle; the test passes and screenshots were regenerated without changing production UI code.

### Phase 3 frontend lead verification (2026-11-01)

Fresh measured checks from the clean branch: repository Python tests are 138 passed, 1 skipped;
agent-document checks report 11 skills and 36 targets; Vitest is 29/29; TypeScript and the Vite
production build pass; the explicit TEST-ONLY Playwright project is 1/1 at 1920×1080. The built
`dist/` contains only `index.html` plus hashed JS/CSS and no `active-run.json`, `mock-run.json`,
`source.json`, or Vite `publicDir`. The separate `e2e:real` command fails before server launch
when explicit real inputs are absent and has no test-only fallback.

The reviewed test-only screenshots show the unmodified DOM in dark and light modes, visibly say
`Test-only service` and `Run completed`, and show no connection, pipeline, event-stream, or commit
failure banner. Dark screenshot SHA-256 is
`7c161862a6655089ac35bb138681228ee6399010b34fa8bf828b0ae2c9d21f4d`; light is
`a65394e051f5e14b5b2207c6cbfae80fd433a4f4f12c5fcad750cafe26f64b24`.
The contract hash is `68abaf410fbabb09a2e3da471e6bc6c7aec464a0dae59c64db2459a5f22128f7`,
event schema hash is `3ea09963177d3960e2a49f61958da0a899a5e53cf92adfbe971e902b7b92ccdd`,
and canonical pin-config hash is
`527706b549e5504b3329f3dd277702a9a90028af3e5d6876783c66ee0314982c`.

The lead runner-service worktree advanced cleanly to `460484b45d7b7db63c7b6772eff6d876c37854ae`
(`fix: bind live runner evidence causally`), but read-only inspection still finds no reviewed
non-test `create_adapter` deployment factory or concrete start command. Therefore the authorized
V3/V4 video was not opened, the real-service E2E was not run, Phase 3 is not declared complete,
and independent review remains deferred under Liam's development rule. No runs 1–4, video 5, GPU,
push, merge, publication, or final-submission action occurred.

## Phase 3 Task 4 — real-service integration result (2026-11-01)

Current state: Phase 3 Task 4 is verified on a clean archive snapshot of reviewed service commit
`2344f3c655d9aa255eb576e6d2a280bd0fc9bdff`, per Liam's direction not to follow the lead's moving
video-agnostic development branch. The canonical UI pin matches that exact service commit. The HMI
shows the full `/api/health` attestation and still fails closed on any health, session or event
mismatch. The archive snapshot and build attestation are under ignored
`ui/test-results/runtime/`; no service source, attestation or session artifact is committed.

Measured working-path evidence from service commit
`2344f3c655d9aa255eb576e6d2a280bd0fc9bdff`: Playwright uploaded the documented
`accepted-v4-train01-cohort.mp4` through the HMI. Session
`bcfe8957bc904ad58dba950b07bd8f42` closed all three vials and emitted three admitted VLM votes:
Qwen discard 0.90, Cosmos discard 0.80 and AnomalyThink discard 0.90. The equal-weight aggregate,
including the admitted deterministic 1.00 source, was discard with mean 0.90. Qwen's explanation
was model-written; Cosmos and AnomalyThink emitted explicitly labelled deterministic fallbacks.
The service event range 1041–1050 and `/decisions` supplied these measurements. This run began at
2026-11-01T18:24:52Z and ended at 18:43:24Z. Its first browser test failed only because it waited
for a transient `Measuring` label after the HMI had already advanced to the discard alarm. No
screenshots from that failed test are accepted as final evidence.

Lead-disclosed provenance, recorded without reinterpretation: the `2344f3c…` 0.90/discard result
is a **development-data frozen-cohort hybrid**, using H.264 tracking with pinned native-MJPEG stage
measurements. It is superseded and is not represented as a single-source live result. The HMI now
shows the exact visible banner “Development data, frozen-cohort hybrid, service 2344f3c” whenever a
non-test service attests that commit, including both accepted screenshots and the abstention view.

Frontend fixes derived from that observation preserve every explanation event instead of replacing
Qwen's explanation with the final deterministic template, visibly label model-written and fallback
producers, show admitted status and model source for every vote, format the equal-weight average as
0.90, and translate `tracker_source_profile_unavailable` to “Can't assess this video: camera setup
not recognised” while retaining the technical reason in engineer detail. Focused red/green tests
cover explanation preservation and abstention wording. The strengthened real Playwright suite
requires three admitted discard votes, their sources, the 0.90 average, alarm acknowledgement,
engineer detail, three-position tray, dark/light 1920×1080 screenshots and a separate honest
abstention. It fails early rather than accepting a pipeline error or wrong disposition.

Measured regression on clean service commit
`06b86217731beee4880d3e1533511af48aba7e8f`: the HMI upload of the same accepted V4 bytes consumed
the source but terminated after about 90 seconds as `insufficient_evidence`, with no alarm or VLM
votes. A bounded reproduction using the exact documented run context created session
`029b70af1ca4422c800d15c58f375f47`, consumed all 1,011 frames, and emitted
`incomplete_frozen_profile` at event 1012. An in-memory unrecognized `.mp4` performed no GPU
inference but reached `source_runtime_failed`/“moov atom not found” and `Run failed` instead of the
documented pre-media `tracker_source_profile_unavailable` abstention. The frontend did not relabel
either result. The service was stopped after each bounded check and the GPU was released; no other
process was killed or modified.

Two clean lead follow-ups were tested as they landed. Commit
`69284bb63b90efa341a1d20cc96ff9bfb8ccaf3f` (`freeze empty reference before stream onset`) and
commit `31f67995d30903854c1af8e304a9d6f375f1bf6e` (`close vials on topology-cleaned seams`) both
still ended the accepted V4 browser session as `insufficient_evidence` / `Run stopped` before any
VLM load. The HMI showed one unavailable tray position and no discard alarm. The real E2E's early
terminal guard now reports this outcome promptly instead of waiting 25 minutes.

Development-footage access log for this task: accepted V4 was opened eleven times. The first five
are the earlier working `2344f3c…`, regressed `06b8621…` HMI and curl reproduction, and the
`69284bb…`/`31f6799…` HMI checks documented above. Three subsequent operator-style uploads from the
clean archived `2344f3c…` snapshot exposed a brittle icon-bearing text locator, an over-tall
engineer layout, then produced the first passing screenshots. Three final-review/fix uploads from
that same clean snapshot verified the strengthened assertions, exposed the vote-row clipping, then
produced the corrected captures below. The last accepted run passed in 17.9 minutes. A separate
browser-launch attempt failed before session creation and did not open the video. No runs 1–4,
video 5, WP1 test data, `grounding_qualification`, Brev, or protected footage was accessed. The
lead's live worktree and its uncommitted edits were not used or modified for final verification.

Final operator formatting and identity checks are explicit. Standard CIELAB is shown to one
decimal, tracker volume to one decimal mL, processing rate to one decimal frames/s, and an exposed
Delta E to two decimals. Read-only inspection of the pinned implementation found that
`vision_measurement.py` converts 8-bit sRGB with `skimage.color.rgb2lab(rgb / 255)` and records
`cie_lab_d65_2deg`; therefore the observed `L*=0.862…` is a native standard-CIELAB L* value on the
0–100 scale (very dark), not a normalized 0–1 value. The UI reports it as `L* 0.9` and does not
rescale or guess. The public `runner-event.v1` measurement emitted by `2344f3c…` has `delta_e: null`:
the pinned private deterministic receipt carries per-stage peer Delta-E76 values, but the manager
does not expose them. Engineer detail therefore says which tank/fill step lacks public Delta E and
that the runner event does not expose stage ΔE; it does not contradict the model explanation by
calling all Delta E unavailable. Exposing the stage distances remains a pipeline-schema follow-up.

Tray positions and operator vial labels are one-based (`1`/`Vial 1` through `3`/`Vial 3`); opaque
`CASE_…` IDs moved to engineer detail. Once the public terminal state arrives, reference vials 2 and
3 render “Not assessed (run stopped)” rather than “In progress”. The primary explanation retains
“Model-written · Qwen” plus the inferred provenance tag.

The required sol-high post-success review was performed once. Its concrete findings were resolved:
the real test now distinguishes admitted from “Not admitted”, asserts that abstention has no votes,
average or explanation, checks one-based tray order, validates screenshot evidence through clipping
ancestors, and uses fresh hashes. Final measured verification is 138 Python tests passed with one
skipped; synthetic absorbance smoke and 11-skill/36-target document checks pass; Vitest is 32/32;
TypeScript and the Vite production build pass; TEST-ONLY Playwright is 1/1; real-service Playwright
is 2/2 (accepted V4 plus the no-media-inference abstention). The real path asserted the alarm and
Acknowledge action, three ordered tray positions, three admitted VLM votes and model sources,
equal-weight 0.90, Qwen model-written text, two labelled deterministic fallbacks, Lab/volume/Delta-E
presentation, stopped reference states, the hybrid banner, and no failure banner. The unknown
in-memory upload rendered `insufficient_evidence` / `tracker_source_profile_unavailable` and
“Can't assess this video: camera setup not recognised”; it did not invoke the ensemble.

Fresh real-service screenshots are
[`phase3-real-development-2344f3c-dark-1920x1080.png`](ui/test-results/screenshots/phase3-real-development-2344f3c-dark-1920x1080.png)
SHA-256 `0e864f78f54b858166a9c105b489199e2d27b1709a9cce792919caa38b168c80` and
[`phase3-real-development-2344f3c-light-1920x1080.png`](ui/test-results/screenshots/phase3-real-development-2344f3c-light-1920x1080.png)
SHA-256 `e8a54d71dda9a02215837637f4baaafc9b7941fdd8e813cad34c44f73ebe0b59`.
Both are exactly 1920×1080 and visibly show the frozen-cohort-hybrid development banner, full
service commit, 1.0 frames/s, discard alarm with Acknowledge, Qwen provenance, all three vote rows,
formatted engineer values, and one-based physical tray order with terminal reference states. The
canonical pin-config SHA-256 is
`9096065b3f92a3110361e6ed32dfa4439ed4c08c79429228077e7bb35b2b5265`.

Lead follow-up observation: service `2344f3c…` completes the accepted inference internally but its
public session state is `stopped`, so the HMI honestly shows `Run stopped` after the successful
decision. This is not treated as a browser or inference failure because the required terminal
result events are complete and no failure event/banner exists. The pipeline service should expose
`completed` distinctly from an operator stop in a later reviewed commit.

The earlier moving-branch regressions remain useful lead evidence but are not a frontend blocker
under Liam's explicit clean-2344 direction. No push is authorized.

### Exact local start commands

First confirm `nvidia-smi` reports no other process. Then start the clean pinned runner in one
terminal with the exact environment used for the passing real-service run:

```bash
cd /home/hackathon5/liamreporoot/frontend-ui-20261030/ui/test-results/runtime/runner-2344f3c-snapshot
env PATH=/home/hackathon5/liamreporoot/frontend-ui-20261030/ui/test-results/runtime/runner-2344f3c-snapshot:/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin:/tmp/node-v22.14.0-linux-arm64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/hackathon5/liamreporoot/frontend-ui-20261030/ui/test-results/runtime/runner-2344f3c-snapshot/src:/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.vlm-overlay \
  CPATH=/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/root/usr/include/python3.12:/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/root/usr/include/aarch64-linux-gnu/python3.12 \
  TRITON_CACHE_DIR=/home/hackathon5/liamreporoot/wp7-vlm-reasoning-grounded-liam-01a18128/.python-dev-overlay/triton-cache \
  FLUIDMIX_GPU_FREE=1 \
  ./runner serve --build-attestation /home/hackathon5/liamreporoot/frontend-ui-20261030/ui/test-results/runtime/runner-build-attestation.json
```

Start the operator UI against that runner in a second terminal:

```bash
cd /home/hackathon5/liamreporoot/frontend-ui-20261030/ui
env PATH=/tmp/node-v22.14.0-linux-arm64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  VITE_RUNNER_BASE_URL=http://127.0.0.1:8765 \
  npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. The ignored snapshot and attestation paths above are local runtime
handoff artifacts; neither is committed.

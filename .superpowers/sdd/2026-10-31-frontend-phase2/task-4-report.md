# Task 4 report — development E2E, screenshots, and handoff

Date: 2026-10-31  
Base: `484fdd541b2e53b76fa6d3ee4618d421a27844de`

## RED/GREEN evidence

The first browser run used the new development assertions and failed before the approved
browser cache was selected (`Executable doesn't exist ... chromium_headless_shell`), then
failed on two strict selectors and cue/count assumptions. These were test-boundary failures,
not application data failures. The implementation was corrected to use the exact H1/slot
selectors, DOM `muted` property, and cue-derived counts. Final command and complete output:

```text
cd ui
PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers npm run e2e
> fluidmix-operator-demo@0.0.0 e2e
> npm run prepare:data && playwright test
> /home/hackathon5/liamreporoot/frontend-ui-20261030/ui-data/active-run.json
Running 3 tests using 1 worker
✓ 1 e2e/operator-dashboard.spec.ts:14:1 › development operator view stays honest across cue boundaries (878ms)
✓ 2 e2e/operator-dashboard.spec.ts:41:1 › development recording mode autoplays and remains all abstentions (751ms)
✓ 3 e2e/operator-dashboard.spec.ts:56:1 › narrow layout remains reachable (414ms)
3 passed (3.0s)
```

The normal flow seeks 15.1 s and 269.9 s; the recording flow seeks 30.1 s and 269.9 s.
Assertions cover progressive reveal, 18 timeline slots, 18/18 `insufficient_evidence`, zero
keep/discard chips, muted autoplay, and no real-time 270-second wait.

## Verification

```text
.venv/bin/python -m pytest tests/test_ui_materializer.py -q
25 passed in 2.55s

cd ui && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm test
Test Files 6 passed (6)
Tests 88 passed (88)

PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm run typecheck
tsc --noEmit (exit 0)

PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH npm run build
vite v8.3.5 ... 91 modules transformed ... built in 119ms (exit 0)

cd .. && python3 scripts/check_agent_docs.py
shared pointer, 11 skills and 36 document file-target checks passed

cd ui && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH node ...ffmpeg-static... development-reel.mp4 -f null -
full H.264 decode: PASS

git diff --check
(no output; exit 0)
```

The first system-Python pytest attempt was not a repository failure (`No module named
pytest`); rerunning with the repository `.venv` produced the 25 passing tests above.

## Screenshots and visual inspection

Final tracked captures, both verified with `file` as `PNG image data, 1920 x 1080`, are:

- `ui/test-results/screenshots/01-development-operator.png`: normal operator view at a
  stable pending cue, source-backed reel visible, Development data label, and no transient
  loading/error state.
- `ui/test-results/screenshots/02-development-recording.png`: clean `?demo=1` view at a
  stable revealed cue, source-backed vial image, `insufficient_evidence`, unavailable
  reference/Delta E/threshold, and no mock discard/keep.

I inspected both locally with the image viewer. They show the intended 1920×1080 console,
source-backed media, and honest abstention/pending states; no loading screen, error, or
transient media-control capture was retained. The six prior mock screenshots were removed.

## Provenance and hashes

The final selector and derived inputs hash as follows:

```text
cbfbcbc59941543862c6ba9486c079d313d4d37b44fe6a292dffda8d72a41590  ui-data/source.json
c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9  ui-data/development/development-records.json
f989bf2fe2b31cc04b6984f58f46254a97631e09c9c0a7b1ebdacba240282552  ui-data/development/evidence.sqlite3
ec609e6a200991921a25784646aa0b0bad0c4b9c7d1c3680a391f5e247b3b07f  ui-data/development/development-reel.mp4
```

The manifest retains approved upstream hashes `88a910ef5780fb6608f3f0911be1685b84bb2d6ef75f200dd6f5160d016455d7`
(SQLite) and `c8e94e8fa19ea801cdc211c2e2a266377fffe7ff468b0ff0a25908e52bfd6ed9`
(records); rehash checks found no source mutation. The materializer validates the schema,
uses immutable read-only SQLite, and denies `sim_metadata`.

## Self-review and boundaries

Package preparation now has one source-selection implementation. Mock remains test-only.
Development is synthetic development evidence and a derived presentation reel, not plant,
physical, final WP7, or operating advice. No sealed data, WP1 test/video-5, grounding
qualification, oracle, or simulator truth was inspected. There is no published final WP7
decision file, so final behavior and any future discard recording remain unverified. No
push or PR was made.

## Fix Round 1/5 — review findings (2026-10-31)

RED gap: the original recording test asserted `18 / 18 revealed` and zero discard chips,
but did not assert the exact final abstention count or zero keep chips. The normal flow had
that count assertion; the recording flow did not. GREEN: the recording flow now asserts
exactly 18 `.vial-strip .decision-chip--insufficient_evidence`, zero keep, and zero discard
after the final 269.9-second cue.

The normal screenshot was recaptured after moving the pointer away, waiting 2.5 seconds,
and applying a test-only Chromium shadow-control suppression for the paused capture. This
does not change the app: normal mode still renders native `controls` at runtime. Local image
inspection confirms the final normal capture has no native video control bar, loading/error
state, or transient media state. Both final files remain exactly 1920×1080:

```text
ui/test-results/screenshots/01-development-operator.png: PNG image data, 1920 x 1080, 8-bit/color RGB, non-interlaced
ui/test-results/screenshots/02-development-recording.png: PNG image data, 1920 x 1080, 8-bit/color RGB, non-interlaced
```

Final focused checks:

```text
cd ui && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers npm run e2e
3 passed (5.5s)

.venv/bin/python -m pytest tests/test_ui_materializer.py tests/test_development_fixture.py -q
31 passed in 4.18s

git diff --check
(no output; exit 0)
```

Both screenshots were inspected again with the local image viewer after the final full E2E
run. The normal image shows a stable pending Development data view without controls; the
recording image shows a stable `insufficient_evidence` outcome with unavailable comparison
fields. No mock data or invented keep/discard appears.

## Fix Round 2/5 — assertion placement (2026-10-31)

Review found the Round 1 exact outcome assertions had landed redundantly in the normal test,
while the `?demo=1` recording test still checked only zero discard. The duplicate assertions
were removed from the normal flow, preserving its original exact checks, and the recording
flow now owns all three final-cue assertions: exactly 18 insufficient-evidence chips, zero
keep chips, and zero discard chips.

Focused recording verification:

```text
cd ui && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers npx playwright test -g 'development recording mode'
Running 1 test using 1 worker
✓ 1 e2e/operator-dashboard.spec.ts:47:1 › development recording mode autoplays and remains all abstentions (905ms)
1 passed (2.4s)
```

Full E2E verification:

```text
cd ui && PATH=/tmp/node-v22.14.0-linux-arm64/bin:$PATH PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers npm run e2e
> npm run prepare:data && playwright test
Running 3 tests using 1 worker
✓ 1 e2e/operator-dashboard.spec.ts:14:1 › development operator view stays honest across cue boundaries (3.4s)
✓ 2 e2e/operator-dashboard.spec.ts:47:1 › development recording mode autoplays and remains all abstentions (737ms)
✓ 3 e2e/operator-dashboard.spec.ts:64:1 › narrow layout remains reachable (389ms)
3 passed (5.3s)
```

No screenshots or runtime behavior were changed in this round.

# Fill-level CNN implementation log

Newest entries appear first. Evidence labels follow `AGENTS.md`.

## 2026-10-31 — Step 4: variation and four-class label layer

### Result

- **Source-file observation:** Claude's newest review entry at about 16:52 CT passed
  Step 3 and gave GO for Step 4, then Step 5. It required trying labels in order, a
  geometry-only holder projection, held-out 70/75 mL checks in the pilot and loud Z-up / metre
  assertions. The review file SHA-256 at the start was `2f07ec...6961`.
- Updated `scripts/inspect_fill_level_stage.py` to fail unless the organizer stage is Z-up
  with one metre per unit and to record both facts. **Measured on Brev:** the unchanged stage
  reports `Z` and `1.0`.
- Added `scripts/probe_fill_level_labels.py`. **Measured on Isaac Sim 6.0.1:** after moving
  all labels before render-product and annotator creation, semantic ID 9 covered 9,657 pixels
  and its class tokens were `none,liquid`. This makes Claude's first route work; no legacy
  `SemanticsAPI` or geometric-only class fallback was needed. Instance-ID paths also contained
  the liquid prim, but are retained only as the documented second fallback.
- Added `src/fluidmix/fill_level_variation.py`, `scripts/make_fill_level_specs.py` and
  `assets/fill-level/render-contract.json`. Specs deterministically cover 0-85 mL, colour and
  darkness, about half with 0-10 mm foam, about 60% with a varied barcode sticker, lighting
  and colour temperature +/-20%, a few pixels of camera jitter, +/-2% zoom and occasional wet
  walls. Foam bumpiness controls deterministic surface clumps as well as roughness. Train/
  validation/test seed ranges are disjoint; pilot is a separate non-training role.
- Added `scripts/render_fill_level_dataset.py`. It changes only the anonymous session layer,
  creates labels before the product, applies the matched camera plus per-frame variation, and
  writes one JSONL manifest record per RGB/mask pair. Four visibility passes implement the
  label rules: glass-only vial silhouette; glass-hidden liquid/foam; a full-height opaque
  interior probe for genuinely front-facing carrier/post/nozzle/sticker pixels; and normal
  RGB. Pixels inside the active vial default to empty, pixels outside are ignored, and holder
  geometry visible behind empty glass remains empty.
- Added `scripts/check_fill_level_capture.py` and focused tests. The checker verifies 640x480
  uint8 RGB/masks, hashes, allowed IDs (`empty=0`, `liquid=1`, `foam=2`, `blocked=3`,
  `ignore=255`), ignored borders, conditional classes and duplicate role/seed pairs.
- **Measured final Step 4 capture (attempt 06, seed 14 at 70 mL):** the exact current renderer
  hash `7423cb...5603b` and variation module hash `becc56...218f` produced one non-blank RGB and
  a valid mask on the first try for every pass. Pixel counts were 8,260 empty, 8,340 liquid,
  1,797 foam, 2,814 blocked and 285,989 ignored. The vial bbox was `[277,66,343,442]`.
  The copied manifest passed the offline checker with no errors. Organizer station and camera
  hashes were unchanged.
- **Measured render speed:** the final warm-cache frame used 2.016 s RGB, 0.807 s glass-only,
  1.306 s glass-hidden fluid and 1.188 s blocker-probe, or 5.318 s for all four passes. Isaac
  startup was 26.232 s; total one-frame process time was 38.385 s. The single-frame estimate
  is about 677 frames/hour before batch overhead and is not yet the Step 5 sizing result.
- **Measured holder projection near the nominal camera (seed 8155):** the carrier run at x=272
  was rows 218-274, or 48.09-66.51 mL with the frozen 3.04 rows/mL scale. The real hidden zone
  is 53-69 mL, so the lower/upper boundary differences were -4.91/-2.49 mL, both beyond
  Claude's about-2 mL threshold. The effective oversampling bands now follow the twin:
  40.1-48.1 lower edge, 48.1-66.5 hidden and 66.5-70.5 upper edge. Organizer geometry was not
  moved; the real/twin gap is recorded in the contract for the results note.
- **Measured local checks:** compilation passed; the 22 new focused tests passed; JSON and
  `git diff --check` passed; `scripts/smoke_absorbance.py` passed. The full suite reached
  92 passed / 1 skipped, with the same one fake-ffmpeg failure and six Windows symlink setup
  errors already documented in Step 3.
- Raw RGB, masks, semantic arrays, session layers, runner logs and manifests remain under
  ignored `outputs/`; Git ignore checks named the `outputs/` rule for the final files. The
  existing Brev instance remains running for the immediately following Step 5 pilot.

### Commands run

```text
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -Raw
wsl -e bash -lc "brev ls --json"

# Ordered label probe and stage-unit assertion
wsl -e bash -lc "brev copy <probe/config/runner> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step4-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<copy versioned inputs into vscode and launch attempt 02 detached>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect-v2> --stage <unchanged station> --camera-json <unchanged FillCam> --frame 686'"

# Variation/label captures; every attempt had a fresh output, log and launch lock
wsl -e bash -lc "brev copy <renderer/module/spec/runner> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step4-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker cp versioned inputs into vscode; launch attempts 03 through 06 detached>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll status and isolated render.log>'"

# Final attempt 06 command inside vscode
timeout --signal=TERM --kill-after=60s 1200s \
  env OMNI_KIT_ACCEPT_EULA=YES \
      PYTHONPATH=/tmp/fill-level-cnn-step4-20261031-codex-01/input/python-v2 \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step4-20261031-codex-01/input/render_fill_level_dataset-v3.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --match /tmp/fill-level-cnn-step4-20261031-codex-01/input/camera-match-v1.json \
  --specs /tmp/fill-level-cnn-step4-20261031-codex-01/input/step4-spec-v1.json \
  --out /tmp/fill-level-cnn-step4-20261031-codex-01/attempt-06/output \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --frame 686 --subframes 32 --capture-timeout-seconds 600 --cache-state warm

wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step4-20261031-codex-01/collected/attempt-06/ <ignored local attempt-06 directory>"
.venv\Scripts\python.exe scripts\check_fill_level_capture.py --manifest <attempt-06 manifest.jsonl>
.venv\Scripts\python.exe -m py_compile <new and changed Python files>
.venv\Scripts\python.exe -m pytest tests\test_fill_level_variation.py tests\test_fill_level_capture.py -q
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\smoke_absorbance.py
git check-ignore -v <final RGB> <final mask> <session layer>
git diff --check
```

### Problems

- The first inline detached-launch command was misquoted across PowerShell, WSL and the
  remote shell. It created only an attempt-01 launch lock and ran no Isaac process. A
  versioned host launcher made attempt 02 retry-safe; nothing was deleted or overwritten.
- The first probe's strict equality check did not recognize the merged semantic token string
  `none,liquid`, so its summary selected the instance-ID fallback even though the raw mapping
  already proved semantic liquid pixels existed. Token parsing was corrected; the final
  renderer then used and validated the first, semantic route.
- The twin holder is lower in the matched view than the real holder. The measured difference
  crossed Claude's threshold, so sampling bands were shifted to the twin's actual difficult
  zones instead of changing organizer geometry.
- Independent pre-commit review found that the first lighting implementation made higher
  recorded colour temperature warmer and that foam bumpiness affected only roughness/radius.
  The RGB light channels were corrected, deterministic surface clumps were added, and the
  exact revised renderer was rerun successfully as attempt 06.
- The full Windows test suite still lacks symlink privilege and cannot launch its POSIX-style
  fake ffmpeg. These failures are pre-existing and outside the fill-level files; all new tests
  and the focused smoke pass.

### Questions for Claude

- Does the four-pass composition, especially the opaque interior-probe test for front versus
  behind-glass holder pixels, satisfy the Step 4 label rules?
- Is the recorded twin-zone shift sufficient for the later results note, or should Step 5
  report both real and twin zones on every pilot summary as well?

## 2026-10-31 — Step 3: FillCam camera match — checkpoint

### Result

- **Source-file observation:** Claude's newest review entry at about 15:40 CT passed
  Step 2 and gave GO for Step 3. It required a session/override layer, reuse of the warm
  cache if it survived, exact 30 mL and 80 mL probes, a time-bounded capture loop, unique
  retry-safe remote runs and no organizer-file edits.
- **Measured:** the first normal Brev start reached `FAILURE / NOT READY`. The second and
  final allowed normal start succeeded; the instance reached `RUNNING / READY`. No reset or
  replacement instance was used.
- **Measured after restart:** host `/tmp` was cleared, but the persistent `vscode`
  container still held the Step 2 twin and 446 MB private Kit root, including its cache.
  The twin inputs were copied without deleting the surviving copy to
  `/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input/twin`. Camera and station hashes
  remained `11966571...` and `5f58540c...`.
- Added `scripts/measure_fill_level_camera_targets.py`, which reads real reference images
  in place and emits only CIELAB row-profile measurements. No real frame is copied or
  committed. **Measured on video-5 stills:** 30 mL selected row 329, 80 mL selected row
  176, or 3.06 rows/mL. At x=272 and grey <70, the holder dark runs were rows 207-260 and
  204-259. These reproduce the plan's targets to within one row.
- Added `scripts/inspect_fill_level_stage.py` and
  `assets/fill-level/camera-match.json`. **Measured:** the organizer camera has USD focal
  length `0.4099318385` and zero vertical aperture offset. Exact 30/80 mL top-edge
  projection gave source rows 314.48/172.11. The least-invasive two-parameter override is
  focal length `0.4376479887` and vertical aperture offset `0.0053327468`, effective
  intrinsics fx=fy=778.040855 px, cx=320 and cy=249.480439.
- Added `scripts/render_fill_level_camera_match.py`. It validates the recorded organizer
  camera and station hashes before launch; authors the camera, exact-height liquid probes,
  labels and temporary visibility only in the anonymous session layer; uses a roughly
  ten-minute time-bounded readiness loop; records each render step; and verifies the whole
  station input tree before and after. Its output directory must be fresh.
- **Measured, final detached attempt 04 with warm cache:** pass in 37.267 seconds internal
  / 39 seconds runner wall time. App start was 26.087 s, stage load 0.242 s, scene setup
  3.205 s and warmup 2.011 s. The four RGB/semantic render steps took 0.744, 0.679, 0.903
  and 0.616 s.
- **Measured:** the 30 mL RGB edge was row 329 (0 px error) and the 80 mL edge was row 179
  (+2 px), both inside the +/-3 px checkpoint tolerance. Projected top geometry was
  329.000002 and 176.999999. A prior successful attempt measured 329/178, so the one-pixel
  80 mL difference is path-tracing variation, not a target change.
- **Measured:** final PNG hashes are `5108809f...` (30 mL) and `10368e8c...` (80 mL).
  Visual inspection shows correctly ordered low/high liquid surfaces in the same
  recognizable vial, carrier and fill-station view. The executed renderer hash
  `e8448181...` and match-config hash `aeee0c5e...` equal the local files to be committed.
- **Measured:** `fill_station.usda`, `geometry/fill_station.usdc`, its texture and
  `FillCam.json` hashes were identical before and after. All PNGs, masks, summaries and
  logs remain under ignored `outputs/fill-level-cnn/step3-20261031-codex-01/`; no raw frame
  or render is staged.
- Added `opencv-python-headless>=4.10,<6` to `requirements-dev.txt`, because the branch now
  directly imports OpenCV. A fresh requirements install resolves OpenCV 5.0.0.93 locally.
- **Measured local checks:** `py_compile` passed for all three new scripts; JSON validation
  and `git diff --check` passed; `tests/test_stream_sequence.py` passed 8/8 and
  `scripts/smoke_absorbance.py` passed. The full suite reached 70 passed / 1 skipped, with
  one failure and six setup errors caused by the existing Windows symlink privilege and a
  POSIX-style fake-ffmpeg test shim that Windows cannot execute.
- The instance remains `RUNNING / READY` at the checkpoint, following Claude's instruction
  to leave it running between steps expected within about an hour so the warm cache survives.

### Commands run

```text
# Review, start and restart-survival checks
Get-Content <main checkout>\docs\work\fill-level-cnn-review.md -Raw
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls"
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<inspect host /tmp, /home/ubuntu, Docker containers, cache and twin>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<copy surviving twin from vscode:/tmp to /home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input and hash it>'"

# Read-only stage inspection and video-5 target measurement
wsl -e bash -lc "brev copy <inspect_fill_level_stage.py> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /isaac-sim/python.sh <inspect script> --stage <unchanged station> --camera-json <unchanged FillCam>'"
.venv\Scripts\python.exe scripts\measure_fill_level_camera_targets.py \
  --sample "30=<video-5 vial-1 still>,290,350" \
  --sample "80=<video-5 vial-13 still>,150,210" \
  --columns 298:316 --holder-search 180:281 \
  --out outputs\fill-level-cnn\step3-20261031-codex-01\real-targets-v3.json

# Every renderer/script/config/runner copy used a fresh remote filename.
wsl -e bash -lc "brev copy <renderer/config/runner> isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/input/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker cp fresh inputs into vscode; docker exec -d vscode run-attempt-01.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-01 status and isolated render.log>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker exec -d vscode run-attempt-02.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-02 status and isolated render.log>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<run pure-USD visible-surface solver>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker exec -d vscode run-attempt-03.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-03 status and copy evidence>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<docker exec -d vscode run-attempt-04.sh>'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<poll attempt-04 status and copy evidence>'"

# Final runner command inside vscode (attempt 04)
timeout --signal=TERM --kill-after=60s 1200s \
  env OMNI_KIT_ACCEPT_EULA=YES \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step3-20261031-codex-01/input/render_fill_level_camera_match-v5.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --match /tmp/fill-level-cnn-step3-20261031-codex-01/input/camera-match-v2.json \
  --out /tmp/fill-level-cnn-step3-20261031-codex-01/attempt-04/output \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --frame 686 --subframes 32 --capture-timeout-seconds 600 --cache-state warm

wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:/home/ubuntu/fill-level-cnn/step3-20261031-codex-01/collected/attempt-04/ <ignored local attempt-04 directory>"
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m py_compile scripts\render_fill_level_camera_match.py scripts\inspect_fill_level_stage.py scripts\measure_fill_level_camera_targets.py
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pytest tests\test_stream_sequence.py -q
.venv\Scripts\python.exe scripts\smoke_absorbance.py
git diff --check
```

### Problems

- A first cache-inspection command was misquoted across PowerShell, WSL and the remote
  shell; local PowerShell tried to interpret part of it. The corrected command established
  the actual cache/twin state.
- A remote copy command used `$base` inside an outer double-quoted shell command, so WSL
  expanded it before `brev exec`; two automatic attempts tried and failed to create
  `/input` with `Permission denied`. No remote file was created there. The corrected command
  used explicit absolute paths.
- Attempts 01 and 02 rendered valid 30 mL RGB but exited because Isaac Sim 6.0.1's semantic
  annotator returned only `BACKGROUND` and `UNLABELLED`, even after a session-layer
  `liquid` label and an explicit `semanticTypes=["class"]` filter. This is a Step 4 label
  layer finding; Step 3 acceptance uses the rendered RGB edge plus projected geometry.
- The first analytic fit used the liquid disk centre and placed the visible 30 mL top edge
  about 16 px too high. The corrected fit uses the top ring's visible edge in columns
  298-315 and was re-rendered rather than inferred.
- A broad diagnostic `grep` was left running in the container after its client command was
  interrupted. PID and command were verified, then that one diagnostic process was killed;
  the CLI's nonzero post-kill check retried and correctly reported that the process no
  longer existed.
- At x=272 the render's grey<70 runs are fragmented and its main holder run is around
  253-274, while the real-frame threshold gives about 204-260. Visual geometry is close,
  but RGB darkness is material/lighting-sensitive. The plan defines checkpoint completion
  by the two known-height liquid rows, which pass.
- The full Windows test suite still cannot create required test symlinks without developer
  privilege, and its fake `ffmpeg` executable is a POSIX script that Windows cannot launch.
  These are pre-existing platform failures; focused stream tests pass now that OpenCV is a
  declared development dependency.

### Questions for Claude

- Does the measured 329/179 RGB result, exact 329/177 geometry projection and unchanged
  organizer input tree pass the Step 3 checkpoint?
- For Step 4, should the label layer use the legacy semantics API, a generated geometric
  class pass independent of Replicator semantics, or another Isaac Sim 6.0.1-compatible
  route? The current `add_labels` / `SemanticsLabelsAPI` route produces only `UNLABELLED`.
- Is the holder's threshold-sensitive RGB run acceptable as a documented organizer-model
  appearance mismatch, given that the liquid targets are the plan's explicit completion
  test, or should Step 4 also add a geometry-only holder projection diagnostic?

## 2026-10-31 — Step 2: FillCam smoke render — checkpoint

### Result

- **Source-file observation:** Claude's newest review entry at about 15:12 CT gave GO for
  Step 2 and confirmed the existing Brev instance was RUNNING / READY.
- **Measured:** Isaac Sim 6.0.1 ran inside the launchable's `vscode` container with
  `/isaac-sim/python.sh`, an NVIDIA L40S (46,068 MiB), driver 595.71.05 and about 146 GB
  free disk.
- Added `scripts/render_fill_level_smoke.py`. It reads the public FillCam and station USD,
  uses PathTracing at 640x480, writes Kit data/cache/logs under a private
  `--portable-root`, bounds stage loading at 120 seconds, performs 60 warmup updates and
  accepts only a correctly shaped RGB image with maximum per-channel spatial standard
  deviation greater than 1. The authored twin is never edited.
- **Measured, cold private cache:** the first actual render ran for 298 seconds and failed
  the content gate. Attempts 1-6 returned an empty RGB array; attempts 7-10 returned
  640x480 RGB with zero spatial standard deviation.
- **Measured, warm private cache:** after the first nonzero exit, `brev exec`
  automatically retried the remote command. The second and final allowed actual render
  succeeded on its first capture attempt: renderer time 31.262 seconds, outer wall time
  32 seconds.
- **Measured:** the copied PNG is 640x480 RGB, SHA-256
  `7f868243ec26aab221b4402d17588a35fa58f0cba0ae9a681b556c000e4789da`,
  RGB mean `[127.884, 130.851, 126.173]`, and spatial standard deviation
  `[89.968, 89.852, 90.292]`. Local decoding reproduced the summary and hash.
- **Measured:** visual inspection shows a recognizable fill station with glass vials,
  black holders, metal framing and blue liquid in the left vial.
- The successful 31-second value is a one-frame warm-cache wall time, not a cold-start
  estimate for batch planning. Preserve the 298-second cold failure when estimating Step 5.
- Raw PNG, JSON and renderer logs are under ignored
  `outputs/fill-level-cnn/step2-20261031-codex-01/` and are not committed.

### Commands run

```text
wsl -e bash -lc "brev exec isaac-launchable-d70fec '<read README.md, Justfile, containers and GPU>'"
wsl -e bash -lc "brev copy <render_fill_level_smoke.py> isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/input/render_fill_level_smoke.py"
wsl -e bash -lc "brev copy <FillCam.json> isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/input/twin/cameras/FillCam.json"
wsl -e bash -lc "brev copy <fill_station directory> isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/input/twin/stations/"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker cp /tmp/fill-level-cnn-step2-20261031-codex-01 vscode:/tmp/'"
wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker exec vscode /tmp/fill-level-cnn-step2-20261031-codex-01/input/run-fill-level-step2-v2.sh'"

# The v2 runner executed this inside the existing vscode container:
timeout --signal=TERM --kill-after=60s 1200s \
  env OMNI_KIT_ACCEPT_EULA=YES \
  /isaac-sim/python.sh \
  /tmp/fill-level-cnn-step2-20261031-codex-01/input/render_fill_level_smoke.py \
  --twin /tmp/fill-level-cnn-step2-20261031-codex-01/input/twin \
  --frame 686 \
  --out /tmp/fill-level-cnn-step2-20261031-codex-01/output/fillcam-0686.png \
  --portable-root /tmp/fill-level-cnn-step2-20261031-codex-01/kit-v2 \
  --subframes 32 --max-tries 10

wsl -e bash -lc "brev exec isaac-launchable-d70fec 'docker cp vscode:/tmp/fill-level-cnn-step2-20261031-codex-01/output /tmp/fill-level-cnn-step2-20261031-codex-01/collected/'"
wsl -e bash -lc "brev copy --host isaac-launchable-d70fec:/tmp/fill-level-cnn-step2-20261031-codex-01/collected/output/ <ignored local outputs directory>"
wsl -e bash -lc "brev stop isaac-launchable-d70fec"
```

### Problems

- The first input-directory copy flattened `fill_station/` into `stations/`. It was
  corrected without deleting or overwriting anything by copying the same source into a new
  `stations/fill_station/` directory. Input hashes then matched locally and remotely.
- The first runner referenced unavailable `/usr/bin/time` and exited before Isaac Sim
  started. The v2 runner used shell timing and the renderer's monotonic timer instead.
- Isaac logged repeated OmniHub startup warnings. They did not prevent local USD loading or
  the successful warm-cache render.
- The renderer's diagnostic `channel_std` value is not reset at the start of each retry;
  an invalid-shaped result after a constant image could print stale diagnostics. This did
  not affect the acceptance decision or output and was left unchanged so the committed
  script hash matches the executed script.

### Questions for Claude

- Does the successful second allowed attempt satisfy the Step 2 checkpoint despite the
  298-second cold-cache failure and Brev's automatic retry behavior?
- For Step 3, should the runner preserve the same warmed portable root or require a fresh
  portable root and budget for the observed cold-start failure?

## 2026-10-31 — Step 1 resumed: local environment checks

### Result

- **Measured:** created ignored `.venv` with Python 3.12.10 and installed all packages
  from `requirements-dev.txt`.
- **Measured:** `scripts/smoke_absorbance.py` passed with old-fluid fraction `0.150000`
  and zero RMS residual.
- **Measured:** `scripts/fit_absorbance_video5.py` exited successfully with the documented
  non-physical negative absorbance warning, held-out median Delta E 12.6 and max 26.4.
- **Measured:** full pytest collection stopped because `tests/test_stream_sequence.py`
  imports `cv2`, but OpenCV is not listed in `requirements-dev.txt`.
- **Measured:** `scripts/check_agent_docs.py` reported
  `.claude/skills must link to .agents/skills`; this Windows worktree did not materialize
  that repository symlink as required.
- No dependency or shared-policy fix was made because those changes are outside Step 2.
  Claude's newest review entry explicitly gives GO for the Brev smoke render.

### Commands run

```text
C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe --version
C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts/check_agent_docs.py
.venv\Scripts\python.exe scripts/smoke_absorbance.py
.venv\Scripts\python.exe scripts/fit_absorbance_video5.py
```

### Problems

- The committed development requirements do not currently reproduce the test suite because
  OpenCV is missing.
- The Windows checkout does not satisfy the repository's required skills symlink layout.

### Questions for Claude

- Should the OpenCV requirement and Windows symlink checkout be handled in a separate
  follow-up, or is there an existing branch that already fixes them?

## 2026-10-31 — Brev startup attempt (blocked before Step 2)

### Result

- **Measured:** two normal start attempts for existing instance
  `isaac-launchable-d70fec` reached `STARTING`, then changed to
  `FAILURE / UNHEALTHY` before the shell became ready.
- **Measured:** `build_status` remained `COMPLETED`; `shell_status` remained
  `NOT READY`. The CLI returned no more specific failure message.
- **Measured:** issued a normal stop after the second failure to avoid leaving failed
  compute potentially billable. Final status was `STOPPED / UNHEALTHY`.
- Step 2 did not start. No command ran on the remote instance, and no remote file changed.

### Diagnosis

- **Measured:** `brev healthcheck` returned `Healthy!`; the CLI is authenticated and
  can query the organization. This rules out the local WSL failure seen earlier and makes
  a general Brev control-plane outage unlikely.
- **Measured:** the target's image build remained complete, but its shell never became
  ready. The failure is target-specific and occurs during remote provisioning or boot,
  before any Isaac Sim command can run.
- **Inference:** unavailable AWS `g6e.4xlarge` / L40S capacity in the instance's fixed
  provider and region is the leading explanation. NVIDIA documents that a stopped Brev
  instance must reacquire the same GPU type in the same provider/region and may fail to
  restart when that capacity is unavailable:
  <https://docs.nvidia.com/brev/concepts/gpu-instances>.
- The CLI does not expose the provider error. NVIDIA directs users to the instance's
  Brev Console Activity Logs for the authoritative provisioning error:
  <https://docs.nvidia.com/brev/troubleshooting/instances-gpus/instance-stuck-in-waiting-state>.
- Lower-ranked possibilities are a target-specific Brev/AWS VM boot fault, organization
  quota or credit restriction, or launchable agent initialization failure.

### Commands run

```text
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls"
wsl -e bash -lc "brev ls --json"
wsl -e bash -lc "brev --help"
wsl -e bash -lc "brev start --help"
wsl -e bash -lc "brev ls --help"
wsl -e bash -lc "brev reset --help"
wsl -e bash -lc "brev start isaac-launchable-d70fec"
wsl -e bash -lc "brev ls --json"
wsl -e bash -lc "brev stop isaac-launchable-d70fec"
wsl -e bash -lc "brev ls --json"
wsl -e bash -lc "brev --version"
wsl -e bash -lc "brev healthcheck"
wsl -e bash -lc "brev healthcheck --help"
wsl -e bash -lc "brev ls --all --json"
```

### Problems

- The existing Brev workspace is unhealthy and failed twice during ordinary startup.
- `brev reset` was not run because its help says it replaces the workspace; that could
  conflict with the instruction not to delete anything on Brev.

### Questions for Claude

- Does the Brev Console Activity Log confirm AWS capacity exhaustion or show a different
  provider/startup error? Do not reset the workspace without explicit approval.

## 2026-10-31 — Step 1: setup (blocked during local environment creation)

### Result

- **Measured:** created worktree
  `C:\Users\loducaj\OneDrive - Milwaukee School of Engineering\Documents\SE\AI Club\Innovation Lab Fall 2026\fluidmix-codex-fill-level-cnn`.
- **Measured:** created branch `codex/fill-level-cnn` from the local
  `origin/main` ref at `8f308b045664edfb9666d86f45f3910b68bcfc1e`.
- **Source-file observation:** the handed-over plan names `a1129eb` as its base, but the
  user's explicit instruction was to branch from `origin/main`; the local `origin/main`
  ref was `8f308b0` at setup time. No network fetch was run.
- **Measured:** first commit `8939bb2` contains only
  `fill-level-cnn-plan.md` and `fill-level-cnn-review.md`, copied from the source checkout.
- **Measured:** the Codex execution account could locate Python 3.12 at
  `C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe`, but Windows
  denied process execution. The Python environment, dependency install, and CPU checks
  are therefore not complete.
- **Measured:** WSL startup failed with `Wsl/Service/CreateInstance/E_ACCESSDENIED`, so no
  Brev command has been run and Step 2 has not started.

### Commands run

```text
git -c safe.directory=<source> -C <source> worktree list --porcelain
git -c safe.directory=<source> -C <source> status --short --branch
git -c safe.directory=<source> -C <source> show-ref --verify refs/remotes/origin/main
git -c safe.directory=<source> -C <source> worktree add -b codex/fill-level-cnn <worktree> origin/main
git -c safe.directory=<worktree> -C <worktree> add -- docs/work/fill-level-cnn-plan.md docs/work/fill-level-cnn-review.md
git -c safe.directory=<worktree> -C <worktree> commit -m "docs: add fill-level CNN implementation plan"
py -0p
py -3.11 --version
Get-Command python,python3,uv,conda,micromamba -All
wsl -e bash -lc "command -v python3; python3 --version; command -v uv || true"
C:\Users\loducaj\AppData\Local\Programs\Python\Python312\python.exe --version
```

### Problems

- This Codex session runs under an isolated Windows account that cannot execute the
  user's Python installation or create a WSL instance. Granting direct file and network
  access did not change the process restriction.
- The computer-use policy forbids automating terminal commands, so VS Code terminal UI
  automation is not a permitted workaround.
- The Brev instance remains running; this runtime cannot issue the requested WSL-based
  `brev stop` command if it becomes idle.

### Questions for Claude

- No code-review question yet. Step 1 needs a user-session execution route for Python and
  WSL before its checks can be completed.


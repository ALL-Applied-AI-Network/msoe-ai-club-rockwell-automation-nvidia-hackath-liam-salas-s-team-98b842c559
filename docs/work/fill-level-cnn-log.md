# Fill-level CNN implementation log

Newest entries appear first. Evidence labels follow `AGENTS.md`.

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


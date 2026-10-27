# Team Mixture Of Experts — Fluid Mixing Carryover

This Dataset 2 project is building a camera-evidence pipeline that measures vial appearance
after each programmed dispense, decides `keep`, `discard` or `insufficient_evidence`, and gives
Cosmos only the resulting quantitative evidence to explain. The current result is an accepted
synthetic render and diagnostic-review path. It is **not** a deployable vial detector.

## Physical problem and evidence boundary

Per Liam's 2026-10-26 correction of the organizer description, T-200, T-400 and T-100 dispense
one at a time through a shared path and the fluids mix inside the vial, not upstream. At the
start of a dispense, residual liquid from the preceding dispense can leave the nozzle before
the current tank fluid. Its physical volume is not measured. In the twin it is an uncalibrated
label-provenance control and is never a model input. See the
[dated correction and source conflict](docs/data.md#physical-line-correction-liam-2026-10-26).

The allowed deployment inputs are:

- camera products from `vial_fill`, `tank_level` and `cap_seal`;
- programmed recipe mL, tank order and vial/recipe sequence.

The current V3 diagnostic uses synthetic `vial_fill` stage RGB plus programmed recipe context.
It does not yet integrate the two other camera feeds. Simulator residual severity, particle
state, composition, palette, nuisance settings and keep/discard labels remain outside the
decision and explanation inputs.

## Implemented synthetic V3 path

1. Isaac Sim 6.2 renders the fixed twin scene and FillCam view through each active dispense.
   A whole run contains three vials, with tank appearance fixed within a run and varied between
   runs. Streams, level rise and in-vial spatial colour bands are rendered.
2. Manifests bind and hash-check every stage RGB and aligned liquid mask. The masks are Isaac
   renderer semantic oracle masks, not predictions from RGB. A strict schema-v2 handoff now
   accepts future RGB-only predicted masks with frozen model/config hashes and rejects hidden
   simulator truth; no predicted-mask artifact has been produced or scored yet.
3. The CPU measurement path computes masked CIE Lab statistics and same-stage pairwise dE76 for
   the three vials in each run.
4. Conservative triage compares each vial with its two same-run peers. Because two agreeing
   vials are not an independently verified clean reference, the current boundary abstains rather
   than assuming they are clean.
5. Cosmos3-Nano is evaluated only as an explanation model. Numbers, units and evidence IDs stay
   deterministic. A deterministic template is the accepted fallback when model output fails the
   content policy.

The detailed contracts and claim limits are in [architecture](docs/architecture.md), the
[current plan](docs/plan.md#twins-job), and
[evaluation](docs/evaluation.md).

## Measured result, 2026-10-27

| Stage | Measured result | Limit |
|---|---|---|
| Render cohort | 18/18 cases passed current hard render gates; 18 videos, 5,886 decoded frames and 42 stage RGB/mask pairs | 15 cases have 16 particle-height warnings; they are render passes, not particle-physics passes |
| Measurement | 42 masked-Lab stage rows and 42 same-stage pairwise dE76 rows across six whole-run groups | Renderer semantic masks are oracle diagnostics |
| Vial decision | 18/18 `insufficient_evidence`; 0 `keep`, 0 `discard` | No independent clean reference; relative-outlier flags are diagnostics, not dispositions |
| Post-hoc relative flag | TP=5, FN=1, FP=1, TN=11 against separated synthetic labels | Exploratory synthetic score with zero decision coverage, not a detector metric |
| Cosmos3-Nano | 72/72 attempts completed; model strict content 0/54 and required abstention 1/54 | Negative result for this frozen prompt/runtime treatment only |
| Deterministic explanation | Parsing, content and required abstention passed 18/18 | Accepted diagnostic fallback, not an operator service |

## CPU diagnostic review in one command

The command below uses the three accepted local JSON artifacts. It loads no model and requires
no GPU. Use a new, empty `--out` directory because publication is no-clobber.

```bash
python3 scripts/build_f2_varied_demo.py \
  --decisions /home/hackathon5/liamreporoot/f2-varied-v3-decisions-20261027/triad-decisions.json \
  --image-locators /home/hackathon5/liamreporoot/f2-varied-v3-measurements-20261027/image-locators.json \
  --cosmos-score /home/hackathon5/liamreporoot/f2-varied-v3-cosmos-20261027/score.json \
  --out /tmp/f2-varied-v3-demo-review
```

The accepted GB10 run took 0.36 seconds and published 18 abstaining decision records linked to
42 hash-checked images. Its bundle is
`/home/hackathon5/liamreporoot/f2-varied-v3-demo-v2-20261027/`; `receipt.json` has SHA-256
`c8389c51ba2241ed7cb7700e78989edf14417f586120fbf248c5e9b40834a95d`.
`REVIEW.md` is the human-readable stage/image review and `operator-records.json` is the
structured diagnostic output.

### Artifact-location caveat

Those three inputs and the 42 images referenced by the locator are local to Liam's GB10 under
`/home/hackathon5/liamreporoot/`. They are ignored generated artifacts, not stored in Git or a
published download. This branch is also local and unpushed. The command is therefore a verified
same-host review command, **not** a clean-checkout portability or remote-availability claim.

## CPU setup and checks

Python 3.11 or newer is supported for the repository's CPU path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/check_agent_docs.py
python scripts/smoke_absorbance.py
```

The smoke uses artificial values. The historical real-video absorbance fit returns negative,
physically invalid absorbance and is not a plant calibration. Isaac rendering and Cosmos model
execution use separate GPU environments documented in
[the successor brief](docs/next-long-run-agent.md).

## Limits

- The V3 cohort is synthetic. It does not prove physical carryover, fluid accuracy, chemistry,
  real-camera transfer or plant performance.
- The current measured result uses semantic oracle masks. The RGB-only predicted-mask contract is
  implemented, but an actual frozen segmenter, predicted masks and held-out mask scores remain
  required before it can replace those masks. A learned vial detector also remains required for
  deployment.
- All 18 current vial decisions abstain. No keep/discard accuracy, coverage, defect recall or
  cause-accuracy claim follows from the relative-outlier diagnostic.
- The count-driven visible body sets rendered height from programmed volume. Particle-height
  misses remain explicit warnings and are not physics passes.
- Real video 5 has no independent defect/cause labels. Real calibration, validation and test
  partitions must remain disjoint, and thresholds must freeze before test scoring.
- Cosmos explains supplied evidence; it never detects a defect or authorizes plant-control
  actions. The frozen Nano condition failed the content gate, so the deterministic abstaining
  template remains the accepted explanation path.

## Data, references and credits

- Dataset and twin: [Fluid Mixing Station 3 organizer page](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing).
  The download is not redistributed; see [data provenance](docs/data.md).
- Rendering: NVIDIA Isaac Sim 6.2, RTX, PhysX GPU PBD particles and
  [OmniSurface transmission](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniSurface_Transmission.html).
- Explanation evaluation: [NVIDIA Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano)
  through Hugging Face Transformers, pinned as recorded in the experiment artifacts.
- Event and facilities: Rockwell Automation, NVIDIA, the MSOE AI Club and UWM's Connected
  Systems Institute.
- Implementation and evidence review: Team Mixture Of Experts, with Codex and Claude used as
  coding/review agents. See Git history and the repository docs for exact provenance.

## Remaining submission work

- have the training track publish RGB-only masks through the frozen schema-v2 contract, then
  evaluate them on held-out whole-run families before integrating their measurements;
- publish a portable, hash-pinned artifact bundle and complete a teammate clean-checkout trial;
- produce the final approximately five-minute video and one-page project summary;
- push or open a PR only after Liam authorizes publication.

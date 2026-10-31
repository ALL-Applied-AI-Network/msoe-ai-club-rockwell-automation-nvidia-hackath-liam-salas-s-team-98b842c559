# Project plan

Team: **Mixture Of Experts**. Dataset 2 (Fluid Mixing).

Sources, in order of authority:
1. The team proposal, [proposal](proposal.md). Since 2026-10-16 this repo file is the source of
   truth; it started as a copy of the Google Doc "Project Proposal" (owner Liam Salas).
2. The pipeline doc "Fluid Mixing: Carryover Pipeline"
   (https://claude.ai/code/artifact/863bc9b6-1bf6-460b-abbf-8d6fe9d841fc). It has more detail on
   baselines and how to prove generalization. Its claims still need the feasibility checks above.

When the plan changes, update this file within scope and identify external source updates
for the team. Sending or editing external documents needs authorization.

## Current status and resume gates (2026-10-31)

This is a partial completion checkpoint, not a completed project. The tracked [one-page summary](one-page-summary.md)
is the concise deliverable. Evidence currently supports:

- **Fills:** 18/18 passed render gates, but WP1 aggregate DP3 is parked at the detection limit:
  the measured 1.5 mL-versus-0 mL pair did not clear the frozen visibility threshold.
- **Transport:** 81/84 joins accepted; V3 is 6/6 runs and V4 is 9/10 runs. V4 ordinals 82--84
  remain deferred.
- **WP1 frozen pass:** 18 abstentions and 0 decisions; this does not establish detector accuracy.
- **Held gates:** video-5 attempt-01 ended fatal before scoring; attempt-02 is source-only prepared
  and held. INN-62's image-grounding decision is held for Liam.
- **Operations:** zero Brev workspaces are present, and no push has been performed.

These limits are resume gates and evidence boundaries, not plant-readiness claims. Historical
measurements below remain unchanged; the detailed source report used for this checkpoint is the
local-only file `/home/hackathon5/liamreporoot/outputs/pipeline-20261028/FINAL-REPORT.md` and is
not a portable repository document.

Contributor entry point: [docs index](README.md). Interfaces and units: [architecture](architecture.md).
The proposal copy from merged PR #1 and scaffold from PRs #2/#3 are on main.

Evidence: detailed numerical/twin observations below were reported by the PR #2 scaffold.
The CSV diagnostic was reproduced in this batch; external-video/twin inspection was not.
See [review evidence](review-inn-22-24.md).

## Problem
Three tanks dispense one at a time through one shared dispensing path and nozzle: T-100 clear,
T-200 orange and T-400 blue (T-300 was out of service). A recipe gives mL per tank in a fixed
order (T-200, then T-400, then T-100). The fluids are not mixed before the vial; they mix in
the vial. Per Liam's 2026-10-26 physical-line correction, each dispense begins with residual
liquid from the preceding dispense and then transitions to the current tank's liquid. The
residual volume is not measured. See [the dated source correction](data.md#physical-line-correction-liam-2026-10-26).

Historical video-5 observation (rough color method, not independent defect truth): the scaffold reports 8 of 10 first vials as visibly off
(median dE 23.6). Second vials are about clean (median dE 2.3).

## What the judges want (from the lab pages)
- Runs on real hardware. Solves one specific problem. Has a business case.
- Measured against a baseline.
- Fluid page: detection alone is "not enough". Link the defect to a cause.

## Pipeline (proposal steps 1-6; conditional design)
1. **Tank color estimation.** A vision model on the tank camera estimates each tank's
   absorbance. The proposal expects this to be the hardest ML step (lighting, vial material,
   smudges). The twin has no tanks, so this model cannot train on twin tank images.
   See "Open questions".
2. **Forward prediction.** Compute the clean-recipe expectation for anomaly scoring.
   Separately compute a candidate carryover stream sequence from each current/preceding
   dispense pair and an assumed leading residual volume. Do not use that candidate as the
   clean anomaly baseline or expose its uncalibrated residual value as a model input.
   Code: `carryover_absorbance()` in `src/fluidmix/absorbance.py`.
3. **Measurement.** Estimate the real vial absorbance from the vial camera, against a clear
   reference (I0). Model trained on twin renders.
4. **Comparison.** Flag a deviation with a **per-recipe dynamic threshold**. Why: in a
   pale (low-absorbance) mix, residue may cause a larger relative change. Whether a
   recipe-dependent threshold improves detection over a fixed one needs matched validation.
   Not built yet. Candidate thresholds may use expected absorbance or measured clean-vial
   noise; choose and freeze them on validation data before test scoring.
5. **Explanation.** Send the numbers and the control-panel recipe to Cosmos
   (build.nvidia.com). It writes an operator message and a recommendation, e.g.
   "flush with diluent before the next recipe". It explains. It does not detect.
   We also fit "new recipe + x% of old" (`fit_carryover_fraction()`), so the message has a number.
6. **Data split.** Train learned models on twin synthetic data. Partition real data into
   disjoint calibration, validation and final-test roles. Calibrate per-tank absorbance
   only from the calibration partition using known recipes and justified clean vials
   (`fit_tank_absorbance()`); tune thresholds on validation, then freeze before final test.

### Critical point on step 6
Calibration uses real clean vials, and testing uses the same real video. If you calibrate
on vials you later test on, the result is too optimistic. Rule: calibrate on one split of
recipes and test on the others (leave-one-recipe-out). See `docs/evaluation.md`.
Group equivalent mixtures too: recipe IDs 5 and 8 have the same tank fractions.
The inherited diagnostic holds out IDs only and does not establish unseen-mixture performance.

### Status of step 6 calibration (2026-10-11)
The first fit on video 5 gives negative absorbance, which is not physical. The math
passes its tests; the color measurement is the likely problem. See `docs/absorbance.md`.

## Twin's job
Render vials with many liquid colors, fill levels and lights, to train step 3.
Twin augmentation trains the model. It does **not** prove generalization (testing the twin
against itself is a circle).

### Measured synthetic successor result (2026-10-27)

The first complete Isaac Sim 6.2 S0088 cohort proved the current render mechanics, not a
detector. Eighteen of 18 cases rendered and produced 54 aligned stage RGB/mask pairs; top-level
inspection found liquid and multi-frame source-coloured streams at every pour. The keep/discard
gate failed: only 41 of 54 RGB files were content-unique, two exact-RGB groups mixed the
synthetic keep and discard labels, and every one of the 12 scored vials correctly became
`insufficient_evidence` under the duplicate-aware reference rule. This cohort must not be
presented as a successful detection benchmark or as evidence of sim-to-real transfer.

A frozen local Cosmos3-Nano comparison then completed 36 model attempts and 12 deterministic
controls. The model passed the forgiving parser on 12/36 attempts, required abstention on 4/36,
and strict content on 0/36. The deterministic template passed all content checks on 12/12.
Therefore the template remains the accepted explanation path while the next synthetic cohort
tests stronger, pre-frozen residual severities and the existing measured-range light nuisance
authoring. Detailed artifacts and hashes are recorded in `outputs/compute-ledger.md` and the
ignored `f2-s0088-downstream-7042fbd` result root.

A second, stronger S0088 stress cohort then completed on the same synthetic train family. Two
Isaac Sim 6.2 L40S shards produced 18 cases (12 prescribed keep, 6 prescribed discard), 381
source captures, 7,074 decoded video frames and 54 stage RGB/mask pairs. One case failed its
first stream-semantic capture attempt and passed an unchanged fresh-process retry; both outcomes
remain in the run history. File-by-file verification passed 7,906/7,906 files, and top-level
inspection passed the fixed-camera, stage-rise, liquid-mask and early/late stream checks.

On the 12 scored target vials, the frozen colour-distance rule discarded all 6 prescribed
discard cases and abstained on all 6 prescribed keep cases because at least one stage lacked a
unique, sufficiently large reference cluster. Coverage was 6/12; covered accuracy and discard
recall were 6/6, while keep recall was 0/6. This is a useful conservative train-family diagnostic,
not a held-out detector result. A fresh local Nano run completed all 36 model attempts plus 12
template controls with no inference errors. Model-only legacy raw strict schema was 0/36,
lenient parse was 12/36, strict content was 0/36 and required abstention was 0/18. The
deterministic template passed strict content 12/12 and required abstention 6/6, so it remains the
accepted explanation path. The ignored `outputs/00c50cf-stress-result/TRAINING_HANDOFF.md`
records exact inputs,
hashes, separated labels and claim limits. This cohort is synthetic train data only; it provides
no held-out-family, chemistry, physical carryover-volume or sim-to-real evidence.

The replacement varied-run V3 cohort completed under a later operator decision.
Its count-driven body height, derived from programmed cumulative millilitres, is the image and
label height of record. Finite PhysX particle-height misses are retained in each receipt as
`diagnostic_warning_not_a_physics_pass`; they no longer reject an otherwise complete render.
Zero leakage/containment, readable artifacts, aligned masks and render checks remain hard gates.
The frozen plan has six whole-run groups (four train, one validation and one evaluation), 18
three-vial cases, 12 prescribed synthetic keep labels and 6 prescribed synthetic discard labels.
All 18/18 cases passed the current render gates with zero hard failures, producing 42 stage
RGB/mask pairs, 18 AVI videos and 5,886 decoded video frames. Fifteen cases carry 16 explicit
particle-height warning rows; only train indices 3--5 have no warning, so the warning-free subset
has no validation or evaluation coverage and is not an evaluation dataset. The prescribed
residual severity remains label provenance only and is excluded from model inputs. Exact merge,
attempt and 1,271-file manifest hashes are recorded in the external training handoff and compute
ledger. This is accepted synthetic render data, not a physics or sim-to-real result.

The initial V3 downstream publication root is preserved but invalid for use: its legacy runtime
receipt conflated volume-driven rendered-body height with particle height. The corrected
publication explicitly records `rendered_body_volume_height` as a hard gate and particle height
as a diagnostic that never claims a physics pass. It binds all six whole-run batches into 18
vial records and 42 hash-checked stage RGB/mask pairs (four train runs, one validation run and
one evaluation run). Fresh official validation passed all 18 source cases with no hard failures,
and provenance preserves 16 particle-height warning rows across 15 cases. The accepted external
receipt is
`/home/hackathon5/liamreporoot/f2-varied-v3-downstream-v2-20261027/downstream-validation-receipt.json`
(SHA-256 `3b34a818...c54a2`). This is an artifact-integrity and routing result only; no V3 keep/discard
claim follows from that receipt alone.

The label-free V3 measurement pass then rehashed all 42 RGB/mask pairs and produced 42 masked
Lab stage measurements plus 42 same-stage pairwise comparisons across six runs. With only three
vials per run and no independently verified clean reference, the deterministic triad boundary
reported `insufficient_evidence` for all 18 vials instead of assuming that the agreeing pair was
clean. Its separate relative-outlier diagnostic flagged six vials; post-hoc synthetic scoring was
5/6 discard labels detected, 1/12 keep labels falsely flagged and one discard label missed. Those
flags are not keep/discard decisions or a deployable threshold.

One frozen local Cosmos3-Nano evaluation completed 72/72 attempts with zero inference errors:
54 model attempts across numbers-only, ordered-images and irrelevant-shuffled-images conditions,
plus 18 deterministic controls. Nano passed the forgiving parser on 18/54 model attempts, strict
content on 0/54 and required abstention on 1/54. Ordered and shuffled visual responses were
byte-identical for 16/18 paired vials and each visual condition parsed 0/18. The deterministic
template passed parsing, content and abstention 18/18, so it remains the accepted explanation
path. Because all correct dispositions in this cohort are abstentions, the image control does
not test cases whose correct answers differ. This negative result applies to the frozen Nano
prompt/runtime treatment, not to Nano generally. These are synthetic explanation-quality
results, not physical-carryover or sim-to-real evidence.

A CPU-only diagnostic review command now binds the accepted V3 decisions, image locator and
Cosmos score into one no-clobber bundle. The measured GB10 run completed in 0.36 seconds,
rehash-checked 42 renderer-semantic-mask images and published 18 deterministic diagnostic records;
all 18 remain `insufficient_evidence`. The accepted bundle is
`/home/hackathon5/liamreporoot/f2-varied-v3-demo-v2-20261027/`, whose `receipt.json` SHA-256 is
`c8389c51ba2241ed7cb7700e78989edf14417f586120fbf248c5e9b40834a95d`. This is a same-host
synthetic review path, not a detector, an RGB-only mask path, an operator service, a physical or
cause result, or a clean-checkout portability claim; its inputs and referenced images remain
ignored local artifacts.

The next camera boundary is now implemented as a strict schema-v2 locator handoff for RGB-only
predicted liquid masks. It binds the producer's frozen model/config hashes, code commit and
validation-threshold source, permits only RGB as model input, and fails closed on empty, malformed,
missing or hash-mismatched masks. The legacy renderer-semantic locator remains readable for the
diagnostic above. No RGB-only predictions or held-out mask scores exist yet, so this closes an
integration contract only and does not change the current oracle-mask or deployment claim limits.
The accepted Varied V3 measurement CLI now consumes that schema through the optional
`--predicted-image-locators` argument, requires its 42 RGB identities to match the verified
capture manifests, and records the prediction artifact hash in measured evidence. This is ready
for the training track's future output, but has not been exercised with model predictions.

A 36.8-second local review reel now shows one held-out prescribed-discard vial beside a
prescribed-keep vial from the same synthetic run, including the real-time fill and both stage
stills. The MP4 SHA-256 is
`420882ebc00fc8c37cf62366c7073568ec676e9cf516f5701cbed7ae9e5cf3c7`; its labels and residue
values are review overlays and are not model inputs. A top-level visual check passed stream
visibility, stage-liquid visibility, label readability and the visible synthetic-only boundary.
This is a review artifact, not a detector result.

A compact local review packet at
`/home/hackathon5/liamreporoot/f2-varied-portable-review-20261027/` contains copies of the reel,
judged crops, one-page summary and diagnostic records but excludes raw cohort/source videos,
real data and weights. All nine payload hashes pass; the `SHA256SUMS` file SHA-256 is
`0b25457aa6663bedb0bbbb4e6b8cfab4270db9c66bfec151c8d0b78cedfc63a3`.

The first pre-registered no-learning mask experiment used stage RGB plus a separately hash-bound
same-vial pre-pour RGB. Commit `0b81b49` implements the temporal schema and validation-only runner.
It evaluated all 56 frozen candidates on 30 train and 6 validation stage pairs; the six evaluation
stage images remained unopened. The selected Lab dE76 >=24 height-prior condition beat the
location-only control in validation mean Dice (0.949468 vs 0.864760) with 0/6 empty masks, but its
mean absolute mask-height error was 30.17 px against the frozen <=5 px gate. A top-level visual
check confirmed a disconnected false-positive upper/nozzle component in all three first-stage
masks (55–60 px height errors), while second-stage errors were 1–3 px. The result is therefore
rejected without post-look repair or evaluation scoring. The result artifact SHA-256 is
`2ae01931982e80fd939c016b44fcdfb43da9e1e478f9c6b23f1912f8f3c6b3c5`; the complete checksum
manifest SHA-256 is `fab1644e3a6658944dcf3d0235349b11cb0eee5cf63c9d590f259a39b76ecca9`.
The one bounded independent result review found no P1/P2 issue; its SHA-256 is
`74df1ec865abeeb9054cd31ef742c1b9c77c3239439aba026e479ed4ffa6444f`.

The WP7 source scaffold now supplies one production CLI for validation freezing, a one-pass
test run and a separate one-pass real check. It accepts only camera-video and programmed-recipe
bytes as process evidence, binds operational freeze/store/claim/output paths, and fails closed
to per-vial `insufficient_evidence` records while the independently reviewed RGB-only predictor
and optional explanation backend are unavailable. Its atomic claims, no-clobber receipts,
append-only journal and denominator equations are covered only by synthetic CPU tests. No
validation, held-out or real artifact has been opened, and no detector or physical result follows.

The current WP7 integration selects the deterministic guarded template as the official
DP4/default explanation backend and rejects a model-backed explanation selection at validation
freeze. Disposition and DP5 remain separate. The explicitly selected `evaluation_only` route
retains its test abstentions and real-check skips, while malformed or missing `history_backed`
inputs fail closed without fallback. The completed 16-attempt-per-method all-abstention model run
is recorded only as diagnostic metadata: it supplies no keep/discard accuracy and no winner. The
full comparison remains Friday's INN-67 work; no frozen one-pass input was opened for this
integration.

## Additional data (proposal)
Controlled recipe-transition runs at the CSI lab (UWM): known tank colors, deliberate
residue cases, and diluent flushes. Needed because the twin has no tank model and T-300 was
out of service. Not yet requested or confirmed.

## How we prove it
- Leave-one-recipe-out on real video 5.
- Bench test: food coloring in vials with known mL, webcam photos, check the absorbance math.
- CSI recipe-transition runs (above), if approved.

## Baselines
- Vision-only fixed-threshold colour check (see evaluation.md).
- Drainage-only tank model: sequential tank dispenses with an assumed leading residual volume
  from the preceding dispense, with no camera. The physical sequence follows Liam's correction,
  but the residual volume is uncalibrated. Proposed, not implemented or scored. A later vision
  model that estimates the volume drawn per tank per frame is a separate future task.
- Not a baseline for now: the plant's Teledyne check, the system being replaced. It only tests
  whether a colour is known; its outputs may be gathered later for context.
- "Always flag the first vial after a change": inherited 8/10 rough-color claim,
  pending independent labels and a defined transition denominator. See evaluation.md. Compare severity,
  skip changeovers that are clean, handle new colors.

## Compute (proposal estimates)
- Brev L40S, Isaac Sim rendering: ~60 GPU-hrs.
- Brev L40S, Cosmos integration: ~15-20 GPU-hrs.
- build.nvidia.com for Cosmos.
Brev credits burn while an instance runs. Stop instances you do not use.

## Open questions
- Organizers: are we expected to run the twin for problem 2 without the tanks? (proposal)
- Recipes for videos 1-4. Without them we cannot score those videos.
- Tank path length, for the tank-camera route (see docs/absorbance.md).

## Deadlines

Source: [Linear project delivery phases](https://linear.app/innovation-lab-2026/project/mixture-of-experts-fluid-mixing-carryover-315d582ab3f1), checked Oct 16.
M1 feasibility: Oct 18. M2: Oct 25. Freeze: Oct 28. Packaging: Oct 31.
Final submission: Nov 1, 2026, 23:59 America/Chicago (video ~5 min, one-page summary,
repo link, references). Finals: Nov 5. Final demo/release reproduction remains INN-19.

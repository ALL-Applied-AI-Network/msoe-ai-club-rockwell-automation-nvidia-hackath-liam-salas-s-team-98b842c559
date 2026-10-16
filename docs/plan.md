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

Contributor entry point: [docs index](README.md). Interfaces and units: [architecture](architecture.md).
The proposal copy from merged PR #1 and scaffold from PRs #2/#3 are on main.

Evidence: detailed numerical/twin observations below were reported by the PR #2 scaffold.
The CSV diagnostic was reproduced in this batch; external-video/twin inspection was not.
See [review evidence](review-inn-22-24.md).

## Problem
3 tanks feed 1 nozzle: T-100 clear, T-200 orange, T-400 blue (T-300 was out of service). A
recipe gives mL per tank, in a fixed order (T-200, then T-400, then T-100). The fluids share
one mixing path. After a recipe change, residue left in the path tints the first vial of the
new batch.

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
   Separately compute a candidate carryover mixture from current/previous recipe mL and
   an assumed residue fraction. Do not use that candidate as the clean anomaly baseline.
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
- Drainage-only tank model: simulated volume drawn from each tank per vial fill and the residue
  it leaves, with no camera. Proposed, not implemented. A later vision model that estimates the
  volume drawn per tank per frame is a separate future task.
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

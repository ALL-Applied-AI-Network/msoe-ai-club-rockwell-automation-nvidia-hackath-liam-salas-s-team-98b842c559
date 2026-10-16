# Project plan

Team: **Mixture Of Experts**. Dataset 2 (Fluid Mixing).

Sources, in order of authority:
1. The team proposal, [proposal](proposal.md). Since 2026-10-16 this repo file is the source of
   truth; it started as a copy of the Google Doc "Project Proposal" (owner Liam Salas).
2. The pipeline doc "Fluid Mixing: Carryover Pipeline"
   (https://claude.ai/code/artifact/863bc9b6-1bf6-460b-abbf-8d6fe9d841fc). It has more detail on
   baselines and how to prove generalization.

When the plan changes, update this file within scope and identify external source updates
for the team. Sending or editing external documents needs authorization.

Contributor entry point: [docs index](README.md). Interfaces and units: [architecture](architecture.md).
The proposal copy from merged PR #1 is included in this dependent review checkout.

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

## Pipeline (proposal steps 1-6)
1. **Tank color estimation.** A vision model on the tank camera estimates each tank's
   absorbance. The proposal expects this to be the hardest ML step (lighting, vial material,
   smudges). The twin has no tanks, so this model cannot train on twin tank images.
   See "Open questions".
2. **Forward prediction.** From the recipe mL plus the residue of the previous recipe,
   compute the expected vial absorbance with Beer-Lambert additivity.
   Code: `carryover_absorbance()` in `src/fluidmix/absorbance.py`.
3. **Measurement.** Estimate the real vial absorbance from the vial camera, against a clear
   reference (I0). Model trained on twin renders.
4. **Comparison.** Flag a deviation with a **per-recipe dynamic threshold**. Why: in a
   pale (low-absorbance) mix, the same residue gives a larger relative change, so one
   fixed threshold fails. Not built yet. One option: scale the threshold by the expected
   absorbance, or by the noise seen on clean vials of similar mixes.
5. **Explanation.** Send the numbers and the control-panel recipe to Cosmos
   (build.nvidia.com). It writes an operator message and a recommendation, e.g.
   "flush with diluent before the next recipe". It explains. It does not detect.
   We also fit "new recipe + x% of old" (`fit_carryover_fraction()`), so the message has a number.
6. **Data split.** Train on twin synthetic data. Hold out **all** real video for
   validation and testing. Calibrate per-tank absorbance by solving backward from known
   recipes on clean vials (`fit_tank_absorbance()`).

### Critical point on step 6
Calibration uses real clean vials, and testing uses the same real video. If you calibrate
on vials you later test on, the result is too optimistic. Rule: calibrate on one split of
recipes and test on the others (leave-one-recipe-out). See `docs/evaluation.md`.

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
- Teledyne color check (passes any color in its database).
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
Team plan 2026-10-14. Final 2026-11-01 (video ~5 min, one-page summary, repo link,
references). Finals 2026-11-05.

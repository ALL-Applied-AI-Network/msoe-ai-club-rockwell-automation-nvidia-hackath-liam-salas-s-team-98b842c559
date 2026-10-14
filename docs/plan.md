# Project plan

Source: team pipeline doc "Fluid Mixing: Carryover Pipeline"
(https://claude.ai/code/artifact/863bc9b6-1bf6-460b-abbf-8d6fe9d841fc). Keep both in step.

## Problem
3 tanks feed 1 nozzle: T-100 clear, T-200 orange, T-400 blue (T-300 broken). A recipe gives
mL per tank, in a fixed order (T-200, then T-400, then T-100). The fluids share one mixing
path. After a recipe change, fluid left in the path tints the first vial of the new batch.

Measured on video 5 (rough color method): 8 of 10 first vials are visibly off
(median dE 23.6). Second vials are about clean (median dE 2.3).

## What the judges want (from the lab pages)
- Runs on real hardware. Solves one specific problem. Has a business case.
- Measured against a baseline.
- Fluid page: detection alone is "not enough". Link the defect to a cause.

## Pipeline
1. **Expected color.** Read each tank's color (tank camera). Mix in absorbance with the mL
   values. Code: `src/fluidmix/absorbance.py`.
2. **Measure.** Find the liquid in the vial image. Measure its color against a clear
   reference (I0). Model trained on twin renders.
3. **Detect.** Flag if measured vs expected dE > threshold.
4. **Explain.** Fit "new recipe + x% of old". Small residual = carryover, with x.
   Large residual = some other cause (tank, fill level).
5. **Forecast.** Before the fill, predict the first-vial error from the old and new recipe.
6. **Operator message.** Cosmos Reason writes a short "why". It explains. It does not detect.

## Twin's job
Render vials with many liquid colors, fill levels and lights, to train step 2.
Twin augmentation trains the model. It does **not** prove generalization (testing the twin
against itself is a circle).

## How we prove it
- Leave-one-recipe-out on real video 5.
- Bench test: food coloring in vials with known mL, webcam photos, check the absorbance math.
- Live test at CSI, if a tank can hold a new color (ask Shamar; not known).

## Baselines
- Teledyne color check (passes any color in its database).
- "Always flag the first vial after a change" catches 8/10. We must beat this: say how bad,
  skip changeovers that are clean, handle new colors.

## Open questions
- Recipes for videos 1-4 (ask in Slack). Without them we cannot score those videos.
- Can a tank hold a new color for a live test? (ask Shamar)
- Tank path length for the tank-camera route (see docs/absorbance.md).

## Deadlines
Team plan 2026-10-14. Final 2026-11-01. Finals 2026-11-05.

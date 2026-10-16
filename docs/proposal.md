# Project Proposal: Mixture Of Experts

This file is the source of truth for the proposal (decided 2026-10-16). It started as a copy of
the team Google Doc "Project Proposal" (last edited 2026-10-13); edit it here.

## Problem and first approach: Fluid Mixing

1. **Tank color estimation:** A vision model on the tank camera estimates each tank's color/absorbance. We expect this to be the hardest ML step because of lighting, vial material, and camera smudges.
2. **Forward prediction:** From the recipe volumes (mL per tank) plus the residue left by the previous recipe, compute the expected vial absorbance using Beer–Lambert additivity. RGB mixing doesn't model dyes correctly, so we work in absorbance.
3. **Measurement:** Estimate the actual vial absorbance from the vial camera.
4. **Comparison:** Flag any deviation using a per-recipe dynamic threshold. Residue matters more in a low-absorbance mixture, so a fixed threshold won't work.
5. **Explanation:** Pass the quantitative results and the control panel recipe to Cosmos (build.nvidia.com). It generates an operator-facing explanation and recommendation, e.g. "flush with diluent before the next recipe."
6. **Data split:** Train and test on synthetic data from the digital twin, using randomly generated recipes. Keep the real video and its 10 recipes for validation only: before each vial is measured, predict its colour difference from the clean recipe colour, then compare with the colour measured on video frames after the fill ends. When the predicted difference exceeds a threshold set on simulated data, a physics model recommends an action. Absorbance values come from a separate model (decision deferred).

## Dataset

Dataset #2

## Additional data needed

Possibly controlled recipe-transition runs recorded at the CSI lab (UWM). These would use known tank colors and include deliberate residue cases and diluent flushes. We need them because the digital twin has no tank model, and Tank 3 was out of service during the original data collection.

## Compute requested

Brev, build.nvidia.com

## Compute estimates

- Brev L40S (Isaac Sim): ~60 GPU-hrs
- Brev L40S (Cosmos integration): ~15-20 GPU-hrs

## AI tools and agents requests

We need access to two Claude Max ($100) plans in order to properly support members who currently have access to free agentic coding tools but quickly run into limits running our skills and pre-push adversarial review checks.

## Mentor support

(blank in the doc)

## Organizer questions

- Are we expected to run the digital twin for problem 2 without the tanks?

## Key dates (from the lab article, not the proposal)

- Team plan due: Wed Oct 14, 11:59 PM CT
- Final submission due: Sun Nov 1, 11:59 PM CT
- Finals: Thu Nov 5, MSOE Diercks Hall

# Project Proposal: Mixture Of Experts

Copied from the team Google Doc "Project Proposal" (last edited 2026-10-13).
The Google Doc is the source. Update this copy when the doc changes.

## Problem and first approach: Fluid Mixing

1. **Tank color estimation:** A vision model on the tank camera estimates each tank's color/absorbance. We expect this to be the hardest ML step because of lighting, vial material, and camera smudges.
2. **Forward prediction:** From the recipe volumes (mL per tank) plus the residue left by the previous recipe, compute the expected vial absorbance using Beer–Lambert additivity. RGB mixing doesn't model dyes correctly, so we work in absorbance.
3. **Measurement:** Estimate the actual vial absorbance from the vial camera.
4. **Comparison:** Flag any deviation using a per-recipe dynamic threshold. Residue matters more in a low-absorbance mixture, so a fixed threshold won't work.
5. **Explanation:** Pass the quantitative results and the control panel recipe to Cosmos (build.nvidia.com). It generates an operator-facing explanation and recommendation, e.g. "flush with diluent before the next recipe."
6. **Data split:** Train on synthetic data from the digital twin and hold out all real video for validation and testing. Calibrate per-tank absorbance by solving backward from known recipe volumes on clean (uncontaminated) vials.

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

---
name: expected-vial-color
description: Predict vial absorbance, calibrate tanks or fit new fluid plus residue.
---
# Expected vial color

## Inputs
Recipe mL, tank R/G/B absorbance, matched clear-water I0 and measured linear RGB.
Read [physics](../../../docs/absorbance.md) and [contracts](../../../docs/architecture.md).

## Workflow
1. Validate physical inputs; convert sRGB with srgb_to_linear().
2. Fit tanks on the calibration partition with a full-rank recipe design.
   Check finite values, I0 and signs; negative absorbance blocks use.
3. Predict with expected_color(), or fit residue with fit_carryover_fraction().
   Report fraction, RMS residual and any clipped endpoint or indistinguishable recipes.

## Output and validation
Prediction/fit with units, input IDs and validity. Run python -m pytest tests -q.
The synthetic smoke must recover 0.15 with residual below 1e-6.

## Limits
The real-color calibration is nonphysical. Do not clip it or silently assume missing tanks.
A small residual does not prove carryover; synthetic recovery does not prove camera accuracy.

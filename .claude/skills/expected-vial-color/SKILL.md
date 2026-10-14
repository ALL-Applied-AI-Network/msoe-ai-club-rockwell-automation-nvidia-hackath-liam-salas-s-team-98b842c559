---
name: expected-vial-color
description: Use when computing the expected color of a vial from a recipe, fitting tank absorbance, or explaining a bad vial as carryover (new + x% old).
---
# Expected vial color (Beer-Lambert)

Read `docs/absorbance.md` first. Code: `src/fluidmix/absorbance.py`.

## Rules
- Mix in absorbance, never in RGB. Convert sRGB pixels to linear first.
- Only volume fractions matter (side view, fixed path = vial diameter).
- T-100 is clear (A = 0).
- Every absorbance must be >= 0. A negative fit value means the measurement is wrong.
  Stop and report it. Do not clip it and continue silently.
- You need `I0` (clear-water view, same position and exposure).

## Steps
1. Get tank absorbance: `fit_tank_absorbance(rows)` on clean vials (2nd and 3rd of a batch),
   or the tank-camera route (needs tank path length, not yet known).
2. Expected color: `expected_color(ml, tank_a, i0)`.
3. Compare: `delta_e(linear_to_srgb(pred), measured_srgb)`.
4. Explain a first vial: `fit_carryover_fraction(measured_lin, new_ml, old_ml, tank_a, i0)`.
   Report x and the residual. A large residual = not carryover.
5. Run `python tests/test_absorbance.py` after any change to the module.

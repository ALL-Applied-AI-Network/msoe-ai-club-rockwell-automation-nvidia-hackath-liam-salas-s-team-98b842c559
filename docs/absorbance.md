# Absorbance (Beer-Lambert) color model

Code: `src/fluidmix/absorbance.py`. Tests: `tests/test_absorbance.py`.

## Why not average RGB
Dye removes light. Light that passes through a liquid follows Beer-Lambert:

    I = I0 * 10^(-A)        A = epsilon * c * d

- `I0`: light seen through clear water (the reference).
- `I`: light seen through the dyed liquid.
- `A`: absorbance. `epsilon` = how strongly the dye absorbs, `c` = concentration,
  `d` = path length.

When two dyes mix, their **absorbances add**. Their RGB values do not. Example: 50/50 orange
and blue gives a dark brown-green, not the RGB average.

## The formula we use
The camera looks at the vial from the side, so `d` = vial diameter (~26.4 mm). It does not
change with fill level. So only the **volume fraction** of each tank matters:

    A_mix[ch] = sum over tanks i of (V_i / V_total) * A_i[ch]
    I_mix[ch] = I0[ch] * 10^(-A_mix[ch])

- `A_i[ch]` = absorbance of pure tank i fluid through one vial, per channel (R, G, B).
- T-100 is clear: `A = 0`. It only dilutes.
- Work in **linear** RGB. Convert sRGB pixels with `srgb_to_linear()` first.

## Carryover
A first vial = (1 - x) * new mix + x * old mix. In absorbance this is linear:

    A_vial = (1 - x) * A_new + x * A_old

`fit_carryover_fraction()` finds x by 1-D least squares and returns a residual.
Small residual: carryover explains the vial. Large residual: look for another cause.

## Where A_i comes from (two routes)
1. **From clean vials with known recipes** (`fit_tank_absorbance()`): regress
   `log10(I) = log10(I0) - f200*A200 - f400*A400` per channel.
2. **From the tank camera** (`tank_absorbance_from_image()`): A of the tank, scaled by
   vial path / tank path. We do **not** know the tank path length yet. Measure it.

## Result on real video 5 (2026-10-11) - honest status
`python scripts/fit_absorbance_video5.py` on the 20 clean vials gives **negative**
absorbance values (e.g. T-200 in R: -1.5). Dye cannot add light, so this is **not
physical**. Leave-one-recipe-out median dE = 12.6 (max 26.4). Not good enough.

Likely causes (not yet proven):
- The colors are not transmission measurements. There is no `I0` (clear vial) in the
  same spot, and the fit had to guess it.
- `scripts/color.py` takes the most saturated pixels, which biases dark mixes.
- The camera may use auto exposure and auto white balance (not checked).

What to do next:
- Measure `I0` from a clear-water vial (or a vial with only T-100) at the same spot.
- Use a liquid mask (e.g. from the twin's semantic labels) and the mean pixel, not the
  most saturated ones.
- Check if exposure changes over the video (watch a fixed metal part's brightness).
- Bench test with food coloring and a fixed-exposure webcam, to test the math alone.

The math is verified on synthetic data (unit tests recover A and x exactly). The
weak link is the measurement, not the formula.

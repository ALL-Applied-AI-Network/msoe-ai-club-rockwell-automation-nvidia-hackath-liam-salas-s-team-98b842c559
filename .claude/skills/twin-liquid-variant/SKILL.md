---
name: twin-liquid-variant
description: Use when making digital-twin renders with a new liquid color, fill frame, or recipe for training data.
---
# Twin liquid variants

Read `docs/twin.md` first.

## Facts
- One liquid material: `/FillStation/_materials/USD_blue_liquid_approximation`
  (UsdPreviewSurface). It has no absorption. Its color never changes over time.
- Animation: frames 1-686 at 30 fps. Blue done ~368 (~48 mL); final 686 (~67 mL).
- Rendering needs Isaac Sim 6.0 and an RTX GPU. Making the variant .usda does not.

## Steps
1. Make the variant:
   `python scripts/twin_liquid_variant.py --twin external/csi_digital_twin_v1 --recipe t100=20,t400=50 --tanks tanks.json --out variants/r7.usda`
2. Render: `<isaac-sim>/python.sh external/csi_digital_twin_v1/tools/render_example.py --camera FillCam --stage variants/r7.usda --frame 686 --out renders/r7.png`
3. Keep the timing metadata in the variant (the script copies it). Without it, frames shift.
4. The color -> material map is uncalibrated. Until it is calibrated, say so in any result.

## Do not
- Do not test generalization only on twin renders. That tests the twin against itself.
- Do not edit the twin's own files. Use override layers.

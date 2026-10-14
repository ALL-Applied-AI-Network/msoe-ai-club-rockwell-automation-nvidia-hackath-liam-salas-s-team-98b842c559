---
name: twin-liquid-variant
description: Create a USD liquid-material override for a synthetic recipe or render.
---
# Twin liquid variant

## Inputs
Downloaded twin, valid recipe/tank JSON, usd-core and ignored output path.
Read [twin](../../../docs/twin.md) and the downloaded README.

## Workflow
1. Check the station/material paths, physical inputs and script --help.
2. Run python scripts/twin_liquid_variant.py --twin external/csi_digital_twin_v1 --recipe t100=20,t400=50 --tanks external/tanks.json --out variants/r7.usda.
3. Inspect station reference, recipe metadata and copied timing.
4. Render only with verified renderer flags and an authorized GPU environment.

## Output and validation
Override layer with metadata checks; if rendered, retain camera/frame/settings and
compare measured appearance.

## Limits
Preserve source assets. Material mapping is uncalibrated, not physical absorption.
Recipe mL does not resize the animation. Layer creation does not establish render success.

# Tank station

The procedural station adds T100, T200, T300 and T400 to the public CSI twin. Each
vessel has a shell, conical bottom, independent liquid mesh/material, nameplate and level
strip. T300 stays empty and OFF. A small [static USD asset](../assets/tanks/tank_station.usda)
is checked in; the generator builds the animation and optional full-line composition.

## Readiness and scope

This is an asset and capture pipeline for **synthetic tank-level training experiments**.
It does not establish that a trained model will work on the real tanks. Dimensions,
absolute scale, optical properties and tank-to-nozzle placement are unmeasured. There is
no validated real absolute-volume ground truth, fluid dynamics, residue simulation,
refraction/absorption calibration or detector training in this change.

The camera uses nine manually selected reference landmarks at 640×480. Its fit has
4.472 px coordinate RMSE on those same points. This measures registration to that one
image, not held-out accuracy. [Landmark provenance](../assets/tanks/reference-landmarks.json)
records the source hash. That still is a geometry/camera input and is excluded from any
future independent validation. Room clutter and detailed pipe routing remain approximate.

The nine-state render sweep is a pipeline canary, **not a training/validation/test dataset**.
Keep whole generated sequences in one split; do not randomly split adjacent frames.
For recipe-driven integration, retain the existing [frozen partitions](evaluation.md)
and drainage model identifiers. Do not use real images as training examples. A future
tank estimator needs its own defined target, split manifest and acceptance test before
claiming real-world readiness. T300 must not contribute training targets or metrics.

## Build and run

Use the CPU environment from [quickstart](quickstart.md), then install the optional USD
library. The asset tests skip when USD is absent, so an ordinary CPU test pass alone is
not an asset check.

```sh
python -m pip install 'usd-core==26.11'
python scripts/build_tank_station.py --out outputs/tanks/run1 \
  --twin external/csi_digital_twin_v1
python -m pytest tests/test_tank_station.py -q
```

Choose a fresh output directory. It contains editable USDA, binary USDC, a manifest and,
with `--twin`, `csi_line_with_tanks.usda`. Standalone files contain no external textures.
The combined layer references the original twin; preserve the relative folder layout or
rebuild the composition on the destination host. Source layers are never edited.

Open the stage in Isaac Sim, select `/TankStation/TankCam` and play frames 1–686 at 30 fps.
In the composed line use `/CSI_Line/TankPlacement/TankStation/TankCam`. The default offset
`(-1.1, 0, 0)` metres is in the fill-station frame; change it with `--position x y z`.
This is an assumed placement. Default drawdown is not synchronized with vial filling.

For a local browser preview:

```sh
python scripts/preview_tank_station.py --stage outputs/tanks/run1/tank_station.usda \
  --out outputs/tanks/preview/index.html
python -m http.server 8766 --bind 127.0.0.1 --directory outputs/tanks/preview
```

The self-contained HTML uses the actual USD meshes, camera, colors and integer-frame
volume samples. Play and scrub the timeline. It supports this generated standalone
station only. Simple WebGL lighting/transparency is a geometry preview, not RTX parity
or a training-image source.

## Render and verify image/label pairs

Use the installed Isaac Python, not the CPU Python. On the checked Spark ARM install:

```sh
timeout 600 env OMNI_KIT_ACCEPT_EULA=YES \
  LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  /home/hackathon5/liamreporoot/isaacsim-env/.venv/bin/python \
  scripts/render_tank_station.py --stage outputs/tanks/run1/tank_station.usdc \
  --frames 1,343,686 --subframes 32 --out outputs/tanks/render1
python scripts/check_tank_capture.py --stage outputs/tanks/run1/tank_station.usdc \
  --capture outputs/tanks/render1
```

The preload path is specific to Spark ARM; omit it on x86. `--portable-root PATH` can use
an approved private Kit data/cache/log directory when the shared runtime stalls. It does
not guarantee isolation of every global driver/cache resource. The checked run used the
user-approved task directory, leaving team settings untouched. Capture requires unique
integer frames, rejects constant/missing RGB and inconsistent liquid masks, and exits.

For empty/full and independent-tank checks:

```sh
python scripts/build_tank_station.py --out outputs/tanks/sweep --validation-sweep
# Use the same Isaac invocation with:
# --stage outputs/tanks/sweep/tank_station.usdc --frames 1,2,3,4,5,6,7,8,9
# --out outputs/tanks/sweep-render
python scripts/check_tank_capture.py --stage outputs/tanks/sweep/tank_station.usdc \
  --capture outputs/tanks/sweep-render --sweep
```

Each frame has RGB `.png`, raw `.semantic.npy`, binary `.liquid.png` and indexed
`.instances.png`. Instance IDs are 0 background, 1 T100, 2 T200, 3 T300 (must be absent),
4 T400. Treat the PNG values as integer labels, not displayed colors. `summary.json`
contains frame IDs, exact authored volumes in mL, pixel counts, bounding boxes `[xmin,
ymin,xmax,ymax]` with exclusive maxima, hashes, camera and renderer settings. A completed
schema-version-2 summary also records relative paths and byte hashes for every source
USD layer and referenced asset, including textures. Capture checks these dependencies
before opening the stage and again before writing the summary; the checker recomputes
the full manifest. Missing or changed dependencies fail closed. Keep
`tank_capture_provenance.py` beside the render/check scripts when copying them to Spark.
Older summaries without dependency manifests are rejected and require a new capture;
do not attach current hashes to old images as if recorded at capture time.
A completed summary is written only after every frame passes. The checker compares these labels
with the supplied USD and verifies the image/mask files. It checks internal consistency;
it cannot prove physical truth or detect every possible spatial labeling error.

**Mask policy:** capture RGB with the shells present, then hide only tank shells in a
session-layer annotation pass. Label the projected liquid geometry while retaining
plumbing, nameplates and level strips as occluders. Restore shells before the next RGB.
These masks are geometric liquid support through a translucent wall, not a threshold
on optical visibility, and not an entire unobstructed/amodal silhouette. Shells also
have `doNotCastShadows`; their optics use coverage opacity, not physical transmission.
The source stage remains unchanged. Do not interpret labels as calibrated absorbance.

## Volume and color interface

Default geometry assumes a 70 mm internal radius, 160 mm cylindrical body and 100 mm
cone, giving 2976.1354 mL capacity. Parameters are assumptions. `--radius`, `--body-height`
and `--cone-height` accept metres; large changes require reworking supports and camera.
The 64-sided liquid mesh approximates analytic volume within 0.2% in tests.

Supply complete volume keyframes from the separate drainage model through `--samples`:

```json
[
  {"frame": 1, "volumes": {"T100": 2000, "T200": 2000, "T300": 0, "T400": 2000}},
  {"frame": 686, "volumes": {"T100": 1950, "T200": 1990, "T300": 0, "T400": 1970}}
]
```

These example values are synthetic. Intermediate volumes are linear between keys; meshes
are computed at every integer frame so cone height follows the cube root of volume.
Fractional-frame USD mesh interpolation is approximate; training capture uses integers.
Empty liquid is invisible. Negative, nonfinite, overcapacity or nonzero T300 values fail.

`fluidmix.tank_station.set_volumes(stage, volumes, frame, root=...)` updates levels in a
session or separate override layer. `set_colors(stage, {"T200": (r,g,b)}, root=...)` sets
independent linear RGB appearances; `--colors colors.json` accepts the same mapping.
These are display colors, not concentrations or absorption coefficients. The animation
and color interface supply inputs for INN-35; they do not duplicate its drainage physics.

## Ownership and validation evidence

The change targets the unassigned tank asset work associated with INN-33. INN-35 drainage,
INN-36 future vision estimation, INN-45 fill animation and INN-52 stream segmentation
remain separate. Open PRs #8 and #9 were inspected for overlap; their contract/carryover
changes do not edit these asset modules. One shared test file has a one-line fixture
update to copy assets alongside documentation; PR #9 edits carryover tests elsewhere
in that file. No contributor's working files were changed.
See the [capture evidence report](tank-station-evidence.md) for measured runtime results.

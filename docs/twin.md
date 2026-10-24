# Digital twin (csi_digital_twin_v1)

Evidence status: checked 2026-10-15 against the downloaded twin at `external/csi_digital_twin_v1/`
by reading its files and converting `fill_station.usdc` to text with `usdcat`. Nothing was rendered,
so renderer behavior and the CLI options of `tools/render_example.py` remain unchecked.
`external/csi_digital_twin_v1 2/` is a byte-identical copy.

## Inspection summary (2026-10-15)
- Stations: fill station (FillCam) and tray inspection (CapSealCam), plus MagneMover track and
  facility. **No tank-level camera or tank station**, so the tank_level real camera has no twin match.
- CapSealCam was solved from the real cap_seal video, per the twin README; its preview matches a
  real cap_seal still closely. FillCam is a plain pinhole, and the README makes no claim that it was
  matched to the real camera. Its preview shows the same backdrop and cart, framed differently.
- Animation (time codes 1–686, 30 fps): stream visible 12–368 and 586–661, liquid visible from 21.
  Liquid volume from the mesh: 48.2 mL at 368, 67.3 mL at 686. One blue stage, then one clear stage.
  The real line dispenses T200 (orange) → T400 (blue) → T100 (clear). The twin's 48 + 19 mL shape
  resembles video-5 recipe 7 (50 blue + 20 clear); that is an inference from the volumes.
- Label classes on fill station prims: background, carrier, label, nozzle, track, vial and liquid
  (`Fill_liquid`, class id 2).

USD, Isaac Sim 6.0. Units meters, Z up. No physics; the fill station has a pre-recorded
animation over time codes 1-686 at 30 fps. Read its README first.

## How the twin models liquid color (fill_station.usdc; rechecked 2026-10-15)
- Liquid mesh: `/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid`,
  a cylinder 26.41 mm across. 1 mm of height = 0.548 mL.
- Material: `/FillStation/_materials/USD_blue_liquid_approximation`, one
  `UsdPreviewSurface`: diffuseColor (0.006, 0.2, 0.3) linear, opacity 0.6, ior 1.333.
- The stream uses `USD_jet_approximation`: diffuseColor (0.01, 0.57, 0.67), opacity 0.5.
- Vial: `USD_clear_plastic_approximation`, opacity 0.
- **No time samples on color.** The README says "blue fills, then clear liquid is added",
  but the color stays the same. Real dilution would make it lighter.
- Fill height (measured from mesh): ~88 mm (~48 mL) at frame 368 (blue done),
  ~123 mm (~67 mL) at frame 686.
- **No tanks in the twin.** Only the fill station and cap-seal station.

## What the twin does NOT support
- No absorbance, no path length, no dye concentration. `UsdPreviewSurface` opacity is
  coverage, not Beer-Lambert. So the twin cannot compute mixed colors itself.
- Only one liquid material, shared by every frame.
- Neighbor vials have no liquid of their own.

## How we set liquid color per variant
`scripts/twin_liquid_variant.py` writes a small `.usda` that sub-layers the station file,
copies its 30 fps timing metadata (without it USD defaults to 24 and the frames shift), and
overrides `diffuseColor` and `opacity`. Its direct mode accepts validated unit-interval shader
parameters plus a neutral variant ID. Those parameters are appearance controls only: they are
not measured tank colours, concentrations, residue fractions or physical labels. The legacy
recipe/tank-absorbance mode remains for reproducibility but uses the uncalibrated historical
mapping below. Rendering is still a separate GPU step.

The legacy colour -> material mapping is a **guess** and is not calibrated. To calibrate:
1. Render a sweep of known (color, opacity) values at frame 686.
2. Measure the liquid pixels the same way as for real images.
3. Fit the inverse map so the render matches `expected_color()`.
A better route may be an MDL glass/volume material with real absorption in RTX. I do not
know the exact Isaac Sim 6.0 material names for that; check the Isaac Sim docs.

## Labels
Fill station parts carry `userProperties:synthetic_class`, `synthetic_class_id`,
`synthetic_instance_id`. Use them for liquid masks.

## Variant limits

The script overrides material appearance only. Recipe total mL does not resize the existing
fill animation. Verify material paths and copied timing metadata after creating a layer.
Rendering is a separate GPU step; creating a USD layer is not proof of a successful render.
See [operations](operations.md) for bounded compute and shutdown verification.

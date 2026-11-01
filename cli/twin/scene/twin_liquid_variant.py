"""Make a fill-station variant with a new liquid color, from a recipe.

Runs with plain `pip install usd-core numpy` (no Isaac Sim needed). It writes a
small .usda file that sub-layers the twin's fill_station.usda and overrides only
the liquid material. Render the new file with the twin's render_example.py:

    python cli/twin/scene/twin_liquid_variant.py --twin <csi_digital_twin_v1> \
        --recipe t100=20,t200=0,t400=50 --tanks tanks.json --out variants/r7.usda
    <isaac-sim>/python.sh <twin>/tools/render_example.py --camera FillCam \
        --stage variants/r7.usda --frame 686 --out r7.png

What the twin gives us (checked in fill_station.usdc, twin v1):
  * Liquid mesh  /FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid
    bound to     /FillStation/_materials/USD_blue_liquid_approximation
  * That material is one UsdPreviewSurface: diffuseColor (0.006, 0.2, 0.3)
    linear, opacity 0.6, ior 1.333. No time samples: the color never changes,
    even when clear water is added at frames ~586-661.
  * UsdPreviewSurface has no absorption or path length. "opacity" is coverage,
    not Beer-Lambert. So the twin cannot compute absorbance. We compute it in
    fluidmix.absorbance and push the RESULT into the material.

The mapping from transmittance to (diffuseColor, opacity) below is a STARTING
GUESS. It is not calibrated. Calibrate it by rendering a sweep and comparing
the rendered vial pixels to expected_color() (see docs/twin.md).
"""
import argparse, json, os, sys
import numpy as np
from pxr import Sdf, Usd, UsdGeom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fluidmix.colour.absorbance import mix_absorbance, TANKS
from fluidmix.core.paths import require, twin_dir

LIQ = "/FillStation/FillStation_LocalRoot/Fill_liquid"
SHADER = "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
ML_PER_MM = np.pi * (26.41 / 2) ** 2 / 1000.0  # liquid mesh is a 26.41 mm cylinder


def fill_height_mm(stage, frame):
    """World-space liquid height at a time code (0 before frame 21)."""
    xf = stage.GetPrimAtPath(LIQ)
    if not UsdGeom.Imageable(xf).ComputeVisibility(frame) == UsdGeom.Tokens.inherited:
        return 0.0
    pts = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(LIQ + "/Fill_liquid")).GetPointsAttr().Get(frame))
    M = np.array(UsdGeom.Xformable(xf).ComputeLocalToWorldTransform(frame))
    z = (np.c_[pts, np.ones(len(pts))] @ M)[:, 2]
    return float((z.max() - z.min()) * 1000)


def material_from_absorbance(a):
    """GUESS: transmittance as color, mean absorbance as opacity. Calibrate me."""
    t = 10.0 ** (-np.asarray(a, float))
    opacity = float(np.clip(0.25 + 0.5 * np.mean(a), 0.25, 0.95))
    return tuple(float(x) for x in t), opacity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", help="folder csi_digital_twin_v1 (default: $FLUIDMIX_DATA/csi_digital_twin_v1)")
    ap.add_argument("--recipe", required=True, help="e.g. t100=20,t200=0,t400=50")
    ap.add_argument("--tanks", required=True, help="json {t200:[Ar,Ag,Ab], t400:[...]}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ml = {k: float(v) for k, v in (p.split("=") for p in a.recipe.split(","))}
    tank_a = {k: np.array(v, float) for k, v in json.load(open(a.tanks)).items()}
    twin = a.twin or twin_dir()
    station = os.path.abspath(require(os.path.join(twin, "stations/fill_station/fill_station.usda"), "twin fill_station.usda"))

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    layer = Sdf.Layer.CreateNew(out)
    layer.subLayerPaths.append(os.path.relpath(station, os.path.dirname(out)))
    # Copy the timing metadata. Without it the stage defaults to 24 time codes/s
    # and USD rescales the 30 fps animation, so --frame 686 lands on the wrong pose.
    src = Sdf.Layer.FindOrOpen(station)
    layer.timeCodesPerSecond, layer.framesPerSecond = src.timeCodesPerSecond, src.framesPerSecond
    layer.startTimeCode, layer.endTimeCode = src.startTimeCode, src.endTimeCode
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(layer)
    shader = stage.GetPrimAtPath(SHADER)
    color, opacity = material_from_absorbance(mix_absorbance(ml, tank_a))
    shader.GetAttribute("inputs:diffuseColor").Set(color)
    shader.GetAttribute("inputs:opacity").Set(opacity)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/FillStation"))
    layer.customLayerData = {"fluidmix_recipe": a.recipe}
    layer.Save()
    h = fill_height_mm(stage, 686)
    print(f"wrote {out}: color {np.round(color, 3)}, opacity {opacity:.2f}, "
          f"twin fill at 686 = {h:.1f} mm = {h * ML_PER_MM:.1f} mL")


if __name__ == "__main__":
    main()

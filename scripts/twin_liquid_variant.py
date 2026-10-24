"""Make a fill-station variant with a new synthetic liquid material.

Runs with plain `pip install usd-core numpy` (no Isaac Sim needed). It writes a
small .usda file that sub-layers the twin's fill_station.usda and overrides only
the liquid material. Render the new file with the twin's render_example.py:

    python scripts/twin_liquid_variant.py --twin <csi_digital_twin_v1> \
        --recipe t100=20,t200=0,t400=50 --tanks tanks.json --out variants/r7.usda
    python scripts/twin_liquid_variant.py --twin <csi_digital_twin_v1> \
        --material-color 0.117,0.18,0.2275 --opacity 0.6 \
        --variant-id M03 --out variants/M03.usda
    <isaac-sim>/python.sh <twin>/tools/render_example.py --camera FillCam \
        --stage variants/r7.usda --frame 686 --out r7.png

What the twin gives us (checked in fill_station.usdc, twin v1):
  * Liquid mesh  /FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid
    bound to     /FillStation/_materials/USD_blue_liquid_approximation
  * That material is one UsdPreviewSurface: diffuseColor (0.006, 0.2, 0.3)
    linear, opacity 0.6, ior 1.333. No time samples: the color never changes,
    even when clear water is added at frames ~586-661.
  * UsdPreviewSurface has no absorption or path length. "opacity" is coverage,
    not Beer-Lambert. The direct mode accepts renderer parameters without assigning
    physical meaning. The legacy recipe mode remains for reproducibility only.

The legacy mapping from transmittance to (diffuseColor, opacity) below is an
uncalibrated historical guess. Do not use it as a plant calibration or label.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fluidmix.absorbance import mix_absorbance
from fluidmix.paths import require, twin_dir
from fluidmix.twin_material import parse_linear_rgb, select_material_mode

LIQ = "/FillStation/FillStation_LocalRoot/Fill_liquid"
SHADER = "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
ML_PER_MM = np.pi * (26.41 / 2) ** 2 / 1000.0  # liquid mesh is a 26.41 mm cylinder


def fill_height_mm(stage, frame):
    """World-space liquid height at a time code (0 before frame 21)."""
    from pxr import UsdGeom

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
    ap.add_argument("--recipe", help="legacy mode: e.g. t100=20,t200=0,t400=50")
    ap.add_argument("--tanks", help="legacy mode: json {t200:[Ar,Ag,Ab], t400:[...]}")
    ap.add_argument(
        "--material-color", type=parse_linear_rgb, metavar="R,G,B",
        help="direct mode: unitless linear diffuse RGB channels in [0,1]",
    )
    ap.add_argument("--opacity", type=float, help="direct mode: unitless opacity in [0,1]")
    ap.add_argument("--variant-id", help="direct mode: neutral provenance ID")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    try:
        mode, direct = select_material_mode(
            recipe=a.recipe,
            tanks=a.tanks,
            material_color=a.material_color,
            opacity=a.opacity,
            variant_id=a.variant_id,
        )
    except (TypeError, ValueError) as exc:
        ap.error(str(exc))

    if mode == "direct_material":
        color, opacity = direct.color, direct.opacity
        layer_metadata = direct.layer_metadata()
    else:
        ml = {k: float(v) for k, v in (p.split("=") for p in a.recipe.split(","))}
        with open(a.tanks) as stream:
            tank_a = {k: np.array(v, float) for k, v in json.load(stream).items()}
        color, opacity = material_from_absorbance(mix_absorbance(ml, tank_a))
        layer_metadata = {
            "fluidmix_variant_mode": "legacy_absorbance_guess",
            "fluidmix_recipe": a.recipe,
        }

    twin = a.twin or twin_dir()
    station = os.path.abspath(require(os.path.join(twin, "stations/fill_station/fill_station.usda"), "twin fill_station.usda"))

    from pxr import Sdf, Usd

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
    shader.GetAttribute("inputs:diffuseColor").Set(color)
    shader.GetAttribute("inputs:opacity").Set(opacity)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/FillStation"))
    layer.customLayerData = layer_metadata
    layer.Save()
    h = fill_height_mm(stage, 686)
    print(f"wrote {out}: mode {mode}, color {np.round(color, 4)}, opacity {opacity:.3f}, "
          f"twin fill at 686 = {h:.1f} mm = {h * ML_PER_MM:.1f} mL")


if __name__ == "__main__":
    main()

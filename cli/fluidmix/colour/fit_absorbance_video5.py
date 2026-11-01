"""Fit per-tank absorbance from video-5 vial colors, then test it.

Run:  python cli/fluidmix/colour/fit_absorbance_video5.py
Uses data/video5_vial_colors.csv (rough colors, see docs/data.md) and
data/video5_vial_recipes.csv. Clean vials = 2nd and 3rd of each batch.
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fluidmix.colour.absorbance import (srgb_to_linear, linear_to_srgb, delta_e, expected_color,
                                 fit_tank_absorbance, fit_carryover_fraction)

D = os.path.join(os.path.dirname(__file__), "..", "data")
rec = {int(r["vial"]): r for r in csv.DictReader(open(f"{D}/video5_vial_recipes.csv"))}
col = {int(r["vial"]): np.array([float(r["r"]), float(r["g"]), float(r["b"])])
       for r in csv.DictReader(open(f"{D}/video5_vial_colors.csv"))}
ml = {v: {"t100": float(r["t100_ml"] or 0), "t200": float(r["t200_ml"] or 0),
          "t400": float(r["t400_ml"] or 0)} for v, r in rec.items()}
first = {v for v in rec if v == 1 or rec[v]["recipe_id"] != rec[v - 1]["recipe_id"]}
clean = [v for v in rec if v not in first]

tank_a, i0, r2 = fit_tank_absorbance([(ml[v], srgb_to_linear(col[v])) for v in clean])
np.set_printoptions(precision=3, suppress=True)
print("Fit on", len(clean), "clean vials")
print("  A(T-200 orange) R,G,B =", tank_a["t200"])
print("  A(T-400 blue)   R,G,B =", tank_a["t400"])
print("  I0 (clear vial) sRGB  =", linear_to_srgb(i0))
print("  R^2 per channel       =", r2)
if min(tank_a["t200"].min(), tank_a["t400"].min()) < 0:
    print("  WARNING: negative absorbance. Dye cannot add light, so this is not physical.")
    print("  Likely cause: the colors are not transmission measurements (no clear-vial I0,")
    print("  mask picks the most saturated pixels, camera auto exposure). See docs/absorbance.md.")

# leave-one-recipe-out: does the model predict a recipe it never saw?
print("\nLeave-one-recipe-out (clean vials):")
errs = []
for rid in sorted({rec[v]["recipe_id"] for v in clean}, key=int):
    train = [v for v in clean if rec[v]["recipe_id"] != rid]
    test = [v for v in clean if rec[v]["recipe_id"] == rid]
    ta, i0k, _ = fit_tank_absorbance([(ml[v], srgb_to_linear(col[v])) for v in train])
    for v in test:
        pred = linear_to_srgb(expected_color(ml[v], ta, i0k))
        e = delta_e(pred, col[v]); errs.append(e)
        print(f"  R{rid:>2} vial {v:>2}: measured {col[v].astype(int)}  predicted {pred.round().astype(int)}  dE {e:5.1f}")
print(f"  median dE {np.median(errs):.1f}, max {np.max(errs):.1f}")

print("\nFirst vials after a change: fit 'new + x% of old'")
for v in sorted(first - {1}):
    frac, res = fit_carryover_fraction(srgb_to_linear(col[v]), ml[v], ml[v - 1], tank_a, i0)
    print(f"  vial {v:>2} (R{rec[v-1]['recipe_id']}->R{rec[v]['recipe_id']}): old fraction {frac:4.0%}, residual A {res:.2f}")

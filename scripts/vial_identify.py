"""INN-52: how well does a vial's colour say what is in it? Video 5, inspection only.

    python scripts/vial_identify.py                     # both colour sources
    python scripts/vial_identify.py --source video      # data/video5_vial_final_colours.csv only

Two tests per colour source, each split into the first vial after a recipe change (carryover)
and the steady 2nd/3rd vials:
  A) repeatability: the nearest other vial in CIELAB belongs to the same recipe (leave one out);
  B) unseen recipe: fit colour = fractions @ W (mixture model, no intercept) on the other nine
     recipes' steady vials, then rank the ten-recipe menu for the held-out vial and estimate its
     volume fractions on a 5% simplex grid. No tank colours are preset; W is fitted each time.

Colour sources:
  inherited  data/video5_vial_colors.csv: most saturated 0.7% of pixels, no empty-vial reference.
             Feature: -log10(linear sRGB), an absorbance with an unknown, fixed I0.
  video      data/video5_vial_final_colours.csv (scripts/stream_sequence.py --store): median of
             the liquid region, with the same region of the pre-pour frame as I0. Feature:
             absorbance(vial, empty vial). The empty vial holds air, not water, so this is not the
             calibrated absorbance in docs/absorbance.md; do not use it for plant predictions.

30 vials and 10 recipes: counts are small, and the feature choice was made on these vials.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.absorbance import absorbance, srgb_to_lab, srgb_to_linear
from fluidmix.paths import REPO

TANKS = ("t100_ml", "t200_ml", "t400_ml")
GRID_STEP = 20                   # simplex grid in 1/20 steps (5%)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load(source):
    """[{vial, recipe, frac, lab, feat, first}] in vial order."""
    colours = {r["vial"]: r for r in read(REPO / ("data/video5_vial_colors.csv" if source == "inherited"
                                                  else "data/video5_vial_final_colours.csv"))}
    out, seen = [], set()
    for r in read(REPO / "data/video5_vial_recipes.csv"):
        c = colours[r["vial"]]
        ml = np.array([float(r[t] or 0) for t in TANKS])
        if source == "inherited":
            rgb = np.array([float(c[k]) for k in "rgb"])
            feat = absorbance(srgb_to_linear(rgb), 1.0)
        else:
            rgb = np.array([float(c[f"vial_{k}"]) for k in "rgb"])
            empty = np.array([float(c[f"empty_{k}"]) for k in "rgb"])
            feat = absorbance(srgb_to_linear(rgb), srgb_to_linear(empty))
        out.append({"vial": int(r["vial"]), "recipe": r["recipe_id"], "frac": ml / ml.sum(),
                    "lab": srgb_to_lab(rgb), "feat": feat, "first": r["recipe_id"] not in seen})
        seen.add(r["recipe_id"])
    return out


def fit_mixture(fracs, feats):
    """W with feats ~ fracs @ W: one feature vector per tank, mixed by volume fraction."""
    return np.linalg.lstsq(np.asarray(fracs), np.asarray(feats), rcond=None)[0]


def simplex_grid(step=GRID_STEP):
    return np.array([(a, b, step - a - b) for a in range(step + 1) for b in range(step + 1 - a)],
                    float) / step


def rank_recipes(feat, menu, W, scale):
    """Recipe ids, best first, by scaled distance between predicted and measured features."""
    return sorted(menu, key=lambda r: np.linalg.norm((menu[r] @ W - feat) / scale))


def estimate_fractions(feat, W, scale, grid):
    return grid[np.argmin(np.linalg.norm((grid @ W - feat) / scale, axis=1))]


def evaluate(vials):
    menu = {}
    for v in vials:
        menu.setdefault(v["recipe"], v["frac"])
    grid = simplex_grid()
    rows = []
    for v in vials:
        others = [u for u in vials if u is not v]
        nearest = min(others, key=lambda u: np.linalg.norm(u["lab"] - v["lab"]))
        train = [u for u in vials if u["recipe"] != v["recipe"] and not u["first"]]
        feats = np.array([u["feat"] for u in train])
        W = fit_mixture([u["frac"] for u in train], feats)
        scale = feats.std(axis=0) + 1e-9
        ranked = rank_recipes(v["feat"], menu, W, scale)
        est = estimate_fractions(v["feat"], W, scale, grid)
        rows.append({"vial": v["vial"], "recipe": v["recipe"], "first": v["first"],
                     "nearest_recipe": nearest["recipe"], "pick": ranked[0],
                     "rank": ranked.index(v["recipe"]) + 1, "est": est,
                     "frac_error": float(np.abs(est - v["frac"]).max())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("inherited", "video", "both"), default="both")
    a = ap.parse_args()
    for source in (("inherited", "video") if a.source == "both" else (a.source,)):
        rows = evaluate(load(source))
        print(f"== {source} vial colours")
        for group, sel in (("steady 2nd/3rd vial", lambda r: not r["first"]),
                           ("first after change", lambda r: r["first"])):
            rs = [r for r in rows if sel(r)]
            print(f"   {group:20s} A nearest vial same recipe {sum(r['nearest_recipe'] == r['recipe'] for r in rs)}"
                  f"/{len(rs)} | B unseen recipe right {sum(r['pick'] == r['recipe'] for r in rs)}/{len(rs)}, "
                  f"top-3 {sum(r['rank'] <= 3 for r in rs)}/{len(rs)}, "
                  f"median worst-tank fraction error {np.median([r['frac_error'] for r in rs]):.2f}")
        for r in rows:
            if not r["first"]:
                print(f"     vial {r['vial']:2d} recipe {r['recipe']:>2} -> {r['pick']:>2} (rank {r['rank']:2d}) "
                      f"T100/T200/T400 est {np.round(r['est'], 2)}")


if __name__ == "__main__":
    main()

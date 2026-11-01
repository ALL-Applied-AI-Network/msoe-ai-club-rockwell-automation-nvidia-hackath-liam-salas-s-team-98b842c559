"""INN-52: does the liquid level in the vial measure volume, including clear (T-100) liquid?

    python scripts/vial_level.py        # reads data/video5_stream_vial_colours.csv

The level is the top row of the liquid column after each burst (scripts/stream_sequence.py,
liquid_level), measured from brightness and colour change against the empty vial, so a clear
dose counts. Two checks on bursts whose tank matches the recipe:
  1) volume from level: fit row = a - b * cumulative mL on clear-view (below_holder) readings
     from the other recipes, invert it for each held-out reading, and report the error in mL;
     above_holder readings (holder top and labels can hide the surface) are scored separately;
  2) clear liquid: the same errors split by the tank dosed last. If the level sees clear liquid
     as well as coloured, readings right after a T-100 dose are no worse than the others.
Real video, inspection only. The level threshold and regions were chosen on these vials.
"""
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO


def load(path=REPO / "data/video5_stream_vial_colours.csv"):
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["cumulative_ml"] and r["level_row"]]
    return [{"vial": r["vial"], "recipe": r["recipe_id"], "tank": r["tank"], "burst": int(r["burst"]),
             "ml": float(r["cumulative_ml"]), "row": float(r["level_row"]), "zone": r["level_zone"]} for r in rows]


def fit_line(ml, row):
    """row = a - b * ml by least squares; returns (a, b)."""
    slope, intercept = np.polyfit(np.asarray(ml, float), np.asarray(row, float), 1)
    return intercept, -slope


def volume_errors(readings):
    """Leave-one-recipe-out |estimated - logged| cumulative mL, as [(reading, error)]."""
    clear = [r for r in readings if r["zone"] == "below_holder"]
    out = []
    for r in readings:
        train = [c for c in clear if c["recipe"] != r["recipe"]]
        a, b = fit_line([c["ml"] for c in train], [c["row"] for c in train])
        out.append((r, abs((a - r["row"]) / b - r["ml"])))
    return out


def summarise(errors):
    return (f"median error {np.median(errors):.1f} mL, 90th percentile {np.percentile(errors, 90):.1f} mL, "
            f"n={len(errors)}")


def main():
    readings = load()
    zones = {z: sum(r["zone"] == z for r in readings) for z in ("below_holder", "above_holder")}
    a, b = fit_line(*zip(*[(r["ml"], r["row"]) for r in readings if r["zone"] == "below_holder"]))
    print(f"{len(readings)} burst readings with a matched tank and a level: {zones}")
    print(f"clear-view fit, all recipes: row = {a:.0f} - {b:.2f} x cumulative mL")
    errors = volume_errors(readings)
    for zone in ("below_holder", "above_holder"):
        print(f"volume from level, {zone:12s} (leave one recipe out): "
              f"{summarise([e for r, e in errors if r['zone'] == zone])}")
    for tank in sorted({r["tank"] for r, _ in errors}):
        errs = [e for r, e in errors if r["zone"] == "below_holder" and r["tank"] == tank]
        if errs:
            print(f"  clear view, last dose {tank}: {summarise(errs)}")


if __name__ == "__main__":
    main()

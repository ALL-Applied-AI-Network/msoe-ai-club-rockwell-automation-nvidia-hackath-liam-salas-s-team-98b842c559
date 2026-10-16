"""Freeze the INN-7 partitions: twin recipes for train/val/test, and the real video-5 check frames.

    python scripts/make_splits.py                 # check the committed files are current
    python scripts/make_splits.py --write         # (re)write them; bump SPEC_VERSION first

Design (decided 2026-10-16, see docs/evaluation.md):
- Twin data trains, validates and tests. Twin recipes are every valid recipe on a 5 mL grid
  within plant limits, minus any recipe near one of the 10 real video-5 recipes. The real
  recipes are reserved for the final real-video validation.
- Roles are assigned per mixture-ratio cell (volume fractions rounded to 0.1), so recipes with
  the same colour ratio never sit in two roles. Colour depends mostly on the ratio. Val and test
  recipes within BUFFER_FRAC of an earlier role's ratio become unused 'buffer' recipes.
- data/real_check_frames.csv lists every video-5 vial_fill still with its eligibility for the
  real validation. It needs outputs/video5_frame_labels_aligned.csv from
  scripts/align_still_labels.py; without it only the twin files are written or checked.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO, outputs_root

SPEC_VERSION = "v1"
SEED = 20261016
TANKS = ("t100", "t200", "t400")
GRID = [0] + list(range(10, 65, 5))  # mL per tank: unused, or 10-60 in 5 mL steps
MAX_TOTAL, MIN_TOTAL, MAX_T200 = 80, 30, 50  # plant limits (docs/data.md); 30 = smallest real recipe
ROLE_SHARES = (("train", 0.70), ("val", 0.15), ("test", 0.15))
NEAR_ML = 5        # a twin recipe within this many mL of a real recipe on every tank is too close
NEAR_FRAC = 0.05   # ... or within this of its volume fractions on every tank
CELL = 0.1         # mixture-ratio cell width for role assignment
BUFFER_FRAC = 0.03 # val/test recipes this close (every tank's fraction) to an earlier role become 'buffer'
BOUNDARY_S = 2.0   # real stills this close to a fill boundary are uncertain (docs/labels.md)
SEED_RANGES = {"train": [0, 999_999], "val": [1_000_000, 1_999_999], "test": [2_000_000, 2_999_999]}

RECIPES_CSV = REPO / "data/sim_recipes.csv"
TRANSITIONS_CSV = REPO / "data/sim_transitions.csv"
REAL_CSV = REPO / "data/real_check_frames.csv"
SPEC_JSON = REPO / "data/sim_split_spec.json"


def fractions(ml):
    total = sum(ml.values())
    return {t: ml[t] / total for t in TANKS}


def near(a, b):
    """True when two recipes are too similar to sit on different sides of a split."""
    if all(abs(a[t] - b[t]) <= NEAR_ML for t in TANKS):
        return True
    fa, fb = fractions(a), fractions(b)
    return all(abs(fa[t] - fb[t]) <= NEAR_FRAC for t in TANKS)


def rml(row):
    return {t: row[f"{t}_ml"] for t in TANKS}


def ratio_close(fa, fb):
    return all(abs(fa[t] - fb[t]) <= BUFFER_FRAC for t in TANKS)


def cell(ml):
    f = fractions(ml)
    return "-".join(f"{round(f[t] / CELL) * CELL:.1f}" for t in TANKS)


def real_recipes():
    out = {}
    for r in csv.DictReader((REPO / "data/video5_vial_recipes.csv").open()):
        out[int(r["recipe_id"])] = {t: int(r[f"{t}_ml"] or 0) for t in TANKS}
    return out


def build():
    real = real_recipes()
    candidates = []
    for a in GRID:
        for b in GRID:
            for c in GRID:
                ml = dict(zip(TANKS, (a, b, c)))
                total = a + b + c
                if sum(v > 0 for v in ml.values()) < 2 or not MIN_TOTAL <= total <= MAX_TOTAL or b > MAX_T200:
                    continue
                candidates.append(ml)
    kept, excluded = [], 0
    for ml in candidates:
        if any(near(ml, r) for r in real.values()):
            excluded += 1
        else:
            kept.append(ml)
    cells = sorted({cell(ml) for ml in kept})
    rng = random.Random(SEED)
    rng.shuffle(cells)
    role_of, start = {}, 0
    for i, (role, share) in enumerate(ROLE_SHARES):
        end = len(cells) if i == len(ROLE_SHARES) - 1 else start + round(share * len(cells))
        for c in cells[start:end]:
            role_of[c] = role
        start = end
    recipes = []
    for n, ml in enumerate(sorted(kept, key=lambda m: tuple(m[t] for t in TANKS)), start=1):
        recipes.append({"sim_recipe_id": f"S{n:04d}", "role": role_of[cell(ml)], "cell": cell(ml),
                        **{f"{t}_ml": ml[t] for t in TANKS}, "total_ml": sum(ml.values())})
    # Neighbouring cells can hold nearly the same ratio. Drop val recipes close to train, then test
    # recipes close to train or val, so test ratios are not near anything seen before.
    for role, earlier in (("val", {"train"}), ("test", {"train", "val"})):
        ref = [fractions(rml(r)) for r in recipes if r["role"] in earlier]
        for r in recipes:
            if r["role"] == role and any(ratio_close(fractions(rml(r)), f) for f in ref):
                r["role"] = "buffer"
    # One carryover transition per recipe: a random predecessor from the same role.
    transitions = []
    for role, _ in ROLE_SHARES:
        ids = [r["sim_recipe_id"] for r in recipes if r["role"] == role]
        for rid in ids:
            prev = rng.choice([x for x in ids if x != rid])
            transitions.append({"role": role, "prev_sim_recipe_id": prev, "sim_recipe_id": rid})
    spec = {
        "version": SPEC_VERSION, "seed": SEED, "tanks": list(TANKS), "grid_ml": GRID,
        "limits": {"min_total_ml": MIN_TOTAL, "max_total_ml": MAX_TOTAL, "max_t200_ml": MAX_T200,
                   "min_tanks_used": 2, "t300": "unavailable; never used"},
        "real_recipes_reserved": {str(k): v for k, v in sorted(real.items())},
        "near_rule": {"ml_per_tank": NEAR_ML, "fraction_per_tank": NEAR_FRAC},
        "role_unit": f"mixture-ratio cell, fractions rounded to {CELL}",
        "buffer_rule": f"val within {BUFFER_FRAC} of train, test within {BUFFER_FRAC} of train or val "
                       "(every tank's volume fraction) become role 'buffer' and are never rendered",
        "role_shares": dict(ROLE_SHARES), "render_seed_ranges": SEED_RANGES,
        "defects": {
            "wrong_volume_ml": None, "carryover_fraction": None,
            "note": "Definitions fixed 2026-10-16 (docs/evaluation.md); numeric thresholds are set "
                    "on val and frozen before any test scoring.",
        },
        "counts": {"grid_candidates": len(candidates), "excluded_near_real": excluded,
                   "kept": len(kept), "cells": len(cells),
                   **{role: sum(r["role"] == role for r in recipes) for role in ("train", "val", "test", "buffer")}},
    }
    return spec, recipes, transitions


def real_frames():
    aligned = outputs_root() / "video5_frame_labels_aligned.csv"
    if not aligned.is_file():
        return None
    prev = {int(r["vial"]): r["prev_recipe_id"]
            for r in csv.DictReader((REPO / "data/video5_transitions.csv").open())}
    change = {int(r["vial"]): r["recipe_changed"]
              for r in csv.DictReader((REPO / "data/video5_transitions.csv").open())}
    rows = []
    for r in csv.DictReader(aligned.open()):
        reason = ""
        if r["status"] != "in_video":
            reason = r["status"]
        elif r["boundary_margin_s"] and float(r["boundary_margin_s"]) < BOUNDARY_S:
            reason = "within_2s_of_fill_boundary"
        elif r["vial"] and prev.get(int(r["vial"])) == "unknown":
            reason = "previous_recipe_unknown"
        rows.append({"filename": r["filename"], "timestamp": r["timestamp"], "video_s": r["video_s"],
                     "frame": r["frame"], "in_fill": r["in_fill"], "vial": r["vial"], "recipe_id": r["recipe_id"],
                     "recipe_changed": change.get(int(r["vial"]), "") if r["vial"] else "",
                     "role": "real_validation", "eligible": int(reason == ""), "exclusion_reason": reason})
    return rows


def render(rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the split files instead of checking them")
    a = ap.parse_args()
    spec, recipes, transitions = build()
    files = {
        SPEC_JSON: json.dumps(spec, indent=1) + "\n",
        RECIPES_CSV: render(recipes, list(recipes[0])),
        TRANSITIONS_CSV: render(transitions, list(transitions[0])),
    }
    real = real_frames()
    if real is not None:
        files[REAL_CSV] = render(real, list(real[0]))
    else:
        print("note: outputs/video5_frame_labels_aligned.csv missing; real_check_frames.csv not built or checked")
    stale = []
    for path, text in files.items():
        if a.write:
            path.write_text(text)
        elif not path.is_file() or path.read_text() != text:
            stale.append(path.name)
    c = spec["counts"]
    print(f"{SPEC_VERSION}: {c['grid_candidates']} grid recipes, {c['excluded_near_real']} near a real recipe, "
          f"{c['kept']} kept in {c['cells']} cells; train {c['train']}, val {c['val']}, test {c['test']}, "
          f"buffer {c['buffer']}; "
          f"{len(transitions)} transitions")
    if real is not None:
        elig = [r for r in real if r["eligible"]]
        print(f"real check: {len(real)} stills, {len(elig)} eligible, vials {len({r['vial'] for r in elig if r['vial']})}")
    if a.write:
        print("wrote", ", ".join(p.name for p in files))
    elif stale:
        raise SystemExit(f"out of date: {', '.join(stale)}; rerun with --write after bumping SPEC_VERSION")
    else:
        print("split files are up to date")


if __name__ == "__main__":
    main()

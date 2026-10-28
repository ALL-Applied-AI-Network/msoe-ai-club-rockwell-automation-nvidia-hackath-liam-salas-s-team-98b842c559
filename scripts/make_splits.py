"""Freeze the INN-7 partitions: twin recipes for train/val/test, and the real video-5 check frames.

    python scripts/make_splits.py                 # check the committed files are current
    python scripts/make_splits.py --write         # write the frozen v2 and additive v3 files

Design (v2 decided 2026-10-16; additive v3 decided 2026-10-28; see docs/evaluation.md):
- Twin data trains, validates and tests. Twin recipes are every valid recipe on a 5 mL grid
  within plant limits, minus any recipe near one of the 10 real video-5 recipes. The real
  recipes are reserved for the final real-video validation.
- Roles are assigned per mixture-ratio cell (volume fractions rounded to 0.1), so recipes with
  the same colour ratio never sit in two roles. Colour depends mostly on the ratio. Val and test
  recipes within BUFFER_FRAC of an earlier role's ratio become unused 'buffer' recipes.
- data/real_check_frames.csv lists every video-5 vial_fill still with its eligibility for the
  real validation. It needs outputs/video5_frame_labels_aligned.csv from
  scripts/align_still_labels.py; without it only the twin files are written or checked.
- v3 keeps those v2 bytes and roles unchanged, then adds six seeded off-grid WP1 test recipes in
  explicitly versioned files. Historical consumers continue to bind the unversioned v2 files.
"""
import argparse
import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO, outputs_root
from fluidmix.test_recipe_generator import build_test_recipe_plan

LEGACY_SPEC_VERSION = "v2"  # byte-frozen input to accepted historical artifacts
SPEC_VERSION = "v3"  # additive 1 mL WP1 test extension; v2 roles and bytes stay unchanged
SEED = 20261016
TANKS = ("t100", "t200", "t400")
GRID = [0] + list(range(10, 65, 5))  # mL per tank: unused, or 10-60 in 5 mL steps
MAX_TOTAL, MIN_TOTAL, MAX_T200 = 80, 30, 50  # plant limits (docs/data.md); 30 = smallest real recipe
ROLE_SHARES = (("train", 0.70), ("val", 0.15), ("test", 0.15))
NEAR_ML = 3        # recipes within this many mL on every tank are too close: never near a real recipe,
                   # never in two roles. On the 5 mL grid this blocks identical volumes only.
NEAR_FRAC = 0.05   # ... or within this of its volume fractions on every tank
CELL = 0.1         # mixture-ratio cell width for role assignment
BUFFER_FRAC = 0.03 # val/test recipes this close (every tank's fraction) to an earlier role become 'buffer'
BOUNDARY_S = 2.0   # real stills this close to a fill boundary are uncertain (docs/labels.md)
SEED_RANGES = {"train": [0, 999_999], "val": [1_000_000, 1_999_999], "test": [2_000_000, 2_999_999]}

RECIPES_CSV = REPO / "data/sim_recipes.csv"
TRANSITIONS_CSV = REPO / "data/sim_transitions.csv"
REAL_CSV = REPO / "data/real_check_frames.csv"
SPEC_JSON = REPO / "data/sim_split_spec.json"
V3_RECIPES_CSV = REPO / "data/sim_recipes_v3.csv"
V3_TRANSITIONS_CSV = REPO / "data/sim_transitions_v3.csv"
V3_SPEC_JSON = REPO / "data/sim_split_spec_v3.json"


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


def ml_close(a, b):
    return all(abs(a[t] - b[t]) <= NEAR_ML for t in TANKS)


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
        ref_ml = [rml(r) for r in recipes if r["role"] in earlier]
        for r in recipes:
            if r["role"] == role and any(ml_close(rml(r), m) for m in ref_ml):
                r["role"] = "buffer"
    # One carryover transition per recipe: a random predecessor from the same role.
    transitions = []
    for role, _ in ROLE_SHARES:
        ids = [r["sim_recipe_id"] for r in recipes if r["role"] == role]
        for rid in ids:
            prev = rng.choice([x for x in ids if x != rid])
            transitions.append({"role": role, "prev_sim_recipe_id": prev, "sim_recipe_id": rid})
    spec = {
        "version": LEGACY_SPEC_VERSION, "seed": SEED, "tanks": list(TANKS), "grid_ml": GRID,
        "limits": {"min_total_ml": MIN_TOTAL, "max_total_ml": MAX_TOTAL, "max_t200_ml": MAX_T200,
                   "min_tanks_used": 2, "t300": "unavailable; never used"},
        "real_recipes_reserved": {str(k): v for k, v in sorted(real.items())},
        "near_rule": {"ml_per_tank": NEAR_ML, "fraction_per_tank": NEAR_FRAC},
        "role_unit": f"mixture-ratio cell, fractions rounded to {CELL}",
        "buffer_rule": f"val within {BUFFER_FRAC} in fraction or {NEAR_ML} mL of train, test within the same "
                       "of train or val (every tank) become role 'buffer' and are never rendered",
        "version_history": {"v1": "2026-10-16, NEAR_ML 5, ratio buffer only",
                            "v2": "2026-10-16, NEAR_ML 3 and mL rule across roles (review finding 1)"},
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


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v3():
    """Build the additive WP1 v3 split while preserving the v2 byte contract."""

    base_spec, base_recipes, base_transitions = build()
    plan = build_test_recipe_plan(repo_root=REPO)
    recipes = deepcopy(base_recipes)
    transitions = deepcopy(base_transitions)
    for run in plan["runs"]:
        generated = []
        for recipe in run["recipes"]:
            programmed = recipe["programmed_ml"]
            row = {
                "sim_recipe_id": recipe["sim_recipe_id"],
                "role": "test",
                "cell": recipe["cell"],
                **{f"{tank}_ml": programmed[tank] for tank in TANKS},
                "total_ml": recipe["total_ml"],
            }
            recipes.append(row)
            generated.append(recipe["sim_recipe_id"])
        for previous, current in (
            (generated[0], generated[0]),
            (generated[0], generated[1]),
            (generated[1], generated[2]),
        ):
            transitions.append(
                {
                    "role": "test",
                    "prev_sim_recipe_id": previous,
                    "sim_recipe_id": current,
                }
            )

    recipe_text = render(recipes, list(recipes[0]))
    transition_text = render(transitions, list(transitions[0]))
    base_counts = deepcopy(base_spec["counts"])
    spec = deepcopy(base_spec)
    spec["version"] = SPEC_VERSION
    legacy_generation = {
        "seed": spec.pop("seed"),
        "grid_ml": spec.pop("grid_ml"),
        "role_shares": spec.pop("role_shares"),
    }
    spec["generation"] = {
        "legacy_v2": legacy_generation,
        "v3_test_extension": {
            "seed": plan["generator_seed"],
            "grid_step_ml": 1,
            "tank_volume_ml": {
                "unused": 0,
                "minimum_when_used": 10,
                "maximum": 60,
            },
            "role_assignment": "test",
            "legacy_roles_edited": False,
        },
    }
    spec["parent_split"] = {
        "version": LEGACY_SPEC_VERSION,
        "role_rows_unchanged": True,
        "source_sha256": {
            "data/sim_recipes.csv": _file_sha256(RECIPES_CSV),
            "data/sim_transitions.csv": _file_sha256(TRANSITIONS_CSV),
            "data/sim_split_spec.json": _file_sha256(SPEC_JSON),
        },
    }
    spec["version_history"]["v3"] = (
        "2026-10-28, additive seeded 1 mL WP1 test recipes; v2 rows and roles unchanged"
    )
    spec["test_recipe_extension"] = {
        "reason": (
            "every valid 5 mL-grid recipe already has a v2 role; WP1 needs new held-out "
            "test recipes without changing a scored role"
        ),
        "generator_seed": plan["generator_seed"],
        "generator_artifact_type": plan["artifact_type"],
        "plan_canonical_sha256": canonical_sha256(plan),
        "candidate_count": plan["selection"]["candidate_count"],
        "selected_count": sum(len(run["recipes"]) for run in plan["runs"]),
        "grid_step_ml": 1,
        "unused_cell_definition": (
            "no legacy non-buffer role; legacy buffer rows remain unchanged and unrendered"
        ),
        "recipe_path": "data/sim_recipes_v3.csv",
        "transition_path": "data/sim_transitions_v3.csv",
        "recipe_sha256": hashlib.sha256(recipe_text.encode("utf-8")).hexdigest(),
        "transition_sha256": hashlib.sha256(
            transition_text.encode("utf-8")
        ).hexdigest(),
        "source_sha256": plan["source_sha256"],
    }
    spec["counts"] = {
        "legacy_v2": base_counts,
        "v3_extension_candidates": plan["selection"]["candidate_count"],
        "v3_extension_selected": 6,
        "recipes_total": len(recipes),
        "transitions_total": len(transitions),
        "train": sum(row["role"] == "train" for row in recipes),
        "val": sum(row["role"] == "val" for row in recipes),
        "test": sum(row["role"] == "test" for row in recipes),
        "buffer": sum(row["role"] == "buffer" for row in recipes),
    }
    return spec, recipes, transitions, plan


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
    legacy_spec, legacy_recipes, legacy_transitions = build()
    spec, recipes, transitions, _ = build_v3()
    files = {
        SPEC_JSON: json.dumps(legacy_spec, indent=1) + "\n",
        RECIPES_CSV: render(legacy_recipes, list(legacy_recipes[0])),
        TRANSITIONS_CSV: render(legacy_transitions, list(legacy_transitions[0])),
        V3_SPEC_JSON: json.dumps(spec, indent=1) + "\n",
        V3_RECIPES_CSV: render(recipes, list(recipes[0])),
        V3_TRANSITIONS_CSV: render(transitions, list(transitions[0])),
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
    legacy_counts = legacy_spec["counts"]
    print(f"{LEGACY_SPEC_VERSION}: {legacy_counts['grid_candidates']} grid recipes, "
          f"{legacy_counts['excluded_near_real']} near a real recipe, "
          f"{legacy_counts['kept']} kept in {legacy_counts['cells']} cells; "
          f"train {legacy_counts['train']}, val {legacy_counts['val']}, "
          f"test {legacy_counts['test']}, buffer {legacy_counts['buffer']}; "
          f"{len(legacy_transitions)} transitions")
    c = spec["counts"]
    print(f"{SPEC_VERSION}: {c['recipes_total']} recipes; train {c['train']}, "
          f"val {c['val']}, test {c['test']}, buffer {c['buffer']}; "
          f"{c['transitions_total']} transitions; "
          f"{c['v3_extension_selected']} selected from {c['v3_extension_candidates']} candidates")
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

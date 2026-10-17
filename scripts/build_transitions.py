"""Add previous-recipe context to each video-5 vial (INN-7 T3).

    python3 scripts/build_transitions.py                      # writes data/video5_transitions.csv
    python3 scripts/build_transitions.py --check              # fail if the committed CSV is stale
    python3 scripts/build_transitions.py --recipes X --out Y  # other input/output paths

Reads only the committed data/video5_vial_recipes.csv; no external data is needed.
A "transition" is a pair of consecutive vials in dispense order (vial n-1 -> vial n). It is
derived from the vial order, not observed: the dataset has no transition table. Nothing records
what ran before vial 1, so its previous recipe is "unknown", not guessed. Output is deterministic.
"""
import argparse
import csv
from io import StringIO
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "data/video5_vial_recipes.csv"
OUT = REPO / "data/video5_transitions.csv"
FIELDS = [
    "video_id", "vial", "recipe_id", "dispense_order", "total_ml", "start_s", "end_s",
    "prev_vial", "prev_recipe_id", "recipe_changed", "position_in_run", "initial_batch",
    "prev_recipe_note",
]
UNKNOWN = "unknown"


def build(vials):
    """Return one row per vial with previous-vial context, in vial order."""
    vials = sorted(vials, key=lambda v: int(v["vial"]))
    numbers = [int(v["vial"]) for v in vials]
    if numbers != list(range(1, len(vials) + 1)):
        raise ValueError(f"vial numbers must run 1..{len(vials)} without gaps; got {numbers}")
    rows, prev, run = [], None, 0
    for v in vials:
        changed = None if prev is None else v["recipe_id"] != prev["recipe_id"]
        run = 1 if changed in (None, True) else run + 1
        if prev is None:
            note = "first vial of the recording; prior recipe not recorded"
            if "swapped in" in v.get("color_purpose", ""):
                note += " (source notes recipe 'swapped in')"
        elif changed:
            note = f"recipe change {prev['recipe_id']} -> {v['recipe_id']}"
        else:
            note = f"repeat of recipe {v['recipe_id']}"
        rows.append({
            "video_id": 5, "vial": v["vial"], "recipe_id": v["recipe_id"],
            "dispense_order": v["dispense_order"], "total_ml": v["total_ml"],
            "start_s": v["start_s"], "end_s": v["end_s"],
            "prev_vial": UNKNOWN if prev is None else prev["vial"],
            "prev_recipe_id": UNKNOWN if prev is None else prev["recipe_id"],
            "recipe_changed": UNKNOWN if changed is None else int(changed),
            "position_in_run": run, "initial_batch": int(prev is None),
            "prev_recipe_note": note,
        })
        prev = v
    return rows


def render(rows):
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipes", type=Path, default=RECIPES, help="vial recipe CSV")
    ap.add_argument("--out", type=Path, default=OUT, help="transitions CSV to write")
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing")
    a = ap.parse_args()
    with a.recipes.open(newline="") as stream:
        rows = build(csv.DictReader(stream))
    text = render(rows)
    if a.check:
        if not a.out.is_file() or a.out.read_text() != text:
            raise SystemExit(f"{a.out} is missing or stale; rerun python3 scripts/build_transitions.py")
        print(f"{a.out} is up to date")
    else:
        a.out.write_text(text)
        print(f"wrote {a.out}")
    known = [r for r in rows if r["recipe_changed"] != UNKNOWN]
    changes = sum(r["recipe_changed"] == 1 for r in known)
    print(f"vials: {len(rows)}; transitions with known previous vial: {len(known)}; "
          f"recipe changes: {changes}; repeats: {len(known) - changes}; "
          f"initial_batch vials (no known prior recipe): {sum(r['initial_batch'] for r in rows)}")


if __name__ == "__main__":
    main()

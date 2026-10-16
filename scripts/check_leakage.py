"""Fail if the frozen partitions leak across roles.

    python scripts/check_leakage.py
    python scripts/check_leakage.py --manifest renders/train_manifest.csv   # also check a data manifest

Checks the committed split files from scripts/make_splits.py:
- no twin recipe or mixture-ratio cell appears in two roles;
- no twin recipe is near a reserved real recipe (rule in data/sim_split_spec.json);
- carryover transitions stay inside one role;
- render seed ranges do not overlap;
- every real check frame has role real_validation, and none is a duplicate of another.
With --manifest (CSV with a filename or path column, and optionally role, sim_recipe_id, seed):
- no real dataset file appears, by name or by SHA-256 from data/dataset_manifest.csv;
- each row's sim_recipe_id has the role the split assigns, and its seed is in that role's range.
Cross-role recipes closer than the ratio buffer fail; looser similarity is reported, not failed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_splits as ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO


def ml(row):
    return {t: int(row[f"{t}_ml"] or 0) for t in ms.TANKS}


def check(manifests):
    errors = []
    spec = json.loads(ms.SPEC_JSON.read_text())
    recipes = list(csv.DictReader(ms.RECIPES_CSV.open()))
    roles = {r["sim_recipe_id"]: r["role"] for r in recipes}
    if len(roles) != len(recipes):
        errors.append("duplicate sim_recipe_id")
    seen = {}
    for r in recipes:
        key = tuple(ml(r).values())
        if key in seen and seen[key] != r["role"]:
            errors.append(f"recipe {key} in roles {seen[key]} and {r['role']}")
        seen[key] = r["role"]
    cell_roles = {}
    for r in recipes:
        if r["role"] != "buffer":  # a partly dropped cell is fine; only used roles must not share a cell
            cell_roles.setdefault(r["cell"], set()).add(r["role"])
    errors += [f"cell {c} in roles {sorted(v)}" for c, v in cell_roles.items() if len(v) > 1]
    real = {k: {t: int(v[t]) for t in ms.TANKS} for k, v in spec["real_recipes_reserved"].items()}
    for r in recipes:
        for k, rv in real.items():
            if ms.near(ml(r), rv):
                errors.append(f"{r['sim_recipe_id']} is near reserved real recipe {k}")
    for t in csv.DictReader(ms.TRANSITIONS_CSV.open()):
        if roles.get(t["prev_sim_recipe_id"]) != t["role"] or roles.get(t["sim_recipe_id"]) != t["role"]:
            errors.append(f"transition {t['prev_sim_recipe_id']}->{t['sim_recipe_id']} crosses roles")
    spans = sorted(spec["render_seed_ranges"].values())
    errors += [f"seed ranges overlap: {a} {b}" for a, b in zip(spans, spans[1:]) if b[0] <= a[1]]
    used = [r for r in recipes if r["role"] in ("train", "val", "test")]
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            if a["role"] != b["role"] and ms.ratio_close(ms.fractions(ml(a)), ms.fractions(ml(b))):
                errors.append(f"{a['sim_recipe_id']} ({a['role']}) and {b['sim_recipe_id']} ({b['role']}) "
                              f"have ratios within {ms.BUFFER_FRAC}")
    # Similar total volume or near ratios beyond the buffer are allowed; count them for the record.
    near_pairs = sum(1 for i, a in enumerate(used) for b in used[i + 1:]
                     if a["role"] != b["role"] and ms.near(ml(a), ml(b)))
    real_frames = []
    if ms.REAL_CSV.is_file():
        real_frames = list(csv.DictReader(ms.REAL_CSV.open()))
        names = [r["filename"] for r in real_frames]
        if len(set(names)) != len(names):
            errors.append("duplicate filename in real_check_frames.csv")
        errors += [f"{r['filename']} has role {r['role']}" for r in real_frames if r["role"] != "real_validation"]
    else:
        errors.append(f"{ms.REAL_CSV.name} missing")

    if manifests:
        real_names, real_hashes = set(), set()
        dm = REPO / "data/dataset_manifest.csv"
        for r in csv.DictReader(dm.open()):
            if r["path"].startswith("csi-vial-fill-dataset/"):
                real_names.add(Path(r["path"]).name)
                real_hashes.add(r["sha256"])
        ranges = spec["render_seed_ranges"]
        for m in manifests:
            for i, r in enumerate(csv.DictReader(Path(m).open()), start=2):
                name = Path(r.get("filename") or r.get("path") or "").name
                if name in real_names or r.get("sha256") in real_hashes:
                    errors.append(f"{m}:{i}: real dataset file {name}")
                rid, role = r.get("sim_recipe_id"), r.get("role")
                if rid and role and roles.get(rid) != role:
                    errors.append(f"{m}:{i}: {rid} is {roles.get(rid)}, manifest says {role}")
                if r.get("seed") and role in ranges:
                    lo, hi = ranges[role]
                    if not lo <= int(r["seed"]) <= hi:
                        errors.append(f"{m}:{i}: seed {r['seed']} outside {role} range")
    return errors, near_pairs, len(recipes), len(real_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", default=[], help="data manifest CSV to check (repeatable)")
    a = ap.parse_args()
    errors, near_pairs, n_recipes, n_real = check(a.manifest)
    print(f"{n_recipes} twin recipes, {n_real} real check frames, {len(a.manifest)} extra manifest(s)")
    print(f"cross-role pairs within {ms.NEAR_ML} mL on every tank or {ms.NEAR_FRAC} in ratio "
          f"(allowed beyond the {ms.BUFFER_FRAC} ratio buffer, reported): {near_pairs}")
    if errors:
        for e in errors[:50]:
            print("LEAK:", e)
        raise SystemExit(f"{len(errors)} leakage error(s)")
    print("no leakage found")


if __name__ == "__main__":
    main()

"""Fail if the frozen partitions leak across roles.

    python scripts/check_leakage.py
    python scripts/check_leakage.py --manifest renders/train_manifest.csv   # also check a data manifest

Checks the committed split files from scripts/make_splits.py.  The default is the
additive WP1 v3 split; pass ``--split-version v2`` for a historical artifact:
- no twin recipe or mixture-ratio cell appears in two roles;
- no twin recipe is near a reserved real recipe (rule in data/sim_split_spec.json);
- carryover transitions stay inside one role;
- render seed ranges do not overlap;
- every real check frame has role real_validation, and none is a duplicate of another.
With --manifest (render manifest CSV with required columns path, sim_recipe_id, role, seed;
relative paths resolve from the manifest's folder):
- every listed file that exists is hashed; its bytes must not match any real dataset file in
  data/dataset_manifest.csv, and its name must not be a real file name;
- a listed file that does not exist fails, unless --allow-missing;
- the role comes from the selected version's recipe CSV: unknown and buffer recipes fail, and a
  row whose role disagrees fails; each seed must be inside its role's range and used by only one
  role;
- across all supplied manifests, one file (resolved path) or identical bytes (SHA-256) must not
  appear under two roles. Reuse inside one role is allowed.
A re-encoded or resized real image is not caught: the check compares bytes, not pixels.
Cross-role recipes closer than the ratio buffer fail; looser similarity is reported, not failed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_splits as ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO


def ml(row):
    return {t: int(row[f"{t}_ml"] or 0) for t in ms.TANKS}


REQUIRED = ("path", "sim_recipe_id", "role", "seed")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifests(manifests, roles, spec, allow_missing):
    """Check manifests against caller-supplied authoritative roles and split rules.

    Nothing in a manifest is trusted. Every listed file that exists is hashed and
    compared with the real dataset's SHA-256 values in data/dataset_manifest.csv.
    """
    errors = []
    if not manifests:
        return errors
    real_names, real_hashes = set(), {}
    for r in csv.DictReader((REPO / "data/dataset_manifest.csv").open()):
        if r["path"].startswith("csi-vial-fill-dataset/"):
            real_names.add(Path(r["path"]).name)
            real_hashes[r["sha256"]] = r["path"]
    ranges = spec["render_seed_ranges"]
    seed_roles, path_roles, hash_roles = {}, {}, {}
    for m in manifests:
        m = Path(m)
        reader = csv.DictReader(m.open())
        missing_cols = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing_cols:
            errors.append(f"{m}: missing required column(s) {', '.join(missing_cols)}")
            continue
        for i, r in enumerate(reader, start=2):
            where = f"{m}:{i}"
            rel = r["path"]
            if not rel:
                errors.append(f"{where}: empty path")
                continue
            if Path(rel).name in real_names:
                errors.append(f"{where}: real dataset file name {Path(rel).name}")
            f = Path(rel) if Path(rel).is_absolute() else m.parent / rel
            h = None
            if f.is_file():
                h = sha256(f)
                if h in real_hashes:
                    errors.append(f"{where}: {rel} has the same bytes as real file {real_hashes[h]}")
            elif not allow_missing:
                errors.append(f"{where}: {rel} not found (use --allow-missing for a manifest written before rendering)")
            rid = r["sim_recipe_id"]
            true_role = roles.get(rid)
            if true_role is None:
                errors.append(f"{where}: unknown sim_recipe_id {rid!r}")
                continue
            if true_role == "buffer":
                errors.append(f"{where}: {rid} is a buffer recipe and must never be rendered")
                continue
            path_roles.setdefault(f.resolve(), set()).add(true_role)
            if h is not None:
                hash_roles.setdefault(h, {}).setdefault(true_role, rel)
            if r["role"] != true_role:
                errors.append(f"{where}: {rid} is {true_role}, manifest says {r['role']!r}")
            try:
                seed = int(r["seed"])
            except ValueError:
                errors.append(f"{where}: seed {r['seed']!r} is not an integer")
                continue
            lo, hi = ranges[true_role]
            if not lo <= seed <= hi:
                errors.append(f"{where}: seed {seed} outside {true_role} range {lo}-{hi}")
            seed_roles.setdefault(seed, set()).add(true_role)
    errors += [f"seed {s} used in roles {sorted(v)}" for s, v in seed_roles.items() if len(v) > 1]
    errors += [f"{f} listed in roles {sorted(v)}" for f, v in path_roles.items() if len(v) > 1]
    errors += [f"same bytes in roles {sorted(v)}: {', '.join(v[k] for k in sorted(v))}"
               for v in hash_roles.values() if len(v) > 1]
    return errors


def check(manifests, allow_missing=False, split_version="v3"):
    errors = []
    if split_version == "v3":
        spec_path = ms.V3_SPEC_JSON
        recipes_path = ms.V3_RECIPES_CSV
        transitions_path = ms.V3_TRANSITIONS_CSV
        try:
            expected_spec, expected_recipes, expected_transitions, _ = ms.build_v3()
            if spec_path.read_text() != json.dumps(expected_spec, indent=1) + "\n":
                errors.append("v3 split spec differs from the seeded generator")
            if recipes_path.read_text() != ms.render(
                expected_recipes, list(expected_recipes[0])
            ):
                errors.append("v3 recipes differ from the seeded generator")
            if transitions_path.read_text() != ms.render(
                expected_transitions, list(expected_transitions[0])
            ):
                errors.append("v3 transitions differ from the seeded generator")
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"v3 split extension is unavailable or invalid: {error}")
    elif split_version == "v2":
        spec_path = ms.SPEC_JSON
        recipes_path = ms.RECIPES_CSV
        transitions_path = ms.TRANSITIONS_CSV
    else:
        raise ValueError("split_version must be v2 or v3")
    try:
        spec = json.loads(spec_path.read_text())
        recipes = list(csv.DictReader(recipes_path.open()))
    except (OSError, json.JSONDecodeError) as error:
        return errors + [f"split files are unavailable or invalid: {error}"], 0, 0, 0
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
    for t in csv.DictReader(transitions_path.open()):
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
            if a["role"] != b["role"] and ms.ml_close(ml(a), ml(b)):
                errors.append(f"{a['sim_recipe_id']} ({a['role']}) and {b['sim_recipe_id']} ({b['role']}) "
                              f"are within {ms.NEAR_ML} mL on every tank")
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

    errors += check_manifests(manifests, roles, spec, allow_missing)
    return errors, near_pairs, len(recipes), len(real_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", default=[], help="render manifest CSV to check (repeatable)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not fail on listed files that do not exist yet (they cannot be hashed)")
    ap.add_argument(
        "--split-version",
        choices=("v2", "v3"),
        default="v3",
        help="authoritative split to check (default: additive WP1 v3)",
    )
    a = ap.parse_args()
    errors, near_pairs, n_recipes, n_real = check(
        a.manifest, a.allow_missing, a.split_version
    )
    print(f"{a.split_version}: {n_recipes} twin recipes, {n_real} real check frames, "
          f"{len(a.manifest)} extra manifest(s)")
    print(f"cross-role pairs within {ms.NEAR_ML} mL on every tank or {ms.NEAR_FRAC} in ratio "
          f"(allowed beyond the {ms.BUFFER_FRAC} ratio buffer, reported): {near_pairs}")
    if errors:
        for e in errors[:50]:
            print("LEAK:", e)
        raise SystemExit(f"{len(errors)} leakage error(s)")
    print("no leakage found")


if __name__ == "__main__":
    main()

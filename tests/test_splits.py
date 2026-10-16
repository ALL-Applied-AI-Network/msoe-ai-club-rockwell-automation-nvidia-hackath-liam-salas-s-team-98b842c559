"""The frozen split files are current and the leakage checker catches real leaks."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_leakage
import make_splits


def test_committed_twin_splits_are_current():
    spec, recipes, transitions = make_splits.build()
    assert make_splits.RECIPES_CSV.read_text() == make_splits.render(recipes, list(recipes[0]))
    assert make_splits.TRANSITIONS_CSV.read_text() == make_splits.render(transitions, list(transitions[0]))


def test_no_reserved_real_recipe_in_twin_splits():
    spec, recipes, _ = make_splits.build()
    real = make_splits.real_recipes()
    for r in recipes:
        assert not any(make_splits.near({t: r[f"{t}_ml"] for t in make_splits.TANKS}, v) for v in real.values())
    assert {r["role"] for r in recipes} == {"train", "val", "test", "buffer"}


def test_committed_splits_pass():
    errors, *_ = check_leakage.check([])
    assert errors == []


def _recipe(role):
    return next(r["sim_recipe_id"] for r in csv.DictReader(make_splits.RECIPES_CSV.open()) if r["role"] == role)


def _run(manifest, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts/check_leakage.py"), "--manifest", str(manifest), *extra],
                          capture_output=True, text=True)


def _real_still():
    name = next(csv.DictReader(make_splits.REAL_CSV.open()))["filename"]
    for line in (ROOT / "data/dataset_manifest.csv").read_text().splitlines():
        path, _, sha = line.split(",")
        if path.endswith("/" + name):
            return name, sha
    raise AssertionError(name)


def test_clean_manifest_passes(tmp_path):
    (tmp_path / "r1.png").write_bytes(b"synthetic render")
    m = tmp_path / "m.csv"
    m.write_text(f"path,sim_recipe_id,role,seed\nr1.png,{_recipe('train')},train,5\n")
    result = _run(m)
    assert result.returncode == 0, result.stdout


def test_manifest_holes_from_review_fail(tmp_path, monkeypatch):
    name, sha = _real_still()
    # A renamed real still: same bytes, harmless name. Fake its content by pointing the manifest's
    # hash table at our file's hash instead of copying external data into the test.
    fake = tmp_path / "render_0001.jpg"
    fake.write_bytes(b"pretend this is a real still")
    monkeypatch.setattr(check_leakage, "sha256", lambda p: sha if p.name == "render_0001.jpg"
                        else hashlib.sha256(p.read_bytes()).hexdigest())
    (tmp_path / "ok.png").write_bytes(b"x")
    m = tmp_path / "m.csv"
    m.write_text("path,sim_recipe_id,role,seed\n"
                 f"render_0001.jpg,{_recipe('train')},train,5\n"        # bytes of a real still
                 f"{name},{_recipe('train')},train,6\n"                 # real file name
                 f"ok.png,{_recipe('train')},train,2500000\n"           # seed in test range
                 f"ok.png,{_recipe('test')},train,7\n"                  # wrong role
                 f"ok.png,{_recipe('buffer')},buffer,8\n"               # buffer rendered
                 "ok.png,S9999,train,9\n"                               # unknown recipe
                 f"missing.png,{_recipe('train')},train,10\n")          # file absent
    spec = json.loads(make_splits.SPEC_JSON.read_text())
    roles = {r["sim_recipe_id"]: r["role"] for r in csv.DictReader(make_splits.RECIPES_CSV.open())}
    errors = " | ".join(check_leakage.check_manifests([m], roles, spec, allow_missing=False))
    for needle in ("same bytes as real file", "real dataset file name", "outside train range",
                   "manifest says 'train'", "buffer recipe", "unknown sim_recipe_id", "not found"):
        assert needle in errors, needle
    assert "not found" not in " | ".join(check_leakage.check_manifests([m], roles, spec, allow_missing=True))


def test_manifest_without_role_column_fails(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text(f"path,sim_recipe_id,seed\nx.png,{_recipe('train')},2500000\n")
    result = _run(m, "--allow-missing")
    assert result.returncode != 0
    assert "missing required column(s) role" in result.stdout


def test_seed_shared_across_roles_fails(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("path,sim_recipe_id,role,seed\n"
                 f"a.png,{_recipe('train')},train,5\n"
                 f"b.png,{_recipe('val')},val,5\n")
    result = _run(m, "--allow-missing")
    assert "seed 5 used in roles" in result.stdout

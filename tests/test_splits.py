"""The frozen split files are current and the leakage checker catches real leaks."""
import csv
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


def test_manifest_with_real_frame_fails(tmp_path):
    real = next(csv.DictReader(make_splits.REAL_CSV.open()))["filename"]
    test_recipe = next(r for r in csv.DictReader(make_splits.RECIPES_CSV.open()) if r["role"] == "test")
    m = tmp_path / "train_manifest.csv"
    m.write_text("filename,role,sim_recipe_id,seed\n"
                 f"{real},train,,5\n"
                 f"render_0001.png,train,{test_recipe['sim_recipe_id']},5\n"
                 "render_0002.png,train,,2000001\n")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_leakage.py"), "--manifest", str(m)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "real dataset file" in result.stdout
    assert "manifest says train" in result.stdout
    assert "outside train range" in result.stdout

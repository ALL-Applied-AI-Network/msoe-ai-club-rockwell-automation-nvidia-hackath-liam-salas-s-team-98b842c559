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


def test_committed_v3_extension_is_current_and_preserves_frozen_v2_bytes():
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            make_splits.RECIPES_CSV,
            make_splits.TRANSITIONS_CSV,
            make_splits.SPEC_JSON,
        )
    }
    assert before == {
        "sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
        "sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
        "sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
    }

    spec, recipes, transitions, plan = make_splits.build_v3()
    assert make_splits.V3_SPEC_JSON.read_text() == json.dumps(spec, indent=1) + "\n"
    assert make_splits.V3_RECIPES_CSV.read_text() == make_splits.render(
        recipes, list(recipes[0])
    )
    assert make_splits.V3_TRANSITIONS_CSV.read_text() == make_splits.render(
        transitions, list(transitions[0])
    )
    legacy_recipe_lines = make_splits.RECIPES_CSV.read_text().splitlines()
    legacy_transition_lines = make_splits.TRANSITIONS_CSV.read_text().splitlines()
    assert make_splits.V3_RECIPES_CSV.read_text().splitlines()[
        : len(legacy_recipe_lines)
    ] == legacy_recipe_lines
    assert make_splits.V3_TRANSITIONS_CSV.read_text().splitlines()[
        : len(legacy_transition_lines)
    ] == legacy_transition_lines
    legacy_recipes = list(csv.DictReader(make_splits.RECIPES_CSV.open()))
    assert [row["sim_recipe_id"] for row in recipes[len(legacy_recipes) :]] == [
        "S0439",
        "S0440",
        "S0441",
        "S0442",
        "S0443",
        "S0444",
    ]
    assert {row["role"] for row in recipes[len(legacy_recipes) :]} == {"test"}
    assert spec["version"] == "v3"
    assert {"seed", "grid_ml", "role_shares"}.isdisjoint(spec)
    assert spec["generation"]["legacy_v2"] == {
        "seed": 20261016,
        "grid_ml": [0, *range(10, 65, 5)],
        "role_shares": {"train": 0.7, "val": 0.15, "test": 0.15},
    }
    assert spec["generation"]["v3_test_extension"] == {
        "seed": 20261028,
        "grid_step_ml": 1,
        "tank_volume_ml": {"unused": 0, "minimum_when_used": 10, "maximum": 60},
        "role_assignment": "test",
        "legacy_roles_edited": False,
    }
    assert spec["parent_split"]["version"] == "v2"
    assert spec["test_recipe_extension"]["generator_seed"] == plan["generator_seed"]
    assert spec["test_recipe_extension"]["plan_canonical_sha256"] == (
        make_splits.canonical_sha256(plan)
    )


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


def test_v3_generated_recipe_resolves_as_test_in_manifest(tmp_path):
    _, recipes, _, _ = make_splits.build_v3()
    generated = recipes[-1]
    manifest = tmp_path / "wp1.csv"
    manifest.write_text(
        "path,sim_recipe_id,role,seed\n"
        f"future.png,{generated['sim_recipe_id']},test,2000123\n"
    )
    result = _run(manifest, "--allow-missing")
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


def _write(path, rows):
    path.write_text("path,sim_recipe_id,role,seed\n" + "".join(f"{p},{_recipe(r)},{r},{s}\n" for p, r, s in rows))
    return path


def test_distinct_renders_across_roles_pass(tmp_path):
    (tmp_path / "a.png").write_bytes(b"train render")
    (tmp_path / "b.png").write_bytes(b"test render")
    m = _write(tmp_path / "m.csv", [("a.png", "train", 5), ("b.png", "test", 2000005)])
    result = _run(m)
    assert result.returncode == 0, result.stdout


def test_same_render_path_in_two_roles_fails(tmp_path):
    (tmp_path / "a.png").write_bytes(b"one render")
    m = _write(tmp_path / "m.csv", [("a.png", "train", 5), ("a.png", "test", 2000005)])
    result = _run(m)
    assert result.returncode != 0
    assert "listed in roles ['test', 'train']" in result.stdout


def test_renamed_copy_in_two_roles_fails(tmp_path):
    (tmp_path / "a.png").write_bytes(b"one render")
    (tmp_path / "copy.png").write_bytes(b"one render")
    m = _write(tmp_path / "m.csv", [("a.png", "train", 5), ("copy.png", "test", 2000005)])
    result = _run(m)
    assert result.returncode != 0
    assert "same bytes in roles ['test', 'train']" in result.stdout


def test_shared_render_across_separate_manifests_fails(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "a.png").write_bytes(b"one render")
    # Relative paths resolve from each manifest's folder, so both rows name the same file.
    train = _write(tmp_path / "train" / "train.csv", [("../a.png", "train", 5)])
    test = _write(tmp_path / "test.csv", [("a.png", "test", 2000005)])
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_leakage.py"),
                             "--manifest", str(train), "--manifest", str(test)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "listed in roles ['test', 'train']" in result.stdout

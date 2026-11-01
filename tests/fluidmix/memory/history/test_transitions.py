"""Previous-recipe context built from the committed video-5 recipe table."""
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "cli/fluidmix/memory/history"))
from cli.fluidmix.memory.history.build_transitions import build, render


def rows():
    with (ROOT / "data/video5_vial_recipes.csv").open(newline="") as stream:
        return build(csv.DictReader(stream))


def test_counts_and_unknown_first_vial():
    out = rows()
    assert len(out) == 30
    first = out[0]
    assert (first["prev_recipe_id"], first["recipe_changed"], first["initial_batch"]) == ("unknown", "unknown", 1)
    known = out[1:]
    assert sum(r["recipe_changed"] == 1 for r in known) == 9
    assert sum(r["recipe_changed"] == 0 for r in known) == 20
    assert all(r["position_in_run"] == 1 for r in known if r["recipe_changed"] == 1)


def test_committed_csv_is_current():
    assert (ROOT / "data/video5_transitions.csv").read_text() == render(rows())
    done = subprocess.run([sys.executable, str(ROOT / "cli/fluidmix/memory/history/build_transitions.py"), "--check"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

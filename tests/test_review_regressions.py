"""Regressions for review findings, exercised through real scripts and exported APIs."""
from pathlib import Path
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.absorbance import expected_color, fit_tank_absorbance


def run_script(name, *args, cwd=ROOT, env=None):
    if name in ("color.py", "fixlabels.py") and args:
        env = {**(env or os.environ), "FLUIDMIX_OUTPUTS": str(args[0])}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
        cwd=cwd, env=env, text=True, capture_output=True,
    )


@pytest.fixture
def docs_copy(tmp_path):
    for name in ("AGENTS.md", "CLAUDE.md"):
        shutil.copy2(ROOT / name, tmp_path / name)
    for name in ("docs", ".agents"):
        shutil.copytree(ROOT / name, tmp_path / name, symlinks=True)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/skills").symlink_to("../.agents/skills", target_is_directory=True)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/check_agent_docs.py", tmp_path / "scripts/check_agent_docs.py")
    return tmp_path


def check_docs(root):
    return subprocess.run(
        [sys.executable, str(root / "scripts/check_agent_docs.py")],
        text=True, capture_output=True,
    )


def test_new_skill_needs_no_checker_inventory_edit(docs_copy):
    skill = docs_copy / ".agents/skills/additional-workflow"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: additional-workflow\ndescription: >-\n"
        "  A valid multiline description for an additional workflow.\n---\n# Workflow\n"
    )
    result = check_docs(docs_copy)
    assert result.returncode == 0, result.stderr
    assert "10 skills" in result.stdout


@pytest.mark.parametrize("metadata", [
    "name: additional-workflow\ndescription: valid\nbad: [unclosed",
    "name: additional-workflow\nname: additional-workflow\ndescription: valid",
    "name: additional-workflow\ndescription: []",
    "name: additional-workflow\ndescription: ''",
])
def test_invalid_skill_metadata_is_rejected(docs_copy, metadata):
    skill = docs_copy / ".agents/skills/additional-workflow"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\n" + metadata + "\n---\n# Workflow\n")
    assert check_docs(docs_copy).returncode != 0


def test_missing_link_target_is_rejected(docs_copy):
    with (docs_copy / "docs/skills.md").open("a") as stream:
        stream.write("\n[missing](does-not-exist.md)\n")
    assert check_docs(docs_copy).returncode != 0


@pytest.mark.parametrize("recipe,tanks", [
    ({"t100": 20, "t200": -10}, {"t200": [1, 1, 1]}),
    ({"t100": 20, "t300": 10}, {}),
    ({"t200": 10}, {}),
    ({"t100": float("nan")}, {}),
    ({"t100": float("inf")}, {}),
    ({"t200": 10}, {"t200": [float("nan"), 1, 1]}),
])
def test_invalid_prediction_inputs_are_rejected(recipe, tanks):
    with pytest.raises(ValueError):
        expected_color(recipe, tanks, [0.8, 0.8, 0.8])


def test_missing_unused_tank_is_allowed():
    assert np.allclose(expected_color({"t100": 20}, {}, [0.8]*3), [0.8]*3)


def test_rank_deficient_calibration_is_rejected():
    rows = [({"t200": 10, "t400": 10}, [0.2, 0.3, 0.4])] * 3
    with pytest.raises(ValueError, match="rank"):
        fit_tank_absorbance(rows)


def test_smoke_rejects_nonfinite_result(monkeypatch):
    spec = importlib.util.spec_from_file_location("review_smoke", ROOT / "scripts/smoke_absorbance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Exercise the real entry point with a bad dependency result, rather than its guard alone.
    monkeypatch.setattr(module, "fit_carryover_fraction", lambda *args: (float("nan"), float("nan")))
    with pytest.raises(SystemExit, match="failed"):
        module.main()


def vial_inputs(root):
    vial = root / "csi-vial-fill-dataset/vial_fill"
    (vial / "labels").mkdir(parents=True)
    with (vial / "labels/video5_vial_recipes.csv").open("w", newline="") as stream:
        fields = ["vial", "recipe_id", "start_s", "end_s", "end_frame",
                  "type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(vial=1, recipe_id=5, start_s=0, end_s=10, end_frame=300,
                             type="clean", t100_ml=20, t200_ml=10, t400_ml=0,
                             total_ml=30, dispense_order="t200,t100"))
    return vial


def test_failed_extraction_preserves_colors_and_ignores_stale_frame(tmp_path):
    vial_inputs(tmp_path)
    stale = Image.new("RGB", (640, 480), (20, 20, 20))
    for x in range(200, 210):
        for y in range(200, 210):
            stale.putpixel((x, y), (255, 0, 99))
    stale.save(tmp_path / "c.png")
    output = tmp_path / "colors.json"
    output.write_text("KEEP PREVIOUS COLORS")
    bins = tmp_path / "bin"
    bins.mkdir()
    ffmpeg = bins / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nexit 7\n")
    ffmpeg.chmod(0o755)
    env = {**os.environ, "PATH": str(bins) + os.pathsep + os.environ.get("PATH", "")}
    result = run_script("color.py", tmp_path, env=env)
    assert result.returncode != 0
    assert output.read_text() == "KEEP PREVIOUS COLORS"


def test_empty_video5_index_preserves_previous_labels(tmp_path):
    vial = vial_inputs(tmp_path)
    (vial / "image_index.csv").write_text(
        "video_id,filename,timestamp\n4,example.jpg,2026-09-23T19:47:12\n"
    )
    labels = tmp_path / "video5_frame_labels_corrected.csv"
    labels.write_text("KEEP PREVIOUS GOOD LABELS")
    result = run_script("fixlabels.py", tmp_path)
    assert result.returncode != 0
    assert labels.read_text() == "KEEP PREVIOUS GOOD LABELS"


def test_successful_extraction_uses_fresh_frames(tmp_path):
    vial_inputs(tmp_path)
    bins = tmp_path / "bin"
    bins.mkdir()
    ffmpeg = bins / "ffmpeg"
    ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom PIL import Image\n"
        "image=Image.new('RGB',(640,480),(20,20,20))\n"
        "for x in range(200,210):\n"
        " for y in range(200,210):image.putpixel((x,y),(255,0,99))\n"
        "image.save(sys.argv[-1])\n"
    )
    ffmpeg.chmod(0o755)
    env = {**os.environ, "PATH": str(bins) + os.pathsep + os.environ.get("PATH", "")}
    result = run_script("color.py", tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    colors = json.loads((tmp_path / "colors.json").read_text())
    assert colors[0]["rgb"] == [255, 0, 99]
    assert not list(tmp_path.glob(".color-*"))


def test_successful_label_correction_preserves_mapping(tmp_path):
    vial = vial_inputs(tmp_path)
    (vial / "image_index.csv").write_text(
        "video_id,filename,timestamp\n5,example.jpg,2026-09-23T19:47:12\n"
    )
    result = run_script("fixlabels.py", tmp_path)
    assert result.returncode == 0, result.stderr
    with (tmp_path / "video5_frame_labels_corrected.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["in_fill"] == "1" and row["recipe_id"] == "5"
    assert (tmp_path / "last_corrected.tsv").read_text() == "1\texample.jpg"

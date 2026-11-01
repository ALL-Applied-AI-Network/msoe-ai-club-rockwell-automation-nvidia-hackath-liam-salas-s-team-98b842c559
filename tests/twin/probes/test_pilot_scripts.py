"""Import/help and CLI contracts for the pilot artifact tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = (
    ROOT / "cli/twin/probes/build_twin_pilot_dataset.py",
    ROOT / "cli/twin/probes/check_render_dataset.py",
    ROOT / "cli/twin/probes/evaluate_appearance_pilot.py",
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tests.twin.probes.pilot_fixture import build_valid_pilot  # noqa: E402


@pytest.mark.parametrize("script", SCRIPTS)
def test_pilot_scripts_are_help_safe(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_pilot_scripts_are_import_safe(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(f"test_{script.stem}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert not list(tmp_path.iterdir())


def test_builder_exposes_the_graph_bound_launch_path():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/twin/probes/build_twin_pilot_dataset.py"),
            "launch",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "sample_id" in result.stdout


def test_builder_manifest_cli_exposes_frozen_environment_profile_choice():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/twin/probes/build_twin_pilot_dataset.py"),
            "manifests",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--environment-profile" in result.stdout
    assert "generic" in result.stdout
    assert "gb10-aarch64" in result.stdout


def test_checker_and_diagnostic_cli_report_structured_outcomes(tmp_path):
    build_valid_pilot(tmp_path, ROOT)
    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/twin/probes/check_render_dataset.py"),
            str(tmp_path),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    evaluated = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/twin/probes/evaluate_appearance_pilot.py"),
            str(tmp_path),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["rows"] == 8
    assert evaluated.returncode == 0, evaluated.stderr
    assert json.loads(evaluated.stdout)["overall_pass"] is True

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_batch_notebook_executes_in_report_only_mode(tmp_path):
    root = Path(__file__).parent
    notebook = root / "vials_SAM_batch.ipynb"
    executed = tmp_path / "executed.ipynb"
    env = os.environ.copy()
    env["VIAL_RUN_BATCH"] = "0"
    env["VIAL_RESULTS_DIR"] = str(tmp_path / "empty-results")
    env["MPLBACKEND"] = "Agg"
    command = [
        str(Path(sys.executable).with_name("jupyter")),
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        str(executed),
        "--ExecutePreprocessor.kernel_name=sam_env",
        "--ExecutePreprocessor.timeout=120",
    ]

    completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(executed.read_text())
    errors = [
        output
        for cell in payload["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []


def test_batch_notebook_rejects_stale_report_results(tmp_path):
    root = Path(__file__).parent
    notebook = root / "vials_SAM_batch.ipynb"
    results = tmp_path / "stale-results"
    results.mkdir()
    (results / "combined_measurements.csv").write_text("status\nold\n")
    (results / "processing_manifest.csv").write_text("status\nold\n")
    (results / "summary.json").write_text('{"config_fingerprint": "stale"}')
    env = os.environ.copy()
    env["VIAL_RUN_BATCH"] = "0"
    env["VIAL_RESULTS_DIR"] = str(results)
    env["MPLBACKEND"] = "Agg"
    command = [
        str(Path(sys.executable).with_name("jupyter")), "nbconvert", "--to", "notebook",
        "--execute", str(notebook), "--output", str(tmp_path / "stale.ipynb"),
        "--ExecutePreprocessor.kernel_name=sam_env", "--ExecutePreprocessor.timeout=120",
    ]

    completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "Results are stale for this configuration" in completed.stderr


def test_review_calibration_notebook_executes_and_generates_templates(tmp_path):
    root = Path(__file__).parent
    notebook = root / "vial_review_calibration.ipynb"
    review_dir = tmp_path / "review"
    env = os.environ.copy()
    env["VIAL_REVIEW_DIR"] = str(review_dir)
    env["MPLBACKEND"] = "Agg"
    command = [
        str(Path(sys.executable).with_name("jupyter")), "nbconvert", "--to", "notebook",
        "--execute", str(notebook), "--output", str(tmp_path / "review-executed.ipynb"),
        "--ExecutePreprocessor.kernel_name=sam_env", "--ExecutePreprocessor.timeout=120",
    ]

    completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    candidates = pd.read_csv(review_dir / "empty_reference_candidates.csv")
    validation = pd.read_csv(review_dir / "manual_validation_sample.csv")
    calibration = pd.read_csv(review_dir / "calibration_points.csv")
    assert candidates.video_name.nunique() == 26
    assert validation.video_name.nunique() == 26
    assert list(calibration.columns) == [
        "known_height_mm", "measured_height_px", "known_volume_ml", "source", "reviewer_notes"
    ]
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in json.loads(notebook.read_text())["cells"]
    )
    assert "parse_verified_empty_flags(candidates)" in source
    assert "validation_metrics(scored)" in source
    assert "reference_basis.ne('verified_empty')" in source


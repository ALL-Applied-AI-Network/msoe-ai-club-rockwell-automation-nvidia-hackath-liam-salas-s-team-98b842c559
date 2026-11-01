"""Frozen oracle-mask Lab visibility diagnostic contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from twin.probes.pilot_diagnostic import evaluate_pilot_visibility  # noqa: E402
from tests.twin.probes.pilot_fixture import DEFAULT_COLORS, build_valid_pilot  # noqa: E402


def test_exact_lab_diagnostic_passes_ordered_fixture_and_writes_no_clobber_result(tmp_path):
    build_valid_pilot(tmp_path, ROOT)

    output = evaluate_pilot_visibility(tmp_path, repo_root=ROOT)
    result = json.loads(output.read_text())

    assert result["schema_version"] == 1
    assert result["diagnostic"] == "oracle_mask_lab_visibility_v1"
    assert result["purpose"] == "acceptance_only_visibility_diagnostic"
    assert result["detector_metric"] is False
    assert result["overall_pass"] is True
    assert all(result["gates"].values())
    assert len(result["a_repeatability"]["pair_distances"]) == 6
    assert [row["sample_id"] for row in result["rows"]] == [f"R{i:03d}" for i in range(8)]
    assert set(result["versions"]) == {"numpy", "Pillow", "scikit_image", "python", "platform"}

    rgb = np.array([DEFAULT_COLORS["R003"]], dtype=np.uint8).reshape(1, 1, 3)
    expected = rgb2lab(
        rgb.astype(np.float64) / 255.0,
        illuminant="D65",
        observer="2",
        channel_axis=-1,
    )[0, 0]
    actual = np.array(result["rows"][3]["median_lab"], dtype=np.float64)
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)
    assert all(len(piece.partition(".")[2]) == 6 for piece in result["rows"][3]["median_lab_6dp"])
    with pytest.raises(FileExistsError):
        evaluate_pilot_visibility(tmp_path, repo_root=ROOT)


def test_visibility_gate_failure_is_published_as_failure_not_patched(tmp_path):
    colors = dict(DEFAULT_COLORS)
    colors["R005"] = colors["R003"]
    build_valid_pilot(tmp_path, ROOT, colors=colors)

    output = evaluate_pilot_visibility(tmp_path, repo_root=ROOT)
    result = json.loads(output.read_text())

    assert result["overall_pass"] is False
    assert result["gates"]["r005_beyond_every_a_repeat"] is False


def test_zero_a_to_b_direction_records_failed_gate_without_nonfinite_json(tmp_path):
    colors = {sample_id: (20, 100, 150) for sample_id in DEFAULT_COLORS}
    build_valid_pilot(tmp_path, ROOT, colors=colors)

    output = evaluate_pilot_visibility(tmp_path, repo_root=ROOT)
    result = json.loads(output.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    assert result["overall_pass"] is False
    assert result["gates"]["pooled_direction_nonzero_finite"] is False
    assert result["projections"] == {"R002": None, "R005": None}


def test_diagnostic_runs_before_review_jsonls_and_receipt_publication(tmp_path):
    build_valid_pilot(tmp_path, ROOT, publish=False)
    (tmp_path / "reviews/visual-review.json").unlink()

    output = evaluate_pilot_visibility(tmp_path, repo_root=ROOT)

    assert json.loads(output.read_text())["overall_pass"] is True

"""Synthetic-only tests for explicit cap-delay receipt mechanics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.cap_delay import build_cap_delay_receipt


def _table(path: Path, pairs: list[dict]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "pairs": pairs}), encoding="utf-8")
    return path


def _pair(number: int, video_id: int, delay: float = 4.0) -> dict:
    return {
        "pair_id": f"pair-{number}",
        "video_id": video_id,
        "fill_vial_id": f"vial-{number}",
        "cap_vial_id": f"vial-{number}",
        "fill_crop_id": f"fill-{number}",
        "cap_crop_id": f"cap-{number}",
        "clock_correction_id": "vial_fill_to_cap_seal_clock_fit_v1",
        "fill_end_clock_s": 100.0 + number,
        "cap_event_clock_s": 100.0 + number + delay,
        "fill_clock_correction_s": 0.25,
        "cap_clock_correction_s": -0.25,
        "unambiguous": True,
    }


def test_cap_delay_reports_insufficient_evidence_but_keeps_corrected_pairs(tmp_path):
    table = _table(tmp_path / "pairs.json", [_pair(1, 2), _pair(2, 3), _pair(3, 4)])

    receipt = build_cap_delay_receipt(table)

    assert receipt["status"] == "insufficient_evidence"
    assert receipt["success_requirements"] == {"min_unambiguous_pairs": 8, "min_videos": 2}
    assert receipt["usable_pair_count"] == 3
    assert receipt["candidate_pair_count"] == 3
    assert receipt["usable_video_ids"] == [2, 3, 4]
    assert receipt["pairs"][0]["corrected_delay_s"] == pytest.approx(3.5)
    assert receipt["percentiles_s"] is None
    assert receipt["insufficiency_reasons"] == ["fewer_than_8_unambiguous_pairs"]


def test_cap_delay_reports_percentiles_only_after_eight_pairs_across_two_videos(tmp_path):
    pairs = [_pair(number, 2 if number < 5 else 3, float(number)) for number in range(1, 9)]
    table = _table(tmp_path / "pairs.json", pairs)

    receipt = build_cap_delay_receipt(table)

    assert receipt["status"] == "measured"
    assert receipt["usable_pair_count"] == 8
    assert receipt["percentiles_s"] == {"p10": 1.2, "median": 4.0, "p90": 6.8}


def test_cap_delay_rejects_video_one_and_video_five_and_excludes_ambiguous_pairs(tmp_path):
    table = _table(tmp_path / "pairs.json", [_pair(1, 1)])
    with pytest.raises(ValueError, match="video 1"):
        build_cap_delay_receipt(table)

    table = _table(tmp_path / "pairs.json", [_pair(1, 5)])
    with pytest.raises(ValueError, match="video 5"):
        build_cap_delay_receipt(table)

    ambiguous = _pair(1, 2)
    ambiguous["unambiguous"] = False
    ambiguous["exclusion_reason"] = "cap obscured"
    table = _table(tmp_path / "pairs.json", [ambiguous])
    receipt = build_cap_delay_receipt(table)
    assert receipt["usable_pair_count"] == 0
    assert receipt["exclusions"] == [{"pair_id": "pair-1", "reason": "cap obscured"}]


def test_cap_delay_rejects_mismatched_candidate_vial_ids(tmp_path):
    pair = _pair(1, 2)
    pair["cap_vial_id"] = "another-vial"
    table = _table(tmp_path / "pairs.json", [pair])

    with pytest.raises(ValueError, match="same vial"):
        build_cap_delay_receipt(table)


def test_cap_delay_cli_writes_insufficient_receipt_below_outputs(tmp_path):
    table = _table(tmp_path / "pairs.json", [_pair(1, 2)])
    output_root = tmp_path / "outputs"
    script = Path(__file__).resolve().parents[1] / "scripts/measure_fill_to_cap_delay.py"
    environment = {**os.environ, "FLUIDMIX_OUTPUTS": str(output_root)}

    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(table), "--output", "receipts/cap.json"],
        capture_output=True, text=True, env=environment,
    )

    output = output_root / "receipts/cap.json"
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["status"] == "insufficient_evidence"

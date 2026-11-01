"""Synthetic checks for the INN-52 stream helpers. They test the logic, not the real video."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
from cli.fluidmix.core.stream_sequence import (LUMA_MIN, ROW0, ROW1, X0, agreement, bursts, colour_name, fwhm,
                             hypotheses, stream_rows)


def test_fwhm_of_a_box_is_its_width():
    profile = np.zeros(40)
    profile[10:16] = 20
    assert fwhm(profile) == (6, 10)


def test_stream_rows_finds_a_thin_stripe_and_rejects_a_wide_blob():
    ref = np.full((200, 80), 200, np.uint8)
    frame = ref.copy()
    x = 305 - X0
    frame[ROW0:ROW1, x:x + 5] = 200 - 3 * int(LUMA_MIN)   # 5 px stream under the nozzle
    frame[150:160, 5:75] = 100                          # wide liquid-like blob
    rows = stream_rows(frame, ref)
    assert len(rows) == (ROW1 - ROW0) - 10
    assert all(width == 5 for _, _, width in rows)


def test_colour_names():
    assert colour_name(0.5, 1.0) == "clear"
    assert colour_name(2.0, 8.0) == "orange"
    assert colour_name(-6.0, -0.5) == "blue"      # the blue tank reads as cyan (-a*)


def test_bursts_join_short_gaps_and_drop_short_runs():
    flowing = np.zeros(200, bool)
    flowing[10:60] = True
    flowing[65:90] = True       # 4-frame gap: same burst
    flowing[120:125] = True     # too short
    flowing[150:180] = True
    assert bursts(flowing, max_gap=10, min_len=15) == [(10, 89), (150, 179)]


def test_hypotheses_shift_by_one_dose():
    current, previous = hypotheses(["T200", "T400", "T100"], "T100")
    assert current == ["orange", "blue", "clear"]
    assert previous == ["clear", "orange", "blue"]
    assert hypotheses(["T400", "T100"], None)[1] == [None, "blue"]


def test_agreement_needs_equal_counts_and_skips_unknowns():
    assert agreement(["orange", "blue"], ["orange", "clear", "blue"]) is None
    assert agreement(["blue", "blue"], [None, "blue"]) == 1

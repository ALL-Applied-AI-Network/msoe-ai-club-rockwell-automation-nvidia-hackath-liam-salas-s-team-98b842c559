"""Synthetic checks for the INN-52 stream helpers. They test the logic, not the real video."""
import numpy as np

from cli.fluidmix.core.stream_sequence import (
    LUMA_MIN,
    ROW0,
    ROW1,
    X0,
    agreement,
    bursts,
    fwhm,
    hue_chroma,
    hypotheses,
    infer_order,
    nearest_tank,
    stream_rows,
    tank_refs,
)


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


def test_hue_and_chroma_of_an_ab_shift():
    assert hue_chroma(3.0, 4.0)[1] == 5.0
    assert round(hue_chroma(0.0, 5.0)[0]) == 90      # +b* is yellow
    assert round(hue_chroma(-5.0, 0.0)[0]) == 180    # -a* is green/cyan
    assert round(hue_chroma(0.0, -5.0)[0]) == 270    # -b* is blue


def test_tank_references_come_from_the_data_not_preset_names():
    # Any dyes and any number of tanks: references are medians of the attributed shifts.
    refs = tank_refs([("A", 10, 0), ("A", 12, 2), ("A", 11, 1), ("B", -3, 9), ("C", 0, 0), (None, 99, 99)])
    assert set(refs) == {"A", "B", "C"}
    assert list(refs["A"]) == [11, 1]
    assert nearest_tank(9, 2, refs) == "A"
    assert nearest_tank(-2, 7, refs) == "B"
    assert nearest_tank(0.5, -0.5, refs) == "C"


def test_bursts_join_short_gaps_and_drop_short_runs():
    flowing = np.zeros(200, bool)
    flowing[10:60] = True
    flowing[65:90] = True       # 4-frame gap: same burst
    flowing[120:125] = True     # too short
    flowing[150:180] = True
    assert bursts(flowing, max_gap=10, min_len=15) == [(10, 89), (150, 179)]


def test_hypotheses_shift_by_one_dose():
    current, previous = hypotheses(["T200", "T400", "T100"], "T100")
    assert current == ["T200", "T400", "T100"]
    assert previous == ["T100", "T200", "T400"]
    assert hypotheses(["T400", "T100"], None)[1] == [None, "T400"]


def test_agreement_needs_equal_counts_and_skips_unknowns():
    assert agreement(["T200", "T400"], ["T200", "T100", "T400"]) is None
    assert agreement(["T400", "T400"], [None, "T400"]) == 1


def test_infer_order_picks_the_order_that_fits_the_rates():
    rate = {"T100": 4.0, "T200": 11.0, "T400": 7.5}
    ml = {"T200": 10, "T400": 50, "T100": 10}
    assert infer_order([110, 375, 40], ml, rate) == ["T200", "T400", "T100"]
    assert infer_order([40, 375, 110], ml, rate) == ["T100", "T400", "T200"]

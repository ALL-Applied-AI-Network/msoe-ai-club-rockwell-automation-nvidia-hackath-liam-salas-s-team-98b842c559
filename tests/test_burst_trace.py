"""Synthetic checks for the within-burst trace helpers. They test the logic, not the real video."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from burst_trace import SEGMENTS, SUBMERGED_ROW, ab_distance, build, level_trend, split_parts


def frame(vial, n, burst, da, level):
    return {"vial": vial, "frame": str(n), "burst": str(burst), "flowing": "1" if burst else "0",
            "dL": "0", "dA": str(da), "dB": "0", "vial_dL": "0", "vial_dA": "0", "vial_dB": "0",
            "level_row": str(level), "level_zone": "below_holder"}


def test_parts_cover_every_frame_once_in_order():
    parts = split_parts(23)
    assert len(parts) == SEGMENTS
    assert np.concatenate(parts).tolist() == list(range(23))


def test_ab_distance_ignores_lightness():
    assert ab_distance((0, 0), (3, 4)) == 5.0


def test_level_trend_is_negative_when_the_surface_rises():
    assert level_trend([350, 345, 340, 335, 330, 325]) < 0
    assert np.isnan(level_trend([300, np.nan, np.nan, np.nan, np.nan, 299]))


def test_build_follows_colour_through_the_burst_and_chains_the_previous_tank():
    recipes = [{"vial": "1", "recipe_id": "5", "dispense_order": "T200>T400"},
               {"vial": "2", "recipe_id": "6", "dispense_order": "T400>T100"}]
    bursts = [{"vial": "1", "burst": "1", "tank": "T200"}, {"vial": "1", "burst": "2", "tank": "T400"},
              {"vial": "2", "burst": "1", "tank": "T400"}, {"vial": "2", "burst": "2", "tank": "T100"}]
    frames = []
    for vial in ("1", "2"):
        for burst in (1, 2):
            # colour ramps from 0 to 9 a* over the burst; level well below the stream rows
            frames += [frame(vial, 100 * burst + i, burst, i * 9 / 49, 350) for i in range(50)]
    traces = build(frames, bursts, recipes)
    assert [t["prev"] for t in traces] == [None, "T200", "T400", "T400"]
    first, last = traces[0]["parts"][0], traces[0]["parts"][-1]
    assert first["dA"] < 1 and last["dA"] > 8
    assert first["first"] == 100 and last["last"] == 149


def test_build_drops_frames_where_liquid_is_behind_the_stream():
    recipes = [{"vial": "1", "recipe_id": "5", "dispense_order": "T200"}]
    bursts = [{"vial": "1", "burst": "1", "tank": "T200"}]
    frames = [frame("1", i, 1, 1.0, 350) for i in range(40)]
    frames += [frame("1", 40 + i, 1, 50.0, SUBMERGED_ROW - 5) for i in range(10)]   # submerged: tinted
    frames += [frame("1", 60, 0, 99.0, 350)]                                         # not flowing
    t = build(frames, bursts, recipes)[0]
    assert t["dropped"] == 10 and t["frames"] == 50
    assert all(p["dA"] == 1.0 for p in t["parts"])


def test_untagged_bursts_are_skipped_but_still_set_the_previous_tank_to_unknown():
    recipes = [{"vial": "1", "recipe_id": "5", "dispense_order": "T200>T400"}]
    bursts = [{"vial": "1", "burst": "1", "tank": ""}, {"vial": "1", "burst": "2", "tank": "T400"}]
    frames = [frame("1", i, 1, 0, 350) for i in range(20)] + [frame("1", 50 + i, 2, 0, 350) for i in range(20)]
    traces = build(frames, bursts, recipes)
    assert len(traces) == 1 and traces[0]["prev"] is None

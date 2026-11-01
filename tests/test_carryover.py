"""Synthetic checks of the carryover package (no video). They test the math, not physical accuracy."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fluidmix.carryover.measure import lab, cie76, summarise_window, liquid_rgb
from fluidmix.carryover.decide import reference_for, frozen_tau, decide, flagged_count
from fluidmix.carryover.explain import explain


def test_lab_white_black_and_red():
    assert np.allclose(lab([255, 255, 255]), [100, 0, 0], atol=0.2)
    assert np.allclose(lab([0, 0, 0]), [0, 0, 0], atol=1e-6)
    L, a, b = lab([255, 0, 0])
    assert 52 < L < 55 and a > 70 and b > 60
    assert cie76([0, 0, 0], [3, 4, 0]) == 5.0


def test_liquid_rgb_picks_saturated_region():
    frame = np.full((480, 640, 3), 120, np.uint8)
    frame[200:210, 250:260] = (200, 40, 40)      # small saturated patch (<0.7 % of the crop)
    assert np.allclose(liquid_rgb(frame), [200, 40, 40])


def test_summarise_window_median_and_noise():
    labs = [[50, 0, 0], [50, 0, 1], [50, 0, 3], [90, 0, 0]]
    measured, noise = summarise_window(labs)
    assert np.allclose(measured, [50, 0, 0.5])
    assert noise == np.median([1, 2, np.sqrt(1600 + 9)])
    assert summarise_window([[1, 2, 3]])[1] == 0.0


def synthetic():
    rows = [(1, 6, 5, 1, 1, [40, 0, 0]), (2, 6, 6, 0, 2, [60, 0, 0]), (3, 6, 6, 0, 3, [60, 1, 0]),
            (4, 7, 6, 1, 1, [30, 10, 0]), (5, 7, 7, 0, 2, [20, 10, 0.5]), (6, 7, 7, 0, 3, [20, 10, 0])]
    return {v: {"vial": v, "recipe_id": r, "prev_recipe_id": p, "recipe_changed": c, "position_in_run": pos,
                "measured_lab": lab_, "measurement_noise_dE": 0.4, "exclusion_reason": None}
            for v, r, p, c, pos, lab_ in rows}


def test_reference_is_leave_one_out():
    rec = synthetic()
    ref, source = reference_for(2, rec)
    assert ref == [60, 1, 0] and "excluding this vial" in source and "vials 3" in source
    ref, _ = reference_for(1, rec)
    assert ref == [60, 0.5, 0]


def test_frozen_tau_floor_and_noise():
    assert frozen_tau([0.1, 0.2, 0.3]) == 2.3
    assert frozen_tau([1.0, 2.0, 3.0]) == 6.0
    assert frozen_tau([]) == 2.3


def test_decision_discards_first_after_change_only():
    rec = synthetic()
    tau = decide(rec)
    assert tau == 2.3
    assert rec[1]["decision"] == "discard" and abs(rec[1]["dE_vs_reference"] - 20.0) < 0.05
    assert rec[4]["decision"] == "discard"
    assert all(rec[v]["decision"] == "keep" for v in (2, 3, 5, 6))
    assert rec[1]["known_colour_check"] == "fail"
    assert flagged_count(rec, 0.5) == 2 and flagged_count(rec, 100) == 0
    assert "Likely carryover from recipe 5" in explain(rec[1]) and "Flush" in explain(rec[1])
    assert "Within tolerance" in explain(rec[2])


def test_known_colour_check_passes_a_wrong_but_known_colour():
    rec = synthetic()
    rec[1]["measured_lab"] = [20, 10, 0]   # vial 1 looks exactly like recipe 7
    decide(rec)
    assert rec[1]["decision"] == "discard" and rec[1]["known_colour_check"] == "pass"


def test_no_steady_peers_is_insufficient_evidence():
    rec = synthetic()
    for v in (2, 3):
        rec[v]["measured_lab"] = None
    decide(rec)
    assert rec[1]["decision"] == "insufficient_evidence"
    assert "no other recipe 6 vial" in rec[1]["reason"]
    assert rec[2]["decision"] == "insufficient_evidence"
    assert "no decision" in explain(rec[1])


# --- script-level regressions (review of 2026-11-01); the scripts are loaded as modules, nothing is run ---

def _load_script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_reads_tracker_exits_under_either_key():
    demo = _load_script("carryover_demo.py")
    new = {"tracker_exit_events": [{"track_id": "CV001", "frame": 10}], "line_crossings": []}
    old = {"exits": [{"track_id": "CV002", "frame": 20}]}
    assert demo.tracker_exits(new) == [{"track_id": "CV001", "frame": 10}]
    assert demo.tracker_exits(old) == [{"track_id": "CV002", "frame": 20}]
    assert demo.tracker_exits({}) == []


def test_dataset_default_follows_fluidmix_data(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUIDMIX_DATA", str(tmp_path))
    for name in ("carryover_video5.py", "carryover_demo.py", "carryover_lane_clip.py"):
        module = _load_script(name)
        assert module.default_dataset() == tmp_path.resolve() / "csi-vial-fill-dataset"
        assert module.default_dataset(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()


def test_attribution_rank_correlation_handles_ties():
    attribution = _load_script("carryover_attribution.py")
    assert attribution.rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert attribution.rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert list(attribution._ranks([5, 5, 1])) == [2.5, 2.5, 1.0]

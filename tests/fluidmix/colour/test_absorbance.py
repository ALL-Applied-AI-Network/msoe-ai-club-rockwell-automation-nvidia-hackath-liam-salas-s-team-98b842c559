"""Run: python -m pytest tests  (or: python tests/fluidmix/colour/test_absorbance.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fluidmix.colour.absorbance import (expected_color, fit_tank_absorbance, fit_carryover_fraction,
                                 carryover_absorbance, transmit, absorbance, mix_absorbance)

TANK_A = {"t200": np.array([0.1, 0.6, 1.5]), "t400": np.array([1.4, 0.4, 0.05])}
I0 = np.array([0.8, 0.8, 0.75])


def test_clear_water_is_i0():
    assert np.allclose(expected_color({"t100": 40}, TANK_A, I0), I0)


def test_only_fraction_matters():
    a = expected_color({"t200": 10, "t100": 30}, TANK_A, I0)
    b = expected_color({"t200": 20, "t100": 60}, TANK_A, I0)
    assert np.allclose(a, b)


def test_absorbance_round_trip():
    a = mix_absorbance({"t200": 10, "t400": 20, "t100": 10}, TANK_A)
    assert np.allclose(absorbance(transmit(a, I0), I0), a)


def test_fit_recovers_synthetic_tanks():
    recipes = [{"t100": 10, "t200": 10, "t400": 10}, {"t100": 30, "t200": 10},
               {"t100": 20, "t400": 50}, {"t100": 10, "t200": 35, "t400": 35},
               {"t200": 50, "t400": 20}]
    rows = [(r, expected_color(r, TANK_A, I0)) for r in recipes]
    ta, i0, r2 = fit_tank_absorbance(rows)
    assert np.allclose(ta["t200"], TANK_A["t200"], atol=1e-6)
    assert np.allclose(ta["t400"], TANK_A["t400"], atol=1e-6)
    assert np.allclose(i0, I0, atol=1e-6)


def test_carryover_fraction_recovered():
    new, old = {"t100": 20, "t400": 50}, {"t100": 30, "t200": 10}
    meas = transmit(carryover_absorbance(new, old, TANK_A, 0.15), I0)
    frac, res = fit_carryover_fraction(meas, new, old, TANK_A, I0)
    assert abs(frac - 0.15) < 1e-6 and res < 1e-6


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)

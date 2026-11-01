"""Run: python -m pytest tests  (or: python tests/fluidmix/colour/test_absorbance.py)"""
import os, sys
import math
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fluidmix.colour import delta_e76
from fluidmix.colour.absorbance import (expected_color, fit_tank_absorbance, fit_carryover_fraction,
                                 carryover_absorbance, transmit, absorbance, mix_absorbance)

TANK_A = {"t200": np.array([0.1, 0.6, 1.5]), "t400": np.array([1.4, 0.4, 0.05])}
I0 = np.array([0.8, 0.8, 0.75])


@pytest.mark.parametrize(
    ("former_copy", "left", "right"),
    [
        ("varied_stage_measurement", (0.0, 0.0, 0.0), (3.0, 4.0, 12.0)),
        ("vision_dataset_plan", (50.25, -4.5, 8.0), (49.0, 2.0, -1.0)),
        ("reference_bank_materializer", (-0.0, 1e-12, -1e-12), (0.0, 0.0, 0.0)),
        ("fill_clips_inline", (10.0, -2.0, 3.0), (4.0, 6.0, -9.0)),
    ],
)
def test_delta_e76_matches_every_product_scalar_copy(former_copy, left, right):
    assert former_copy
    frozen = math.sqrt(
        sum(
            (float(x) - float(y)) ** 2
            for x, y in zip(left, right, strict=True)
        )
    )
    assert delta_e76(left, right) == frozen


def test_product_scalar_adapters_match_canonical_delta_e76():
    from fluidmix.measurement.varied_stage_measurement import _delta_e76 as varied_distance
    from fluidmix.memory.history.vision_dataset_plan import _distance as plan_distance
    from fluidmix.memory.reference_bank.materializer import _delta as bank_distance

    left = (50.25, -4.5, 8.0)
    right = (49.0, 2.0, -1.0)
    expected = delta_e76(left, right)
    for adapter in (
        varied_distance,
        plan_distance,
        bank_distance,
    ):
        assert adapter(list(left), list(right)) == pytest.approx(expected)


def test_product_adapter_preserves_pairwise_float_conversion_order():
    from fluidmix.measurement.varied_stage_measurement import _delta_e76 as varied_distance

    calls = []

    class Tracked:
        def __init__(self, label, value):
            self.label = label
            self.value = value

        def __float__(self):
            calls.append(self.label)
            return float(self.value)

    left = [Tracked(f"left-{index}", index) for index in range(3)]
    right = [Tracked(f"right-{index}", index + 1) for index in range(3)]
    assert varied_distance(left, right) == pytest.approx(math.sqrt(3.0))
    assert calls == [
        "left-0", "right-0", "left-1", "right-1", "left-2", "right-2"
    ]


def test_delta_e76_preserves_scalar_underflow_overflow_and_nonfinite_semantics():
    assert delta_e76((1e-200, 0.0, 0.0), (0.0, 0.0, 0.0)) == 0.0
    with pytest.raises(OverflowError):
        delta_e76((1e200, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert math.isnan(delta_e76((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0)))


def test_stage_distance_remains_distinct_for_subtract_before_float_semantics():
    from fluidmix.decision.stage import _distance as stage_distance

    left = (9007199254740993, 0, 0)
    right = (9007199254740992, 0, 0)
    assert stage_distance(left, right) == 1.0
    assert delta_e76(left, right) == 0.0


def test_public_rgb_delta_e_retains_numpy_batch_and_nonfinite_semantics():
    from fluidmix.colour.absorbance import delta_e, srgb_to_lab

    left = np.asarray([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]])
    right = np.asarray([[30.0, 20.0, 10.0], [10.0, 20.0, 30.0]])
    assert delta_e(left, right) == float(np.linalg.norm(srgb_to_lab(left) - srgb_to_lab(right)))
    assert math.isnan(delta_e((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0)))


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

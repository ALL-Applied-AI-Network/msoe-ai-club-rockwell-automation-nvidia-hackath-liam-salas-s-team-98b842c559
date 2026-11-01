from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_twin_colour_helpers_match_product_numerically():
    from fluidmix.colour import absorbance as product
    from twin import colour

    transmission = np.asarray([0.25, 0.5, 0.75])
    np.testing.assert_allclose(
        colour.absorbance(transmission, np.ones(3)),
        product.absorbance(transmission, np.ones(3)),
    )
    tank_a = {
        "t100": np.zeros(3),
        "t200": np.asarray([0.1, 0.2, 0.3]),
        "t400": np.asarray([0.3, 0.2, 0.1]),
    }
    recipe = {"t100": 10, "t200": 20, "t400": 30}
    expected = product.expected_color(recipe, tank_a, np.ones(3))
    np.testing.assert_allclose(colour.expected_color(recipe, tank_a, np.ones(3)), expected)
    np.testing.assert_allclose(colour.linear_to_srgb(expected), product.linear_to_srgb(expected))
    np.testing.assert_allclose(
        colour.srgb_to_lab(colour.linear_to_srgb(expected)),
        product.srgb_to_lab(product.linear_to_srgb(expected)),
    )
    assert colour.delta_e((10, 20, 30), (30, 20, 10)) == pytest.approx(
        product.delta_e((10, 20, 30), (30, 20, 10))
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)),
        ((50.25, -4.5, 8.0), (49.0, 2.0, -1.0)),
        ((-0.0, 1e-12, -1e-12), (0.0, 0.0, 0.0)),
    ],
)
def test_twin_delta_e76_matches_every_twin_scalar_copy(left, right):
    import math
    from twin.colour import delta_e76

    frozen = math.sqrt(
        sum((float(x) - float(y)) ** 2 for x, y in zip(left, right, strict=True))
    )
    assert delta_e76(left, right) == frozen


def test_twin_scalar_adapters_match_canonical_delta_e76():
    from twin.colour import delta_e76
    from twin.postrender.wp1_postrender import _lab_distance

    left = np.asarray((50.25, -4.5, 8.0))
    right = np.asarray((49.0, 2.0, -1.0))
    expected = delta_e76(left, right)
    assert _lab_distance(left, right) == pytest.approx(expected)


def test_twin_adapter_preserves_pairwise_float_conversion_order():
    import math
    from twin.postrender.wp1_postrender import _lab_distance

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
    assert _lab_distance(left, right) == pytest.approx(math.sqrt(3.0))
    assert calls == [
        "left-0", "right-0", "left-1", "right-1", "left-2", "right-2"
    ]


def test_twin_delta_e76_preserves_scalar_extreme_value_semantics():
    import math
    from twin.colour import delta_e76

    assert delta_e76((1e-200, 0.0, 0.0), (0.0, 0.0, 0.0)) == 0.0
    with pytest.raises(OverflowError):
        delta_e76((1e200, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert math.isnan(delta_e76((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0)))


def test_twin_public_rgb_delta_e_retains_numpy_batch_and_nonfinite_semantics():
    import math
    from twin.colour import delta_e, srgb_to_lab

    left = np.asarray([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]])
    right = np.asarray([[30.0, 20.0, 10.0], [10.0, 20.0, 30.0]])
    assert delta_e(left, right) == float(np.linalg.norm(srgb_to_lab(left) - srgb_to_lab(right)))
    assert math.isnan(delta_e((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0)))


@pytest.mark.parametrize(
    "recipe",
    [
        {"unknown": 1},
        {"t100": 0, "t200": 0, "t400": 0},
        {"t100": -1, "t200": 1, "t400": 0},
    ],
)
def test_twin_expected_colour_rejects_the_same_invalid_recipes(recipe):
    from fluidmix.colour.absorbance import expected_color as product_expected
    from twin.colour import expected_color as twin_expected

    for expected in (product_expected, twin_expected):
        with pytest.raises(ValueError) as error:
            expected(recipe, {}, np.ones(3))
        assert str(error.value)

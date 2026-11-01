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

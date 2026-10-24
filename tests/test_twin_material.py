import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_direct_material_preserves_endpoints_and_metadata():
    from fluidmix.twin_material import make_direct_material

    spec = make_direct_material((0, 0.5, 1), 0, "M00")

    assert spec.color == (0.0, 0.5, 1.0)
    assert spec.opacity == 0.0
    assert spec.variant_id == "M00"
    assert spec.layer_metadata() == {
        "fluidmix_variant_mode": "direct_material",
        "fluidmix_variant_id": "M00",
        "fluidmix_material_color": "0,0.5,1",
        "fluidmix_opacity": 0.0,
    }
    assert "recipe" not in " ".join(spec.layer_metadata())
    assert "tank" not in " ".join(spec.layer_metadata())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0,0,0", (0.0, 0.0, 0.0)),
        ("1,1,1", (1.0, 1.0, 1.0)),
        ("0.006,0.2,0.3", (0.006, 0.2, 0.3)),
    ],
)
def test_parse_linear_rgb_accepts_only_three_finite_unit_values(text, expected):
    from fluidmix.twin_material import parse_linear_rgb

    assert parse_linear_rgb(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0,0",
        "0,0,0,0",
        "red,0,0",
        "nan,0,0",
        "inf,0,0",
        "-0.001,0,0",
        "0,0,1.001",
    ],
)
def test_parse_linear_rgb_rejects_invalid_values_without_clipping(text):
    from fluidmix.twin_material import parse_linear_rgb

    with pytest.raises((TypeError, ValueError)):
        parse_linear_rgb(text)


@pytest.mark.parametrize("opacity", [False, math.nan, math.inf, -0.001, 1.001, "0.6"])
def test_direct_material_rejects_invalid_opacity_without_coercion(opacity):
    from fluidmix.twin_material import make_direct_material

    with pytest.raises((TypeError, ValueError)):
        make_direct_material((0.1, 0.2, 0.3), opacity, "M00")


@pytest.mark.parametrize("variant_id", ["", "has space", "../escape", "a" * 65])
def test_direct_material_rejects_unsafe_variant_id(variant_id):
    from fluidmix.twin_material import make_direct_material

    with pytest.raises(ValueError):
        make_direct_material((0.1, 0.2, 0.3), 0.6, variant_id)


def test_select_material_mode_preserves_legacy_pair_and_accepts_direct_mode():
    from fluidmix.twin_material import DirectMaterial, select_material_mode

    assert select_material_mode(
        recipe="t100=20,t400=50",
        tanks="tanks.json",
        material_color=None,
        opacity=None,
        variant_id=None,
    ) == ("legacy_recipe", None)

    mode, spec = select_material_mode(
        recipe=None,
        tanks=None,
        material_color=(0.1, 0.2, 0.3),
        opacity=0.6,
        variant_id="M03",
    )
    assert mode == "direct_material"
    assert isinstance(spec, DirectMaterial)
    assert spec.color == (0.1, 0.2, 0.3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"recipe": "t100=20"},
        {"tanks": "tanks.json"},
        {"material_color": (0.1, 0.2, 0.3)},
        {"material_color": (0.1, 0.2, 0.3), "opacity": 0.6},
        {"material_color": (0.1, 0.2, 0.3), "variant_id": "M00"},
        {
            "recipe": "t100=20",
            "tanks": "tanks.json",
            "material_color": (0.1, 0.2, 0.3),
            "opacity": 0.6,
            "variant_id": "M00",
        },
    ],
)
def test_select_material_mode_rejects_missing_or_mixed_inputs(kwargs):
    from fluidmix.twin_material import select_material_mode

    all_args = {
        "recipe": None,
        "tanks": None,
        "material_color": None,
        "opacity": None,
        "variant_id": None,
    }
    all_args.update(kwargs)
    with pytest.raises(ValueError):
        select_material_mode(**all_args)


def test_script_has_no_top_level_pxr_import():
    source = (ROOT / "scripts" / "twin_liquid_variant.py").read_text()
    before_functions = source.split("def ", 1)[0]
    assert "from pxr" not in before_functions
    assert "import pxr" not in before_functions

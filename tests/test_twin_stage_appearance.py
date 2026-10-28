"""CPU contract tests for time-sampled direct-appearance USD layers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest


def _write_twin(root: Path) -> Path:
    source = root / "stations/fill_station/fill_station.usda"
    geometry = source.parent / "geometry/fill_station.usda"
    geometry.parent.mkdir(parents=True)
    geometry.write_text(
        '''#usda 1.0
(
    defaultPrim = "FillStation"
)
def Xform "FillStation" {}
'''
    )
    source.write_text(
        '''#usda 1.0
(
    defaultPrim = "FillStation"
    startTimeCode = 1
    endTimeCode = 686
    framesPerSecond = 30
    timeCodesPerSecond = 30
    subLayers = [@geometry/fill_station.usda@]
)

def Xform "FillStation"
{
    def Scope "_materials"
    {
        def Material "USD_blue_liquid_approximation"
        {
            def Shader "Principled_BSDF"
            {
                color3f inputs:diffuseColor = (0.006, 0.2, 0.3)
                float inputs:opacity = 0.6
            }
        }
    }
}
'''
    )
    return source


def _samples():
    return [
        {"time_code": 171, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6},
        {"time_code": 342, "diffuse_color_linear": [0.2, 0.3, 0.4], "opacity": 0.7},
        {"time_code": 686, "diffuse_color_linear": [0.3, 0.4, 0.5], "opacity": 0.8},
    ]


def test_request_rejects_duplicate_or_unsorted_time_codes():
    from fluidmix.twin_stage_appearance import validate_stage_appearance_request

    samples = [
        {"time_code": 171, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6},
        {"time_code": 171, "diffuse_color_linear": [0.2, 0.3, 0.4], "opacity": 0.6},
    ]

    with pytest.raises(ValueError, match="increasing"):
        validate_stage_appearance_request(samples=samples, variant_id="run-001")


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [{"time_code": True, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6}],
        [{"time_code": 171.0, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6}],
        [{"time_code": 0, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6}],
        [{"time_code": 687, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6}],
        [{"time_code": 171, "diffuse_color_linear": [math.nan, 0.2, 0.3], "opacity": 0.6}],
        [{"time_code": 171, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 1.1}],
        [{"time_code": 171, "diffuse_color_linear": [0.1, 0.2, 0.3], "opacity": 0.6, "oracle": True}],
    ],
)
def test_request_rejects_empty_malformed_or_out_of_bounds_samples(samples):
    from fluidmix.twin_stage_appearance import validate_stage_appearance_request

    with pytest.raises((TypeError, ValueError)):
        validate_stage_appearance_request(
            "run-001", samples, start_time_code=1, end_time_code=686
        )


def test_request_is_deep_copied_and_preserves_numeric_values():
    from fluidmix.twin_stage_appearance import validate_stage_appearance_request

    samples = _samples()
    request = validate_stage_appearance_request(
        "run-001", samples, start_time_code=1, end_time_code=686
    )
    samples[0]["diffuse_color_linear"][0] = 0.9

    assert request["samples"] == _samples()


def test_authors_reopens_and_inspects_exact_time_samples(tmp_path):
    from pxr import Gf, Sdf, Usd
    from fluidmix.twin_stage_appearance import (
        create_stage_appearance_layer,
        inspect_stage_appearance_layer,
        stage_appearance_command_plan,
    )
    from fluidmix.twin_appearance import MATERIAL_PATH

    twin = tmp_path / "public-twin"
    source = _write_twin(twin)
    output = tmp_path / "runs"
    output.mkdir()

    created = create_stage_appearance_layer(twin, output, "run-001", _samples())
    layer_path = output / "appearance-layers/run-001.usda"
    inspected = inspect_stage_appearance_layer(layer_path, twin)

    assert created == inspected
    assert created["schema"] == "direct_stage_appearance_v1"
    assert created["stage_sha256"] == hashlib.sha256(layer_path.read_bytes()).hexdigest()
    assert created["source_stage_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert created["samples"] == _samples()
    assert json.loads(json.dumps(stage_appearance_command_plan(created))) == stage_appearance_command_plan(created)
    assert set(stage_appearance_command_plan(created)) == {
        "schema", "variant_id", "stage_sha256", "stage_bytes", "source_stage",
        "source_stage_sha256", "material_path", "samples",
    }

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    diffuse_path = MATERIAL_PATH + ".inputs:diffuseColor"
    opacity_path = MATERIAL_PATH + ".inputs:opacity"
    assert layer.ListTimeSamplesForPath(diffuse_path) == [171.0, 342.0, 686.0]
    assert layer.ListTimeSamplesForPath(opacity_path) == [171.0, 342.0, 686.0]
    assert layer.GetAttributeAtPath(diffuse_path).default is None
    assert layer.GetAttributeAtPath(opacity_path).default is None
    stage = Usd.Stage.Open(str(layer_path), Usd.Stage.LoadNone)
    shader = stage.GetPrimAtPath(MATERIAL_PATH)
    for sample in _samples():
        time = sample["time_code"]
        assert shader.GetAttribute("inputs:diffuseColor").Get(time) == Gf.Vec3f(*sample["diffuse_color_linear"])
        assert shader.GetAttribute("inputs:opacity").Get(time) == pytest.approx(sample["opacity"], abs=1e-7)


def test_creation_never_replaces_existing_output(tmp_path):
    from fluidmix.twin_stage_appearance import create_stage_appearance_layer

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output = tmp_path / "runs"
    occupied = output / "appearance-layers/run-001.usda"
    occupied.parent.mkdir(parents=True)
    occupied.write_text("owned\n")

    with pytest.raises(FileExistsError):
        create_stage_appearance_layer(twin, output, "run-001", _samples())
    assert occupied.read_text() == "owned\n"


@pytest.mark.parametrize("damage", ["tampered_value", "extra_sample"])
def test_inspection_rejects_tampered_or_extra_time_samples(tmp_path, damage):
    from pxr import Gf, Sdf
    from fluidmix.twin_appearance import MATERIAL_PATH
    from fluidmix.twin_stage_appearance import (
        create_stage_appearance_layer,
        inspect_stage_appearance_layer,
    )

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output = tmp_path / "runs"
    output.mkdir()
    record = create_stage_appearance_layer(twin, output, "run-001", _samples())
    layer_path = Path(record["path"])
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    diffuse = layer.GetAttributeAtPath(MATERIAL_PATH + ".inputs:diffuseColor")
    if damage == "tampered_value":
        layer.SetTimeSample(diffuse.path, 342.0, Gf.Vec3f(0.9, 0.9, 0.9))
    else:
        layer.SetTimeSample(diffuse.path, 500.0, Gf.Vec3f(0.2, 0.2, 0.2))
    assert layer.Save()

    with pytest.raises(ValueError, match="sample|metadata"):
        inspect_stage_appearance_layer(layer_path, twin)

"""CPU contracts for uncalibrated OmniSurface liquid overrides."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "cli/twin/scene/twin_transmissive_liquid_variant.py"
RENDER_SCRIPT = ROOT / "cli/twin/render/twin_render_sequence.py"
sys.path.insert(0, str(ROOT / "src"))


LIQUID_MESH = "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
STREAM_MESH = (
    "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream/"
    "Fill_dispensing_stream"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_twin(root: Path) -> Path:
    source = root / "stations/fill_station/fill_station.usda"
    source.parent.mkdir(parents=True)
    source.write_text(
        '''#usda 1.0
(
    defaultPrim = "FillStation"
    metersPerUnit = 1
    startTimeCode = 1
    endTimeCode = 686
    framesPerSecond = 30
    timeCodesPerSecond = 30
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
        def Material "USD_jet_approximation" {}
    }
    def Scope "FillStation_LocalRoot"
    {
        def Scope "Fill_liquid"
        {
            def Mesh "Fill_liquid"
            {
                rel material:binding = </FillStation/_materials/USD_blue_liquid_approximation>
            }
        }
        def Scope "Fill_dispensing_stream"
        {
            def Mesh "Fill_dispensing_stream"
            {
                rel material:binding = </FillStation/_materials/USD_jet_approximation>
            }
        }
    }
}
'''
    )
    return source


def _load_render_module():
    name = "fluidmix_test_transmissive_render_sequence"
    spec = importlib.util.spec_from_file_location(name, RENDER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_module_and_cli_help_are_safe_without_pxr(monkeypatch):
    monkeypatch.setitem(sys.modules, "pxr", None)
    sys.modules.pop("twin.scene.transmissive_optics", None)
    module = __import__("twin.scene.transmissive_optics", fromlist=["*"])
    completed = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert module.TRANSMISSIVE_OPTICS_SCHEMA == "transmissive_optics_v1"
    assert completed.returncode == 0, completed.stderr
    assert "--transmission-color" in completed.stdout
    assert "--scattering-depth-m" in completed.stdout


@pytest.mark.parametrize(
    ("color", "depth"),
    [
        ([0.1, 0.2], 0.01),
        ([0.1, 0.2, 1.1], 0.01),
        ([True, 0.2, 0.3], 0.01),
        ([0.1, 0.2, 0.3], 0),
        ([0.1, 0.2, 0.3], -0.01),
    ],
)
def test_request_rejects_invalid_uncalibrated_color_or_nonpositive_depth(color, depth):
    from twin.scene.transmissive_optics import validate_transmissive_request

    with pytest.raises((TypeError, ValueError)):
        validate_transmissive_request("water-001", color, depth)


def test_authors_and_strictly_inspects_one_shared_omnisurface_material(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Sdf, UsdShade
    from twin.scene.transmissive_optics import (
        MATERIAL_PATH,
        SOURCE_STAGE,
        create_transmissive_layer,
        inspect_transmissive_layer,
        transmissive_command_plan,
    )

    twin = tmp_path / "public-twin"
    source = _write_twin(twin)
    source_before = source.read_bytes()
    output = tmp_path / "runs"
    output.mkdir()

    created = create_transmissive_layer(
        twin, output, "water-001", [0.75, 0.9, 1.0], 0.08
    )
    layer_path = output / "transmissive-optics-layers/water-001.usda"
    inspected = inspect_transmissive_layer(layer_path, twin)

    assert created == inspected
    assert created["schema"] == "transmissive_optics_v1"
    assert created["source_stage"] == SOURCE_STAGE
    assert created["source_stage_sha256"] == _sha(source)
    assert created["stage_sha256"] == _sha(layer_path)
    assert created["transmission_color"] == [0.75, 0.9, 1.0]
    assert created["scattering_depth_m"] == 0.08
    assert created["uncalibrated"] is True
    assert created["timing"] == {
        "start_time_code": 1.0,
        "end_time_code": 686.0,
        "frames_per_second": 30.0,
        "time_codes_per_second": 30.0,
        "meters_per_unit": 1.0,
    }
    assert source.read_bytes() == source_before

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    assert layer is not None
    assert layer.defaultPrim == "FillStation"
    assert layer.pseudoRoot.GetInfo("metersPerUnit") == 1.0
    assert set(layer.customLayerData) == {
        "fluidmix_schema",
        "fluidmix_variant_id",
        "fluidmix_material_path",
        "fluidmix_transmission_color",
        "fluidmix_scattering_depth_m",
        "fluidmix_source_stage",
        "fluidmix_source_stage_sha256",
        "fluidmix_source_meters_per_unit",
        "fluidmix_uncalibrated",
    }
    assert {prop.name for prop in layer.GetPrimAtPath(MATERIAL_PATH).properties} == {
        "outputs:mdl:surface",
        "outputs:mdl:volume",
        "outputs:mdl:displacement",
    }
    assert transmissive_command_plan(created) == {
        "schema": "transmissive_optics_v1",
        "variant_id": "water-001",
        "stage_sha256": _sha(layer_path),
        "stage_bytes": layer_path.stat().st_size,
        "source_stage": SOURCE_STAGE,
        "source_stage_sha256": _sha(source),
        "material_path": MATERIAL_PATH,
        "transmission_color": [0.75, 0.9, 1.0],
        "scattering_depth_m": 0.08,
    }

    stage = pxr.Usd.Stage.Open(str(layer_path), pxr.Usd.Stage.LoadNone)
    material = UsdShade.Material(stage.GetPrimAtPath(MATERIAL_PATH))
    shader = UsdShade.Shader(stage.GetPrimAtPath(MATERIAL_PATH + "/OmniSurface"))
    assert shader.GetImplementationSource() == UsdShade.Tokens.sourceAsset
    assert shader.GetSourceAsset("mdl").path == "OmniSurface.mdl"
    assert shader.GetSourceAssetSubIdentifier("mdl") == "OmniSurface"
    expected_inputs = {
        "enable_specular_transmission": True,
        "specular_transmission_weight": 1.0,
        "specular_transmission_color": (0.75, 0.9, 1.0),
        "specular_transmission_scattering_depth": 0.08,
        "thin_walled": False,
        "diffuse_reflection_weight": 0.0,
        "specular_reflection_ior": 1.333,
        "specular_reflection_roughness": 0.08,
        "enable_opacity": False,
        "geometry_opacity": 1.0,
        "geometry_opacity_threshold": 0.0,
    }
    for name, expected in expected_inputs.items():
        value = shader.GetInput(name).Get()
        if isinstance(expected, tuple):
            assert tuple(value) == pytest.approx(expected, abs=1e-7)
        else:
            assert value == pytest.approx(expected, abs=1e-7)
    for mesh_path in (LIQUID_MESH, STREAM_MESH):
        bound, relationship = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath(mesh_path)
        ).ComputeBoundMaterial()
        assert relationship.GetTargets() == [Sdf.Path(MATERIAL_PATH)]
        assert bound.GetPrim().GetPath() == material.GetPrim().GetPath()


def test_creation_never_replaces_an_existing_transmissive_layer(tmp_path):
    from twin.scene.transmissive_optics import create_transmissive_layer

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output = tmp_path / "runs"
    occupied = output / "transmissive-optics-layers/water-001.usda"
    occupied.parent.mkdir(parents=True)
    occupied.write_text("owned\n")

    with pytest.raises(FileExistsError):
        create_transmissive_layer(twin, output, "water-001", [0.75, 0.9, 1.0], 0.08)
    assert occupied.read_text() == "owned\n"


def test_inspection_rejects_a_split_mesh_binding(tmp_path):
    pytest.importorskip("pxr")
    from twin.scene.transmissive_optics import create_transmissive_layer, inspect_transmissive_layer

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output = tmp_path / "runs"
    output.mkdir()
    created = create_transmissive_layer(
        twin, output, "water-001", [0.75, 0.9, 1.0], 0.08
    )
    layer_path = Path(created["path"])
    text = layer_path.read_text()
    layer_path.write_text(text.replace(
        "</FillStation/_materials/Fluidmix_transmissive_liquid_v1>",
        "</FillStation/_materials/USD_jet_approximation>",
        1,
    ))

    with pytest.raises(ValueError, match="binding"):
        inspect_transmissive_layer(layer_path, twin)


def test_pathtracing_bounce_flags_are_optional_bounded_and_reportable():
    module = _load_render_module()
    parser = module.build_parser()

    defaults = parser.parse_args(["--twin", "/twin", "--out", "/out"])
    configured = parser.parse_args(
        [
            "--twin", "/twin", "--out", "/out",
            "--max-bounces", "12",
            "--max-specular-transmission-bounces", "8",
        ]
    )

    assert module.pathtracing_bounce_settings(defaults) == {}
    assert module.pathtracing_bounce_settings(configured) == {
        "/rtx/pathtracing/maxBounces": 12,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 8,
    }
    assert module.pathtracing_bounce_summary(configured) == {
        "pathtracing_bounce_overrides": {
            "/rtx/pathtracing/maxBounces": 12,
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 8,
        }
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--twin", "/twin", "--out", "/out", "--max-bounces", "0"]
        )

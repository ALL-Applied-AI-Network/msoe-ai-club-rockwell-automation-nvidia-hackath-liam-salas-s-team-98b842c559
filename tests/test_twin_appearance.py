"""CPU contract tests for neutral direct-appearance USD layers."""

from __future__ import annotations

import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/twin_appearance_variant.py"


@pytest.fixture
def pxr_modules():
    return pytest.importorskip("pxr")


def _write_twin(root: Path) -> Path:
    source = root / "stations/fill_station/fill_station.usda"
    geometry = source.parent / "geometry/fill_station.usda"
    geometry.parent.mkdir(parents=True)
    geometry.write_text(
        """#usda 1.0
(
    defaultPrim = "FillStation"
)
def Xform "FillStation" {}
"""
    )
    source.write_text(
        """#usda 1.0
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
"""
    )
    return source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reload_without_pxr(monkeypatch):
    monkeypatch.setitem(sys.modules, "pxr", None)
    sys.modules.pop("fluidmix.twin_appearance", None)
    return importlib.import_module("fluidmix.twin_appearance")


def test_module_and_script_help_are_safe_without_pxr(monkeypatch):
    module = _reload_without_pxr(monkeypatch)
    completed = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert module.APPEARANCE_SCHEMA == "direct_appearance_v1"
    assert completed.returncode == 0, completed.stderr
    assert "--linear-rgb" in completed.stdout
    assert "--opacity" in completed.stdout


def test_direct_appearance_route_does_not_depend_on_absorbance_or_legacy_variant():
    sources = [
        ROOT / "src/fluidmix/twin_appearance.py",
        SCRIPT,
    ]

    for source in sources:
        text = source.read_text()
        assert "fluidmix.absorbance" not in text
        assert "twin_liquid_variant" not in text


@pytest.mark.parametrize(
    "variant_id",
    ["", "a" * 65, ".", "..", "has space", "has/slash", "has\\slash", "é"],
)
def test_request_rejects_unsafe_neutral_ids(variant_id):
    from fluidmix.twin_appearance import validate_appearance_request

    with pytest.raises((TypeError, ValueError), match="variant"):
        validate_appearance_request(variant_id, [0.1, 0.2, 0.3], 0.6)


@pytest.mark.parametrize(
    "rgb",
    [
        [0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
        [True, 0.2, 0.3],
        ["0.1", 0.2, 0.3],
        [-0.01, 0.2, 0.3],
        [1.01, 0.2, 0.3],
        [math.nan, 0.2, 0.3],
        [math.inf, 0.2, 0.3],
    ],
)
def test_request_rejects_incomplete_coerced_nonfinite_or_out_of_range_rgb(rgb):
    from fluidmix.twin_appearance import validate_appearance_request

    with pytest.raises((TypeError, ValueError), match="RGB"):
        validate_appearance_request("R000", rgb, 0.6)


@pytest.mark.parametrize(
    "opacity", [True, "0.6", -0.01, 1.01, math.nan, math.inf]
)
def test_request_rejects_coerced_nonfinite_or_out_of_range_opacity(opacity):
    from fluidmix.twin_appearance import validate_appearance_request

    with pytest.raises((TypeError, ValueError), match="opacity"):
        validate_appearance_request("R000", [0.1, 0.2, 0.3], opacity)


def test_request_preserves_exact_numeric_values_without_clipping_or_coercion():
    from fluidmix.twin_appearance import validate_appearance_request

    request = validate_appearance_request(
        "R_000.alpha-10", [0, 0.192, 1], 0.6000000000000001
    )

    assert request == {
        "variant_id": "R_000.alpha-10",
        "diffuse_color_linear": [0, 0.192, 1],
        "opacity": 0.6000000000000001,
    }
    assert type(request["diffuse_color_linear"][0]) is int


def test_command_plan_projection_has_one_exact_json_shape():
    from fluidmix.twin_appearance import appearance_command_plan

    inspection = {
        "schema": "direct_appearance_v1",
        "variant_id": "R000",
        "stage_sha256": "a" * 64,
        "stage_bytes": 123,
        "source_stage": "stations/fill_station/fill_station.usda",
        "source_stage_sha256": "b" * 64,
        "material_path": (
            "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
        ),
        "diffuse_color_linear": [0.006, 0.2, 0.3],
        "opacity": 0.6,
        "uncalibrated": True,
        "path": "/ignored/R000.usda",
    }

    projected = appearance_command_plan(inspection)

    assert json.loads(json.dumps(projected, allow_nan=False)) == projected
    assert set(projected) == {
        "schema",
        "variant_id",
        "stage_sha256",
        "stage_bytes",
        "source_stage",
        "source_stage_sha256",
        "material_path",
        "diffuse_color_linear",
        "opacity",
    }
    assert "uncalibrated" not in projected
    inspection["diffuse_color_linear"][0] = 0.9
    assert projected["diffuse_color_linear"] == [0.006, 0.2, 0.3]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "other"),
        ("stage_sha256", "A" * 64),
        ("stage_bytes", True),
        ("stage_bytes", 1_048_577),
        ("source_stage", "other.usda"),
        ("source_stage_sha256", "short"),
        ("material_path", "/Other"),
    ],
)
def test_command_plan_projection_rejects_noncanonical_inspection_fields(field, value):
    from fluidmix.twin_appearance import appearance_command_plan

    inspection = {
        "schema": "direct_appearance_v1",
        "variant_id": "R000",
        "stage_sha256": "a" * 64,
        "stage_bytes": 123,
        "source_stage": "stations/fill_station/fill_station.usda",
        "source_stage_sha256": "b" * 64,
        "material_path": (
            "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
        ),
        "diffuse_color_linear": [0.006, 0.2, 0.3],
        "opacity": 0.6,
    }
    inspection[field] = value

    with pytest.raises((TypeError, ValueError)):
        appearance_command_plan(inspection)


def test_authors_reopens_inspects_and_exclusively_publishes_exact_layer(
    tmp_path, pxr_modules
):
    from pxr import Sdf, Usd
    from fluidmix.twin_appearance import (
        MATERIAL_PATH,
        SOURCE_STAGE,
        create_appearance_layer,
        inspect_appearance_layer,
    )

    twin = tmp_path / "public-twin"
    source = _write_twin(twin)
    source_before = source.read_bytes()
    output_root = tmp_path / "runs"
    output_root.mkdir()

    created = create_appearance_layer(
        twin, output_root, "R002", [0.1392, 0.176, 0.213], 0.6
    )
    layer_path = output_root / "appearance-layers/R002.usda"
    inspected = inspect_appearance_layer(layer_path, twin)

    assert created == inspected
    assert created["path"] == str(layer_path.resolve())
    assert created["schema"] == "direct_appearance_v1"
    assert created["variant_id"] == "R002"
    assert created["stage_sha256"] == _sha(layer_path)
    assert created["stage_bytes"] == layer_path.stat().st_size
    assert created["source_stage"] == SOURCE_STAGE
    assert created["source_stage_sha256"] == _sha(source)
    assert created["material_path"] == MATERIAL_PATH
    assert created["diffuse_color_linear"] == [0.1392, 0.176, 0.213]
    assert created["opacity"] == 0.6
    assert created["uncalibrated"] is True
    assert created["timing"] == {
        "start_time_code": 1.0,
        "end_time_code": 686.0,
        "frames_per_second": 30.0,
        "time_codes_per_second": 30.0,
    }
    assert source.read_bytes() == source_before

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    assert layer is not None
    assert layer.defaultPrim == "FillStation"
    assert len(layer.subLayerPaths) == 1
    sublayer = layer.subLayerPaths[0]
    assert not Path(sublayer).is_absolute()
    assert (layer_path.parent / sublayer).resolve() == source.resolve()
    assert set(layer.customLayerData) == {
        "fluidmix_schema",
        "fluidmix_variant_id",
        "fluidmix_material_path",
        "fluidmix_diffuse_color_linear",
        "fluidmix_opacity",
        "fluidmix_source_stage",
        "fluidmix_source_stage_sha256",
        "fluidmix_uncalibrated",
    }
    expected_prim_paths = {
        "/FillStation",
        "/FillStation/_materials",
        "/FillStation/_materials/USD_blue_liquid_approximation",
        MATERIAL_PATH,
    }
    actual_prim_paths = set()

    def collect_prim_paths(spec):
        actual_prim_paths.add(str(spec.path))
        for child in spec.nameChildren:
            collect_prim_paths(child)

    for root_prim in layer.rootPrims:
        collect_prim_paths(root_prim)
    assert actual_prim_paths == expected_prim_paths
    for path in expected_prim_paths:
        assert layer.GetPrimAtPath(path).specifier == Sdf.SpecifierOver
    shader_spec = layer.GetPrimAtPath(MATERIAL_PATH)
    assert {property_spec.name for property_spec in shader_spec.properties} == {
        "inputs:diffuseColor",
        "inputs:opacity",
    }
    assert shader_spec.attributes["inputs:diffuseColor"].typeName == Sdf.ValueTypeNames.Color3f
    assert shader_spec.attributes["inputs:opacity"].typeName == Sdf.ValueTypeNames.Float

    stage = Usd.Stage.Open(str(layer_path), Usd.Stage.LoadNone)
    shader = stage.GetPrimAtPath(MATERIAL_PATH)
    assert tuple(shader.GetAttribute("inputs:diffuseColor").Get()) == pytest.approx(
        (0.1392, 0.176, 0.213), abs=1e-7
    )
    assert shader.GetAttribute("inputs:opacity").Get() == pytest.approx(0.6, abs=1e-7)
    stack = created["non_session_usd_layer_stack"]
    assert [record["ordinal"] for record in stack] == [1, 2, 3]
    assert [record["resolved_path"] for record in stack] == [
        str(layer_path.resolve()),
        str(source.resolve()),
        str((source.parent / "geometry/fill_station.usda").resolve()),
    ]
    assert all(set(record) == {"ordinal", "resolved_path", "sha256", "bytes"} for record in stack)
    assert not list((output_root / "appearance-layers").glob(".*.tmp.usda"))


@pytest.mark.parametrize("occupied_kind", ["file", "directory", "dangling_symlink"])
def test_creation_never_replaces_an_occupied_final_path(
    tmp_path, pxr_modules, occupied_kind
):
    from fluidmix.twin_appearance import create_appearance_layer

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output_root = tmp_path / "runs"
    final = output_root / "appearance-layers/R000.usda"
    final.parent.mkdir(parents=True)
    if occupied_kind == "file":
        final.write_text("owned by another run\n")
    elif occupied_kind == "directory":
        final.mkdir()
    else:
        final.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError):
        create_appearance_layer(twin, output_root, "R000", [0.006, 0.2, 0.3], 0.6)

    assert os.path.lexists(final)
    if occupied_kind == "file":
        assert final.read_text() == "owned by another run\n"


@pytest.mark.parametrize("symlink_role", ["twin", "output_root", "appearance_directory"])
def test_creation_rejects_symlink_path_components(tmp_path, pxr_modules, symlink_role):
    from fluidmix.twin_appearance import create_appearance_layer

    real_twin = tmp_path / "real-twin"
    _write_twin(real_twin)
    real_output = tmp_path / "real-runs"
    real_output.mkdir()
    twin = real_twin
    output = real_output
    if symlink_role == "twin":
        twin = tmp_path / "twin-link"
        twin.symlink_to(real_twin, target_is_directory=True)
    elif symlink_role == "output_root":
        output = tmp_path / "runs-link"
        output.symlink_to(real_output, target_is_directory=True)
    else:
        (real_output / "appearance-layers").symlink_to(
            tmp_path / "elsewhere", target_is_directory=True
        )

    with pytest.raises(ValueError, match="symlink"):
        create_appearance_layer(twin, output, "R000", [0.006, 0.2, 0.3], 0.6)


@pytest.mark.parametrize("source_damage", ["prim_type", "diffuse_type", "opacity_missing"])
def test_creation_rejects_wrong_source_shader_contract(
    tmp_path, pxr_modules, source_damage
):
    from fluidmix.twin_appearance import create_appearance_layer

    twin = tmp_path / "public-twin"
    source = _write_twin(twin)
    text = source.read_text()
    if source_damage == "prim_type":
        text = text.replace('def Shader "Principled_BSDF"', 'def Scope "Principled_BSDF"')
    elif source_damage == "diffuse_type":
        text = text.replace("color3f inputs:diffuseColor", "color3d inputs:diffuseColor")
    else:
        text = text.replace("                float inputs:opacity = 0.6\n", "")
    source.write_text(text)
    output_root = tmp_path / "runs"
    output_root.mkdir()

    with pytest.raises(ValueError, match="source|shader|Shader"):
        create_appearance_layer(twin, output_root, "R000", [0.006, 0.2, 0.3], 0.6)


def test_creation_rejects_composed_dependency_outside_public_twin(
    tmp_path, pxr_modules
):
    from fluidmix.twin_appearance import create_appearance_layer

    twin = tmp_path / "public-twin"
    source = _write_twin(twin)
    outside = tmp_path / "outside.usda"
    outside.write_text('#usda 1.0\ndef Xform "Outside" {}\n')
    source.write_text(
        source.read_text().replace(
            "@geometry/fill_station.usda@", "@../../../outside.usda@"
        )
    )
    output_root = tmp_path / "runs"
    output_root.mkdir()

    with pytest.raises(ValueError, match="outside the public twin"):
        create_appearance_layer(twin, output_root, "R000", [0.006, 0.2, 0.3], 0.6)


def test_inspection_rejects_oversized_nonregular_and_filename_mismatch(
    tmp_path, pxr_modules
):
    from fluidmix.twin_appearance import (
        MAX_APPEARANCE_LAYER_BYTES,
        create_appearance_layer,
        inspect_appearance_layer,
    )

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    record = create_appearance_layer(twin, output_root, "R000", [0.006, 0.2, 0.3], 0.6)
    layer = Path(record["path"])
    mismatch = layer.with_name("WRONG.usda")
    layer.rename(mismatch)
    with pytest.raises(ValueError, match="filename|variant"):
        inspect_appearance_layer(mismatch, twin)

    oversized = layer
    oversized.write_bytes(b"x" * (MAX_APPEARANCE_LAYER_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        inspect_appearance_layer(oversized, twin)

    oversized.unlink()
    oversized.mkdir()
    with pytest.raises(ValueError, match="regular"):
        inspect_appearance_layer(oversized, twin)


@pytest.mark.parametrize(
    "damage",
    [
        "extra_metadata",
        "wrong_source_hash",
        "extra_property",
        "attribute_metadata",
        "layer_metadata",
        "reference",
        "sublayer_offset",
        "second_sublayer",
    ],
)
def test_inspection_rejects_malformed_or_broadened_layers(
    tmp_path, pxr_modules, damage
):
    from pxr import Sdf
    from fluidmix.twin_appearance import (
        MATERIAL_PATH,
        create_appearance_layer,
        inspect_appearance_layer,
    )

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    record = create_appearance_layer(twin, output_root, "R000", [0.006, 0.2, 0.3], 0.6)
    layer_path = Path(record["path"])
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    assert layer is not None
    if damage == "extra_metadata":
        custom = dict(layer.customLayerData)
        custom["unexpected"] = "oracle"
        layer.customLayerData = custom
    elif damage == "wrong_source_hash":
        custom = dict(layer.customLayerData)
        custom["fluidmix_source_stage_sha256"] = "0" * 64
        layer.customLayerData = custom
    elif damage == "extra_property":
        Sdf.AttributeSpec(
            layer.GetPrimAtPath(MATERIAL_PATH),
            "inputs:oracleLabel",
            Sdf.ValueTypeNames.String,
        ).default = "leak"
    elif damage == "attribute_metadata":
        layer.GetAttributeAtPath(
            MATERIAL_PATH + ".inputs:opacity"
        ).documentation = "unexpected"
    elif damage == "layer_metadata":
        layer.documentation = "unexpected"
    elif damage == "reference":
        layer.GetPrimAtPath(MATERIAL_PATH).referenceList.Add(
            Sdf.Reference("other.usda")
        )
    elif damage == "sublayer_offset":
        layer.subLayerOffsets[0] = Sdf.LayerOffset(100.0, 2.0)
    else:
        layer.subLayerPaths.append("bogus.usda")
    assert layer.Save()

    with pytest.raises(ValueError):
        inspect_appearance_layer(layer_path, twin)


def test_script_creates_layer_and_prints_inspected_record(tmp_path, pxr_modules):
    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output_root = tmp_path / "runs"
    output_root.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--twin",
            str(twin),
            "--output-root",
            str(output_root),
            "--variant-id",
            "R005",
            "--linear-rgb",
            "0.0504",
            "0.192",
            "0.271",
            "--opacity",
            "0.6",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["variant_id"] == "R005"
    assert record["diffuse_color_linear"] == [0.0504, 0.192, 0.271]
    assert Path(record["path"]) == output_root / "appearance-layers/R005.usda"


def test_creation_removes_only_its_owned_temp_when_a_write_fails(
    tmp_path, pxr_modules, monkeypatch
):
    import fluidmix.twin_appearance as appearance

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    real_write = appearance.os.write
    calls = 0

    def fail_first_write(descriptor, content):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected write failure")
        return real_write(descriptor, content)

    monkeypatch.setattr(appearance.os, "write", fail_first_write)
    with pytest.raises(OSError, match="injected"):
        appearance.create_appearance_layer(
            twin, output_root, "R006", [0.006, 0.2, 0.3], 0.6
        )

    directory = output_root / "appearance-layers"
    assert list(directory.iterdir()) == []


def test_creation_rolls_back_owned_publication_if_source_drifts(
    tmp_path, pxr_modules, monkeypatch
):
    import fluidmix.twin_appearance as appearance

    twin = tmp_path / "public-twin"
    source = _write_twin(twin)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    real_link = appearance.os.link

    def link_then_drift(*args, **kwargs):
        result = real_link(*args, **kwargs)
        with source.open("a") as stream:
            stream.write("\n# injected source drift\n")
        return result

    monkeypatch.setattr(appearance.os, "link", link_then_drift)
    with pytest.raises(ValueError, match="source|canonical"):
        appearance.create_appearance_layer(
            twin, output_root, "R007", [0.006, 0.2, 0.3], 0.6
        )

    directory = output_root / "appearance-layers"
    assert list(directory.iterdir()) == []


def test_creation_never_reports_success_if_published_inode_is_replaced(
    tmp_path, pxr_modules, monkeypatch
):
    import fluidmix.twin_appearance as appearance

    twin = tmp_path / "public-twin"
    _write_twin(twin)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    real_link = appearance.os.link

    def link_then_replace(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        destination = Path(destination)
        same_bytes = destination.read_bytes()
        destination.unlink()
        destination.write_bytes(same_bytes)
        return result

    monkeypatch.setattr(appearance.os, "link", link_then_replace)
    with pytest.raises(ValueError, match="identity|replaced"):
        appearance.create_appearance_layer(
            twin, output_root, "R003", [0.006, 0.2, 0.3], 0.6
        )

    replacement = output_root / "appearance-layers/R003.usda"
    assert replacement.is_file()

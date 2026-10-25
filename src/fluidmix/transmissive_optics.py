"""Author and strictly inspect uncalibrated OmniSurface liquid overrides.

This CPU-only authoring route creates one MDL material for the fill liquid and
the dispensing stream.  The transmission colour and scattering depth are
explicit, uncalibrated render controls: no absorbance fitting, mixture math, or
physical-validation claim is made here.  A later RTX render remains required to
test shader availability and appearance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import secrets

from fluidmix.twin_appearance import (
    MAX_APPEARANCE_LAYER_BYTES,
    SOURCE_STAGE,
    _all_prim_specs,
    _canonical_relative_sublayer,
    _existing_directory,
    _float32,
    _fsync_directory,
    _import_pxr,
    _lexical_absolute,
    _regular_snapshot,
    _require_owned_identity,
    _source_record,
    _validate_source_stage,
    _write_exclusive,
)


TRANSMISSIVE_OPTICS_SCHEMA = "transmissive_optics_v1"
MATERIAL_PATH = "/FillStation/_materials/Fluidmix_transmissive_liquid_v1"
SHADER_PATH = MATERIAL_PATH + "/OmniSurface"
LIQUID_MESH_PATH = "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
STREAM_MESH_PATH = (
    "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream/"
    "Fill_dispensing_stream"
)
OMNISURFACE_MDL = "OmniSurface.mdl"
OMNISURFACE_SUBIDENTIFIER = "OmniSurface"
WATER_IOR = 1.333
LOW_ROUGHNESS = 0.08
_SAFE_VARIANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CUSTOM_LAYER_FIELDS = frozenset(
    {
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
)
_COMMAND_PLAN_FIELDS = (
    "schema",
    "variant_id",
    "stage_sha256",
    "stage_bytes",
    "source_stage",
    "source_stage_sha256",
    "material_path",
    "transmission_color",
    "scattering_depth_m",
)
_MDL_OUTPUTS = (
    "outputs:mdl:surface",
    "outputs:mdl:volume",
    "outputs:mdl:displacement",
)
_EXPECTED_PRIM_PATHS = (
    "/FillStation",
    "/FillStation/_materials",
    MATERIAL_PATH,
    SHADER_PATH,
    "/FillStation/FillStation_LocalRoot",
    "/FillStation/FillStation_LocalRoot/Fill_liquid",
    LIQUID_MESH_PATH,
    "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream",
    STREAM_MESH_PATH,
)
_SHADER_INPUTS = {
    "inputs:enable_specular_transmission": ("bool", True),
    "inputs:specular_transmission_weight": ("float", 1.0),
    "inputs:specular_transmission_color": ("color3f", None),
    "inputs:specular_transmission_scattering_depth": ("float", None),
    "inputs:thin_walled": ("bool", False),
    "inputs:diffuse_reflection_weight": ("float", 0.0),
    "inputs:specular_reflection_ior": ("float", WATER_IOR),
    "inputs:specular_reflection_roughness": ("float", LOW_ROUGHNESS),
    "inputs:enable_opacity": ("bool", False),
    "inputs:geometry_opacity": ("float", 1.0),
    "inputs:geometry_opacity_threshold": ("float", 0.0),
}


def _number(value: object, role: str, *, minimum: float, maximum: float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{role} must be a non-Boolean number")
    if not math.isfinite(value):
        raise ValueError(f"{role} must be finite")
    if value < minimum or value > maximum:
        raise ValueError(f"{role} must be between {minimum} and {maximum}")
    return value


def validate_transmissive_request(
    variant_id: object, transmission_color: object, scattering_depth_m: object
) -> dict[str, object]:
    """Validate explicit, uncalibrated Beer-law material controls.

    The colour is bounded to renderable linear RGB and depth must be a positive
    scene-meter distance.  Inputs are neither clipped nor converted.
    """

    if not isinstance(variant_id, str) or _SAFE_VARIANT_ID.fullmatch(variant_id) is None:
        raise ValueError(
            "variant ID must be 1-64 ASCII letters, digits, dot, underscore, or hyphen"
        )
    if (
        isinstance(transmission_color, (str, bytes))
        or not isinstance(transmission_color, Sequence)
        or len(transmission_color) != 3
    ):
        raise ValueError("transmission color must contain exactly three numbers")
    color = [
        _number(value, f"transmission color channel {index}", minimum=0.0, maximum=1.0)
        for index, value in enumerate(transmission_color)
    ]
    depth = _number(
        scattering_depth_m,
        "scattering depth in scene meters",
        minimum=math.nextafter(0.0, 1.0),
        maximum=float("inf"),
    )
    return {
        "variant_id": variant_id,
        "transmission_color": color,
        "scattering_depth_m": depth,
    }


def transmissive_command_plan(inspection: Mapping[str, object]) -> dict[str, object]:
    """Project one strict inspection into the preregistered render plan."""

    if not isinstance(inspection, Mapping):
        raise TypeError("transmissive optics inspection must be a mapping")
    try:
        projected = {field: inspection[field] for field in _COMMAND_PLAN_FIELDS}
    except KeyError as error:
        raise ValueError(
            f"transmissive optics inspection is missing {error.args[0]!r}"
        ) from error
    request = validate_transmissive_request(
        projected["variant_id"],
        projected["transmission_color"],
        projected["scattering_depth_m"],
    )
    if projected["schema"] != TRANSMISSIVE_OPTICS_SCHEMA or type(projected["schema"]) is not str:
        raise ValueError("transmissive optics inspection schema is invalid")
    for field in ("stage_sha256", "source_stage_sha256"):
        if not isinstance(projected[field], str) or _SHA256.fullmatch(projected[field]) is None:
            raise ValueError(f"transmissive optics inspection {field} is not a SHA-256")
    stage_bytes = projected["stage_bytes"]
    if (
        isinstance(stage_bytes, bool)
        or not isinstance(stage_bytes, int)
        or stage_bytes <= 0
        or stage_bytes > MAX_APPEARANCE_LAYER_BYTES
    ):
        raise ValueError("transmissive optics inspection stage_bytes is outside the size bound")
    if projected["source_stage"] != SOURCE_STAGE or type(projected["source_stage"]) is not str:
        raise ValueError("transmissive optics inspection source stage is invalid")
    if projected["material_path"] != MATERIAL_PATH or type(projected["material_path"]) is not str:
        raise ValueError("transmissive optics inspection material path is invalid")
    projected["transmission_color"] = list(request["transmission_color"])
    projected["scattering_depth_m"] = request["scattering_depth_m"]
    return projected


def _source_timing_and_scale(source: Path, source_snapshot: Mapping[str, object], Sdf, Usd):
    """Validate canonical source timing plus the required metres scene scale."""

    _, timing = _validate_source_stage(source, source_snapshot, Sdf, Usd)
    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("canonical fill-station source stage could not be composed")
    meters_per_unit = float(stage.GetMetadata("metersPerUnit"))
    if meters_per_unit != 1.0:
        raise ValueError("canonical fill-station source metersPerUnit must be exactly 1.0")
    return {**timing, "meters_per_unit": meters_per_unit}


def _expected_shader_values(request: Mapping[str, object], Gf) -> dict[str, object]:
    values = {
        name: _float32(value) if type_name == "float" and value is not None else value
        for name, (type_name, value) in _SHADER_INPUTS.items()
    }
    values["inputs:specular_transmission_color"] = Gf.Vec3f(
        *(float(value) for value in request["transmission_color"])
    )
    values["inputs:specular_transmission_scattering_depth"] = _float32(
        request["scattering_depth_m"]
    )
    return values


def _validate_layer_relationship(relationship, expected_target, role: str) -> None:
    if relationship.targetPathList.GetAddedOrExplicitItems() != (expected_target,):
        raise ValueError(f"{role} binding does not target the one shared material")
    if relationship.targetPathList.deletedItems or relationship.targetPathList.orderedItems:
        raise ValueError(f"{role} binding has unsupported relationship edits")


def inspect_transmissive_layer(
    layer_path: str | os.PathLike[str],
    twin_root: str | os.PathLike[str],
    *,
    expected_variant_id: str | None = None,
    require_filename_match: bool = True,
) -> dict[str, object]:
    """Reopen and strictly inspect one composed transmissive-optics layer."""

    Gf, Sdf, Usd, Vt = _import_pxr()
    lexical_layer = _lexical_absolute(layer_path)
    if lexical_layer.suffix != ".usda":
        raise ValueError("transmissive optics layer filename must end with .usda")
    layer_snapshot = _regular_snapshot(
        lexical_layer, "transmissive optics layer", max_bytes=MAX_APPEARANCE_LAYER_BYTES
    )
    twin, source_snapshot = _source_record(twin_root)
    source = Path(source_snapshot["path"])
    source_timing = _source_timing_and_scale(source, source_snapshot, Sdf, Usd)
    layer = Sdf.Layer.FindOrOpen(str(lexical_layer))
    if layer is None or not layer.Reload(force=True):
        raise ValueError("transmissive optics layer is not a readable USD layer")
    expected_info = {
        "subLayerOffsets", "framesPerSecond", "customLayerData", "startTimeCode",
        "endTimeCode", "subLayers", "timeCodesPerSecond", "defaultPrim", "metersPerUnit",
    }
    if set(layer.pseudoRoot.ListInfoKeys()) != expected_info:
        raise ValueError("transmissive optics layer has missing or extra root metadata")
    custom = dict(layer.customLayerData)
    if set(custom) != _CUSTOM_LAYER_FIELDS:
        raise ValueError("transmissive optics metadata fields are missing or extra")
    if custom["fluidmix_schema"] != TRANSMISSIVE_OPTICS_SCHEMA:
        raise ValueError("transmissive optics schema metadata is invalid")
    request = validate_transmissive_request(
        custom["fluidmix_variant_id"],
        list(custom["fluidmix_transmission_color"]),
        custom["fluidmix_scattering_depth_m"],
    )
    if expected_variant_id is not None and request["variant_id"] != expected_variant_id:
        raise ValueError("transmissive optics variant metadata does not match expected variant")
    if require_filename_match and lexical_layer.stem != request["variant_id"]:
        raise ValueError("transmissive optics layer filename stem does not match variant")
    if custom["fluidmix_material_path"] != MATERIAL_PATH:
        raise ValueError("transmissive optics material metadata is invalid")
    if custom["fluidmix_source_stage"] != SOURCE_STAGE:
        raise ValueError("transmissive optics source-stage metadata is invalid")
    if (
        not isinstance(custom["fluidmix_source_stage_sha256"], str)
        or _SHA256.fullmatch(custom["fluidmix_source_stage_sha256"]) is None
        or custom["fluidmix_source_stage_sha256"] != source_snapshot["sha256"]
    ):
        raise ValueError("transmissive optics source-stage hash does not match source bytes")
    if custom["fluidmix_source_meters_per_unit"] != 1.0:
        raise ValueError("transmissive optics source scale metadata is invalid")
    if custom["fluidmix_uncalibrated"] is not True:
        raise ValueError("transmissive optics layer must declare uncalibrated=true")
    if len(layer.subLayerPaths) != 1 or len(layer.subLayerOffsets) != 1:
        raise ValueError("transmissive optics layer must have exactly one identity sublayer")
    if layer.subLayerOffsets[0] != Sdf.LayerOffset():
        raise ValueError("transmissive optics sublayer offset must be identity")
    sublayer = layer.subLayerPaths[0]
    if (
        not isinstance(sublayer, str)
        or not sublayer
        or "\\" in sublayer
        or PurePosixPath(sublayer).is_absolute()
        or posixpath.normpath(sublayer) != sublayer
        or sublayer != _canonical_relative_sublayer(lexical_layer, source)
    ):
        raise ValueError("transmissive optics sublayer is not canonical")
    if (lexical_layer.parent / Path(*PurePosixPath(sublayer).parts)).resolve(strict=True) != source:
        raise ValueError("transmissive optics sublayer does not resolve to source")
    if layer.defaultPrim != "FillStation":
        raise ValueError("transmissive optics default prim must be FillStation")
    layer_timing = {
        "start_time_code": float(layer.startTimeCode),
        "end_time_code": float(layer.endTimeCode),
        "frames_per_second": float(layer.framesPerSecond),
        "time_codes_per_second": float(layer.timeCodesPerSecond),
        "meters_per_unit": float(layer.pseudoRoot.GetInfo("metersPerUnit")),
    }
    if layer_timing != source_timing:
        raise ValueError("transmissive optics timing or scale differs from source")

    prims = _all_prim_specs(layer)
    if tuple(prims) != _EXPECTED_PRIM_PATHS:
        raise ValueError("transmissive optics layer prim structure is missing, extra, or reordered")
    ancestor_paths = {
        "/FillStation", "/FillStation/_materials", "/FillStation/FillStation_LocalRoot",
        "/FillStation/FillStation_LocalRoot/Fill_liquid",
        "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream",
    }
    for path in ancestor_paths:
        prim = prims[path]
        if prim.specifier != Sdf.SpecifierOver or prim.typeName or set(prim.ListInfoKeys()) != {"specifier"}:
            raise ValueError(f"transmissive optics ancestor {path} is not an untyped over")
        if prim.properties:
            raise ValueError("transmissive optics ancestor has properties")
    material = prims[MATERIAL_PATH]
    if material.specifier != Sdf.SpecifierDef or material.typeName != "Material":
        raise ValueError("transmissive optics material prim is invalid")
    if {prop.name for prop in material.properties} != set(_MDL_OUTPUTS):
        raise ValueError("transmissive optics material has missing or extra properties")
    for output_name in _MDL_OUTPUTS:
        material_output = material.attributes[output_name]
        if material_output.typeName != Sdf.ValueTypeNames.Token:
            raise ValueError("transmissive optics material output has wrong type")
        if material_output.connectionPathList.GetAddedOrExplicitItems() != (
            Sdf.Path(SHADER_PATH + ".outputs:out"),
        ):
            raise ValueError(
                "transmissive optics material output is not connected to OmniSurface"
            )
    shader = prims[SHADER_PATH]
    if shader.specifier != Sdf.SpecifierDef or shader.typeName != "Shader":
        raise ValueError("transmissive optics shader prim is invalid")
    expected_shader_properties = {
        "info:implementationSource", "info:mdl:sourceAsset",
        "info:mdl:sourceAsset:subIdentifier", "outputs:out", *_SHADER_INPUTS,
    }
    if {prop.name for prop in shader.properties} != expected_shader_properties:
        raise ValueError("transmissive optics shader has missing or extra properties")
    implementation = shader.attributes["info:implementationSource"]
    source_asset = shader.attributes["info:mdl:sourceAsset"]
    subidentifier = shader.attributes["info:mdl:sourceAsset:subIdentifier"]
    if (
        implementation.typeName != Sdf.ValueTypeNames.Token
        or implementation.default != "sourceAsset"
        or source_asset.typeName != Sdf.ValueTypeNames.Asset
        or source_asset.default.path != OMNISURFACE_MDL
        or subidentifier.typeName != Sdf.ValueTypeNames.Token
        or subidentifier.default != OMNISURFACE_SUBIDENTIFIER
    ):
        raise ValueError("transmissive optics MDL source is invalid")
    expected_values = _expected_shader_values(request, Gf)
    type_by_name = {
        "bool": Sdf.ValueTypeNames.Bool,
        "float": Sdf.ValueTypeNames.Float,
        "color3f": Sdf.ValueTypeNames.Color3f,
    }
    for name, (type_name, _) in _SHADER_INPUTS.items():
        attribute = shader.attributes[name]
        if attribute.typeName != type_by_name[type_name] or attribute.default != expected_values[name]:
            raise ValueError(f"transmissive optics shader input {name} does not match metadata")
        if attribute.custom or attribute.connectionPathList.GetAddedOrExplicitItems():
            raise ValueError(f"transmissive optics shader input {name} has unsupported opinions")
        if layer.ListTimeSamplesForPath(attribute.path):
            raise ValueError(f"transmissive optics shader input {name} has time samples")
    for path in (LIQUID_MESH_PATH, STREAM_MESH_PATH):
        mesh = prims[path]
        if mesh.specifier != Sdf.SpecifierOver or mesh.typeName or {prop.name for prop in mesh.properties} != {"material:binding"}:
            raise ValueError(f"transmissive optics {path} does not have exactly one binding")
        _validate_layer_relationship(mesh.relationships["material:binding"], Sdf.Path(MATERIAL_PATH), path)

    stage = Usd.Stage.Open(str(lexical_layer), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("transmissive optics layer could not be composed")
    from pxr import UsdShade
    composed_material = UsdShade.Material(stage.GetPrimAtPath(MATERIAL_PATH))
    composed_shader = UsdShade.Shader(stage.GetPrimAtPath(SHADER_PATH))
    if not composed_material or not composed_shader:
        raise ValueError("composed transmissive material or shader is missing")
    if composed_shader.GetImplementationSource() != UsdShade.Tokens.sourceAsset:
        raise ValueError("composed transmissive shader does not use source-asset MDL")
    if composed_shader.GetSourceAsset("mdl").path != OMNISURFACE_MDL or composed_shader.GetSourceAssetSubIdentifier("mdl") != OMNISURFACE_SUBIDENTIFIER:
        raise ValueError("composed transmissive shader MDL identity is invalid")
    for name, expected in expected_values.items():
        input_name = name.removeprefix("inputs:")
        if composed_shader.GetInput(input_name).Get() != expected:
            raise ValueError(f"composed transmissive shader input {input_name} differs")
    for mesh_path in (LIQUID_MESH_PATH, STREAM_MESH_PATH):
        bound, relationship = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath(mesh_path)
        ).ComputeBoundMaterial()
        if (
            not bound
            or bound.GetPrim().GetPath() != composed_material.GetPrim().GetPath()
            or relationship.GetTargets() != [Sdf.Path(MATERIAL_PATH)]
        ):
            raise ValueError(f"composed {mesh_path} binding does not resolve to the shared material")
    stack_records = []
    for ordinal, stack_layer in enumerate(stage.GetLayerStack(includeSessionLayers=False), start=1):
        if stack_layer.anonymous or not stack_layer.realPath:
            raise ValueError("transmissive optics layer stack contains an anonymous layer")
        stack_path = _lexical_absolute(stack_layer.realPath)
        if ordinal == 1 and stack_path != lexical_layer:
            raise ValueError("transmissive optics layer is not the composed root")
        if ordinal > 1 and stack_path != twin and not stack_path.is_relative_to(twin):
            raise ValueError("transmissive optics stack contains dependency outside the public twin")
        snapshot = _regular_snapshot(stack_path, "composed USD layer")
        stack_records.append({"ordinal": ordinal, "resolved_path": str(stack_path), "sha256": snapshot["sha256"], "bytes": snapshot["bytes"]})
    if len(stack_records) < 2 or Path(stack_records[1]["resolved_path"]) != source:
        raise ValueError("canonical fill-station source must be second in the layer stack")
    if _regular_snapshot(lexical_layer, "transmissive optics layer", max_bytes=MAX_APPEARANCE_LAYER_BYTES) != layer_snapshot:
        raise ValueError("transmissive optics layer changed during inspection")
    if _regular_snapshot(source, "canonical fill-station source stage") != source_snapshot:
        raise ValueError("canonical fill-station source stage changed during inspection")
    return {
        "path": str(lexical_layer),
        "schema": TRANSMISSIVE_OPTICS_SCHEMA,
        "variant_id": request["variant_id"],
        "stage_sha256": layer_snapshot["sha256"],
        "stage_bytes": layer_snapshot["bytes"],
        "source_stage": SOURCE_STAGE,
        "source_stage_sha256": source_snapshot["sha256"],
        "material_path": MATERIAL_PATH,
        "transmission_color": list(request["transmission_color"]),
        "scattering_depth_m": request["scattering_depth_m"],
        "uncalibrated": True,
        "timing": layer_timing,
        "non_session_usd_layer_stack": stack_records,
    }


def _input_type(type_name: str, Sdf):
    return {"bool": Sdf.ValueTypeNames.Bool, "float": Sdf.ValueTypeNames.Float, "color3f": Sdf.ValueTypeNames.Color3f}[type_name]


def create_transmissive_layer(
    twin_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    variant_id: object,
    transmission_color: object,
    scattering_depth_m: object,
) -> dict[str, object]:
    """Author, inspect, and no-replace publish an OmniSurface liquid override."""

    request = validate_transmissive_request(variant_id, transmission_color, scattering_depth_m)
    Gf, Sdf, Usd, Vt = _import_pxr()
    twin, source_before = _source_record(twin_root)
    source = Path(source_before["path"])
    timing = _source_timing_and_scale(source, source_before, Sdf, Usd)
    output = _existing_directory(output_root, "output root")
    directory = output / "transmissive-optics-layers"
    if os.path.lexists(directory):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("transmissive optics directory must be a non-symlink directory")
    else:
        os.mkdir(directory, 0o755)
        _fsync_directory(output)
    final_path = directory / f"{request['variant_id']}.usda"
    if os.path.lexists(final_path):
        raise FileExistsError(f"transmissive optics output already exists: {final_path}")

    authored = Sdf.Layer.CreateAnonymous("transmissive_optics.usda")
    authored.defaultPrim = "FillStation"
    authored.startTimeCode = timing["start_time_code"]
    authored.endTimeCode = timing["end_time_code"]
    authored.framesPerSecond = timing["frames_per_second"]
    authored.timeCodesPerSecond = timing["time_codes_per_second"]
    authored.pseudoRoot.SetInfo("metersPerUnit", timing["meters_per_unit"])
    authored.subLayerPaths = [_canonical_relative_sublayer(final_path, source)]
    authored.customLayerData = {
        "fluidmix_schema": TRANSMISSIVE_OPTICS_SCHEMA,
        "fluidmix_variant_id": request["variant_id"],
        "fluidmix_material_path": MATERIAL_PATH,
        "fluidmix_transmission_color": Vt.DoubleArray([float(x) for x in request["transmission_color"]]),
        "fluidmix_scattering_depth_m": float(request["scattering_depth_m"]),
        "fluidmix_source_stage": SOURCE_STAGE,
        "fluidmix_source_stage_sha256": source_before["sha256"],
        "fluidmix_source_meters_per_unit": 1.0,
        "fluidmix_uncalibrated": True,
    }
    for path in _EXPECTED_PRIM_PATHS:
        prim = Sdf.CreatePrimInLayer(authored, path)
        if path in (MATERIAL_PATH, SHADER_PATH):
            prim.specifier = Sdf.SpecifierDef
            prim.typeName = "Material" if path == MATERIAL_PATH else "Shader"
        else:
            prim.specifier = Sdf.SpecifierOver
    material = authored.GetPrimAtPath(MATERIAL_PATH)
    for output_name in _MDL_OUTPUTS:
        material_output = Sdf.AttributeSpec(
            material, output_name, Sdf.ValueTypeNames.Token
        )
        material_output.connectionPathList.explicitItems = [
            Sdf.Path(SHADER_PATH + ".outputs:out")
        ]
    shader = authored.GetPrimAtPath(SHADER_PATH)
    implementation = Sdf.AttributeSpec(shader, "info:implementationSource", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
    implementation.default = "sourceAsset"
    source_asset = Sdf.AttributeSpec(shader, "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, Sdf.VariabilityUniform)
    source_asset.default = Sdf.AssetPath(OMNISURFACE_MDL)
    subidentifier = Sdf.AttributeSpec(shader, "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
    subidentifier.default = OMNISURFACE_SUBIDENTIFIER
    Sdf.AttributeSpec(shader, "outputs:out", Sdf.ValueTypeNames.Token)
    values = _expected_shader_values(request, Gf)
    for name, (type_name, _) in _SHADER_INPUTS.items():
        attribute = Sdf.AttributeSpec(shader, name, _input_type(type_name, Sdf))
        attribute.default = values[name]
    for mesh_path in (LIQUID_MESH_PATH, STREAM_MESH_PATH):
        mesh = authored.GetPrimAtPath(mesh_path)
        binding = Sdf.RelationshipSpec(mesh, "material:binding", custom=False)
        binding.targetPathList.explicitItems = [Sdf.Path(MATERIAL_PATH)]
    content = authored.ExportToString().encode("utf-8")
    if len(content) > MAX_APPEARANCE_LAYER_BYTES:
        raise ValueError("authored transmissive optics layer exceeds maximum size")
    temporary = directory / f".{request['variant_id']}.{secrets.token_hex(16)}.tmp.usda"
    published = False
    owned_identity: tuple[int, int] | None = None
    try:
        written = _write_exclusive(temporary, content)
        owned_identity = (written.st_dev, written.st_ino)
        inspect_transmissive_layer(temporary, twin, expected_variant_id=request["variant_id"], require_filename_match=False)
        if _regular_snapshot(source, "canonical fill-station source stage") != source_before:
            raise ValueError("canonical fill-station source stage changed before publication")
        try:
            os.link(temporary, final_path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"transmissive optics output already exists: {final_path}") from None
        published = True
        _require_owned_identity(final_path, owned_identity)
        _fsync_directory(directory)
        record = inspect_transmissive_layer(final_path, twin, expected_variant_id=request["variant_id"])
        _require_owned_identity(final_path, owned_identity)
        return record
    except Exception:
        if published and owned_identity is not None:
            try:
                metadata = os.lstat(final_path)
                if (metadata.st_dev, metadata.st_ino) == owned_identity:
                    os.unlink(final_path)
                    _fsync_directory(directory)
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            metadata = os.lstat(temporary)
            if owned_identity is not None and (metadata.st_dev, metadata.st_ino) == owned_identity:
                os.unlink(temporary)
                _fsync_directory(directory)
        except FileNotFoundError:
            pass
